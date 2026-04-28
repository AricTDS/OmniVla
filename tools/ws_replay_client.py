#!/usr/bin/env python3
"""
Replay / stream client for OmniVLA WebSocket v1.

文件列表（一次性回放，结束发 stop_session）:
  cd <repo> && conda run -n omnivla python -u tools/ws_replay_client.py \\
    --ws-url ws://127.0.0.1:8000/ws/v1/omnivla \\
    --goal example/20260414-192532_goal.jpg \\
    --frames example/20260414-192532_current.jpg example/20260414-192536.jpg

文件夹持续定频流（循环读图、Ctrl+C 结束，默认 10Hz）:
  conda run -n omnivla python -u tools/ws_replay_client.py \\
    --goal example/20260414-192532_goal.jpg \\
    --frames-dir example/frames_in \\
    --stream-hz 10

日志：与 server 共用 server/log_utils.emit_event，默认落盘 ./logs/ws_replay_client_<时间戳>.log；
可用 --log-file、环境变量 OMNIVLA_CLIENT_LOG_FILE，或 --no-log-file 仅终端。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import websockets
except ModuleNotFoundError as e:  # noqa: FUN101
    print(
        "缺少依赖 websockets。请任选其一：\n"
        "  1) 在 omnivla 环境中运行（推荐）: conda run -n omnivla python -u tools/ws_replay_client.py ...\n"
        "  2) 或: pip install websockets",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

# 与 server 共用 log_utils；须能从仓库根 import server（见下方 sys.path）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os

from server.log_utils import default_client_replay_log_file_path, emit_event

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def client_log(args: argparse.Namespace, event: str, **kwargs: Any) -> None:
    """与 server 相同：终端 + 可选落盘到 log_file_path（见 server.log_utils.emit_event）。"""
    path = getattr(args, "log_file_path", None)
    emit_event("CLIENT", event, log_file_path=path, **kwargs)


def log_received_waypoints(args: argparse.Namespace, seq: int, resp: Dict[str, Any]) -> None:
    """按「batch step [4 维]」逐行打印 waypoints，与 run_omnivla 侧输出风格一致，便于对照日志。"""
    path = getattr(args, "log_file_path", None)
    wps = resp.get("waypoints")
    if wps is None:
        client_log(args, "trajectory_waypoints_missing", seq=seq)
        return
    if not isinstance(wps, list):
        client_log(args, "trajectory_waypoints_bad_type", seq=seq, typ=str(type(wps)))
        return

    emit_event("CLIENT", f"waypoints seq={seq} batch=0", log_file_path=path)
    for i, row in enumerate(wps):
        if isinstance(row, (list, tuple)) and len(row) >= 4:
            a, b, c, d = (float(row[j]) for j in range(4))
            line = f"  batch=0 step={i}  [{a:.8f}, {b:.8f}, {c:.8f}, {d:.8f}]"
        else:
            try:
                inner = json.dumps(row, ensure_ascii=False)
            except (TypeError, ValueError):
                inner = str(row)
            line = f"  batch=0 step={i}  {inner}"
        emit_event("CLIENT", line, log_file_path=path)


def _argparse_stream_hz(s: str) -> float:
    """--stream-hz 只能为数字；若误传图片路径则给出明确提示。"""
    s = s.strip()
    try:
        v = float(s)
    except ValueError as e:
        sl = s.lower()
        looks_path = (
            "/" in s
            or "\\" in s
            or any(sl.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"))
        )
        if looks_path:
            raise argparse.ArgumentTypeError(
                f"误把路径当成 --stream-hz: {s!r}。"
                f" 频率只填数字，例如 --stream-hz 10；"
                f" 单张图用 --frames {s}；多图/目录用 --frames-dir <文件夹>"
            ) from e
        raise argparse.ArgumentTypeError(f"无效浮点数: {s!r}") from e
    if v <= 0:
        raise argparse.ArgumentTypeError("stream-hz 必须 > 0")
    return v


def image_to_b64(image_path: Path) -> str:
    raw = image_path.read_bytes()
    return base64.b64encode(raw).decode("ascii")


def list_image_files(dir_path: Path) -> List[Path]:
    if not dir_path.is_dir():
        raise NotADirectoryError(f"not a directory: {dir_path}")
    out: List[Path] = []
    for p in sorted(dir_path.iterdir(), key=lambda x: (x.name.lower(), str(x))):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return out


async def recv_until(
    ws: websockets.ClientConnection,
    timeout_s: float,
    expected_type: str | None = None,
    expected_seq: int | None = None,
) -> Dict[str, Any]:
    end_ts = time.perf_counter() + timeout_s
    while True:
        remaining = end_ts - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("receive timeout")

        msg_text = await asyncio.wait_for(ws.recv(), timeout=remaining)
        msg = json.loads(msg_text)

        if expected_type is not None and msg.get("type") != expected_type:
            continue

        if expected_seq is not None and msg.get("seq") != expected_seq:
            continue

        return msg


def _print_summary(
    args: argparse.Namespace, client_rtt_ms: List[float], server_total_ms: List[int]
) -> None:
    if client_rtt_ms:
        p95 = statistics.quantiles(client_rtt_ms, n=20)[-1] if len(client_rtt_ms) >= 20 else max(client_rtt_ms)
        client_log(
            args,
            "summary_client_rtt",
            avg=f"{statistics.mean(client_rtt_ms):.1f}",
            p95_ms=f"{p95:.1f}",
            n=len(client_rtt_ms),
        )
    if server_total_ms:
        p95 = statistics.quantiles(server_total_ms, n=20)[-1] if len(server_total_ms) >= 20 else max(server_total_ms)
        client_log(
            args,
            "summary_server_total",
            avg_ms=f"{statistics.mean(server_total_ms):.1f}",
            p95_ms=f"{p95:.1f}",
            n=len(server_total_ms),
        )


async def run_replay_list(args: argparse.Namespace) -> int:
    """原有逻辑：--frames 列表，repeat 轮后结束。"""
    goal_path = Path(args.goal).resolve()
    frame_paths = [Path(p).resolve() for p in args.frames]

    if not goal_path.exists():
        raise FileNotFoundError(f"goal image not found: {goal_path}")
    for frame_path in frame_paths:
        if not frame_path.exists():
            raise FileNotFoundError(f"frame image not found: {frame_path}")

    goal_b64 = image_to_b64(goal_path)
    frame_b64_list = [image_to_b64(p) for p in frame_paths]

    client_log(
        args,
        "list_mode",
        ws_url=args.ws_url,
        goal=str(goal_path),
        num_frames=len(frame_paths),
        repeat=args.repeat,
    )

    client_rtt_ms: List[float] = []
    server_total_ms: List[int] = []

    async with websockets.connect(args.ws_url, max_size=50 * 1024 * 1024) as ws:
        start_payload: Dict[str, Any] = {
            "type": "start_session",
            "session_id": args.session_id,
            "instruction": args.instruction,
            "goal_image_jpeg_b64": goal_b64,
            "config": {"fps": min(args.fps, 10)},
        }
        await ws.send(json.dumps(start_payload))

        start_resp = await recv_until(ws, timeout_s=args.timeout_s)
        if start_resp.get("type") == "error":
            client_log(args, "start_session_error", response=str(start_resp))
            return 2
        client_log(args, "start_session_ack", response=str(start_resp))

        seq = args.start_seq
        for _ in range(args.repeat):
            for frame_path, frame_b64 in zip(frame_paths, frame_b64_list):
                frame_payload = {
                    "type": "frame",
                    "session_id": args.session_id,
                    "seq": seq,
                    "timestamp_ms": int(time.time() * 1000),
                    "current_image_jpeg_b64": frame_b64,
                }

                t0 = time.perf_counter()
                await ws.send(json.dumps(frame_payload))
                resp = await recv_until(
                    ws,
                    timeout_s=args.timeout_s,
                    expected_seq=seq,
                )
                rtt_ms = (time.perf_counter() - t0) * 1000.0

                if resp.get("type") == "error":
                    client_log(args, "frame_error", seq=seq, response=str(resp))
                    return 3

                timing = resp.get("timing_ms", {})
                total_ms = int(timing.get("total", -1))
                infer_ms = int(timing.get("infer", -1))
                client_log(
                    args,
                    "trajectory",
                    seq=seq,
                    frame=frame_path.name,
                    linear=f"{float(resp.get('linear_vel', 0.0)):.4f}",
                    angular=f"{float(resp.get('angular_vel', 0.0)):.4f}",
                    modality_id=int(resp.get("modality_id", -1)),
                    server_total_ms=total_ms,
                    infer_ms=infer_ms,
                    client_rtt_ms=f"{rtt_ms:.1f}",
                )
                log_received_waypoints(args, seq, resp)
                if total_ms >= 0:
                    server_total_ms.append(total_ms)
                client_rtt_ms.append(rtt_ms)

                seq += 1
                if args.send_interval_ms > 0:
                    await asyncio.sleep(args.send_interval_ms / 1000.0)

        await ws.send(json.dumps({"type": "stop_session", "session_id": args.session_id}))
        stop_resp = await recv_until(ws, timeout_s=args.timeout_s)
        client_log(args, "stop_session_done", response=str(stop_resp))

    _print_summary(args, client_rtt_ms, server_total_ms)
    return 0


async def run_dir_stream(args: argparse.Namespace) -> int:
    """--frames-dir：定频从文件夹读图、循环轮询，按 Ctrl+C 或 kill 前发 stop_session。"""
    goal_path = Path(args.goal).resolve()
    dir_path = Path(args.frames_dir).resolve()

    if not goal_path.exists():
        raise FileNotFoundError(f"goal image not found: {goal_path}")

    # 与 server settings max_fps=10 对齐
    raw_hz = float(args.stream_hz)
    if raw_hz > 10.0:
        client_log(
            args,
            "stream_hz_clamped",
            raw_hz=raw_hz,
            effective_hz=10.0,
            note="server max_fps=10",
        )
    effective_hz = min(max(raw_hz, 0.1), 10.0)
    period_s = 1.0 / effective_hz
    fps_cfg = min(10, max(1, int(round(effective_hz))))

    def refresh_paths() -> List[Path]:
        paths = list_image_files(dir_path)
        if not paths:
            raise FileNotFoundError(
                f"no image files in {dir_path} (扩展名: {sorted(IMAGE_SUFFIXES)})"
            )
        return paths

    paths = refresh_paths()
    rescan_sec = max(0.0, float(args.rescan_sec))

    client_log(
        args,
        "dir_stream_mode",
        ws_url=args.ws_url,
        goal=str(goal_path),
        frames_dir=str(dir_path),
        num_files=len(paths),
        effective_hz=f"{effective_hz}",
        start_session_fps=fps_cfg,
    )
    if rescan_sec > 0:
        client_log(args, "rescan", rescan_sec=rescan_sec, note="定期重扫目录")

    goal_b64 = image_to_b64(goal_path)
    client_rtt_ms: List[float] = []
    server_total_ms: List[int] = []
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()

    def _on_signal() -> None:
        client_log(args, "signal_stop", note="即将停止流")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except (NotImplementedError, RuntimeError, ValueError):
            pass  # Windows/子线程等：靠 KeyboardInterrupt 或进程信号结束

    idx = 0
    last_rescan = time.perf_counter()

    async with websockets.connect(args.ws_url, max_size=50 * 1024 * 1024) as ws:
        start_payload: Dict[str, Any] = {
            "type": "start_session",
            "session_id": args.session_id,
            "instruction": args.instruction,
            "goal_image_jpeg_b64": goal_b64,
            "config": {"fps": fps_cfg},
        }
        await ws.send(json.dumps(start_payload))

        start_resp = await recv_until(ws, timeout_s=args.timeout_s)
        if start_resp.get("type") == "error":
            client_log(args, "start_session_error", response=str(start_resp))
            return 2
        client_log(args, "start_session_ack", response=str(start_resp))

        seq = args.start_seq
        try:
            while not stop.is_set():
                t_tick = time.perf_counter()

                if rescan_sec > 0 and (t_tick - last_rescan) >= rescan_sec:
                    try:
                        new_paths = refresh_paths()
                        if len(new_paths) != len(paths):
                            client_log(
                                args,
                                "dir_rescan",
                                before=len(paths),
                                after=len(new_paths),
                            )
                        paths = new_paths
                    except FileNotFoundError as e:
                        client_log(args, "dir_rescan_error", error=str(e))
                    last_rescan = t_tick

                frame_path = paths[idx % len(paths)]
                idx += 1

                try:
                    frame_b64 = image_to_b64(frame_path)
                except OSError as e:
                    client_log(args, "read_skip", path=str(frame_path), error=str(e))
                    await asyncio.sleep(period_s)
                    continue

                frame_payload = {
                    "type": "frame",
                    "session_id": args.session_id,
                    "seq": seq,
                    "timestamp_ms": int(time.time() * 1000),
                    "current_image_jpeg_b64": frame_b64,
                }

                t0 = time.perf_counter()
                await ws.send(json.dumps(frame_payload))
                resp = await recv_until(
                    ws,
                    timeout_s=args.timeout_s,
                    expected_seq=seq,
                )
                rtt_ms = (time.perf_counter() - t0) * 1000.0

                if resp.get("type") == "error":
                    client_log(args, "frame_error", seq=seq, response=str(resp))
                    return 3

                timing = resp.get("timing_ms", {})
                total_ms = int(timing.get("total", -1))
                infer_ms = int(timing.get("infer", -1))
                client_log(
                    args,
                    "trajectory",
                    seq=seq,
                    file=frame_path.name,
                    linear=f"{float(resp.get('linear_vel', 0.0)):.4f}",
                    angular=f"{float(resp.get('angular_vel', 0.0)):.4f}",
                    modality_id=int(resp.get("modality_id", -1)),
                    server_total_ms=total_ms,
                    infer_ms=infer_ms,
                    rtt_ms=f"{rtt_ms:.1f}",
                )
                log_received_waypoints(args, seq, resp)
                if total_ms >= 0:
                    server_total_ms.append(total_ms)
                client_rtt_ms.append(rtt_ms)
                seq += 1

                elapsed = time.perf_counter() - t_tick
                to_sleep = period_s - elapsed
                if to_sleep > 0:
                    await asyncio.sleep(to_sleep)
        except KeyboardInterrupt:
            client_log(args, "keyboard_interrupt", note="准备停止流")
        finally:
            try:
                await ws.send(json.dumps({"type": "stop_session", "session_id": args.session_id}))
                stop_resp = await recv_until(ws, timeout_s=min(5.0, args.timeout_s))
                client_log(args, "stop_session_done", response=str(stop_resp))
            except (asyncio.CancelledError, OSError, websockets.exceptions.WebSocketException) as e:  # noqa: BLE001
                client_log(args, "stop_session_error", error=str(e))
            try:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    try:
                        loop.remove_signal_handler(sig)
                    except (NotImplementedError, RuntimeError, ValueError):
                        pass
            except Exception:  # noqa: BLE001
                pass

    _print_summary(args, client_rtt_ms, server_total_ms)
    return 0


async def run(args: argparse.Namespace) -> int:
    if bool(args.frames_dir) and bool(args.frames):
        client_log(args, "arg_error", message="不能同时指定 --frames 与 --frames-dir")
        return 1
    if not args.frames_dir and not args.frames:
        client_log(args, "arg_error", message="需要 --frames 或 --frames-dir")
        return 1
    if args.frames_dir:
        return await run_dir_stream(args)
    return await run_replay_list(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="向 OmniVLA WebSocket 回放图片：文件列表 或 文件夹定频流。",
        epilog=(
            "示例(目录+10Hz):  python -u tools/ws_replay_client.py --goal G.jpg --frames-dir ./frames --stream-hz 10\n"
            "示例(单张/多张):  python -u tools/ws_replay_client.py --goal G.jpg --frames a.jpg b.jpg"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ws-url", type=str, default="ws://127.0.0.1:8000/ws/v1/omnivla")
    parser.add_argument("--session-id", type=str, default="sess_demo")
    parser.add_argument("--goal", type=str, required=True)
    parser.add_argument(
        "--frames",
        type=str,
        nargs="*",
        default=[],
        help="当前相机帧：一个或多个图片路径；与 --frames-dir 二选一",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="从目录持续定频发图：按文件名排序的 .jpg/.jpeg/.png/…，轮询；按 Ctrl+C 结束",
    )
    parser.add_argument(
        "--stream-hz",
        type=_argparse_stream_hz,
        default=10.0,
        help="仅 --frames-dir：定频(次/秒)，如 10。勿写图片路径；单张图用 --frames。默认 10。",
    )
    parser.add_argument(
        "--rescan-sec",
        type=float,
        default=0.0,
        help="仅 --frames-dir：每 N 秒重扫目录更新文件列表；0=启动时只扫一次",
    )
    parser.add_argument("--instruction", type=str, default="move toward blue trash bin")
    parser.add_argument(
        "--fps", type=int, default=4, help="仅列表模式: start_session.config.fps（1–10）"
    )
    parser.add_argument("--repeat", type=int, default=1, help="仅列表模式: 整组帧重复轮数")
    parser.add_argument("--start-seq", type=int, default=0)
    parser.add_argument(
        "--send-interval-ms", type=int, default=0, help="仅列表模式: 每帧发送后额外休眠(ms)"
    )
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="客户端日志文件（与 server 的 emit_event 同格式落盘）。未写则读环境 OMNIVLA_CLIENT_LOG_FILE，再否则 ./logs/ws_replay_client_<时间戳>.log",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="仅输出到终端，不落盘",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.no_log_file:
        args.log_file_path = None
    elif args.log_file is not None:
        args.log_file_path = str(Path(args.log_file).resolve())
    else:
        env = os.getenv("OMNIVLA_CLIENT_LOG_FILE", "").strip()
        if env:
            args.log_file_path = env
        else:
            args.log_file_path = default_client_replay_log_file_path()
    client_log(
        args,
        "client_start",
        log_to=args.log_file_path or "(仅终端)",
        session_id=args.session_id,
    )
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
