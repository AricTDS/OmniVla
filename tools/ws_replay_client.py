#!/usr/bin/env python3
"""
Replay client for OmniVLA WebSocket v1.

Usage example:
  conda run -n omnivla python tools/ws_replay_client.py \
    --ws-url ws://127.0.0.1:8000/ws/v1/omnivla \
    --goal example/20260414-192532_goal.jpg \
    --frames example/20260414-192532_current.jpg example/20260414-192536.jpg \
    --instruction "move forward and avoid right obstacle"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import websockets


def image_to_b64(image_path: Path) -> str:
    raw = image_path.read_bytes()
    return base64.b64encode(raw).decode("ascii")


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
            # Skip unexpected message type and continue waiting.
            continue

        if expected_seq is not None and msg.get("seq") != expected_seq:
            # Skip stale/out-of-order trajectory and continue waiting.
            continue

        return msg


async def run(args: argparse.Namespace) -> int:
    goal_path = Path(args.goal).resolve()
    frame_paths = [Path(p).resolve() for p in args.frames]

    if not goal_path.exists():
        raise FileNotFoundError(f"goal image not found: {goal_path}")
    for frame_path in frame_paths:
        if not frame_path.exists():
            raise FileNotFoundError(f"frame image not found: {frame_path}")

    goal_b64 = image_to_b64(goal_path)
    frame_b64_list = [image_to_b64(p) for p in frame_paths]

    print(f"[client] ws_url={args.ws_url}")
    print(f"[client] goal={goal_path}")
    print(f"[client] frames={len(frame_paths)} repeat={args.repeat}")

    client_rtt_ms: List[float] = []
    server_total_ms: List[int] = []

    async with websockets.connect(args.ws_url, max_size=50 * 1024 * 1024) as ws:
        start_payload: Dict[str, Any] = {
            "type": "start_session",
            "session_id": args.session_id,
            "instruction": args.instruction,
            "goal_image_jpeg_b64": goal_b64,
            "config": {"fps": args.fps},
        }
        await ws.send(json.dumps(start_payload))

        start_resp = await recv_until(ws, timeout_s=args.timeout_s)
        if start_resp.get("type") == "error":
            print(f"[client] start_session error: {start_resp}")
            return 2
        print(f"[client] start_session ack: {start_resp}")

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
                    print(f"[client] frame seq={seq} error: {resp}")
                    return 3

                timing = resp.get("timing_ms", {})
                total_ms = int(timing.get("total", -1))
                infer_ms = int(timing.get("infer", -1))
                print(
                    "[client] seq={} frame={} linear={:.4f} angular={:.4f} modality={} "
                    "server_total={}ms infer={}ms client_rtt={:.1f}ms".format(
                        seq,
                        frame_path.name,
                        float(resp.get("linear_vel", 0.0)),
                        float(resp.get("angular_vel", 0.0)),
                        int(resp.get("modality_id", -1)),
                        total_ms,
                        infer_ms,
                        rtt_ms,
                    )
                )
                if total_ms >= 0:
                    server_total_ms.append(total_ms)
                client_rtt_ms.append(rtt_ms)

                seq += 1
                if args.send_interval_ms > 0:
                    await asyncio.sleep(args.send_interval_ms / 1000.0)

        await ws.send(json.dumps({"type": "stop_session", "session_id": args.session_id}))
        stop_resp = await recv_until(ws, timeout_s=args.timeout_s)
        print(f"[client] stop_session resp: {stop_resp}")

    if client_rtt_ms:
        print(
            "[summary] client_rtt_ms avg={:.1f} p95={:.1f} n={}".format(
                statistics.mean(client_rtt_ms),
                statistics.quantiles(client_rtt_ms, n=20)[-1] if len(client_rtt_ms) >= 20 else max(client_rtt_ms),
                len(client_rtt_ms),
            )
        )
    if server_total_ms:
        print(
            "[summary] server_total_ms avg={:.1f} p95={:.1f} n={}".format(
                statistics.mean(server_total_ms),
                statistics.quantiles(server_total_ms, n=20)[-1] if len(server_total_ms) >= 20 else max(server_total_ms),
                len(server_total_ms),
            )
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay example images to OmniVLA WebSocket service.")
    parser.add_argument("--ws-url", type=str, default="ws://127.0.0.1:8000/ws/v1/omnivla")
    parser.add_argument("--session-id", type=str, default="sess_demo")
    parser.add_argument("--goal", type=str, required=True)
    parser.add_argument("--frames", type=str, nargs="+", required=True)
    parser.add_argument("--instruction", type=str, default="move toward blue trash bin")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--start-seq", type=int, default=0)
    parser.add_argument("--send-interval-ms", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
