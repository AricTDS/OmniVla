from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import ValidationError

from .log_utils import emit_event
from .schemas import (
    AckMessage,
    ErrorMessage,
    FrameMessage,
    PingMessage,
    PongMessage,
    StartSessionMessage,
    StopSessionMessage,
    TimingMessage,
    TrajectoryMessage,
    dump_model,
    parse_client_message,
)
from .session_manager import SessionState

router = APIRouter()


@dataclass
class PendingFrame:
    session_id: str
    seq: int
    enqueued_at: float
    current_image_pil: Image.Image
    goal_image_pil: Image.Image
    instruction: Optional[str]


def decode_jpeg_b64(image_b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(image_b64, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("invalid jpeg payload") from exc


def log_ws(event: str, log_file_path: Optional[str] = None, **kwargs) -> None:
    emit_event("WS", event, log_file_path=log_file_path, **kwargs)


@router.websocket("/ws/v1/omnivla")
async def omnivla_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    client = websocket.client
    client_addr = f"{client.host}:{client.port}" if client else "unknown"
    log_file_path = getattr(websocket.app.state, "log_file_path", None)

    def ws_log(event: str, **kwargs) -> None:
        log_ws(event, log_file_path=log_file_path, client=client_addr, **kwargs)

    ws_log("connected")

    session_manager = websocket.app.state.session_manager
    frame_buffer = websocket.app.state.frame_buffer
    engine = websocket.app.state.engine

    send_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    async def send_model(model) -> None:
        async with send_lock:
            await websocket.send_json(dump_model(model))

    async def send_error(
        error_code: str,
        message: str,
        session_id: Optional[str] = None,
        seq: Optional[int] = None,
    ) -> None:
        ws_log(
            "send_error",
            error_code=error_code,
            session_id=session_id,
            seq=seq,
            message=message,
        )
        await send_model(
            ErrorMessage(
                session_id=session_id,
                seq=seq,
                error_code=error_code,
                message=message,
            )
        )

    async def consumer_loop() -> None:
        while not stop_event.is_set():
            pending = await frame_buffer.pop(timeout_s=0.2)
            if pending is None:
                continue

            try:
                queue_wait_ms = int(round((time.perf_counter() - pending.enqueued_at) * 1000))
                ws_log(
                    "frame_dequeued",
                    session_id=pending.session_id,
                    seq=pending.seq,
                    queue_wait_ms=queue_wait_ms,
                )
                infer_start = time.perf_counter()
                infer_result = await asyncio.to_thread(
                    engine.infer,
                    pending.current_image_pil,
                    pending.goal_image_pil,
                    pending.instruction,
                )
                infer_ms = int(round((time.perf_counter() - infer_start) * 1000))

                post_start = time.perf_counter()
                trajectory = TrajectoryMessage(
                    session_id=pending.session_id,
                    seq=pending.seq,
                    linear_vel=infer_result["linear_vel"],
                    angular_vel=infer_result["angular_vel"],
                    modality_id=infer_result["modality_id"],
                    waypoints=infer_result["waypoints"],
                    timing_ms=TimingMessage(
                        queue_wait=queue_wait_ms,
                        infer=max(infer_ms, int(infer_result.get("infer_ms", infer_ms))),
                        post=0,
                        total=0,
                    ),
                )
                post_ms = int(round((time.perf_counter() - post_start) * 1000))
                trajectory.timing_ms.post = post_ms
                trajectory.timing_ms.total = trajectory.timing_ms.queue_wait + trajectory.timing_ms.infer + post_ms
                await send_model(trajectory)
                ws_log(
                    "trajectory_sent",
                    session_id=pending.session_id,
                    seq=pending.seq,
                    infer_ms=trajectory.timing_ms.infer,
                    total_ms=trajectory.timing_ms.total,
                    linear=f"{trajectory.linear_vel:.4f}",
                    angular=f"{trajectory.angular_vel:.4f}",
                    modality_id=trajectory.modality_id,
                )
            except Exception as exc:  # noqa: BLE001
                ws_log(
                    "infer_exception",
                    session_id=pending.session_id,
                    seq=pending.seq,
                    error=str(exc),
                )
                await send_error(
                    error_code="E_INTERNAL",
                    message=f"inference failed: {exc}",
                    session_id=pending.session_id,
                    seq=pending.seq,
                )

    consumer_task = asyncio.create_task(consumer_loop(), name="omnivla-ws-consumer")

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                ws_log("bad_json", payload_len=len(raw_text))
                await send_error("E_BAD_REQUEST", "invalid json")
                continue

            try:
                msg = parse_client_message(payload)
            except (ValidationError, ValueError) as exc:
                ws_log("invalid_payload", error=str(exc))
                await send_error("E_BAD_REQUEST", f"invalid payload: {exc}")
                continue

            if isinstance(msg, StartSessionMessage):
                ws_log(
                    "start_session_received",
                    session_id=msg.session_id,
                    fps=msg.config.fps,
                    has_instruction=bool(msg.instruction),
                )
                try:
                    goal_image_pil = decode_jpeg_b64(msg.goal_image_jpeg_b64)
                except ValueError as exc:
                    await send_error("E_IMAGE_DECODE", str(exc), session_id=msg.session_id)
                    continue

                await session_manager.start_session(
                    SessionState(
                        session_id=msg.session_id,
                        goal_image_pil=goal_image_pil,
                        instruction=msg.instruction,
                        fps=msg.config.fps,
                    )
                )
                await frame_buffer.clear()
                await send_model(AckMessage(session_id=msg.session_id, op="start_session", message="ok"))
                ws_log("start_session_ack", session_id=msg.session_id)
                continue

            if isinstance(msg, FrameMessage):
                ws_log("frame_received", session_id=msg.session_id, seq=msg.seq)
                session = await session_manager.get_session(msg.session_id)
                if session is None:
                    await send_error(
                        "E_SESSION_NOT_FOUND",
                        "session not found, please call start_session first",
                        session_id=msg.session_id,
                        seq=msg.seq,
                    )
                    continue

                try:
                    current_image_pil = decode_jpeg_b64(msg.current_image_jpeg_b64)
                except ValueError as exc:
                    await send_error("E_IMAGE_DECODE", str(exc), session_id=msg.session_id, seq=msg.seq)
                    continue

                await frame_buffer.push(
                    PendingFrame(
                        session_id=msg.session_id,
                        seq=msg.seq,
                        enqueued_at=time.perf_counter(),
                        current_image_pil=current_image_pil,
                        goal_image_pil=session.goal_image_pil,
                        instruction=session.instruction,
                    )
                )
                await session_manager.update_last_seq(msg.session_id, msg.seq)
                ws_log("frame_enqueued", session_id=msg.session_id, seq=msg.seq)
                continue

            if isinstance(msg, StopSessionMessage):
                ws_log("stop_session_received", session_id=msg.session_id)
                stopped = await session_manager.stop_session(msg.session_id)
                await frame_buffer.clear()
                if not stopped:
                    await send_error("E_SESSION_NOT_FOUND", "session not found", session_id=msg.session_id)
                else:
                    await send_model(AckMessage(session_id=msg.session_id, op="stop_session", message="ok"))
                    ws_log("stop_session_ack", session_id=msg.session_id)
                continue

            if isinstance(msg, PingMessage):
                await send_model(PongMessage(session_id=msg.session_id, timestamp_ms=msg.timestamp_ms))
                ws_log("pong_sent", session_id=msg.session_id)
                continue

            await send_error("E_BAD_REQUEST", "unsupported message type")

    except WebSocketDisconnect:
        ws_log("disconnected")
    finally:
        stop_event.set()
        consumer_task.cancel()
        await frame_buffer.clear()
        await session_manager.clear()
        ws_log("cleanup_done")
