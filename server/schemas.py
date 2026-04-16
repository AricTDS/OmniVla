from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, Field


class StartSessionConfig(BaseModel):
    fps: int = Field(default=4, ge=1, le=10)


class StartSessionMessage(BaseModel):
    type: Literal["start_session"]
    session_id: str = Field(min_length=1, max_length=64)
    instruction: Optional[str] = Field(default=None, max_length=512)
    goal_image_jpeg_b64: str = Field(min_length=1)
    config: StartSessionConfig = Field(default_factory=StartSessionConfig)


class FrameMessage(BaseModel):
    type: Literal["frame"]
    session_id: str = Field(min_length=1, max_length=64)
    seq: int = Field(ge=0)
    timestamp_ms: Optional[int] = None
    current_image_jpeg_b64: str = Field(min_length=1)


class StopSessionMessage(BaseModel):
    type: Literal["stop_session"]
    session_id: str = Field(min_length=1, max_length=64)


class PingMessage(BaseModel):
    type: Literal["ping"]
    session_id: Optional[str] = None
    timestamp_ms: Optional[int] = None


ClientMessage = Union[StartSessionMessage, FrameMessage, StopSessionMessage, PingMessage]


class AckMessage(BaseModel):
    type: Literal["ack"] = "ack"
    session_id: Optional[str] = None
    op: str
    message: str = "ok"


class TimingMessage(BaseModel):
    queue_wait: int
    infer: int
    post: int
    total: int


class TrajectoryMessage(BaseModel):
    type: Literal["trajectory"] = "trajectory"
    session_id: str
    seq: int
    linear_vel: float
    angular_vel: float
    modality_id: int
    waypoints: List[List[float]]
    timing_ms: TimingMessage


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    session_id: Optional[str] = None
    seq: Optional[int] = None
    error_code: str
    message: str


class PongMessage(BaseModel):
    type: Literal["pong"] = "pong"
    session_id: Optional[str] = None
    timestamp_ms: Optional[int] = None


def parse_model(model_cls: Type[BaseModel], payload: Dict[str, Any]) -> BaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def dump_model(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def parse_client_message(payload: Dict[str, Any]) -> ClientMessage:
    msg_type = payload.get("type")
    type_to_model: Dict[str, Type[BaseModel]] = {
        "start_session": StartSessionMessage,
        "frame": FrameMessage,
        "stop_session": StopSessionMessage,
        "ping": PingMessage,
    }
    model_cls = type_to_model.get(msg_type)
    if model_cls is None:
        raise ValueError(f"unsupported message type: {msg_type}")
    return parse_model(model_cls, payload)  # type: ignore[return-value]
