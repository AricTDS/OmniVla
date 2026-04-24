from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


def default_log_file_path() -> str:
    """未设置 OMNIVLA_LOG_FILE 时使用，文件名带启动时间戳。"""
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"./logs/omnivla_server_{ts}.log"


@dataclass(frozen=True)
class ServerSettings:
    model_path: str = "./omnivla-original"
    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 8000
    save_dir: str = "./inference"
    log_file_path: str = field(default_factory=default_log_file_path)
    default_fps: int = 4
    max_fps: int = 10

    @staticmethod
    def from_env() -> "ServerSettings":
        log_path = os.getenv("OMNIVLA_LOG_FILE", "").strip()
        if not log_path:
            log_path = default_log_file_path()
        return ServerSettings(
            model_path=os.getenv("OMNIVLA_MODEL_PATH", "./omnivla-original"),
            device=os.getenv("OMNIVLA_DEVICE", "cuda:0"),
            host=os.getenv("OMNIVLA_HOST", "0.0.0.0"),
            port=int(os.getenv("OMNIVLA_PORT", "8000")),
            save_dir=os.getenv("OMNIVLA_SAVE_DIR", "./inference"),
            log_file_path=log_path,
            default_fps=int(os.getenv("OMNIVLA_DEFAULT_FPS", "4")),
            max_fps=int(os.getenv("OMNIVLA_MAX_FPS", "10")),
        )
