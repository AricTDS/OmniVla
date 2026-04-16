from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerSettings:
    model_path: str = "./omnivla-original"
    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 8000
    save_dir: str = "./inference"
    default_fps: int = 4
    max_fps: int = 10

    @staticmethod
    def from_env() -> "ServerSettings":
        return ServerSettings(
            model_path=os.getenv("OMNIVLA_MODEL_PATH", "./omnivla-original"),
            device=os.getenv("OMNIVLA_DEVICE", "cuda:0"),
            host=os.getenv("OMNIVLA_HOST", "0.0.0.0"),
            port=int(os.getenv("OMNIVLA_PORT", "8000")),
            save_dir=os.getenv("OMNIVLA_SAVE_DIR", "./inference"),
            default_fps=int(os.getenv("OMNIVLA_DEFAULT_FPS", "4")),
            max_fps=int(os.getenv("OMNIVLA_MAX_FPS", "10")),
        )
