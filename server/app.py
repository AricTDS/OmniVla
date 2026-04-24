from __future__ import annotations

import asyncio
from typing import Dict

from fastapi import FastAPI

from .frame_buffer import LatestFrameBuffer
from .log_utils import emit_event
from .omnivla_engine import OmniVLAEngine
from .session_manager import SessionManager
from .settings import ServerSettings
from .ws_router import router as ws_router


def create_app() -> FastAPI:
    settings = ServerSettings.from_env()
    app = FastAPI(title="OmniVLA Realtime WS Service", version="v1")

    app.state.settings = settings
    app.state.log_file_path = settings.log_file_path
    app.state.session_manager = SessionManager()
    app.state.frame_buffer = LatestFrameBuffer()
    app.state.engine = OmniVLAEngine(
        model_path=settings.model_path,
        device=settings.device,
        save_dir=settings.save_dir,
        enable_visualization=False,
    )

    @app.on_event("startup")
    async def on_startup() -> None:
        emit_event(
            "APP",
            "startup_begin",
            log_file_path=settings.log_file_path,
            model_path=settings.model_path,
            device=settings.device,
            log_file=settings.log_file_path,
        )
        # 加载较慢，放到线程里避免阻塞事件循环心跳
        await asyncio.to_thread(app.state.engine.load)
        emit_event(
            "APP",
            "startup_ready",
            log_file_path=settings.log_file_path,
            engine_ready=bool(app.state.engine.ready),
        )

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> Dict[str, bool]:
        return {"ready": bool(app.state.engine.ready)}

    app.include_router(ws_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = ServerSettings.from_env()
    uvicorn.run("server.app:app", host=cfg.host, port=cfg.port, reload=False)
