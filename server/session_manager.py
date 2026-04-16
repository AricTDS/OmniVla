from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image


@dataclass
class SessionState:
    session_id: str
    goal_image_pil: Image.Image
    instruction: Optional[str] = None
    fps: int = 4
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    last_seq: int = -1


class SessionManager:
    """v1: 单会话优先。新 start_session 会覆盖旧会话。"""

    def __init__(self) -> None:
        self._session: Optional[SessionState] = None
        self._lock = asyncio.Lock()

    async def start_session(self, session: SessionState) -> None:
        async with self._lock:
            self._session = session

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        async with self._lock:
            if self._session and self._session.session_id == session_id:
                return self._session
            return None

    async def update_last_seq(self, session_id: str, seq: int) -> None:
        async with self._lock:
            if self._session and self._session.session_id == session_id:
                self._session.last_seq = seq

    async def stop_session(self, session_id: str) -> bool:
        async with self._lock:
            if self._session and self._session.session_id == session_id:
                self._session = None
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._session = None

    async def active_session_id(self) -> Optional[str]:
        async with self._lock:
            return self._session.session_id if self._session else None
