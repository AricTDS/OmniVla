from __future__ import annotations

import asyncio
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class LatestFrameBuffer(Generic[T]):
    """仅保留最新帧。push 会覆盖旧帧，pop 取出后清空。"""

    def __init__(self) -> None:
        self._item: Optional[T] = None
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def push(self, item: T) -> None:
        async with self._lock:
            self._item = item
            self._event.set()

    async def pop(self, timeout_s: Optional[float] = None) -> Optional[T]:
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None

        async with self._lock:
            item = self._item
            self._item = None
            self._event.clear()
            return item

    async def clear(self) -> None:
        async with self._lock:
            self._item = None
            self._event.clear()
