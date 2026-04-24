from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

_LOG_FILE_LOCK = threading.Lock()


def _ensure_parent_dir(log_file_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(log_file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def default_client_replay_log_file_path() -> str:
    """与 default_log_file_path 类似，供 tools/ws_replay_client 等客户端落盘。"""
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"./logs/ws_replay_client_{ts}.log"


def _format_event(prefix: str, event: str, **kwargs: Any) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    detail = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
    if detail:
        return f"[{prefix}][{ts}] {event} | {detail}"
    return f"[{prefix}][{ts}] {event}"


def emit_event(prefix: str, event: str, log_file_path: Optional[str] = None, **kwargs: Any) -> None:
    """打印到终端并同步追加到日志文件。"""
    line = _format_event(prefix=prefix, event=event, **kwargs)
    print(line, flush=True)

    if not log_file_path:
        return

    _ensure_parent_dir(log_file_path)
    with _LOG_FILE_LOCK:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
