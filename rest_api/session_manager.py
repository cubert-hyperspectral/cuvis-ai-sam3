"""Thread-safe session metadata store."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor


@dataclass
class SessionInfo:
    """Lightweight metadata kept alongside the SAM3 inference state."""

    session_id: str
    video_path: str
    num_frames: int
    frame_rate: float
    width: int
    height: int
    created_at: float = field(default_factory=time.time)


class SessionManager:
    """Thread-safe registry of active sessions with expiry support."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionInfo] = {}
        self._lock = threading.Lock()

    def register(
        self,
        session_id: str,
        video_path: str,
        inference_state: dict,
    ) -> SessionInfo:
        """Register a new session after ``start_session`` succeeds."""
        video = inference_state.get("_video_iterator")
        info = SessionInfo(
            session_id=session_id,
            video_path=video_path,
            num_frames=inference_state["num_frames"],
            frame_rate=getattr(video, "frame_rate", 0.0),
            width=inference_state["orig_width"],
            height=inference_state["orig_height"],
        )
        with self._lock:
            self._sessions[session_id] = info
        return info

    def get(self, session_id: str) -> SessionInfo:
        """Return session info or raise ``KeyError``."""
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError:
                raise KeyError(f"Session {session_id!r} not found") from None

    def remove(self, session_id: str) -> None:
        """Remove session metadata (idempotent)."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_expired(
        self,
        timeout_seconds: int,
        predictor: Sam3VideoPredictor,
    ) -> list[str]:
        """Close and remove sessions older than *timeout_seconds*.

        Returns the list of expired session IDs.
        """
        now = time.time()
        expired: list[str] = []
        with self._lock:
            for sid, info in list(self._sessions.items()):
                if now - info.created_at > timeout_seconds:
                    expired.append(sid)

        # Close outside the lock to avoid holding it during model cleanup.
        for sid in expired:
            predictor.close_session(sid)
            self.remove(sid)

        return expired

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)
