from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from app.models import AudioFrame


class AudioCaptureError(RuntimeError):
    """Raised when audio capture cannot continue."""


class AudioSource(Protocol):
    def frames(self) -> Iterator[AudioFrame]:
        ...

    def close(self) -> None:
        ...

