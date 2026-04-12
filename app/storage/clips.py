from __future__ import annotations

import hashlib
import uuid
import wave
from datetime import UTC
from pathlib import Path

import numpy as np

from app.models import ClipMetadata, CompletedEvent


class ClipStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def save(self, event: CompletedEvent) -> ClipMetadata | None:
        if event.clip_samples.size == 0:
            return None
        timestamp = event.summary.started_at.astimezone()
        clip_dir = self.base_dir / timestamp.strftime("%Y/%m/%d")
        clip_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"event_{timestamp.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
        path = clip_dir / file_name

        pcm = np.clip(event.clip_samples, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype("<i2")
        raw_bytes = pcm_i16.tobytes()
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(event.sample_rate)
            handle.writeframes(raw_bytes)

        return ClipMetadata(
            path=path,
            sample_rate=event.sample_rate,
            channels=1,
            duration_seconds=float(event.clip_samples.size / event.sample_rate),
            byte_size=path.stat().st_size,
            sha1=hashlib.sha1(raw_bytes).hexdigest(),
        )
