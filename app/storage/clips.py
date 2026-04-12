from __future__ import annotations

import hashlib
import logging
import uuid
import wave
from pathlib import Path

import numpy as np

from app.models import ClipMetadata, CompletedEvent
from app.storage.spectrogram import SpectrogramRenderer

LOGGER = logging.getLogger(__name__)


class ClipStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.spectrogram_renderer = SpectrogramRenderer()

    @staticmethod
    def estimate_wav_size(sample_count: int) -> int:
        return max(0, sample_count) * 2 + 44

    def estimate_total_size(self, sample_count: int) -> int:
        return self.estimate_wav_size(sample_count) + self.spectrogram_renderer.estimate_jpeg_size()

    def save(self, event: CompletedEvent) -> ClipMetadata | None:
        if event.clip_samples.size == 0:
            return None
        timestamp = event.summary.started_at.astimezone()
        clip_dir = self.base_dir / timestamp.strftime("%Y/%m/%d")
        clip_dir.mkdir(parents=True, exist_ok=True)
        stem = f"event_{timestamp.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}"
        path = clip_dir / f"{stem}.wav"
        spectrogram_path = clip_dir / f"{stem}.jpg"

        pcm = np.clip(event.clip_samples, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype("<i2")
        raw_bytes = pcm_i16.tobytes()
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(event.sample_rate)
            handle.writeframes(raw_bytes)

        spectrogram_byte_size = 0
        try:
            spectrogram_byte_size = self.spectrogram_renderer.save(event.clip_samples, event.sample_rate, spectrogram_path)
        except Exception as exc:
            LOGGER.warning("Failed to save spectrogram preview for %s: %s", path.name, exc)
            spectrogram_path = None

        return ClipMetadata(
            path=path,
            spectrogram_path=spectrogram_path,
            sample_rate=event.sample_rate,
            channels=1,
            duration_seconds=float(event.clip_samples.size / event.sample_rate),
            byte_size=path.stat().st_size,
            spectrogram_byte_size=spectrogram_byte_size,
            sha1=hashlib.sha1(raw_bytes).hexdigest(),
        )
