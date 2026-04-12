from __future__ import annotations

import wave
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from app.capture.base import AudioCaptureError
from app.models import AudioFrame


class WavFileSource:
    def __init__(self, path: Path, sample_rate: int, frame_duration_seconds: float) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.frame_duration_seconds = frame_duration_seconds
        self._handle: wave.Wave_read | None = None

    def frames(self) -> Iterator[AudioFrame]:
        samples_per_frame = int(self.sample_rate * self.frame_duration_seconds)
        with wave.open(str(self.path), "rb") as handle:
            self._handle = handle
            if handle.getsampwidth() != 2:
                raise AudioCaptureError(f"{self.path} is not a 16-bit PCM WAV file.")
            if handle.getframerate() != self.sample_rate:
                raise AudioCaptureError(
                    f"{self.path} has sample rate {handle.getframerate()} Hz, expected {self.sample_rate} Hz."
                )

            channels = handle.getnchannels()
            if channels < 1:
                raise AudioCaptureError(f"{self.path} does not expose a readable audio channel.")

            file_started_at = datetime.fromtimestamp(self.path.stat().st_mtime).astimezone()
            frame_index = 0
            while True:
                raw = handle.readframes(samples_per_frame)
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                if samples.size < samples_per_frame:
                    samples = np.pad(samples, (0, samples_per_frame - samples.size))
                yield AudioFrame(
                    samples=samples,
                    started_at=file_started_at + timedelta(seconds=frame_index * self.frame_duration_seconds),
                    duration_seconds=self.frame_duration_seconds,
                    source_name=self.path.name,
                )
                frame_index += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
