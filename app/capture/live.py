from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime

import numpy as np

from app.capture.base import AudioCaptureError
from app.models import AudioFrame

LOGGER = logging.getLogger(__name__)


class LiveAudioSource:
    def __init__(
        self,
        sample_rate: int,
        frame_duration_seconds: float,
        arecord_binary: str = "arecord",
        arecord_device: str | None = None,
        channels: int = 1,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_seconds = frame_duration_seconds
        self.arecord_binary = arecord_binary
        self.arecord_device = arecord_device
        self.channels = channels
        self._process: subprocess.Popen[bytes] | None = None

    def _command(self) -> list[str]:
        command = [
            self.arecord_binary,
            "-q",
            "-f",
            "S16_LE",
            "-r",
            str(self.sample_rate),
            "-c",
            str(self.channels),
            "-t",
            "raw",
        ]
        if self.arecord_device:
            command.extend(["-D", self.arecord_device])
        return command

    def frames(self) -> Iterator[AudioFrame]:
        samples_per_frame = int(self.sample_rate * self.frame_duration_seconds)
        bytes_per_frame = samples_per_frame * self.channels * 2
        command = self._command()
        LOGGER.info("Starting live capture with command: %s", " ".join(command))
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=bytes_per_frame * 4,
        )

        if self._process.stdout is None or self._process.stderr is None:
            raise AudioCaptureError("Failed to open arecord pipes.")

        try:
            while True:
                started_at = datetime.now(tz=UTC)
                chunk = self._process.stdout.read(bytes_per_frame)
                if len(chunk) != bytes_per_frame:
                    stderr_output = self._process.stderr.read().decode("utf-8", errors="replace").strip()
                    raise AudioCaptureError(
                        "Live audio source stopped unexpectedly."
                        + (f" arecord output: {stderr_output}" if stderr_output else "")
                    )
                samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
                if self.channels > 1:
                    samples = samples.reshape(-1, self.channels).mean(axis=1)
                yield AudioFrame(
                    samples=samples,
                    started_at=started_at,
                    duration_seconds=self.frame_duration_seconds,
                    source_name=self.arecord_device or "live",
                )
        finally:
            self.close()

    def close(self) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

