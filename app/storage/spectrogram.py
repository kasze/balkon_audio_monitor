from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class SpectrogramRenderer:
    def __init__(self, width: int = 640, height: int = 240) -> None:
        self.width = width
        self.height = height

    @staticmethod
    def estimate_jpeg_size() -> int:
        return 120_000

    def save(self, samples: np.ndarray, sample_rate: int, path: Path) -> int:
        image = self._build_image(samples, sample_rate)
        image.save(path, format="JPEG", quality=82, optimize=True)
        return path.stat().st_size

    def _build_image(self, samples: np.ndarray, sample_rate: int) -> Image.Image:
        normalized = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
        if normalized.size == 0:
            normalized = np.zeros(1024, dtype=np.float32)

        window_size = 512 if sample_rate >= 16_000 else 256
        hop_size = max(window_size // 4, 64)
        padded = normalized
        if padded.size < window_size:
            padded = np.pad(padded, (0, window_size - padded.size))

        frame_count = 1 + max(0, (padded.size - window_size) // hop_size)
        if frame_count <= 0:
            frame_count = 1

        window = np.hanning(window_size).astype(np.float32)
        columns: list[np.ndarray] = []
        for index in range(frame_count):
            start = index * hop_size
            chunk = padded[start : start + window_size]
            if chunk.size < window_size:
                chunk = np.pad(chunk, (0, window_size - chunk.size))
            spectrum = np.fft.rfft(chunk * window)
            magnitude = np.abs(spectrum).astype(np.float32)
            columns.append(magnitude)

        matrix = np.stack(columns, axis=1)
        matrix = np.log10(np.maximum(matrix, 1e-6))
        matrix -= matrix.min()
        max_value = float(matrix.max()) if matrix.size else 0.0
        if max_value > 0.0:
            matrix /= max_value
        matrix = np.power(matrix, 0.7)
        matrix = np.flipud(matrix)
        image_bytes = np.clip(matrix * 255.0, 0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(image_bytes)
        return image.resize((self.width, self.height), resample=Image.Resampling.BILINEAR)
