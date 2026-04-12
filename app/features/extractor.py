from __future__ import annotations

import math

import numpy as np

from app.models import AudioFrame, FrameFeatures


class FeatureExtractor:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._previous_spectrum: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_spectrum = None

    def extract(self, frame: AudioFrame) -> FrameFeatures:
        samples = frame.samples.astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
        dbfs = 20.0 * math.log10(max(rms, 1e-9))

        windowed = samples * np.hanning(samples.size)
        spectrum = np.abs(np.fft.rfft(windowed))
        power = np.square(spectrum) + 1e-12
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / self.sample_rate)

        power_sum = float(power.sum())
        centroid = float(np.sum(freqs * power) / power_sum)

        dominant_index = int(np.argmax(power[1:]) + 1) if power.size > 1 else 0
        dominant_freq = float(freqs[dominant_index]) if dominant_index < freqs.size else 0.0

        low_ratio = self._band_ratio(freqs, power, 50.0, 250.0)
        mid_ratio = self._band_ratio(freqs, power, 250.0, 1_000.0)
        high_ratio = self._band_ratio(freqs, power, 1_000.0, 4_000.0)

        normalized = power / power_sum
        spectral_flux = 0.0
        if self._previous_spectrum is not None and self._previous_spectrum.shape == normalized.shape:
            spectral_flux = float(np.mean(np.square(normalized - self._previous_spectrum)))
        self._previous_spectrum = normalized

        flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
        zero_crossing_rate = float(np.mean(np.abs(np.diff(np.signbit(samples)))))

        return FrameFeatures(
            started_at=frame.started_at,
            duration_seconds=frame.duration_seconds,
            source_name=frame.source_name,
            rms=rms,
            dbfs=dbfs,
            spectral_centroid_hz=centroid,
            dominant_freq_hz=dominant_freq,
            low_band_ratio=low_ratio,
            mid_band_ratio=mid_ratio,
            high_band_ratio=high_ratio,
            spectral_flux=spectral_flux,
            flatness=flatness,
            zero_crossing_rate=zero_crossing_rate,
        )

    @staticmethod
    def _band_ratio(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            return 0.0
        total = float(power.sum())
        return float(power[mask].sum() / total) if total else 0.0

