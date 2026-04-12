from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from app.models import FrameFeatures, NoiseInterval


@dataclass(slots=True)
class _NoiseAccumulatorState:
    features: list[FrameFeatures]


class NoiseIntervalCollector:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.reset()

    def reset(self) -> None:
        self._state = _NoiseAccumulatorState(features=[])

    def process(self, features: FrameFeatures) -> NoiseInterval | None:
        self._state.features.append(features)
        total_duration = sum(item.duration_seconds for item in self._state.features)
        if total_duration < self.interval_seconds:
            return None

        return self.flush()

    def flush(self) -> NoiseInterval | None:
        if not self._state.features:
            return None
        items = self._state.features
        self._state.features = []
        dbfs_values = np.array([item.dbfs for item in items], dtype=np.float32)
        rms_values = np.array([item.rms for item in items], dtype=np.float32)

        return NoiseInterval(
            source_name=items[0].source_name,
            started_at=items[0].started_at,
            ended_at=items[-1].started_at + timedelta(seconds=items[-1].duration_seconds),
            avg_rms=float(rms_values.mean()),
            avg_dbfs=float(dbfs_values.mean()),
            max_dbfs=float(dbfs_values.max()),
            avg_centroid_hz=float(np.mean([item.spectral_centroid_hz for item in items])),
            low_band_ratio=float(np.mean([item.low_band_ratio for item in items])),
            mid_band_ratio=float(np.mean([item.mid_band_ratio for item in items])),
            high_band_ratio=float(np.mean([item.high_band_ratio for item in items])),
        )
