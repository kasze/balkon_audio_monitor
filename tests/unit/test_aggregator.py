from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from app.aggregate.event_aggregator import EventAggregator
from app.config import AggregationConfig
from app.models import AudioFrame, FrameFeatures


def make_frame(index: int) -> AudioFrame:
    return AudioFrame(
        samples=np.ones(8000, dtype=np.float32) * 0.1,
        started_at=datetime.now().astimezone() + timedelta(seconds=index * 0.5),
        duration_seconds=0.5,
        source_name="test",
    )


def make_features(index: int, dbfs: float = -35.0) -> FrameFeatures:
    return FrameFeatures(
        started_at=datetime.now().astimezone() + timedelta(seconds=index * 0.5),
        duration_seconds=0.5,
        source_name="test",
        rms=0.1,
        dbfs=dbfs,
        spectral_centroid_hz=900.0,
        dominant_freq_hz=700.0 + index * 30.0,
        low_band_ratio=0.3,
        mid_band_ratio=0.5,
        high_band_ratio=0.2,
        spectral_flux=0.02,
        flatness=0.2,
        zero_crossing_rate=0.1,
    )


def test_aggregator_merges_short_gap_into_single_event() -> None:
    aggregator = EventAggregator(
        AggregationConfig(
            pre_roll_seconds=0.5,
            post_roll_seconds=1.0,
            min_event_seconds=1.0,
            max_clip_seconds=5.0,
            max_event_seconds=30.0,
        ),
        sample_rate=16_000,
        frame_duration_seconds=0.5,
    )

    pattern = [True, True, False, True, True, False, False, False]
    completed = []
    for index, active in enumerate(pattern):
        completed.extend(aggregator.process(make_frame(index), make_features(index), active))

    assert len(completed) == 1
    assert completed[0].summary.duration_seconds >= 2.0
    assert completed[0].clip_samples.size > 0

