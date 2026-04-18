from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from app.models import ClassifierDecision, CompletedEvent, EventSummary
from app.models import NoiseInterval
from app.storage.database import SQLiteRepository


def _noise_interval(started_at: datetime, avg_dbfs: float, max_dbfs: float) -> NoiseInterval:
    return NoiseInterval(
        source_name="test",
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=1),
        avg_rms=0.1,
        avg_dbfs=avg_dbfs,
        max_dbfs=max_dbfs,
        avg_centroid_hz=500.0,
        low_band_ratio=0.3,
        mid_band_ratio=0.4,
        high_band_ratio=0.3,
    )


def _event(started_at: datetime) -> CompletedEvent:
    summary = EventSummary(
        source_name="test",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=4),
        duration_seconds=4.0,
        frame_count=8,
        peak_dbfs=-12.0,
        mean_dbfs=-20.0,
        mean_centroid_hz=500.0,
        dominant_freq_hz=700.0,
        dominant_span_hz=120.0,
        low_band_ratio=0.3,
        mid_band_ratio=0.4,
        high_band_ratio=0.3,
        mean_flux=0.1,
        mean_flatness=0.1,
        rms_modulation_depth=0.2,
        dominant_modulation_hz=0.7,
    )
    return CompletedEvent(summary=summary, clip_samples=np.array([], dtype=np.float32), sample_rate=16_000)


def test_dashboard_groups_noise_into_ten_minute_buckets(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    day_start = datetime(2026, 4, 12, 12, 1).astimezone()

    repository.insert_event(_event(day_start), ClassifierDecision("yamnet_litert", "1", "speech", 0.9, {}), None)
    repository.insert_event(_event(day_start + timedelta(minutes=7)), ClassifierDecision("yamnet_litert", "1", "speech", 0.9, {}), None)
    repository.insert_event(_event(day_start + timedelta(minutes=11)), ClassifierDecision("yamnet_litert", "1", "speech", 0.9, {}), None)
    repository.insert_noise_interval(_noise_interval(day_start, -30.0, -10.0))
    repository.insert_noise_interval(_noise_interval(day_start + timedelta(minutes=7), -20.0, -8.0))
    repository.insert_noise_interval(_noise_interval(day_start + timedelta(minutes=11), -40.0, -12.0))

    dashboard = repository.get_dashboard(day_start.strftime("%Y-%m-%d"), 10)

    assert [row["bucket_start"] for row in dashboard["ten_minute"]] == [
        "2026-04-12 12:00:00",
        "2026-04-12 12:10:00",
    ]
    assert round(float(dashboard["ten_minute"][0]["avg_dbfs"]), 1) == -25.0
    assert round(float(dashboard["ten_minute"][0]["max_dbfs"]), 1) == -8.0
    assert int(dashboard["ten_minute"][0]["event_count"]) == 2
    assert round(float(dashboard["ten_minute"][1]["avg_dbfs"]), 1) == -40.0
    assert int(dashboard["ten_minute"][1]["event_count"]) == 1


def test_dashboard_groups_noise_by_hour_and_six_hour_for_wider_ranges(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    start = datetime(2026, 4, 12, 12, 1).astimezone()

    repository.insert_event(_event(start), ClassifierDecision("yamnet_litert", "1", "speech", 0.9, {}), None)
    repository.insert_event(_event(start + timedelta(minutes=7)), ClassifierDecision("yamnet_litert", "1", "speech", 0.9, {}), None)
    repository.insert_event(_event(start + timedelta(days=40)), ClassifierDecision("yamnet_litert", "1", "speech", 0.9, {}), None)
    repository.insert_noise_interval(_noise_interval(start, -30.0, -10.0))
    repository.insert_noise_interval(_noise_interval(start + timedelta(days=1), -20.0, -8.0))
    repository.insert_noise_interval(_noise_interval(start + timedelta(days=40), -40.0, -12.0))

    weekly = repository.get_dashboard_range(
        started_at="2026-04-12 00:00:00",
        ended_at="2026-04-20 00:00:00",
        recent_limit=10,
        bucket_mode="hour",
    )
    yearly = repository.get_dashboard_range(
        started_at="2026-01-01 00:00:00",
        ended_at="2027-01-01 00:00:00",
        recent_limit=10,
        bucket_mode="six_hour",
    )

    assert [row["bucket_start"] for row in weekly["ten_minute"]] == [
        "2026-04-12 12:00:00",
        "2026-04-13 12:00:00",
    ]
    assert [int(row["event_count"]) for row in weekly["ten_minute"]] == [2, 0]
    assert [row["bucket_start"] for row in yearly["ten_minute"]] == [
        "2026-04-12 12:00:00",
        "2026-04-13 12:00:00",
        "2026-05-22 12:00:00",
    ]
    assert [int(row["event_count"]) for row in yearly["ten_minute"]] == [2, 0, 1]
