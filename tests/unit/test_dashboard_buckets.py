from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

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


def test_dashboard_groups_noise_into_ten_minute_buckets(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    day_start = datetime(2026, 4, 12, 12, 1).astimezone()

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
    assert round(float(dashboard["ten_minute"][1]["avg_dbfs"]), 1) == -40.0
