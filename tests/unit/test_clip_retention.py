from __future__ import annotations

from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from app.config import StorageConfig
from app.models import ClassifierDecision, ClipMetadata, CompletedEvent, EventSummary
from app.storage.database import SQLiteRepository
from app.storage.retention import ClipRetentionManager


def _build_event(started_at: datetime) -> CompletedEvent:
    summary = EventSummary(
        source_name="test",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=2),
        duration_seconds=2.0,
        frame_count=4,
        peak_dbfs=-8.0,
        mean_dbfs=-18.0,
        mean_centroid_hz=900.0,
        dominant_freq_hz=700.0,
        dominant_span_hz=120.0,
        low_band_ratio=0.4,
        mid_band_ratio=0.4,
        high_band_ratio=0.2,
        mean_flux=0.2,
        mean_flatness=0.1,
        rms_modulation_depth=0.3,
        dominant_modulation_hz=0.6,
    )
    return CompletedEvent(summary=summary, clip_samples=np.zeros(32_000, dtype=np.float32), sample_rate=16_000)


def _insert_clip_event(
    repository: SQLiteRepository,
    clip_dir: Path,
    clip_name: str,
    byte_size: int,
    started_at: datetime,
) -> int:
    clip_path = clip_dir / clip_name
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"\x01" * byte_size)
    clip = ClipMetadata(
        path=clip_path,
        sample_rate=16_000,
        channels=1,
        duration_seconds=2.0,
        byte_size=byte_size,
        sha1=f"{clip_name:0<40}"[:40],
    )
    decision = ClassifierDecision(
        classifier_name="test",
        classifier_version="1",
        category="street_background",
        confidence=0.5,
        details={},
    )
    persisted = repository.insert_event(_build_event(started_at), decision, clip)
    return persisted.event_id


def _set_clip_created_at(repository: SQLiteRepository, event_id: int, created_at: datetime) -> None:
    timestamp = created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    with repository._connect() as connection:  # noqa: SLF001 - test helper
        row = connection.execute("SELECT clip_id FROM events WHERE id = ?", (event_id,)).fetchone()
        assert row is not None
        connection.execute("UPDATE clips SET created_at = ? WHERE id = ?", (timestamp, int(row["clip_id"])))
        connection.commit()


def test_prepare_for_clip_deletes_oldest_when_size_limit_exceeded(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clips"
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    now = datetime.now().astimezone()
    first_event = _insert_clip_event(repository, clip_dir, "older.wav", 450_000, now - timedelta(minutes=2))
    second_event = _insert_clip_event(repository, clip_dir, "newer.wav", 450_000, now - timedelta(minutes=1))
    _set_clip_created_at(repository, first_event, now - timedelta(days=2))
    _set_clip_created_at(repository, second_event, now - timedelta(days=1))

    manager = ClipRetentionManager(
        StorageConfig(
            database_path=repository.database_path,
            clip_dir=clip_dir,
            keep_clips=True,
            clip_max_megabytes=1,
            clip_max_age_days=0,
            min_free_disk_megabytes=0,
        ),
        repository,
    )

    assert manager.prepare_for_clip(200_000) is True
    assert not (clip_dir / "older.wav").exists()
    assert (clip_dir / "newer.wav").exists()
    assert repository.get_event(first_event)["clip_path"] is None
    assert repository.get_event(second_event)["clip_path"] is not None


def test_enforce_limits_deletes_clips_older_than_retention_window(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clips"
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    now = datetime.now().astimezone()
    expired_event = _insert_clip_event(repository, clip_dir, "expired.wav", 128_000, now - timedelta(days=10))
    fresh_event = _insert_clip_event(repository, clip_dir, "fresh.wav", 128_000, now - timedelta(days=1))
    _set_clip_created_at(repository, expired_event, now - timedelta(days=10))
    _set_clip_created_at(repository, fresh_event, now - timedelta(days=1))

    manager = ClipRetentionManager(
        StorageConfig(
            database_path=repository.database_path,
            clip_dir=clip_dir,
            keep_clips=True,
            clip_max_megabytes=0,
            clip_max_age_days=7,
            min_free_disk_megabytes=0,
        ),
        repository,
    )

    manager.enforce_limits()

    assert not (clip_dir / "expired.wav").exists()
    assert (clip_dir / "fresh.wav").exists()
    assert repository.get_event(expired_event)["clip_path"] is None
    assert repository.get_event(fresh_event)["clip_path"] is not None


def test_prepare_for_clip_returns_false_when_free_space_reserve_cannot_be_met(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    manager = ClipRetentionManager(
        StorageConfig(
            database_path=repository.database_path,
            clip_dir=clip_dir,
            keep_clips=True,
            clip_max_megabytes=0,
            clip_max_age_days=0,
            min_free_disk_megabytes=1,
        ),
        repository,
    )
    monkeypatch.setattr(
        "app.storage.retention.shutil.disk_usage",
        lambda _path: DiskUsage(total=10_000_000, used=9_900_000, free=100_000),
    )

    assert manager.prepare_for_clip(200_000) is False
