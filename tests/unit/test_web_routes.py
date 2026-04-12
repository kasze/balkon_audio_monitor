from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from app.config import AppConfig, StorageConfig
from app.models import ClassifierDecision, ClipMetadata, CompletedEvent, EventSummary
from app.pipeline import RuntimeStatus
from app.storage.database import SQLiteRepository
from app.web.app import create_app


def _build_event(category: str, started_at: datetime) -> CompletedEvent:
    summary = EventSummary(
        source_name="test",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=4),
        duration_seconds=4.0,
        frame_count=8,
        peak_dbfs=-10.0,
        mean_dbfs=-20.0,
        mean_centroid_hz=900.0,
        dominant_freq_hz=700.0,
        dominant_span_hz=120.0,
        low_band_ratio=0.4,
        mid_band_ratio=0.4,
        high_band_ratio=0.2,
        mean_flux=0.1,
        mean_flatness=0.1,
        rms_modulation_depth=0.2,
        dominant_modulation_hz=0.7,
    )
    return CompletedEvent(summary=summary, clip_samples=np.array([], dtype=np.float32), sample_rate=16_000)


def test_category_page_lists_filtered_events(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    now = datetime.now().astimezone().replace(microsecond=0)

    repository.insert_event(
        _build_event("ambulance", now),
        ClassifierDecision("yamnet_litert", "1", "ambulance", 0.9, {}),
        None,
    )
    repository.insert_event(
        _build_event("street_background", now - timedelta(minutes=5)),
        ClassifierDecision("yamnet_litert", "1", "street_background", 0.7, {}),
        None,
    )

    app = create_app(
        repository,
        RuntimeStatus(),
        AppConfig(
            base_dir=tmp_path,
            storage=StorageConfig(database_path=repository.database_path, clip_dir=tmp_path / "clips"),
        ),
    )
    client = app.test_client()

    response = client.get("/categories/ambulance")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "/events/1" in html
    assert "Karetka / syrena karetki" in html
    assert "/categories/ambulance" in html
    assert "/events/2" not in html


def test_manual_label_updates_event_and_category_views(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    now = datetime.now().astimezone().replace(microsecond=0)

    repository.insert_event(
        _build_event("street_background", now),
        ClassifierDecision("yamnet_litert", "1", "street_background", 0.7, {}),
        None,
    )

    app = create_app(
        repository,
        RuntimeStatus(),
        AppConfig(
            base_dir=tmp_path,
            storage=StorageConfig(database_path=repository.database_path, clip_dir=tmp_path / "clips"),
        ),
    )
    client = app.test_client()

    response = client.post("/events/1/label", data={"user_label": "ambulance"}, follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Ręczna korekta:" in html
    assert "Karetka / syrena karetki" in html

    category_response = client.get("/categories/ambulance")
    assert category_response.status_code == 200
    assert "/events/1" in category_response.get_data(as_text=True)


def test_manual_label_accepts_full_yamnet_label(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    now = datetime.now().astimezone().replace(microsecond=0)

    repository.insert_event(
        _build_event("street_background", now),
        ClassifierDecision("yamnet_litert", "1", "street_background", 0.7, {}),
        None,
    )

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "yamnet_class_map.csv").write_text(
        "index,mid,display_name\n0,/m/09x0r,Speech\n1,/m/01h8n0,Conversation\n",
        encoding="utf-8",
    )

    app = create_app(
        repository,
        RuntimeStatus(),
        AppConfig(
            base_dir=tmp_path,
            storage=StorageConfig(database_path=repository.database_path, clip_dir=tmp_path / "clips"),
        ),
    )
    client = app.test_client()

    response = client.post("/events/1/label", data={"user_label": "Speech"}, follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Mowa" in html


def test_event_details_show_clip_duration_separately_from_event_duration(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    now = datetime.now().astimezone().replace(microsecond=0)

    clip_path = tmp_path / "clips" / "sample.wav"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"RIFFdemo")

    spectrogram_path = tmp_path / "clips" / "sample.jpg"
    spectrogram_path.write_bytes(b"jpeg")

    repository.insert_event(
        _build_event("street_background", now),
        ClassifierDecision("yamnet_litert", "1", "street_background", 0.7, {}),
        ClipMetadata(
            path=clip_path,
            spectrogram_path=spectrogram_path,
            sample_rate=16_000,
            channels=1,
            duration_seconds=4.0,
            byte_size=clip_path.stat().st_size,
            spectrogram_byte_size=spectrogram_path.stat().st_size,
            sha1="demo",
        ),
    )

    app = create_app(
        repository,
        RuntimeStatus(),
        AppConfig(
            base_dir=tmp_path,
            storage=StorageConfig(database_path=repository.database_path, clip_dir=tmp_path / "clips"),
        ),
    )
    client = app.test_client()

    response = client.get("/events/1")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Czas zdarzenia:</strong> 4.0 s" in html
    assert "Czas próbki audio:</strong> 4.0 s" in html


def test_health_includes_human_uptime(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    status = RuntimeStatus()
    status.update(started_at="2026-04-10T09:40:00+02:00")

    app = create_app(
        repository,
        status,
        AppConfig(
            base_dir=tmp_path,
            storage=StorageConfig(database_path=repository.database_path, clip_dir=tmp_path / "clips"),
        ),
    )
    client = app.test_client()

    from app.web import app as web_app_module

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = cls.fromisoformat("2026-04-12T23:01:00+02:00")
            return current.astimezone(tz) if tz is not None else current

    original_datetime = web_app_module.datetime
    web_app_module.datetime = FrozenDateTime
    try:
        response = client.get("/health")
    finally:
        web_app_module.datetime = original_datetime

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["uptime_human"] == "2d 13h 21m"
