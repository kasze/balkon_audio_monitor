from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from app.config import AppConfig, StorageConfig
from app.models import ClassifierDecision, CompletedEvent, EventSummary
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
    assert "Ambulance / syrena karetki" in html
    assert "/categories/ambulance" in html
    assert "/events/2" not in html
