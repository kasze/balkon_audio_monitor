from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from app.config import AppConfig, ClassifierConfig, StorageConfig, load_config
from app.models import ClassifierDecision, ClipMetadata, CompletedEvent, EventSummary, NoiseInterval
from app.pipeline import RuntimeStatus
from app.storage.database import SQLiteRepository
from app.web.app import create_app


def _build_event(category: str, started_at: datetime, *, duration_seconds: float = 4.0) -> CompletedEvent:
    summary = EventSummary(
        source_name="test",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=duration_seconds),
        duration_seconds=duration_seconds,
        frame_count=max(1, int(duration_seconds * 2)),
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


def test_birdnet_results_show_badge_in_lists(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    now = datetime.now().astimezone().replace(microsecond=0)

    repository.insert_event(
        _build_event("Bird vocalization, bird call, bird song", now),
        ClassifierDecision(
            "birdnet_remote",
            "1",
            "Bogatka",
            0.88,
            {
                "used_external_api": True,
                "external_api_name": "BirdNET API",
                "birdnet_common_name": "Bogatka",
                "birdnet_scientific_name": "Parus major",
                "birdnet_trigger_labels": ["Bird", "Animal"],
                "birdnet_trigger_summary": "BirdNET uruchomiono przez etykietę ptasią i zwierzęcą YAMNet",
            },
        ),
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

    dashboard_html = client.get("/").get_data(as_text=True)
    birds_html = client.get("/birds").get_data(as_text=True)

    assert "BirdNET" in dashboard_html
    assert "BirdNET" in birds_html


def test_recent_events_show_birdnet_badge_for_remote_results(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    now = datetime.now().astimezone().replace(microsecond=0)

    repository.insert_event(
        _build_event("Bird vocalization, bird call, bird song", now),
        ClassifierDecision(
            "birdnet_remote",
            "1",
            "Bogatka",
            0.88,
            {
                "used_external_api": True,
                "external_api_name": "BirdNET API",
                "birdnet_common_name": "Bogatka",
                "birdnet_scientific_name": "Parus major",
                "birdnet_trigger_labels": ["Bird", "Animal"],
                "birdnet_trigger_summary": "BirdNET uruchomiono przez etykietę ptasią i zwierzęcą YAMNet",
            },
        ),
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

    html = client.get("/").get_data(as_text=True)
    assert "BirdNET" in html


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


def test_event_details_and_lists_prefer_clip_duration_over_event_duration(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    now = datetime.now().astimezone().replace(microsecond=0)

    clip_path = tmp_path / "clips" / "sample.wav"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"RIFFdemo")

    spectrogram_path = tmp_path / "clips" / "sample.jpg"
    spectrogram_path.write_bytes(b"jpeg")

    repository.insert_event(
        _build_event("street_background", now, duration_seconds=90.0),
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
    assert "Czas nagrania:</strong> 4.0 s" in html
    assert "Czas pełnego zdarzenia:</strong> 90.0 s" in html

    dashboard_response = client.get("/")
    assert dashboard_response.status_code == 200
    dashboard_html = dashboard_response.get_data(as_text=True)
    assert "Czas nagrania" in dashboard_html
    assert "90.0 s" not in dashboard_html
    assert "4.0 s" in dashboard_html


def test_health_includes_human_uptime(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    status = RuntimeStatus()

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

    original_read_text = web_app_module.Path.read_text

    def fake_read_text(self, encoding="utf-8"):
        if str(self) == "/proc/uptime":
            return "220860.42 99999.00\n"
        return original_read_text(self, encoding=encoding)

    web_app_module.Path.read_text = fake_read_text
    try:
        response = client.get("/health")
    finally:
        web_app_module.Path.read_text = original_read_text

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["system_uptime_human"] == "2d 13h 21m"


def test_birds_page_and_dashboard_show_recent_bird_species(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    now = datetime.now().astimezone().replace(microsecond=0)

    repository.insert_event(
        _build_event("Bird vocalization, bird call, bird song", now),
        ClassifierDecision(
            "birdnet_remote",
            "1",
            "Bogatka",
            0.88,
            {
                "used_external_api": True,
                "external_api_name": "BirdNET API",
                "birdnet_common_name": "Bogatka",
                "birdnet_scientific_name": "Parus major",
            },
        ),
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

    dashboard_response = client.get("/")
    assert dashboard_response.status_code == 200
    dashboard_html = dashboard_response.get_data(as_text=True)
    assert 'href="/birds"' in dashboard_html
    assert "Bogatka" in dashboard_html
    assert "BirdNET" in dashboard_html

    birds_response = client.get("/birds")
    assert birds_response.status_code == 200
    birds_html = birds_response.get_data(as_text=True)
    assert "Ostatnio rozpoznane gatunki" in birds_html
    assert "Bogatka" in birds_html
    assert "BirdNET" in birds_html
    assert "/events/1" in birds_html
    assert "street_background" not in birds_html


def test_dashboard_supports_period_navigation(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    now = datetime(2026, 4, 13, 10, 0).astimezone().replace(microsecond=0)

    repository.insert_noise_interval(
        NoiseInterval(
            source_name="test",
            started_at=now,
            ended_at=now + timedelta(minutes=1),
            avg_rms=0.1,
            avg_dbfs=-30.0,
            max_dbfs=-10.0,
            avg_centroid_hz=500.0,
            low_band_ratio=0.3,
            mid_band_ratio=0.4,
            high_band_ratio=0.3,
        )
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

    response = client.get("/?period=week&date=2026-04-13")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Tydzień" in html
    assert 'class="segment-link active"' in html
    assert 'data-calendar-toggle' in html
    assert 'href="/birds?period=week&amp;date=2026-04-13"' in html


def test_settings_page_renders_and_saves_config(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "audio_monitor.sqlite3")
    repository.initialize()
    config_path = tmp_path / "configs" / "config.yaml"
    app = create_app(
        repository,
        RuntimeStatus(),
        AppConfig(
            base_dir=tmp_path,
            storage=StorageConfig(database_path=repository.database_path, clip_dir=tmp_path / "clips"),
            classifier=ClassifierConfig(
                yamnet_model_path=tmp_path / "models" / "yamnet.tflite",
                yamnet_class_map_path=tmp_path / "models" / "yamnet_class_map.csv",
            ),
            config_path=config_path,
        ),
    )
    client = app.test_client()

    response = client.get("/settings")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Ustawienia" in html
    assert "Balkon miejski" in html
    assert "Pokój / wnętrze" in html

    post_response = client.post(
        "/settings",
        data={
            "preset": "room",
            "audio.arecord_device_mode": "auto",
            "audio.arecord_device": "",
            "audio.sample_rate": "16000",
            "audio.channels": "1",
            "audio.frame_duration_seconds": "0.5",
            "audio.retry_backoff_seconds": "5",
            "detection.initial_noise_floor_dbfs": "-58",
            "detection.activation_margin_db": "11",
            "detection.release_margin_db": "4",
            "detection.min_event_dbfs": "-44",
            "detection.min_active_frames": "3",
            "detection.max_inactive_frames": "2",
            "detection.noise_floor_alpha": "0.04",
            "aggregation.noise_interval_seconds": "5",
            "aggregation.pre_roll_seconds": "1.0",
            "aggregation.post_roll_seconds": "0.5",
            "aggregation.min_event_seconds": "1.0",
            "aggregation.focus_clip_seconds": "8.0",
            "aggregation.max_clip_seconds": "30.0",
            "aggregation.max_event_seconds": "25.0",
            "classifier.backend": "yamnet",
            "classifier.yamnet_num_threads": "1",
            "classifier.yamnet_max_analysis_seconds": "12.0",
            "classifier.yamnet_max_windows": "24",
            "classifier.yamnet_min_category_score": "0.08",
            "classifier.yamnet_top_k": "8",
            "classifier.birdnet_timeout_seconds": "15",
            "classifier.birdnet_min_confidence": "0.20",
            "classifier.birdnet_num_results": "5",
            "classifier.birdnet_locale": "pl",
            "classifier.birdnet_api_url": "http://birdnet.local",
            "storage.keep_clips": "true",
            "storage.clip_max_megabytes": "512",
            "storage.clip_max_age_days": "10",
            "storage.min_free_disk_megabytes": "256",
            "storage.database_path": str(repository.database_path),
            "storage.clip_dir": str(tmp_path / "clips"),
            "web.host": "0.0.0.0",
            "web.port": "8080",
            "web.recent_events_limit": "25",
            "web.dashboard_history_hours": "24",
            "logging.level": "INFO",
        },
    )

    assert post_response.status_code == 200
    saved = load_config(config_path)
    assert saved.classifier.birdnet_api_url == "http://birdnet.local"
    assert saved.detection.activation_margin_db == 9.0
    assert saved.detection.min_event_dbfs == -47.0
    assert saved.aggregation.max_event_seconds == 15.0
