from __future__ import annotations

import os
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from app.config import AppConfig, StorageConfig
from app.web.app import (
    _describe_classifier_decision,
    _format_dbfs,
    _format_local_timestamp,
    _format_uptime_seconds,
    _read_cpu_load_percent,
    _read_cpu_temperature_c,
    _read_disk_free_gb,
    _read_memory_available_gb,
    _read_system_uptime_seconds,
    _read_system_status,
    _translate_classifier_name,
    _translate_label,
)


@pytest.fixture
def warsaw_timezone() -> None:
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is not available on this platform")

    original_timezone = os.environ.get("TZ")
    os.environ["TZ"] = "Europe/Warsaw"
    time.tzset()
    try:
        yield
    finally:
        if original_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_timezone
        time.tzset()


def test_format_local_timestamp_uses_local_timezone(warsaw_timezone: None) -> None:
    formatted = _format_local_timestamp("2026-04-12T10:57:32.228818+00:00")
    assert formatted == "2026-04-12 12:57:32.22"


def test_format_local_timestamp_returns_original_on_parse_error() -> None:
    assert _format_local_timestamp("not-a-timestamp") == "not-a-timestamp"


def test_format_dbfs_normalizes_negative_zero() -> None:
    assert _format_dbfs(-0.00003) == "0.0 dBFS"
    assert _format_dbfs(0.0) == "0.0 dBFS"
    assert _format_dbfs(-12.34) == "-12.3 dBFS"


def test_format_uptime_returns_human_readable_duration() -> None:
    assert _format_uptime_seconds(220_860) == "2d 13h 21m"


def test_describe_classifier_decision_marks_cache_reuse() -> None:
    trace = _describe_classifier_decision(
        {
            "classifier_name": "yamnet_litert",
            "classifier_version": "1",
            "details": {
                "cache_hit": True,
                "cache_similarity": 0.9987,
                "cache_source_event_id": 42,
                "top_labels": [{"label": "Siren", "mean_score": 0.41, "peak_score": 0.87}],
                "category_scores": {"ambulance": 0.87, "street_background": 0.23},
            },
        }
    )

    assert trace["source"] == "cache_reuse"
    assert trace["used_external_api"] is False
    assert trace["cache_source_event_id"] == 42
    assert trace["top_labels"][0]["label"] == "Siren"
    assert trace["category_scores"][0]["category"] == "ambulance"
    assert trace["classifier_label"] == "Lokalny YAMNet (LiteRT)"
    assert trace["source_label"] == "Ponowne użycie z lokalnego cache"


def test_describe_classifier_decision_marks_external_api_when_present() -> None:
    trace = _describe_classifier_decision(
        {
            "classifier_name": "birdnet_remote",
            "classifier_version": "1",
            "details": {
                "used_external_api": True,
                "external_api_name": "BirdNET API",
                "birdnet_common_name": "Bogatka",
                "birdnet_scientific_name": "Parus major",
                "birdnet_trigger_labels": ["Bird", "Bird vocalization, bird call, bird song"],
            },
        }
    )

    assert trace["source"] == "external_api"
    assert trace["used_external_api"] is True
    assert trace["external_api_name"] == "BirdNET API"
    assert trace["birdnet_common_name"] == "Bogatka"
    assert trace["birdnet_scientific_name"] == "Parus major"
    assert trace["birdnet_trigger_labels"] == ["Bird", "Bird vocalization, bird call, bird song"]


def test_read_cpu_load_percent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.web.app.os.cpu_count", lambda: 4)
    monkeypatch.setattr("app.web.app.os.getloadavg", lambda: (1.5, 1.0, 0.5))

    assert _read_cpu_load_percent() == 37.5


def test_read_system_uptime_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.web.app.Path.read_text", lambda self, encoding="utf-8": "220860.42 99999.00\n")

    assert _read_system_uptime_seconds() == 220860.42


def test_read_cpu_temperature_from_sysfs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.web.app.Path.exists", lambda self: True)
    monkeypatch.setattr("app.web.app.Path.read_text", lambda self, encoding="utf-8": "48678\n")

    assert _read_cpu_temperature_c() == 48.7


def test_read_cpu_temperature_from_vcgencmd_when_sysfs_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.web.app.Path.exists", lambda self: False)
    monkeypatch.setattr(
        "app.web.app.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args=args[0], returncode=0, stdout="temp=51.2'C\n", stderr=""),
    )

    assert _read_cpu_temperature_c() == 51.2


def test_read_memory_and_disk_and_compose_system_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])
    meminfo = "MemTotal:       2048000 kB\nMemAvailable:   1048576 kB\n"
    monkeypatch.setattr("app.web.app.Path.read_text", lambda self, encoding="utf-8": meminfo)
    monkeypatch.setattr(
        "app.web.app.shutil.disk_usage",
        lambda _path: DiskUsage(total=10, used=3, free=7 * 1024 * 1024 * 1024),
    )
    monkeypatch.setattr("app.web.app.os.cpu_count", lambda: 2)
    monkeypatch.setattr("app.web.app.os.getloadavg", lambda: (0.5, 0.4, 0.3))
    monkeypatch.setattr("app.web.app.Path.exists", lambda self: True)
    monkeypatch.setattr(
        "app.web.app.Path.read_text",
        lambda self, encoding="utf-8": (
            "220860.42 99999.00\n"
            if str(self) == "/proc/uptime"
            else "52000\n"
            if str(self) == "/sys/class/thermal/thermal_zone0/temp"
            else meminfo
        ),
    )

    config = AppConfig(
        base_dir=tmp_path,
        storage=StorageConfig(database_path=tmp_path / "db.sqlite3", clip_dir=tmp_path / "clips"),
    )
    status = _read_system_status(config)

    assert _read_memory_available_gb() == 1.0
    assert _read_disk_free_gb(tmp_path) == 7.0
    assert status["uptime_seconds"] == 220860.42
    assert status["cpu_percent"] == 25.0
    assert status["cpu_temperature_c"] == 52.0
    assert status["memory_available_gb"] == 1.0
    assert status["disk_free_gb"] == 7.0


def test_translate_label_and_classifier_name() -> None:
    assert _translate_label("Speech") == "Mowa"
    assert _translate_label("ambulance") == "Karetka / syrena karetki"
    assert _translate_label("speech") == "Mowa ludzka"
    assert _translate_classifier_name("yamnet_litert") == "Lokalny YAMNet (LiteRT)"
