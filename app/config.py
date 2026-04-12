from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


@dataclass(slots=True, frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    frame_duration_seconds: float = 0.5
    arecord_binary: str = "arecord"
    arecord_device: str | None = None
    retry_backoff_seconds: float = 5.0


@dataclass(slots=True, frozen=True)
class DetectionConfig:
    initial_noise_floor_dbfs: float = -58.0
    activation_margin_db: float = 9.0
    release_margin_db: float = 4.0
    min_event_dbfs: float = -48.0
    min_active_frames: int = 2
    max_inactive_frames: int = 3
    noise_floor_alpha: float = 0.04


@dataclass(slots=True, frozen=True)
class AggregationConfig:
    noise_interval_seconds: float = 5.0
    pre_roll_seconds: float = 1.0
    post_roll_seconds: float = 1.0
    min_event_seconds: float = 1.0
    max_clip_seconds: float = 30.0
    max_event_seconds: float = 90.0


@dataclass(slots=True, frozen=True)
class StorageConfig:
    database_path: Path
    clip_dir: Path
    keep_clips: bool = True


@dataclass(slots=True, frozen=True)
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    recent_events_limit: int = 20
    dashboard_history_hours: int = 24


@dataclass(slots=True, frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(slots=True, frozen=True)
class AppConfig:
    base_dir: Path
    audio: AudioConfig = field(default_factory=AudioConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    storage: StorageConfig = field(
        default_factory=lambda: StorageConfig(
            database_path=Path("data/db/audio_monitor.sqlite3"),
            clip_dir=Path("data/clips"),
        )
    )
    web: WebConfig = field(default_factory=WebConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _defaults(base_dir: Path) -> dict[str, Any]:
    return {
        "audio": asdict(AudioConfig()),
        "detection": asdict(DetectionConfig()),
        "aggregation": asdict(AggregationConfig()),
        "storage": {
            "database_path": str((base_dir / "data/db/audio_monitor.sqlite3").resolve()),
            "clip_dir": str((base_dir / "data/clips").resolve()),
            "keep_clips": True,
        },
        "web": asdict(WebConfig()),
        "logging": asdict(LoggingConfig()),
    }


def load_config(path: str | Path | None = None) -> AppConfig:
    base_dir = Path.cwd().resolve()
    config_path = Path(path).resolve() if path else None
    merged = _defaults(base_dir)
    if config_path and config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        merged = _merge(merged, loaded)

    storage = merged["storage"]
    return AppConfig(
        base_dir=base_dir,
        audio=AudioConfig(**merged["audio"]),
        detection=DetectionConfig(**merged["detection"]),
        aggregation=AggregationConfig(**merged["aggregation"]),
        storage=StorageConfig(
            database_path=_resolve_path(base_dir, storage["database_path"]),
            clip_dir=_resolve_path(base_dir, storage["clip_dir"]),
            keep_clips=bool(storage.get("keep_clips", True)),
        ),
        web=WebConfig(**merged["web"]),
        logging=LoggingConfig(**merged["logging"]),
    )
