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
    activation_margin_db: float = 11.0
    release_margin_db: float = 4.0
    min_event_dbfs: float = -44.0
    min_active_frames: int = 3
    max_inactive_frames: int = 2
    noise_floor_alpha: float = 0.04


@dataclass(slots=True, frozen=True)
class AggregationConfig:
    noise_interval_seconds: float = 5.0
    pre_roll_seconds: float = 1.0
    post_roll_seconds: float = 0.5
    min_event_seconds: float = 1.0
    focus_clip_seconds: float = 8.0
    max_clip_seconds: float = 30.0
    max_event_seconds: float = 25.0


@dataclass(slots=True, frozen=True)
class StorageConfig:
    database_path: Path
    clip_dir: Path
    keep_clips: bool = True
    clip_max_megabytes: int = 1024
    clip_max_age_days: int = 14
    min_free_disk_megabytes: int = 512

    @property
    def clip_max_bytes(self) -> int:
        return max(0, self.clip_max_megabytes) * 1024 * 1024

    @property
    def min_free_disk_bytes(self) -> int:
        return max(0, self.min_free_disk_megabytes) * 1024 * 1024


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
class ClassifierConfig:
    backend: str = "yamnet"
    reuse_similarity_threshold: float = 0.992
    reuse_confidence_threshold: float = 0.60
    similarity_cache_limit: int = 200
    similarity_lookback_days: int = 30
    yamnet_model_path: Path = Path("models/yamnet.tflite")
    yamnet_class_map_path: Path = Path("models/yamnet_class_map.csv")
    yamnet_model_url: str = "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
    yamnet_class_map_url: str = (
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    )
    yamnet_num_threads: int = 1
    yamnet_max_analysis_seconds: float = 12.0
    yamnet_max_windows: int = 24
    yamnet_min_category_score: float = 0.08
    yamnet_top_k: int = 8
    birdnet_api_url: str = ""
    birdnet_timeout_seconds: float = 15.0
    birdnet_min_confidence: float = 0.20
    birdnet_num_results: int = 5
    birdnet_locale: str = "pl"


@dataclass(slots=True, frozen=True)
class AppConfig:
    base_dir: Path
    audio: AudioConfig = field(default_factory=AudioConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    storage: StorageConfig = field(
        default_factory=lambda: StorageConfig(
            database_path=Path("data/db/audio_monitor.sqlite3"),
            clip_dir=Path("data/clips"),
        )
    )
    web: WebConfig = field(default_factory=WebConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    config_path: Path | None = None


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
        "classifier": {
            **asdict(ClassifierConfig()),
            "yamnet_model_path": str((base_dir / "models/yamnet.tflite").resolve()),
            "yamnet_class_map_path": str((base_dir / "models/yamnet_class_map.csv").resolve()),
        },
        "storage": {
            "database_path": str((base_dir / "data/db/audio_monitor.sqlite3").resolve()),
            "clip_dir": str((base_dir / "data/clips").resolve()),
            "keep_clips": True,
            "clip_max_megabytes": 1024,
            "clip_max_age_days": 14,
            "min_free_disk_megabytes": 512,
        },
        "web": asdict(WebConfig()),
        "logging": asdict(LoggingConfig()),
    }


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path).resolve() if path else None
    if config_path is None:
        base_dir = Path.cwd().resolve()
    elif config_path.parent.name == "configs":
        base_dir = config_path.parent.parent.resolve()
    else:
        base_dir = config_path.parent.resolve()
    merged = _defaults(base_dir)
    if config_path and config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        merged = _merge(merged, loaded)

    classifier = merged["classifier"]
    storage = merged["storage"]
    return AppConfig(
        base_dir=base_dir,
        audio=AudioConfig(**merged["audio"]),
        detection=DetectionConfig(**merged["detection"]),
        aggregation=AggregationConfig(**merged["aggregation"]),
        classifier=ClassifierConfig(
            backend=classifier["backend"],
            reuse_similarity_threshold=float(classifier["reuse_similarity_threshold"]),
            reuse_confidence_threshold=float(classifier["reuse_confidence_threshold"]),
            similarity_cache_limit=int(classifier["similarity_cache_limit"]),
            similarity_lookback_days=int(classifier["similarity_lookback_days"]),
            yamnet_model_path=_resolve_path(base_dir, classifier["yamnet_model_path"]),
            yamnet_class_map_path=_resolve_path(base_dir, classifier["yamnet_class_map_path"]),
            yamnet_model_url=classifier["yamnet_model_url"],
            yamnet_class_map_url=classifier["yamnet_class_map_url"],
            yamnet_num_threads=int(classifier["yamnet_num_threads"]),
            yamnet_max_analysis_seconds=float(classifier["yamnet_max_analysis_seconds"]),
            yamnet_max_windows=int(classifier["yamnet_max_windows"]),
            yamnet_min_category_score=float(classifier["yamnet_min_category_score"]),
            yamnet_top_k=int(classifier["yamnet_top_k"]),
            birdnet_api_url=str(classifier.get("birdnet_api_url", "")),
            birdnet_timeout_seconds=float(classifier.get("birdnet_timeout_seconds", 15.0)),
            birdnet_min_confidence=float(classifier.get("birdnet_min_confidence", 0.20)),
            birdnet_num_results=int(classifier.get("birdnet_num_results", 5)),
            birdnet_locale=str(classifier.get("birdnet_locale", "pl")),
        ),
        storage=StorageConfig(
            database_path=_resolve_path(base_dir, storage["database_path"]),
            clip_dir=_resolve_path(base_dir, storage["clip_dir"]),
            keep_clips=bool(storage.get("keep_clips", True)),
            clip_max_megabytes=int(storage.get("clip_max_megabytes", 1024)),
            clip_max_age_days=int(storage.get("clip_max_age_days", 14)),
            min_free_disk_megabytes=int(storage.get("min_free_disk_megabytes", 512)),
        ),
        web=WebConfig(**merged["web"]),
        logging=LoggingConfig(**merged["logging"]),
        config_path=config_path,
    )


def save_config(config: AppConfig, path: str | Path | None = None) -> Path:
    target_path = Path(path).resolve() if path else (config.config_path or config.base_dir / "configs/config.yaml")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio": {
            "sample_rate": int(config.audio.sample_rate),
            "channels": int(config.audio.channels),
            "frame_duration_seconds": float(config.audio.frame_duration_seconds),
            "arecord_binary": config.audio.arecord_binary,
            "arecord_device": config.audio.arecord_device,
            "retry_backoff_seconds": float(config.audio.retry_backoff_seconds),
        },
        "detection": {
            "initial_noise_floor_dbfs": float(config.detection.initial_noise_floor_dbfs),
            "activation_margin_db": float(config.detection.activation_margin_db),
            "release_margin_db": float(config.detection.release_margin_db),
            "min_event_dbfs": float(config.detection.min_event_dbfs),
            "min_active_frames": int(config.detection.min_active_frames),
            "max_inactive_frames": int(config.detection.max_inactive_frames),
            "noise_floor_alpha": float(config.detection.noise_floor_alpha),
        },
        "aggregation": {
            "noise_interval_seconds": float(config.aggregation.noise_interval_seconds),
            "pre_roll_seconds": float(config.aggregation.pre_roll_seconds),
            "post_roll_seconds": float(config.aggregation.post_roll_seconds),
            "min_event_seconds": float(config.aggregation.min_event_seconds),
            "focus_clip_seconds": float(config.aggregation.focus_clip_seconds),
            "max_clip_seconds": float(config.aggregation.max_clip_seconds),
            "max_event_seconds": float(config.aggregation.max_event_seconds),
        },
        "classifier": {
            "backend": config.classifier.backend,
            "reuse_similarity_threshold": float(config.classifier.reuse_similarity_threshold),
            "reuse_confidence_threshold": float(config.classifier.reuse_confidence_threshold),
            "similarity_cache_limit": int(config.classifier.similarity_cache_limit),
            "similarity_lookback_days": int(config.classifier.similarity_lookback_days),
            "yamnet_model_path": _serialize_path(config.base_dir, config.classifier.yamnet_model_path),
            "yamnet_class_map_path": _serialize_path(config.base_dir, config.classifier.yamnet_class_map_path),
            "yamnet_model_url": config.classifier.yamnet_model_url,
            "yamnet_class_map_url": config.classifier.yamnet_class_map_url,
            "yamnet_num_threads": int(config.classifier.yamnet_num_threads),
            "yamnet_max_analysis_seconds": float(config.classifier.yamnet_max_analysis_seconds),
            "yamnet_max_windows": int(config.classifier.yamnet_max_windows),
            "yamnet_min_category_score": float(config.classifier.yamnet_min_category_score),
            "yamnet_top_k": int(config.classifier.yamnet_top_k),
            "birdnet_api_url": config.classifier.birdnet_api_url,
            "birdnet_timeout_seconds": float(config.classifier.birdnet_timeout_seconds),
            "birdnet_min_confidence": float(config.classifier.birdnet_min_confidence),
            "birdnet_num_results": int(config.classifier.birdnet_num_results),
            "birdnet_locale": config.classifier.birdnet_locale,
        },
        "storage": {
            "database_path": _serialize_path(config.base_dir, config.storage.database_path),
            "clip_dir": _serialize_path(config.base_dir, config.storage.clip_dir),
            "keep_clips": bool(config.storage.keep_clips),
            "clip_max_megabytes": int(config.storage.clip_max_megabytes),
            "clip_max_age_days": int(config.storage.clip_max_age_days),
            "min_free_disk_megabytes": int(config.storage.min_free_disk_megabytes),
        },
        "web": {
            "host": config.web.host,
            "port": int(config.web.port),
            "recent_events_limit": int(config.web.recent_events_limit),
            "dashboard_history_hours": int(config.web.dashboard_history_hours),
        },
        "logging": {
            "level": config.logging.level,
        },
    }
    target_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target_path


def _serialize_path(base_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path.resolve())
