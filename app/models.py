from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class AudioFrame:
    samples: np.ndarray
    started_at: datetime
    duration_seconds: float
    source_name: str


@dataclass(slots=True)
class FrameFeatures:
    started_at: datetime
    duration_seconds: float
    source_name: str
    rms: float
    dbfs: float
    spectral_centroid_hz: float
    dominant_freq_hz: float
    low_band_ratio: float
    mid_band_ratio: float
    high_band_ratio: float
    spectral_flux: float
    flatness: float
    zero_crossing_rate: float


@dataclass(slots=True)
class DetectionState:
    is_active: bool
    noise_floor_dbfs: float
    activation_score: float


@dataclass(slots=True)
class NoiseInterval:
    source_name: str
    started_at: datetime
    ended_at: datetime
    avg_rms: float
    avg_dbfs: float
    max_dbfs: float
    avg_centroid_hz: float
    low_band_ratio: float
    mid_band_ratio: float
    high_band_ratio: float


@dataclass(slots=True)
class EventSummary:
    source_name: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    frame_count: int
    peak_dbfs: float
    mean_dbfs: float
    mean_centroid_hz: float
    dominant_freq_hz: float
    dominant_span_hz: float
    low_band_ratio: float
    mid_band_ratio: float
    high_band_ratio: float
    mean_flux: float
    mean_flatness: float
    rms_modulation_depth: float
    dominant_modulation_hz: float
    details: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CompletedEvent:
    summary: EventSummary
    clip_samples: np.ndarray
    sample_rate: int


@dataclass(slots=True)
class ClipMetadata:
    path: Path
    sample_rate: int
    channels: int
    duration_seconds: float
    byte_size: int
    sha1: str


@dataclass(slots=True)
class StoredClip:
    clip_id: int
    path: Path
    byte_size: int
    created_at: str


@dataclass(slots=True)
class ClassifierDecision:
    classifier_name: str
    classifier_version: str
    category: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PersistedEvent:
    event_id: int
    category: str
    clip_path: str | None


@dataclass(slots=True)
class ClassificationOutcome:
    decision: ClassifierDecision
    signature_hash: str | None = None
    signature: list[float] = field(default_factory=list)


@dataclass(slots=True)
class ClassifierCacheEntry:
    event_id: int
    classifier_name: str
    classifier_version: str
    category: str
    confidence: float
    signature_hash: str
    signature: list[float]
    details: dict[str, Any] = field(default_factory=dict)
