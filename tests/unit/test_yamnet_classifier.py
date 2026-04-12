from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from app.classify.service import (
    AppClassifier,
    YAMNetClassifier,
    YAMNetModelOutput,
    compute_audio_signature,
)
from app.config import ClassifierConfig
from app.models import (
    ClassifierCacheEntry,
    ClassifierDecision,
    CompletedEvent,
    EventSummary,
)


class FakeRepository:
    def __init__(self, entries: list[ClassifierCacheEntry] | None = None) -> None:
        self.entries = entries or []
        self.inserted: list[dict[str, object]] = []

    def list_classifier_cache_entries(
        self,
        classifier_name: str,
        min_confidence: float,
        lookback_days: int,
        limit: int,
    ) -> list[ClassifierCacheEntry]:
        return list(self.entries)

    def insert_classifier_cache_entry(self, **kwargs) -> None:
        self.inserted.append(kwargs)


def make_event(samples: np.ndarray | None = None) -> CompletedEvent:
    clip_samples = samples if samples is not None else np.sin(np.linspace(0.0, 50.0, 16_000)).astype(np.float32)
    started_at = datetime.now().astimezone()
    summary = EventSummary(
        source_name="test",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=float(clip_samples.size / 16_000)),
        duration_seconds=float(clip_samples.size / 16_000),
        frame_count=8,
        peak_dbfs=-20.0,
        mean_dbfs=-25.0,
        mean_centroid_hz=900.0,
        dominant_freq_hz=950.0,
        dominant_span_hz=200.0,
        low_band_ratio=0.2,
        mid_band_ratio=0.5,
        high_band_ratio=0.3,
        mean_flux=0.02,
        mean_flatness=0.25,
        rms_modulation_depth=0.1,
        dominant_modulation_hz=2.0,
    )
    return CompletedEvent(summary=summary, clip_samples=clip_samples, sample_rate=16_000)


def test_app_classifier_reuses_cached_decision(monkeypatch) -> None:
    event = make_event()
    signature_hash, signature = compute_audio_signature(event.clip_samples, event.sample_rate)
    repository = FakeRepository(
        entries=[
            ClassifierCacheEntry(
                event_id=42,
                classifier_name="yamnet_litert",
                classifier_version="1",
                category="airplane",
                confidence=0.88,
                signature_hash=signature_hash,
                signature=signature,
                details={"top_labels": [{"label": "Fixed-wing aircraft, airplane", "mean_score": 0.88}]},
            )
        ]
    )
    classifier = AppClassifier(ClassifierConfig(), repository)  # type: ignore[arg-type]
    monkeypatch.setattr(
        classifier.yamnet,
        "classify",
        lambda _event: (_ for _ in ()).throw(AssertionError("YAMNet should not run when cache hits")),
    )

    outcome = classifier.classify(event)

    assert outcome.decision.category == "airplane"
    assert outcome.decision.details["cache_hit"] is True
    assert outcome.decision.details["cache_source_event_id"] == 42


def test_yamnet_mapping_prefers_specific_emergency_label() -> None:
    classifier = YAMNetClassifier(ClassifierConfig())
    output = YAMNetModelOutput(
        mean_scores={
            "Ambulance (siren)": 0.42,
            "Siren": 0.20,
            "Traffic noise, roadway noise": 0.05,
        },
        peak_scores={
            "Ambulance (siren)": 0.70,
            "Siren": 0.40,
            "Traffic noise, roadway noise": 0.12,
        },
        top_labels=[],
    )

    category, confidence, category_scores = classifier._map_to_domain_category(output)

    assert category == "ambulance"
    assert confidence > 0.40
    assert category_scores["ambulance"] > category_scores["street_background"]


def test_yamnet_mapping_falls_back_to_background() -> None:
    classifier = YAMNetClassifier(ClassifierConfig())
    output = YAMNetModelOutput(
        mean_scores={"Traffic noise, roadway noise": 0.18, "Vehicle": 0.07},
        peak_scores={"Traffic noise, roadway noise": 0.25, "Vehicle": 0.12},
        top_labels=[],
    )

    category, confidence, category_scores = classifier._map_to_domain_category(output)

    assert category == "street_background"
    assert confidence >= 0.18
    assert category_scores["street_background"] > 0.0
