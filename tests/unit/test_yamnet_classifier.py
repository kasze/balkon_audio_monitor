from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from app.classify.service import (
    AppClassifier,
    YAMNetClassifier,
    YAMNetModelOutput,
    _extract_bird_trigger_labels,
    compute_audio_signature,
)
from app.config import AppConfig, ClassifierConfig, StorageConfig
from app.models import ClassificationOutcome, ClassifierDecision, CompletedEvent, EventSummary
from app.pipeline import AudioPipeline, RuntimeStatus
from app.storage.database import SQLiteRepository


class FakeRepository:
    def __init__(self) -> None:
        self.inserted: list[dict[str, object]] = []

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


def test_app_classifier_always_runs_yamnet_even_for_identical_sample(monkeypatch) -> None:
    event = make_event()
    repository = FakeRepository()
    classifier = AppClassifier(ClassifierConfig(), repository)  # type: ignore[arg-type]
    calls: list[CompletedEvent] = []
    monkeypatch.setattr(
        classifier.yamnet,
        "classify",
        lambda sample_event: calls.append(sample_event)
        or ClassifierDecision("yamnet_litert", "1", "airplane", 0.88, {"cache_hit": False}),
    )

    outcome = classifier.classify(event)

    assert outcome.decision.category == "airplane"
    assert outcome.decision.details["cache_hit"] is False
    assert calls == [event]


def test_app_classifier_does_not_store_similarity_cache_entries(monkeypatch) -> None:
    event = make_event()
    repository = FakeRepository()
    classifier = AppClassifier(ClassifierConfig(), repository)  # type: ignore[arg-type]
    outcome = ClassificationOutcome(
        decision=ClassifierDecision("yamnet_litert", "1", "ambulance", 0.99, {"cache_hit": False}),
        signature_hash="demo",
        signature=[0.1, 0.2],
    )

    classifier.remember(outcome, 7)

    assert repository.inserted == []


def test_extract_bird_trigger_labels_covers_birds_and_animals() -> None:
    decision = ClassifierDecision(
        "yamnet_litert",
        "1",
        "Bird vocalization, bird call, bird song",
        0.91,
        {
            "resolved_label": "Bird vocalization, bird call, bird song",
            "resolved_label_score": 0.91,
            "top_labels": [
                {"label": "Bird", "mean_score": 0.32, "peak_score": 0.67},
                {"label": "Bird flight, flapping wings", "mean_score": 0.12, "peak_score": 0.21},
                {"label": "Animal", "mean_score": 0.15, "peak_score": 0.22},
                {"label": "Turkey", "mean_score": 0.11, "peak_score": 0.19},
                {"label": "Chicken, rooster", "mean_score": 0.09, "peak_score": 0.18},
                {"label": "Speech", "mean_score": 0.05, "peak_score": 0.08},
            ],
        },
    )

    labels = _extract_bird_trigger_labels(decision, 0.08)

    assert labels == [
        "Bird vocalization, bird call, bird song",
        "Bird",
        "Bird flight, flapping wings",
        "Animal",
        "Turkey",
        "Chicken, rooster",
    ]


def test_app_classifier_uses_birdnet_for_bird_sounds(monkeypatch) -> None:
    event = make_event()
    repository = FakeRepository()
    classifier = AppClassifier(ClassifierConfig(birdnet_api_url="http://birdnet.local"), repository)  # type: ignore[arg-type]
    monkeypatch.setattr(
        classifier.yamnet,
        "classify",
        lambda _event: ClassifierDecision(
            "yamnet_litert",
            "1",
            "Bird vocalization, bird call, bird song",
            0.91,
            {
                "resolved_label": "Bird vocalization, bird call, bird song",
                "resolved_label_score": 0.91,
                "top_labels": [{"label": "Bird", "mean_score": 0.32, "peak_score": 0.67}],
                "cache_hit": False,
            },
        ),
    )
    monkeypatch.setattr(
        classifier.birdnet,
        "identify",
        lambda _event, trigger_labels, decision: ClassifierDecision(
            "birdnet_remote",
            "1",
            "Bogatka",
            0.88,
            {
                **decision.details,
                "used_external_api": True,
                "external_api_name": "BirdNET API",
                "birdnet_common_name": "Bogatka",
                "birdnet_scientific_name": "Parus major",
                "birdnet_trigger_labels": trigger_labels,
            },
        ),
    )

    outcome = classifier.classify(event)

    assert outcome.decision.classifier_name == "birdnet_remote"
    assert outcome.decision.category == "Bogatka"
    assert outcome.decision.details["birdnet_scientific_name"] == "Parus major"
    assert outcome.decision.details["birdnet_trigger_labels"] == ["Bird vocalization, bird call, bird song", "Bird"]


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

    category, confidence, category_scores, resolved_label, resolved_label_score = classifier._map_to_domain_category(
        output
    )

    assert category == "ambulance"
    assert confidence > 0.40
    assert resolved_label == "ambulance"
    assert resolved_label_score == confidence
    assert category_scores["ambulance"] > category_scores["street_background"]


def test_yamnet_mapping_falls_back_to_background() -> None:
    classifier = YAMNetClassifier(ClassifierConfig())
    output = YAMNetModelOutput(
        mean_scores={"Traffic noise, roadway noise": 0.18, "Vehicle": 0.07},
        peak_scores={"Traffic noise, roadway noise": 0.25, "Vehicle": 0.12},
        top_labels=[],
    )

    category, confidence, category_scores, resolved_label, resolved_label_score = classifier._map_to_domain_category(
        output
    )

    assert category == "street_background"
    assert confidence >= 0.18
    assert resolved_label == "Traffic noise, roadway noise"
    assert resolved_label_score == confidence
    assert category_scores["street_background"] > 0.0


def test_yamnet_mapping_uses_raw_yamnet_label_for_strong_non_domain_label() -> None:
    classifier = YAMNetClassifier(ClassifierConfig())
    output = YAMNetModelOutput(
        mean_scores={"Speech": 0.733, "Inside, small room": 0.032, "Silence": 0.009},
        peak_scores={"Speech": 0.980, "Inside, small room": 0.059, "Silence": 0.109},
        top_labels=[],
    )

    category, confidence, category_scores, resolved_label, resolved_label_score = classifier._map_to_domain_category(
        output
    )

    assert category == "Speech"
    assert confidence > 0.80
    assert resolved_label == "Speech"
    assert resolved_label_score == confidence
    assert max(category_scores.values()) == 0.0


def test_yamnet_mapping_discards_background_like_labels() -> None:
    classifier = YAMNetClassifier(ClassifierConfig())
    output = YAMNetModelOutput(
        mean_scores={"White noise": 0.42, "Silence": 0.11, "Speech": 0.07},
        peak_scores={"White noise": 0.91, "Silence": 0.24, "Speech": 0.15},
        top_labels=[],
    )

    category, confidence, category_scores, resolved_label, resolved_label_score = classifier._map_to_domain_category(
        output
    )

    assert category == "discarded"
    assert confidence > 0.70
    assert resolved_label == "White noise"
    assert resolved_label_score == confidence
    assert category_scores["street_background"] == 0.0


def test_pipeline_skips_discarded_events(tmp_path) -> None:
    database_path = tmp_path / "audio_monitor.sqlite3"
    repository = SQLiteRepository(database_path)
    repository.initialize()
    config = AppConfig(base_dir=tmp_path, storage=StorageConfig(database_path=database_path, clip_dir=tmp_path / "clips"))
    pipeline = AudioPipeline(config, repository, RuntimeStatus())

    class FakeClassifier:
        def classify(self, _event):
            return ClassificationOutcome(
                decision=ClassifierDecision(
                    "yamnet_litert",
                    "1",
                    "discarded",
                    0.91,
                    {"resolved_label": "White noise"},
                )
            )

        def remember(self, _outcome, _event_id):
            raise AssertionError("discarded events must not be remembered")

    pipeline.classifier = FakeClassifier()  # type: ignore[assignment]
    pipeline._persist_event(make_event())

    assert repository.recent_events_count() == 0
