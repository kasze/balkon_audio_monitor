from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime
from threading import Event, Lock

from app.aggregate.event_aggregator import EventAggregator
from app.classify.service import AppClassifier
from app.aggregate.noise_intervals import NoiseIntervalCollector
from app.config import AppConfig
from app.features.extractor import FeatureExtractor
from app.models import AudioFrame
from app.storage.clips import ClipStore
from app.storage.database import SQLiteRepository
from app.storage.retention import ClipRetentionManager
from app.detect.detector import AdaptiveEnergyDetector

LOGGER = logging.getLogger(__name__)


class RuntimeStatus:
    def __init__(self) -> None:
        self._lock = Lock()
        self._data = {
            "worker_state": "idle",
            "started_at": datetime.now().astimezone().isoformat(),
            "audio_available": False,
            "audio_device": None,
            "audio_device_mode": None,
            "audio_device_name": None,
            "last_frame_at": None,
            "last_error": None,
            "events_written": 0,
            "intervals_written": 0,
            "source_name": None,
        }

    def update(self, **values: object) -> None:
        with self._lock:
            self._data.update(values)

    def increment(self, key: str) -> None:
        with self._lock:
            self._data[key] = int(self._data.get(key, 0)) + 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._data)


class AudioPipeline:
    def __init__(self, config: AppConfig, repository: SQLiteRepository, status: RuntimeStatus) -> None:
        self.config = config
        self.repository = repository
        self.status = status
        self.extractor = FeatureExtractor(config.audio.sample_rate)
        self.detector = AdaptiveEnergyDetector(config.detection)
        self.noise_collector = NoiseIntervalCollector(config.aggregation.noise_interval_seconds)
        self.aggregator = EventAggregator(
            config.aggregation,
            config.audio.sample_rate,
            config.audio.frame_duration_seconds,
        )
        self.classifier = AppClassifier(config.classifier, repository)
        self.clip_store = ClipStore(config.storage.clip_dir)
        self.retention = ClipRetentionManager(config.storage, repository)

    def reset_runtime_state(self) -> None:
        self.extractor.reset()
        self.detector.reset()
        self.noise_collector.reset()
        self.aggregator.reset()

    def process_stream(self, frames: Iterable[AudioFrame], stop_event: Event | None = None) -> None:
        try:
            for frame in frames:
                if stop_event and stop_event.is_set():
                    break
                self.process_frame(frame)
        finally:
            self._flush()

    def process_frame(self, frame: AudioFrame) -> None:
        features = self.extractor.extract(frame)
        detection_state = self.detector.process(features)
        self.status.update(
            worker_state="running",
            audio_available=True,
            last_frame_at=frame.started_at.isoformat(),
            last_error=None,
            source_name=frame.source_name,
        )

        noise_interval = self.noise_collector.process(features)
        if noise_interval is not None:
            self.repository.insert_noise_interval(noise_interval)
            self.status.increment("intervals_written")

        for completed in self.aggregator.process(frame, features, detection_state.is_active):
            self._persist_event(completed)

    def _flush(self) -> None:
        noise_interval = self.noise_collector.flush()
        if noise_interval is not None:
            self.repository.insert_noise_interval(noise_interval)
            self.status.increment("intervals_written")
        for completed in self.aggregator.flush():
            self._persist_event(completed)

    def _persist_event(self, completed) -> None:
        outcome = self.classifier.classify(completed)
        decision = outcome.decision
        if decision.category == "discarded":
            LOGGER.info(
                "Discarded event duration=%.1fs reason=%s",
                completed.summary.duration_seconds,
                decision.details.get("resolved_label") if isinstance(decision.details, dict) else "discarded_label",
            )
            return
        clip = None
        if self.config.storage.keep_clips:
            estimated_clip_bytes = self.clip_store.estimate_total_size(completed.clip_samples.size)
            if self.retention.prepare_for_clip(estimated_clip_bytes):
                try:
                    clip = self.clip_store.save(completed)
                except OSError as exc:
                    LOGGER.warning("Failed to save clip audio: %s", exc)
        persisted = self.repository.insert_event(completed, decision, clip)
        if clip is not None:
            self.retention.enforce_limits()
        self.classifier.remember(outcome, persisted.event_id)
        self.status.increment("events_written")
        LOGGER.info(
            "Persisted event id=%s category=%s confidence=%.2f duration=%.1fs clip=%s",
            persisted.event_id,
            persisted.category,
            decision.confidence,
            completed.summary.duration_seconds,
            persisted.clip_path,
        )
