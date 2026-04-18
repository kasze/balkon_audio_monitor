from __future__ import annotations

import logging
from collections.abc import Iterable
from collections import deque
from datetime import datetime
from threading import Condition, Event, Lock

import numpy as np

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


class LiveAudioBuffer:
    def __init__(self, sample_rate: int, max_seconds: float = 20.0) -> None:
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._chunks: deque[tuple[int, datetime, bytes, float]] = deque()
        self._duration_seconds = 0.0
        self._sequence = 0

    def append(self, started_at: datetime, samples: np.ndarray, duration_seconds: float) -> None:
        if samples.size == 0 or duration_seconds <= 0:
            return
        chunk = np.asarray(samples, dtype=np.float32).copy()
        pcm = np.clip(chunk, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype("<i2")
        chunk_bytes = pcm_i16.tobytes()
        with self._lock:
            self._sequence += 1
            self._chunks.append((self._sequence, started_at, chunk_bytes, duration_seconds))
            self._duration_seconds += duration_seconds
            while self._chunks and self._duration_seconds > self.max_seconds:
                _old_sequence, _old_started_at, _old_bytes, old_duration = self._chunks.popleft()
                self._duration_seconds -= old_duration
            self._condition.notify_all()

    def snapshot(self, seconds: float) -> np.ndarray:
        if seconds <= 0:
            return np.array([], dtype=np.float32)
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)
            selected: list[bytes] = []
            total = 0.0
            for _sequence, _started_at, chunk_bytes, duration_seconds in reversed(self._chunks):
                selected.append(chunk_bytes)
                total += duration_seconds
                if total >= seconds:
                    break
        if not selected:
            return np.array([], dtype=np.float32)
        pcm_i16 = np.frombuffer(b"".join(reversed(selected)), dtype="<i2")
        return pcm_i16.astype(np.float32) / 32768.0

    def snapshot_wav_bytes(self, seconds: float) -> bytes:
        samples = self.snapshot(seconds)
        if samples.size == 0:
            return b""
        return _build_wav_bytes(samples, self.sample_rate)

    def stream_wav_bytes(self, seconds: float, stop_event: Event | None = None):
        yield _build_wav_header(self.sample_rate, 1, 2, 0xFFFFFFFF)
        initial = self.snapshot_wav_bytes(seconds)
        if initial:
            # Strip RIFF header and stream the PCM payload immediately.
            yield initial[44:]

        last_sequence = self._sequence
        while True:
            if stop_event and stop_event.is_set():
                break
            with self._condition:
                self._condition.wait(timeout=1.0)
                if stop_event and stop_event.is_set():
                    break
                pending = [chunk_bytes for seq, _started_at, chunk_bytes, _duration_seconds in self._chunks if seq > last_sequence]
                if self._chunks:
                    last_sequence = self._chunks[-1][0]
            if not pending:
                continue
            for chunk_bytes in pending:
                yield chunk_bytes


def _build_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype("<i2")
    import io
    import wave

    handle = io.BytesIO()
    with wave.open(handle, "wb") as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(sample_rate)
        wav_handle.writeframes(pcm_i16.tobytes())
    return handle.getvalue()


def _build_wav_header(sample_rate: int, channels: int, sample_width: int, data_size: int) -> bytes:
    import struct

    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", min(0xFFFFFFFF, 36 + data_size)),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, sample_width * 8),
            b"data",
            struct.pack("<I", min(0xFFFFFFFF, data_size)),
        ]
    )


class AudioPipeline:
    def __init__(self, config: AppConfig, repository: SQLiteRepository, status: RuntimeStatus) -> None:
        self.config = config
        self.repository = repository
        self.status = status
        self.live_audio_buffer = LiveAudioBuffer(config.audio.sample_rate)
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
        self.live_audio_buffer.append(frame.started_at, frame.samples, frame.duration_seconds)
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
