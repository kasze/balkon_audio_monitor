from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from app.capture.wav import WavFileSource
from app.config import AppConfig, AudioConfig, LoggingConfig, StorageConfig, WebConfig
from app.config import AggregationConfig, DetectionConfig
from app.pipeline import AudioPipeline, RuntimeStatus
from app.storage.database import SQLiteRepository


def write_siren(path: Path, sample_rate: int = 16_000, duration: float = 8.0) -> None:
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    sweep = 700 + 500 * np.sin(2 * math.pi * 0.6 * t)
    phase = 2 * math.pi * np.cumsum(sweep) / sample_rate
    samples = 0.35 * np.sin(phase) + 0.02 * np.random.default_rng(2).normal(size=t.shape)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm_i16.tobytes())


def test_offline_pipeline_persists_event_and_clip(tmp_path: Path) -> None:
    database_path = tmp_path / "audio_monitor.sqlite3"
    clip_dir = tmp_path / "clips"
    wav_path = tmp_path / "demo_siren.wav"
    write_siren(wav_path)

    config = AppConfig(
        base_dir=tmp_path,
        audio=AudioConfig(sample_rate=16_000, channels=1, frame_duration_seconds=0.5),
        detection=DetectionConfig(
            initial_noise_floor_dbfs=-60.0,
            activation_margin_db=7.0,
            release_margin_db=3.0,
            min_event_dbfs=-50.0,
            min_active_frames=2,
            max_inactive_frames=3,
            noise_floor_alpha=0.05,
        ),
        aggregation=AggregationConfig(
            noise_interval_seconds=5.0,
            pre_roll_seconds=0.5,
            post_roll_seconds=1.0,
            min_event_seconds=1.0,
            max_clip_seconds=10.0,
            max_event_seconds=30.0,
        ),
        storage=StorageConfig(database_path=database_path, clip_dir=clip_dir, keep_clips=True),
        web=WebConfig(),
        logging=LoggingConfig(level="INFO"),
    )

    repository = SQLiteRepository(database_path)
    repository.initialize()
    pipeline = AudioPipeline(config, repository, RuntimeStatus())
    source = WavFileSource(wav_path, config.audio.sample_rate, config.audio.frame_duration_seconds)
    pipeline.reset_runtime_state()
    pipeline.process_stream(source.frames())

    assert repository.recent_events_count() >= 1
    event = repository.get_event(1)
    assert event is not None
    dashboard = repository.get_dashboard(event["started_at"][:10], 10)
    assert event["category"] in {"ambulance", "police", "fire_truck", "street_background"}
    assert event["clip_path"] is not None
    assert event["spectrogram_path"] is not None
    assert Path(event["clip_path"]).exists()
    assert Path(event["spectrogram_path"]).exists()
    assert dashboard["recent_events"]
