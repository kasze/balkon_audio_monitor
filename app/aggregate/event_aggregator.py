from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from app.config import AggregationConfig
from app.models import AudioFrame, CompletedEvent, EventSummary, FrameFeatures


@dataclass(slots=True)
class _EventState:
    frames: list[FrameFeatures] = field(default_factory=list)
    clip_chunks: list[np.ndarray] = field(default_factory=list)
    clip_samples: int = 0
    inactive_frames: int = 0
    source_name: str = ""


class EventAggregator:
    def __init__(self, config: AggregationConfig, sample_rate: int, frame_duration_seconds: float) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self.frame_duration_seconds = frame_duration_seconds
        self.pre_roll_frames = max(0, round(config.pre_roll_seconds / frame_duration_seconds))
        self.post_roll_frames = max(1, round(config.post_roll_seconds / frame_duration_seconds))
        self.focus_clip_frames = max(1, round(config.focus_clip_seconds / frame_duration_seconds))
        self.clip_limit_samples = int(sample_rate * config.max_clip_seconds)
        self.reset()

    def reset(self) -> None:
        self._history: deque[tuple[AudioFrame, FrameFeatures]] = deque(maxlen=self.pre_roll_frames)
        self._state: _EventState | None = None

    def process(
        self,
        frame: AudioFrame,
        features: FrameFeatures,
        is_active: bool,
    ) -> list[CompletedEvent]:
        completed: list[CompletedEvent] = []

        if self._state is None and is_active:
            self._state = _EventState(source_name=frame.source_name)
            for previous_frame, previous_features in self._history:
                self._append(previous_frame, previous_features)
            self._append(frame, features)
        elif self._state is not None:
            self._append(frame, features)
            if is_active:
                self._state.inactive_frames = 0
            else:
                self._state.inactive_frames += 1

            duration_seconds = sum(item.duration_seconds for item in self._state.frames)
            if duration_seconds >= self.config.max_event_seconds:
                completed.append(self._finalize())
            elif self._state.inactive_frames > self.post_roll_frames:
                completed.append(self._finalize())

        self._history.append((frame, features))
        return [event for event in completed if event.summary.duration_seconds >= self.config.min_event_seconds]

    def flush(self) -> list[CompletedEvent]:
        if self._state is None:
            return []
        event = self._finalize()
        if event.summary.duration_seconds < self.config.min_event_seconds:
            return []
        return [event]

    def _append(self, frame: AudioFrame, features: FrameFeatures) -> None:
        assert self._state is not None
        self._state.frames.append(features)
        if self._state.clip_samples < self.clip_limit_samples:
            remaining = self.clip_limit_samples - self._state.clip_samples
            clip_chunk = frame.samples[:remaining]
            self._state.clip_chunks.append(clip_chunk.copy())
            self._state.clip_samples += clip_chunk.size

    def _finalize(self) -> CompletedEvent:
        assert self._state is not None
        state = self._state
        self._state = None
        summary = self._summarize(state.frames)
        clip_samples, focus_details = self._build_focus_clip(state.frames, state.clip_chunks)
        summary.details.update(focus_details)
        return CompletedEvent(summary=summary, clip_samples=clip_samples, sample_rate=self.sample_rate)

    def _build_focus_clip(
        self,
        frames: list[FrameFeatures],
        clip_chunks: list[np.ndarray],
    ) -> tuple[np.ndarray, dict[str, float]]:
        if not clip_chunks:
            return np.array([], dtype=np.float32), {
                "focus_clip_duration_seconds": 0.0,
                "focus_clip_start_offset_seconds": 0.0,
                "clip_trimmed": 0.0,
            }

        full_clip = np.concatenate(clip_chunks)
        available_frames = min(len(frames), len(clip_chunks))
        if available_frames == 0:
            return full_clip, {
                "focus_clip_duration_seconds": float(full_clip.size / self.sample_rate),
                "focus_clip_start_offset_seconds": 0.0,
                "clip_trimmed": 0.0,
            }

        if available_frames <= self.focus_clip_frames:
            return full_clip, {
                "focus_clip_duration_seconds": float(full_clip.size / self.sample_rate),
                "focus_clip_start_offset_seconds": 0.0,
                "clip_trimmed": 0.0,
            }

        dbfs_values = np.array([item.dbfs for item in frames[:available_frames]], dtype=np.float32)
        peak_index = int(np.argmax(dbfs_values))
        start_frame = max(0, peak_index - self.focus_clip_frames // 2)
        max_start = max(0, available_frames - self.focus_clip_frames)
        start_frame = min(start_frame, max_start)
        end_frame = min(available_frames, start_frame + self.focus_clip_frames)

        focused_clip = np.concatenate(clip_chunks[start_frame:end_frame])
        return focused_clip, {
            "focus_clip_duration_seconds": float(focused_clip.size / self.sample_rate),
            "focus_clip_start_offset_seconds": float(start_frame * self.frame_duration_seconds),
            "clip_trimmed": 1.0,
        }

    def _summarize(self, frames: list[FrameFeatures]) -> EventSummary:
        dbfs_values = np.array([item.dbfs for item in frames], dtype=np.float32)
        centroid_values = np.array([item.spectral_centroid_hz for item in frames], dtype=np.float32)
        dominant_values = np.array([item.dominant_freq_hz for item in frames], dtype=np.float32)
        flux_values = np.array([item.spectral_flux for item in frames], dtype=np.float32)
        flatness_values = np.array([item.flatness for item in frames], dtype=np.float32)
        rms_values = np.array([item.rms for item in frames], dtype=np.float32)
        low_values = np.array([item.low_band_ratio for item in frames], dtype=np.float32)
        mid_values = np.array([item.mid_band_ratio for item in frames], dtype=np.float32)
        high_values = np.array([item.high_band_ratio for item in frames], dtype=np.float32)

        duration_seconds = sum(item.duration_seconds for item in frames)
        dominant_modulation_hz = self._estimate_modulation_hz(dominant_values, frames[0].duration_seconds)
        mean_rms = float(rms_values.mean())
        rms_modulation_depth = float(rms_values.std() / mean_rms) if mean_rms else 0.0

        return EventSummary(
            source_name=frames[0].source_name,
            started_at=frames[0].started_at,
            ended_at=frames[-1].started_at + timedelta(seconds=frames[-1].duration_seconds),
            duration_seconds=duration_seconds,
            frame_count=len(frames),
            peak_dbfs=float(dbfs_values.max()),
            mean_dbfs=float(dbfs_values.mean()),
            mean_centroid_hz=float(centroid_values.mean()),
            dominant_freq_hz=float(dominant_values.mean()),
            dominant_span_hz=float(dominant_values.max() - dominant_values.min()),
            low_band_ratio=float(low_values.mean()),
            mid_band_ratio=float(mid_values.mean()),
            high_band_ratio=float(high_values.mean()),
            mean_flux=float(flux_values.mean()),
            mean_flatness=float(flatness_values.mean()),
            rms_modulation_depth=rms_modulation_depth,
            dominant_modulation_hz=dominant_modulation_hz,
            details={
                "dominant_freq_std_hz": float(dominant_values.std()),
                "zero_flux_frames": float((flux_values < 0.001).sum()),
            },
        )

    @staticmethod
    def _estimate_modulation_hz(values: np.ndarray, frame_hop_seconds: float) -> float:
        if values.size < 4:
            return 0.0
        centered = values - values.mean()
        if not np.any(centered):
            return 0.0
        spectrum = np.abs(np.fft.rfft(centered))
        freqs = np.fft.rfftfreq(values.size, d=frame_hop_seconds)
        mask = (freqs >= 0.25) & (freqs <= 12.0)
        if not np.any(mask):
            return 0.0
        candidate_index = np.argmax(spectrum[mask])
        return float(freqs[mask][candidate_index])
