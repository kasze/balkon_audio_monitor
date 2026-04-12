from __future__ import annotations

from app.config import DetectionConfig
from app.models import DetectionState, FrameFeatures


class AdaptiveEnergyDetector:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.noise_floor_dbfs = self.config.initial_noise_floor_dbfs
        self.active = False
        self._candidate_frames = 0
        self._inactive_frames = 0

    def process(self, features: FrameFeatures) -> DetectionState:
        activation_score = features.dbfs - self.noise_floor_dbfs
        onset = (
            activation_score >= self.config.activation_margin_db
            and features.dbfs >= self.config.min_event_dbfs
        )
        sustain = activation_score >= self.config.release_margin_db

        if not self.active:
            if onset:
                self._candidate_frames += 1
            else:
                self._candidate_frames = 0
                self._update_noise_floor(features.dbfs)

            if self._candidate_frames >= self.config.min_active_frames:
                self.active = True
                self._inactive_frames = 0
        else:
            if sustain or features.spectral_flux > 0.001:
                self._inactive_frames = 0
            else:
                self._inactive_frames += 1
            if self._inactive_frames > self.config.max_inactive_frames:
                self.active = False
                self._candidate_frames = 0
                self._inactive_frames = 0
                self._update_noise_floor(features.dbfs)

        if not self.active:
            self._update_noise_floor(features.dbfs)

        return DetectionState(
            is_active=self.active,
            noise_floor_dbfs=self.noise_floor_dbfs,
            activation_score=activation_score,
        )

    def _update_noise_floor(self, dbfs: float) -> None:
        alpha = self.config.noise_floor_alpha
        self.noise_floor_dbfs = (1.0 - alpha) * self.noise_floor_dbfs + alpha * dbfs

