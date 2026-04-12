from __future__ import annotations

from datetime import datetime

from app.config import DetectionConfig
from app.detect.detector import AdaptiveEnergyDetector
from app.models import FrameFeatures


def make_features(dbfs: float, spectral_flux: float = 0.0) -> FrameFeatures:
    return FrameFeatures(
        started_at=datetime.now().astimezone(),
        duration_seconds=0.5,
        source_name="test",
        rms=10 ** (dbfs / 20),
        dbfs=dbfs,
        spectral_centroid_hz=800.0,
        dominant_freq_hz=900.0,
        low_band_ratio=0.2,
        mid_band_ratio=0.5,
        high_band_ratio=0.3,
        spectral_flux=spectral_flux,
        flatness=0.2,
        zero_crossing_rate=0.1,
    )


def test_detector_activates_after_sustained_energy() -> None:
    detector = AdaptiveEnergyDetector(
        DetectionConfig(
            initial_noise_floor_dbfs=-60.0,
            activation_margin_db=8.0,
            release_margin_db=4.0,
            min_event_dbfs=-48.0,
            min_active_frames=2,
            max_inactive_frames=2,
            noise_floor_alpha=0.05,
        )
    )

    detector.process(make_features(-60.0))
    first = detector.process(make_features(-45.0, spectral_flux=0.002))
    second = detector.process(make_features(-44.0, spectral_flux=0.003))

    assert first.is_active is False
    assert second.is_active is True


def test_detector_releases_after_multiple_quiet_frames() -> None:
    detector = AdaptiveEnergyDetector(DetectionConfig(max_inactive_frames=2))

    detector.process(make_features(-42.0, spectral_flux=0.01))
    detector.process(make_features(-41.0, spectral_flux=0.01))
    assert detector.process(make_features(-40.0, spectral_flux=0.02)).is_active is True

    detector.process(make_features(-58.0))
    detector.process(make_features(-59.0))
    released = detector.process(make_features(-60.0))

    assert released.is_active is False

