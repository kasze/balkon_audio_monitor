from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000


def save_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm_i16.tobytes())


def generate_siren(duration: float = 8.0) -> np.ndarray:
    t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    sweep = 650 + 450 * np.sin(2 * math.pi * 0.55 * t)
    phase = 2 * math.pi * np.cumsum(sweep) / SAMPLE_RATE
    tone = 0.35 * np.sin(phase)
    noise = 0.02 * np.random.default_rng(7).normal(size=t.shape)
    return tone + noise


def generate_aircraft(duration: float = 12.0) -> np.ndarray:
    t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    rumble = 0.28 * np.sin(2 * math.pi * 120 * t)
    harmonic = 0.12 * np.sin(2 * math.pi * 240 * t)
    envelope = np.linspace(0.2, 1.0, t.size)
    noise = 0.03 * np.random.default_rng(9).normal(size=t.shape)
    return envelope * (rumble + harmonic) + noise


if __name__ == "__main__":
    target_dir = Path(__file__).resolve().parent
    save_wav(target_dir / "demo_siren.wav", generate_siren())
    save_wav(target_dir / "demo_aircraft.wav", generate_aircraft())

