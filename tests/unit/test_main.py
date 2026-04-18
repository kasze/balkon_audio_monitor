from __future__ import annotations

from pathlib import Path

from app.audio_devices import CaptureDevice, CaptureSelection
from app.config import AppConfig, load_config
from app.main import init_config


def test_init_config_creates_config_with_detected_device(tmp_path: Path, monkeypatch) -> None:
    config = AppConfig(
        base_dir=tmp_path,
        config_path=tmp_path / "configs" / "config.yaml",
    )

    def fake_select_capture_device(**_kwargs):
        return CaptureSelection(
            device_spec="plughw:2,0",
            source_name="C-Media USB Headphone Set / USB Audio",
            mode="auto",
            devices=(
                CaptureDevice(
                    card_index=2,
                    card_id="Set",
                    card_name="C-Media USB Headphone Set",
                    device_index=0,
                    device_id="USB Audio",
                    device_name="USB Audio",
                ),
            ),
        )

    monkeypatch.setattr("app.main.select_capture_device", fake_select_capture_device)

    assert init_config(config) == 0

    saved = load_config(config.config_path)
    assert saved.audio.arecord_device == "plughw:2,0"
    assert saved.storage.database_path == (tmp_path / "data/db/audio_monitor.sqlite3").resolve()
