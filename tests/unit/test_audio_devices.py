from __future__ import annotations

from unittest.mock import patch

from app.audio_devices import parse_arecord_hardware_list, select_capture_device


ARECORD_OUTPUT = """
**** List of CAPTURE Hardware Devices ****
card 2: Set [C-Media USB Headphone Set], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 3: Generic [Generic Analog], device 0: Mic [Mic]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
"""


def test_parse_arecord_hardware_list_extracts_devices() -> None:
    devices = parse_arecord_hardware_list(ARECORD_OUTPUT)

    assert len(devices) == 2
    assert devices[0].card_index == 2
    assert devices[0].device_spec == "plughw:2,0"
    assert devices[0].is_usb is True


def test_select_capture_device_prefers_usb_when_auto() -> None:
    with patch("app.audio_devices.list_capture_devices") as mocked:
        mocked.return_value = parse_arecord_hardware_list(ARECORD_OUTPUT)
        selection = select_capture_device()

    assert selection.mode == "auto"
    assert selection.device_spec == "plughw:2,0"
    assert "C-Media USB Headphone Set" in selection.source_name


def test_select_capture_device_honors_manual_override() -> None:
    with patch("app.audio_devices.list_capture_devices") as mocked:
        mocked.return_value = parse_arecord_hardware_list(ARECORD_OUTPUT)
        selection = select_capture_device(configured_device="hw:9,1")

    assert selection.mode == "manual"
    assert selection.device_spec == "hw:9,1"
