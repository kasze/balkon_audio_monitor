from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from app.capture.base import AudioCaptureError

_ARECORD_LINE_RE = re.compile(
    r"^card (?P<card_index>\d+): (?P<card_id>\S+) \[(?P<card_name>.+?)\], "
    r"device (?P<device_index>\d+): (?P<device_id>.+?) \[(?P<device_name>.+?)\]$"
)


@dataclass(slots=True, frozen=True)
class CaptureDevice:
    card_index: int
    card_id: str
    card_name: str
    device_index: int
    device_id: str
    device_name: str

    @property
    def is_usb(self) -> bool:
        text = f"{self.card_name} {self.device_id} {self.device_name}".lower()
        return "usb" in text or "c-media" in text

    @property
    def device_spec(self) -> str:
        # `plughw` keeps the app tolerant to cards that expose only 44.1/48 kHz natively.
        return f"plughw:{self.card_index},{self.device_index}"

    @property
    def source_name(self) -> str:
        return f"{self.card_name} / {self.device_name}"


@dataclass(slots=True, frozen=True)
class CaptureSelection:
    device_spec: str
    source_name: str
    mode: str
    devices: tuple[CaptureDevice, ...]

    @property
    def description(self) -> str:
        return f"{self.device_spec} ({self.source_name}, {self.mode})"


def parse_arecord_hardware_list(output: str) -> list[CaptureDevice]:
    devices: list[CaptureDevice] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = _ARECORD_LINE_RE.match(line)
        if match is None:
            continue
        devices.append(
            CaptureDevice(
                card_index=int(match.group("card_index")),
                card_id=match.group("card_id"),
                card_name=match.group("card_name"),
                device_index=int(match.group("device_index")),
                device_id=match.group("device_id"),
                device_name=match.group("device_name"),
            )
        )
    return devices


def list_capture_devices(arecord_binary: str = "arecord") -> list[CaptureDevice]:
    try:
        result = subprocess.run(
            [arecord_binary, "-l"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except FileNotFoundError as exc:
        raise AudioCaptureError(f"Missing binary: {arecord_binary}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioCaptureError("Timed out while listing ALSA capture devices.") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise AudioCaptureError(stderr or "Failed to list ALSA capture devices.")
    return parse_arecord_hardware_list(result.stdout)


def select_capture_device(
    arecord_binary: str = "arecord",
    configured_device: str | None = None,
) -> CaptureSelection:
    if configured_device:
        try:
            devices = tuple(list_capture_devices(arecord_binary))
        except AudioCaptureError:
            devices = ()
        return CaptureSelection(
            device_spec=configured_device,
            source_name=configured_device,
            mode="manual",
            devices=devices,
        )
    devices = tuple(list_capture_devices(arecord_binary))
    if not devices:
        raise AudioCaptureError("No ALSA capture device detected. Check `arecord -l` and the USB audio card.")

    selected = sorted(devices, key=_device_sort_key, reverse=True)[0]
    return CaptureSelection(
        device_spec=selected.device_spec,
        source_name=selected.source_name,
        mode="auto",
        devices=devices,
    )


def _device_sort_key(device: CaptureDevice) -> tuple[int, int, int]:
    score = 0
    text = f"{device.card_name} {device.device_id} {device.device_name}".lower()
    if device.is_usb:
        score += 100
    if "mic" in text or "input" in text or "capture" in text:
        score += 10
    return (score, -device.card_index, -device.device_index)
