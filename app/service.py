from __future__ import annotations

import logging
import signal
import subprocess
from pathlib import Path
from threading import Event, Thread

from waitress import create_server

from app.audio_devices import CaptureSelection, select_capture_device
from app.capture.base import AudioCaptureError
from app.capture.live import LiveAudioSource
from app.config import AppConfig
from app.pipeline import AudioPipeline, RuntimeStatus
from app.storage.database import SQLiteRepository
from app.web.app import create_app

LOGGER = logging.getLogger(__name__)


class LiveCaptureWorker:
    def __init__(self, config: AppConfig, pipeline: AudioPipeline, status: RuntimeStatus) -> None:
        self.config = config
        self.pipeline = pipeline
        self.status = status
        self.stop_event = Event()
        self._thread = Thread(target=self._run, name="live-capture-worker", daemon=True)
        self._source: LiveAudioSource | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._source is not None:
            self._source.close()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.pipeline.reset_runtime_state()
            try:
                selection = self._resolve_capture_selection()
                self._source = LiveAudioSource(
                    sample_rate=self.config.audio.sample_rate,
                    frame_duration_seconds=self.config.audio.frame_duration_seconds,
                    arecord_binary=self.config.audio.arecord_binary,
                    arecord_device=selection.device_spec,
                    source_name=selection.source_name,
                    channels=self.config.audio.channels,
                )
                self.status.update(
                    worker_state="starting",
                    audio_device=selection.device_spec,
                    audio_device_mode=selection.mode,
                    audio_device_name=selection.source_name,
                )
                self.pipeline.process_stream(self._source.frames(), stop_event=self.stop_event)
            except AudioCaptureError as exc:
                LOGGER.warning("Audio capture unavailable: %s", exc)
                self.status.update(worker_state="degraded", audio_available=False, last_error=str(exc))
            except Exception as exc:  # pragma: no cover - defensive logging around runtime loop
                LOGGER.exception("Unexpected failure in live capture loop")
                self.status.update(worker_state="error", audio_available=False, last_error=str(exc))
            finally:
                if self._source is not None:
                    self._source.close()
                    self._source = None
            if self.stop_event.wait(self.config.audio.retry_backoff_seconds):
                break
        self.status.update(worker_state="stopped", audio_available=False)

    def _resolve_capture_selection(self) -> CaptureSelection:
        selection = select_capture_device(
            arecord_binary=self.config.audio.arecord_binary,
            configured_device=self.config.audio.arecord_device,
        )
        LOGGER.info("Selected ALSA capture device: %s", selection.description)
        return selection


def run_web_server(repository: SQLiteRepository, status: RuntimeStatus, config: AppConfig) -> None:
    application = create_app(repository, status, config)
    server = create_server(application, host=config.web.host, port=config.web.port, threads=2)
    shutting_down = False

    def shutdown(signum: int, _frame) -> None:
        nonlocal shutting_down
        shutting_down = True
        LOGGER.info("Received signal %s, stopping web server.", signum)
        server.close()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    LOGGER.info("Web panel listening on %s:%s", config.web.host, config.web.port)
    try:
        server.run()
    except OSError as exc:
        if shutting_down:
            LOGGER.info("Web server closed during shutdown: %s", exc)
            return
        raise


class MonitorServiceRunner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.repository = SQLiteRepository(config.storage.database_path)
        self.status = RuntimeStatus()
        self.pipeline = AudioPipeline(config, self.repository, self.status)
        self.worker = LiveCaptureWorker(config, self.pipeline, self.status)

    def run(self) -> None:
        self.repository.initialize()
        self.config.storage.clip_dir.mkdir(parents=True, exist_ok=True)
        self.worker.start()
        try:
            run_web_server(self.repository, self.status, self.config)
        finally:
            self.worker.stop()
            self.worker.join(timeout=5)


def probe_audio_input(config: AppConfig) -> tuple[bool, str]:
    try:
        selection = select_capture_device(
            arecord_binary=config.audio.arecord_binary,
            configured_device=config.audio.arecord_device,
        )
    except AudioCaptureError as exc:
        return False, str(exc)
    command = [
        config.audio.arecord_binary,
        "-q",
        "-D",
        selection.device_spec,
        "-f",
        "S16_LE",
        "-r",
        str(config.audio.sample_rate),
        "-c",
        str(config.audio.channels),
        "-d",
        "1",
        "-t",
        "raw",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=5)
    except FileNotFoundError:
        return False, f"Missing binary: {config.audio.arecord_binary}"
    except subprocess.TimeoutExpired:
        return False, "Audio probe timed out after 5 seconds."
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        if "Device or resource busy" in stderr:
            return True, f"{selection.description}: device is busy, likely already in use by the running service."
        return False, f"{selection.description}: {stderr or 'Audio probe failed.'}"
    if not result.stdout:
        return False, f"{selection.description}: audio probe produced no samples."
    return True, f"{selection.description}: captured {len(result.stdout)} bytes from audio input."
