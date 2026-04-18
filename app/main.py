from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from app.audio_devices import select_capture_device
from app.capture.base import AudioCaptureError
from app.capture.live import LiveAudioSource
from app.capture.wav import WavFileSource
from app.config import AppConfig, StorageConfig, load_config, save_config
from app.logging_setup import configure_logging
from app.pipeline import AudioPipeline, RuntimeStatus
from app.service import MonitorServiceRunner, probe_audio_input, run_web_server
from app.storage.database import SQLiteRepository

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balcony audio monitor MVP")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Initialize SQLite schema.")
    subparsers.add_parser("service", help="Run live capture and web panel in one process.")
    subparsers.add_parser("web", help="Run only the web panel.")
    subparsers.add_parser("run-live", help="Run only live capture pipeline.")
    subparsers.add_parser("init-config", help="Create a local config and auto-detect the audio device.")
    subparsers.add_parser("check-audio", help="Probe ALSA input with a short capture.")
    subparsers.add_parser("detect-audio", help="List ALSA capture devices and show the selected input.")

    analyze_wav = subparsers.add_parser("analyze-wav", help="Analyze a single WAV file.")
    analyze_wav.add_argument("path", help="Path to a WAV file.")

    analyze_dir = subparsers.add_parser("analyze-dir", help="Analyze every WAV file in a directory.")
    analyze_dir.add_argument("directory", help="Path to a directory containing WAV files.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.logging.level)
    try:
        if args.command == "service":
            MonitorServiceRunner(config).run()
            return 0

        repository = SQLiteRepository(config.storage.database_path)
        repository.initialize()
        status = RuntimeStatus()
        pipeline = AudioPipeline(config, repository, status)

        if args.command == "init-db":
            LOGGER.info("Database initialized at %s", config.storage.database_path)
            return 0

        if args.command == "init-config":
            return init_config(config)

        if args.command == "web":
            run_web_server(repository, status, config, pipeline.live_audio_buffer)
            return 0

        if args.command == "run-live":
            return run_live_capture(config, pipeline)

        if args.command == "check-audio":
            ok, message = probe_audio_input(config)
            print(message)
            return 0 if ok else 1

        if args.command == "detect-audio":
            selection = select_capture_device(
                arecord_binary=config.audio.arecord_binary,
                configured_device=config.audio.arecord_device,
            )
            print(f"Selected: {selection.description}")
            if selection.devices:
                print("Candidates:")
                for device in selection.devices:
                    print(f"- {device.device_spec} | {device.source_name}")
            else:
                print("Candidates: none")
            return 0

        if args.command == "analyze-wav":
            return analyze_wav(Path(args.path), config, pipeline)

        if args.command == "analyze-dir":
            return analyze_directory(Path(args.directory), config, pipeline)

        raise ValueError(f"Unsupported command: {args.command}")
    except AudioCaptureError as exc:
        LOGGER.error("%s", exc)
        return 1


def run_live_capture(config: AppConfig, pipeline: AudioPipeline) -> int:
    selection = select_capture_device(
        arecord_binary=config.audio.arecord_binary,
        configured_device=config.audio.arecord_device,
    )
    source = LiveAudioSource(
        sample_rate=config.audio.sample_rate,
        frame_duration_seconds=config.audio.frame_duration_seconds,
        arecord_binary=config.audio.arecord_binary,
        arecord_device=selection.device_spec,
        source_name=selection.source_name,
        channels=config.audio.channels,
    )
    pipeline.reset_runtime_state()
    try:
        pipeline.process_stream(source.frames())
    finally:
        source.close()
    return 0


def init_config(config: AppConfig) -> int:
    target_path = config.config_path or (config.base_dir / "configs/config.yaml")
    if target_path.exists():
        print(f"Config already exists at {target_path}")
        return 0

    storage = StorageConfig(
        database_path=(config.base_dir / "data/db/audio_monitor.sqlite3").resolve(),
        clip_dir=(config.base_dir / "data/clips").resolve(),
        keep_clips=config.storage.keep_clips,
        clip_max_megabytes=config.storage.clip_max_megabytes,
        clip_max_age_days=config.storage.clip_max_age_days,
        min_free_disk_megabytes=config.storage.min_free_disk_megabytes,
    )
    selected_description = None
    audio = config.audio
    try:
        selection = select_capture_device(
            arecord_binary=config.audio.arecord_binary,
            configured_device=config.audio.arecord_device,
        )
        selected_description = selection.description
        if config.audio.arecord_device is None and selection.device_spec:
            audio = replace(audio, arecord_device=selection.device_spec)
    except AudioCaptureError as exc:
        LOGGER.warning("Audio device auto-detection failed: %s", exc)

    bootstrap_config = AppConfig(
        base_dir=config.base_dir,
        audio=audio,
        detection=config.detection,
        aggregation=config.aggregation,
        classifier=config.classifier,
        storage=storage,
        web=config.web,
        logging=config.logging,
        config_path=target_path,
    )

    saved_path = save_config(bootstrap_config, target_path)
    print(f"Created config at {saved_path}")
    if selected_description:
        print(f"Selected capture device: {selected_description}")
    else:
        print("Audio device remains auto-detected at runtime.")
    return 0


def analyze_wav(path: Path, config: AppConfig, pipeline: AudioPipeline) -> int:
    LOGGER.info("Analyzing WAV file: %s", path)
    pipeline.reset_runtime_state()
    source = WavFileSource(path, config.audio.sample_rate, config.audio.frame_duration_seconds)
    try:
        pipeline.process_stream(source.frames())
    finally:
        source.close()
    return 0


def analyze_directory(directory: Path, config: AppConfig, pipeline: AudioPipeline) -> int:
    wav_files = sorted(path for path in directory.glob("*.wav"))
    if not wav_files:
        LOGGER.warning("No WAV files found in %s", directory)
        return 1
    for wav_file in wav_files:
        analyze_wav(wav_file, config, pipeline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
