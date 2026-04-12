# Codex Repo Guide

## Goal

Keep context small by default while preserving full debugging and implementation capability.

## Read This First

- Start from the narrowest module that matches the task.
- Prefer `rg` scoped to `app/`, `tests/`, `configs/`, `scripts/`, `README.md`, `DEVELOPMENT.md`.
- Do not scan the whole repository unless the task is explicitly architectural.
- `.rgignore` excludes generated and binary-heavy paths by default. Override only when the task truly requires it.

## High-Value Entry Points

- CLI and app wiring: `app/main.py`
- Runtime loop and worker lifecycle: `app/service.py`, `app/pipeline.py`
- Classification: `app/classify/service.py`, `app/classify/heuristics.py`
- Event aggregation: `app/aggregate/event_aggregator.py`
- Storage and queries: `app/storage/database.py`, `app/storage/retention.py`, `app/storage/clips.py`
- Web UI: `app/web/app.py`, `app/web/templates/`
- Configuration: `app/config.py`, `configs/config.yaml`

## Minimal File Sets By Task

- Audio capture issue:
  `app/audio_devices.py`, `app/capture/`, `app/service.py`, relevant tests
- Detection or event segmentation issue:
  `app/detect/`, `app/aggregate/event_aggregator.py`, `app/pipeline.py`, relevant tests
- Classification issue:
  `app/classify/service.py`, `app/models.py`, `app/config.py`, `tests/unit/test_yamnet_classifier.py`
- Database or dashboard data issue:
  `app/storage/database.py`, `app/web/app.py`, relevant template, relevant tests
- Retention or clip persistence issue:
  `app/storage/retention.py`, `app/storage/clips.py`, `app/pipeline.py`, relevant tests
- CLI or operational issue:
  `app/main.py`, `scripts/`, `README.md`, `DEVELOPMENT.md`

## Default Search Patterns

- Code only:
  `rg -n "pattern" app tests configs scripts README.md DEVELOPMENT.md`
- File discovery:
  `rg --files app tests configs scripts`
- Database references:
  `rg -n "insert_event|get_dashboard|list_events|classifier_cache" app tests`

## Avoid By Default

- `.venv/`, `data/`, `models/`, generated logs, SQLite files, WAV clips, downloaded model assets
- broad reads of `README.md` when one module and its tests are enough
- loading unrelated templates or tests

## When To Override `.rgignore`

- Inspecting downloaded YAMNet assets
- Diagnosing local SQLite contents
- Verifying generated clips or spectrograms

Use a targeted command instead of disabling ignore globally.
