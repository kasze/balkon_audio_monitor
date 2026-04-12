from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from app.config import StorageConfig
from app.models import StoredClip
from app.storage.database import SQLiteRepository

LOGGER = logging.getLogger(__name__)


class ClipRetentionManager:
    def __init__(self, config: StorageConfig, repository: SQLiteRepository) -> None:
        self.config = config
        self.repository = repository

    def prepare_for_clip(self, estimated_clip_bytes: int) -> bool:
        self._prune_by_age()
        if self._has_capacity_for(estimated_clip_bytes):
            return True

        deleted_count, freed_bytes = self._prune_oldest_until(
            lambda: self._has_capacity_for(estimated_clip_bytes)
        )
        if deleted_count:
            LOGGER.info(
                "Freed %s old clips before save, reclaimed %.1f MiB",
                deleted_count,
                freed_bytes / (1024 * 1024),
            )

        if self._has_capacity_for(estimated_clip_bytes):
            return True

        LOGGER.warning(
            "Skipping clip save because storage limits are still exceeded: free=%s MiB reserve=%s MiB clip_total=%s MiB clip_limit=%s MiB",
            self._disk_free_megabytes(),
            self.config.min_free_disk_megabytes,
            round(self.repository.total_clip_bytes() / (1024 * 1024), 1),
            self.config.clip_max_megabytes,
        )
        return False

    def enforce_limits(self) -> None:
        self._prune_by_age()
        deleted_count, freed_bytes = self._prune_oldest_until(self._within_limits)
        if deleted_count:
            LOGGER.info(
                "Clip retention removed %s old clips, reclaimed %.1f MiB",
                deleted_count,
                freed_bytes / (1024 * 1024),
            )

    def _prune_by_age(self) -> None:
        if self.config.clip_max_age_days <= 0:
            return
        cutoff = (datetime.now().astimezone() - timedelta(days=self.config.clip_max_age_days)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        deleted_count = 0
        freed_bytes = 0
        while True:
            clips = self.repository.list_clips_created_before(cutoff, limit=32)
            if not clips:
                break
            for clip in clips:
                freed_bytes += self._delete_clip(clip)
                deleted_count += 1
        if deleted_count:
            LOGGER.info(
                "Clip retention removed %s clips older than %s days, reclaimed %.1f MiB",
                deleted_count,
                self.config.clip_max_age_days,
                freed_bytes / (1024 * 1024),
            )

    def _prune_oldest_until(self, predicate: Callable[[], bool]) -> tuple[int, int]:
        deleted_count = 0
        freed_bytes = 0
        while not predicate():
            clips = self.repository.list_oldest_clips(limit=32)
            if not clips:
                break
            progress_made = False
            for clip in clips:
                freed_bytes += self._delete_clip(clip)
                deleted_count += 1
                progress_made = True
                if predicate():
                    break
            if not progress_made:
                break
        return deleted_count, freed_bytes

    def _has_capacity_for(self, estimated_clip_bytes: int) -> bool:
        return self._clip_total_within_limit(estimated_clip_bytes) and self._free_disk_within_limit(
            estimated_clip_bytes
        )

    def _within_limits(self) -> bool:
        return self._clip_total_within_limit(0) and self._free_disk_within_limit(0)

    def _clip_total_within_limit(self, extra_bytes: int) -> bool:
        if self.config.clip_max_bytes <= 0:
            return True
        return self.repository.total_clip_bytes() + extra_bytes <= self.config.clip_max_bytes

    def _free_disk_within_limit(self, extra_bytes: int) -> bool:
        if self.config.min_free_disk_bytes <= 0:
            return True
        return self._disk_usage().free - extra_bytes >= self.config.min_free_disk_bytes

    def _disk_free_megabytes(self) -> float:
        return round(self._disk_usage().free / (1024 * 1024), 1)

    def _disk_usage(self):
        path = self.config.clip_dir if self.config.clip_dir.exists() else self.config.clip_dir.parent
        return shutil.disk_usage(path)

    def _delete_clip(self, clip: StoredClip) -> int:
        reclaimed_bytes = 0
        paths = [clip.path]
        if clip.spectrogram_path is not None:
            paths.append(clip.spectrogram_path)
        for path in paths:
            try:
                if path.exists():
                    reclaimed_bytes += path.stat().st_size
                    path.unlink()
            except OSError as exc:
                LOGGER.warning("Failed to remove artifact %s: %s", path, exc)
                return 0

        self.repository.delete_clip(clip.clip_id)
        self._prune_empty_directories(clip.path.parent)
        return reclaimed_bytes

    def _prune_empty_directories(self, path: Path) -> None:
        base_dir = self.config.clip_dir.resolve()
        current = path.resolve()
        while current != base_dir and base_dir in current.parents:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent
