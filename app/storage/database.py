from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.models import (
    ClassifierCacheEntry,
    ClassifierDecision,
    ClipMetadata,
    CompletedEvent,
    NoiseInterval,
    PersistedEvent,
    StoredClip,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _serialize_summary(event: CompletedEvent) -> str:
    summary = asdict(event.summary)
    summary["started_at"] = _iso(event.summary.started_at)
    summary["ended_at"] = _iso(event.summary.ended_at)
    return json.dumps(summary)


class SQLiteRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS noise_intervals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    avg_rms REAL NOT NULL,
                    avg_dbfs REAL NOT NULL,
                    max_dbfs REAL NOT NULL,
                    avg_centroid_hz REAL NOT NULL,
                    low_band_ratio REAL NOT NULL,
                    mid_band_ratio REAL NOT NULL,
                    high_band_ratio REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    spectrogram_path TEXT,
                    sample_rate INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    byte_size INTEGER NOT NULL,
                    spectrogram_byte_size INTEGER NOT NULL DEFAULT 0,
                    sha1 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    peak_dbfs REAL NOT NULL,
                    mean_dbfs REAL NOT NULL,
                    mean_centroid_hz REAL NOT NULL,
                    dominant_freq_hz REAL NOT NULL,
                    dominant_span_hz REAL NOT NULL,
                    low_band_ratio REAL NOT NULL,
                    mid_band_ratio REAL NOT NULL,
                    high_band_ratio REAL NOT NULL,
                    clip_id INTEGER,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (clip_id) REFERENCES clips(id)
                );

                CREATE TABLE IF NOT EXISTS classifier_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    classifier_name TEXT NOT NULL,
                    classifier_version TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (event_id) REFERENCES events(id)
                );

                CREATE TABLE IF NOT EXISTS classifier_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    classifier_name TEXT NOT NULL,
                    classifier_version TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    signature_hash TEXT NOT NULL,
                    signature_json TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (event_id) REFERENCES events(id)
                );

                CREATE TABLE IF NOT EXISTS hourly_stats (
                    bucket_start TEXT PRIMARY KEY,
                    avg_dbfs REAL NOT NULL,
                    max_dbfs REAL NOT NULL,
                    interval_count INTEGER NOT NULL,
                    event_count INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS daily_stats (
                    day TEXT PRIMARY KEY,
                    avg_dbfs REAL NOT NULL,
                    max_dbfs REAL NOT NULL,
                    interval_count INTEGER NOT NULL,
                    event_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_noise_intervals_started_at ON noise_intervals(started_at);
                CREATE INDEX IF NOT EXISTS idx_events_started_at ON events(started_at);
                CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);
                CREATE INDEX IF NOT EXISTS idx_classifier_cache_lookup
                    ON classifier_cache(classifier_name, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_classifier_cache_signature_hash
                    ON classifier_cache(classifier_name, signature_hash);
                """
            )
            self._ensure_column(connection, "clips", "spectrogram_path", "TEXT")
            self._ensure_column(connection, "clips", "spectrogram_byte_size", "INTEGER NOT NULL DEFAULT 0")
            connection.commit()

    def insert_noise_interval(self, interval: NoiseInterval) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO noise_intervals (
                    source_name, started_at, ended_at, avg_rms, avg_dbfs, max_dbfs,
                    avg_centroid_hz, low_band_ratio, mid_band_ratio, high_band_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interval.source_name,
                    _iso(interval.started_at),
                    _iso(interval.ended_at),
                    interval.avg_rms,
                    interval.avg_dbfs,
                    interval.max_dbfs,
                    interval.avg_centroid_hz,
                    interval.low_band_ratio,
                    interval.mid_band_ratio,
                    interval.high_band_ratio,
                ),
            )
            self._upsert_interval_stats(connection, interval)
            connection.commit()

    def insert_event(
        self,
        event: CompletedEvent,
        decision: ClassifierDecision,
        clip: ClipMetadata | None,
    ) -> PersistedEvent:
        created_at = _iso(datetime.now().astimezone())
        with closing(self._connect()) as connection:
            clip_id = None
            clip_path = None
            spectrogram_path = None
            if clip is not None:
                cursor = connection.execute(
                    """
                    INSERT INTO clips (
                        path, spectrogram_path, sample_rate, channels, duration_seconds,
                        byte_size, spectrogram_byte_size, sha1, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(clip.path),
                        str(clip.spectrogram_path) if clip.spectrogram_path else None,
                        clip.sample_rate,
                        clip.channels,
                        clip.duration_seconds,
                        clip.byte_size,
                        clip.spectrogram_byte_size,
                        clip.sha1,
                        created_at,
                    ),
                )
                clip_id = int(cursor.lastrowid)
                clip_path = str(clip.path)
                spectrogram_path = str(clip.spectrogram_path) if clip.spectrogram_path else None

            summary = event.summary
            cursor = connection.execute(
                """
                INSERT INTO events (
                    source_name, started_at, ended_at, duration_seconds, category, confidence,
                    peak_dbfs, mean_dbfs, mean_centroid_hz, dominant_freq_hz, dominant_span_hz,
                    low_band_ratio, mid_band_ratio, high_band_ratio, clip_id, summary_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.source_name,
                    _iso(summary.started_at),
                    _iso(summary.ended_at),
                    summary.duration_seconds,
                    decision.category,
                    decision.confidence,
                    summary.peak_dbfs,
                    summary.mean_dbfs,
                    summary.mean_centroid_hz,
                    summary.dominant_freq_hz,
                    summary.dominant_span_hz,
                    summary.low_band_ratio,
                    summary.mid_band_ratio,
                    summary.high_band_ratio,
                    clip_id,
                    _serialize_summary(event),
                    created_at,
                ),
            )
            event_id = int(cursor.lastrowid)
            connection.execute(
                """
                INSERT INTO classifier_decisions (
                    event_id, classifier_name, classifier_version, category, confidence, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    decision.classifier_name,
                    decision.classifier_version,
                    decision.category,
                    decision.confidence,
                    json.dumps(decision.details),
                    created_at,
                ),
            )
            self._upsert_event_stats(connection, event)
            connection.commit()
        return PersistedEvent(
            event_id=event_id,
            category=decision.category,
            clip_path=clip_path,
            spectrogram_path=spectrogram_path,
        )

    def get_dashboard(self, day: str, recent_limit: int = 20) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            summary = connection.execute(
                """
                SELECT event_count, avg_dbfs, max_dbfs
                FROM daily_stats
                WHERE day = ?
                """,
                (day,),
            ).fetchone()
            categories = connection.execute(
                """
                SELECT category, COUNT(*) AS total
                FROM events
                WHERE substr(started_at, 1, 10) = ?
                GROUP BY category
                ORDER BY total DESC, category ASC
                """,
                (day,),
            ).fetchall()
            hourly = connection.execute(
                """
                SELECT bucket_start, avg_dbfs, max_dbfs, event_count
                FROM hourly_stats
                WHERE substr(bucket_start, 1, 10) = ?
                ORDER BY bucket_start ASC
                """,
                (day,),
            ).fetchall()
            recent = connection.execute(
                """
                SELECT id, started_at, ended_at, duration_seconds, category, confidence, peak_dbfs
                FROM events
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (recent_limit,),
            ).fetchall()
        return {
            "summary": dict(summary) if summary else {"event_count": 0, "avg_dbfs": None, "max_dbfs": None},
            "categories": [dict(row) for row in categories],
            "hourly": [dict(row) for row in hourly],
            "recent_events": [dict(row) for row in recent],
        }

    def get_event(self, event_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT e.*, c.path AS clip_path, c.spectrogram_path AS spectrogram_path,
                       c.duration_seconds AS clip_duration, c.sample_rate AS clip_sample_rate
                FROM events e
                LEFT JOIN clips c ON c.id = e.clip_id
                WHERE e.id = ?
                """,
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            decision_rows = connection.execute(
                """
                SELECT classifier_name, classifier_version, category, confidence, details_json, created_at
                FROM classifier_decisions
                WHERE event_id = ?
                ORDER BY created_at ASC
                """,
                (event_id,),
            ).fetchall()

        event = dict(row)
        event["summary"] = json.loads(event["summary_json"])
        event["decisions"] = [
            {**dict(item), "details": json.loads(item["details_json"])} for item in decision_rows
        ]
        return event

    def insert_classifier_cache_entry(
        self,
        event_id: int,
        classifier_name: str,
        classifier_version: str,
        category: str,
        confidence: float,
        signature_hash: str,
        signature: list[float],
        details: dict[str, Any],
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO classifier_cache (
                    event_id, classifier_name, classifier_version, category, confidence,
                    signature_hash, signature_json, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    classifier_name,
                    classifier_version,
                    category,
                    confidence,
                    signature_hash,
                    json.dumps(signature),
                    json.dumps(details),
                    _iso(datetime.now().astimezone()),
                ),
            )
            connection.commit()

    def list_classifier_cache_entries(
        self,
        classifier_name: str,
        min_confidence: float,
        lookback_days: int,
        limit: int,
    ) -> list[ClassifierCacheEntry]:
        created_after = _iso(datetime.now().astimezone() - timedelta(days=lookback_days))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT event_id, classifier_name, classifier_version, category, confidence,
                       signature_hash, signature_json, details_json
                FROM classifier_cache
                WHERE classifier_name = ?
                  AND confidence >= ?
                  AND created_at >= ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (classifier_name, min_confidence, created_after, limit),
            ).fetchall()
        return [
            ClassifierCacheEntry(
                event_id=int(row["event_id"]),
                classifier_name=str(row["classifier_name"]),
                classifier_version=str(row["classifier_version"]),
                category=str(row["category"]),
                confidence=float(row["confidence"]),
                signature_hash=str(row["signature_hash"]),
                signature=list(json.loads(row["signature_json"])),
                details=dict(json.loads(row["details_json"])),
            )
            for row in rows
        ]

    def total_clip_bytes(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(byte_size + COALESCE(spectrogram_byte_size, 0)), 0) AS total FROM clips"
            ).fetchone()
        return int(row["total"]) if row else 0

    def list_oldest_clips(self, limit: int) -> list[StoredClip]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, path, spectrogram_path, byte_size, spectrogram_byte_size, created_at
                FROM clips
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            StoredClip(
                clip_id=int(row["id"]),
                path=Path(str(row["path"])),
                spectrogram_path=Path(str(row["spectrogram_path"])) if row["spectrogram_path"] else None,
                byte_size=int(row["byte_size"]),
                spectrogram_byte_size=int(row["spectrogram_byte_size"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def list_clips_created_before(self, created_before: str, limit: int) -> list[StoredClip]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, path, spectrogram_path, byte_size, spectrogram_byte_size, created_at
                FROM clips
                WHERE created_at < ?
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (created_before, limit),
            ).fetchall()
        return [
            StoredClip(
                clip_id=int(row["id"]),
                path=Path(str(row["path"])),
                spectrogram_path=Path(str(row["spectrogram_path"])) if row["spectrogram_path"] else None,
                byte_size=int(row["byte_size"]),
                spectrogram_byte_size=int(row["spectrogram_byte_size"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def delete_clip(self, clip_id: int) -> None:
        with closing(self._connect()) as connection:
            connection.execute("UPDATE events SET clip_id = NULL WHERE clip_id = ?", (clip_id,))
            connection.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
            connection.commit()

    def recent_events_count(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM events").fetchone()
        return int(row["total"]) if row else 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_column(self, connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        if any(str(row["name"]) == column for row in rows):
            return
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _upsert_interval_stats(self, connection: sqlite3.Connection, interval: NoiseInterval) -> None:
        bucket_start = interval.started_at.astimezone().strftime("%Y-%m-%d %H:00:00")
        day = interval.started_at.astimezone().strftime("%Y-%m-%d")
        connection.execute(
            """
            INSERT INTO hourly_stats (bucket_start, avg_dbfs, max_dbfs, interval_count, event_count)
            VALUES (?, ?, ?, 1, 0)
            ON CONFLICT(bucket_start) DO UPDATE SET
                avg_dbfs = ((hourly_stats.avg_dbfs * hourly_stats.interval_count) + excluded.avg_dbfs)
                    / (hourly_stats.interval_count + 1),
                max_dbfs = MAX(hourly_stats.max_dbfs, excluded.max_dbfs),
                interval_count = hourly_stats.interval_count + 1
            """,
            (bucket_start, interval.avg_dbfs, interval.max_dbfs),
        )
        connection.execute(
            """
            INSERT INTO daily_stats (day, avg_dbfs, max_dbfs, interval_count, event_count, updated_at)
            VALUES (?, ?, ?, 1, 0, ?)
            ON CONFLICT(day) DO UPDATE SET
                avg_dbfs = ((daily_stats.avg_dbfs * daily_stats.interval_count) + excluded.avg_dbfs)
                    / (daily_stats.interval_count + 1),
                max_dbfs = MAX(daily_stats.max_dbfs, excluded.max_dbfs),
                interval_count = daily_stats.interval_count + 1,
                updated_at = excluded.updated_at
            """,
            (day, interval.avg_dbfs, interval.max_dbfs, _iso(datetime.now().astimezone())),
        )

    def _upsert_event_stats(self, connection: sqlite3.Connection, event: CompletedEvent) -> None:
        bucket_start = event.summary.started_at.astimezone().strftime("%Y-%m-%d %H:00:00")
        day = event.summary.started_at.astimezone().strftime("%Y-%m-%d")
        connection.execute(
            """
            INSERT INTO hourly_stats (bucket_start, avg_dbfs, max_dbfs, interval_count, event_count)
            VALUES (?, ?, ?, 0, 1)
            ON CONFLICT(bucket_start) DO UPDATE SET
                event_count = hourly_stats.event_count + 1
            """,
            (bucket_start, event.summary.mean_dbfs, event.summary.peak_dbfs),
        )
        connection.execute(
            """
            INSERT INTO daily_stats (day, avg_dbfs, max_dbfs, interval_count, event_count, updated_at)
            VALUES (?, ?, ?, 0, 1, ?)
            ON CONFLICT(day) DO UPDATE SET
                event_count = daily_stats.event_count + 1,
                updated_at = excluded.updated_at
            """,
            (day, event.summary.mean_dbfs, event.summary.peak_dbfs, _iso(datetime.now().astimezone())),
        )
