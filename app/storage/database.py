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


def _bucket_expression(bucket_mode: str) -> str:
    if bucket_mode == "month":
        return "substr(started_at, 1, 7) || '-01 00:00:00'"
    if bucket_mode == "day":
        return "substr(started_at, 1, 10) || ' 00:00:00'"
    return (
        "substr(started_at, 1, 14)"
        " || printf('%02d', (CAST(substr(started_at, 15, 2) AS INTEGER) / 10) * 10)"
        " || ':00'"
    )


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
                    user_label TEXT,
                    user_label_updated_at TEXT,
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
                CREATE INDEX IF NOT EXISTS idx_events_started_category ON events(started_at, category);
                CREATE INDEX IF NOT EXISTS idx_events_user_label_started ON events(user_label, started_at);
                CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);
                CREATE INDEX IF NOT EXISTS idx_classifier_decisions_name_event
                    ON classifier_decisions(classifier_name, event_id);
                CREATE INDEX IF NOT EXISTS idx_classifier_cache_lookup
                    ON classifier_cache(classifier_name, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_classifier_cache_signature_hash
                    ON classifier_cache(classifier_name, signature_hash);
                """
            )
            self._ensure_column(connection, "events", "user_label", "TEXT")
            self._ensure_column(connection, "events", "user_label_updated_at", "TEXT")
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
        day_start = datetime.strptime(day, "%Y-%m-%d")
        day_end = day_start + timedelta(days=1)
        return self.get_dashboard_range(
            started_at=day_start.strftime("%Y-%m-%d %H:%M:%S"),
            ended_at=day_end.strftime("%Y-%m-%d %H:%M:%S"),
            recent_limit=recent_limit,
            bucket_mode="ten_minute",
        )

    def get_dashboard_range(
        self,
        *,
        started_at: str,
        ended_at: str,
        recent_limit: int = 20,
        bucket_mode: str = "ten_minute",
    ) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            summary_events = connection.execute(
                """
                SELECT COUNT(*) AS event_count
                FROM events
                WHERE started_at >= ? AND started_at < ?
                """,
                (started_at, ended_at),
            ).fetchone()
            summary_noise = self._fetch_noise_summary(connection, started_at, ended_at, bucket_mode)
            categories = connection.execute(
                """
                SELECT COALESCE(user_label, category) AS category, COUNT(*) AS total
                FROM events
                WHERE started_at >= ? AND started_at < ?
                GROUP BY COALESCE(user_label, category)
                ORDER BY total DESC, COALESCE(user_label, category) ASC
                """,
                (started_at, ended_at),
            ).fetchall()
            bucket_rows = self._fetch_noise_buckets(connection, started_at, ended_at, bucket_mode)
            recent = connection.execute(
                """
                SELECT
                    e.id,
                    e.started_at,
                    e.ended_at,
                    COALESCE(c.duration_seconds, e.duration_seconds) AS duration_seconds,
                    COALESCE(user_label, category) AS category,
                    e.confidence,
                    e.peak_dbfs,
                    e.user_label
                FROM events e
                LEFT JOIN clips c ON c.id = e.clip_id
                WHERE e.started_at >= ? AND e.started_at < ?
                ORDER BY e.started_at DESC
                LIMIT ?
                """,
                (started_at, ended_at, recent_limit),
            ).fetchall()
            bird_species = connection.execute(
                """
                SELECT
                    e.id,
                    e.started_at,
                    e.ended_at,
                    COALESCE(c.duration_seconds, e.duration_seconds) AS duration_seconds,
                    d.category AS species,
                    d.confidence,
                    e.peak_dbfs
                FROM events e
                JOIN classifier_decisions d ON d.event_id = e.id
                LEFT JOIN clips c ON c.id = e.clip_id
                WHERE e.started_at >= ? AND e.started_at < ?
                  AND d.classifier_name = 'birdnet_remote'
                ORDER BY e.started_at DESC
                LIMIT ?
                """,
                (started_at, ended_at, recent_limit),
            ).fetchall()
            bird_species_counts = connection.execute(
                """
                SELECT d.category AS species, COUNT(*) AS total
                FROM events e
                JOIN classifier_decisions d ON d.event_id = e.id
                WHERE e.started_at >= ? AND e.started_at < ?
                  AND d.classifier_name = 'birdnet_remote'
                GROUP BY d.category
                ORDER BY total DESC, d.category ASC
                LIMIT ?
                """,
                (started_at, ended_at, recent_limit),
            ).fetchall()
        summary = {
            "event_count": int(summary_events["event_count"]) if summary_events and summary_events["event_count"] is not None else 0,
            "avg_dbfs": float(summary_noise["avg_dbfs"]) if summary_noise and summary_noise["avg_dbfs"] is not None else None,
            "max_dbfs": float(summary_noise["max_dbfs"]) if summary_noise and summary_noise["max_dbfs"] is not None else None,
        }
        return {
            "summary": summary,
            "categories": [dict(row) for row in categories],
            "ten_minute": [dict(row) for row in bucket_rows],
            "recent_noise": [dict(row) for row in bucket_rows],
            "recent_events": [dict(row) for row in recent],
            "bird_species": [dict(row) for row in bird_species],
            "bird_species_counts": [dict(row) for row in bird_species_counts],
        }

    def _fetch_noise_summary(
        self,
        connection: sqlite3.Connection,
        started_at: str,
        ended_at: str,
        bucket_mode: str,
    ) -> sqlite3.Row | None:
        if bucket_mode in {"day", "month"}:
            start_day = started_at[:10]
            end_day = ended_at[:10]
            return connection.execute(
                """
                SELECT
                    CASE WHEN SUM(interval_count) > 0
                         THEN SUM(avg_dbfs * interval_count) / SUM(interval_count)
                         ELSE NULL
                    END AS avg_dbfs,
                    MAX(max_dbfs) AS max_dbfs
                FROM daily_stats
                WHERE day >= ? AND day < ? AND interval_count > 0
                """,
                (start_day, end_day),
            ).fetchone()
        return connection.execute(
            """
            SELECT AVG(avg_dbfs) AS avg_dbfs, MAX(max_dbfs) AS max_dbfs
            FROM noise_intervals
            WHERE started_at >= ? AND started_at < ?
            """,
            (started_at, ended_at),
        ).fetchone()

    def _fetch_noise_buckets(
        self,
        connection: sqlite3.Connection,
        started_at: str,
        ended_at: str,
        bucket_mode: str,
    ) -> list[sqlite3.Row]:
        if bucket_mode == "day":
            return connection.execute(
                """
                SELECT
                    day || ' 00:00:00' AS bucket_start,
                    avg_dbfs,
                    max_dbfs,
                    interval_count
                FROM daily_stats
                WHERE day >= ? AND day < ? AND interval_count > 0
                ORDER BY day ASC
                """,
                (started_at[:10], ended_at[:10]),
            ).fetchall()
        if bucket_mode == "month":
            return connection.execute(
                """
                SELECT
                    substr(day, 1, 7) || '-01 00:00:00' AS bucket_start,
                    SUM(avg_dbfs * interval_count) / SUM(interval_count) AS avg_dbfs,
                    MAX(max_dbfs) AS max_dbfs,
                    SUM(interval_count) AS interval_count
                FROM daily_stats
                WHERE day >= ? AND day < ? AND interval_count > 0
                GROUP BY substr(day, 1, 7)
                ORDER BY bucket_start ASC
                """,
                (started_at[:10], ended_at[:10]),
            ).fetchall()
        return connection.execute(
            """
            SELECT
                """
            + _bucket_expression(bucket_mode)
            + """
                AS bucket_start,
                AVG(avg_dbfs) AS avg_dbfs,
                MAX(max_dbfs) AS max_dbfs,
                COUNT(*) AS interval_count
            FROM noise_intervals
            WHERE started_at >= ? AND started_at < ?
            GROUP BY bucket_start
            ORDER BY bucket_start ASC
            """,
            (started_at, ended_at),
        ).fetchall()

    def list_events(
        self,
        *,
        category: str | None = None,
        day: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if day:
            day_start = datetime.strptime(day, "%Y-%m-%d")
            day_end = day_start + timedelta(days=1)
            return self.list_events_range(
                category=category,
                started_at=day_start.strftime("%Y-%m-%d %H:%M:%S"),
                ended_at=day_end.strftime("%Y-%m-%d %H:%M:%S"),
                limit=limit,
            )
        return self.list_events_range(category=category, started_at=None, ended_at=None, limit=limit)

    def list_events_range(
        self,
        *,
        category: str | None = None,
        started_at: str | None,
        ended_at: str | None,
        limit: int | None = 100,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                e.id,
                e.started_at,
                e.ended_at,
                COALESCE(c.duration_seconds, e.duration_seconds) AS duration_seconds,
                COALESCE(e.user_label, e.category) AS category,
                e.confidence,
                e.peak_dbfs,
                e.user_label
            FROM events e
            LEFT JOIN clips c ON c.id = e.clip_id
        """
        conditions: list[str] = []
        params: list[Any] = []
        if category:
            conditions.append("COALESCE(e.user_label, e.category) = ?")
            params.append(category)
        if started_at:
            conditions.append("e.started_at >= ?")
            params.append(started_at)
        if ended_at:
            conditions.append("e.started_at < ?")
            params.append(ended_at)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY e.started_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with closing(self._connect()) as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def list_bird_events(
        self,
        *,
        day: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if day:
            day_start = datetime.strptime(day, "%Y-%m-%d")
            day_end = day_start + timedelta(days=1)
            return self.list_bird_events_range(
                started_at=day_start.strftime("%Y-%m-%d %H:%M:%S"),
                ended_at=day_end.strftime("%Y-%m-%d %H:%M:%S"),
                limit=limit,
            )
        return self.list_bird_events_range(started_at=None, ended_at=None, limit=limit)

    def list_bird_events_range(
        self,
        *,
        started_at: str | None,
        ended_at: str | None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                   e.id,
                   e.started_at,
                   e.ended_at,
                   COALESCE(c.duration_seconds, e.duration_seconds) AS duration_seconds,
                   d.category AS species,
                   d.confidence,
                   e.peak_dbfs
            FROM events e
            JOIN classifier_decisions d ON d.event_id = e.id
            LEFT JOIN clips c ON c.id = e.clip_id
            WHERE d.classifier_name = 'birdnet_remote'
        """
        params: list[Any] = []
        if started_at:
            query += " AND e.started_at >= ?"
            params.append(started_at)
        if ended_at:
            query += " AND e.started_at < ?"
            params.append(ended_at)
        query += " ORDER BY e.started_at DESC LIMIT ?"
        params.append(limit)

        with closing(self._connect()) as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def get_event(self, event_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT e.*, c.path AS clip_path, c.spectrogram_path AS spectrogram_path,
                       c.duration_seconds AS clip_duration, c.sample_rate AS clip_sample_rate,
                       COALESCE(e.user_label, e.category) AS effective_category
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

    def set_event_user_label(self, event_id: int, user_label: str | None) -> bool:
        normalized = user_label.strip() if user_label else None
        updated_at = _iso(datetime.now().astimezone()) if normalized else None
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE events
                SET user_label = ?, user_label_updated_at = ?
                WHERE id = ?
                """,
                (normalized, updated_at, event_id),
            )
            connection.commit()
        return cursor.rowcount > 0

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
                SELECT cc.event_id, cc.classifier_name, cc.classifier_version,
                       COALESCE(e.user_label, cc.category) AS category,
                       CASE WHEN e.user_label IS NOT NULL THEN MAX(cc.confidence, 0.99) ELSE cc.confidence END AS confidence,
                       cc.signature_hash, cc.signature_json, cc.details_json, e.user_label
                FROM classifier_cache cc
                LEFT JOIN events e ON e.id = cc.event_id
                WHERE cc.classifier_name = ?
                  AND (cc.confidence >= ? OR e.user_label IS NOT NULL)
                  AND cc.created_at >= ?
                ORDER BY cc.created_at DESC
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
                details={
                    **dict(json.loads(row["details_json"])),
                    **(
                        {
                            "manual_label": True,
                            "manual_label_category": str(row["user_label"]),
                        }
                        if row["user_label"]
                        else {}
                    ),
                },
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
