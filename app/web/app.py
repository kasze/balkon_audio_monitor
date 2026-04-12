from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, send_file

from app.classify.heuristics import CATEGORY_LABELS
from app.config import AppConfig
from app.pipeline import RuntimeStatus
from app.storage.database import SQLiteRepository


def create_app(repository: SQLiteRepository, status: RuntimeStatus, config: AppConfig) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.context_processor
    def inject_helpers() -> dict[str, object]:
        return {
            "category_labels": CATEGORY_LABELS,
            "describe_classifier_decision": _describe_classifier_decision,
            "format_local_timestamp": _format_local_timestamp,
            "status": status.snapshot(),
        }

    @app.get("/")
    def index():
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        dashboard = repository.get_dashboard(today, config.web.recent_events_limit)
        chart = _build_chart(dashboard["hourly"])
        return render_template(
            "index.html",
            day=today,
            dashboard=dashboard,
            chart=chart,
        )

    @app.get("/events/<int:event_id>")
    def event_details(event_id: int):
        event = repository.get_event(event_id)
        if event is None:
            abort(404)
        return render_template("event.html", event=event)

    @app.get("/clips/<int:event_id>")
    def clip_audio(event_id: int):
        event = repository.get_event(event_id)
        if event is None or not event.get("clip_path"):
            abort(404)
        return send_file(Path(event["clip_path"]))

    @app.get("/health")
    def health():
        snapshot = status.snapshot()
        snapshot["database_path"] = str(config.storage.database_path)
        return jsonify(snapshot)

    return app


def _build_chart(rows: list[dict[str, object]]) -> dict[str, object] | None:
    if not rows:
        return None
    labels = [str(row["bucket_start"])[11:16] for row in rows]
    values = [float(row["avg_dbfs"]) for row in rows]
    width = 760
    height = 220
    min_value = min(values)
    max_value = max(values)
    if max_value - min_value < 1.0:
        max_value += 0.5
        min_value -= 0.5

    points: list[str] = []
    for index, value in enumerate(values):
        x = 20 + index * ((width - 40) / max(1, len(values) - 1))
        normalized = (value - min_value) / (max_value - min_value)
        y = height - 20 - normalized * (height - 40)
        points.append(f"{x:.1f},{y:.1f}")

    return {
        "width": width,
        "height": height,
        "polyline": " ".join(points),
        "labels": labels,
        "min_value": round(min_value, 1),
        "max_value": round(max_value, 1),
    }


def _format_local_timestamp(value: str | None) -> str:
    if not value:
        return "brak"

    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return value

    local_timestamp = timestamp.astimezone()
    hundredths = local_timestamp.microsecond // 10_000
    return f"{local_timestamp:%Y-%m-%d %H:%M:%S}.{hundredths:02d}"


def _describe_classifier_decision(decision: dict[str, object]) -> dict[str, object]:
    details = decision.get("details")
    normalized_details = details if isinstance(details, dict) else {}
    classifier_name = str(decision.get("classifier_name") or "unknown")
    cache_hit = bool(normalized_details.get("cache_hit"))
    external_api_name = normalized_details.get("external_api_name") or normalized_details.get("api_name")
    used_external_api = bool(normalized_details.get("used_external_api") or external_api_name)

    if used_external_api:
        source = "external_api"
        source_label = f"Zewnetrzne API: {external_api_name or 'nieznane'}"
    elif cache_hit:
        source = "cache_reuse"
        source_label = "Reuse z lokalnego cache"
    elif classifier_name.startswith("yamnet"):
        source = "local_yamnet"
        source_label = "Lokalny YAMNet (LiteRT)"
    elif classifier_name.startswith("heuristic"):
        source = "heuristic_fallback"
        source_label = "Fallback heurystyczny"
    else:
        source = "other"
        source_label = classifier_name

    top_labels_raw = normalized_details.get("top_labels")
    top_labels: list[dict[str, object]] = []
    if isinstance(top_labels_raw, list):
        for item in top_labels_raw[:5]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "unknown")
            mean_score = item.get("mean_score")
            peak_score = item.get("peak_score")
            top_labels.append(
                {
                    "label": label,
                    "mean_score": float(mean_score) if isinstance(mean_score, int | float) else None,
                    "peak_score": float(peak_score) if isinstance(peak_score, int | float) else None,
                }
            )

    category_scores_raw = normalized_details.get("category_scores")
    category_scores: list[dict[str, object]] = []
    if isinstance(category_scores_raw, dict):
        sorted_scores = sorted(
            ((str(name), float(score)) for name, score in category_scores_raw.items() if isinstance(score, int | float)),
            key=lambda item: item[1],
            reverse=True,
        )
        category_scores = [{"category": name, "score": score} for name, score in sorted_scores[:5]]

    return {
        "classifier_name": classifier_name,
        "classifier_version": str(decision.get("classifier_version") or "-"),
        "source": source,
        "source_label": source_label,
        "used_external_api": used_external_api,
        "external_api_name": str(external_api_name) if external_api_name else None,
        "cache_hit": cache_hit,
        "cache_similarity": normalized_details.get("cache_similarity"),
        "cache_source_event_id": normalized_details.get("cache_source_event_id"),
        "fallback_reason": normalized_details.get("fallback_reason"),
        "top_labels": top_labels,
        "category_scores": category_scores,
    }
