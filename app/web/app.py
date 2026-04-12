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
        return {"category_labels": CATEGORY_LABELS, "status": status.snapshot()}

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
