from __future__ import annotations

import csv
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

from app.classify.heuristics import CATEGORY_LABELS
from app.config import AppConfig
from app.pipeline import RuntimeStatus
from app.storage.database import SQLiteRepository

CLASSIFIER_LABELS = {
    "yamnet_litert": "Lokalny YAMNet (LiteRT)",
    "heuristic_baseline": "Heurystyczny klasyfikator bazowy",
    "heuristic_fallback": "Heurystyczny fallback",
    "birdnet_remote": "Zdalny BirdNET",
}

YAMNET_LABEL_TRANSLATIONS = {
    "Speech": "Mowa",
    "Conversation": "Rozmowa",
    "Narration, speech": "Narracja, wypowiedź",
    "Narration, monologue": "Narracja, monolog",
    "Male speech, man speaking": "Męska mowa",
    "Female speech, woman speaking": "Kobieca mowa",
    "Child speech, kid speaking": "Mowa dziecka",
    "Shout": "Krzyk",
    "Yell": "Wrzask",
    "Whispering": "Szept",
    "Singing": "Śpiew",
    "Choir": "Chór",
    "Female singing": "Śpiew kobiecy",
    "Male singing": "Śpiew męski",
    "Babbling": "Gaworzenie",
    "Laughter": "Śmiech",
    "Chuckle, chortle": "Chichot",
    "Snicker": "Parsknięcie śmiechem",
    "Inside, small room": "Wewnątrz, mały pokój",
    "Silence": "Cisza",
    "Animal": "Zwierzę",
    "Ambulance (siren)": "Karetka (syrena)",
    "Siren": "Syrena",
    "Civil defense siren": "Syrena alarmowa obrony cywilnej",
    "Police car (siren)": "Radiowóz (syrena)",
    "Car alarm": "Alarm samochodowy",
    "Fire engine, fire truck (siren)": "Wóz strażacki (syrena)",
    "Truck": "Ciężarówka",
    "Fixed-wing aircraft, airplane": "Samolot",
    "Aircraft": "Statek powietrzny",
    "Aircraft engine": "Silnik samolotu",
    "Jet engine": "Silnik odrzutowy",
    "Helicopter": "Śmigłowiec",
    "Traffic noise, roadway noise": "Hałas uliczny",
    "Vehicle": "Pojazd",
    "Car": "Samochód",
    "Engine": "Silnik",
    "Outside, urban or manmade": "Na zewnątrz, środowisko miejskie lub sztuczne",
    "Burping, eructation": "Beknięcie",
    "Bird": "Ptak",
    "Bird vocalization, bird call, bird song": "Głos ptaka, śpiew ptaka",
    "Bird flight, flapping wings": "Lot ptaka, trzepot skrzydeł",
}


def create_app(repository: SQLiteRepository, status: RuntimeStatus, config: AppConfig) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    manual_label_options = _load_manual_label_options(config.classifier.yamnet_class_map_path)
    manual_label_values = {item["value"] for item in manual_label_options}

    @app.context_processor
    def inject_helpers() -> dict[str, object]:
        return {
            "category_labels": CATEGORY_LABELS,
            "describe_classifier_decision": _describe_classifier_decision,
            "format_dbfs": _format_dbfs,
            "format_local_timestamp": _format_local_timestamp,
            "manual_label_options": manual_label_options,
            "translate_label": _translate_label,
            "system_status": _read_system_status(config),
            "status": status.snapshot(),
        }

    @app.get("/")
    def index():
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        dashboard = repository.get_dashboard(today, config.web.recent_events_limit)
        chart = _build_chart(dashboard["ten_minute"])
        recent_chart = _build_chart(dashboard["recent_noise"], label_slice=slice(11, 19))
        return render_template(
            "index.html",
            day=today,
            dashboard=dashboard,
            chart=chart,
            recent_chart=recent_chart,
        )

    @app.get("/events/<int:event_id>")
    def event_details(event_id: int):
        event = repository.get_event(event_id)
        if event is None:
            abort(404)
        return render_template("event.html", event=event)

    @app.post("/events/<int:event_id>/label")
    def update_event_label(event_id: int):
        raw_label = request.form.get("user_label")
        normalized_label = raw_label if raw_label in manual_label_values else None
        if raw_label and normalized_label is None:
            abort(400)
        updated = repository.set_event_user_label(event_id, normalized_label)
        if not updated:
            abort(404)
        return redirect(url_for("event_details", event_id=event_id))

    @app.get("/categories/<path:category>")
    def category_events(category: str):
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        events = repository.list_events(category=category, day=today, limit=200)
        if not events:
            abort(404)
        return render_template(
            "category.html",
            category=category,
            category_label=_translate_label(category),
            day=today,
            events=events,
        )

    @app.get("/clips/<int:event_id>")
    def clip_audio(event_id: int):
        event = repository.get_event(event_id)
        if event is None or not event.get("clip_path") or not Path(event["clip_path"]).exists():
            abort(404)
        return send_file(Path(event["clip_path"]))

    @app.get("/spectrograms/<int:event_id>")
    def spectrogram_image(event_id: int):
        event = repository.get_event(event_id)
        if event is None or not event.get("spectrogram_path") or not Path(event["spectrogram_path"]).exists():
            abort(404)
        return send_file(Path(event["spectrogram_path"]))

    @app.get("/health")
    def health():
        snapshot = status.snapshot()
        snapshot["database_path"] = str(config.storage.database_path)
        snapshot["system_status"] = _read_system_status(config)
        return jsonify(snapshot)

    return app


def _build_chart(rows: list[dict[str, object]], label_slice: slice = slice(11, 16)) -> dict[str, object] | None:
    if not rows:
        return None
    labels = [str(row["bucket_start"])[label_slice] for row in rows]
    values = [float(row["avg_dbfs"]) for row in rows]
    min_value = min(values)
    max_value = max(values)
    if max_value - min_value < 1.0:
        max_value += 0.5
        min_value -= 0.5

    return {
        "labels": labels,
        "values": [round(value, 3) for value in values],
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


def _format_dbfs(value: float | int | None, decimals: int = 1) -> str:
    if value is None:
        return "brak danych"
    normalized = float(value)
    threshold = 0.5 * (10 ** (-decimals))
    if abs(normalized) < threshold:
        normalized = 0.0
    return f"{normalized:.{decimals}f} dBFS"


def _describe_classifier_decision(decision: dict[str, object]) -> dict[str, object]:
    details = decision.get("details")
    normalized_details = details if isinstance(details, dict) else {}
    classifier_name = str(decision.get("classifier_name") or "unknown")
    cache_hit = bool(normalized_details.get("cache_hit"))
    external_api_name = normalized_details.get("external_api_name") or normalized_details.get("api_name")
    used_external_api = bool(normalized_details.get("used_external_api") or external_api_name)

    if used_external_api:
        source = "external_api"
        source_label = f"Zewnętrzne API: {external_api_name or 'nieznane'}"
    elif cache_hit:
        source = "cache_reuse"
        source_label = "Ponowne użycie z lokalnego cache"
        if normalized_details.get("manual_feedback_applied"):
            source_label = "Ponowne użycie z lokalnego cache po ręcznej korekcie"
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
        "classifier_label": _translate_classifier_name(classifier_name),
        "classifier_version": str(decision.get("classifier_version") or "-"),
        "source": source,
        "source_label": source_label,
        "used_external_api": used_external_api,
        "external_api_name": str(external_api_name) if external_api_name else None,
        "cache_hit": cache_hit,
        "cache_similarity": normalized_details.get("cache_similarity"),
        "cache_source_event_id": normalized_details.get("cache_source_event_id"),
        "fallback_reason": normalized_details.get("fallback_reason"),
        "birdnet_common_name": (
            str(normalized_details.get("birdnet_common_name")) if normalized_details.get("birdnet_common_name") else None
        ),
        "birdnet_scientific_name": (
            str(normalized_details.get("birdnet_scientific_name"))
            if normalized_details.get("birdnet_scientific_name")
            else None
        ),
        "birdnet_trigger_labels": [
            str(item) for item in normalized_details.get("birdnet_trigger_labels", []) if isinstance(item, str)
        ]
        if isinstance(normalized_details.get("birdnet_trigger_labels"), list)
        else [],
        "resolved_label": str(normalized_details.get("resolved_label")) if normalized_details.get("resolved_label") else None,
        "resolved_label_score": (
            float(normalized_details.get("resolved_label_score"))
            if isinstance(normalized_details.get("resolved_label_score"), int | float)
            else None
        ),
        "cache_category_promoted": bool(normalized_details.get("cache_category_promoted")),
        "top_labels": top_labels,
        "category_scores": category_scores,
    }


def _translate_classifier_name(name: str) -> str:
    return CLASSIFIER_LABELS.get(name, _translate_label(name))


def _translate_label(value: str | None) -> str:
    if not value:
        return "-"
    if value in CATEGORY_LABELS:
        return CATEGORY_LABELS[value]
    if value in YAMNET_LABEL_TRANSLATIONS:
        return YAMNET_LABEL_TRANSLATIONS[value]
    return value


def _load_manual_label_options(class_map_path: Path) -> list[dict[str, str]]:
    values: dict[str, str] = {key: label for key, label in CATEGORY_LABELS.items()}
    try:
        with class_map_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_label = (row.get("display_name") or "").strip()
                if raw_label and raw_label not in values:
                    values[raw_label] = _translate_label(raw_label)
    except OSError:
        pass

    ordered = sorted(values.items(), key=lambda item: item[1].casefold())
    return [{"value": value, "label": label} for value, label in ordered]


def _read_system_status(config: AppConfig) -> dict[str, object]:
    return {
        "cpu_percent": _read_cpu_load_percent(),
        "cpu_temperature_c": _read_cpu_temperature_c(),
        "memory_available_gb": _read_memory_available_gb(),
        "disk_free_gb": _read_disk_free_gb(config.storage.database_path.parent),
    }


def _read_cpu_load_percent() -> float | None:
    try:
        cpu_count = max(os.cpu_count() or 1, 1)
        load_avg, _, _ = os.getloadavg()
    except (AttributeError, OSError):
        return None
    return round((load_avg / cpu_count) * 100.0, 1)


def _read_memory_available_gb() -> float | None:
    meminfo_path = Path("/proc/meminfo")
    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return round(int(parts[1]) / (1024 * 1024), 2)
    except (OSError, ValueError):
        return None
    return None


def _read_cpu_temperature_c() -> float | None:
    thermal_zone_path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        if thermal_zone_path.exists():
            raw_value = thermal_zone_path.read_text(encoding="utf-8").strip()
            return round(float(raw_value) / 1000.0, 1)
    except (OSError, ValueError):
        pass

    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    output = result.stdout.strip()
    if not output.startswith("temp=") or "'" not in output:
        return None

    try:
        numeric_value = output.split("=", 1)[1].split("'", 1)[0]
        return round(float(numeric_value), 1)
    except ValueError:
        return None


def _read_disk_free_gb(path: Path) -> float | None:
    probe_path = path if path.exists() else path.parent
    try:
        usage = shutil.disk_usage(probe_path)
    except OSError:
        return None
    return round(usage.free / (1024 * 1024 * 1024), 2)
