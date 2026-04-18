from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import threading
from functools import lru_cache
from io import BytesIO
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_file, url_for

from app.audio_devices import AudioCaptureError, list_capture_devices
from app.classify.heuristics import CATEGORY_LABELS
from app.config import (
    AggregationConfig,
    AppConfig,
    AudioConfig,
    ClassifierConfig,
    DetectionConfig,
    LoggingConfig,
    StorageConfig,
    WebConfig,
    load_config,
    save_config,
)
from app.pipeline import LiveAudioBuffer, RuntimeStatus
from app.storage.database import SQLiteRepository

CLASSIFIER_LABELS = {
    "yamnet_litert": "Lokalny YAMNet (LiteRT)",
    "heuristic_baseline": "Heurystyczny klasyfikator bazowy",
    "heuristic_fallback": "Heurystyczny tryb awaryjny",
}

WORKER_STATE_LABELS = {
    "idle": "bezczynny",
    "running": "działa",
    "starting": "uruchamianie",
    "degraded": "ograniczony",
    "error": "błąd",
    "stopped": "zatrzymany",
}

PERIOD_LABELS = {
    "day": "Dzień",
    "week": "Tydzień",
    "month": "Miesiąc",
    "year": "Rok",
}

SETTINGS_PRESETS = {
    "balcony_city": {
        "label": "Balkon miejski",
        "description": "Mniej czuły na stały szum uliczny, lepszy do monitoringu zewnętrznego.",
        "detection": {
            "activation_margin_db": 12.0,
            "min_event_dbfs": -43.0,
            "min_active_frames": 3,
            "max_inactive_frames": 2,
        },
        "aggregation": {
            "post_roll_seconds": 0.5,
            "max_event_seconds": 20.0,
            "focus_clip_seconds": 8.0,
        },
    },
    "yard_nature": {
        "label": "Ogród / ptaki",
        "description": "Bardziej czuły na krótkie i delikatniejsze sygnały, dobry do ptaków i przyrody.",
        "detection": {
            "activation_margin_db": 8.0,
            "min_event_dbfs": -50.0,
            "min_active_frames": 2,
            "max_inactive_frames": 3,
        },
        "aggregation": {
            "post_roll_seconds": 1.0,
            "max_event_seconds": 18.0,
            "focus_clip_seconds": 10.0,
        },
    },
    "room": {
        "label": "Pokój / wnętrze",
        "description": "Zrównoważony profil do mowy i typowych dźwięków w pomieszczeniu.",
        "detection": {
            "activation_margin_db": 9.0,
            "min_event_dbfs": -47.0,
            "min_active_frames": 2,
            "max_inactive_frames": 2,
        },
        "aggregation": {
            "post_roll_seconds": 0.5,
            "max_event_seconds": 15.0,
            "focus_clip_seconds": 6.0,
        },
    },
    "custom": {
        "label": "Własne ustawienia",
        "description": "Nie nadpisuje suwaków. Użyj, jeśli chcesz stroić parametry ręcznie.",
    },
}

SETTINGS_SECTIONS = (
    {
        "id": "preset",
        "title": "Preset czułości",
        "fields": (
            {
                "name": "preset",
                "label": "Profil pracy",
                "kind": "radio",
                "description": "Szybki wybór zestawu parametrów pod typowe zastosowanie.",
                "options": [
                    {"value": key, "label": value["label"], "description": value["description"]}
                    for key, value in SETTINGS_PRESETS.items()
                ],
            },
        ),
    },
    {
        "id": "audio",
        "title": "Audio",
        "fields": (
            {
                "name": "audio.arecord_device_mode",
                "label": "Wybór wejścia audio",
                "kind": "radio",
                "description": "Auto wybiera najlepsze wejście capture. Ręczny wybór wymusza konkretne urządzenie.",
                "options": [
                    {"value": "auto", "label": "Automatyczny wybór"},
                    {"value": "manual", "label": "Ręczny wybór"},
                ],
            },
            {
                "name": "audio.arecord_device",
                "label": "Urządzenie capture",
                "kind": "select",
                "description": "Lista urządzeń wykrytych przez `arecord -l`.",
            },
            {
                "name": "audio.sample_rate",
                "label": "Częstotliwość próbkowania",
                "kind": "select",
                "description": "Wyższa wartość daje więcej detali, ale zwiększa obciążenie CPU.",
                "options": [
                    {"value": "8000", "label": "8 kHz"},
                    {"value": "16000", "label": "16 kHz"},
                    {"value": "32000", "label": "32 kHz"},
                    {"value": "48000", "label": "48 kHz"},
                ],
            },
            {
                "name": "audio.channels",
                "label": "Kanały audio",
                "kind": "radio",
                "description": "Mono jest lżejsze i zwykle wystarcza do monitoringu.",
                "options": [
                    {"value": "1", "label": "Mono"},
                    {"value": "2", "label": "Stereo"},
                ],
            },
            {
                "name": "audio.level_display_mode",
                "label": "Skala prezentacji",
                "kind": "radio",
                "description": "Wybierz, czy pokazywać surowe dBFS z urządzenia, czy szacowane dBA po kalibracji.",
                "options": [
                    {"value": "raw", "label": "Surowe dBFS"},
                    {"value": "calibrated", "label": "Szacowane dBA"},
                ],
            },
            {
                "name": "audio.calibration_slope",
                "label": "Współczynnik kalibracji",
                "kind": "range",
                "description": "Skala liniowa od dBFS do dBA. Na podstawie Twoich punktów startowych jest w okolicy 2.9.",
                "min": 0.5,
                "max": 6.0,
                "step": 0.01,
                "unit": "",
            },
            {
                "name": "audio.calibration_offset_db",
                "label": "Przesunięcie kalibracji",
                "kind": "range",
                "description": "Stały offset używany razem ze współczynnikiem kalibracji.",
                "min": 0.0,
                "max": 200.0,
                "step": 0.1,
                "unit": "dB",
            },
            {
                "name": "audio.frame_duration_seconds",
                "label": "Długość ramki analizy",
                "kind": "range",
                "description": "Krótsze ramki szybciej reagują, dłuższe są stabilniejsze na tle.",
                "min": 0.25,
                "max": 1.0,
                "step": 0.05,
                "unit": "s",
            },
            {
                "name": "audio.retry_backoff_seconds",
                "label": "Czas ponownej próby po błędzie wejścia",
                "kind": "range",
                "description": "Ile sekund czekać przed następną próbą otwarcia urządzenia audio.",
                "min": 1.0,
                "max": 30.0,
                "step": 1.0,
                "unit": "s",
            },
        ),
    },
    {
        "id": "detection",
        "title": "Detekcja zdarzeń",
        "fields": (
            {
                "name": "detection.initial_noise_floor_dbfs",
                "label": "Początkowy poziom tła",
                "kind": "range",
                "description": "Niższa wartość oznacza, że system startuje z założeniem cichszego otoczenia.",
                "min": -80.0,
                "max": -30.0,
                "step": 1.0,
                "unit": "dBFS",
            },
            {
                "name": "detection.activation_margin_db",
                "label": "Próg aktywacji ponad tło",
                "kind": "range",
                "description": "Im wyżej, tym trudniej wywołać zdarzenie samym szumem tła.",
                "min": 4.0,
                "max": 20.0,
                "step": 0.5,
                "unit": "dB",
            },
            {
                "name": "detection.release_margin_db",
                "label": "Próg podtrzymania zdarzenia",
                "kind": "range",
                "description": "Niższa wartość wydłuża zdarzenia, wyższa szybciej je kończy.",
                "min": 1.0,
                "max": 12.0,
                "step": 0.5,
                "unit": "dB",
            },
            {
                "name": "detection.min_event_dbfs",
                "label": "Minimalna głośność zdarzenia",
                "kind": "range",
                "description": "Dolny próg głośności. Pomaga odsiać bardzo ciche tło.",
                "min": -70.0,
                "max": -20.0,
                "step": 1.0,
                "unit": "dBFS",
            },
            {
                "name": "detection.min_active_frames",
                "label": "Minimalna liczba aktywnych ramek",
                "kind": "range",
                "description": "Więcej ramek zmniejsza fałszywe alarmy, ale opóźnia start zdarzenia.",
                "min": 1,
                "max": 8,
                "step": 1,
                "unit": "ramek",
            },
            {
                "name": "detection.max_inactive_frames",
                "label": "Dopuszczalna liczba cichych ramek w środku zdarzenia",
                "kind": "range",
                "description": "Wyższa wartość bardziej skleja przerywane dźwięki w jeden event.",
                "min": 1,
                "max": 8,
                "step": 1,
                "unit": "ramek",
            },
            {
                "name": "detection.noise_floor_alpha",
                "label": "Szybkość uczenia poziomu tła",
                "kind": "range",
                "description": "Wyżej = szybsza adaptacja do zmian tła. Niżej = większa stabilność.",
                "min": 0.01,
                "max": 0.20,
                "step": 0.01,
                "unit": "",
            },
        ),
    },
    {
        "id": "aggregation",
        "title": "Łączenie i cięcie zdarzeń",
        "fields": (
            {
                "name": "aggregation.noise_interval_seconds",
                "label": "Okno statystyk hałasu",
                "kind": "range",
                "description": "Jak długie interwały trafiają do statystyk wykresów.",
                "min": 1.0,
                "max": 30.0,
                "step": 1.0,
                "unit": "s",
            },
            {
                "name": "aggregation.pre_roll_seconds",
                "label": "Pre-roll",
                "kind": "range",
                "description": "Ile sekund dodać przed wykrytym początkiem zdarzenia.",
                "min": 0.0,
                "max": 5.0,
                "step": 0.5,
                "unit": "s",
            },
            {
                "name": "aggregation.post_roll_seconds",
                "label": "Post-roll",
                "kind": "range",
                "description": "Ile ciszy jeszcze trzymać w tym samym zdarzeniu po ustaniu sygnału.",
                "min": 0.0,
                "max": 5.0,
                "step": 0.5,
                "unit": "s",
            },
            {
                "name": "aggregation.min_event_seconds",
                "label": "Minimalny czas zdarzenia",
                "kind": "range",
                "description": "Krótsze epizody są odrzucane jako zbyt krótkie.",
                "min": 0.5,
                "max": 10.0,
                "step": 0.5,
                "unit": "s",
            },
            {
                "name": "aggregation.focus_clip_seconds",
                "label": "Długość próbki do odsłuchu i klasyfikacji",
                "kind": "range",
                "description": "Jak długi wycinek zapisać wokół najmocniejszego fragmentu zdarzenia.",
                "min": 2.0,
                "max": 20.0,
                "step": 1.0,
                "unit": "s",
            },
            {
                "name": "aggregation.max_clip_seconds",
                "label": "Maksymalna długość pełnego klipu roboczego",
                "kind": "range",
                "description": "Twardy limit dla bufora audio przed wycięciem focus clipu.",
                "min": 5.0,
                "max": 60.0,
                "step": 1.0,
                "unit": "s",
            },
            {
                "name": "aggregation.max_event_seconds",
                "label": "Maksymalny czas pojedynczego zdarzenia",
                "kind": "range",
                "description": "Chroni przed ciągnięciem jednego eventu przez długie tło lub ciągły hałas.",
                "min": 5.0,
                "max": 120.0,
                "step": 1.0,
                "unit": "s",
            },
        ),
    },
    {
        "id": "classifier",
        "title": "Klasyfikacja",
        "fields": (
            {
                "name": "classifier.backend",
                "label": "Backend klasyfikacji",
                "kind": "radio",
                "description": "YAMNet daje pełną klasyfikację. Heurystyka to lżejszy tryb awaryjny.",
                "options": [
                    {"value": "yamnet", "label": "YAMNet"},
                    {"value": "heuristic", "label": "Heurystyka"},
                ],
            },
            {
                "name": "classifier.yamnet_num_threads",
                "label": "Liczba wątków YAMNet",
                "kind": "range",
                "description": "Więcej wątków może przyspieszyć analizę kosztem CPU.",
                "min": 1,
                "max": 4,
                "step": 1,
                "unit": "wątki",
            },
            {
                "name": "classifier.yamnet_max_analysis_seconds",
                "label": "Maksymalny czas próbki dla YAMNet",
                "kind": "range",
                "description": "Dłuższy fragment może dać lepszą klasyfikację, ale kosztuje więcej CPU.",
                "min": 2.0,
                "max": 30.0,
                "step": 1.0,
                "unit": "s",
            },
            {
                "name": "classifier.yamnet_max_windows",
                "label": "Maksymalna liczba okien analizy",
                "kind": "range",
                "description": "Ogranicza koszt obliczeń dla długich próbek.",
                "min": 4,
                "max": 64,
                "step": 1,
                "unit": "okien",
            },
            {
                "name": "classifier.yamnet_min_category_score",
                "label": "Minimalny score kategorii YAMNet",
                "kind": "range",
                "description": "Wyższa wartość daje mniej, ale pewniejszych klasyfikacji.",
                "min": 0.01,
                "max": 0.50,
                "step": 0.01,
                "unit": "",
            },
            {
                "name": "classifier.min_persist_confidence",
                "label": "Minimalna pewność zapisu eventu",
                "kind": "range",
                "description": "Eventy z niższą pewnością po klasyfikacji są ignorowane i nie trafiają do bazy.",
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "unit": "",
            },
            {
                "name": "classifier.yamnet_top_k",
                "label": "Liczba top etykiet YAMNet",
                "kind": "range",
                "description": "Ile najwyżej ocenionych klas pokazać w śladzie klasyfikacji.",
                "min": 3,
                "max": 15,
                "step": 1,
                "unit": "etykiet",
            },
        ),
    },
    {
        "id": "storage",
        "title": "Przechowywanie",
        "fields": (
            {
                "name": "storage.keep_clips",
                "label": "Zapisuj próbki audio",
                "kind": "radio",
                "description": "Wyłączenie zostawia zdarzenia i statystyki, ale bez plików WAV i widma.",
                "options": [
                    {"value": "true", "label": "Tak"},
                    {"value": "false", "label": "Nie"},
                ],
            },
            {
                "name": "storage.clip_max_megabytes",
                "label": "Maksymalny rozmiar katalogu z klipami",
                "kind": "range",
                "description": "Po przekroczeniu limitu stare próbki są usuwane przez retencję.",
                "min": 64,
                "max": 4096,
                "step": 64,
                "unit": "MB",
            },
            {
                "name": "storage.clip_max_age_days",
                "label": "Maksymalny wiek próbki",
                "kind": "range",
                "description": "Po ilu dniach stare próbki mogą zostać usunięte.",
                "min": 1,
                "max": 90,
                "step": 1,
                "unit": "dni",
            },
            {
                "name": "storage.min_free_disk_megabytes",
                "label": "Minimalna wolna przestrzeń",
                "kind": "range",
                "description": "Rezerwa wolnego miejsca na dysku utrzymywana przez retencję.",
                "min": 64,
                "max": 4096,
                "step": 64,
                "unit": "MB",
            },
            {
                "name": "storage.database_path",
                "label": "Ścieżka bazy danych",
                "kind": "text",
                "description": "Zmiana wymaga restartu usługi i migracji danych, jeśli baza ma zostać przeniesiona.",
            },
            {
                "name": "storage.clip_dir",
                "label": "Katalog próbek audio",
                "kind": "text",
                "description": "Gdzie zapisywać WAV i obrazy widma.",
            },
        ),
    },
    {
        "id": "web",
        "title": "Panel WWW i logi",
        "fields": (
            {
                "name": "web.host",
                "label": "Adres nasłuchu panelu",
                "kind": "select",
                "description": "0.0.0.0 wystawia panel w sieci, 127.0.0.1 tylko lokalnie.",
                "options": [
                    {"value": "0.0.0.0", "label": "0.0.0.0 (cała sieć)"},
                    {"value": "127.0.0.1", "label": "127.0.0.1 (tylko lokalnie)"},
                ],
            },
            {
                "name": "web.port",
                "label": "Port panelu WWW",
                "kind": "number",
                "description": "Port nasłuchu interfejsu webowego.",
                "min": 1024,
                "max": 65535,
                "step": 1,
            },
            {
                "name": "web.recent_events_limit",
                "label": "Liczba ostatnich zdarzeń na panelu",
                "kind": "range",
                "description": "Ile najnowszych eventów pokazywać na stronie głównej.",
                "min": 5,
                "max": 200,
                "step": 5,
                "unit": "zdarzeń",
            },
            {
                "name": "web.dashboard_history_hours",
                "label": "Zakres historii panelu",
                "kind": "range",
                "description": "Ile godzin historii przeznaczyć dla widoków panelu.",
                "min": 1,
                "max": 168,
                "step": 1,
                "unit": "h",
            },
            {
                "name": "logging.level",
                "label": "Poziom logowania",
                "kind": "select",
                "description": "Ile szczegółów zapisywać do logów aplikacji.",
                "options": [
                    {"value": "DEBUG", "label": "DEBUG"},
                    {"value": "INFO", "label": "INFO"},
                    {"value": "WARNING", "label": "WARNING"},
                    {"value": "ERROR", "label": "ERROR"},
                ],
            },
        ),
    },
)

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
    "Fowl": "Drób",
    "Chicken, rooster": "Kura, kogut",
    "Cluck": "Gdaczenie",
    "Crowing, cock-a-doodle-doo": "Kukuryku",
    "Turkey": "Indyk",
    "Gobble": "Gulgotanie",
    "Duck": "Kaczka",
    "Quack": "Kwa-kwa",
    "Goose": "Gęś",
    "Honk": "Trąbienie gęsi",
    "Pigeon, dove": "Gołąb, gołębica",
    "Coo": "Grukanie",
    "Crow": "Wrona",
    "Caw": "Kraknięcie",
    "Owl": "Sowa",
    "Hoot": "Puhanie",
    "Dog": "Pies",
    "Whimper (dog)": "Skowyt psa",
    "Howl": "Wycie",
    "Growling": "Warczenie",
    "Cat": "Kot",
    "Caterwaul": "Koci wrzask",
    "Mouse": "Mysz",
    "Pig": "Świnia",
    "Oink": "Kwik",
    "Goat": "Koza",
    "Bleat": "Beczenie",
    "Sheep": "Owca",
    "Horse": "Koń",
    "Moo": "Muczenie",
    "Cattle, bovinae": "Bydło",
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


def create_app(
    repository: SQLiteRepository,
    status: RuntimeStatus,
    config: AppConfig,
    live_audio_buffer: LiveAudioBuffer | None = None,
) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    config_state = {"value": config}

    def current_config() -> AppConfig:
        return config_state["value"]

    def current_manual_label_options() -> list[dict[str, str]]:
        class_map_path = current_config().classifier.yamnet_class_map_path
        cache_key = _manual_label_options_cache_key(class_map_path)
        return [
            {"value": value, "label": label}
            for value, label in _manual_label_options_for_path(*cache_key)
        ]

    def current_manual_label_values() -> set[str]:
        class_map_path = current_config().classifier.yamnet_class_map_path
        cache_key = _manual_label_options_cache_key(class_map_path)
        return {value for value, _label in _manual_label_options_for_path(*cache_key)}

    @app.context_processor
    def inject_helpers() -> dict[str, object]:
        cfg = current_config()
        return {
            "category_labels": CATEGORY_LABELS,
            "describe_classifier_decision": _describe_classifier_decision,
            "format_audio_level": lambda value, decimals=1: _format_audio_level(
                value,
                cfg.audio,
                decimals=decimals,
            ),
            "format_dbfs": _format_dbfs,
            "audio_level_unit": _audio_level_unit(cfg.audio),
            "format_local_timestamp": _format_local_timestamp,
            "format_uptime_seconds": _format_uptime_seconds,
            "format_worker_state": _format_worker_state,
            "live_audio_available": live_audio_buffer is not None,
            "manual_label_options": current_manual_label_options(),
            "nav_links": [
                {"href": url_for("index"), "label": "Panel"},
                {"href": url_for("sleep_health"), "label": "Zdrowie"},
                {"href": url_for("settings"), "label": "Ustawienia"},
                {"action": "live", "label": "Live"},
            ],
            "settings_sections": SETTINGS_SECTIONS,
            "settings_presets": SETTINGS_PRESETS,
            "translate_label": _translate_label,
            "system_status": _read_system_status(cfg),
            "status": status.snapshot(),
        }

    @app.get("/")
    def index():
        cfg = current_config()
        range_state = _resolve_range_state(request.args)
        dashboard = repository.get_dashboard_range(
            started_at=range_state["started_at"],
            ended_at=range_state["ended_at"],
            recent_limit=20,
            bucket_mode=range_state["bucket_mode"],
        )
        recent_events = repository.list_events_range(category=None, started_at=None, ended_at=None, limit=20)
        chart = _build_chart(
            dashboard["ten_minute"],
            period=range_state["period"],
        )
        return render_template(
            "index.html",
            range_state=range_state,
            dashboard=dashboard,
            recent_events=recent_events,
            range_events_url=url_for("range_events_api", period=range_state["period"], date=range_state["date"]),
            chart=chart,
            live_audio_available=live_audio_buffer is not None,
            audio_level_mode=cfg.audio.level_display_mode,
            audio_calibration_slope=cfg.audio.calibration_slope,
            audio_calibration_offset_db=cfg.audio.calibration_offset_db,
        )

    @app.get("/api/events")
    def range_events_api():
        range_state = _resolve_range_state(request.args)
        events = repository.list_events_range(
            category=None,
            started_at=range_state["started_at"],
            ended_at=range_state["ended_at"],
            limit=None,
        )
        return jsonify(
            {
                "events": [
                    {
                        "id": row["id"],
                        "started_at": row["started_at"],
                        "category": row["category"],
                        "category_label": _translate_label(str(row["category"])),
                        "duration_label": f"{float(row['duration_seconds']):.1f} s",
                        "peak_label": _format_audio_level(row["peak_dbfs"], current_config().audio),
                        "confidence_label": f"{float(row['confidence']):.2f}",
                        "event_url": url_for("event_details", event_id=row["id"]),
                        "category_url": url_for(
                            "category_events",
                            category=row["category"],
                            period=range_state["period"],
                            date=range_state["date"],
                        ),
                    }
                    for row in events
                ]
            }
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
        normalized_label = raw_label if raw_label in current_manual_label_values() else None
        if raw_label and normalized_label is None:
            abort(400)
        updated = repository.set_event_user_label(event_id, normalized_label)
        if not updated:
            abort(404)
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
            return jsonify(
                {
                    "message": "Etykieta zapisana.",
                    "event_id": event_id,
                    "user_label": normalized_label,
                    "user_label_label": _translate_label(normalized_label) if normalized_label else None,
                }
            )
        return redirect(url_for("event_details", event_id=event_id))

    @app.get("/categories/<path:category>")
    def category_events(category: str):
        range_state = _resolve_range_state(request.args)
        events = repository.list_events_range(
            category=category,
            started_at=range_state["started_at"],
            ended_at=range_state["ended_at"],
            limit=200,
        )
        if not events:
            abort(404)
        return render_template(
            "category.html",
            category=category,
            category_label=_translate_label(category),
            range_state=range_state,
            events=events,
        )

    @app.get("/zdrowie")
    def sleep_health():
        range_state = _resolve_range_state(request.args)
        events = repository.list_events_range(
            category=None,
            started_at=range_state["started_at"],
            ended_at=range_state["ended_at"],
            limit=500,
        )
        health = _build_sleep_health(events, range_state)
        return render_template(
            "health.html",
            range_state=range_state,
            health=health,
        )

    @app.get("/api/live-audio")
    def live_audio():
        if live_audio_buffer is None:
            abort(404)
        try:
            seconds = float(request.args.get("seconds", 12.0))
        except ValueError:
            seconds = 12.0
        seconds = max(3.0, min(seconds, 30.0))
        wav_bytes = live_audio_buffer.snapshot_wav_bytes(seconds)
        if not wav_bytes:
            abort(404)
        return send_file(
            BytesIO(wav_bytes),
            mimetype="audio/wav",
            as_attachment=False,
            download_name="live-audio.wav",
            max_age=0,
        )

    @app.get("/api/live-level")
    def live_level():
        if live_audio_buffer is None:
            abort(404)
        stats = live_audio_buffer.current_level_stats_dbfs(5.0)
        level_dbfs = stats["level_dbfs"]
        min_dbfs = stats["min_dbfs"]
        max_dbfs = stats["max_dbfs"]
        snapshot = status.snapshot()
        if level_dbfs is None:
            level_dbfs = snapshot.get("last_frame_dbfs")
        if min_dbfs is None:
            min_dbfs = snapshot.get("last_frame_min_dbfs", level_dbfs)
        if max_dbfs is None:
            max_dbfs = snapshot.get("last_frame_max_dbfs", level_dbfs)
        cfg = current_config()
        return jsonify(
            {
                "window_seconds": 5.0,
                "raw_level_dbfs": level_dbfs,
                "raw_min_dbfs": min_dbfs,
                "raw_max_dbfs": max_dbfs,
                "raw_level_label": _format_dbfs(level_dbfs),
                "raw_min_label": _format_dbfs(min_dbfs),
                "raw_max_label": _format_dbfs(max_dbfs),
                "display_level_label": _format_audio_level(level_dbfs, cfg.audio),
                "display_min_label": _format_audio_level(min_dbfs, cfg.audio),
                "display_max_label": _format_audio_level(max_dbfs, cfg.audio),
                "updated_at": datetime.now().astimezone().isoformat(),
            }
        )

    @app.get("/api/live-audio-stream")
    def live_audio_stream():
        if live_audio_buffer is None:
            abort(404)
        try:
            seconds = float(request.args.get("seconds", 12.0))
        except ValueError:
            seconds = 12.0
        seconds = max(3.0, min(seconds, 30.0))

        def generate():
            yield from live_audio_buffer.stream_wav_bytes(seconds)

        return Response(
            generate(),
            mimetype="audio/wav",
            direct_passthrough=True,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/settings", methods=["GET", "POST"])
    def settings():
        cfg = current_config()
        message = None
        if request.method == "POST":
            updated_config = _build_config_from_form(cfg, request.form)
            saved_path = save_config(updated_config)
            config_state["value"] = load_config(saved_path)
            cfg = current_config()
            message = (
                f"Ustawienia zapisane w {saved_path}. Część zmian zacznie działać po restarcie usługi audio-monitor."
            )
            if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
                return jsonify({"message": message, "saved_path": str(saved_path)})
        return render_template(
            "settings.html",
            config=cfg,
            message=message,
            settings_values=_settings_values(cfg),
            capture_device_options=_capture_device_options(cfg),
        )

    @app.post("/settings/restart")
    def settings_restart():
        _schedule_audio_monitor_restart()
        message = "Restart aplikacji audio-monitor został zaplanowany."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
            return jsonify({"message": message, "service": "audio-monitor"})
        return render_template(
            "settings.html",
            config=current_config(),
            message=message,
            settings_values=_settings_values(current_config()),
            capture_device_options=_capture_device_options(current_config()),
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
        cfg = current_config()
        snapshot["database_path"] = str(cfg.storage.database_path)
        system_status = _read_system_status(cfg)
        snapshot["system_status"] = system_status
        snapshot["system_uptime_human"] = _format_uptime_seconds(system_status.get("uptime_seconds"))
        return jsonify(snapshot)

    return app


def _build_chart(
    rows: list[dict[str, object]],
    *,
    period: str,
) -> dict[str, object] | None:
    if not rows:
        return None
    labels = [_format_chart_bucket_label(str(row["bucket_start"]), period) for row in rows]
    values = [float(row["avg_dbfs"]) for row in rows]
    counts = [int(row.get("event_count", 0) or 0) for row in rows]
    min_value = min(values)
    max_value = max(values)
    if max_value - min_value < 1.0:
        max_value += 0.5
        min_value -= 0.5

    return {
        "labels": labels,
        "values": [round(value, 3) for value in values],
        "counts": counts,
        "min_value": round(min_value, 1),
        "max_value": round(max_value, 1),
    }


def _build_sleep_health(events: list[dict[str, object]], range_state: dict[str, object]) -> dict[str, object]:
    parsed: list[dict[str, object]] = []
    for row in events:
        summary_raw = row.get("summary_json")
        if not summary_raw:
            continue
        try:
            summary = json.loads(str(summary_raw))
        except json.JSONDecodeError:
            continue
        score, reasons = _score_snore_candidate(summary)
        parsed.append(
            {
                "id": int(row["id"]),
                "started_at": str(row["started_at"]),
                "ended_at": str(row["ended_at"]),
                "duration_seconds": float(row["duration_seconds"]),
                "category": str(row["category"]),
                "peak_dbfs": float(row["peak_dbfs"]),
                "confidence": float(row["confidence"]),
                "summary": summary,
                "snore_score": score,
                "snore_reasons": reasons,
                "event_url": url_for("event_details", event_id=row["id"]),
            }
        )

    candidates = [row for row in parsed if row["snore_score"] >= 0.55]
    candidates.sort(key=lambda item: (float(item["snore_score"]), str(item["started_at"])))
    candidates.reverse()

    span_seconds = _range_seconds(range_state["started_at"], range_state["ended_at"])
    span_hours = max(span_seconds / 3600.0, 1e-6)
    candidate_durations = [float(item["duration_seconds"]) for item in candidates]
    candidate_starts = [datetime.fromisoformat(str(item["started_at"])) for item in sorted(candidates, key=lambda item: str(item["started_at"]))]
    gaps = [
        (later - earlier).total_seconds()
        for earlier, later in zip(candidate_starts, candidate_starts[1:], strict=False)
    ]
    if not gaps and len(candidate_starts) == 1:
        gaps = []

    snore_index = len(candidates) / span_hours
    snore_burden_seconds = sum(candidate_durations)
    burden_percent = (snore_burden_seconds / span_seconds * 100.0) if span_seconds else 0.0
    mean_score = sum(float(item["snore_score"]) for item in candidates) / len(candidates) if candidates else 0.0
    mean_peak = sum(float(item["peak_dbfs"]) for item in candidates) / len(candidates) if candidates else None
    mean_duration = sum(candidate_durations) / len(candidate_durations) if candidate_durations else None
    mean_gap = sum(gaps) / len(gaps) if gaps else None
    max_gap = max(gaps) if gaps else None
    long_pause_count = sum(1 for gap in gaps if gap >= 15.0)
    gap_cv = (float(np.std(np.array(gaps, dtype=np.float32)) / np.mean(np.array(gaps, dtype=np.float32))) if len(gaps) >= 2 and np.mean(np.array(gaps, dtype=np.float32)) else None)
    apnea_risk = _sleep_risk_score(snore_index, long_pause_count, max_gap, gap_cv, mean_score)
    snore_gauge_width = min(max(snore_index * 4.0, 0.0), 100.0)
    apnea_risk_width = min(max(apnea_risk, 0.0), 100.0)

    return {
        "summary": {
            "event_count": len(events),
            "candidate_count": len(candidates),
            "snore_burden_seconds": snore_burden_seconds,
            "burden_percent": burden_percent,
            "snore_index_per_hour": snore_index,
            "mean_score": mean_score,
            "mean_peak_dbfs": mean_peak,
            "mean_duration_seconds": mean_duration,
            "mean_gap_seconds": mean_gap,
            "max_gap_seconds": max_gap,
            "gap_cv": gap_cv,
            "long_pause_count": long_pause_count,
            "apnea_risk_score": apnea_risk,
            "snore_gauge_width": snore_gauge_width,
            "apnea_risk_width": apnea_risk_width,
            "apnea_risk_label": _sleep_risk_label(apnea_risk, len(candidates)),
        },
        "candidates": candidates[:40],
    }


def _resolve_range_state(args) -> dict[str, object]:
    today = datetime.now().astimezone().date()
    period = args.get("period", "day")
    if period not in PERIOD_LABELS:
        period = "day"

    raw_date = args.get("date")
    try:
        anchor = date.fromisoformat(raw_date) if raw_date else today
    except ValueError:
        anchor = today

    range_start, range_end, nav_date, title = _period_bounds(period, anchor)
    previous_date = _shift_anchor(nav_date, period, -1)
    next_date = _shift_anchor(nav_date, period, 1)
    bucket_mode, label_slice = _bucket_mode_for_period(period)

    return {
        "period": period,
        "period_label": PERIOD_LABELS[period],
        "date": nav_date.isoformat(),
        "started_at": datetime.combine(range_start, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S"),
        "ended_at": datetime.combine(range_end, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S"),
        "title": title,
        "previous_date": previous_date.isoformat(),
        "next_date": next_date.isoformat(),
        "bucket_mode": bucket_mode,
    }


def _period_bounds(period: str, anchor: date) -> tuple[date, date, date, str]:
    if period == "week":
        start = anchor - timedelta(days=anchor.weekday())
        end = start + timedelta(days=7)
        title = f"{start:%Y-%m-%d} - {(end - timedelta(days=1)):%Y-%m-%d}"
        return start, end, start, title
    if period == "month":
        start = anchor.replace(day=1)
        next_month = _add_months(start, 1)
        title = start.strftime("%Y-%m")
        return start, next_month, start, title
    if period == "year":
        start = anchor.replace(month=1, day=1)
        end = start.replace(year=start.year + 1)
        title = start.strftime("%Y")
        return start, end, start, title
    start = anchor
    end = anchor + timedelta(days=1)
    return start, end, anchor, anchor.strftime("%Y-%m-%d")


def _shift_anchor(anchor: date, period: str, delta: int) -> date:
    if period == "week":
        return anchor + timedelta(days=7 * delta)
    if period == "month":
        return _add_months(anchor, delta)
    if period == "year":
        return anchor.replace(year=anchor.year + delta)
    return anchor + timedelta(days=delta)


def _add_months(value: date, months: int) -> date:
    total_month = value.month - 1 + months
    year = value.year + total_month // 12
    month = total_month % 12 + 1
    return date(year, month, 1)


def _bucket_mode_for_period(period: str) -> tuple[str, slice]:
    if period == "year":
        return "month", slice(0, 7)
    if period == "week":
        return "hour", slice(5, 13)
    if period == "month":
        return "six_hour", slice(5, 13)
    return "ten_minute", slice(11, 16)


def _format_chart_bucket_label(bucket_start: str, period: str) -> str:
    try:
        timestamp = datetime.fromisoformat(bucket_start).astimezone()
    except ValueError:
        return bucket_start

    if period == "year":
        return timestamp.strftime("%Y-%m")
    if period in {"week", "month"}:
        return timestamp.strftime("%Y-%m-%d %H:%M")
    return timestamp.strftime("%H:%M")


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


def _format_uptime_seconds(value: float | int | None) -> str:
    if value is None:
        return "brak"
    total_seconds = max(int(float(value)), 0)
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, _seconds = divmod(remainder, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if days or hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def _format_dbfs(value: float | int | None, decimals: int = 1) -> str:
    if value is None:
        return "brak danych"
    normalized = float(value)
    threshold = 0.5 * (10 ** (-decimals))
    if abs(normalized) < threshold:
        normalized = 0.0
    return f"{normalized:.{decimals}f} dBFS"


def _audio_level_unit(audio: AudioConfig) -> str:
    return "dBA" if audio.level_display_mode == "calibrated" else "dBFS"


def _calibrate_audio_level(value: float | int | None, audio: AudioConfig) -> float | None:
    if value is None:
        return None
    dbfs_value = float(value)
    if audio.level_display_mode != "calibrated":
        return dbfs_value
    return dbfs_value * float(audio.calibration_slope) + float(audio.calibration_offset_db)


def _format_audio_level(value: float | int | None, audio: AudioConfig, decimals: int = 1) -> str:
    calibrated = _calibrate_audio_level(value, audio)
    if calibrated is None:
        return "brak danych"
    threshold = 0.5 * (10 ** (-decimals))
    if abs(calibrated) < threshold:
        calibrated = 0.0
    return f"{calibrated:.{decimals}f} {_audio_level_unit(audio)}"


def _range_seconds(started_at: str, ended_at: str) -> float:
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(ended_at)
    except ValueError:
        return 0.0
    return max((end - start).total_seconds(), 0.0)


def _score_snore_candidate(summary: dict[str, object]) -> tuple[float, list[str]]:
    duration_seconds = float(summary.get("duration_seconds") or 0.0)
    peak_dbfs = float(summary.get("peak_dbfs") or -120.0)
    mean_dbfs = float(summary.get("mean_dbfs") or -120.0)
    dominant_freq_hz = float(summary.get("dominant_freq_hz") or 0.0)
    low_band_ratio = float(summary.get("low_band_ratio") or 0.0)
    mid_band_ratio = float(summary.get("mid_band_ratio") or 0.0)
    mean_flux = float(summary.get("mean_flux") or 0.0)
    mean_flatness = float(summary.get("mean_flatness") or 1.0)
    rms_modulation_depth = float(summary.get("rms_modulation_depth") or 0.0)
    dominant_modulation_hz = float(summary.get("dominant_modulation_hz") or 0.0)

    score = 0.0
    reasons: list[str] = []

    if 0.35 <= duration_seconds <= 8.0:
        score += 0.20
        reasons.append("krótki epizod")
    if 90.0 <= dominant_freq_hz <= 1400.0:
        score += 0.18
        reasons.append("zakres częstotliwości")
    if low_band_ratio + mid_band_ratio >= 0.55:
        score += 0.22
        reasons.append("nisko-średnie pasmo")
    if mean_flatness <= 0.55:
        score += 0.16
        reasons.append("tonalny charakter")
    if -55.0 <= mean_dbfs <= -15.0:
        score += 0.10
        reasons.append("typowa głośność")
    if 0.01 <= mean_flux <= 0.65:
        score += 0.06
    if rms_modulation_depth >= 0.08:
        score += 0.05
    if dominant_modulation_hz >= 0.6:
        score += 0.03
    if peak_dbfs >= -50.0:
        score += 0.02

    return min(score, 1.0), reasons


def _sleep_risk_score(
    snore_index_per_hour: float,
    long_pause_count: int,
    max_gap_seconds: float | None,
    gap_cv: float | None,
    mean_score: float,
) -> float:
    score = 0.0
    score += min(snore_index_per_hour / 25.0, 1.0) * 35.0
    score += min(long_pause_count / 6.0, 1.0) * 25.0
    if max_gap_seconds is not None:
        score += min(max_gap_seconds / 45.0, 1.0) * 20.0
    if gap_cv is not None:
        score += min(gap_cv / 1.2, 1.0) * 10.0
    score += min(mean_score, 1.0) * 10.0
    return max(0.0, min(score, 100.0))


def _sleep_risk_label(score: float, candidate_count: int) -> str:
    if candidate_count < 4:
        return "Za mało danych"
    if score < 30.0:
        return "Niskie ryzyko"
    if score < 55.0:
        return "Obserwuj"
    if score < 75.0:
        return "Podwyższone ryzyko"
    return "Wysokie ryzyko"


def _capture_device_options(config: AppConfig) -> list[dict[str, str]]:
    options = [{"value": "", "label": "Automatyczny wybór"}]
    try:
        devices = list_capture_devices(config.audio.arecord_binary)
    except AudioCaptureError:
        devices = []
    for device in devices:
        options.append(
            {
                "value": device.device_spec,
                "label": f"{device.device_spec} - {device.source_name}",
            }
        )
    if config.audio.arecord_device and config.audio.arecord_device not in {item["value"] for item in options}:
        options.append({"value": config.audio.arecord_device, "label": f"{config.audio.arecord_device} - ręcznie zapisane"})
    return options


def _settings_values(config: AppConfig) -> dict[str, object]:
    return {
        "preset": _detect_matching_preset(config),
        "audio.arecord_device_mode": "manual" if config.audio.arecord_device else "auto",
        "audio.arecord_device": config.audio.arecord_device or "",
        "audio.sample_rate": config.audio.sample_rate,
        "audio.channels": config.audio.channels,
        "audio.level_display_mode": config.audio.level_display_mode,
        "audio.calibration_slope": config.audio.calibration_slope,
        "audio.calibration_offset_db": config.audio.calibration_offset_db,
        "audio.frame_duration_seconds": config.audio.frame_duration_seconds,
        "audio.retry_backoff_seconds": config.audio.retry_backoff_seconds,
        "detection.initial_noise_floor_dbfs": config.detection.initial_noise_floor_dbfs,
        "detection.activation_margin_db": config.detection.activation_margin_db,
        "detection.release_margin_db": config.detection.release_margin_db,
        "detection.min_event_dbfs": config.detection.min_event_dbfs,
        "detection.min_active_frames": config.detection.min_active_frames,
        "detection.max_inactive_frames": config.detection.max_inactive_frames,
        "detection.noise_floor_alpha": config.detection.noise_floor_alpha,
        "aggregation.noise_interval_seconds": config.aggregation.noise_interval_seconds,
        "aggregation.pre_roll_seconds": config.aggregation.pre_roll_seconds,
        "aggregation.post_roll_seconds": config.aggregation.post_roll_seconds,
        "aggregation.min_event_seconds": config.aggregation.min_event_seconds,
        "aggregation.focus_clip_seconds": config.aggregation.focus_clip_seconds,
        "aggregation.max_clip_seconds": config.aggregation.max_clip_seconds,
        "aggregation.max_event_seconds": config.aggregation.max_event_seconds,
        "classifier.backend": config.classifier.backend,
        "classifier.yamnet_num_threads": config.classifier.yamnet_num_threads,
        "classifier.yamnet_max_analysis_seconds": config.classifier.yamnet_max_analysis_seconds,
        "classifier.yamnet_max_windows": config.classifier.yamnet_max_windows,
        "classifier.yamnet_min_category_score": config.classifier.yamnet_min_category_score,
        "classifier.min_persist_confidence": config.classifier.min_persist_confidence,
        "classifier.yamnet_top_k": config.classifier.yamnet_top_k,
        "storage.keep_clips": "true" if config.storage.keep_clips else "false",
        "storage.clip_max_megabytes": config.storage.clip_max_megabytes,
        "storage.clip_max_age_days": config.storage.clip_max_age_days,
        "storage.min_free_disk_megabytes": config.storage.min_free_disk_megabytes,
        "storage.database_path": _display_path(config.base_dir, config.storage.database_path),
        "storage.clip_dir": _display_path(config.base_dir, config.storage.clip_dir),
        "web.host": config.web.host,
        "web.port": config.web.port,
        "web.recent_events_limit": config.web.recent_events_limit,
        "web.dashboard_history_hours": config.web.dashboard_history_hours,
        "logging.level": config.logging.level,
    }


def _detect_matching_preset(config: AppConfig) -> str:
    for preset_name, preset in SETTINGS_PRESETS.items():
        if preset_name == "custom":
            continue
        detection_matches = all(
            getattr(config.detection, key) == value for key, value in preset.get("detection", {}).items()
        )
        aggregation_matches = all(
            getattr(config.aggregation, key) == value for key, value in preset.get("aggregation", {}).items()
        )
        if detection_matches and aggregation_matches:
            return preset_name
    return "custom"


def _build_config_from_form(config: AppConfig, form) -> AppConfig:
    preset_name = form.get("preset", "custom")
    detection_values = {
        "initial_noise_floor_dbfs": _float_from_form(form, "detection.initial_noise_floor_dbfs"),
        "activation_margin_db": _float_from_form(form, "detection.activation_margin_db"),
        "release_margin_db": _float_from_form(form, "detection.release_margin_db"),
        "min_event_dbfs": _float_from_form(form, "detection.min_event_dbfs"),
        "min_active_frames": _int_from_form(form, "detection.min_active_frames"),
        "max_inactive_frames": _int_from_form(form, "detection.max_inactive_frames"),
        "noise_floor_alpha": _float_from_form(form, "detection.noise_floor_alpha"),
    }
    aggregation_values = {
        "noise_interval_seconds": _float_from_form(form, "aggregation.noise_interval_seconds"),
        "pre_roll_seconds": _float_from_form(form, "aggregation.pre_roll_seconds"),
        "post_roll_seconds": _float_from_form(form, "aggregation.post_roll_seconds"),
        "min_event_seconds": _float_from_form(form, "aggregation.min_event_seconds"),
        "focus_clip_seconds": _float_from_form(form, "aggregation.focus_clip_seconds"),
        "max_clip_seconds": _float_from_form(form, "aggregation.max_clip_seconds"),
        "max_event_seconds": _float_from_form(form, "aggregation.max_event_seconds"),
    }
    if preset_name in SETTINGS_PRESETS and preset_name != "custom":
        detection_values.update(SETTINGS_PRESETS[preset_name].get("detection", {}))
        aggregation_values.update(SETTINGS_PRESETS[preset_name].get("aggregation", {}))

    arecord_device_mode = form.get("audio.arecord_device_mode", "auto")
    arecord_device = form.get("audio.arecord_device", "").strip() if arecord_device_mode == "manual" else None
    if arecord_device == "":
        arecord_device = None

    config_path = config.config_path
    return AppConfig(
        base_dir=config.base_dir,
        audio=AudioConfig(
            sample_rate=_int_from_form(form, "audio.sample_rate"),
            channels=_int_from_form(form, "audio.channels"),
            level_display_mode=form.get("audio.level_display_mode", config.audio.level_display_mode),
            calibration_slope=_float_from_form(form, "audio.calibration_slope"),
            calibration_offset_db=_float_from_form(form, "audio.calibration_offset_db"),
            frame_duration_seconds=_float_from_form(form, "audio.frame_duration_seconds"),
            arecord_binary=config.audio.arecord_binary,
            arecord_device=arecord_device,
            retry_backoff_seconds=_float_from_form(form, "audio.retry_backoff_seconds"),
        ),
        detection=DetectionConfig(**detection_values),
        aggregation=AggregationConfig(**aggregation_values),
        classifier=ClassifierConfig(
            backend=form.get("classifier.backend", config.classifier.backend),
            reuse_similarity_threshold=config.classifier.reuse_similarity_threshold,
            reuse_confidence_threshold=config.classifier.reuse_confidence_threshold,
            similarity_cache_limit=config.classifier.similarity_cache_limit,
            similarity_lookback_days=config.classifier.similarity_lookback_days,
            yamnet_model_path=config.classifier.yamnet_model_path,
            yamnet_class_map_path=config.classifier.yamnet_class_map_path,
            yamnet_model_url=config.classifier.yamnet_model_url,
            yamnet_class_map_url=config.classifier.yamnet_class_map_url,
            yamnet_num_threads=_int_from_form(form, "classifier.yamnet_num_threads"),
            yamnet_max_analysis_seconds=_float_from_form(form, "classifier.yamnet_max_analysis_seconds"),
            yamnet_max_windows=_int_from_form(form, "classifier.yamnet_max_windows"),
            yamnet_min_category_score=_float_from_form(form, "classifier.yamnet_min_category_score"),
            min_persist_confidence=_float_from_form(form, "classifier.min_persist_confidence"),
            yamnet_top_k=_int_from_form(form, "classifier.yamnet_top_k"),
        ),
        storage=StorageConfig(
            database_path=_resolve_settings_path(config.base_dir, form.get("storage.database_path", "")),
            clip_dir=_resolve_settings_path(config.base_dir, form.get("storage.clip_dir", "")),
            keep_clips=_bool_from_form(form, "storage.keep_clips"),
            clip_max_megabytes=_int_from_form(form, "storage.clip_max_megabytes"),
            clip_max_age_days=_int_from_form(form, "storage.clip_max_age_days"),
            min_free_disk_megabytes=_int_from_form(form, "storage.min_free_disk_megabytes"),
        ),
        web=WebConfig(
            host=form.get("web.host", config.web.host),
            port=_int_from_form(form, "web.port"),
            recent_events_limit=_int_from_form(form, "web.recent_events_limit"),
            dashboard_history_hours=_int_from_form(form, "web.dashboard_history_hours"),
        ),
        logging=LoggingConfig(level=form.get("logging.level", config.logging.level)),
        config_path=config_path,
    )


def _resolve_settings_path(base_dir: Path, value: str) -> Path:
    raw_value = value.strip()
    if not raw_value:
        return base_dir
    return Path(raw_value).resolve() if Path(raw_value).is_absolute() else (base_dir / raw_value).resolve()


def _display_path(base_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _int_from_form(form, key: str) -> int:
    return int(float(form.get(key, 0)))


def _float_from_form(form, key: str) -> float:
    return float(form.get(key, 0.0))


def _bool_from_form(form, key: str) -> bool:
    return form.get(key, "false") == "true"


def _schedule_audio_monitor_restart() -> None:
    timer = threading.Timer(1.0, _restart_audio_monitor_service)
    timer.daemon = True
    timer.start()


def _restart_audio_monitor_service() -> None:
    os._exit(1)


def _describe_classifier_decision(decision: dict[str, object]) -> dict[str, object]:
    details = decision.get("details")
    normalized_details = details if isinstance(details, dict) else {}
    classifier_name = str(decision.get("classifier_name") or "unknown")
    cache_hit = bool(normalized_details.get("cache_hit"))
    external_api_name = normalized_details.get("external_api_name") or normalized_details.get("api_name")
    used_external_api = bool(normalized_details.get("used_external_api") or external_api_name)
    external_api_label = "Zewnętrzne API" if external_api_name else None

    if used_external_api:
        source = "external_api"
        source_label = "Zewnętrzne API"
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
        "external_api_name": external_api_label,
        "cache_hit": cache_hit,
        "cache_similarity": normalized_details.get("cache_similarity"),
        "cache_source_event_id": normalized_details.get("cache_source_event_id"),
        "fallback_reason": normalized_details.get("fallback_reason"),
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


def _format_worker_state(value: str | None) -> str:
    if not value:
        return WORKER_STATE_LABELS["idle"]
    return WORKER_STATE_LABELS.get(value, value)


def _translate_label(value: str | None) -> str:
    if not value:
        return "-"
    if value in CATEGORY_LABELS:
        return CATEGORY_LABELS[value]
    if value in YAMNET_LABEL_TRANSLATIONS:
        return YAMNET_LABEL_TRANSLATIONS[value]
    return value


def _manual_label_options_cache_key(class_map_path: Path) -> tuple[str, int, int]:
    try:
        stat_result = class_map_path.stat()
        return (str(class_map_path.resolve()), stat_result.st_mtime_ns, stat_result.st_size)
    except OSError:
        return (str(class_map_path.resolve()), 0, 0)


@lru_cache(maxsize=8)
def _manual_label_options_for_path(class_map_path: str, _mtime_ns: int, _size: int) -> tuple[tuple[str, str], ...]:
    values: dict[str, str] = {key: label for key, label in CATEGORY_LABELS.items()}
    try:
        with Path(class_map_path).open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_label = (row.get("display_name") or "").strip()
                if raw_label and raw_label not in values:
                    values[raw_label] = _translate_label(raw_label)
    except OSError:
        pass

    ordered = sorted(values.items(), key=lambda item: item[1].casefold())
    return tuple((value, label) for value, label in ordered)


def _read_system_status(config: AppConfig) -> dict[str, object]:
    return {
        "uptime_seconds": _read_system_uptime_seconds(),
        "cpu_percent": _read_cpu_load_percent(),
        "cpu_temperature_c": _read_cpu_temperature_c(),
        "memory_available_gb": _read_memory_available_gb(),
        "disk_free_gb": _read_disk_free_gb(config.storage.database_path.parent),
    }


def _read_systemd_service_status(service_name: str) -> dict[str, str | None]:
    status = {"active": None, "enabled": None}
    try:
        active_result = subprocess.run(
            ["systemctl", "is-active", service_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        status["active"] = active_result.stdout.strip() or active_result.stderr.strip() or None
    except (OSError, subprocess.SubprocessError):
        return status

    try:
        enabled_result = subprocess.run(
            ["systemctl", "is-enabled", service_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        status["enabled"] = enabled_result.stdout.strip() or enabled_result.stderr.strip() or None
    except (OSError, subprocess.SubprocessError):
        return status

    return status


def _read_system_uptime_seconds() -> float | None:
    uptime_path = Path("/proc/uptime")
    try:
        raw_value = uptime_path.read_text(encoding="utf-8").split()[0]
        return float(raw_value)
    except (OSError, ValueError, IndexError):
        return None


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
