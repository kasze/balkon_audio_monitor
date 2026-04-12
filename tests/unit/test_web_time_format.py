from __future__ import annotations

import os
import time

import pytest

from app.web.app import _describe_classifier_decision, _format_local_timestamp


@pytest.fixture
def warsaw_timezone() -> None:
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is not available on this platform")

    original_timezone = os.environ.get("TZ")
    os.environ["TZ"] = "Europe/Warsaw"
    time.tzset()
    try:
        yield
    finally:
        if original_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_timezone
        time.tzset()


def test_format_local_timestamp_uses_local_timezone(warsaw_timezone: None) -> None:
    formatted = _format_local_timestamp("2026-04-12T10:57:32.228818+00:00")
    assert formatted == "2026-04-12 12:57:32.22"


def test_format_local_timestamp_returns_original_on_parse_error() -> None:
    assert _format_local_timestamp("not-a-timestamp") == "not-a-timestamp"


def test_describe_classifier_decision_marks_cache_reuse() -> None:
    trace = _describe_classifier_decision(
        {
            "classifier_name": "yamnet_litert",
            "classifier_version": "1",
            "details": {
                "cache_hit": True,
                "cache_similarity": 0.9987,
                "cache_source_event_id": 42,
                "top_labels": [{"label": "Siren", "mean_score": 0.41, "peak_score": 0.87}],
                "category_scores": {"ambulance": 0.87, "street_background": 0.23},
            },
        }
    )

    assert trace["source"] == "cache_reuse"
    assert trace["used_external_api"] is False
    assert trace["cache_source_event_id"] == 42
    assert trace["top_labels"][0]["label"] == "Siren"
    assert trace["category_scores"][0]["category"] == "ambulance"


def test_describe_classifier_decision_marks_external_api_when_present() -> None:
    trace = _describe_classifier_decision(
        {
            "classifier_name": "birdnet_remote",
            "classifier_version": "1",
            "details": {
                "used_external_api": True,
                "external_api_name": "BirdNET Cloud",
            },
        }
    )

    assert trace["source"] == "external_api"
    assert trace["used_external_api"] is True
    assert trace["external_api_name"] == "BirdNET Cloud"
