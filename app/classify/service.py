from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import shutil
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from app.config import ClassifierConfig
from app.models import (
    ClassificationOutcome,
    ClassifierDecision,
    CompletedEvent,
)
from app.storage.database import SQLiteRepository
from app.classify.heuristics import HeuristicEventClassifier

LOGGER = logging.getLogger(__name__)
BIRDNET_API_NAME = "BirdNET API"

YAMNET_CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "ambulance": {
        "Ambulance (siren)": 1.0,
        "Siren": 0.30,
        "Civil defense siren": 0.20,
    },
    "police": {
        "Police car (siren)": 1.0,
        "Siren": 0.30,
        "Civil defense siren": 0.15,
        "Car alarm": 0.10,
    },
    "fire_truck": {
        "Fire engine, fire truck (siren)": 1.0,
        "Siren": 0.30,
        "Civil defense siren": 0.20,
        "Truck": 0.08,
    },
    "airplane": {
        "Fixed-wing aircraft, airplane": 1.0,
        "Aircraft": 0.70,
        "Aircraft engine": 0.55,
        "Jet engine": 0.45,
    },
    "helicopter": {
        "Helicopter": 1.0,
        "Aircraft": 0.25,
    },
    "street_background": {
        "Traffic noise, roadway noise": 1.0,
        "Vehicle": 0.28,
        "Car": 0.24,
        "Truck": 0.20,
        "Engine": 0.16,
        "Outside, urban or manmade": 0.24,
    },
}

YAMNET_DISCARD_LABELS = {
    "White noise",
    "Silence",
    "Inside, small room",
}


@dataclass(slots=True)
class YAMNetModelOutput:
    mean_scores: dict[str, float]
    peak_scores: dict[str, float]
    top_labels: list[dict[str, float | str]]


@dataclass(slots=True)
class BirdNETPrediction:
    species_label: str
    score: float
    scientific_name: str | None
    common_name: str


class BirdNETClient:
    def __init__(self, config: ClassifierConfig) -> None:
        self.config = config

    @property
    def is_enabled(self) -> bool:
        return bool(self.config.birdnet_api_url.strip())

    def identify(self, event: CompletedEvent, trigger_labels: list[str], yamnet_decision: ClassifierDecision) -> ClassifierDecision | None:
        if not self.is_enabled or event.clip_samples.size == 0:
            return None

        audio_bytes = _build_wav_bytes(event.clip_samples, event.sample_rate)
        meta = {
            "locale": self.config.birdnet_locale,
            "num_results": self.config.birdnet_num_results,
            "min_conf": self.config.birdnet_min_confidence,
            "save": False,
        }
        boundary = f"----bam-birdnet-{uuid.uuid4().hex}"
        body = _build_multipart_form_data(
            boundary,
            fields={"meta": json.dumps(meta)},
            file_field_name="audio",
            filename="event.wav",
            content_type="audio/wav",
            file_bytes=audio_bytes,
        )
        request = urllib.request.Request(
            self.config.birdnet_api_url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "bam/0.1",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.config.birdnet_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if str(payload.get("msg")) != "success":
            raise RuntimeError(f"BirdNET API returned {payload.get('msg') or 'error'}")

        raw_results = payload.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            return None

        predictions = [_parse_birdnet_prediction(item) for item in raw_results]
        predictions = [item for item in predictions if item is not None]
        if not predictions:
            return None

        best = predictions[0]
        if best.score < self.config.birdnet_min_confidence:
            return None

        return ClassifierDecision(
            classifier_name="birdnet_remote",
            classifier_version="1",
            category=best.common_name,
            confidence=best.score,
            details={
                **yamnet_decision.details,
                "used_external_api": True,
                "external_api_name": BIRDNET_API_NAME,
                "birdnet_species_label": best.species_label,
                "birdnet_common_name": best.common_name,
                "birdnet_scientific_name": best.scientific_name,
                "birdnet_results": [
                    {
                        "species_label": item.species_label,
                        "common_name": item.common_name,
                        "scientific_name": item.scientific_name,
                        "score": item.score,
                    }
                    for item in predictions[: self.config.birdnet_num_results]
                ],
                "birdnet_trigger_labels": trigger_labels,
            },
        )


class AppClassifier:
    def __init__(self, config: ClassifierConfig, repository: SQLiteRepository) -> None:
        self.config = config
        self.repository = repository
        self.yamnet = YAMNetClassifier(config)
        self.heuristics = HeuristicEventClassifier()
        self.birdnet = BirdNETClient(config)

    def classify(self, event: CompletedEvent) -> ClassificationOutcome:
        if self.config.backend == "heuristic":
            return ClassificationOutcome(decision=self.heuristics.classify(event.summary))

        signature_hash, signature = compute_audio_signature(event.clip_samples, event.sample_rate)
        try:
            decision = self.yamnet.classify(event)
            trigger_labels = _extract_bird_trigger_labels(decision, self.config.yamnet_min_category_score)
            decision.details = {
                **decision.details,
                "birdnet_trigger_labels": trigger_labels,
            }
            if trigger_labels:
                if not self.birdnet.is_enabled:
                    decision.details = {
                        **decision.details,
                        "birdnet_lookup_status": "disabled",
                        "birdnet_lookup_reason": "BirdNET API nie jest skonfigurowane",
                    }
                else:
                    try:
                        birdnet_decision = self.birdnet.identify(event, trigger_labels, decision)
                    except Exception as exc:  # pragma: no cover - network/runtime dependent
                        LOGGER.warning("BirdNET API lookup failed, keeping YAMNet decision: %s", exc)
                        decision.details = {
                            **decision.details,
                            "birdnet_lookup_status": "error",
                            "birdnet_lookup_reason": str(exc),
                        }
                    else:
                        if birdnet_decision is not None:
                            decision = birdnet_decision
                        else:
                            decision.details = {
                                **decision.details,
                                "birdnet_lookup_status": "no_result",
                                "birdnet_lookup_reason": "BirdNET nie zwrócił wyniku powyżej progu",
                            }
        except Exception as exc:  # pragma: no cover - exercised on device when runtime/model fails
            LOGGER.warning("YAMNet unavailable, falling back to heuristics: %s", exc)
            decision = self.heuristics.classify(event.summary)
            decision.classifier_name = "heuristic_fallback"
            decision.details = {**decision.details, "fallback_reason": str(exc)}

        return ClassificationOutcome(decision=decision, signature_hash=signature_hash, signature=signature)

    def remember(self, outcome: ClassificationOutcome, event_id: int) -> None:
        return


class YAMNetClassifier:
    classifier_name = "yamnet_litert"
    classifier_version = "1"

    def __init__(self, config: ClassifierConfig) -> None:
        self.config = config
        self._lock = Lock()
        self._interpreter = None
        self._input_index: int | None = None
        self._input_shape: tuple[int, ...] | None = None
        self._score_output_index: int | None = None
        self._class_names: list[str] | None = None

    def classify(self, event: CompletedEvent) -> ClassifierDecision:
        model_output = self._classify_samples(event.clip_samples)
        category, confidence, category_scores, resolved_label, resolved_label_score = self._map_to_domain_category(
            model_output
        )
        return ClassifierDecision(
            classifier_name=self.classifier_name,
            classifier_version=self.classifier_version,
            category=category,
            confidence=confidence,
            details={
                "top_labels": model_output.top_labels,
                "category_scores": category_scores,
                "resolved_label": resolved_label,
                "resolved_label_score": resolved_label_score,
                "cache_hit": False,
            },
        )

    def _classify_samples(self, samples: np.ndarray) -> YAMNetModelOutput:
        interpreter = self._ensure_model_loaded()
        input_index = self._input_index
        score_output_index = self._score_output_index
        assert input_index is not None
        assert score_output_index is not None

        inference_batches = self._build_inference_batches(samples)
        score_chunks: list[np.ndarray] = []

        with self._lock:
            for batch in inference_batches:
                interpreter.resize_tensor_input(input_index, batch.shape, strict=False)
                interpreter.allocate_tensors()
                interpreter.set_tensor(input_index, batch)
                interpreter.invoke()
                scores = np.asarray(interpreter.get_tensor(score_output_index), dtype=np.float32)
                scores = scores.reshape(-1, scores.shape[-1])
                score_chunks.append(scores)

        all_scores = np.concatenate(score_chunks, axis=0)
        mean_scores = np.mean(all_scores, axis=0)
        peak_scores = np.max(all_scores, axis=0)
        class_names = self._class_names or []

        top_indices = np.argsort(mean_scores)[::-1][: self.config.yamnet_top_k]
        top_labels = [
            {
                "label": class_names[index],
                "mean_score": round(float(mean_scores[index]), 6),
                "peak_score": round(float(peak_scores[index]), 6),
            }
            for index in top_indices
        ]

        mean_by_label = {class_names[index]: float(mean_scores[index]) for index in range(len(class_names))}
        peak_by_label = {class_names[index]: float(peak_scores[index]) for index in range(len(class_names))}
        return YAMNetModelOutput(mean_scores=mean_by_label, peak_scores=peak_by_label, top_labels=top_labels)

    def _map_to_domain_category(
        self,
        output: YAMNetModelOutput,
    ) -> tuple[str, float, dict[str, float], str | None, float | None]:
        label_scores: dict[str, float] = {}
        for label, mean_value in output.mean_scores.items():
            peak_value = output.peak_scores.get(label, mean_value)
            label_scores[label] = 0.35 * mean_value + 0.65 * peak_value

        top_label = max(label_scores.items(), key=lambda item: item[1], default=(None, 0.0))
        top_label_name, top_label_score = top_label

        category_scores: dict[str, float] = {}
        for category, weights in YAMNET_CATEGORY_WEIGHTS.items():
            weighted_scores = [label_scores.get(label, 0.0) * weight for label, weight in weights.items()]
            category_scores[category] = round(float(max(weighted_scores, default=0.0)), 6)

        if top_label_name in YAMNET_DISCARD_LABELS:
            return (
                "discarded",
                round(float(top_label_score), 6),
                category_scores,
                top_label_name,
                round(float(top_label_score), 6),
            )

        non_background = {k: v for k, v in category_scores.items() if k != "street_background"}
        top_category = max(non_background, key=non_background.get, default="street_background")
        top_score = non_background.get(top_category, 0.0)
        if top_score >= self.config.yamnet_min_category_score:
            return top_category, round(float(top_score), 6), category_scores, top_category, round(float(top_score), 6)

        background_score = category_scores.get("street_background", 0.0)
        if background_score >= self.config.yamnet_min_category_score:
            return (
                "street_background",
                round(float(background_score), 6),
                category_scores,
                "Traffic noise, roadway noise",
                round(float(background_score), 6),
            )

        if top_label_name and top_label_score >= self.config.yamnet_min_category_score:
            return (
                str(top_label_name),
                round(float(top_label_score), 6),
                category_scores,
                str(top_label_name),
                round(float(top_label_score), 6),
            )

        fallback_confidence = max(float(top_label_score), background_score, 0.50)
        return "street_background", round(float(fallback_confidence), 6), category_scores, top_label_name, round(
            float(top_label_score), 6
        )

    def _build_inference_batches(self, samples: np.ndarray) -> list[np.ndarray]:
        normalized = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
        max_samples = int(self.config.yamnet_max_analysis_seconds * 16_000)
        if normalized.size > max_samples:
            normalized = normalized[:max_samples]

        input_shape = self._input_shape or ()
        flat_input_length = _infer_input_length(input_shape)
        if flat_input_length is None:
            return [_reshape_input_batch(normalized, input_shape)]

        if normalized.size <= flat_input_length:
            padded = np.pad(normalized, (0, flat_input_length - normalized.size))
            return [_reshape_input_batch(padded, input_shape)]

        hop = max(flat_input_length // 2, 1)
        starts = list(range(0, normalized.size - flat_input_length + 1, hop))
        if len(starts) > self.config.yamnet_max_windows:
            starts = evenly_spaced_indices(starts, self.config.yamnet_max_windows)

        batches: list[np.ndarray] = []
        for start in starts:
            window = normalized[start : start + flat_input_length]
            batches.append(_reshape_input_batch(window, input_shape))
        return batches

    def _ensure_model_loaded(self):
        with self._lock:
            if self._interpreter is not None:
                return self._interpreter

            self._ensure_assets()
            tflite = load_litert_module()
            interpreter = tflite.Interpreter(
                model_path=str(self.config.yamnet_model_path),
                num_threads=self.config.yamnet_num_threads,
            )
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()
            class_names = load_class_names(self.config.yamnet_class_map_path)

            score_output_index = None
            for detail in output_details:
                shape = tuple(int(item) for item in detail["shape"])
                if shape and shape[-1] == len(class_names):
                    score_output_index = int(detail["index"])

            if score_output_index is None:
                raise RuntimeError("YAMNet output tensor with class scores was not found.")

            self._interpreter = interpreter
            self._input_index = int(input_details["index"])
            self._input_shape = tuple(int(item) for item in input_details["shape"])
            self._score_output_index = score_output_index
            self._class_names = class_names
            LOGGER.info("Loaded YAMNet model from %s", self.config.yamnet_model_path)
            return self._interpreter

    def _ensure_assets(self) -> None:
        self.config.yamnet_model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config.yamnet_model_path.exists():
            download_file(self.config.yamnet_model_url, self.config.yamnet_model_path)
        if not self.config.yamnet_class_map_path.exists():
            download_file(self.config.yamnet_class_map_url, self.config.yamnet_class_map_path)


def compute_audio_signature(samples: np.ndarray, sample_rate: int) -> tuple[str, list[float]]:
    if samples.size == 0:
        signature = [0.0] * 26
        return hashlib.sha1(json.dumps(signature).encode("utf-8")).hexdigest(), signature

    frame_length = 2048
    hop = 1024
    max_frames = 64
    usable = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    starts = list(range(0, max(usable.size - frame_length + 1, 1), hop))
    if not starts:
        starts = [0]
    if len(starts) > max_frames:
        starts = evenly_spaced_indices(starts, max_frames)

    spectra: list[np.ndarray] = []
    window = np.hanning(frame_length).astype(np.float32)
    for start in starts:
        chunk = usable[start : start + frame_length]
        if chunk.size < frame_length:
            chunk = np.pad(chunk, (0, frame_length - chunk.size))
        spectrum = np.abs(np.fft.rfft(chunk * window))
        spectra.append(np.square(spectrum))

    average_spectrum = np.mean(np.stack(spectra, axis=0), axis=0)
    bands = np.array_split(average_spectrum, 24)
    signature = np.array([np.log1p(float(np.mean(band))) for band in bands], dtype=np.float32)
    duration_feature = min(float(usable.size / max(sample_rate, 1)) / 30.0, 1.0)
    rms_feature = float(np.sqrt(np.mean(np.square(usable)) + 1e-12))
    combined = np.concatenate([signature, np.array([duration_feature, rms_feature], dtype=np.float32)])
    norm = float(np.linalg.norm(combined))
    if norm > 0.0:
        combined /= norm
    quantized = [round(float(value), 6) for value in combined.tolist()]
    signature_hash = hashlib.sha1(json.dumps(quantized).encode("utf-8")).hexdigest()
    return signature_hash, quantized


def _extract_bird_trigger_labels(decision: ClassifierDecision, min_score: float) -> list[str]:
    details = decision.details if isinstance(decision.details, dict) else {}
    labels: list[str] = []
    resolved_label = details.get("resolved_label")
    resolved_score = details.get("resolved_label_score")
    if isinstance(resolved_label, str) and _should_run_birdnet_for_yamnet_label(resolved_label):
        score = float(resolved_score) if isinstance(resolved_score, int | float) else 0.0
        if score >= min_score:
            labels.append(resolved_label)

    top_labels = details.get("top_labels")
    if isinstance(top_labels, list):
        for item in top_labels:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if not isinstance(label, str) or not _should_run_birdnet_for_yamnet_label(label):
                continue
            mean_score = float(item.get("mean_score")) if isinstance(item.get("mean_score"), int | float) else 0.0
            peak_score = float(item.get("peak_score")) if isinstance(item.get("peak_score"), int | float) else 0.0
            if max(mean_score, peak_score) >= min_score:
                labels.append(label)

    deduped: list[str] = []
    for label in labels:
        if label not in deduped:
            deduped.append(label)
    return deduped


_BIRD_TRIGGER_EXACT_LABELS = {
    "Bird",
    "Bird vocalization, bird call, bird song",
    "Bird flight, flapping wings",
    "Animal",
    "Domestic animals, pets",
    "Wild animals",
    "Livestock, farm animals, working animals",
    "Fowl",
    "Chicken, rooster",
    "Cluck",
    "Crowing, cock-a-doodle-doo",
    "Turkey",
    "Gobble",
    "Duck",
    "Quack",
    "Goose",
    "Honk",
    "Pigeon, dove",
    "Coo",
    "Crow",
    "Caw",
    "Owl",
    "Hoot",
    "Canidae, dogs, wolves",
    "Dog",
    "Whimper (dog)",
    "Howl",
    "Growling",
    "Cat",
    "Caterwaul",
    "Rodents, rats, mice",
    "Whimper (dog)",
    "Roaring cats (lions, tigers)",
    "Mouse",
    "Pig",
    "Oink",
    "Goat",
    "Bleat",
    "Sheep",
    "Horse",
    "Moo",
    "Cattle, bovinae",
}


def _should_run_birdnet_for_yamnet_label(label: str) -> bool:
    normalized = label.casefold()
    if "bird" in normalized or "animal" in normalized:
        return True
    return label in _BIRD_TRIGGER_EXACT_LABELS


def _build_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype("<i2")
    handle = io.BytesIO()
    with wave.open(handle, "wb") as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(sample_rate)
        wav_handle.writeframes(pcm_i16.tobytes())
    return handle.getvalue()


def _build_multipart_form_data(
    boundary: str,
    *,
    fields: dict[str, str],
    file_field_name: str,
    filename: str,
    content_type: str,
    file_bytes: bytes,
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field_name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks)


def _parse_birdnet_prediction(item: object) -> BirdNETPrediction | None:
    if not isinstance(item, list | tuple) or len(item) < 2:
        return None
    species_label = str(item[0]).strip()
    if not species_label:
        return None
    try:
        score = float(item[1])
    except (TypeError, ValueError):
        return None
    scientific_name, common_name = _split_species_label(species_label)
    return BirdNETPrediction(
        species_label=species_label,
        score=score,
        scientific_name=scientific_name,
        common_name=common_name,
    )


def _split_species_label(species_label: str) -> tuple[str | None, str]:
    if "_" not in species_label:
        return None, species_label
    scientific_name, common_name = species_label.split("_", 1)
    return scientific_name or None, common_name or species_label


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.shape != right.shape:
        return 0.0
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def evenly_spaced_indices(items: list[int], limit: int) -> list[int]:
    if len(items) <= limit:
        return items
    positions = np.linspace(0, len(items) - 1, num=limit, dtype=int)
    return [items[index] for index in positions]


def _infer_input_length(shape: tuple[int, ...]) -> int | None:
    if not shape:
        return None
    positive_dims = [dimension for dimension in shape if dimension > 1]
    if not positive_dims:
        return None
    return int(np.prod(positive_dims))


def _reshape_input_batch(samples: np.ndarray, input_shape: tuple[int, ...]) -> np.ndarray:
    if len(input_shape) == 1:
        return samples.astype(np.float32)
    return samples.reshape(1, -1).astype(np.float32)


def download_file(url: str, target_path: Path) -> None:
    LOGGER.info("Downloading %s to %s", url, target_path)
    request = urllib.request.Request(url, headers={"User-Agent": "bam/0.1"})
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with urllib.request.urlopen(request, timeout=60) as response, temporary_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary_path.replace(target_path)


def load_class_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["display_name"] for row in reader]


def load_litert_module():
    errors: list[str] = []
    try:
        import ai_edge_litert.interpreter as tflite

        return tflite
    except Exception as exc:  # pragma: no cover - platform specific
        errors.append(f"ai_edge_litert: {exc}")
    try:
        import tflite_runtime.interpreter as tflite

        return tflite
    except Exception as exc:  # pragma: no cover - platform specific
        errors.append(f"tflite_runtime: {exc}")
    try:
        import tensorflow as tf  # type: ignore

        return tf.lite
    except Exception as exc:  # pragma: no cover - platform specific
        errors.append(f"tensorflow.lite: {exc}")
    raise RuntimeError("No LiteRT/TFLite interpreter is available. " + " | ".join(errors))
