from __future__ import annotations

from app.models import ClassifierDecision, EventSummary


CATEGORY_LABELS = {
    "Snoring": "Chrapanie",
    "ambulance": "Karetka / syrena karetki",
    "police": "Policja / syrena radiowozu",
    "fire_truck": "Straż pożarna / syrena wozu strażackiego",
    "airplane": "Samolot",
    "helicopter": "Śmigłowiec",
    "street_background": "Nieokreślony hałas uliczny / tło",
    "speech": "Mowa ludzka",
    "other_detected": "Inny rozpoznany dźwięk",
    "discarded": "Odrzucone tło",
}


class HeuristicEventClassifier:
    classifier_name = "heuristic_baseline"
    classifier_version = "1"

    def classify(self, summary: EventSummary) -> ClassifierDecision:
        rules: list[str] = []

        if self._looks_like_rotary_aircraft(summary):
            rules.append("low-band energy with clear amplitude modulation")
            return self._decision("helicopter", 0.76, summary, rules)

        if self._looks_like_airplane(summary):
            rules.append("long low-frequency broadband event")
            return self._decision("airplane", 0.72, summary, rules)

        if self._looks_like_siren(summary):
            if summary.mean_centroid_hz < 900:
                rules.append("sirena o niskim centrum widma")
                return self._decision("fire_truck", 0.65, summary, rules)
            if summary.high_band_ratio > 0.18 or summary.dominant_modulation_hz >= 1.4:
                rules.append("wysokie pasmo i szybsza modulacja")
                return self._decision("police", 0.64, summary, rules)
            rules.append("szeroki sweep w pasmie syren")
            return self._decision("ambulance", 0.66, summary, rules)

        if summary.mean_dbfs < -42.0 and summary.low_band_ratio < 0.35:
            rules.append("slabe zdarzenie o niskiej energii")
            return self._decision("street_background", 0.55, summary, rules)

        rules.append("fallback do kategorii tla ulicznego")
        return self._decision("street_background", 0.51, summary, rules)

    @staticmethod
    def _looks_like_siren(summary: EventSummary) -> bool:
        return (
            2.0 <= summary.duration_seconds <= 25.0
            and summary.dominant_freq_hz >= 350.0
            and summary.dominant_span_hz >= 250.0
            and summary.mid_band_ratio >= 0.30
            and summary.mean_flatness <= 0.60
        )

    @staticmethod
    def _looks_like_airplane(summary: EventSummary) -> bool:
        return (
            summary.duration_seconds >= 6.0
            and summary.low_band_ratio >= 0.45
            and summary.mean_centroid_hz <= 700.0
            and summary.mean_flux <= 0.02
            and summary.dominant_span_hz <= 180.0
        )

    @staticmethod
    def _looks_like_rotary_aircraft(summary: EventSummary) -> bool:
        return (
            summary.duration_seconds >= 4.0
            and summary.low_band_ratio >= 0.40
            and summary.mean_centroid_hz <= 900.0
            and summary.rms_modulation_depth >= 0.15
            and 1.5 <= summary.dominant_modulation_hz <= 10.0
        )

    def _decision(
        self,
        category: str,
        confidence: float,
        summary: EventSummary,
        rules: list[str],
    ) -> ClassifierDecision:
        return ClassifierDecision(
            classifier_name=self.classifier_name,
            classifier_version=self.classifier_version,
            category=category,
            confidence=confidence,
            details={
                "rules": rules,
                "peak_dbfs": summary.peak_dbfs,
                "dominant_freq_hz": summary.dominant_freq_hz,
                "dominant_span_hz": summary.dominant_span_hz,
            },
        )
