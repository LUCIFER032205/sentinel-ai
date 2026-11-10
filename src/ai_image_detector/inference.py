from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import MODEL_PATH, THRESHOLD_PATH
from .data import load_image_for_inference

DEFAULT_THRESHOLD = 0.55
DEFAULT_BAND = 0.10


@dataclass
class PredictionResult:
    label: str
    confidence: float
    ai_probability: float
    low_confidence: bool


@dataclass
class CalibrationConfig:
    threshold: float
    band_low: float
    band_high: float


def load_calibration(calibration_path: Path = THRESHOLD_PATH) -> CalibrationConfig:
    """Read the decision threshold produced by calibrate_thresholds.py.

    Falls back to DEFAULT_THRESHOLD when the file is absent, and says so loudly.
    A silent fallback is how a deployment ends up scoring at a different threshold
    from the one that was tuned, with nothing to catch the divergence.
    """
    threshold = DEFAULT_THRESHOLD
    band_low = threshold - DEFAULT_BAND
    band_high = threshold + DEFAULT_BAND

    if not calibration_path.exists():
        warnings.warn(
            f"No calibration file at {calibration_path}. Falling back to "
            f"DEFAULT_THRESHOLD={DEFAULT_THRESHOLD} and a +/-{DEFAULT_BAND} band. "
            "This is the hardcoded default, NOT a tuned threshold - if the model "
            "has been retrained, these verdicts are calibrated for a different one.",
            RuntimeWarning,
            stacklevel=2,
        )

    if calibration_path.exists():
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        threshold = float(payload.get("threshold", threshold))
        # calibrate_thresholds.py still writes these two keys. They no longer
        # produce a separate "Uncertain" verdict - they mark the band where the
        # model is close to a coin flip, which the result reports as a flag.
        band_low = float(payload.get("uncertain_low", threshold - DEFAULT_BAND))
        band_high = float(payload.get("uncertain_high", threshold + DEFAULT_BAND))

    # Round: band_low is computed as threshold - DEFAULT_BAND, which gives
    # 0.45000000000000007 in binary floating point and shows up in the UI.
    threshold = round(float(np.clip(threshold, 0.01, 0.99)), 4)
    band_low = round(float(np.clip(band_low, 0.0, 1.0)), 4)
    band_high = round(float(np.clip(band_high, 0.0, 1.0)), 4)
    if band_low >= band_high:
        band_low = max(0.0, threshold - DEFAULT_BAND)
        band_high = min(1.0, threshold + DEFAULT_BAND)

    return CalibrationConfig(threshold=threshold, band_low=band_low, band_high=band_high)


def load_trained_model(model_path: Path = MODEL_PATH) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train the model first."
        )
    return tf.keras.models.load_model(model_path)


def predict_image_bytes(
    model: tf.keras.Model,
    image_bytes: bytes,
    calibration: CalibrationConfig | None = None,
) -> PredictionResult:
    image = load_image_for_inference(image_bytes)

    # Average the original and horizontally flipped score. Measured over 1600
    # held-out images this beats a single pass (87.0%) and beats averaging over
    # four rotations (88.4%) at a quarter of the cost - two forward passes, not
    # eight. Taking the *minimum* over rotations, which this used to do in
    # "default" mode, scored 84.2% and cost 35 points of AI recall.
    batch = np.stack([image, np.flip(image, axis=1)], axis=0)
    ai_probability = float(np.mean(model.predict(batch, verbose=0).ravel()))

    calibration = calibration or load_calibration()
    label = "AI-generated" if ai_probability >= calibration.threshold else "Real"
    low_confidence = calibration.band_low < ai_probability < calibration.band_high

    if ai_probability >= calibration.threshold:
        confidence = (ai_probability - calibration.threshold) / (1.0 - calibration.threshold)
    else:
        confidence = (calibration.threshold - ai_probability) / calibration.threshold
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return PredictionResult(
        label=label,
        confidence=confidence,
        ai_probability=ai_probability,
        low_confidence=low_confidence,
    )
