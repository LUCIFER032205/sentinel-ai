from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import MODEL_PATH, THRESHOLD_PATH
from .data import load_image_for_inference

DEFAULT_THRESHOLD = 0.50
DEFAULT_UNCERTAIN_MARGIN = 0.12


@dataclass
class PredictionResult:
    label: str
    confidence: float
    ai_probability: float
    uncertain: bool


@dataclass
class CalibrationConfig:
    threshold: float
    uncertain_low: float
    uncertain_high: float


def load_calibration(calibration_path: Path = THRESHOLD_PATH) -> CalibrationConfig:
    threshold = DEFAULT_THRESHOLD
    uncertain_low = max(0.0, threshold - DEFAULT_UNCERTAIN_MARGIN)
    uncertain_high = min(1.0, threshold + DEFAULT_UNCERTAIN_MARGIN)

    if calibration_path.exists():
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        threshold = float(payload.get("threshold", threshold))
        uncertain_low = float(payload.get("uncertain_low", uncertain_low))
        uncertain_high = float(payload.get("uncertain_high", uncertain_high))

    threshold = float(np.clip(threshold, 0.01, 0.99))
    uncertain_low = float(np.clip(uncertain_low, 0.0, 1.0))
    uncertain_high = float(np.clip(uncertain_high, 0.0, 1.0))
    if uncertain_low >= uncertain_high:
        uncertain_low = max(0.0, threshold - DEFAULT_UNCERTAIN_MARGIN)
        uncertain_high = min(1.0, threshold + DEFAULT_UNCERTAIN_MARGIN)

    return CalibrationConfig(
        threshold=threshold,
        uncertain_low=uncertain_low,
        uncertain_high=uncertain_high,
    )


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
    orientation_conservative: bool = False,
) -> PredictionResult:
    image = load_image_for_inference(image_bytes)

    if orientation_conservative:
        rotations = [image, np.rot90(image, 1), np.rot90(image, 2), np.rot90(image, 3)]
        batch = np.stack(
            [variant for rotated in rotations for variant in (rotated, np.flip(rotated, axis=1))],
            axis=0,
        )
        rotation_scores = model.predict(batch, verbose=0).ravel().reshape(len(rotations), 2).mean(axis=1)
        # In real-photo safe mode, require the AI signal to survive orientation changes.
        ai_probability = float(np.min(rotation_scores))
    else:
        # Average the original + horizontally flipped predictions for more stable scores.
        batch = np.stack([image, np.flip(image, axis=1)], axis=0)
        ai_probability = float(np.mean(model.predict(batch, verbose=0).ravel()))

    calibration = calibration or load_calibration()
    uncertain = calibration.uncertain_low < ai_probability < calibration.uncertain_high
    label = "Uncertain"
    if not uncertain:
        label = "AI-generated" if ai_probability >= calibration.threshold else "Real"

    if ai_probability >= calibration.threshold:
        confidence = (ai_probability - calibration.threshold) / (1.0 - calibration.threshold)
    else:
        confidence = (calibration.threshold - ai_probability) / calibration.threshold
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return PredictionResult(
        label=label,
        confidence=confidence,
        ai_probability=ai_probability,
        uncertain=uncertain,
    )
