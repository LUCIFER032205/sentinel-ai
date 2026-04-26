from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from src.ai_image_detector.config import MODEL_PATH, PROCESSED_DATA_DIR, SEED, THRESHOLD_PATH
from src.ai_image_detector.data import load_image_for_inference


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in VALID_SUFFIXES)


def balanced_sample(real_files: list[Path], fake_files: list[Path], per_class: int) -> tuple[list[Path], np.ndarray]:
    rng = np.random.default_rng(SEED)
    count = min(len(real_files), len(fake_files), per_class)
    real_sample = rng.choice(real_files, size=count, replace=False).tolist()
    fake_sample = rng.choice(fake_files, size=count, replace=False).tolist()
    paths = real_sample + fake_sample
    labels = np.array([0] * count + [1] * count)
    order = rng.permutation(len(paths))
    return [paths[i] for i in order], labels[order]


def predict_paths(model: tf.keras.Model, paths: list[Path], batch_size: int) -> np.ndarray:
    batches: list[np.ndarray] = []
    current: list[np.ndarray] = []

    for path in paths:
        image = load_image_for_inference(path.read_bytes())
        current.append(image)
        if len(current) == batch_size:
            batches.append(np.stack(current, axis=0))
            current = []

    if current:
        batches.append(np.stack(current, axis=0))

    probabilities: list[float] = []
    for batch in batches:
        flipped = np.flip(batch, axis=2)
        augmented_batch = np.concatenate([batch, flipped], axis=0)
        preds = model.predict(augmented_batch, verbose=0).ravel()
        averaged = (preds[: len(batch)] + preds[len(batch) :]) / 2.0
        probabilities.extend(averaged.tolist())

    return np.array(probabilities)


def find_best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    best_threshold = 0.5
    best_accuracy = -1.0
    best_f1 = -1.0

    for threshold in np.linspace(0.2, 0.8, 241):
        predicted = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(labels, predicted)
        f1 = f1_score(labels, predicted, pos_label=1, zero_division=0)
        if accuracy > best_accuracy or (accuracy == best_accuracy and f1 > best_f1):
            best_threshold = float(threshold)
            best_accuracy = float(accuracy)
            best_f1 = float(f1)

    default_predictions = (probabilities >= 0.5).astype(int)
    default_accuracy = float(accuracy_score(labels, default_predictions))
    default_f1 = float(f1_score(labels, default_predictions, pos_label=1, zero_division=0))

    # Avoid over-tuning the UI to a tiny validation gain. A very low threshold can
    # make normal gallery photos look AI-generated in real-world use.
    if best_accuracy < default_accuracy + 0.02:
        best_threshold = 0.5
        best_accuracy = default_accuracy
        best_f1 = default_f1

    best_threshold = float(np.clip(best_threshold, 0.35, 0.65))
    return {
        "threshold": best_threshold,
        "uncertain_low": float(max(0.0, best_threshold - 0.10)),
        "uncertain_high": float(min(1.0, best_threshold + 0.10)),
        "validation_accuracy_default_0_5": default_accuracy,
        "validation_accuracy": best_accuracy,
        "validation_f1_fake": best_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh decision thresholds for the trained model.")
    parser.add_argument("--per-class", type=int, default=2000, help="Number of real and fake images to sample.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=THRESHOLD_PATH)
    args = parser.parse_args()

    real_files = image_files(PROCESSED_DATA_DIR / "real")
    fake_files = image_files(PROCESSED_DATA_DIR / "fake")
    if not real_files or not fake_files:
        raise FileNotFoundError("Expected images in data/processed/real and data/processed/fake.")

    paths, labels = balanced_sample(real_files, fake_files, args.per_class)
    model = tf.keras.models.load_model(str(MODEL_PATH))
    probabilities = predict_paths(model, paths, args.batch_size)
    threshold_info = find_best_threshold(labels, probabilities)
    threshold_info["sample_count_per_class"] = int(len(paths) // 2)
    threshold_info["model_path"] = str(MODEL_PATH)

    args.output.write_text(json.dumps(threshold_info, indent=2), encoding="utf-8")
    print(json.dumps(threshold_info, indent=2))


if __name__ == "__main__":
    main()
