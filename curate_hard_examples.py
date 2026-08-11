from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.ai_image_detector.data import VALID_SUFFIXES, iter_image_files, preprocess_image


def iter_images(folder: Path) -> list[Path]:
    return list(iter_image_files(folder))


def score_images(model: tf.keras.Model, files: list[Path]) -> list[tuple[Path, float]]:
    scored: list[tuple[Path, float]] = []
    for file_path in files:
        image = cv2.imread(str(file_path))
        if image is None:
            continue
        x = preprocess_image(image)
        pred = float(model.predict(np.expand_dims(x, axis=0), verbose=0)[0][0])
        scored.append((file_path, pred))
    return scored


def clear_previous_curated(processed_real: Path, processed_fake: Path) -> None:
    for p in processed_real.glob("hard_real_*"):
        p.unlink(missing_ok=True)
    for p in processed_fake.glob("hard_fake_*"):
        p.unlink(missing_ok=True)


def copy_selected(
    selected_real: list[tuple[Path, float]],
    selected_fake: list[tuple[Path, float]],
    processed_real: Path,
    processed_fake: Path,
) -> None:
    for i, (src, score) in enumerate(selected_real):
        dst = processed_real / f"hard_real_{i:04d}_{score:.4f}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    for i, (src, score) in enumerate(selected_fake):
        dst = processed_fake / f"hard_fake_{i:04d}_{score:.4f}{src.suffix.lower()}"
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard holdout examples into training.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"E:\ML mini-project"),
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=300,
        help="How many hard examples to add per class",
    )
    args = parser.parse_args()

    project_root = args.project_root
    model_path = project_root / "artifacts" / "ai_image_detector.keras"
    holdout_real = project_root / "data" / "holdout_newdb" / "real"
    holdout_fake = project_root / "data" / "holdout_newdb" / "fake"
    processed_real = project_root / "data" / "processed" / "real"
    processed_fake = project_root / "data" / "processed" / "fake"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    real_scored = score_images(model, iter_images(holdout_real))
    fake_scored = score_images(model, iter_images(holdout_fake))

    # Hard real: model thinks fake (highest AI probability).
    hard_real = sorted(real_scored, key=lambda x: x[1], reverse=True)[: args.per_class]
    # Hard fake: model thinks real (lowest AI probability).
    hard_fake = sorted(fake_scored, key=lambda x: x[1])[: args.per_class]

    clear_previous_curated(processed_real, processed_fake)
    copy_selected(hard_real, hard_fake, processed_real, processed_fake)

    print(f"Added hard real examples: {len(hard_real)}")
    print(f"Added hard fake examples: {len(hard_fake)}")
    if hard_real:
        print(f"Hard real score range: {hard_real[-1][1]:.4f} -> {hard_real[0][1]:.4f}")
    if hard_fake:
        print(f"Hard fake score range: {hard_fake[0][1]:.4f} -> {hard_fake[-1][1]:.4f}")


if __name__ == "__main__":
    main()

