from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from src.ai_image_detector.config import IMAGE_SIZE, MODEL_PATH
from src.ai_image_detector.data import load_image_for_inference, preprocess_image
from src.ai_image_detector.inference import load_calibration


def resize_with_padding(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(rgb)
    image.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", IMAGE_SIZE, (128, 128, 128))
    left = (IMAGE_SIZE[0] - image.width) // 2
    top = (IMAGE_SIZE[1] - image.height) // 2
    canvas.paste(image, (left, top))
    return preprocess_input(np.asarray(canvas).astype("float32"))


def predict(model: tf.keras.Model, image: np.ndarray) -> float:
    batch = np.stack([image, np.flip(image, axis=1)], axis=0)
    return float(np.mean(model.predict(batch, verbose=0).ravel()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw model scores for one image.")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    model = tf.keras.models.load_model(str(MODEL_PATH))
    calibration = load_calibration()
    image_bytes = args.image.read_bytes()

    current = load_image_for_inference(image_bytes)
    pil = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    rgb = np.asarray(pil)

    variants = {
        "current_loader": current,
        "exif_transposed_stretched": preprocess_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
        "exif_transposed_padded": resize_with_padding(rgb),
        "rotated_90_padded": resize_with_padding(np.rot90(rgb, 1)),
        "rotated_180_padded": resize_with_padding(np.rot90(rgb, 2)),
        "rotated_270_padded": resize_with_padding(np.rot90(rgb, 3)),
    }

    print(f"Threshold: {calibration.threshold:.2f}")
    print(f"Uncertain band: {calibration.uncertain_low:.2f}-{calibration.uncertain_high:.2f}")
    for name, image in variants.items():
        probability = predict(model, image)
        if calibration.uncertain_low < probability < calibration.uncertain_high:
            label = "Uncertain"
        else:
            label = "AI-generated" if probability >= calibration.threshold else "Real"
        print(f"{name}: {probability:.4f} -> {label}")


if __name__ == "__main__":
    main()
