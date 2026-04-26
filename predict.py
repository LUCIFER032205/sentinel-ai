from __future__ import annotations

import argparse
from pathlib import Path

from src.ai_image_detector.inference import load_trained_model, predict_image_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether an image is AI-generated.")
    parser.add_argument("image", type=Path, help="Path to the image file")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = load_trained_model()
    result = predict_image_bytes(model, args.image.read_bytes())

    print(f"Prediction: {result.label}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"AI probability: {result.ai_probability:.2%}")


if __name__ == "__main__":
    main()

