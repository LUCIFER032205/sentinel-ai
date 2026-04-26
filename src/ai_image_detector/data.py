from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .config import CLASS_NAMES, FAKE_CLASS_NAME, IMAGE_SIZE, REAL_CLASS_NAME


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_image_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
            yield path


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image_bgr, IMAGE_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = rgb.astype("float32")
    return preprocess_input(array)


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []

    for label_name in CLASS_NAMES:
        class_dir = data_dir / label_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected class folder '{label_name}' inside {data_dir}."
            )

        label = 0 if label_name == REAL_CLASS_NAME else 1
        for image_path in iter_image_files(class_dir):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            images.append(preprocess_image(image))
            labels.append(label)
            paths.append(image_path)

    if not images:
        raise ValueError(
            f"No images found in {data_dir}. Put files under '{REAL_CLASS_NAME}/' "
            f"and '{FAKE_CLASS_NAME}/'."
        )

    return np.array(images), np.array(labels), paths


def load_image_for_inference(image_bytes: bytes) -> np.ndarray:
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    return preprocess_image(image)

