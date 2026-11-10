from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .config import CLASS_NAMES, FAKE_CLASS_NAME, IMAGE_SIZE, REAL_CLASS_NAME, SEED


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


def list_dataset(data_dir: Path, validate: bool = True) -> tuple[list[Path], np.ndarray]:
    """List image paths and labels WITHOUT decoding them.

    load_dataset() below decodes every image into one resident float32 array:
    at 224x224x3 that is ~600 KB each, so the 48,707-image corpus needs ~29 GB
    before batching. This returns paths instead, so a tf.data pipeline can
    stream and decode per batch.
    """
    paths: list[Path] = []
    labels: list[int] = []
    unreadable: list[Path] = []

    for label_name in CLASS_NAMES:
        class_dir = data_dir / label_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected class folder '{label_name}' inside {data_dir}."
            )
        label = 0 if label_name == REAL_CLASS_NAME else 1
        for image_path in iter_image_files(class_dir):
            # Header-only readability check: ~0.4 ms per file against ~5 ms for a
            # full decode. Doing it here keeps the tf.data cardinality known,
            # which Keras needs to compute steps_per_epoch.
            if validate and not cv2.haveImageReader(str(image_path)):
                unreadable.append(image_path)
                continue
            paths.append(image_path)
            labels.append(label)

    if unreadable:
        warnings.warn(
            f"Skipped {len(unreadable)} unreadable file(s) under {data_dir}, "
            f"first few: {[p.name for p in unreadable[:3]]}",
            RuntimeWarning,
            stacklevel=2,
        )

    if not paths:
        raise ValueError(
            f"No images found in {data_dir}. Put files under '{REAL_CLASS_NAME}/' "
            f"and '{FAKE_CLASS_NAME}/'."
        )
    return paths, np.array(labels)


def decode_and_preprocess(path_bytes: bytes) -> tuple[np.ndarray, bool]:
    """Decode one image with cv2 and run the exact serving preprocessing.

    Deliberately uses cv2 rather than tf.io.decode_image + tf.image.resize:
    inference.py decodes with cv2 and resizes with cv2.INTER_LINEAR, and a
    different resize kernel here would train the model on a subtly different
    distribution from the one it serves.

    Returns (image, ok). Unreadable files come back as ok=False so the caller
    can drop them rather than train on a blank frame.
    """
    image = cv2.imread(path_bytes.decode("utf-8"))
    if image is None:
        return np.zeros((*IMAGE_SIZE, 3), dtype="float32"), np.bool_(False)
    return preprocess_image(image), np.bool_(True)


def paths_to_dataset(
    paths: list[Path],
    labels: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool = False,
    cache_path: str | None = None,
    seed: int = SEED,
) -> tf.data.Dataset:
    """Stream images from disk instead of holding the corpus in memory.

    Unreadable files are dropped, not zero-filled. Set cache_path to trade disk
    for repeated decode cost - note the cache stores preprocessed float32
    tensors, so it is roughly 600 KB per image and can be larger than the
    source images.
    """
    path_strings = [str(path) for path in paths]
    dataset = tf.data.Dataset.from_tensor_slices(
        (path_strings, labels.astype("float32"))
    )

    def _decode(path, label):
        image, ok = tf.numpy_function(
            decode_and_preprocess, [path], (tf.float32, tf.bool)
        )
        image.set_shape((*IMAGE_SIZE, 3))
        ok.set_shape(())
        return image, label, ok

    dataset = dataset.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda image, label, ok: ok)
    dataset = dataset.map(
        lambda image, label, ok: (image, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # filter() erases cardinality, and Keras then cannot compute steps_per_epoch
    # (it warns "input ran out of data" and treats the epoch as unbounded).
    # list_dataset already header-checked every path, so nothing should be
    # dropped here; assert that, so a surprise drop fails loudly instead of
    # silently shortening every epoch.
    dataset = dataset.apply(
        tf.data.experimental.assert_cardinality(len(path_strings))
    )

    if cache_path is not None:
        dataset = dataset.cache(cache_path)
    if shuffle:
        dataset = dataset.shuffle(
            min(len(path_strings), 4096), seed=seed, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


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

