"""Memory-efficient training using TensorFlow data generators."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.ai_image_detector.config import (
    ARTIFACTS_DIR,
    IMAGE_SIZE,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
    SEED,
    THRESHOLD_PATH,
    TRAINING_PLOT_PATH,
)
from src.ai_image_detector.model import build_model, unfreeze_for_fine_tuning


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def create_dataset(
    data_dir: Path,
    batch_size: int,
    augment: bool = False,
    shuffle: bool = False,
    subset: str | None = None,
    validation_split: float = 0.0,
    seed: int = SEED,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset from directory with streaming."""

    def parse_image(file_path, label):
        # Read and decode image
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32)
        # MobileNetV2 preprocessing
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label

    def augment_image(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.clip_by_value(image, -1.0, 1.0)  # Keep in MobileNetV2 range
        return image, label

    # Get file paths and labels
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"

    real_files = [str(p) for p in real_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    fake_files = [str(p) for p in fake_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    file_paths = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)

    print(f"Found {len(real_files)} real images")
    print(f"Found {len(fake_files)} fake images")
    print(f"Total: {len(file_paths)} images")

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), 10000), seed=seed)

    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(file_paths)


def split_dataset(
    data_dir: Path,
    batch_size: int,
    validation_split: float = 0.3,
    test_split: float = 0.15,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, int]:
    """Split dataset into train/val/test."""

    def parse_image(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label

    def augment_image(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.clip_by_value(image, -1.0, 1.0)  # Keep in MobileNetV2 range
        return image, label

    # Get file paths and labels
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"

    real_files = sorted([str(p) for p in real_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    fake_files = sorted([str(p) for p in fake_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])

    # Balance and shuffle
    np.random.seed(seed)
    min_count = min(len(real_files), len(fake_files))
    real_files = np.random.choice(real_files, min_count, replace=False).tolist()
    fake_files = np.random.choice(fake_files, min_count, replace=False).tolist()

    file_paths = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)

    # Shuffle together
    indices = np.random.permutation(len(file_paths))
    file_paths = [file_paths[i] for i in indices]
    labels = [labels[i] for i in indices]

    # Calculate split indices
    total = len(file_paths)
    test_count = int(total * test_split)
    val_count = int(total * validation_split)
    train_count = total - val_count - test_count

    train_files = file_paths[:train_count]
    train_labels = labels[:train_count]
    val_files = file_paths[train_count:train_count + val_count]
    val_labels = labels[train_count:train_count + val_count]
    test_files = file_paths[train_count + val_count:]
    test_labels = labels[train_count + val_count:]

    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.shuffle(buffer_size=min(len(train_files), 5000), seed=seed)
    train_ds = train_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_ds = test_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, len(val_files), len(test_files)


def save_training_plot(history) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(TRAINING_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved training plot to {TRAINING_PLOT_PATH}")


def evaluate_model(model, test_ds, test_count, threshold=0.5):
    """Evaluate model on test set."""
    # Collect predictions
    y_true = []
    y_pred = []
    y_probs = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_probs.extend(probs.flatten().tolist())
        y_pred.extend((probs >= threshold).flatten().astype(int).tolist())
        y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=["real", "fake"], output_dict=True, zero_division=0)

    metrics = {
        "test_accuracy": float(acc),
        "test_f1_fake": float(f1),
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "classification_report": report,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test F1 (fake): {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return metrics


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found at {PROCESSED_DATA_DIR}")

    batch_size = get_env_int("BATCH_SIZE", 32)
    frozen_epochs = get_env_int("FROZEN_EPOCHS", 10)
    finetune_epochs = get_env_int("FINETUNE_EPOCHS", 15)

    print("Creating datasets...")
    train_ds, val_ds, test_ds, val_count, test_count = split_dataset(
        PROCESSED_DATA_DIR, batch_size=batch_size
    )

    print(f"\nBuilding model...")
    model = build_model()

    # Stage 1: Train with frozen base
    print(f"\n{'='*50}")
    print("Stage 1: Training with frozen base")
    print(f"{'='*50}")

    callbacks_frozen = [
        EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_auc", mode="max", save_best_only=True),
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=frozen_epochs,
        callbacks=callbacks_frozen,
        verbose=1,
    )

    # Stage 2: Fine-tune
    print(f"\n{'='*50}")
    print("Stage 2: Fine-tuning")
    print(f"{'='*50}")

    model = tf.keras.models.load_model(str(MODEL_PATH))
    unfreeze_for_fine_tuning(model, trainable_layers=45)

    callbacks_finetune = [
        EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_auc", mode="max", save_best_only=True),
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=finetune_epochs,
        callbacks=callbacks_finetune,
        verbose=1,
    )

    # Evaluate
    print(f"\n{'='*50}")
    print("Final Evaluation")
    print(f"{'='*50}")

    model = tf.keras.models.load_model(str(MODEL_PATH))
    evaluate_model(model, test_ds, test_count)

    # Save plots
    class CombinedHistory:
        def __init__(self, h1, h2):
            self.history = {}
            for key in h1.history:
                self.history[key] = h1.history[key] + h2.history[key]

    save_training_plot(CombinedHistory(history1, history2))

    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
