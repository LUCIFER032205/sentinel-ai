"""Fast training with balanced 10k subset."""
from __future__ import annotations

import json
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
    get_env_int,
)
from src.ai_image_detector.data import VALID_SUFFIXES
from src.ai_image_detector.model import build_model, unfreeze_for_fine_tuning
from train import combine_histories, save_training_plot


def split_dataset(
    data_dir: Path,
    batch_size: int,
    max_samples: int = 5000,  # Limit per class
    validation_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = SEED,
):
    """Split dataset into train/val/test with limited samples for fast training."""

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
        image = tf.clip_by_value(image, -1.0, 1.0)
        return image, label

    # Get file paths and labels
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"

    real_files = sorted([str(p) for p in real_dir.glob("*") if p.suffix.lower() in VALID_SUFFIXES])
    fake_files = sorted([str(p) for p in fake_dir.glob("*") if p.suffix.lower() in VALID_SUFFIXES])

    # Balance and limit. Keep phone_real_* examples in every fast run so the
    # model learns real gallery/portrait/selfie artifacts instead of treating
    # them as AI clues.
    np.random.seed(seed)
    min_count = min(len(real_files), len(fake_files), max_samples)

    phone_real_files = [path for path in real_files if Path(path).name.startswith("phone_real_")]
    other_real_files = [path for path in real_files if not Path(path).name.startswith("phone_real_")]

    phone_count = min(len(phone_real_files), min_count)
    remaining_real_count = min_count - phone_count
    if phone_count:
        phone_indices = np.random.choice(len(phone_real_files), phone_count, replace=False)
        phone_sample = [phone_real_files[i] for i in phone_indices]
    else:
        phone_sample = []

    if remaining_real_count:
        real_indices = np.random.choice(len(other_real_files), remaining_real_count, replace=False)
        other_real_sample = [other_real_files[i] for i in real_indices]
    else:
        other_real_sample = []

    fake_indices = np.random.choice(len(fake_files), min_count, replace=False)

    real_sample = phone_sample + other_real_sample
    fake_sample = [fake_files[i] for i in fake_indices]

    file_paths = real_sample + fake_sample
    labels = [0] * len(real_sample) + [1] * len(fake_sample)

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

    print(f"Using {min_count} images per class (limited from {len(real_files)} available)")
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.shuffle(buffer_size=min(len(train_files), 2000), seed=seed)
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


def evaluate_model(model, test_ds, test_count, threshold=0.5):
    """Evaluate model on test set."""
    y_true, y_pred, y_probs = [], [], []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_probs.extend(probs.flatten().tolist())
        y_pred.extend((probs >= threshold).flatten().astype(int).tolist())
        y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

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
    frozen_epochs = get_env_int("FROZEN_EPOCHS", 8)  # Reduced for speed
    finetune_epochs = get_env_int("FINETUNE_EPOCHS", 10)  # Reduced for speed
    max_samples = 5000  # Limit to 5k per class = 10k total

    print("Creating datasets with 10k balanced subset...")
    train_ds, val_ds, test_ds, val_count, test_count = split_dataset(
        PROCESSED_DATA_DIR, batch_size=batch_size, max_samples=max_samples
    )

    print(f"\nBuilding model...")
    model = build_model()

    # Stage 1: Train with frozen base
    print(f"\n{'='*50}")
    print("Stage 1: Training with frozen base")
    print(f"{'='*50}")

    callbacks_frozen = [
        EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
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
        EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
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

    save_training_plot(combine_histories(history1, history2))

    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
