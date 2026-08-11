from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from calibrate_thresholds import find_best_threshold
from src.ai_image_detector.config import (
    ARTIFACTS_DIR,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
    SEED,
    THRESHOLD_PATH,
    TRAINING_PLOT_PATH,
    get_env_int,
)
from src.ai_image_detector.data import load_dataset
from src.ai_image_detector.model import build_model, unfreeze_for_fine_tuning


def make_datasets(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=SEED,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=y_temp,
    )

    augmenter = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ]
    )

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(len(x_train), seed=SEED)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.map(
        lambda images, labels: (augmenter(images, training=True), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, x_val, y_val, x_test, y_test


def combine_histories(
    first_history: tf.keras.callbacks.History,
    second_history: tf.keras.callbacks.History,
) -> dict[str, list[float]]:
    combined: dict[str, list[float]] = {}
    for history in (first_history.history, second_history.history):
        for key, values in history.items():
            combined.setdefault(key, []).extend(values)
    return combined


def save_training_plot(history_data: dict[str, list[float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_data["accuracy"], label="Train")
    axes[0].plot(history_data["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history_data["loss"], label="Train")
    axes[1].plot(history_data["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(TRAINING_PLOT_PATH, dpi=150)
    plt.close(fig)


def predict_probabilities(
    model: tf.keras.Model,
    x: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return model.predict(dataset, verbose=0).ravel()


def evaluate_and_save_metrics(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold_info: dict[str, float],
) -> None:
    threshold = float(threshold_info["threshold"])
    results = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    predictions = predict_probabilities(model, x_test, batch_size=32)
    predicted_classes = (predictions >= threshold).astype(int)
    predicted_classes_default = (predictions >= 0.5).astype(int)

    report = classification_report(
        y_test,
        predicted_classes,
        target_names=["real", "fake"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predicted_classes).tolist()

    metrics = {
        "evaluation": {key: float(value) for key, value in results.items()},
        "thresholding": {
            "default_threshold": 0.5,
            "calibrated_threshold": threshold,
            "test_accuracy_default": float(accuracy_score(y_test, predicted_classes_default)),
            "test_accuracy_calibrated": float(accuracy_score(y_test, predicted_classes)),
            "test_f1_fake_calibrated": float(f1_score(y_test, predicted_classes, pos_label=1)),
        },
        "confusion_matrix": matrix,
        "classification_report": report,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PROCESSED_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found at {PROCESSED_DATA_DIR}. "
            "Create data/processed/real and data/processed/fake and put images there."
        )

    batch_size = get_env_int("BATCH_SIZE", 32)
    frozen_epochs = get_env_int("FROZEN_EPOCHS", 10)
    finetune_epochs = get_env_int("FINETUNE_EPOCHS", 18)

    x, y, _ = load_dataset(PROCESSED_DATA_DIR)
    train_ds, val_ds, x_val, y_val, x_test, y_test = make_datasets(x, y, batch_size=batch_size)
    model = build_model()

    callbacks_frozen = [
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    frozen_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=frozen_epochs,
        callbacks=callbacks_frozen,
        verbose=1,
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    unfreeze_for_fine_tuning(model, trainable_layers=45)

    callbacks_finetune = [
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    finetune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=finetune_epochs,
        callbacks=callbacks_finetune,
        verbose=1,
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    val_predictions = predict_probabilities(model, x_val, batch_size=32)
    threshold_info = find_best_threshold(y_val, val_predictions)
    THRESHOLD_PATH.write_text(json.dumps(threshold_info, indent=2), encoding="utf-8")

    save_training_plot(combine_histories(frozen_history, finetune_history))
    evaluate_and_save_metrics(model, x_test, y_test, threshold_info)

    print(f"Training complete. Model saved to: {MODEL_PATH}")
    print(f"Threshold config saved to: {THRESHOLD_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Training plot saved to: {TRAINING_PLOT_PATH}")


if __name__ == "__main__":
    main()
