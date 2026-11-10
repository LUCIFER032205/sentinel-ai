from __future__ import annotations

import json
import os

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
from src.ai_image_detector.data import list_dataset, paths_to_dataset
from src.ai_image_detector.model import build_model, unfreeze_for_fine_tuning


def make_datasets(
    paths: list,
    y: np.ndarray,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
    """Split by PATH, not by decoded image.

    The corpus is 48,707 images; decoding it eagerly needs ~29 GB resident.
    Everything below streams from disk instead, so only labels stay in memory.
    """
    paths_train, paths_temp, y_train, y_temp = train_test_split(
        paths,
        y,
        test_size=0.3,
        random_state=SEED,
        stratify=y,
    )
    paths_val, paths_test, y_val, y_test = train_test_split(
        paths_temp,
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
            # value_range MUST match the tensor range. data.py applies
            # preprocess_input at load time, so images arrive in [-1, 1]; the
            # Keras default of (0, 255) clips every negative value to 0.0,
            # which blanks roughly 55% of each augmented image.
            tf.keras.layers.RandomContrast(0.1, value_range=(-1, 1)),
        ]
    )

    cache_dir = os.getenv("TRAIN_CACHE_DIR")

    def cache_for(name: str) -> str | None:
        # Opt-in: the cache stores preprocessed float32 tensors, ~600 KB per
        # image, so caching the full corpus costs tens of GB of disk.
        return None if cache_dir is None else os.path.join(cache_dir, name)

    train_ds = paths_to_dataset(
        paths_train, y_train, batch_size, shuffle=True, cache_path=cache_for("train")
    )
    train_ds = train_ds.map(
        lambda images, labels: (augmenter(images, training=True), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = paths_to_dataset(paths_val, y_val, batch_size, cache_path=cache_for("val"))
    # Unaugmented, unshuffled passes used for threshold fitting and final metrics.
    val_eval_ds = paths_to_dataset(paths_val, y_val, batch_size)
    test_eval_ds = paths_to_dataset(paths_test, y_test, batch_size)

    return train_ds, val_ds, val_eval_ds, test_eval_ds, y_val, y_test


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
    dataset: tf.data.Dataset,
    flip_average: bool = True,
) -> np.ndarray:
    """Score a dataset the way the server scores an upload.

    flip_average defaults to True because inference.py averages the image and
    its horizontal mirror. Fitting the decision threshold on single-pass scores
    while serving flip-averaged ones means tuning on a different distribution
    from the one the threshold gates.
    """
    plain = model.predict(dataset, verbose=0).ravel()
    if not flip_average:
        return plain
    flipped = model.predict(
        dataset.map(
            lambda images, labels: (tf.image.flip_left_right(images), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        ),
        verbose=0,
    ).ravel()
    return (plain + flipped) / 2.0


def evaluate_and_save_metrics(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    y_test: np.ndarray,
    threshold_info: dict[str, float],
) -> None:
    threshold = float(threshold_info["threshold"])
    results = model.evaluate(test_ds, verbose=0, return_dict=True)
    predictions = predict_probabilities(model, test_ds)
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

    paths, y = list_dataset(PROCESSED_DATA_DIR)
    print(f"Found {len(paths)} images in {PROCESSED_DATA_DIR} (streaming, not preloaded).")
    train_ds, val_ds, val_eval_ds, test_eval_ds, y_val, y_test = make_datasets(
        paths, y, batch_size=batch_size
    )
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
    val_predictions = predict_probabilities(model, val_eval_ds)
    threshold_info = find_best_threshold(y_val, val_predictions)
    THRESHOLD_PATH.write_text(json.dumps(threshold_info, indent=2), encoding="utf-8")

    save_training_plot(combine_histories(frozen_history, finetune_history))
    evaluate_and_save_metrics(model, test_eval_ds, y_test, threshold_info)

    print(f"Training complete. Model saved to: {MODEL_PATH}")
    print(f"Threshold config saved to: {THRESHOLD_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Training plot saved to: {TRAINING_PLOT_PATH}")


if __name__ == "__main__":
    main()
