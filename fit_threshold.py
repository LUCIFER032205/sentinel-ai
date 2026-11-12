"""Fit the threshold on the VALIDATION split under the stated preference:
upset as few real users as possible, while still catching >= 90% of AI images.

The holdout plays no part. It is only used afterwards to check the result.
"""
import sys, json
sys.path.insert(0, r"E:\ML mini-project")
import numpy as np, tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from calibrate_thresholds import find_threshold_min_recall
from src.ai_image_detector.config import PROCESSED_DATA_DIR, SEED, MODEL_PATH, THRESHOLD_PATH
from src.ai_image_detector.data import list_dataset, paths_to_dataset
from src.ai_image_detector.inference import load_trained_model

MIN_RECALL = 0.90  # "AI read as real" no worse than ~10%

paths, y = list_dataset(PROCESSED_DATA_DIR)
_, p_temp, _, y_temp = train_test_split(paths, y, test_size=0.3, random_state=SEED, stratify=y)
p_val, _, y_val, _ = train_test_split(p_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)
print(f"validation split: {len(p_val)} images", flush=True)

model = load_trained_model()
ds = paths_to_dataset(p_val, y_val, 32)
plain = model.predict(ds, verbose=0).ravel()
flipped = model.predict(ds.map(lambda x, l: (tf.image.flip_left_right(x), l)), verbose=0).ravel()
probs = (plain + flipped) / 2.0

info = find_threshold_min_recall(y_val, probs, MIN_RECALL)
info["sample_count_per_class"] = int(min((y_val == 0).sum(), (y_val == 1).sum()))
info["model_path"] = str(MODEL_PATH)
info["tuning_note"] = (
    "Fitted on the validation split only, flip-averaged, with the objective "
    "'maximise real-photo specificity subject to AI-recall >= 0.90'. That encodes the "
    "stated preference that a real photo wrongly flagged as AI is the more costly error, "
    "with missed AI images acceptable up to about 10%. Plain accuracy was the wrong "
    "objective because it treats both mistakes as equally expensive. The holdout sets "
    "played no part in choosing this value."
)

print()
print(json.dumps({k: v for k, v in info.items() if k != "tuning_note"}, indent=2))

print()
print("  what the constraint picked, on validation:")
print(f"    threshold            {info['threshold']}")
print(f"    AI-recall            {info.get('validation_ai_recall', float('nan'))*100:.1f}%  (target >= {MIN_RECALL*100:.0f}%)")
print(f"    real-specificity     {info.get('validation_real_specificity', float('nan'))*100:.1f}%")
print(f"    -> AI missed         {(1-info.get('validation_ai_recall', float('nan')))*100:.1f}%")
print(f"    -> real flagged      {(1-info.get('validation_real_specificity', float('nan')))*100:.1f}%")

Path(THRESHOLD_PATH).write_text(json.dumps(info, indent=2), encoding="utf-8")
print(f"\nwritten to {THRESHOLD_PATH}")
