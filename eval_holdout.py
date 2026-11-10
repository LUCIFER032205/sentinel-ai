"""Score a labelled holdout folder with the live served configuration.

This is the only committed code that evaluates `data/holdout_*`. It reproduces the
exact scoring path the API uses (flip-averaged, threshold from
`artifacts/thresholds.json`) and can apply the robustness perturbations that
accuracy claims are measured against.

    python eval_holdout.py data/holdout_newdb
    python eval_holdout.py data/holdout_newdb --noise 16
    python eval_holdout.py data/holdout_newdb --jpeg 30 --scale 0.5
    python eval_holdout.py data/holdout_newdb --limit 300 --check 0.80

The folder must contain `real/` and `fake/` subfolders.

UNITS: --noise sigma and --jpeg quality are applied to the decoded 0-255 BGR image
BEFORE preprocessing and BEFORE the 224x224 resize, which is how the published
robustness numbers were measured. Noise applied after preprocessing is in [-1, 1]
units and is ~127.5x smaller for the same number - see docs/architecture-improvement-plan.md.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from src.ai_image_detector.config import FAKE_CLASS_NAME, REAL_CLASS_NAME
from src.ai_image_detector.data import iter_image_files, preprocess_image
from src.ai_image_detector.inference import load_calibration, load_trained_model

# Perturbations are applied in this order: geometry, then compression, then noise.
# Noise is applied last so that JPEG does not partially smooth it away.


def apply_scale(image: np.ndarray, factor: float) -> np.ndarray:
    """Downscale then restore size, to mimic a resized/re-uploaded image."""
    height, width = image.shape[:2]
    small = cv2.resize(
        image,
        (max(1, int(width * factor)), max(1, int(height * factor))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else image


def apply_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def collect(folder: Path, limit: int | None, seed: int) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for class_name, label in ((REAL_CLASS_NAME, 0), (FAKE_CLASS_NAME, 1)):
        class_dir = folder / class_name
        if not class_dir.is_dir():
            raise SystemExit(
                f"Expected '{class_name}/' inside {folder}. "
                f"Pass a folder containing '{REAL_CLASS_NAME}/' and '{FAKE_CLASS_NAME}/'."
            )
        files = list(iter_image_files(class_dir))
        if not files:
            raise SystemExit(f"No readable images in {class_dir}.")
        if limit is not None and limit < len(files):
            files = random.Random(seed).sample(files, limit)
        items += [(path, label) for path in files]
    return items


def score(model, items, args) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, labels).

    Two distinct noise stages, deliberately separate:
      --noise        the ATTACK. Applied to the image, simulating what an adversary
                     or a noisy camera does before the detector ever sees it.
      --score-noise  a DEFENCE. A grain floor the detector itself adds to every
                     input, so both classes arrive equally grainy and the relative
                     noise-floor cue is destroyed. Applied after the attack.
      --tta-noise    a DEFENCE. Score the image clean and grainy, average both.

    Flip-averaging is always on, matching inference.py. --tta-noise doubles the
    variants per image from 2 to 4.
    """
    rng = np.random.default_rng(args.seed)
    per_image = 4 if args.tta_noise else 2
    probabilities: list[float] = []
    labels: list[int] = []
    batch: list[np.ndarray] = []
    pending: list[int] = []
    skipped = 0

    def flush() -> None:
        if not batch:
            return
        preds = (
            model.predict(np.stack(batch), verbose=0)
            .ravel()
            .reshape(-1, per_image)
            .mean(1)
        )
        probabilities.extend(float(value) for value in preds)
        labels.extend(pending)
        batch.clear()
        pending.clear()

    for index, (path, label) in enumerate(items):
        image = cv2.imread(str(path))
        if image is None:
            skipped += 1
            continue
        # --- attack stages ---
        if args.scale is not None:
            image = apply_scale(image, args.scale)
        if args.jpeg is not None:
            image = apply_jpeg(image, args.jpeg)
        if args.noise:
            image = apply_noise(image, args.noise, rng)
        # --- defence stage: grain floor added by the detector itself ---
        if args.score_noise:
            image = apply_noise(image, args.score_noise, rng)

        prepared = preprocess_image(image)
        batch.append(prepared)
        batch.append(np.flip(prepared, axis=1))
        if args.tta_noise:
            noisy = preprocess_image(apply_noise(image, args.tta_noise, rng))
            batch.append(noisy)
            batch.append(np.flip(noisy, axis=1))
        pending.append(label)

        if len(batch) >= args.batch_size * per_image:
            flush()
        if args.progress and (index + 1) % 250 == 0:
            print(f"  ...{index + 1}/{len(items)}", file=sys.stderr, flush=True)
    flush()

    if skipped:
        print(f"warning: skipped {skipped} unreadable file(s)", file=sys.stderr)
    return np.array(probabilities), np.array(labels)


def interval(rate: float, n: int) -> float:
    """Half-width of the 95% confidence interval, in percentage points."""
    if n == 0:
        return float("nan")
    return 1.96 * math.sqrt(rate * (1.0 - rate) / n) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a labelled holdout folder with the served configuration.",
    )
    parser.add_argument("folder", type=Path, help="Folder containing real/ and fake/")
    parser.add_argument("--noise", type=float, default=None,
                        help="ATTACK: Gaussian sigma in 0-255 pixel units, applied before preprocessing")
    parser.add_argument("--score-noise", type=float, default=None,
                        help="DEFENCE (grain-floor normalisation): sigma the detector adds to "
                             "every input after the attack, so both classes arrive equally "
                             "grainy. Combine with --noise to test it against the attack.")
    parser.add_argument("--tta-noise", type=float, default=None,
                        help="DEFENCE (noise-level TTA): also score a grainy copy at this sigma "
                             "and average with the clean score. Doubles forward passes to 4.")
    parser.add_argument("--jpeg", type=int, default=None, help="Re-encode at this JPEG quality (1-100)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Downscale by this factor then restore size, e.g. 0.5")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the threshold from artifacts/thresholds.json")
    parser.add_argument("--limit", type=int, default=None, help="Images per class (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling and noise")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--check", type=float, default=None,
                        help="Exit non-zero if ACCURACY falls below this fraction, e.g. 0.80")
    parser.add_argument("--check-recall", type=float, default=None,
                        help="Exit non-zero if AI-RECALL falls below this fraction, e.g. 0.70. "
                             "Use this for robustness gates: under --noise the metric that "
                             "matters is recall, because noise pushes AI images toward 'Real' "
                             "while real-specificity rises.")
    args = parser.parse_args()

    if args.jpeg is not None and not 1 <= args.jpeg <= 100:
        raise SystemExit("--jpeg must be between 1 and 100.")
    if args.scale is not None and not 0 < args.scale <= 1:
        raise SystemExit("--scale must be greater than 0 and at most 1.")

    threshold = args.threshold if args.threshold is not None else load_calibration().threshold
    items = collect(args.folder, args.limit, args.seed)
    model = load_trained_model()
    probabilities, labels = score(model, items, args)
    if probabilities.size == 0:
        raise SystemExit("No images could be scored.")

    predicted_ai = probabilities >= threshold
    is_ai = labels == 1
    accuracy = float((predicted_ai == is_ai).mean())
    recall = float(predicted_ai[is_ai].mean()) if is_ai.any() else float("nan")
    specificity = float((~predicted_ai[~is_ai]).mean()) if (~is_ai).any() else float("nan")

    attacks = [
        f"{name}={value:g}" for name, value in
        (("scale", args.scale), ("jpeg", args.jpeg), ("noise", args.noise))
        if value
    ]
    defences = [
        f"{name}={value:g}" for name, value in
        (("score-noise", args.score_noise), ("tta-noise", args.tta_noise))
        if value
    ]
    print(f"folder     {args.folder}")
    print(f"images     {probabilities.size}  ({int((~is_ai).sum())} real / {int(is_ai.sum())} AI)")
    print(f"threshold  {threshold:.3f}" + ("" if args.threshold is None else "  (overridden)"))
    print(f"attack     {', '.join(attacks) if attacks else 'none'}")
    print(f"defence    {', '.join(defences) if defences else 'none'}")
    print()
    print(f"  accuracy          {accuracy*100:5.1f}%  +/-{interval(accuracy, probabilities.size):.1f}")
    print(f"  AI-recall         {recall*100:5.1f}%  +/-{interval(recall, int(is_ai.sum())):.1f}")
    print(f"  real-specificity  {specificity*100:5.1f}%  +/-{interval(specificity, int((~is_ai).sum())):.1f}")
    print()
    print("  intervals are 95% and assume independent images; bursts of near-duplicate")
    print("  photos make the true interval wider than shown.")

    gates = [
        ("accuracy", accuracy, args.check, "--check"),
        ("AI-recall", recall, args.check_recall, "--check-recall"),
    ]
    failed = False
    for name, value, floor, flag in gates:
        if floor is None:
            continue
        if value < floor:
            print(f"\nFAIL: {name} {value:.3f} is below {flag} {floor:.3f}")
            failed = True
        else:
            print(f"\nPASS: {name} {value:.3f} meets {flag} {floor:.3f}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
