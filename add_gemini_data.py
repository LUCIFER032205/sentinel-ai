from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

from src.ai_image_detector.config import PROCESSED_DATA_DIR

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Folder not found: {directory}")
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    ]
    if not files:
        raise ValueError(f"No valid images found in: {directory}")
    return sorted(files)


def save_as_png(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="PNG")


def remove_prefix_files(directory: Path, prefix: str) -> int:
    removed = 0
    for path in directory.glob(f"{prefix}_*.png"):
        path.unlink()
        removed += 1
    return removed


def copy_subset(
    files: list[Path],
    destination_dir: Path,
    prefix: str,
    count: int,
    seed: int,
) -> int:
    rng = random.Random(seed)
    candidates = list(files)
    rng.shuffle(candidates)
    selected = candidates[:count]

    written = 0
    for index, source in enumerate(selected):
        destination = destination_dir / f"{prefix}_{index:05d}_{source.stem}.png"
        save_as_png(source, destination)
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add Gemini-focused fake images plus matched real images into training folders."
    )
    parser.add_argument(
        "--gemini-dir",
        type=Path,
        required=True,
        help="Folder containing Gemini-generated images (fake).",
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        required=True,
        help="Folder containing real photos for balancing.",
    )
    parser.add_argument(
        "--gemini-count",
        type=int,
        default=700,
        help="How many Gemini fake images to add.",
    )
    parser.add_argument(
        "--real-count",
        type=int,
        default=None,
        help="How many real images to add. Defaults to --gemini-count.",
    )
    parser.add_argument(
        "--fake-prefix",
        type=str,
        default="gemini_fake",
        help="Filename prefix for copied Gemini fake images.",
    )
    parser.add_argument(
        "--real-prefix",
        type=str,
        default="gemini_match_real",
        help="Filename prefix for copied real images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset sampling.",
    )
    parser.add_argument(
        "--replace-prefix",
        action="store_true",
        help="Delete existing files with the same prefixes before copying new ones.",
    )
    args = parser.parse_args()

    real_count = args.gemini_count if args.real_count is None else args.real_count
    fake_out_dir = PROCESSED_DATA_DIR / "fake"
    real_out_dir = PROCESSED_DATA_DIR / "real"
    fake_out_dir.mkdir(parents=True, exist_ok=True)
    real_out_dir.mkdir(parents=True, exist_ok=True)

    gemini_files = list_images(args.gemini_dir)
    real_files = list_images(args.real_dir)

    if len(gemini_files) < args.gemini_count:
        raise ValueError(
            f"Requested {args.gemini_count} Gemini images, but only {len(gemini_files)} found."
        )
    if len(real_files) < real_count:
        raise ValueError(
            f"Requested {real_count} real images, but only {len(real_files)} found."
        )

    if args.replace_prefix:
        removed_fake = remove_prefix_files(fake_out_dir, args.fake_prefix)
        removed_real = remove_prefix_files(real_out_dir, args.real_prefix)
        print(f"Removed {removed_fake} fake files with prefix '{args.fake_prefix}'.")
        print(f"Removed {removed_real} real files with prefix '{args.real_prefix}'.")

    added_fake = copy_subset(
        files=gemini_files,
        destination_dir=fake_out_dir,
        prefix=args.fake_prefix,
        count=args.gemini_count,
        seed=args.seed,
    )
    added_real = copy_subset(
        files=real_files,
        destination_dir=real_out_dir,
        prefix=args.real_prefix,
        count=real_count,
        seed=args.seed + 11,
    )

    print(f"Added {added_fake} Gemini fake images to {fake_out_dir}")
    print(f"Added {added_real} real images to {real_out_dir}")
    print("Now run: python train.py")


if __name__ == "__main__":
    main()
