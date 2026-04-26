from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


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


def random_resized_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    if width < 32 or height < 32:
        return image

    min_scale = 0.86
    crop_scale = rng.uniform(min_scale, 1.0)
    crop_w = max(24, int(width * crop_scale))
    crop_h = max(24, int(height * crop_scale))

    if crop_w >= width or crop_h >= height:
        return image

    left = rng.randint(0, width - crop_w)
    top = rng.randint(0, height - crop_h)
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def add_sensor_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    sigma = rng.uniform(1.0, 6.0)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(
        loc=0.0, scale=sigma, size=array.shape
    )
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def jpeg_roundtrip(image: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(42, 92)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def augment_once(image: Image.Image, rng: random.Random) -> Image.Image:
    out = image.convert("RGB")

    # Randomly apply geometric and photometric effects seen in real captures.
    if rng.random() < 0.75:
        out = random_resized_crop(out, rng)

    if rng.random() < 0.50:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if rng.random() < 0.75:
        angle = rng.uniform(-7.0, 7.0)
        out = out.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

    if rng.random() < 0.80:
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.82, 1.20))
    if rng.random() < 0.80:
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.82, 1.25))
    if rng.random() < 0.70:
        out = ImageEnhance.Color(out).enhance(rng.uniform(0.82, 1.18))
    if rng.random() < 0.45:
        out = ImageEnhance.Sharpness(out).enhance(rng.uniform(0.75, 1.35))

    if rng.random() < 0.40:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.4)))

    if rng.random() < 0.65:
        out = add_sensor_noise(out, rng)

    if rng.random() < 0.55:
        out = jpeg_roundtrip(out, rng)

    return out


def clear_existing_outputs(output_dir: Path) -> int:
    removed = 0
    for path in output_dir.glob("gemini_aug_*.png"):
        path.unlink()
        removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create augmented Gemini image variants for model retraining."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"E:\gemini_images"),
        help="Folder containing original Gemini images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\ML mini-project\data\raw\gemini_augmented"),
        help="Where augmented images will be written.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=600,
        help="Total number of augmented outputs to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentation.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Remove previous generated files (gemini_aug_*.png) before writing.",
    )
    args = parser.parse_args()

    if args.target_count <= 0:
        raise ValueError("--target-count must be a positive integer.")

    source_files = list_images(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_existing:
        removed = clear_existing_outputs(args.output_dir)
        print(f"Removed {removed} existing generated files from {args.output_dir}")

    base_seed = args.seed
    total_written = 0
    source_count = len(source_files)

    while total_written < args.target_count:
        source_index = total_written % source_count
        variant_round = total_written // source_count
        source = source_files[source_index]

        rng = random.Random(base_seed + source_index * 1009 + variant_round * 9173)
        with Image.open(source) as image:
            augmented = augment_once(image, rng)
            output_name = f"gemini_aug_{total_written:05d}_{source.stem}.png"
            augmented.save(args.output_dir / output_name, format="PNG")

        total_written += 1

    print(f"Generated {total_written} augmented Gemini images in {args.output_dir}")


if __name__ == "__main__":
    main()

