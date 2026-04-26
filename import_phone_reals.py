from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from src.ai_image_detector.config import PROCESSED_DATA_DIR


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def stable_name(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"phone_real_{path.stem}_{digest}{path.suffix.lower()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import real phone photos into the training dataset.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--target", type=Path, default=PROCESSED_DATA_DIR / "real")
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")

    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0

    for image_path in sorted(source.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_SUFFIXES:
            continue
        output_path = target / stable_name(image_path)
        if output_path.exists():
            skipped += 1
            continue
        shutil.copy2(image_path, output_path)
        copied += 1

    print(f"Copied: {copied}")
    print(f"Skipped existing: {skipped}")
    print(f"Target: {target}")


if __name__ == "__main__":
    main()
