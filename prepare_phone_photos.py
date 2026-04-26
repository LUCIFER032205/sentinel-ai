from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


HEIC_SUFFIXES = {".heic", ".heif"}
VIDEO_SUFFIXES = {".mp4", ".mov"}


def unique_jpg_path(source: Path) -> Path:
    candidate = source.with_suffix(".jpg")
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = source.with_name(f"{source.stem}_heic_{index}.jpg")
        if not candidate.exists():
            return candidate
        index += 1


def assert_inside_root(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_root not in resolved_path.parents and resolved_path != resolved_root:
        raise ValueError(f"Refusing to touch path outside root: {path}")


def convert_heic(root: Path, quality: int) -> int:
    register_heif_opener()
    converted = 0
    for source in sorted(root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in HEIC_SUFFIXES:
            continue
        assert_inside_root(source, root)
        target = unique_jpg_path(source)
        with Image.open(source) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb.save(target, "JPEG", quality=quality, optimize=True)
        converted += 1
        print(f"converted: {source} -> {target}")
    return converted


def delete_videos(root: Path) -> int:
    deleted = 0
    for source in sorted(root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        assert_inside_root(source, root)
        source.unlink()
        deleted += 1
        print(f"deleted: {source}")
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real phone photos for training.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--delete-videos", action="store_true")
    args = parser.parse_args()

    root = args.folder.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")

    converted = convert_heic(root, quality=args.quality)
    deleted = delete_videos(root) if args.delete_videos else 0
    print(f"HEIC converted: {converted}")
    print(f"Videos deleted: {deleted}")


if __name__ == "__main__":
    main()
