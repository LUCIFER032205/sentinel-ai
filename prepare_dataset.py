from __future__ import annotations

import argparse
import base64
import io
import time
from pathlib import Path

import requests
from PIL import Image

from src.ai_image_detector.config import PROCESSED_DATA_DIR


def save_image(image_bytes: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.save(destination, format="PNG")


def save_pil_image(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination, format="PNG")


def download_image_url(image_url: str, destination: Path, retries: int = 5) -> None:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(image_url, timeout=120)
            response.raise_for_status()
            save_image(response.content, destination)
            return
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Failed to download image after {retries} attempts") from last_error


def fetch_rows(
    dataset_name: str,
    offset: int,
    length: int,
    retries: int = 5,
) -> list[dict]:
    rows_api = "https://datasets-server.huggingface.co/rows"
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            response = requests.get(
                rows_api,
                params={
                    "dataset": dataset_name,
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": length,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("rows", [])
        except requests.RequestException as exc:
            last_error = exc
            if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 429:
                time.sleep(30 * (attempt + 1))
            else:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch dataset rows after {retries} attempts") from last_error


def download_subset(
    dataset_name: str,
    real_target: int,
    fake_target: int,
    seed: int,
    start_offset: int,
) -> tuple[int, int]:
    real_dir = PROCESSED_DATA_DIR / "real"
    fake_dir = PROCESSED_DATA_DIR / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    real_count = len(list(real_dir.glob("*.png")))
    fake_count = len(list(fake_dir.glob("*.png")))

    if dataset_name == "OwensLab/CommunityForensics-Small":
        offset = start_offset
        page_size = 10

        while real_count < real_target or fake_count < fake_target:
            rows = fetch_rows(dataset_name, offset, page_size)
            if not rows:
                break

            for item in rows:
                row = item["row"]
                label = int(row["label"])
                image_name = Path(row["image_name"]).stem
                image_bytes = base64.b64decode(row["image_data"])

                if label == 0 and real_count < real_target:
                    destination = real_dir / f"communityforensics_real_{real_count:05d}_{image_name}.png"
                    save_image(image_bytes, destination)
                    real_count += 1
                elif label == 1 and fake_count < fake_target:
                    destination = fake_dir / f"communityforensics_fake_{fake_count:05d}_{image_name}.png"
                    save_image(image_bytes, destination)
                    fake_count += 1

                if real_count >= real_target and fake_count >= fake_target:
                    break

            offset += page_size

    elif dataset_name == "Parveshiiii/AI-vs-Real":
        offset = start_offset
        page_size = 20

        while real_count < real_target or fake_count < fake_target:
            rows = fetch_rows(dataset_name, offset, page_size)
            if not rows:
                break

            for item in rows:
                row = item["row"]
                label = int(row["binary_label"])
                image_url = row["image"]["src"]
                row_idx = item["row_idx"]

                if label == 1 and real_count < real_target:
                    destination = real_dir / f"aivsreal_real_{real_count:05d}_{row_idx}.png"
                    download_image_url(image_url, destination)
                    real_count += 1
                elif label == 0 and fake_count < fake_target:
                    destination = fake_dir / f"aivsreal_fake_{fake_count:05d}_{row_idx}.png"
                    download_image_url(image_url, destination)
                    fake_count += 1

                if real_count >= real_target and fake_count >= fake_target:
                    break

            offset += page_size
    else:
        raise ValueError(f"Unsupported dataset source: {dataset_name}")

    return real_count, fake_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a balanced starter subset of CommunityForensics-Small."
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=2000,
        help="Number of real and fake images to download",
    )
    parser.add_argument(
        "--real-count",
        type=int,
        default=None,
        help="Override the real-image target count",
    )
    parser.add_argument(
        "--fake-count",
        type=int,
        default=None,
        help="Override the fake-image target count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stream shuffling",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="OwensLab/CommunityForensics-Small",
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Starting row offset for resumable API downloads",
    )
    args = parser.parse_args()

    real_target = args.per_class if args.real_count is None else args.real_count
    fake_target = args.per_class if args.fake_count is None else args.fake_count

    real_count, fake_count = download_subset(
        dataset_name=args.dataset,
        real_target=real_target,
        fake_target=fake_target,
        seed=args.seed,
        start_offset=args.start_offset,
    )

    print(f"Downloaded {real_count} real images to {PROCESSED_DATA_DIR / 'real'}")
    print(f"Downloaded {fake_count} fake images to {PROCESSED_DATA_DIR / 'fake'}")


if __name__ == "__main__":
    main()
