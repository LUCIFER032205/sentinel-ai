from __future__ import annotations

import argparse
import io
import os
import random
import time
from collections import deque
from pathlib import Path

from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'google-genai'. Install it with: pip install google-genai"
    ) from exc


VARIATION_SUFFIXES = [
    "Use a different camera angle and framing from previous outputs.",
    "Change people appearance, clothing, and background details.",
    "Make lighting natural but different from earlier variants.",
    "Keep realism high with physically correct shadows and reflections.",
    "Use candid composition with natural human expressions.",
    "Ensure realistic hand anatomy and face details.",
    "Keep colors true-to-life and avoid over-saturation.",
    "Add subtle environmental clutter to look like a real photo.",
    "Use a different focal length feel while keeping realism.",
    "Avoid any stylization; keep it camera-photo realistic.",
]


def read_prompt_bank(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt bank not found: {path}")
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def next_image_index(output_dir: Path, prefix: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"{prefix}_*.png"))
    if not existing:
        return 1
    last = existing[-1].stem
    # expected: prefix_00001
    try:
        return int(last.split("_")[-1]) + 1
    except ValueError:
        return len(existing) + 1


def response_to_image(response) -> Image.Image:
    parts = []
    if getattr(response, "parts", None):
        parts = response.parts
    elif getattr(response, "candidates", None):
        first = response.candidates[0]
        parts = getattr(first.content, "parts", []) if first else []

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue

        as_image = getattr(part, "as_image", None)
        if callable(as_image):
            image = as_image()
            if image is not None:
                return image

        data = getattr(inline, "data", None)
        if data:
            if isinstance(data, str):
                import base64

                data = base64.b64decode(data)
            return Image.open(io.BytesIO(data)).convert("RGB")

    raise RuntimeError("No image found in model response.")


def generate_one_image(
    client: genai.Client,
    model: str,
    prompt: str,
) -> Image.Image:
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )
    return response_to_image(response)


def enforce_rate_limit(request_timestamps: deque[float], max_rpm: int) -> None:
    if max_rpm <= 0:
        return

    window_seconds = 60.0
    now = time.monotonic()
    while request_timestamps and (now - request_timestamps[0]) >= window_seconds:
        request_timestamps.popleft()

    if len(request_timestamps) < max_rpm:
        return

    wait_seconds = window_seconds - (now - request_timestamps[0]) + 0.05
    wait_seconds = max(wait_seconds, 0.05)
    print(f"rate-limit wait: sleeping {wait_seconds:.2f}s to stay under {max_rpm} req/min")
    time.sleep(wait_seconds)

    now = time.monotonic()
    while request_timestamps and (now - request_timestamps[0]) >= window_seconds:
        request_timestamps.popleft()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate many Gemini images from a prompt bank.")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "gemini_prompt_bank_500.txt",
        help="Path to prompt-bank text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder to save generated images.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=700,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash-image",
        help="Gemini image model id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for prompt variation.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.3,
        help="Delay between successful requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=9,
        help="Maximum API requests per minute.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retries per image on transient failures.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="gemini_api",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (optional, overrides environment).",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="Path to a text file containing only the API key.",
    )
    args = parser.parse_args()

    prompts = read_prompt_bank(args.prompt_file)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    start_idx = next_image_index(output_dir, args.prefix)

    resolved_api_key = None
    if args.api_key:
        resolved_api_key = args.api_key.strip()
    elif args.api_key_file:
        if not args.api_key_file.exists():
            raise FileNotFoundError(f"API key file not found: {args.api_key_file}")
        resolved_api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    else:
        resolved_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not resolved_api_key:
        raise RuntimeError(
            "No API key found. Provide --api-key, --api-key-file, or set GEMINI_API_KEY."
        )

    client = genai.Client(api_key=resolved_api_key)
    generated = 0
    attempts = 0
    consecutive_failures = 0
    request_timestamps: deque[float] = deque()

    while generated < args.count:
        global_index = start_idx + generated
        base_prompt = prompts[(global_index - 1) % len(prompts)]
        variation = rng.choice(VARIATION_SUFFIXES)
        full_prompt = f"{base_prompt} {variation}"

        destination = output_dir / f"{args.prefix}_{global_index:05d}.png"
        success = False
        error_text = ""

        for retry in range(1, args.max_retries + 1):
            attempts += 1
            try:
                enforce_rate_limit(request_timestamps, args.max_rpm)
                request_timestamps.append(time.monotonic())
                image = generate_one_image(client=client, model=args.model, prompt=full_prompt)
                image.convert("RGB").save(destination, format="PNG")
                generated += 1
                success = True
                consecutive_failures = 0
                print(f"[{generated}/{args.count}] saved {destination.name}")
                time.sleep(args.sleep_seconds)
                break
            except Exception as exc:  # broad on purpose for API transport/model errors
                error_text = str(exc)
                backoff = min(20.0, 2.0 * retry)
                print(f"retry {retry}/{args.max_retries} for {destination.name}: {error_text}")
                time.sleep(backoff)

        if not success:
            consecutive_failures += 1
            print(f"failed image {destination.name}: {error_text}")
            print("Moving to next prompt and retrying this index.")
            if consecutive_failures >= 12:
                raise RuntimeError(
                    "Stopping generation after repeated failures. "
                    "Check model id, API key, billing, or rate limits."
                )

    print(f"Done. Target count: {args.count}, attempts: {attempts}")
    print(f"Images folder: {output_dir}")


if __name__ == "__main__":
    main()
