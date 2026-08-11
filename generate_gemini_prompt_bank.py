from __future__ import annotations

import argparse
import random
from pathlib import Path


SUBJECTS = [
    "a family of four",
    "two close friends",
    "an elderly couple",
    "a young mother and child",
    "a father and daughter",
    "three college students",
    "a street musician",
    "a chef in a busy kitchen",
    "a nurse and patient",
    "an office team",
    "a shopkeeper",
    "a cyclist",
    "a tourist couple",
    "a farmer",
    "a fisherman",
    "a teacher with students",
    "a wedding couple",
    "a barista",
    "a mechanic",
    "a delivery worker",
    "a basketball player",
    "a runner",
    "a hiker",
    "a painter",
    "a carpenter",
]

LOCATIONS = [
    "on a tropical beach",
    "at a crowded train station",
    "in a small home kitchen",
    "inside a city cafe",
    "in a modern office",
    "at a local street market",
    "in a public park",
    "on a quiet suburban street",
    "in a village field",
    "at a riverside walkway",
    "inside a classroom",
    "in a hospital corridor",
    "at a rooftop gathering",
    "in a bookstore",
    "inside a shopping mall",
    "at a bus stop",
    "in a restaurant patio",
    "on a mountain trail",
    "inside a metro coach",
    "at a sports court",
]

ACTIONS = [
    "laughing naturally",
    "having a candid conversation",
    "walking together",
    "looking at a phone",
    "sharing a meal",
    "playing with a pet",
    "carrying groceries",
    "working on a task",
    "taking a short break",
    "drinking coffee",
    "reading a document",
    "taking a selfie",
    "helping each other",
    "packing travel bags",
    "watching the surroundings",
    "preparing food",
    "fixing equipment",
    "celebrating a small moment",
    "waiting in line",
    "crossing the street",
]

LIGHTING = [
    "soft morning light",
    "golden hour sunlight",
    "overcast daylight",
    "bright midday sun",
    "warm indoor lighting",
    "cool fluorescent indoor light",
    "late evening ambient light",
    "rainy day natural light",
    "window side directional light",
    "mixed indoor-outdoor lighting",
]

CAMERA = [
    "shot on a 35mm lens",
    "shot on a 50mm lens",
    "shot on an 85mm lens",
    "DSLR-style depth of field",
    "smartphone camera realism",
    "eye-level perspective",
    "slightly low-angle composition",
    "documentary photography style",
    "candid street photo style",
    "natural handheld framing",
]

COMPOSITION = [
    "wide composition with environment context",
    "medium shot with natural body language",
    "close candid portrait framing",
    "rule-of-thirds composition",
    "off-center framing",
    "foreground-background depth",
    "realistic background clutter",
    "subtle motion in the background",
    "natural subject occlusion",
    "balanced scene geometry",
]

DETAILS = [
    "realistic skin texture and pores",
    "natural hand and finger anatomy",
    "consistent shadows and reflections",
    "accurate fabric folds and wrinkles",
    "true-to-life colors without oversaturation",
    "subtle sensor noise and fine grain",
    "physically plausible perspective",
    "clean but realistic edge transitions",
    "authentic facial expressions",
    "real-world object proportions",
]

BASE_SUFFIX = (
    "No text, no watermark, no logo, no border, no frame. "
    "Do not make it look like CGI, 3D render, digital art, or illustration."
)


def build_prompt(entry: tuple[str, str, str, str, str, str, str]) -> str:
    subject, location, action, lighting, camera, composition, detail = entry
    return (
        f"Photorealistic candid photo of {subject} {action} {location}, "
        f"{lighting}, {camera}, {composition}, {detail}. {BASE_SUFFIX}"
    )


def generate_prompts(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts: list[str] = []

    while len(prompts) < count:
        entry = (
            rng.choice(SUBJECTS),
            rng.choice(LOCATIONS),
            rng.choice(ACTIONS),
            rng.choice(LIGHTING),
            rng.choice(CAMERA),
            rng.choice(COMPOSITION),
            rng.choice(DETAILS),
        )
        prompt = build_prompt(entry)
        if prompt not in prompts:
            prompts.append(prompt)
    return prompts


def write_outputs(output_dir: Path, prompts: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "gemini_prompt_bank_500.txt"
    txt_path.write_text("\n".join(prompts), encoding="utf-8")
    return txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 500-prompt Gemini image bank.")
    parser.add_argument("--count", type=int, default=500, help="Number of prompts to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Where to write prompt-bank files.",
    )
    args = parser.parse_args()

    prompts = generate_prompts(count=args.count, seed=args.seed)
    txt_path = write_outputs(args.output_dir, prompts)

    print(f"Wrote {len(prompts)} prompts to:")
    print(f"- {txt_path}")


if __name__ == "__main__":
    main()
