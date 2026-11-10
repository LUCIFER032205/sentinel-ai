---
title: SENTINEL_AI
emoji: "🤖"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: FastAPI AI image detector with batch scan support
---

# SENTINEL_AI

A FastAPI app that tells whether an image is AI-generated or a real photo. It uses a fine-tuned MobileNetV2 model and serves a dark HTML frontend.

- Single-image and batch prediction
- One tuned decision threshold, read at startup from `artifacts/thresholds.json` - no modes to choose
- Scores close to the threshold are returned with a `low_confidence` flag rather than a separate verdict
- ~89% accuracy on held-out images (80% AI recall, 86% real-photo specificity at threshold 0.55)

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Open http://127.0.0.1:8000.

Predict from the terminal:

```bash
python predict.py path\to\image.jpg
```

## Docker

```bash
docker build -t sentinel-ai .
docker run --rm -p 7860:7860 sentinel-ai
```

Open http://127.0.0.1:7860.

## API

| Route | Description |
|-------|-------------|
| `GET /` | Frontend (`static/index.html`) |
| `GET /health` | Health check |
| `GET /calibration` | Live threshold and borderline band |
| `POST /predict` | Form field: `file` |
| `POST /predict/batch` | Form field: `files` (multiple) |

Response:

```json
{ "label": "AI-generated", "ai_probability": 0.94, "confidence": 0.88, "low_confidence": false }
```

`label` is always `Real` or `AI-generated`. `low_confidence` is true when the score falls
inside the borderline band around the threshold, where the model is close to a coin flip
(about 7% of uploads).

## Accuracy and limits

Measured on 1,600 held-out images (`data/holdout_newdb`, `data/holdout_hard_mined`):

| Input | Accuracy |
|-------|----------|
| Ordinary images | ~89% |
| Adversarially hard-mined images | ~73% |
| JPEG re-compression down to q30, downscaling, screenshots | no measurable change |
| Images with added grain (Gaussian noise, sigma 16) | 75%, AI recall falls to 56% |

Two known gaps:

- **Added grain defeats it.** Noise pushes AI images toward the "Real" verdict much harder
  than it pushes real photos (mean shift -0.29 vs -0.08). Training has no noise augmentation,
  so grain is off-distribution. Adding it to the augmentation in `train.py` is the fix.
- **Non-square images are barely tested.** Roughly 89% of the training set and 100% of the
  real-photo holdout set are square 1024x1024. Phone photos and screenshots are not, and no
  number here covers them.

## Retraining

```bash
pip install -r requirements-full.txt
python train.py
```

Training reads images from `data/processed/` (not in git) and writes `artifacts/ai_image_detector.keras`, `metrics.json` and `thresholds.json`.

## Project structure

```
main.py                   FastAPI app
static/index.html         Frontend
src/ai_image_detector/    Config, preprocessing, model, inference
artifacts/                Trained model (Git LFS)
train.py                  Training
calibrate_thresholds.py   Threshold tuning, used by train.py
predict.py                Command-line prediction
REPORT.md                 Project report (also as .docx)
```

## Deployment

The model is stored with Git LFS, so run `git lfs install` before cloning.

- **GitHub:** https://github.com/LUCIFER032205/sentinel-ai (remote `github`)
- **Hugging Face Space:** Docker SDK on port 7860 (remote `origin`). The front matter at the top of this file is the Space config. Push `main` to `origin` to redeploy.
