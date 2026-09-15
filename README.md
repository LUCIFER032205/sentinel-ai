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
- Two scan modes: `default` (fewer false alarms) and `sensitive` (catches more AI images)
- 88.8% accuracy on a 1,000-image test set (F1 0.884 on AI-generated images)

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
| `POST /predict` | Form fields: `file`, optional `mode` (`default` or `sensitive`) |
| `POST /predict/batch` | Form fields: `files` (multiple), optional `mode` |

Response:

```json
{ "label": "AI-generated", "ai_probability": 0.94, "confidence": 0.88 }
```

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
