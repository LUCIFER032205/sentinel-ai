---
title: SENTINEL_AI
emoji: "🛡️"
colorFrom: cyan
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# SENTINEL_AI

SENTINEL_AI is a FastAPI-powered AI image detector with a dark cyberpunk HTML frontend.

The app supports:

- single-image prediction
- batch prediction
- `Default Scan`
- `AI-Sensitive`

## Hugging Face Spaces deployment

This repository is prepared for a Hugging Face Docker Space.

Included deployment files:

- `Dockerfile`
- `.dockerignore`
- this `README.md` with Space metadata
- `requirements.txt` for a lighter inference-only build
- `requirements-full.txt` if you want the older training and Streamlit tooling locally

Hugging Face will build the Docker image and run the app on port `7860`.

## App routes

- `GET /` serves the frontend from `static/index.html`
- `POST /predict` accepts one uploaded image
- `POST /predict/batch` accepts multiple uploaded images

The app loads the trained model at startup from:

- `artifacts/ai_image_detector.keras`

## Local development

### Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

If you want the older training and Streamlit dependencies too:

```bash
pip install -r requirements-full.txt
```

### Docker

```bash
docker build -t sentinel-ai .
docker run --rm -p 7860:7860 sentinel-ai
```

Then open:

- `http://127.0.0.1:7860`

## Project structure

- `main.py`: FastAPI entrypoint
- `static/index.html`: frontend UI
- `src/ai_image_detector/`: shared inference code
- `artifacts/`: trained model and related artifacts
- `predict.py`: terminal prediction helper
- `train.py`: model training script

## Notes

- The app requires a trained model artifact before deployment.
- `data/` is not needed for inference-only deployment, so it is excluded from the Docker build context.
- If you retrain the model, make sure the updated `artifacts/ai_image_detector.keras` is included before pushing to the Space.
- The model file `artifacts/ai_image_detector.keras` is not excluded by `.dockerignore`, so it will be copied into the Docker image.

## Publish to Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Choose `Docker` as the SDK.
3. Push this repository to the Space repository.
4. Wait for the Docker build to finish.
5. Open the Space URL once the build is green.

Example git flow:

```bash
git init
git lfs install
git add .gitattributes
git add .
git commit -m "Prepare Hugging Face Docker Space"
git branch -M main
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push -u origin main
```

If you already have a Space repository, you can also clone that repo first and copy this project into it before committing and pushing.

After deploy, these routes should be available:

- `/`
- `/health`
- `/predict`
- `/predict/batch`
