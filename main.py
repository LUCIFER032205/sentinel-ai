from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.ai_image_detector.inference import (
    CalibrationConfig,
    PredictionResult,
    load_trained_model,
    predict_image_bytes,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

MODE_CONFIGS = {
    "default": {
        "calibration": CalibrationConfig(
            threshold=0.65,
            uncertain_low=0.45,
            uncertain_high=0.70,
        ),
        "orientation_conservative": True,
    },
    "sensitive": {
        "calibration": CalibrationConfig(
            threshold=0.40,
            uncertain_low=0.30,
            uncertain_high=0.50,
        ),
        "orientation_conservative": False,
    },
}

app = FastAPI(title="SENTINEL_AI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def cache_model() -> None:
    app.state.model = load_trained_model()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def get_mode_settings(mode: str) -> dict:
    settings = MODE_CONFIGS.get(mode)
    if settings is None:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")
    return settings


def serialize_prediction(result: PredictionResult) -> dict[str, float | str]:
    return {
        "label": result.label,
        "ai_probability": float(result.ai_probability),
        "confidence": float(result.confidence),
    }


async def run_prediction(upload: UploadFile, mode: str) -> dict[str, float | str]:
    payload = await upload.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    settings = get_mode_settings(mode)
    try:
        result = await run_in_threadpool(
            predict_image_bytes,
            app.state.model,
            payload,
            settings["calibration"],
            settings["orientation_conservative"],
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process '{upload.filename or 'upload'}' as an image.",
        ) from exc

    return serialize_prediction(result)


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Form("default"),
) -> dict[str, float | str]:
    return await run_prediction(file, mode)


@app.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    mode: str = Form("default"),
) -> list[dict[str, float | str]]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    results: list[dict[str, float | str]] = []
    for upload in files:
        row = await run_prediction(upload, mode)
        row["filename"] = upload.filename or "upload"
        results.append(row)

    return results
