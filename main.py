from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.ai_image_detector.inference import (
    load_calibration,
    load_trained_model,
    predict_image_bytes,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="SENTINEL_AI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def cache_model() -> None:
    app.state.model = load_trained_model()
    # Read the threshold from artifacts/thresholds.json rather than hardcoding
    # it, so retraining re-tunes the served detector automatically.
    app.state.calibration = load_calibration()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/calibration")
async def calibration() -> dict[str, float]:
    """Expose the live threshold so the frontend cannot drift out of sync."""
    config = app.state.calibration
    return {
        "threshold": config.threshold,
        "band_low": config.band_low,
        "band_high": config.band_high,
    }


async def run_prediction(upload: UploadFile) -> dict[str, float | str | bool]:
    payload = await upload.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = await run_in_threadpool(
            predict_image_bytes,
            app.state.model,
            payload,
            app.state.calibration,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process '{upload.filename or 'upload'}' as an image.",
        ) from exc

    return {
        "label": result.label,
        "ai_probability": float(result.ai_probability),
        "confidence": float(result.confidence),
        "low_confidence": bool(result.low_confidence),
    }


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, float | str | bool]:
    return await run_prediction(file)


@app.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
) -> list[dict[str, float | str | bool]]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    results: list[dict[str, float | str | bool]] = []
    for upload in files:
        row = await run_prediction(upload)
        row["filename"] = upload.filename or "upload"
        results.append(row)

    return results
