from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "ai_image_detector.keras"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
TRAINING_PLOT_PATH = ARTIFACTS_DIR / "training_curves.png"
THRESHOLD_PATH = ARTIFACTS_DIR / "thresholds.json"

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["real", "fake"]
REAL_CLASS_NAME = "real"
FAKE_CLASS_NAME = "fake"
SEED = 42
