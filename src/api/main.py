"""FastAPI application. Train the pipeline before starting this service."""
from contextlib import asynccontextmanager
from pathlib import Path
import json
import joblib
from fastapi import FastAPI
from api.db import create_tables
from api.routes.predict import router as predict_router
from api.routes.history import router as history_router

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "src" / "models"
MODEL_ARTIFACTS = {}


def load_artifacts():
    metadata_path = MODELS_DIR / "model_metadata.json"
    model_path = MODELS_DIR / "ladle_life_pipeline.pkl"
    if not metadata_path.exists() or not model_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    MODEL_ARTIFACTS.update({
        "pipeline": joblib.load(model_path),
        "model_version": metadata["model_version"],
        "feature_cols": metadata["feature_columns"],
        "meta": metadata,
    })
    return True


@asynccontextmanager
async def lifespan(app):
    create_tables()
    load_artifacts()
    yield


app = FastAPI(title="Ladle Life Prediction API", version="2.0.0", lifespan=lifespan)
app.include_router(predict_router)
app.include_router(history_router)


@app.get("/health")
def health():
    return {"status": "ok" if MODEL_ARTIFACTS else "model_not_trained", "model_version": MODEL_ARTIFACTS.get("model_version")}
