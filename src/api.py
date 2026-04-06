from __future__ import annotations

from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import (
    BEST_MODEL_FILENAME,
    MODELS_DIR,
    SCALE_COLUMNS,
    SCALER_FILENAME,
)

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first txn")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0.0)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = MODELS_DIR / BEST_MODEL_FILENAME
    scaler_path = MODELS_DIR / SCALER_FILENAME
    if not model_path.exists() or not scaler_path.exists():
        raise RuntimeError(
            f"Model or scaler not found. Run scripts/run_pipeline.py first.\n"
            f"Looked for:\n  {model_path}\n  {scaler_path}"
        )
    state["model"] = joblib.load(model_path)
    state["scaler"] = joblib.load(scaler_path)
    state["threshold"] = 0.5
    yield
    state.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="Serves a trained classifier for credit card fraud detection.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction) -> PredictionResponse:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([txn.model_dump()])[FEATURE_ORDER]
    row[SCALE_COLUMNS] = state["scaler"].transform(row[SCALE_COLUMNS])

    proba = float(state["model"].predict_proba(row)[0, 1])
    threshold = float(state["threshold"])
    return PredictionResponse(
        fraud_probability=proba,
        is_fraud=proba >= threshold,
        threshold=threshold,
    )
