"""Project-wide configuration: paths, seeds, hyperparameters."""
from __future__ import annotations

from pathlib import Path

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _p in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- Reproducibility --------------------------------------------------------
RANDOM_SEED = 42

# --- Data -------------------------------------------------------------------
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
DATASET_CSV_NAME = "creditcard.csv"
TARGET_COLUMN = "Class"
TEST_SIZE = 0.2

# --- Feature groups ---------------------------------------------------------
# Columns that need scaling (V1..V28 are already PCA-transformed).
SCALE_COLUMNS = ["Amount", "Time"]

# --- Model hyperparameters --------------------------------------------------
LOGREG_PARAMS = {
    "max_iter": 2000,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "n_jobs": -1,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.1,
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
}

# --- Serving ----------------------------------------------------------------
BEST_MODEL_FILENAME = "best_model.joblib"
SCALER_FILENAME = "scaler.joblib"
API_HOST = "127.0.0.1"
API_PORT = 8000
