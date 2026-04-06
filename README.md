# Fraud Detection

Credit card fraud detection pipeline using the Kaggle `mlg-ulb/creditcardfraud` dataset.

## Stack
- **uv** for dependency management
- **scikit-learn**, **xgboost**, **imbalanced-learn** for modeling
- **FastAPI** + **uvicorn** for serving

## Setup
```bash
uv sync
```

## Run (from PyCharm)
Right-click and run:
- `scripts/run_pipeline.py` — downloads data, trains all models, saves best model + plots
- `scripts/run_api.py` — launches FastAPI server on `localhost:8000`

Swagger UI: http://localhost:8000/docs

## Project Layout
```
src/
├── config.py       # paths, seed, hyperparams
├── data.py         # dataset download + split
├── preprocess.py   # scaling + resampling
├── train.py        # LogReg / RF / XGBoost
├── evaluate.py     # PR curves, confusion matrices
└── api.py          # FastAPI app

scripts/
├── run_pipeline.py # end-to-end pipeline
└── run_api.py      # launch API server

data/     # downloaded dataset (gitignored)
models/   # serialized models
outputs/  # plots and metrics
```

## Class Imbalance
The dataset is extremely imbalanced (~0.17% fraud). The pipeline compares three strategies:
1. **Class weights** — `class_weight='balanced'`
2. **SMOTE** — synthetic minority oversampling
3. **Random undersampling** — downsample majority class

Resampling is applied **only to training data** to keep test evaluation honest.

## Evaluation
- **Precision-Recall curves** (not ROC — more informative under extreme imbalance)
- **Confusion matrices**
- **Average Precision, F1, Precision, Recall**
