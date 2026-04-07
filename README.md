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

notebooks/
└── 01_eda.ipynb    # exploratory data analysis

data/     # downloaded dataset (gitignored)
models/   # serialized models
outputs/  # plots and metrics
```

## EDA
`notebooks/01_eda.ipynb` explores the dataset before modeling: class imbalance, amount distributions, hour-of-day fraud patterns, and which PCA features discriminate fraud best. Open in PyCharm's Jupyter integration and run all cells.

## Experiment: Strategy Comparison
`scripts/run_pipeline.py` trains all three model families (LogReg, Random Forest, XGBoost) under all three imbalance-handling strategies (class weights, SMOTE, random undersampling) — nine runs total — and compares them. Results are saved as individual PR curves and confusion matrices per run, plus a combined `outputs/pr_curves_combined.png` overlaying all nine.

Key finding on this dataset: class-weighted training beats SMOTE for all three models, and random undersampling (which reduces the training set to ~800 rows) collapses performance. The best model (`xgboost` + class weights) is what the API serves.

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
