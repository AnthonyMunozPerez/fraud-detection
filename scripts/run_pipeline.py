"""End-to-end training pipeline.

Run this from PyCharm (right-click -> Run). It will:
  1. Download the Kaggle credit card fraud dataset
  2. Split into train/test (stratified)
  3. Scale features
  4. Train LogReg, Random Forest, and XGBoost with class-weighting
  5. Evaluate each with PR curves + confusion matrices
  6. Save the best model (by average precision) to models/
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running this file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib

from src.config import BEST_MODEL_FILENAME, MODELS_DIR, SCALER_FILENAME
from src.data import load_dataframe, split_features_target, stratified_split
from src.evaluate import evaluate_model, summarize
from src.preprocess import prepare
from src.train import train_all


def main() -> None:
    print("=" * 60)
    print("Fraud Detection Training Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading dataset...")
    df = load_dataframe()
    print(f"  rows={len(df):,}  columns={df.shape[1]}")
    print(f"  class distribution:\n{df['Class'].value_counts().to_string()}")

    print("\n[2/5] Splitting train/test...")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"  train={len(X_train):,}  test={len(X_test):,}")
    print(f"  train fraud rate: {y_train.mean():.4%}")
    print(f"  test fraud rate:  {y_test.mean():.4%}")

    print("\n[3/5] Scaling features (RobustScaler on Amount + Time)...")
    prepared = prepare(X_train, X_test, y_train, y_test)

    print("\n[4/5] Training models...")
    print("  Strategy: class_weight='balanced' / scale_pos_weight")
    print("  (SMOTE + undersampling helpers are available in src/preprocess.py)")
    trained = train_all(prepared.X_train, prepared.y_train)
    for t in trained:
        print(f"  trained: {t.name}")

    print("\n[5/5] Evaluating...")
    results = [
        evaluate_model(t.name, t.model, prepared.X_test, prepared.y_test)
        for t in trained
    ]

    summary = summarize(results)
    print("\n--- Model comparison (sorted by average precision) ---")
    print(summary.to_string(index=False))

    best_name = summary.iloc[0]["model"]
    best = next(t for t in trained if t.name == best_name)
    print(f"\nBest model: {best_name}")

    model_path = MODELS_DIR / BEST_MODEL_FILENAME
    scaler_path = MODELS_DIR / SCALER_FILENAME
    joblib.dump(best.model, model_path)
    joblib.dump(prepared.scaler, scaler_path)
    print(f"  saved model  -> {model_path}")
    print(f"  saved scaler -> {scaler_path}")

    print("\nDone. Plots saved in outputs/.")


if __name__ == "__main__":
    main()
