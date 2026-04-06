from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib

from src.config import BEST_MODEL_FILENAME, MODELS_DIR, OUTPUTS_DIR, SCALER_FILENAME
from src.data import load_dataframe, split_features_target, stratified_split
from src.evaluate import evaluate_model, summarize
from src.preprocess import prepare
from src.train import train_all


def main() -> None:
    df = load_dataframe()
    print(f"rows={len(df):,}  columns={df.shape[1]}")
    print(f"class distribution:\n{df['Class'].value_counts().to_string()}")

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"train={len(X_train):,}  test={len(X_test):,}")
    print(f"train fraud rate: {y_train.mean():.4%}")
    print(f"test fraud rate:  {y_test.mean():.4%}")

    prepared = prepare(X_train, X_test, y_train, y_test)

    trained = train_all(prepared.X_train, prepared.y_train)
    for t in trained:
        print(f"trained: {t.name}")

    results = [
        evaluate_model(t.name, t.model, prepared.X_test, prepared.y_test)
        for t in trained
    ]

    summary = summarize(results)
    print(summary.to_string(index=False))

    best_name = summary.iloc[0]["model"]
    best = next(t for t in trained if t.name == best_name)
    print(f"best model: {best_name}")

    model_path = MODELS_DIR / BEST_MODEL_FILENAME
    scaler_path = MODELS_DIR / SCALER_FILENAME
    joblib.dump(best.model, model_path)
    joblib.dump(prepared.scaler, scaler_path)
    print(f"saved model  -> {model_path}")
    print(f"saved scaler -> {scaler_path}")
    print(f"plots saved to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
