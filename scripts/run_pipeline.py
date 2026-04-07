from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib

from src.config import BEST_MODEL_FILENAME, MODELS_DIR, OUTPUTS_DIR, SCALER_FILENAME
from src.data import load_dataframe, split_features_target, stratified_split
from src.evaluate import (
    EvalResult,
    evaluate_model,
    plot_pr_curves_combined,
    summarize,
)
from src.preprocess import ResampleStrategy, prepare, resample
from src.train import TrainedModel, train_logreg, train_random_forest, train_xgboost


STRATEGIES: list[ResampleStrategy] = ["none", "smote", "undersample"]
TRAINERS = [
    ("logistic_regression", train_logreg),
    ("random_forest", train_random_forest),
    ("xgboost", train_xgboost),
]


def main() -> None:
    df = load_dataframe()
    print(f"rows={len(df):,}  columns={df.shape[1]}")

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"train={len(X_train):,}  test={len(X_test):,}")
    print(f"train fraud rate: {y_train.mean():.4%}")
    print(f"test fraud rate:  {y_test.mean():.4%}")

    prepared = prepare(X_train, X_test, y_train, y_test)

    all_results: list[EvalResult] = []
    all_trained: dict[str, TrainedModel] = {}
    pr_curves: list[tuple[str, object, object]] = []

    for strategy in STRATEGIES:
        X_tr, y_tr = resample(prepared.X_train, prepared.y_train, strategy)
        use_balanced = strategy == "none"
        print(f"\nstrategy={strategy}  train_rows={len(X_tr):,}  balanced_weights={use_balanced}")

        for model_name, train_fn in TRAINERS:
            trained = train_fn(X_tr, y_tr, use_balanced=use_balanced)
            tag = f"{model_name}_{strategy}"
            result = evaluate_model(tag, trained.model, prepared.X_test, prepared.y_test)
            result.strategy = strategy
            all_results.append(result)
            all_trained[tag] = trained

            y_scores = trained.model.predict_proba(prepared.X_test)[:, 1]
            pr_curves.append((tag, prepared.y_test, y_scores))
            print(f"  {tag}: AP={result.average_precision:.4f}  F1={result.f1:.4f}")

    print()
    summary = summarize(all_results)
    print(summary.to_string(index=False))

    best_tag = summary.iloc[0]["model"]
    best = all_trained[best_tag]

    model_path = MODELS_DIR / BEST_MODEL_FILENAME
    scaler_path = MODELS_DIR / SCALER_FILENAME
    joblib.dump(best.model, model_path)
    joblib.dump(prepared.scaler, scaler_path)

    combined_path = plot_pr_curves_combined(pr_curves, OUTPUTS_DIR / "pr_curves_combined.png")

    print(f"\nbest model: {best_tag}")
    print(f"saved model  -> {model_path}")
    print(f"saved scaler -> {scaler_path}")
    print(f"combined PR  -> {combined_path}")
    print(f"per-run plots saved to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
