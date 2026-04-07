from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.config import OUTPUTS_DIR


@dataclass
class EvalResult:
    name: str
    average_precision: float
    f1: float
    precision: float
    recall: float
    confusion: np.ndarray
    strategy: str = ""
    plot_paths: dict[str, Path] = field(default_factory=dict)


def plot_precision_recall(
    name: str, y_true: pd.Series, y_scores: np.ndarray, out_dir: Path
) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    path = out_dir / f"pr_curve_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(name: str, cm: np.ndarray, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")

    path = out_dir / f"confusion_matrix_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pr_curves_combined(
    curves: list[tuple[str, pd.Series, np.ndarray]],
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, y_true, y_scores in curves:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", linewidth=1.5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — all models × strategies")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def evaluate_model(
    name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path = OUTPUTS_DIR,
) -> EvalResult:
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    result = EvalResult(
        name=name,
        average_precision=float(average_precision_score(y_test, y_scores)),
        f1=float(f1_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred)),
        confusion=confusion_matrix(y_test, y_pred),
    )
    result.plot_paths["pr_curve"] = plot_precision_recall(name, y_test, y_scores, out_dir)
    result.plot_paths["confusion_matrix"] = plot_confusion_matrix(name, result.confusion, out_dir)
    return result


def summarize(results: list[EvalResult]) -> pd.DataFrame:
    has_strategy = any(r.strategy for r in results)
    rows = []
    for r in results:
        row = {"model": r.name}
        if has_strategy:
            row["strategy"] = r.strategy
        row.update(
            {
                "avg_precision": round(r.average_precision, 4),
                "f1": round(r.f1, 4),
                "precision": round(r.precision, 4),
                "recall": round(r.recall, 4),
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("avg_precision", ascending=False)
    return df.reset_index(drop=True)
