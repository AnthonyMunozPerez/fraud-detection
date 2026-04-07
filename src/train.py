from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import LOGREG_PARAMS, RF_PARAMS, XGB_PARAMS


@dataclass
class TrainedModel:
    name: str
    model: Any


def train_logreg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_balanced: bool = True,
) -> TrainedModel:
    params = {**LOGREG_PARAMS}
    if not use_balanced:
        params["class_weight"] = None
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return TrainedModel(name="logistic_regression", model=model)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_balanced: bool = True,
) -> TrainedModel:
    params = {**RF_PARAMS}
    if not use_balanced:
        params["class_weight"] = None
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return TrainedModel(name="random_forest", model=model)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_balanced: bool = True,
) -> TrainedModel:
    params = {**XGB_PARAMS}
    if use_balanced:
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        params["scale_pos_weight"] = neg / max(pos, 1)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return TrainedModel(name="xgboost", model=model)
