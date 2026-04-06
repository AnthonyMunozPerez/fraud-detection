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


def train_logreg(X_train: pd.DataFrame, y_train: pd.Series) -> TrainedModel:
    model = LogisticRegression(**LOGREG_PARAMS)
    model.fit(X_train, y_train)
    return TrainedModel(name="logistic_regression", model=model)


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> TrainedModel:
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    return TrainedModel(name="random_forest", model=model)


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> TrainedModel:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    return TrainedModel(name="xgboost", model=model)


def train_all(X_train: pd.DataFrame, y_train: pd.Series) -> list[TrainedModel]:
    return [
        train_logreg(X_train, y_train),
        train_random_forest(X_train, y_train),
        train_xgboost(X_train, y_train),
    ]
