from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import RobustScaler

from src.config import RANDOM_SEED, SCALE_COLUMNS

ResampleStrategy = Literal["none", "smote", "undersample"]


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: RobustScaler


def fit_scaler(X_train: pd.DataFrame) -> RobustScaler:
    scaler = RobustScaler()
    scaler.fit(X_train[SCALE_COLUMNS])
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    X_scaled = X.copy()
    X_scaled[SCALE_COLUMNS] = scaler.transform(X[SCALE_COLUMNS])
    return X_scaled


def prepare(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> PreparedData:
    scaler = fit_scaler(X_train)
    return PreparedData(
        X_train=apply_scaler(X_train, scaler),
        X_test=apply_scaler(X_test, scaler),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        scaler=scaler,
    )


def resample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: ResampleStrategy,
) -> tuple[pd.DataFrame, pd.Series]:
    if strategy == "none":
        return X_train, y_train
    if strategy == "smote":
        sampler = SMOTE(random_state=RANDOM_SEED)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown resample strategy: {strategy}")

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res
