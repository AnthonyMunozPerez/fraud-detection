from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    DATASET_CSV_NAME,
    KAGGLE_DATASET,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)


def download_dataset() -> Path:
    local_csv = DATA_DIR / DATASET_CSV_NAME
    if local_csv.exists():
        return local_csv

    import kagglehub

    print(f"Downloading {KAGGLE_DATASET} via kagglehub...")
    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    source_csv = cache_dir / DATASET_CSV_NAME
    if not source_csv.exists():
        # kagglehub may extract to a subdirectory; search for it.
        matches = list(cache_dir.rglob(DATASET_CSV_NAME))
        if not matches:
            raise FileNotFoundError(
                f"Could not find {DATASET_CSV_NAME} inside {cache_dir}"
            )
        source_csv = matches[0]

    shutil.copy2(source_csv, local_csv)
    return local_csv


def load_dataframe() -> pd.DataFrame:
    return pd.read_csv(download_dataset())


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
