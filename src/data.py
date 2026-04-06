"""Dataset download, loading, and train/test splitting."""
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
    """Download the Kaggle credit card fraud dataset via kagglehub.

    Copies the CSV into the project's local data/ directory on first run
    so PyCharm users don't need to dig into kagglehub's cache.
    """
    local_csv = DATA_DIR / DATASET_CSV_NAME
    if local_csv.exists():
        return local_csv

    import kagglehub  # imported lazily so tests don't require network

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
    print(f"Dataset saved to {local_csv}")
    return local_csv


def load_dataframe() -> pd.DataFrame:
    """Load the credit card fraud dataset as a pandas DataFrame."""
    csv_path = download_dataset()
    df = pd.read_csv(csv_path)
    return df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target vector y."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split that preserves the fraud class ratio."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
