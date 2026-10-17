from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "phone_specs_training.csv"
DEFAULT_MODEL_PATH = PROJECT_DIR / "mobile_specs_model.joblib"

FEATURE_COLUMNS = ["budget", "use_case", "priority"]
TARGET_COLUMNS = ["ram_gb", "storage_gb", "battery_mah", "camera_mp"]
DATA_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS
CATEGORICAL_COLUMNS = ["use_case", "priority"]

SPEC_OPTIONS = {
    "ram_gb": [4, 6, 8, 12, 16, 24, 32],
    "storage_gb": [64, 128, 256, 512, 1024],
    "battery_mah": [4000, 4500, 5000, 5500, 6000, 6500, 7000],
    "camera_mp": [12, 16, 32, 50, 64, 108, 200],
}


def validate_training_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in DATA_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

    data = frame[DATA_COLUMNS].copy()
    for column in ["budget", *TARGET_COLUMNS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"{column} must contain only numeric values.")
        if not data[column].map(lambda value: isfinite(float(value))).all():
            raise ValueError(f"{column} must contain finite numeric values.")
        if (data[column] <= 0).any():
            raise ValueError(f"{column} values must be greater than zero.")

    for column in CATEGORICAL_COLUMNS:
        data[column] = data[column].astype("string").str.strip()
        if data[column].isna().any() or data[column].eq("").any():
            raise ValueError(f"{column} must be filled in for every row.")

    if len(data) < 8:
        raise ValueError("Add at least 8 complete examples before training.")
    return data


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
                    ],
                    remainder="passthrough",
                ),
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=160,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )


def predict_specifications(
    model: Any,
    budget: float,
    use_case: str,
    priority: str,
) -> dict[str, int]:
    request = pd.DataFrame(
        [{"budget": budget, "use_case": use_case, "priority": priority}],
        columns=FEATURE_COLUMNS,
    )
    prediction = model.predict(request)[0]
    return {
        target: min(
            SPEC_OPTIONS[target],
            key=lambda option: abs(option - predicted_value),
        )
        for target, predicted_value in zip(TARGET_COLUMNS, prediction)
    }


def load_model_bundle(path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Model file has an invalid format: {path}")
    return bundle
