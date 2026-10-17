from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from mobile_spec_model import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_model,
    validate_training_data,
)


def train(data_path: Path, model_path: Path) -> None:
    try:
        source_data = pd.read_csv(data_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Could not read training data at {data_path}: {error}") from error

    data = validate_training_data(source_data)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS],
        data[TARGET_COLUMNS],
        test_size=0.25,
        random_state=42,
    )
    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    errors = mean_absolute_error(y_test, predictions, multioutput="raw_values")
    bundle = {
        "model": model,
        "use_cases": sorted(data["use_case"].astype(str).unique().tolist()),
        "priorities": sorted(data["priority"].astype(str).unique().tolist()),
        "budget_range": (
            float(data["budget"].min()),
            float(data["budget"].max()),
        ),
        "training_rows": len(data),
        "mae": dict(zip(TARGET_COLUMNS, errors)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    print(f"Trained with {len(x_train)} rows; tested on {len(x_test)} held-out rows.")
    print("Held-out mean absolute error:")
    for target, error in bundle["mae"].items():
        print(f"  {target}: {error:.2f}")
    print(f"Saved trained model to: {model_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a phone-specification model from a labeled CSV."
    )
    parser.add_argument(
        "data",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CSV file to train from (default: {DEFAULT_DATA_PATH.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to save the model (default: {DEFAULT_MODEL_PATH.name})",
    )
    args = parser.parse_args()

    try:
        train(args.data, args.output)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
