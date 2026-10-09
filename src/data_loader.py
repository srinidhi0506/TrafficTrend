"""
data_loader.py
--------------
Step 1 of the pipeline: find and load the raw CSV file.

Where the dataset must be placed
    TrafficTrend/data/Metro_Interstate_Traffic_Volume.csv
(The UCI download is a zip. Inside it the file may be named
 'Metro_Interstate_Traffic_Volume.csv.gz'. Pandas can read .csv.gz directly,
 so you may place either the .csv or the .csv.gz file in the data/ folder.)

How the application finds it
    PROJECT_ROOT is calculated from this file's own location, so the project
    works no matter which folder your terminal is currently in.

This module never invents data. If the file is missing it raises a clear error.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # .../TrafficTrend
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_FILENAMES = [
    "Metro_Interstate_Traffic_Volume.csv",
    "Metro_Interstate_Traffic_Volume.csv.gz",
]

DATASET_URL = "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"

# Columns the UCI dataset is expected to contain
REQUIRED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]


# ---------------------------------------------------------------------------
# Custom errors (so the dashboard can show friendly messages)
# ---------------------------------------------------------------------------
class DatasetNotFoundError(FileNotFoundError):
    """Raised when the dataset file is not inside the data/ folder."""


class MissingColumnsError(ValueError):
    """Raised when the CSV does not have the columns this project needs."""


def dataset_help_message() -> str:
    """Human-readable instructions shown when the dataset is missing."""
    return (
        "The dataset file was not found.\n\n"
        f"1. Open: {DATASET_URL}\n"
        "2. Click 'Download' (a zip file is downloaded).\n"
        "3. Extract the zip. Inside you will find 'Metro_Interstate_Traffic_Volume.csv' "
        "(or 'Metro_Interstate_Traffic_Volume.csv.gz').\n"
        f"4. Copy that file into this folder:\n   {DATA_DIR}\n"
        "5. Refresh this page."
    )


def find_dataset_path(custom_path=None) -> Path:
    """
    Return the path of the dataset file.

    Order of search:
      1. custom_path, if given
      2. data/Metro_Interstate_Traffic_Volume.csv (or .csv.gz)
      3. If data/ contains exactly one .csv/.csv.gz file, use that one
    """
    if custom_path is not None:
        path = Path(custom_path)
        if path.exists():
            return path
        raise DatasetNotFoundError(f"File not found: {path}\n\n{dataset_help_message()}")

    for name in DATASET_FILENAMES:
        candidate = DATA_DIR / name
        if candidate.exists():
            return candidate

    if DATA_DIR.exists():
        others = sorted(list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.csv.gz")))
        if len(others) == 1:
            return others[0]

    raise DatasetNotFoundError(dataset_help_message())


def load_raw_data(custom_path=None) -> pd.DataFrame:
    """Load the raw CSV exactly as stored (no cleaning) and validate columns."""
    path = find_dataset_path(custom_path)
    try:
        df = pd.read_csv(path)   # pandas reads .csv and .csv.gz automatically
    except Exception as exc:     # corrupted file, wrong format, etc.
        raise ValueError(f"Could not read '{path.name}' as a CSV file: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MissingColumnsError(
            f"The file '{path.name}' is missing required column(s): {missing}.\n"
            f"Columns found: {list(df.columns)}\n"
            "Please download the original UCI 'Metro Interstate Traffic Volume' file."
        )
    return df


def describe_raw_data(df: pd.DataFrame) -> dict:
    """Shape, column names, data types and missing-value counts of a DataFrame."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {c: int(n) for c, n in df.isna().sum().items()},
    }


if __name__ == "__main__":
    # Quick manual check:  python -m src.data_loader
    try:
        data = load_raw_data()
        info = describe_raw_data(data)
        print("Shape:", (info["rows"], info["columns"]))
        print("Columns:", info["column_names"])
    except (DatasetNotFoundError, MissingColumnsError, ValueError) as err:
        print("ERROR:", err)
