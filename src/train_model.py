"""
train_model.py
--------------
Trains and compares Linear Regression and Random Forest to predict traffic_volume.

Run from the project root (optional, the dashboard also does this):
    python -m src.train_model

DESIGN DECISIONS (explain these in your viva)
1. Chronological split: data is sorted by time; the first 80% is training, the last 20% is
   testing. A random split would mix future rows into training, which makes the test
   unrealistically easy for time-dependent data.
2. No target leakage: features are only information known BEFORE/AT the hour being
   predicted (calendar fields and weather). We do NOT use the previous hour's traffic or any
   value derived from traffic_volume (e.g. group averages of traffic).
3. Preprocessing is inside a Pipeline and fitted ONLY on the training rows
   (median imputation, scaling, one-hot encoding). The test set is only transformed.
4. Prediction vs forecasting: these models predict traffic GIVEN time + weather. To forecast
   the future you would need a weather forecast as input, which this project does not use.
"""

from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import MODELS_DIR

TARGET = "traffic_volume"
CATEGORICAL_FEATURES = ["hour", "day_of_week", "month", "weather_main"]
NUMERIC_FEATURES = ["temp_c", "rain_1h", "snow_1h", "clouds_all"]
BINARY_FEATURES = ["is_weekend", "is_holiday"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES

MODEL_PATH = MODELS_DIR / "best_traffic_model.joblib"

FEATURE_AVAILABILITY = pd.DataFrame([
    ("hour, day_of_week, month, is_weekend", "Calendar", "Yes - known for any future date"),
    ("is_holiday", "Calendar", "Yes - holidays are known in advance"),
    ("temp_c, rain_1h, snow_1h, clouds_all, weather_main", "Weather", "Only if a weather forecast is supplied; "
     "the dataset contains the observed weather of that hour"),
    ("traffic_volume of previous hours (lags)", "Past traffic", "NOT used - would need live traffic data and "
     "the dataset has time gaps"),
    ("year", "Calendar", "NOT used - a model cannot learn a future year it never saw"),
], columns=["Feature(s)", "Type", "Available at prediction time?"])


def make_preprocessor() -> ColumnTransformer:
    """Imputation + scaling + one-hot encoding (fitted only when pipeline.fit is called on train data)."""
    return ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ])


def build_models() -> dict:
    return {
        "Baseline (training mean)": Pipeline([("prep", make_preprocessor()), ("model", DummyRegressor(strategy="mean"))]),
        "Linear Regression": Pipeline([("prep", make_preprocessor()), ("model", LinearRegression())]),
        "Random Forest": Pipeline([("prep", make_preprocessor()),
                                   ("model", RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                                                   n_jobs=-1, random_state=42))]),
    }


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """First (1 - test_fraction) of the time-ordered rows = train, remainder = test."""
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")
    ordered = df.sort_values("date_time").reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_fraction))
    if cut < 100 or len(ordered) - cut < 20:
        raise ValueError("Not enough rows to train and test a model.")
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def check_columns(df: pd.DataFrame):
    missing = [c for c in FEATURE_COLUMNS + [TARGET, "date_time"] if c not in df.columns]
    if missing:
        raise ValueError(f"Cleaned data is missing required column(s): {missing}")


def evaluate(y_true, y_pred) -> dict:
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def train_and_evaluate(df: pd.DataFrame, test_fraction: float = 0.2, save: bool = True) -> dict:
    """
    Train all models, evaluate on the held-out final period, save the best real model.
    Returns a dict with results table, predictions and fitted pipelines.
    """
    check_columns(df)
    train, test = chronological_split(df, test_fraction)
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET]

    rows, fitted, preds = [], {}, {}
    for name, pipe in build_models().items():
        pipe.fit(X_train, y_train)
        pred = np.clip(pipe.predict(X_test), 0, None)     # traffic cannot be negative
        fitted[name], preds[name] = pipe, pred
        m = evaluate(y_test, pred)
        rows.append({"Model": name, **m})

    results = pd.DataFrame(rows)
    real = results[results["Model"] != "Baseline (training mean)"]
    best_name = real.sort_values("RMSE").iloc[0]["Model"]    # lowest RMSE wins

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": fitted[best_name], "model_name": best_name, "feature_columns": FEATURE_COLUMNS,
                     "metrics": real[real["Model"] == best_name].iloc[0].to_dict(),
                     "train_period": (str(train["date_time"].min()), str(train["date_time"].max())),
                     "test_period": (str(test["date_time"].min()), str(test["date_time"].max())),
                     "trained_at": datetime.now().isoformat(timespec="seconds")}, MODEL_PATH)

    return {"results": results, "best_model_name": best_name, "models": fitted, "predictions": preds,
            "train": train, "test": test, "model_path": MODEL_PATH if save else None}


def load_saved_model(path=MODEL_PATH):
    """Load the saved artifact (dict) or return None if it does not exist."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def predict_traffic(pipeline, hour, day_of_week, month, weather_main="Clear",
                    temp_c=10.0, rain_1h=0.0, snow_1h=0.0, clouds_all=40, is_holiday=0) -> float:
    """Predict one hour. Uses the same FEATURE_COLUMNS as training."""
    row = pd.DataFrame([{
        "hour": hour, "day_of_week": day_of_week, "month": month, "weather_main": weather_main,
        "temp_c": temp_c, "rain_1h": rain_1h, "snow_1h": snow_1h, "clouds_all": clouds_all,
        "is_weekend": int(day_of_week >= 5), "is_holiday": int(is_holiday),
    }])[FEATURE_COLUMNS]
    return float(max(0.0, pipeline.predict(row)[0]))


def feature_importance(pipeline, top_n: int = 15):
    """Random Forest feature importances (None for models without them)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    names = pipeline.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    out = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    return out.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data
    clean, _ = clean_data(load_raw_data())
    out = train_and_evaluate(clean)
    print(out["results"].round(3).to_string(index=False))
    print("Best model:", out["best_model_name"], "-> saved to", out["model_path"])
