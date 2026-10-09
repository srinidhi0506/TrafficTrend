"""Tests for the analysis and modelling modules (small synthetic fixtures)."""

import numpy as np
import pandas as pd
import pytest

from src import anomaly_detection as ad
from src import peak_analysis as pk
from src import train_model as tm
from src.data_loader import REQUIRED_COLUMNS
from src.preprocessing import clean_data


def make_clean(n_days=60, seed=0):
    """Synthetic hourly data with a rush-hour pattern; one injected spike."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2017-01-01", periods=24 * n_days, freq="h")
    hours = np.asarray(times.hour)
    base = 500 + 3000 * np.exp(-((hours - 8) ** 2) / 8) + 3500 * np.exp(-((hours - 17) ** 2) / 8)
    traffic = (base * np.where(np.asarray(times.dayofweek) >= 5, 0.6, 1.0) + rng.normal(0, 80, len(times)))
    traffic = np.clip(traffic, 0, None).round()
    if len(traffic) > 100:
        traffic[100] = 9000
    raw = pd.DataFrame({"holiday": "None", "temp": 280 + rng.normal(0, 3, len(times)), "rain_1h": 0.0,
                        "snow_1h": 0.0, "clouds_all": rng.integers(0, 100, len(times)),
                        "weather_main": "Clear", "weather_description": "sky is clear",
                        "date_time": times.astype(str), "traffic_volume": traffic})[REQUIRED_COLUMNS]
    df, _ = clean_data(raw)
    return df


def test_chronological_split_has_no_overlap_and_keeps_order():
    df = make_clean()
    train, test = tm.chronological_split(df, 0.2)
    assert train["date_time"].max() < test["date_time"].min()
    assert len(train) + len(test) == len(df)


def test_split_rejects_too_little_data():
    with pytest.raises(ValueError):
        tm.chronological_split(make_clean(n_days=2), 0.2)


def test_features_do_not_contain_target():
    assert tm.TARGET not in tm.FEATURE_COLUMNS


def test_training_returns_metrics_and_saves_model(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(tm, "MODELS_DIR", tmp_path)
    out = tm.train_and_evaluate(make_clean(), 0.2, save=True)
    assert {"MAE", "RMSE", "R2"} <= set(out["results"].columns)
    assert (tmp_path / "m.joblib").exists()
    saved = tm.load_saved_model(tmp_path / "m.joblib")
    assert saved["feature_columns"] == tm.FEATURE_COLUMNS
    value = tm.predict_traffic(saved["model"], 8, 1, 3)
    assert value >= 0


def test_iqr_flags_injected_spike():
    result = ad.detect_iqr_anomalies(make_clean())
    assert result.loc[100, "is_anomaly"] and result.loc[100, "anomaly_type"] == "High"
    counts = ad.anomaly_counts(result)
    assert counts["flagged"] == counts["high"] + counts["low"]


def test_peak_hours_found_from_data():
    df = make_clean()
    top = pk.top_busiest_hours(df, 5)
    assert len(top) == 5
    assert int(top.iloc[0]["hour"]) in (16, 17, 18)
    assert len(pk.generate_observations(df)) >= 4
