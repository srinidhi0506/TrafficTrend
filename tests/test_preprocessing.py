"""
Basic tests for data loading and preprocessing.
Run from the project root:   python -m pytest -v
These tests use tiny hand-made DataFrames (test fixtures), not the real dataset,
so they run even before you download the CSV.
"""

import pandas as pd
import pytest

from src.data_loader import (REQUIRED_COLUMNS, DatasetNotFoundError, MissingColumnsError,
                             describe_raw_data, load_raw_data)
from src.preprocessing import clean_data, filter_by_date, outlier_summary


def make_raw():
    """Six rows with deliberate problems: a duplicate, a repeated timestamp, 0 K temp, huge rain."""
    rows = [
        # holiday, temp, rain, snow, clouds, weather_main, weather_desc, date_time, traffic
        ("None", 280.0, 0.0, 0.0, 40, "Clouds", "scattered clouds", "2018-01-05 08:00:00", 5000),   # Friday
        ("None", 280.0, 0.0, 0.0, 40, "Clouds", "scattered clouds", "2018-01-05 08:00:00", 5000),   # exact duplicate
        ("None", 281.0, 0.0, 0.0, 90, "Rain", "light rain", "2018-01-05 08:00:00", 5000),           # same hour, other weather
        ("None", 0.0, 0.0, 0.0, 10, "Clear", "sky is clear", "2018-01-06 03:00:00", 300),           # Saturday, 0 K temp
        ("Labor Day", 285.0, 9831.3, 0.0, 20, "Clear", "sky is clear", "2018-01-07 00:00:00", 900),  # Sunday, rain error
        ("None", 285.0, 0.0, 0.0, 20, "Clear", "sky is clear", "2018-01-07 01:00:00", 700),
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(DatasetNotFoundError) as err:
        load_raw_data(tmp_path / "does_not_exist.csv")
    assert "download" in str(err.value).lower() or "Download" in str(err.value)


def test_missing_columns_detected(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad, index=False)
    with pytest.raises(MissingColumnsError):
        load_raw_data(bad)


def test_load_valid_csv_and_gzip(tmp_path):
    raw = make_raw()
    for name in ["ok.csv", "ok.csv.gz"]:
        path = tmp_path / name
        raw.to_csv(path, index=False)
        loaded = load_raw_data(path)
        assert loaded.shape == raw.shape
    info = describe_raw_data(loaded)
    assert info["rows"] == 6 and "traffic_volume" in info["column_names"]


def test_duplicates_removed_one_row_per_timestamp():
    df, report = clean_data(make_raw())
    assert df["date_time"].is_unique
    assert report["exact_duplicates_removed"] == 1
    assert report["duplicate_timestamps_removed"] == 1
    assert len(df) == 4


def test_impossible_values_become_nan_but_rows_are_kept():
    df, report = clean_data(make_raw())
    assert report["invalid_values_set_to_nan"]["temp"] == 1
    assert report["invalid_values_set_to_nan"]["rain_1h"] == 1
    saturday = df[df["date_time"] == "2018-01-06 03:00:00"].iloc[0]
    assert pd.isna(saturday["temp_c"]) and saturday["traffic_volume"] == 300


def test_time_features_and_weekend_flag():
    df, _ = clean_data(make_raw())
    friday = df[df["date_time"] == "2018-01-05 08:00:00"].iloc[0]
    assert friday["hour"] == 8 and friday["day_name"] == "Friday" and friday["is_weekend"] == 0
    assert friday["month"] == 1 and friday["month_name"] == "Jan"
    sat = df[df["date_time"] == "2018-01-06 03:00:00"].iloc[0]
    assert sat["is_weekend"] == 1
    assert abs(friday["temp_c"] - (280.0 - 273.15)) < 1e-9


def test_holiday_label_spread_over_whole_day():
    df, _ = clean_data(make_raw())
    sunday = df[df["date_time"].dt.date == pd.Timestamp("2018-01-07").date()]
    assert (sunday["holiday"] == "Labor Day").all() and (sunday["is_holiday"] == 1).all()


def test_missing_text_categories_filled_and_negative_traffic_removed():
    raw = make_raw()
    raw.loc[0, "weather_main"] = None
    raw.loc[5, "traffic_volume"] = -5
    df, _ = clean_data(raw)
    assert "Unknown" in set(df["weather_main"])
    assert (df["traffic_volume"] >= 0).all()


def test_outliers_flagged_not_removed_and_date_filter():
    df, _ = clean_data(make_raw())
    summary = outlier_summary(df)
    assert "traffic_volume" in set(summary["column"])
    assert len(df) == 4                              # nothing removed by the outlier check
    part = filter_by_date(df, "2018-01-06", "2018-01-06")
    assert len(part) == 1
