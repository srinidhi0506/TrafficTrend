"""
preprocessing.py
----------------
Cleans the raw Metro Interstate Traffic Volume data and creates time features.

Every decision is recorded in a human-readable list (report["steps"]) so the
dashboard can display it and you can explain it during your viva.

KEY IDEA - "unusual but valid" vs "data error"
    * A data ERROR is physically impossible or contradicts the meaning of the
      column (e.g. air temperature of 0 Kelvin = -273 C, or 9,831 mm of rain in
      one hour). We cannot trust such a value, so we mark it as missing (NaN).
    * An unusual but VALID observation is rare but possible (e.g. very heavy
      traffic on a holiday evening, or very low traffic at 3 a.m.). These are real
      measurements, so we KEEP them. Statistical outliers are only flagged,
      never deleted automatically.

IMPORTANT - why numeric missing values are NOT filled here
    If we filled missing weather values with a median of the WHOLE dataset, the
    test period would influence the training data (a mild form of leakage).
    So this module only marks bad values as NaN. The machine-learning pipeline
    (train_model.py) fills them using a median learned from the TRAINING data only.
"""

import numpy as np
import pandas as pd

# Plausibility limits used to detect data errors (documented, easy to change)
MIN_VALID_TEMP_KELVIN = 200.0     # about -73 C; 0 K is impossible
MAX_VALID_TEMP_KELVIN = 330.0     # about 57 C
MAX_VALID_RAIN_MM_PER_HOUR = 100.0  # far above any realistic hourly rainfall for this region
MAX_VALID_SNOW_MM_PER_HOUR = 100.0

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

WEATHER_NUMERIC_COLUMNS = ["temp_c", "rain_1h", "snow_1h", "clouds_all"]


def clean_data(raw: pd.DataFrame):
    """
    Clean the raw DataFrame.

    Returns
    -------
    df : pandas.DataFrame   cleaned data sorted by time, with extra time features
    report : dict           counts and a list of explained steps
    """
    steps = []
    report = {"rows_raw": int(len(raw))}
    df = raw.copy()

    # 1. Parse date_time -----------------------------------------------------
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    bad_dates = int(df["date_time"].isna().sum())
    df = df.dropna(subset=["date_time"])
    report["invalid_dates_removed"] = bad_dates
    steps.append(
        f"Converted 'date_time' to datetime. Removed {bad_dates} row(s) whose date "
        "could not be parsed (a row without a valid time cannot be used in a time analysis)."
    )

    # 2. Make numeric columns numeric ---------------------------------------
    for col in ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Missing target -----------------------------------------------------
    missing_target = int(df["traffic_volume"].isna().sum())
    df = df.dropna(subset=["traffic_volume"])
    negative_target = int((df["traffic_volume"] < 0).sum())
    df = df[df["traffic_volume"] >= 0]
    report["missing_or_negative_target_removed"] = missing_target + negative_target
    steps.append(
        f"Removed {missing_target + negative_target} row(s) with missing or negative "
        "'traffic_volume' (the target value must exist and a vehicle count cannot be negative)."
    )

    # 3b. Holiday label -> whole calendar day ------------------------------------
    # In this dataset the holiday name is written ONLY on the 00:00 row of the holiday
    # (e.g. 'Christmas Day' at 2012-12-25 00:00; the other 23 hours say 'None').
    # A holiday affects the whole day, so we copy the label to every hour of that date.
    # We do it BEFORE removing duplicates so a label is never lost with a dropped row.
    # (pandas reads the text 'None' as missing, so both cases are handled.)
    has_label = df["holiday"].notna() & (df["holiday"].astype(str) != "None")
    labels = (df.loc[has_label, ["date_time", "holiday"]]
                .assign(_date=lambda d: d["date_time"].dt.normalize())
                .drop_duplicates("_date")
                .set_index("_date")["holiday"])
    report["holiday_days"] = int(len(labels))
    steps.append(
        f"Holiday labels: the raw file writes a holiday name only on the 00:00 row of {len(labels)} "
        "holiday date(s). We copied the label to all hours of that calendar date, because a holiday "
        "changes traffic for the whole day, not just at midnight. Done before duplicate removal so "
        "no label is lost."
    )

    # 4. Exact duplicate rows ------------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    exact_dups = before - len(df)
    report["exact_duplicates_removed"] = int(exact_dups)
    steps.append(
        f"Removed {exact_dups} exact duplicate row(s) (every column identical). "
        "These add no information and would double-count an observation."
    )

    # 5. Repeated timestamps with different weather rows -----------------------
    before = len(df)
    df = df.sort_values("date_time", kind="stable")
    df = df.drop_duplicates(subset="date_time", keep="first")
    dup_times = before - len(df)
    report["duplicate_timestamps_removed"] = int(dup_times)
    steps.append(
        f"Found {dup_times} extra row(s) that share a timestamp with another row "
        "(the original data can list several weather descriptions for the same hour). "
        "The dataset has ONE traffic count per hour, so we kept the first record per "
        "timestamp to get one row = one hour. This avoids counting the same hour twice."
    )

    # 6. Text columns: fill missing categories -------------------------------
    df["holiday"] = (df["date_time"].dt.normalize().map(labels).fillna("None").astype(str))
    df["weather_main"] = df["weather_main"].fillna("Unknown").astype(str)
    df["weather_description"] = df["weather_description"].fillna("Unknown").astype(str)
    steps.append(
        "Filled missing text categories: days without a holiday -> 'None', "
        "missing weather text -> 'Unknown'. A missing label is itself a valid category."
    )

    # 7. Impossible numeric values -> NaN (NOT deleted rows) ------------------
    invalid = {}
    mask = (df["temp"] < MIN_VALID_TEMP_KELVIN) | (df["temp"] > MAX_VALID_TEMP_KELVIN)
    invalid["temp"] = int(mask.sum())
    df.loc[mask, "temp"] = np.nan

    mask = (df["rain_1h"] < 0) | (df["rain_1h"] > MAX_VALID_RAIN_MM_PER_HOUR)
    invalid["rain_1h"] = int(mask.sum())
    df.loc[mask, "rain_1h"] = np.nan

    mask = (df["snow_1h"] < 0) | (df["snow_1h"] > MAX_VALID_SNOW_MM_PER_HOUR)
    invalid["snow_1h"] = int(mask.sum())
    df.loc[mask, "snow_1h"] = np.nan

    mask = (df["clouds_all"] < 0) | (df["clouds_all"] > 100)
    invalid["clouds_all"] = int(mask.sum())
    df.loc[mask, "clouds_all"] = np.nan

    report["invalid_values_set_to_nan"] = invalid
    steps.append(
        "Marked physically impossible weather values as missing (NaN) instead of deleting "
        f"the row: {invalid}. Limits: temperature {MIN_VALID_TEMP_KELVIN}-{MAX_VALID_TEMP_KELVIN} K, "
        f"rain/snow 0-{MAX_VALID_RAIN_MM_PER_HOUR} mm per hour, clouds 0-100 %. "
        "The traffic count in that row is still valid, so the row is kept. "
        "These NaN values are filled later using training data only (no leakage)."
    )

    # 8. Zero traffic: keep but report ---------------------------------------
    zero_traffic = int((df["traffic_volume"] == 0).sum())
    report["zero_traffic_rows_kept"] = zero_traffic
    steps.append(
        f"{zero_traffic} row(s) have traffic_volume = 0. They are KEPT: a zero count can be "
        "genuine (very quiet hour) or a sensor/recording gap, and the dataset does not tell us which. "
        "They are visible in the anomaly section so you can inspect them."
    )

    # 9. Feature engineering --------------------------------------------------
    df = df.sort_values("date_time").reset_index(drop=True)
    df["temp_c"] = df["temp"] - 273.15
    df["hour"] = df["date_time"].dt.hour
    df["day"] = df["date_time"].dt.day
    df["month"] = df["date_time"].dt.month
    df["year"] = df["date_time"].dt.year
    df["day_of_week"] = df["date_time"].dt.dayofweek          # 0 = Monday ... 6 = Sunday
    df["day_name"] = df["day_of_week"].map(lambda i: DAY_NAMES[i])
    df["month_name"] = df["month"].map(lambda i: MONTH_NAMES[i - 1])
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = (df["holiday"] != "None").astype(int)
    df["date"] = df["date_time"].dt.normalize()
    steps.append(
        "Created time features from 'date_time': hour, day, month, year, day_of_week "
        "(0=Monday), day_name, month_name, is_weekend (Sat/Sun = 1), is_holiday and temp_c "
        "(Kelvin - 273.15). Models cannot use a raw timestamp, but they can use these numbers."
    )

    # 10. Hourly gaps (information only) --------------------------------------
    if len(df) > 1:
        gaps = df["date_time"].diff().dropna()
        long_gaps = int((gaps > pd.Timedelta(hours=1)).sum())
        longest_gap_days = float(gaps.max() / pd.Timedelta(days=1))
    else:
        long_gaps, longest_gap_days = 0, 0.0
    report["time_gaps"] = long_gaps
    report["longest_gap_days"] = round(longest_gap_days, 1)
    steps.append(
        f"Checked continuity: {long_gaps} place(s) where consecutive records are more than one hour "
        f"apart (longest gap about {longest_gap_days:.1f} days). Gaps are NOT filled, because "
        "inventing traffic counts would be fake data. Lag-based features are avoided for the same reason."
    )

    report["rows_clean"] = int(len(df))
    report["missing_after_cleaning"] = {c: int(n) for c, n in df.isna().sum().items() if n > 0}
    report["steps"] = steps
    return df, report


def outlier_summary(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Count statistical outliers per column with the IQR rule
    (below Q1 - 1.5*IQR or above Q3 + 1.5*IQR).

    These rows are NOT removed. A statistical outlier can be a perfectly valid
    observation; deleting it would hide real behaviour.
    """
    if columns is None:
        columns = ["traffic_volume", "temp_c", "rain_1h", "snow_1h", "clouds_all"]
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < low) | (series > high)).sum())
        rows.append({
            "column": col,
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "lower_fence": round(float(low), 2),
            "upper_fence": round(float(high), 2),
            "outliers": count,
            "outlier_%": round(100 * count / len(series), 2),
        })
    return pd.DataFrame(rows)


def filter_by_date(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Return rows whose date is between start and end (inclusive)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    mask = (df["date_time"] >= start) & (df["date_time"] < end + pd.Timedelta(days=1))
    return df.loc[mask].copy()
