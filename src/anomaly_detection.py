"""
anomaly_detection.py
--------------------
Explainable anomaly detection with the Interquartile Range (IQR) rule.

WHY IQR (and not Isolation Forest)?
  * It is easy to explain in a viva: "a value is unusual if it is far outside the
    middle 50% of similar observations".
  * It makes no assumption that data is normally distributed.
  * We can show the exact lower/upper limits for every observation.

WHY compare WITHIN the same hour-of-day and day-type (weekday/weekend)?
  Traffic at 8 a.m. is naturally very different from traffic at 3 a.m. If we used
  one global limit, a normal rush hour could look 'high' and a normal night hour
  could look 'low'. Comparing like with like (e.g. weekday 8 a.m. vs other weekday
  8 a.m.) finds observations that are unusual FOR THEIR OWN TIME SLOT.

IMPORTANT: A flagged observation is only a statistical anomaly. It is NOT proof of
an accident, road closure, sensor failure or any real incident.
"""

import pandas as pd
import plotly.graph_objects as go

DEFAULT_GROUP_COLUMNS = ["hour", "is_weekend"]


def detect_iqr_anomalies(df: pd.DataFrame, multiplier: float = 1.5,
                         group_columns=None) -> pd.DataFrame:
    """
    Add anomaly columns to a copy of df:
      lower_limit, upper_limit : IQR limits for the observation's group
      is_anomaly               : True if outside the limits
      anomaly_type             : 'High', 'Low' or 'Normal'
    group_columns=[] means one global limit for all rows.
    """
    if group_columns is None:
        group_columns = DEFAULT_GROUP_COLUMNS
    out = df.copy()
    target = out["traffic_volume"]

    if group_columns:
        grouped = out.groupby(group_columns)["traffic_volume"]
        q1 = grouped.transform(lambda s: s.quantile(0.25))
        q3 = grouped.transform(lambda s: s.quantile(0.75))
    else:
        q1 = pd.Series(target.quantile(0.25), index=out.index)
        q3 = pd.Series(target.quantile(0.75), index=out.index)

    iqr = q3 - q1
    out["lower_limit"] = q1 - multiplier * iqr
    out["upper_limit"] = q3 + multiplier * iqr
    out["is_anomaly"] = (target < out["lower_limit"]) | (target > out["upper_limit"])
    out["anomaly_type"] = "Normal"
    out.loc[target > out["upper_limit"], "anomaly_type"] = "High"
    out.loc[target < out["lower_limit"], "anomaly_type"] = "Low"
    return out


def anomaly_table(result: pd.DataFrame) -> pd.DataFrame:
    """Table of flagged observations, most extreme first."""
    flagged = result[result["is_anomaly"]].copy()
    if flagged.empty:
        return flagged
    # How far outside the limits (in vehicles)?
    flagged["distance_outside_limit"] = (
        (flagged["traffic_volume"] - flagged["upper_limit"]).clip(lower=0)
        + (flagged["lower_limit"] - flagged["traffic_volume"]).clip(lower=0)
    ).round(0)
    cols = ["date_time", "day_name", "hour", "traffic_volume", "anomaly_type",
            "lower_limit", "upper_limit", "distance_outside_limit", "weather_main", "holiday"]
    flagged = flagged[cols].sort_values("distance_outside_limit", ascending=False)
    flagged["lower_limit"] = flagged["lower_limit"].round(0)
    flagged["upper_limit"] = flagged["upper_limit"].round(0)
    return flagged.reset_index(drop=True)


def anomaly_counts(result: pd.DataFrame) -> dict:
    return {
        "total": int(len(result)),
        "flagged": int(result["is_anomaly"].sum()),
        "high": int((result["anomaly_type"] == "High").sum()),
        "low": int((result["anomaly_type"] == "Low").sum()),
        "percent": round(100 * float(result["is_anomaly"].mean()), 2) if len(result) else 0.0,
    }


def fig_anomalies(result: pd.DataFrame) -> go.Figure:
    """Traffic over time with flagged observations highlighted in red/orange."""
    normal = result[~result["is_anomaly"]]
    high = result[result["anomaly_type"] == "High"]
    low = result[result["anomaly_type"] == "Low"]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=normal["date_time"], y=normal["traffic_volume"], mode="markers",
                               name="Normal", marker=dict(size=3, color="#94a3b8", opacity=0.45)))
    fig.add_trace(go.Scattergl(x=high["date_time"], y=high["traffic_volume"], mode="markers",
                               name="Unusually high", marker=dict(size=6, color="#dc2626")))
    fig.add_trace(go.Scattergl(x=low["date_time"], y=low["traffic_volume"], mode="markers",
                               name="Unusually low", marker=dict(size=6, color="#f97316")))
    fig.update_layout(title="Hourly traffic volume with statistically unusual observations highlighted",
                      xaxis_title="Date and time", yaxis_title="Traffic volume (vehicles/hour)",
                      legend=dict(orientation="h", y=1.08))
    return fig


def fig_anomalies_by_hour(result: pd.DataFrame) -> go.Figure:
    counts = (result[result["is_anomaly"]].groupby(["hour", "anomaly_type"])
              .size().reset_index(name="count"))
    fig = go.Figure()
    for kind, color in [("High", "#dc2626"), ("Low", "#f97316")]:
        sub = counts[counts["anomaly_type"] == kind]
        fig.add_trace(go.Bar(x=sub["hour"], y=sub["count"], name=kind, marker_color=color))
    fig.update_layout(barmode="stack", title="Number of flagged observations by hour of day",
                      xaxis_title="Hour of day", yaxis_title="Flagged observations")
    fig.update_xaxes(dtick=2)
    return fig
