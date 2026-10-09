"""
peak_analysis.py
----------------
Finds busy hours/days and compares weekday and weekend traffic.
Every number and sentence is calculated from the data you pass in
(for example the date range selected in the dashboard) - nothing is hardcoded.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average, median and count of traffic for each hour of the day (0-23)."""
    out = (df.groupby("hour")["traffic_volume"]
             .agg(mean="mean", median="median", observations="count")
             .reset_index())
    out["mean"] = out["mean"].round(1)
    return out


def top_busiest_hours(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The n hours of the day with the highest AVERAGE traffic volume."""
    prof = hourly_profile(df).sort_values("mean", ascending=False).head(n).copy()
    prof["time_window"] = prof["hour"].map(lambda h: f"{h:02d}:00 - {h:02d}:59")
    return prof[["time_window", "hour", "mean", "median", "observations"]].reset_index(drop=True)


def top_busiest_observations(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The n single hourly records with the highest traffic volume."""
    cols = ["date_time", "day_name", "hour", "traffic_volume", "weather_main", "holiday"]
    return df.nlargest(n, "traffic_volume")[cols].reset_index(drop=True)


def day_of_week_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average traffic for each day of the week (Monday first)."""
    out = (df.groupby(["day_of_week", "day_name"])["traffic_volume"]
             .mean().round(1).reset_index(name="mean_traffic"))
    return out.sort_values("day_of_week").reset_index(drop=True)


def weekday_weekend_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average traffic per hour, separately for weekdays and weekends."""
    tmp = df.copy()
    tmp["day_type"] = tmp["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    out = (tmp.groupby(["hour", "day_type"])["traffic_volume"]
              .mean().round(1).reset_index(name="mean_traffic"))
    return out


def high_traffic_periods(df: pd.DataFrame, quantile: float = 0.75):
    """
    Find hours of the day with CONSISTENTLY high traffic.

    Definition used here:
      threshold = the given quantile (default 75th percentile) of all hourly volumes
      An hour of the day is 'consistently high' if at least 50% of its observations
      are above the threshold.
    Returns (table, threshold).
    """
    threshold = float(df["traffic_volume"].quantile(quantile))
    tmp = df.assign(above=df["traffic_volume"] > threshold)
    out = (tmp.groupby("hour")
              .agg(mean_traffic=("traffic_volume", "mean"),
                   share_above_threshold=("above", "mean"))
              .reset_index())
    out["mean_traffic"] = out["mean_traffic"].round(1)
    out["share_above_threshold"] = (100 * out["share_above_threshold"]).round(1)
    out["consistently_high"] = out["share_above_threshold"] >= 50
    return out, threshold


def _hours_to_ranges(hours) -> str:
    """Turn [7, 8, 15, 16, 17] into '07:00-08:59, 15:00-17:59'."""
    hours = sorted(int(h) for h in hours)
    if not hours:
        return "none"
    ranges, start, prev = [], hours[0], hours[0]
    for h in hours[1:]:
        if h == prev + 1:
            prev = h
            continue
        ranges.append((start, prev))
        start = prev = h
    ranges.append((start, prev))
    return ", ".join(f"{a:02d}:00-{b:02d}:59" for a, b in ranges)


def generate_observations(df: pd.DataFrame) -> list:
    """Return a list of plain-English findings calculated from df."""
    if df.empty:
        return ["No data in the selected range."]

    notes = []
    prof = hourly_profile(df)
    busiest = prof.loc[prof["mean"].idxmax()]
    quietest = prof.loc[prof["mean"].idxmin()]
    notes.append(
        f"Busiest hour of the day on average: {int(busiest['hour']):02d}:00 "
        f"(about {busiest['mean']:,.0f} vehicles/hour). Quietest: {int(quietest['hour']):02d}:00 "
        f"(about {quietest['mean']:,.0f} vehicles/hour)."
    )

    dow = day_of_week_profile(df)
    best, worst = dow.loc[dow["mean_traffic"].idxmax()], dow.loc[dow["mean_traffic"].idxmin()]
    notes.append(
        f"Busiest day of the week on average: {best['day_name']} ({best['mean_traffic']:,.0f}); "
        f"quietest: {worst['day_name']} ({worst['mean_traffic']:,.0f})."
    )

    wk = df.loc[df["is_weekend"] == 0, "traffic_volume"]
    we = df.loc[df["is_weekend"] == 1, "traffic_volume"]
    if len(wk) and len(we):
        diff = 100 * (wk.mean() - we.mean()) / we.mean() if we.mean() else float("nan")
        direction = "higher" if diff >= 0 else "lower"
        notes.append(
            f"Weekday average ({wk.mean():,.0f}) is {abs(diff):.1f}% {direction} than the weekend "
            f"average ({we.mean():,.0f}) in this selection."
        )
        # Peak hour separately for weekday and weekend
        wprof = weekday_weekend_profile(df)
        pk = wprof.loc[wprof.groupby("day_type")["mean_traffic"].idxmax()]
        text = "; ".join(f"{r.day_type} peak at {int(r.hour):02d}:00 ({r.mean_traffic:,.0f})"
                         for r in pk.itertuples())
        notes.append(f"Peak hour by day type: {text}.")

    table, thr = high_traffic_periods(df)
    high_hours = table.loc[table["consistently_high"], "hour"].tolist()
    notes.append(
        f"Hours where at least half of all observations exceed {thr:,.0f} vehicles/hour "
        f"(the 75th percentile): {_hours_to_ranges(high_hours)}."
    )

    top1 = top_busiest_observations(df, 1).iloc[0]
    notes.append(
        f"Single highest record: {int(top1['traffic_volume']):,} vehicles on "
        f"{top1['date_time']:%d %b %Y} at {int(top1['hour']):02d}:00 ({top1['day_name']})."
    )
    return notes


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def fig_top_hours(df: pd.DataFrame, n: int = 5) -> go.Figure:
    top = top_busiest_hours(df, n)
    fig = px.bar(top, x="time_window", y="mean",
                 labels={"time_window": "Hour of day", "mean": "Average traffic volume (vehicles/hour)"},
                 title=f"Top {n} busiest hours of the day (by average volume)",
                 text="mean", color_discrete_sequence=["#2563eb"])
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    return fig


def fig_weekday_vs_weekend(df: pd.DataFrame) -> go.Figure:
    prof = weekday_weekend_profile(df)
    fig = px.line(prof, x="hour", y="mean_traffic", color="day_type", markers=True,
                  labels={"hour": "Hour of day", "mean_traffic": "Average traffic volume (vehicles/hour)",
                          "day_type": "Day type"},
                  title="Average hourly traffic: weekdays vs weekends",
                  color_discrete_map={"Weekday": "#2563eb", "Weekend": "#f97316"})
    fig.update_xaxes(dtick=2)
    return fig


def fig_high_traffic_hours(df: pd.DataFrame) -> go.Figure:
    table, thr = high_traffic_periods(df)
    colors = table["consistently_high"].map({True: "#dc2626", False: "#94a3b8"})
    fig = go.Figure(go.Bar(x=table["hour"], y=table["share_above_threshold"],
                           marker_color=colors,
                           hovertemplate="Hour %{x}:00<br>%{y:.1f}% of records above threshold<extra></extra>"))
    fig.add_hline(y=50, line_dash="dash", line_color="#475569",
                  annotation_text="50% line", annotation_position="top left")
    fig.update_layout(title=f"Share of records above {thr:,.0f} vehicles/hour, by hour (red = consistently high)",
                      xaxis_title="Hour of day", yaxis_title="% of records above threshold")
    fig.update_xaxes(dtick=2)
    return fig
