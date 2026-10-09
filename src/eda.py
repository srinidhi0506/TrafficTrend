"""
eda.py
------
Exploratory Data Analysis charts (Plotly). Each function returns
(figure, interpretation_text). The text is calculated from the data passed in,
so it changes when you change the date filter.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import DAY_NAMES, MONTH_NAMES

BLUE = "#2563eb"
Y_LABEL = "Traffic volume (vehicles/hour)"


def _empty(msg="No data in the selected range."):
    fig = go.Figure()
    fig.update_layout(title=msg)
    return fig, msg


def fig_distribution(df: pd.DataFrame):
    if df.empty:
        return _empty()
    fig = px.histogram(df, x="traffic_volume", nbins=50, color_discrete_sequence=[BLUE],
                       labels={"traffic_volume": Y_LABEL, "count": "Number of hours"},
                       title="Distribution of hourly traffic volume")
    fig.update_layout(yaxis_title="Number of hourly records", bargap=0.03)
    s = df["traffic_volume"]
    text = (f"Mean = {s.mean():,.0f}, median = {s.median():,.0f}, standard deviation = {s.std():,.0f}, "
            f"skewness = {s.skew():.2f}. ")
    low_share = 100 * (s < s.quantile(0.25)).mean()
    # Detect number of peaks from the histogram counts (a peak = a bin higher than both neighbours,
    # after light smoothing) so the sentence below is calculated, not assumed.
    counts, _ = np.histogram(s, bins=20)
    smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    peaks = int(sum(1 for i in range(1, len(smooth) - 1)
                    if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] > 0.3 * smooth.max()))
    shape = (f"The histogram shows about {peaks} distinct peak(s)"
             + (", which suggests different traffic regimes (e.g. quiet hours vs busy hours). " if peaks >= 2 else ". "))
    text += shape + f"{low_share:.0f}% of records fall below the 25th percentile ({s.quantile(0.25):,.0f})."
    return fig, text


def fig_by_hour(df: pd.DataFrame):
    if df.empty:
        return _empty()
    g = df.groupby("hour")["traffic_volume"].agg(["mean", "std"]).reset_index()
    fig = go.Figure(go.Bar(x=g["hour"], y=g["mean"], marker_color=BLUE,
                           error_y=dict(type="data", array=g["std"].fillna(0), visible=True, color="#94a3b8"),
                           hovertemplate="Hour %{x}:00<br>Average: %{y:,.0f}<extra></extra>"))
    fig.update_layout(title="Average traffic volume by hour of day (bars = mean, whiskers = std. deviation)",
                      xaxis_title="Hour of day (0-23)", yaxis_title="Average " + Y_LABEL.lower())
    fig.update_xaxes(dtick=1)
    hi, lo = g.loc[g["mean"].idxmax()], g.loc[g["mean"].idxmin()]
    text = (f"Traffic is highest around {int(hi['hour']):02d}:00 (average {hi['mean']:,.0f}) and lowest around "
            f"{int(lo['hour']):02d}:00 (average {lo['mean']:,.0f}). Long whiskers mean that the same hour varies a lot "
            "between days (e.g. weekday vs weekend).")
    return fig, text


def fig_by_day(df: pd.DataFrame):
    if df.empty:
        return _empty()
    g = df.groupby(["day_of_week"])["traffic_volume"].mean().reindex(range(7)).reset_index()
    g["day_name"] = [DAY_NAMES[i] for i in g["day_of_week"]]
    fig = px.bar(g, x="day_name", y="traffic_volume", color_discrete_sequence=[BLUE],
                 labels={"day_name": "Day of week", "traffic_volume": "Average " + Y_LABEL.lower()},
                 title="Average traffic volume by day of week")
    fig.update_traces(hovertemplate="%{x}<br>Average: %{y:,.0f}<extra></extra>")
    g2 = g.dropna()
    hi, lo = g2.loc[g2["traffic_volume"].idxmax()], g2.loc[g2["traffic_volume"].idxmin()]
    text = (f"{hi['day_name']} has the highest average ({hi['traffic_volume']:,.0f}); {lo['day_name']} has the lowest "
            f"({lo['traffic_volume']:,.0f}). The gap between them is "
            f"{100 * (hi['traffic_volume'] - lo['traffic_volume']) / lo['traffic_volume']:.1f}% of the lowest value.")
    return fig, text


def fig_by_month(df: pd.DataFrame):
    if df.empty:
        return _empty()
    g = df.groupby("month")["traffic_volume"].mean().reindex(range(1, 13)).reset_index()
    g["month_name"] = [MONTH_NAMES[i - 1] for i in g["month"]]
    fig = px.bar(g, x="month_name", y="traffic_volume", color_discrete_sequence=[BLUE],
                 labels={"month_name": "Month", "traffic_volume": "Average " + Y_LABEL.lower()},
                 title="Average traffic volume by month")
    fig.update_traces(hovertemplate="%{x}<br>Average: %{y:,.0f}<extra></extra>")
    g2 = g.dropna()
    hi, lo = g2.loc[g2["traffic_volume"].idxmax()], g2.loc[g2["traffic_volume"].idxmin()]
    text = (f"Highest monthly average: {hi['month_name']} ({hi['traffic_volume']:,.0f}); lowest: {lo['month_name']} "
            f"({lo['traffic_volume']:,.0f}). Note: months are not covered equally in every year (the data has gaps), "
            "so treat monthly differences as indicative rather than exact.")
    return fig, text


def fig_heatmap(df: pd.DataFrame):
    if df.empty:
        return _empty()
    pivot = (df.pivot_table(index="day_of_week", columns="hour", values="traffic_volume", aggfunc="mean")
               .reindex(index=range(7), columns=range(24)))
    fig = go.Figure(go.Heatmap(z=pivot.values, x=list(range(24)), y=DAY_NAMES, colorscale="YlOrRd",
                               colorbar=dict(title="Avg vehicles/hour"),
                               hovertemplate="%{y}, %{x}:00<br>Average: %{z:,.0f}<extra></extra>"))
    fig.update_layout(title="Average traffic volume: day of week x hour of day",
                      xaxis_title="Hour of day", yaxis_title="Day of week", yaxis=dict(autorange="reversed"))
    fig.update_xaxes(dtick=1)
    stacked = pivot.stack()
    (d, h), top = stacked.idxmax(), stacked.max()
    (d2, h2), low = stacked.idxmin(), stacked.min()
    text = (f"The hottest cell is {DAY_NAMES[d]} at {h:02d}:00 (average {top:,.0f}); the coolest is {DAY_NAMES[d2]} "
            f"at {h2:02d}:00 ({low:,.0f}). Compare weekday rows with Saturday/Sunday rows to see how the daily "
            "rhythm changes.")
    return fig, text


def fig_time_series(df: pd.DataFrame, resample: str = "Daily"):
    if df.empty:
        return _empty()
    rule = {"Hourly": None, "Daily": "D", "Weekly": "W", "Monthly": "MS"}[resample]
    ts = df.set_index("date_time")["traffic_volume"]
    if rule:
        ts = ts.resample(rule).mean().dropna()   # empty periods (data gaps) are dropped, not invented
    fig = px.line(ts.reset_index(), x="date_time", y="traffic_volume",
                  labels={"date_time": "Date", "traffic_volume": f"{resample} average traffic volume"},
                  title=f"Traffic volume over time ({resample.lower()} average)")
    fig.update_traces(line_color=BLUE, connectgaps=False)
    fig.update_xaxes(rangeslider_visible=True)
    text = (f"Shows {len(ts):,} {resample.lower()} points from {ts.index.min():%d %b %Y} to {ts.index.max():%d %b %Y}. "
            "Straight lines across long stretches indicate gaps in the recorded data, not real traffic behaviour. "
            "Use the range slider to zoom.")
    return fig, text


WEATHER_CHOICES = {
    "Temperature (C)": "temp_c",
    "Rain in last hour (mm)": "rain_1h",
    "Snow in last hour (mm)": "snow_1h",
    "Cloud cover (%)": "clouds_all",
}


def fig_weather_numeric(df: pd.DataFrame, label: str):
    col = WEATHER_CHOICES[label]
    d = df[[col, "traffic_volume"]].dropna()
    if d.empty:
        return _empty()
    sample = d.sample(min(4000, len(d)), random_state=0)
    fig = px.scatter(sample, x=col, y="traffic_volume", opacity=0.35, color_discrete_sequence=["#64748b"],
                     labels={col: label, "traffic_volume": Y_LABEL},
                     title=f"{label} vs traffic volume (sample of up to 4,000 hours)")
    # Binned average line so that the trend is visible
    if d[col].nunique() > 10:
        bins = pd.qcut(d[col], q=min(12, d[col].nunique()), duplicates="drop")
        trend = d.groupby(bins, observed=True).agg(x=(col, "mean"), y=("traffic_volume", "mean")).reset_index(drop=True)
        fig.add_trace(go.Scatter(x=trend["x"], y=trend["y"], mode="lines+markers", name="Binned average",
                                 line=dict(color="#dc2626", width=3)))
    r = d[col].corr(d["traffic_volume"])
    strength = "very weak" if abs(r) < 0.1 else "weak" if abs(r) < 0.3 else "moderate" if abs(r) < 0.5 else "strong"
    text = (f"Pearson correlation = {r:.3f} ({strength} linear relationship). Correlation does not prove causation: "
            "weather and time of day are related (it is colder at night), and time of day strongly drives traffic.")
    return fig, text


def fig_weather_category(df: pd.DataFrame):
    if df.empty:
        return _empty()
    g = (df.groupby("weather_main")["traffic_volume"].agg(["mean", "count"]).reset_index()
           .sort_values("mean", ascending=False))
    fig = px.bar(g, x="weather_main", y="mean", color_discrete_sequence=[BLUE], hover_data={"count": True},
                 labels={"weather_main": "Weather condition", "mean": "Average " + Y_LABEL.lower(),
                         "count": "Hours observed"},
                 title="Average traffic volume by weather condition")
    hi, lo = g.iloc[0], g.iloc[-1]
    text = (f"Highest average: {hi['weather_main']} ({hi['mean']:,.0f}, n={int(hi['count'])}); lowest: "
            f"{lo['weather_main']} ({lo['mean']:,.0f}, n={int(lo['count'])}). Categories with few hours "
            "are unreliable. Differences can also reflect WHEN the weather happens (e.g. fog at dawn), not only weather itself.")
    return fig, text


def fig_correlation(df: pd.DataFrame):
    cols = ["traffic_volume", "hour", "day_of_week", "month", "is_weekend", "is_holiday",
            "temp_c", "rain_1h", "snow_1h", "clouds_all"]
    corr = df[cols].corr()
    if df.empty or corr.isna().all().all():
        return _empty()
    fig = px.imshow(corr.round(2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    title="Correlation heatmap (Pearson) of numerical variables", aspect="auto")
    target = corr["traffic_volume"].drop("traffic_volume").dropna()
    top = target.abs().sort_values(ascending=False).head(3).index
    items = ", ".join(f"{c} ({target[c]:+.2f})" for c in top)
    text = (f"Strongest linear relationships with traffic_volume: {items}. Pearson correlation only captures "
            "straight-line relationships; 'hour' has a curved (rush-hour) pattern, so its correlation understates its importance.")
    return fig, text
