"""
clustering.py  (OPTIONAL feature G)
-----------------------------------
K-Means clustering of hourly observations into statistical traffic patterns.

Features used: traffic_volume, hour (as sin/cos so that 23:00 is close to 00:00),
is_weekend, temp_c and clouds_all.
All features are standardised (mean 0, std 1) before clustering because K-Means uses
distances and would otherwise be dominated by the feature with the biggest numbers
(traffic_volume is in the thousands, is_weekend is 0 or 1).

Clusters are STATISTICAL groupings. They are not guaranteed real-world categories.
"""

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CLUSTER_BASE_COLUMNS = ["traffic_volume", "hour", "is_weekend", "temp_c", "clouds_all"]


def run_kmeans(df: pd.DataFrame, k: int = 4, random_state: int = 42):
    """
    Returns (df_with_cluster_column, summary_table, silhouette_score).
    Rows with missing weather values are excluded from clustering only.
    """
    data = df.dropna(subset=CLUSTER_BASE_COLUMNS).copy()
    if len(data) < k * 10:
        raise ValueError("Not enough rows to cluster. Choose a larger date range or fewer clusters.")

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    features = ["traffic_volume", "hour_sin", "hour_cos", "is_weekend", "temp_c", "clouds_all"]

    scaled = StandardScaler().fit_transform(data[features])
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    data["cluster"] = model.fit_predict(scaled)

    # Silhouette on a sample (it is slow on very large data)
    sample = min(5000, len(data))
    score = float(silhouette_score(scaled, data["cluster"], sample_size=sample, random_state=random_state))

    summary = (data.groupby("cluster")
                   .agg(observations=("traffic_volume", "size"),
                        avg_traffic=("traffic_volume", "mean"),
                        avg_hour=("hour", "mean"),
                        weekend_share=("is_weekend", "mean"),
                        avg_temp_c=("temp_c", "mean"),
                        avg_clouds=("clouds_all", "mean"))
                   .round(2).reset_index())
    summary["weekend_share"] = (100 * summary["weekend_share"]).round(1)
    summary = summary.rename(columns={"weekend_share": "weekend_%"})
    summary["description"] = summary.apply(lambda r: describe_cluster(r, data), axis=1)
    data["cluster"] = data["cluster"].astype(str)
    return data, summary, score


def describe_cluster(row, data: pd.DataFrame) -> str:
    """Plain-English description generated from the cluster's own statistics."""
    overall = data["traffic_volume"].mean()
    ratio = row["avg_traffic"] / overall if overall else 1
    if ratio >= 1.15:
        level = "Higher-than-average traffic"
    elif ratio <= 0.6:
        level = "Very low traffic"
    elif ratio <= 0.9:
        level = "Lower-than-average traffic"
    else:
        level = "Around-average traffic"
    day = "mostly weekends" if row["weekend_%"] >= 60 else (
        "mostly weekdays" if row["weekend_%"] <= 15 else "mixed weekdays/weekends")
    cl = int(row["cluster"])
    hours = data.loc[data["cluster"] == cl, "hour"]
    common = sorted(hours.value_counts().head(3).index)       # 3 most frequent hours (no wrap-around problem)
    common_text = ", ".join(f"{int(h):02d}:00" for h in common)
    return f"{level}, {day}; most common hours: {common_text}"


def fig_clusters(data: pd.DataFrame, max_points: int = 6000):
    """Scatter plot: hour vs traffic volume coloured by cluster."""
    plot_df = data.sample(min(max_points, len(data)), random_state=1).copy()
    plot_df["day_type"] = plot_df["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    fig = px.scatter(plot_df, x="hour", y="traffic_volume", color="cluster",
                     symbol="day_type", opacity=0.6,
                     hover_data=["date_time", "temp_c", "clouds_all"],
                     labels={"hour": "Hour of day", "traffic_volume": "Traffic volume (vehicles/hour)",
                             "cluster": "Cluster", "day_type": "Day type"},
                     title="K-Means clusters of hourly traffic observations (random sample of points)")
    return fig
