"""
app.py - TrafficTrend dashboard
Run from the project root with:   python -m streamlit run app.py
"""

import inspect

import pandas as pd
import plotly.express as px
import streamlit as st

from src import anomaly_detection as ad
from src import clustering as cl
from src import eda
from src import peak_analysis as pk
from src import train_model as tm
from src.data_loader import (DATASET_URL, DatasetNotFoundError, MissingColumnsError,
                             describe_raw_data, load_raw_data)
from src.preprocessing import DAY_NAMES, MONTH_NAMES, clean_data, filter_by_date, outlier_summary

st.set_page_config(page_title="TrafficTrend", page_icon="🚦", layout="wide")

# ---------------------------------------------------------------------------
# Styling (works in light and dark themes because backgrounds are translucent)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 60%, #0ea5e9 100%); color: white;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: .3rem 0 0 0; opacity: .92;}
.kpi {border: 1px solid rgba(100,116,139,.35); border-radius: 12px; padding: .9rem 1rem;
      background: rgba(37,99,235,.07);}
.kpi .label {font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; opacity: .75;}
.kpi .value {font-size: 1.7rem; font-weight: 700; line-height: 1.2;}
.kpi .sub {font-size: .78rem; opacity: .7;}
.note {border-left: 4px solid #2563eb; padding: .5rem .9rem; margin: .4rem 0 1rem 0;
       background: rgba(37,99,235,.07); border-radius: 0 8px 8px 0; font-size: .93rem;}
.warn {border-left: 4px solid #f59e0b; padding: .5rem .9rem; margin: .4rem 0 1rem 0;
       background: rgba(245,158,11,.10); border-radius: 0 8px 8px 0; font-size: .93rem;}
</style>
""", unsafe_allow_html=True)

_NEW_PLOTLY_API = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig, key=None):
    """Display a Plotly figure full-width (works on old and new Streamlit versions)."""
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), template="plotly_white")
    if _NEW_PLOTLY_API:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def note(text, warn=False):
    st.markdown(f'<div class="{"warn" if warn else "note"}">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data / model loading (so reruns are fast)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_and_clean():
    raw = load_raw_data()
    raw_info = describe_raw_data(raw)
    clean, report = clean_data(raw)
    return clean, report, raw_info, raw.head(10)


@st.cache_resource(show_spinner="Training models (first run only, may take a minute)...")
def train_cached(_df, n_rows, test_fraction):
    # _df is not hashed by Streamlit; n_rows and test_fraction make the cache key.
    return tm.train_and_evaluate(_df, test_fraction=test_fraction, save=True)


try:
    data, report, raw_info, raw_head = load_and_clean()
except DatasetNotFoundError as err:
    st.markdown('<div class="hero"><h1>🚦 TrafficTrend</h1><p>Dataset not found</p></div>', unsafe_allow_html=True)
    st.error("Dataset file not found - the dashboard cannot show results without the real data.")
    st.code(str(err))
    st.stop()
except (MissingColumnsError, ValueError) as err:
    st.error("The dataset could not be used.")
    st.code(str(err))
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🚦 TrafficTrend")
PAGES = ["Overview", "Data Quality", "Exploratory Analysis", "Peak Traffic Analysis",
         "Traffic Prediction", "Unusual Traffic Detection", "Traffic Pattern Clustering", "About & Guide"]
page = st.sidebar.radio("Navigate", PAGES)

st.sidebar.markdown("---")
st.sidebar.subheader("Date filter")
min_d, max_d = data["date_time"].min().date(), data["date_time"].max().date()
picked = st.sidebar.date_input("Analysis period", value=(min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(picked, (list, tuple)) and len(picked) == 2:
    start_d, end_d = picked
else:
    start_d, end_d = min_d, max_d     # user is still choosing the second date
df = filter_by_date(data, start_d, end_d)
st.sidebar.caption(f"{len(df):,} hourly records selected.\n\nThe date filter applies to every page except model "
                   "training, which always uses the full dataset.")
st.sidebar.markdown("---")
st.sidebar.caption("Data: UCI Metro Interstate Traffic Volume (I-94 Westbound, Minneapolis-St. Paul, USA). "
                   "Not Hyderabad data.")

st.markdown('<div class="hero"><h1>🚦 TrafficTrend</h1>'
            '<p>Traffic congestion pattern analysis and traffic volume prediction using historical hourly data '
            'from a highway in the Minneapolis-Saint Paul area (USA).</p></div>', unsafe_allow_html=True)

if df.empty:
    st.warning("No records in the selected date range. Choose a different period in the sidebar.")
    st.stop()

# ===========================================================================
# PAGES
# ===========================================================================
if page == "Overview":
    prof = pk.hourly_profile(df)
    busiest = prof.loc[prof["mean"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total observations", f"{len(df):,}", "hourly records in selected period")
    kpi(c2, "Average traffic volume", f"{df['traffic_volume'].mean():,.0f}", "vehicles per hour")
    kpi(c3, "Maximum recorded volume", f"{int(df['traffic_volume'].max()):,}", "vehicles in one hour")
    kpi(c4, "Busiest hour (average)", f"{int(busiest['hour']):02d}:00", f"avg {busiest['mean']:,.0f} vehicles/hour")
    st.write("")
    note("<b>Where is this data from?</b> The dataset (UCI repository) records hourly westbound traffic volume on "
         "Interstate 94 between Minneapolis and St. Paul, Minnesota, USA, together with weather and holiday "
         "information. It does <b>not</b> represent Hyderabad or any Indian road.")
    st.subheader("Key insights (calculated from the selected period)")
    for line in pk.generate_observations(df):
        st.markdown(f"- {line}")
    st.subheader("Traffic over time")
    resample = st.radio("Averaging window", ["Daily", "Weekly", "Monthly", "Hourly"], index=1, horizontal=True)
    fig, text = eda.fig_time_series(df, resample)
    show(fig)
    note(text)

elif page == "Data Quality":
    st.header("Data loading and preprocessing")
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Raw rows", f"{raw_info['rows']:,}", f"{raw_info['columns']} columns")
    kpi(c2, "Rows after cleaning", f"{report['rows_clean']:,}", "one row per hour")
    kpi(c3, "Rows removed", f"{report['rows_raw'] - report['rows_clean']:,}", "duplicates / invalid rows")
    st.write("")
    st.subheader("Raw dataset structure")
    st.write("**Column names:**", ", ".join(raw_info["column_names"]))
    st.dataframe(pd.DataFrame({"column": list(raw_info["dtypes"]), "data type": list(raw_info["dtypes"].values()),
                               "missing values": [raw_info["missing_per_column"][c] for c in raw_info["dtypes"]]}),
                 hide_index=True)
    st.write("**First 10 raw rows:**")
    st.dataframe(raw_head, hide_index=True)
    st.subheader("Preprocessing decisions (each step explained)")
    for i, step in enumerate(report["steps"], 1):
        st.markdown(f"**{i}.** {step}")
    st.subheader("Outlier check (IQR rule) - flagged, NOT deleted")
    note("<b>Unusual but valid vs data error:</b> a data error is impossible (e.g. 0 Kelvin temperature) and was "
         "set to missing. A statistical outlier (e.g. heavy rain) is rare but possible, so it is kept. "
         "Rain and snow are 0 most of the time, so any rainy hour looks like an 'outlier' to the IQR rule - "
         "that does not make it wrong.")
    st.dataframe(outlier_summary(data), hide_index=True)
    st.subheader("Missing values remaining after cleaning")
    st.write(report["missing_after_cleaning"] or "None")
    st.caption("Remaining missing weather values are filled later inside the model pipeline using training data only.")

elif page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    tabs = st.tabs(["Distribution", "By hour", "By day", "By month", "Heatmap", "Over time", "Weather", "Correlation"])
    with tabs[0]:
        fig, text = eda.fig_distribution(df); show(fig); note(text)
    with tabs[1]:
        fig, text = eda.fig_by_hour(df); show(fig); note(text)
    with tabs[2]:
        fig, text = eda.fig_by_day(df); show(fig); note(text)
    with tabs[3]:
        fig, text = eda.fig_by_month(df); show(fig); note(text)
    with tabs[4]:
        fig, text = eda.fig_heatmap(df); show(fig); note(text)
    with tabs[5]:
        res = st.radio("Averaging window", ["Daily", "Weekly", "Monthly", "Hourly"], index=1, horizontal=True, key="eda_res")
        fig, text = eda.fig_time_series(df, res); show(fig); note(text)
    with tabs[6]:
        choice = st.selectbox("Numeric weather variable", list(eda.WEATHER_CHOICES))
        fig, text = eda.fig_weather_numeric(df, choice); show(fig); note(text)
        fig, text = eda.fig_weather_category(df); show(fig); note(text)
    with tabs[7]:
        fig, text = eda.fig_correlation(df); show(fig); note(text)

elif page == "Peak Traffic Analysis":
    st.header("Peak traffic analysis")
    col1, col2 = st.columns(2)
    with col1:
        show(pk.fig_top_hours(df, 5))
    with col2:
        show(pk.fig_weekday_vs_weekend(df))
    show(pk.fig_high_traffic_hours(df))
    st.subheader("Data-driven observations")
    for line in pk.generate_observations(df):
        st.markdown(f"- {line}")
    st.subheader("Top 5 busiest single hours recorded")
    st.dataframe(pk.top_busiest_observations(df, 5), hide_index=True)
    st.subheader("Top 5 busiest hours of the day (average)")
    st.dataframe(pk.top_busiest_hours(df, 5), hide_index=True)

elif page == "Traffic Prediction":
    st.header("Traffic volume prediction")
    note("<b>Prediction is not the same as forecasting.</b> These models predict the traffic of an hour <i>given</i> its "
         "calendar information and its weather. To forecast tomorrow's traffic you would also need tomorrow's weather "
         "forecast, which is not part of this dataset. Past traffic values are deliberately not used as inputs.", warn=True)
    st.subheader("Features and when they are available")
    st.dataframe(tm.FEATURE_AVAILABILITY, hide_index=True)

    test_pct = st.sidebar.slider("Test set size (latest data)", 10, 40, 20, step=5, help="Used on the Prediction page")
    out = train_cached(data, len(data), test_pct / 100)
    results, preds, test = out["results"], out["predictions"], out["test"]

    st.write(f"**Chronological split:** training = {out['train']['date_time'].min():%d %b %Y} to "
             f"{out['train']['date_time'].max():%d %b %Y} ({len(out['train']):,} rows); testing = "
             f"{test['date_time'].min():%d %b %Y} to {test['date_time'].max():%d %b %Y} ({len(test):,} rows).")
    st.subheader("Model comparison on the unseen test period")
    st.dataframe(results.round(3), hide_index=True)
    melt = results.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig = px.bar(melt, x="Model", y="Value", color="Model", facet_col="Metric",
                 title="Model comparison (lower MAE/RMSE is better; higher R2 is better)",
                 labels={"Value": "Metric value"})
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(showticklabels=False, title="")
    show(fig)
    best = out["best_model_name"]
    b = results[results["Model"] == best].iloc[0]
    base = results[results["Model"].str.startswith("Baseline")].iloc[0]
    note(f"Best model by RMSE on this test period: <b>{best}</b> (MAE {b['MAE']:,.0f}, RMSE {b['RMSE']:,.0f}, "
         f"R2 {b['R2']:.3f}). The baseline that always predicts the training average has RMSE {base['RMSE']:,.0f}. "
         f"These numbers describe this test period only. The best model was saved to <code>models/{tm.MODEL_PATH.name}</code>.")

    st.subheader("Actual vs predicted traffic")
    model_name = st.selectbox("Model to inspect", [m for m in preds if not m.startswith("Baseline")],
                              index=[m for m in preds if not m.startswith("Baseline")].index(best))
    t_min, t_max = test["date_time"].min().date(), test["date_time"].max().date()
    c1, c2 = st.columns(2)
    start = c1.date_input("Window start", value=t_min, min_value=t_min, max_value=t_max)
    days = c2.slider("Window length (days)", 1, 30, 7)
    view = test.assign(Predicted=preds[model_name]).rename(columns={"traffic_volume": "Actual"})
    view = view[(view["date_time"] >= pd.Timestamp(start)) & (view["date_time"] < pd.Timestamp(start) + pd.Timedelta(days=days))]
    if view.empty:
        st.info("No test records in that window - pick another start date.")
    else:
        fig = px.line(view, x="date_time", y=["Actual", "Predicted"], markers=True,
                      labels={"date_time": "Date and time", "value": "Traffic volume (vehicles/hour)", "variable": ""},
                      title=f"Actual vs predicted - {model_name}",
                      color_discrete_map={"Actual": "#2563eb", "Predicted": "#f97316"})
        show(fig)
    samp = test.assign(Predicted=preds[model_name]).rename(columns={"traffic_volume": "Actual"})
    samp = samp.sample(min(3000, len(samp)), random_state=0)
    fig = px.scatter(samp, x="Actual", y="Predicted", opacity=0.4, color_discrete_sequence=["#2563eb"],
                     hover_data=["date_time"], title=f"Actual vs predicted scatter - {model_name} (points on the diagonal = perfect)")
    mx = float(max(samp["Actual"].max(), samp["Predicted"].max()))
    fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(color="#dc2626", dash="dash"))
    fig.update_xaxes(title="Actual traffic volume"); fig.update_yaxes(title="Predicted traffic volume")
    show(fig)

    imp = tm.feature_importance(out["models"]["Random Forest"])
    if imp is not None:
        fig = px.bar(imp.iloc[::-1], x="importance", y="feature", orientation="h", color_discrete_sequence=["#2563eb"],
                     title="Random Forest: top 15 feature importances (feature_value = one-hot category)",
                     labels={"importance": "Importance", "feature": "Feature"})
        show(fig)

    st.subheader("Try a prediction")
    st.caption(f"Uses the best model ({best}). Weather values are inputs you supply - they are not forecasts.")
    c1, c2, c3, c4 = st.columns(4)
    hour = c1.slider("Hour", 0, 23, 8)
    dname = c2.selectbox("Day of week", DAY_NAMES)
    mname = c3.selectbox("Month", MONTH_NAMES, index=5)
    wmain = c4.selectbox("Weather", sorted(data["weather_main"].unique()))
    c1, c2, c3, c4 = st.columns(4)
    temp = c1.slider("Temperature (C)", -30, 40, 15)
    rain = c2.number_input("Rain last hour (mm)", 0.0, 100.0, 0.0)
    clouds = c3.slider("Cloud cover (%)", 0, 100, 40)
    holiday = c4.checkbox("Public holiday")
    pred = tm.predict_traffic(out["models"][best], hour, DAY_NAMES.index(dname), MONTH_NAMES.index(mname) + 1,
                              wmain, temp, rain, 0.0, clouds, int(holiday))
    st.metric("Predicted traffic volume", f"{pred:,.0f} vehicles/hour")

elif page == "Unusual Traffic Detection":
    st.header("Unusual traffic pattern detection (IQR method)")
    note("<b>Method:</b> an hour is flagged when its traffic is below Q1 - k x IQR or above Q3 + k x IQR, compared with "
         "other observations <i>in the same hour of day and day type</i> (weekday/weekend). IQR is simple, explainable "
         "and does not assume a normal distribution.")
    c1, c2 = st.columns(2)
    k = c1.slider("Sensitivity multiplier k (smaller = more flags)", 1.0, 3.0, 1.5, 0.1)
    mode = c2.radio("Compare against", ["Same hour & day type (recommended)", "All hours together"], horizontal=True)
    groups = ad.DEFAULT_GROUP_COLUMNS if mode.startswith("Same") else []
    result = ad.detect_iqr_anomalies(df, k, groups)
    counts = ad.anomaly_counts(result)
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Flagged observations", f"{counts['flagged']:,}", f"{counts['percent']}% of {counts['total']:,}")
    kpi(c2, "Unusually high", f"{counts['high']:,}")
    kpi(c3, "Unusually low", f"{counts['low']:,}")
    kpi(c4, "Multiplier k", f"{k}")
    st.write("")
    show(ad.fig_anomalies(result))
    show(ad.fig_anomalies_by_hour(result))
    table = ad.anomaly_table(result)
    st.subheader("Flagged observations (most extreme first)")
    if table.empty:
        st.info("No observations were flagged with these settings.")
    else:
        st.dataframe(table.head(500), hide_index=True)
        st.caption(f"Showing up to 500 of {len(table):,} flagged rows.")
        st.download_button("Download all flagged rows (CSV)", table.to_csv(index=False).encode("utf-8"),
                           "flagged_traffic_observations.csv", "text/csv")
    note("A statistical anomaly is <b>not</b> proof of an accident, road closure or sensor fault. It only means the value "
         "is unusual compared with similar hours in this dataset. Extra information would be needed to explain it.", warn=True)

elif page == "Traffic Pattern Clustering":
    st.header("Traffic pattern clustering (K-Means, optional)")
    note("Observations are grouped by traffic volume, hour of day (as sin/cos), weekend flag, temperature and cloud cover. "
         "Features are standardised first. Clusters are statistical patterns, not guaranteed real-world traffic categories.")
    k = st.slider("Number of clusters (k)", 2, 6, 4)
    try:
        clustered, summary, score = cl.run_kmeans(df, k)
    except ValueError as err:
        st.warning(str(err)); st.stop()
    st.metric("Silhouette score (sample)", f"{score:.3f}",
              help="Between -1 and 1. Higher means better separated clusters. Values below about 0.25 mean weak structure.")
    show(cl.fig_clusters(clustered))
    st.subheader("Cluster summary")
    st.dataframe(summary, hide_index=True)
    note("The 'description' column is generated automatically from each cluster's averages. Check it against the table "
         "before quoting it in your report.")

else:
    st.header("About & guide")
    st.markdown(f"""
**Dataset:** [UCI Metro Interstate Traffic Volume]({DATASET_URL}) - hourly westbound traffic on I-94 near
Minneapolis-St. Paul, Minnesota (USA), with weather and holidays, from {data['date_time'].min():%b %Y} to {data['date_time'].max():%b %Y}.

**Pipeline:** CSV -> `data_loader.py` -> `preprocessing.py` -> analysis modules (`eda.py`, `peak_analysis.py`,
`anomaly_detection.py`, `clustering.py`, `train_model.py`) -> this Streamlit dashboard.

**Limitations**
- Data is from the USA; patterns will not transfer directly to Indian cities.
- The data has time gaps (see Data Quality), so lag-based forecasting was not attempted.
- Models use observed weather; this is prediction given conditions, not true future forecasting.
- Correlation and anomaly flags do not prove causes.
""")
