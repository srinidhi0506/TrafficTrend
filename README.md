# 🚦 TrafficTrend — Traffic Congestion Pattern Analysis and Traffic Volume Forecasting

A beginner-friendly Data Analytics project (B.Tech CSE). It analyses historical hourly traffic data,
finds peak hours, studies the effect of weather, detects unusual observations, groups traffic patterns
and predicts traffic volume with machine learning — all in an interactive Streamlit dashboard.

> **Dataset origin (please state this in your report/viva):** UCI *Metro Interstate Traffic Volume* —
> hourly westbound traffic on **Interstate 94 between Minneapolis and St. Paul, Minnesota, USA**
> (Oct 2012 – Sep 2018). It is **not** Hyderabad data and conclusions must not be presented as such.

---

## 1. Project structure

```
TrafficTrend/
├── app.py                  # Streamlit dashboard (run this)
├── requirements.txt        # Python packages
├── pytest.ini              # makes `python -m pytest` find the src/ package
├── README.md
├── data/
│   └── .gitkeep            # PUT THE DOWNLOADED DATASET HERE
├── models/
│   └── .gitkeep            # best trained model is saved here (best_traffic_model.joblib)
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # finds + loads the CSV, validates columns, friendly errors
│   ├── preprocessing.py    # cleaning, outlier report, time features (every step explained)
│   ├── eda.py              # 8 EDA charts with data-driven interpretations
│   ├── train_model.py      # Linear Regression + Random Forest, chronological split, Joblib
│   ├── peak_analysis.py    # busiest hours, weekday vs weekend, observations
│   ├── anomaly_detection.py# IQR anomaly detection (explainable)
│   └── clustering.py       # OPTIONAL K-Means clustering
└── tests/
    ├── test_preprocessing.py
    └── test_analysis.py
```

**Data flow:** `data/*.csv(.gz)` → `data_loader` → `preprocessing` → (`eda`, `peak_analysis`,
`anomaly_detection`, `clustering`, `train_model`) → `app.py` (dashboard).

**Where does the app look for the dataset?** `src/data_loader.py` computes the project root from its own location,
then looks for `data/Metro_Interstate_Traffic_Volume.csv` or `data/Metro_Interstate_Traffic_Volume.csv.gz`.
If `data/` contains exactly one `.csv`/`.csv.gz` file, that file is used. The dataset is stored in one place only.

---

## 2. Windows 10/11 setup — step by step

### 2.1 Install Python and check it
1. Go to <https://www.python.org/downloads/windows/> and download a **64-bit Python 3.11 or 3.12** installer.
2. Run it and **tick "Add python.exe to PATH"** on the first screen, then *Install Now*.
3. Open **Command Prompt** or PowerShell and check:
   ```
   py --version
   ```
   You should see something like `Python 3.12.x`.

### 2.2 Install VS Code
Download from <https://code.visualstudio.com/>, install, then open VS Code → Extensions (Ctrl+Shift+X) →
install **Python** (by Microsoft).

### 2.3 Create the project folder and open it
1. In File Explorer create `C:\Users\<YourName>\Documents\TrafficTrend` (or any folder you prefer).
2. VS Code → **File → Open Folder…** → select `TrafficTrend`.
3. Create the files and folders exactly as in the structure above (right-click in the Explorer panel →
   *New Folder* / *New File*). Paste the code of each file. Filenames and capital letters must match exactly.
   `src/__init__.py` can be an empty file.

### 2.4 Open the terminal inside VS Code
**Terminal → New Terminal** (shortcut Ctrl+`). The path shown at the prompt must end with `TrafficTrend`.
Every command below is run **from the `TrafficTrend` folder (the project root)** unless stated otherwise.

### 2.5 Create and activate a virtual environment
```
py -m venv .venv
```
Activate it:

| Terminal type | Command |
|---|---|
| PowerShell (default in VS Code) | `.venv\Scripts\Activate.ps1` |
| Command Prompt | `.venv\Scripts\activate.bat` |

When active, the prompt starts with `(.venv)`.

**If PowerShell says "running scripts is disabled on this system":** do *not* change system security settings.
Simply use Command Prompt instead: in the VS Code terminal panel click the **▾ arrow next to the + button →
Command Prompt**, make sure you are in the project folder (`cd` to it if needed), and run
`.venv\Scripts\activate.bat`.

### 2.6 Install packages (with the environment active, in the project root)
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2.7 Download the dataset and place it
1. Open <https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume> and click **Download**.
2. Extract the zip. You get `Metro_Interstate_Traffic_Volume.csv.gz`.
3. **You do not need to unzip the `.gz`** — pandas reads it directly. Copy the file into `TrafficTrend\data\`.
   (If you do have a plain `.csv`, that works too.)

Quick check (optional, project root):
```
python -m src.data_loader
```
It should print the shape `(48204, 9)` and the column names.

### 2.8 Run the dashboard
```
python -m streamlit run app.py
```
Streamlit normally opens your browser automatically. If not, open **http://localhost:8501** manually
(the exact address is printed in the terminal). If Streamlit asks for an e-mail on first run, just press Enter.

### 2.9 Run the tests (project root)
```
python -m pytest -v
```

### 2.10 Stop, deactivate, restart later
* **Stop the server:** click in the terminal and press **Ctrl+C**.
* **Deactivate the virtual environment:** type `deactivate`.
* **Restart on another day:** open VS Code → *File → Open Recent* → `TrafficTrend` → open a terminal →
  `.venv\Scripts\Activate.ps1` (or `activate.bat` in Command Prompt) → `python -m streamlit run app.py`.
  You do **not** need to reinstall packages.

---

## 3. How to verify each feature

| Feature | Where to look | What a correct result looks like |
|---|---|---|
| Data loading | *Data Quality* page | Raw rows 48,204 and 9 columns; list of preprocessing steps |
| Preprocessing | *Data Quality* | Duplicates removed; invalid values counted; outlier table |
| EDA (8 charts) | *Exploratory Analysis* tabs | Each tab shows a chart and a blue interpretation box |
| Peak analysis | *Peak Traffic Analysis* | Top-5 tables, weekday vs weekend line chart, observations |
| Prediction | *Traffic Prediction* | Table with MAE/RMSE/R² for 3 rows (baseline, LR, RF); `models/best_traffic_model.joblib` appears |
| Anomalies | *Unusual Traffic Detection* | Count cards, red/orange points, table, CSV download |
| Clustering | *Traffic Pattern Clustering* | Scatter plot, silhouette score, summary table |
| Date filter | Sidebar | KPI values and charts change when the period changes |

The first visit to *Traffic Prediction* trains the models (about a minute on a typical laptop); later visits are instant
because the result is cached.

### Troubleshooting
| Problem | Fix |
|---|---|
| `'py' is not recognized` | Reinstall Python and tick *Add python.exe to PATH*; reopen VS Code. |
| `Activate.ps1 cannot be loaded… disabled` | Use Command Prompt + `activate.bat` (see 2.5). |
| `No module named streamlit` / `src` | Environment not active (no `(.venv)` in prompt) or terminal is not in the project root. |
| `streamlit` is not recognized | Always use `python -m streamlit run app.py`. |
| Red box "Dataset file not found" | Copy the file into `data\` (see 2.7) and refresh the page. |
| Port already in use | Close the old terminal, or run `python -m streamlit run app.py --server.port 8502`. |
| Browser shows an old page | Press **R** in the page or click *Rerun*; hard-refresh with Ctrl+F5. |
| `pip install` fails with build errors | Use Python 3.11 or 3.12 (very new versions may lack pre-built packages). |

---

## 4. Syllabus mapping

| Unit | Topic | Where in this project | Status |
|---|---|---|---|
| I — Data Management | Ingestion, quality checks, missing values, duplicates, preprocessing | `data_loader.py`, `preprocessing.py`, *Data Quality* page | Implemented |
| II — Data Visualization | Univariate (histogram), bivariate (weather vs traffic), multivariate (heatmaps) | `eda.py`, `peak_analysis.py` | Implemented |
| III — Data Analysis | EDA, interpretation, analytics workflow | All dashboard pages, data-driven insight text | Implemented |
| IV — Data Modelling | Correlation, Linear Regression, Random Forest, MAE/RMSE/R² | `eda.fig_correlation`, `train_model.py` | Implemented |
| V — Objective Segmentation | K-Means (optional), anomaly analysis, time-dependent analysis | `clustering.py`, `anomaly_detection.py`, chronological split | K-Means is optional-but-included; IQR anomaly detection implemented. **ARIMA is NOT implemented.** |

---

## 5. Academic documentation

### 5.1 Abstract
TrafficTrend is an interactive Data Analytics system that studies hourly traffic volume on a highway in the
Minneapolis–Saint Paul area using the UCI Metro Interstate Traffic Volume dataset. The system cleans the data,
explores daily, weekly and seasonal patterns, examines weather relationships, identifies peak periods, flags
statistically unusual observations, groups hours into traffic patterns with K-Means and compares Linear Regression
with Random Forest for predicting traffic volume using a time-ordered train/test split. Results are presented in a
Streamlit dashboard. *(Add your own measured results in Section 5.9.)*

### 5.2 Problem statement
Traffic congestion planning needs an understanding of when traffic peaks, how it varies by day and season, and how
weather and holidays relate to it. Raw hourly records are hard to interpret manually. There is a need for a simple
tool that converts historical data into clear patterns, anomaly flags and a predictive model.

### 5.3 Objectives
1. Clean and validate the historical traffic dataset and document each preprocessing decision.
2. Identify peak hours and days and compare weekday and weekend behaviour.
3. Study relationships between weather variables and traffic volume.
4. Build and compare regression models (Linear Regression, Random Forest) with a chronological evaluation.
5. Detect unusually high or low observations with an explainable method.
6. Present all findings in an interactive dashboard.

### 5.4 Scope
Included: one highway-segment dataset, hourly granularity, offline CSV analysis, local dashboard.
Not included: live traffic feeds, route-level analysis, Indian road data, deep-learning or ARIMA forecasting,
real-time deployment.

### 5.5 Existing system vs proposed system
| Existing (manual/basic) | Proposed (TrafficTrend) |
|---|---|
| Spreadsheet inspection of raw hourly rows | Automated cleaning with documented steps |
| Static charts, no filtering | Interactive Plotly charts and date filter |
| Patterns noticed informally | Calculated peak hours, weekday/weekend comparison |
| No quantified weather effect | Correlation and category comparisons |
| No predictive model | Linear Regression and Random Forest with MAE, RMSE, R² |
| Unusual values noticed by chance | IQR anomaly detection per hour-of-day and day type |

### 5.6 System architecture
```
 UCI CSV (data/) ──► data_loader ──► preprocessing ──► cleaned DataFrame (cached)
                                                         │
        ┌──────────────┬──────────────┬─────────────────┼───────────────┬───────────────┐
        ▼              ▼              ▼                 ▼               ▼               ▼
      eda.py     peak_analysis   anomaly_detection  clustering     train_model.py   (tests)
        │              │              │                 │               │
        └──────────────┴──────────────┴────── app.py (Streamlit + Plotly) ◄── models/*.joblib
```

### 5.7 Dataset description
Source: UCI Machine Learning Repository, *Metro Interstate Traffic Volume*. Hourly westbound I-94 traffic volume
measured by an automatic traffic recorder between Minneapolis and St. Paul, with weather from
a nearby station.

| Column | Meaning |
|---|---|
| `holiday` | US national / regional holiday name, or `None` |
| `temp` | Temperature in Kelvin |
| `rain_1h` | Rain in the last hour (mm) |
| `snow_1h` | Snow in the last hour (mm) |
| `clouds_all` | Cloud cover (%) |
| `weather_main` / `weather_description` | Short / detailed weather text |
| `date_time` | Local CST hour of the record |
| `traffic_volume` | Hourly traffic volume (target) |

Properties you will see when you run the project (the dashboard computes the exact numbers): the file has 48,204 rows
but many repeated timestamps (the same hour listed with several weather descriptions), a few exact duplicates,
impossible values (e.g. 0 K temperature, a rain value of several thousand mm in one hour), and long time gaps.
The holiday name appears only on the 00:00 row of a holiday, which `preprocessing.py` extends to the whole day.

### 5.8 Methodology
1. **Ingestion** – load CSV/CSV.GZ, validate required columns.
2. **Cleaning** – parse dates; drop rows without valid time/target; remove exact duplicates; keep one row per timestamp;
   mark impossible weather values as missing (rows kept); keep valid outliers; never fill time gaps.
3. **Feature engineering** – hour, day, month, day of week, weekend flag, holiday flag, temperature in °C.
4. **EDA** – distribution, hourly/daily/monthly profiles, heatmap, time series, weather relationships, correlation.
5. **Peak analysis** – averages by hour/day type, top-5 lists, "consistently high" hours (≥50% of records above the
   75th-percentile threshold).
6. **Anomaly detection** – IQR rule within each (hour, weekday/weekend) group.
7. **Clustering (optional)** – standardised features → K-Means.
8. **Modelling** – chronological 80/20 split; preprocessing inside a scikit-learn `Pipeline` fitted on training rows
   only; evaluate with MAE, RMSE, R²; compare with a "predict the training mean" baseline; save the best real model.

### 5.9 Algorithms
* **Linear Regression** – fits a weighted sum of the inputs (hour, day, month and weather are one-hot encoded so a
  rush-hour *shape* can be learned). Fast and interpretable, but cannot capture complex interactions
  (e.g. "weekday AND 5 pm").
* **Random Forest Regressor** – averages many decision trees trained on random subsets; captures non-linear patterns and
  interactions without feature scaling. Less interpretable; can overfit if not tuned.
* **IQR anomaly detection** – flag values below Q1 − k·IQR or above Q3 + k·IQR (k = 1.5 by default).
* **K-Means** – assigns each point to the nearest of *k* centroids; needs standardised features; the silhouette score
  measures how well separated clusters are.
* **Metrics** – MAE (average absolute error, same unit as traffic), RMSE (penalises large errors),
  R² (share of variance explained; 0 = no better than predicting the mean).

### 5.10 Experimental results template (fill in after YOUR run)
| Item | Your value |
|---|---|
| Rows after cleaning | `[fill]` |
| Train period / Test period | `[fill]` / `[fill]` |
| Baseline (training mean): MAE / RMSE / R² | `[fill]` |
| Linear Regression: MAE / RMSE / R² | `[fill]` |
| Random Forest: MAE / RMSE / R² | `[fill]` |
| Best model (lowest RMSE) | `[fill]` |
| Busiest hour (average) / quietest hour | `[fill]` |
| Weekday vs weekend difference (%) | `[fill]` |
| Anomalies flagged (k = 1.5): total / high / low | `[fill]` |
| K-Means (k = 4) silhouette score | `[fill]` |

Only state a model is "accurate" if these measured numbers support it (compare against the baseline row).

### 5.11 Limitations
* Single highway segment in the USA; not transferable to Hyderabad or other cities.
* Time gaps (the longest is many months) make lag-based or ARIMA forecasting unreliable without special handling.
* The models use *observed* weather of the same hour; true forecasting would need weather forecasts.
* Only one temporal split is evaluated; results can differ for other periods (no time-series cross-validation).
* Repeated timestamps were reduced to the first record, which discards alternative weather descriptions.
* Anomalies are statistical; the data has no incident labels, so causes cannot be confirmed.
* Clusters are statistical groupings, and their silhouette score may be modest.

### 5.12 Future enhancements
Time-series cross-validation; hyper-parameter tuning; gradient boosting; ARIMA/SARIMA or Prophet on a gap-free
sub-period; weather forecast API for real forecasting; Isolation Forest comparison; multi-city data; deployment on
Streamlit Community Cloud.

### 5.13 Conclusion
TrafficTrend demonstrates a complete analytics workflow: data management, visualization, analysis, modelling and
segmentation. It shows how calendar and weather information relate to traffic volume and how model quality should be
measured honestly on a chronological hold-out period. Final conclusions should quote your measured results.

### 5.14 Viva questions and simple answers
1. **Why did you use a chronological split?** Traffic is time-ordered; a random split lets the model see the future,
   making the test unrealistically easy.
2. **What is target leakage?** Using information in the inputs that would not exist at prediction time or is derived from the
   target. We use only calendar and weather features and no traffic-based features.
3. **Prediction vs forecasting?** Prediction uses known inputs of the same hour (including observed weather). Forecasting
   needs inputs for a future time, e.g. a weather forecast.
4. **Why one-hot encode hour?** The hour–traffic relation is not a straight line; treating hours as categories lets Linear
   Regression learn a different level for each hour.
5. **Why did you not delete outliers?** Valid extremes are real behaviour; only impossible values (0 K, thousands of mm rain)
   are errors, and they were set to missing.
6. **Why fill missing values inside the pipeline?** So medians come only from training data (avoids leakage).
7. **Why does Random Forest usually beat Linear Regression here?** It captures interactions such as weekday × rush hour.
   (Check this claim against *your* results table.)
8. **What does R² = 0 mean?** The model is no better than always predicting the average.
9. **Why is the baseline model included?** To prove the real models add value.
10. **Why IQR for anomalies?** Simple, no normality assumption, explainable; computed per hour/day type so normal rush hours
    are not flagged.
11. **Does an anomaly mean an accident?** No. It only means unusual compared to similar hours.
12. **Why standardise before K-Means?** K-Means uses distances; large-scale features (traffic in thousands) would dominate.
13. **What is the silhouette score?** A −1 to 1 measure of how well points fit their own cluster compared to others.
14. **Why remove duplicate timestamps?** The data has one traffic count per hour; repeats would double-count hours.
15. **Is this Hyderabad traffic?** No — it is a Minneapolis–St. Paul highway; patterns are illustrative.
16. **Why is the model cached with Joblib/Streamlit?** Training is slow; caching avoids retraining on each rerun.
17. **Did you use ARIMA?** No. It is listed as future work.

---

## 6. Screenshots and evaluation presentation

**Screenshots (Windows):** press **Win + Shift + S**, drag over the dashboard area, then paste into Word/PowerPoint.
Take one per page: Overview (KPIs), Data Quality, each EDA tab (heatmap, hour chart, correlation), Peak Analysis,
Prediction (comparison table + actual-vs-predicted), Anomalies, Clustering. Plotly charts have a camera icon
(top-right on hover) to download a PNG.

**Suggested 8–10 minute demo flow**
1. State the dataset origin (Minneapolis–St. Paul, USA) and the objective.
2. *Data Quality*: explain duplicates, invalid values, "error vs valid outlier".
3. *EDA*: hour chart and heatmap — explain the rush-hour pattern shown by *your* data.
4. *Peak Analysis*: weekday vs weekend.
5. *Prediction*: chronological split, metrics, baseline, and the prediction-vs-forecasting caveat.
6. *Anomalies* and *Clustering*: what they do and do not prove.
7. Limitations and future work; run `python -m pytest -v` live if asked.

Tips: start the app and open every page once before the evaluation (so the model is already cached); keep the
terminal visible; be ready to open `preprocessing.py` and `train_model.py` and explain them.

---

## 7. Final submission checklist
- [ ] Project runs from a fresh terminal: activate `.venv` → `python -m streamlit run app.py`
- [ ] Dataset is in `data/` (do not submit it if your department forbids it; mention the download link instead)
- [ ] `python -m pytest -v` passes
- [ ] Results table in 5.10 filled with **your** measured numbers
- [ ] Screenshots of all pages collected
- [ ] Report contains abstract, problem, objectives, scope, architecture, methodology, results, limitations, conclusion
- [ ] You can explain: cleaning decisions, chronological split, leakage, metrics, IQR, K-Means
- [ ] Statement that data is from Minneapolis–St. Paul, not Hyderabad
- [ ] `.venv/` folder excluded from the zip you submit (it is large and machine-specific)
