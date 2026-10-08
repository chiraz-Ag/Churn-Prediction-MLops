"""
monitoring/dashboard.py

Streamlit dashboard for ChurnGuard AI monitoring. Reads logs/predictions.csv
(written by the API's log_prediction()) and shows:
  - Key numbers: total predictions, churn rate, average probability
  - Charts: probability distribution, predictions by contract type,
    predictions over time
  - A button to (re)generate the Evidently drift report, shown inline

Run from the repo root with:
    streamlit run monitoring/dashboard.py
"""

import os
import subprocess
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

PREDICTIONS_LOG_PATH = os.path.join(REPO_ROOT, "logs", "predictions.csv")
DRIFT_REPORT_PATH = os.path.join(REPO_ROOT, "monitoring", "reports", "drift_report.html")
DRIFT_SCRIPT_PATH = os.path.join(REPO_ROOT, "monitoring", "drift_report.py")

st.set_page_config(page_title="ChurnGuard AI — Monitoring", layout="wide")
st.title("ChurnGuard AI — Monitoring Dashboard")


@st.cache_data(ttl=10)
def load_predictions() -> pd.DataFrame | None:
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return None
    df = pd.read_csv(PREDICTIONS_LOG_PATH)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


df = load_predictions()

if df is None:
    st.warning(
        f"No prediction log found at `{PREDICTIONS_LOG_PATH}`.\n\n"
        "Call the API's `/predict` endpoint at least once (or run "
        "`python monitoring/populate_test_logs.py`), then refresh this page."
    )
    st.stop()

if df.empty:
    st.warning("The prediction log exists but is empty. Call `/predict` a few times, then refresh.")
    st.stop()

# ---------------------------------------------------------------------------
# Key numbers
# ---------------------------------------------------------------------------
total_predictions = len(df)
churn_rate = (df["churn_prediction"] == "Yes").mean()
avg_probability = df["churn_probability"].mean()
last_updated = df["timestamp"].max()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total predictions logged", f"{total_predictions:,}")
col2.metric("Predicted churn rate", f"{churn_rate:.1%}")
col3.metric("Average churn probability", f"{avg_probability:.2f}")
col4.metric("Last prediction", last_updated.strftime("%Y-%m-%d %H:%M") if pd.notna(last_updated) else "—")

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Churn probability distribution")
    hist_data = pd.cut(
        df["churn_probability"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
    ).value_counts().sort_index()
    hist_data.index = hist_data.index.astype(str)
    st.bar_chart(hist_data)

with right:
    st.subheader("Predictions by contract type")
    if "Contract" in df.columns:
        contract_counts = df["Contract"].value_counts()
        st.bar_chart(contract_counts)
    else:
        st.info("No `Contract` column in the log.")

st.subheader("Predictions over time")
if df["timestamp"].notna().any():
    per_day = df.set_index("timestamp").resample("D").size()
    per_day.name = "Predictions"
    st.line_chart(per_day)
else:
    st.info("No valid timestamps to plot.")

st.divider()

# ---------------------------------------------------------------------------
# Data drift
# ---------------------------------------------------------------------------
st.subheader("Data drift report")
st.caption(
    "Compares the logged predictions above against the reference training "
    "population (data/processed/X_train_raw.csv) using Evidently."
)

if st.button("Generate / refresh drift report", type="primary"):
    with st.spinner("Running monitoring/drift_report.py..."):
        result = subprocess.run(
            [sys.executable, DRIFT_SCRIPT_PATH],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    if result.returncode == 0:
        st.success("Drift report generated.")
        with st.expander("Console output"):
            st.code(result.stdout)
    else:
        st.error("Drift report generation failed. See details below.")
        st.code(result.stdout + "\n" + result.stderr)

if os.path.exists(DRIFT_REPORT_PATH):
    with open(DRIFT_REPORT_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=800, scrolling=True)
else:
    st.info(
        "No drift report found yet. Click the button above to generate "
        f"one (needs `{os.path.relpath(os.path.join(REPO_ROOT, 'data', 'processed', 'X_train_raw.csv'), REPO_ROOT)}` "
        "— run `python monitoring/generate_reference.py` first if missing)."
    )