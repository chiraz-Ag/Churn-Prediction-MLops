"""
monitoring/drift_report.py

Compares the live traffic logged by the API (logs/predictions.csv) against
the reference dataset (data/processed/X_train_raw.csv, the historical
customer population) using Evidently's DataDriftPreset, and writes an HTML
report you can open in a browser.

Requires:
    - data/processed/X_train_raw.csv  (run monitoring/generate_reference.py first)
    - logs/predictions.csv            (created automatically by the API;
                                        call /predict a few times first)

Run from the repo root with:
    python monitoring/drift_report.py
"""

import os
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

REFERENCE_PATH = os.path.join(REPO_ROOT, "data", "processed", "X_train_raw.csv")
CURRENT_PATH = os.path.join(REPO_ROOT, "logs", "predictions.csv")
REPORT_DIR = os.path.join(REPO_ROOT, "monitoring", "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "drift_report.html")

# Minimum number of logged predictions before a drift report is statistically
# meaningful. Below this, Evidently will still run, but the result is noisy
# since drift tests (K-S, Z-test, ...) need enough samples to be reliable.
MIN_CURRENT_ROWS = 30


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(
            f"Reference file not found: {REFERENCE_PATH}\n"
            "Run: python monitoring/generate_reference.py"
        )
    if not os.path.exists(CURRENT_PATH):
        raise FileNotFoundError(
            f"Prediction log not found: {CURRENT_PATH}\n"
            "Call the API's /predict endpoint at least once so it gets created."
        )

    reference_df = pd.read_csv(REFERENCE_PATH)
    current_df = pd.read_csv(CURRENT_PATH)

    if current_df.empty:
        raise ValueError(
            f"{CURRENT_PATH} exists but has no rows yet. "
            "Call /predict a few times first, then re-run this script."
        )

    # Keep only the raw customer feature columns both files have in common —
    # this drops logs/predictions.csv's extra timestamp/churn_prediction/
    # churn_probability columns, which aren't part of the reference and
    # aren't meaningful inputs to compare for drift.
    common_cols = [c for c in reference_df.columns if c in current_df.columns]
    missing_from_current = [c for c in reference_df.columns if c not in current_df.columns]
    if missing_from_current:
        print(
            f"Warning: columns present in the reference but missing from "
            f"the prediction log were skipped: {missing_from_current}"
        )

    return reference_df[common_cols], current_df[common_cols]


def summarize_drift(result) -> None:
    """Print a short, readable summary to the console from the Evidently result."""
    result_dict = result.dict()
    metrics = result_dict.get("metrics", [])

    drifted_count_metric = next(
        (m for m in metrics if m.get("metric_name", "").startswith("DriftedColumnsCount")),
        None,
    )

    print("\n--- Drift summary ---")
    if drifted_count_metric:
        count = drifted_count_metric["value"]["count"]
        share = drifted_count_metric["value"]["share"]
        print(f"Drifted columns (per Evidently's own count): {int(count)} ({share:.1%} of all columns)")
    else:
        print("Could not find the DriftedColumnsCount summary metric.")

    # Per-column detail. Evidently picks the test method per column
    # automatically (it depends on the column type AND the reference
    # dataset size — p-value tests like K-S/Z-test/chi-square for smaller
    # references, distance-based tests like Jensen-Shannon/Wasserstein for
    # larger ones), and the two families disagree on drift direction:
    #   - "... p_value" methods:  drifted when value < threshold
    #   - "... distance" methods: drifted when value > threshold
    # Getting this backwards would silently mismatch Evidently's own count
    # above, so the direction is picked explicitly per method name below.
    rows = []
    for m in metrics:
        name = m.get("metric_name", "")
        if name.startswith("ValueDrift") and "column=" in name:
            column = name.split("column=")[1].split(",")[0]
            config = m.get("config", {})
            method = config.get("method", "?")
            threshold = config.get("threshold")
            value = m.get("value")

            drifted = None
            if isinstance(value, (int, float)) and isinstance(threshold, (int, float)):
                if "p_value" in method:
                    drifted = value < threshold
                elif "distance" in method:
                    drifted = value > threshold

            rows.append((column, method, value, threshold, drifted))

    if rows:
        rows.sort(key=lambda r: (r[4] is not True, r[0]))
        print("\nPer-column test results:")
        for column, method, value, threshold, drifted in rows:
            value_str = f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
            flag = "DRIFTED" if drifted else ("ok" if drifted is False else "?")
            print(f"  [{flag:<7}] {column:<20} method={method:<28} value={value_str} (threshold={threshold})")

    print(f"\nFull interactive HTML report: {REPORT_PATH}")
    print("---------------------\n")


def main():
    reference_df, current_df = load_data()

    if len(current_df) < MIN_CURRENT_ROWS:
        print(
            f"Warning: only {len(current_df)} logged prediction(s) found "
            f"(recommended minimum: {MIN_CURRENT_ROWS}). The report below "
            "will still be generated, but treat its conclusions with caution "
            "until more traffic has been logged.\n"
        )

    report = Report(metrics=[DataDriftPreset()])
    try:
        result = report.run(reference_data=reference_df, current_data=current_df)
    except Exception as e:
        if len(current_df) < MIN_CURRENT_ROWS:
            raise RuntimeError(
                f"Evidently failed to compute drift, most likely because there "
                f"are only {len(current_df)} logged prediction(s) — too few for "
                f"its statistical tests to run. Call /predict at least "
                f"{MIN_CURRENT_ROWS} times (varying the input values) to build "
                f"up logs/predictions.csv, then try again.\n"
                f"(Original error: {e})"
            ) from e
        raise RuntimeError(f"Evidently failed to compute drift: {e}") from e

    os.makedirs(REPORT_DIR, exist_ok=True)
    result.save_html(REPORT_PATH)
    print(f"HTML report written to: {REPORT_PATH}")

    summarize_drift(result)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)