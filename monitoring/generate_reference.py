"""
monitoring/generate_reference.py

Builds data/processed/X_train_raw.csv — the raw (non-encoded) reference
dataset used by monitoring/drift_report.py as the "known good" baseline
for Evidently's data drift detection.

Source: data/processed/telco_churn_cleaned.csv (the full cleaned dataset
from the EDA notebook, 7043 customers). We use the whole cleaned dataset
rather than only the original training split, since the exact train/test
split used in data_prep.ipynb isn't reproducible from this script alone —
and for a drift *reference* baseline (not model training), using the full
historical population is a standard and valid choice.

Only the columns that the API's CustomerInput schema actually receives are
kept, so this file has exactly the same raw columns as logs/predictions.csv
(minus the timestamp/prediction/probability columns that only exist in the
live log). That's what makes the two files comparable in drift_report.py.

Run from the repo root with:
    python monitoring/generate_reference.py
"""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

SOURCE_PATH = os.path.join(REPO_ROOT, "data", "processed", "telco_churn_cleaned.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "X_train_raw.csv")

# Exactly the fields the API's CustomerInput schema accepts, in the same
# order as logs/predictions.csv (minus timestamp/prediction/probability).
RAW_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def main():
    if not os.path.exists(SOURCE_PATH):
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_PATH}\n"
            "Make sure notebooks/eda_churnguard_v2.ipynb has been run and "
            "telco_churn_cleaned.csv exists."
        )

    df = pd.read_csv(SOURCE_PATH)

    missing_cols = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Expected column(s) missing from {SOURCE_PATH}: {missing_cols}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    reference_df = df[RAW_FEATURE_COLUMNS].copy()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    reference_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Reference file written to: {OUTPUT_PATH}")
    print(f"Shape: {reference_df.shape}")
    print(f"Columns: {reference_df.columns.tolist()}")


if __name__ == "__main__":
    main()