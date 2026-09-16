"""
ChurnGuard AI — Prediction API

Loads the trained model + the fitted encoder/feature metadata produced by
data_prep.ipynb, and exposes:
  - GET  /health   basic liveness check
  - POST /predict  churn probability + prediction for one customer

Run locally with:
    uvicorn src.api.main:app --reload
(run from the repo root so the relative MODEL_DIR path resolves correctly)
"""

import os
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file, not the current working directory,
# so the API works the same whether launched from repo root or elsewhere.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.joblib")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.joblib")
CATEGORICAL_COLS_PATH = os.path.join(MODELS_DIR, "categorical_cols.joblib")
NUMERIC_COLS_PATH = os.path.join(MODELS_DIR, "numeric_cols.joblib")

# ---------------------------------------------------------------------------
# Load model + preprocessing artifacts ONCE at startup, not per-request —
# loading from disk is slow, and the artifacts never change while the
# server is running.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ChurnGuard AI API",
    description="Predicts customer churn probability for telecom customers.",
    version="1.0.0",
)

model = None
encoder = None
feature_columns = None
categorical_cols = None
numeric_cols = None


@app.on_event("startup")
def load_artifacts():
    global model, encoder, feature_columns, categorical_cols, numeric_cols
    missing = [p for p in [MODEL_PATH, ENCODER_PATH, FEATURE_COLUMNS_PATH,
                            CATEGORICAL_COLS_PATH, NUMERIC_COLS_PATH] if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            "Missing required artifact file(s): " + ", ".join(missing) +
            "\nRun notebooks/data_prep.ipynb and notebooks/train_churnguard.ipynb "
            "(or your teammate's training notebook) first."
        )

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    categorical_cols = joblib.load(CATEGORICAL_COLS_PATH)
    numeric_cols = joblib.load(NUMERIC_COLS_PATH)


# ---------------------------------------------------------------------------
# Request schema — one customer's raw attributes, validated by Pydantic.
# Field names/types mirror the original Telco dataset columns (pre-encoding).
# ---------------------------------------------------------------------------
# Kept as a module-level constant (not a class attribute) so it's not a
# mutable default value living directly in the class body.
EXAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100, description="Months as a customer")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_CUSTOMER})

class PredictionResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float
    risk_level: Literal["Low", "Medium", "High"]


# ---------------------------------------------------------------------------
# Feature engineering — MUST mirror data_prep.ipynb exactly, or the model
# receives data shaped differently than what it was trained on.
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12m", "12-24m", "24-48m", "48m+"],
    )

    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)
    df["avg_charge_per_service"] = df["TotalCharges"] / df["num_services"].replace(0, 1)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    return df


def encode_and_align(df: pd.DataFrame) -> pd.DataFrame:
    binary_cols = [c for c in df.columns if set(df[c].dropna().unique()) <= {"Yes", "No"}]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # Cast to plain string: sklearn's OneHotEncoder chokes on pandas
    # "category" dtype (e.g. tenure_bucket from pd.cut) when checking for
    # unknown categories internally.
    cat_input = df[categorical_cols].astype(str)

    encoded = pd.DataFrame(
        encoder.transform(cat_input),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )
    numeric_part = df[numeric_cols].reset_index(drop=True)
    encoded = encoded.reset_index(drop=True)
    full = pd.concat([numeric_part, encoded], axis=1)

    # Reindex to the exact training-time column order — fills any column
    # the model expects but this request didn't produce (e.g. a category
    # not present in this single row) with 0.
    full = full.reindex(columns=feature_columns, fill_value=0)
    return full


def risk_level_from_probability(p: float) -> str:
    if p < 0.33:
        return "Low"
    if p < 0.66:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model not loaded",
        "model_type": type(model).__name__ if model is not None else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        raw_df = pd.DataFrame([customer.model_dump()])
        engineered = engineer_features(raw_df)
        X = encode_and_align(engineered)

        probability = float(model.predict_proba(X)[0, 1])
        prediction = "Yes" if probability >= 0.5 else "No"

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=risk_level_from_probability(probability),
        )
    except Exception as e:
        # Intentional catch-all at the API boundary: any failure in feature
        # engineering, encoding, or the model itself becomes a clean 400
        # for the client, not a raw 500 with an internal stack trace.
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}") from e