"""
Tests for the FastAPI endpoints defined in src/api/main.py.

Uses FastAPI's TestClient, which calls the app in-process (no real network
request, no need to have `uvicorn` running separately). Using it as a
context manager (`with TestClient(app) as client`) triggers the app's
startup event, so the model/encoder/artifacts are actually loaded before
each test runs, exactly like they would be in a real deployment.

Run with:
    pytest tests/test_api.py -v
(run from the repo root so the src.api.main import resolves correctly)
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


VALID_CUSTOMER = {
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

# A long-tenure, two-year-contract customer with many add-on services —
# the profile the EDA identified as low churn risk, used to sanity-check
# that the API's output direction makes sense, not just that it responds.
LOW_RISK_CUSTOMER = {
    **VALID_CUSTOMER,
    "tenure": 60,
    "Contract": "Two year",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
    "MonthlyCharges": 90.0,
    "TotalCharges": 5400.0,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_ok_when_model_loaded(client):
    """After startup, /health must report status ok and the loaded model's
    class name, confirming the artifacts were found and loaded correctly."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_type"] is not None


# ---------------------------------------------------------------------------
# /predict — happy path
# ---------------------------------------------------------------------------

def test_predict_returns_valid_response_shape(client):
    """A well-formed customer must produce a 200 response with all three
    expected fields, correctly typed."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200

    body = response.json()
    assert body["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in ("Low", "Medium", "High")


def test_predict_probability_matches_prediction_label(client):
    """churn_prediction and churn_probability must be internally consistent:
    'Yes' only when probability >= 0.5, 'No' otherwise. Guards against the
    two fields silently disagreeing after a future code change."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    body = response.json()

    if body["churn_prediction"] == "Yes":
        assert body["churn_probability"] >= 0.5
    else:
        assert body["churn_probability"] < 0.5


def test_predict_low_risk_profile_scores_lower_than_high_risk(client):
    """Sanity check on model direction, not just API plumbing: a customer
    matching the EDA's low-churn profile (long tenure, two-year contract,
    security/support add-ons) should score a lower churn probability than
    the short-tenure, month-to-month VALID_CUSTOMER profile."""
    low_risk_response = client.post("/predict", json=LOW_RISK_CUSTOMER)
    high_risk_response = client.post("/predict", json=VALID_CUSTOMER)

    low_risk_proba = low_risk_response.json()["churn_probability"]
    high_risk_proba = high_risk_response.json()["churn_probability"]

    assert low_risk_proba < high_risk_proba


# ---------------------------------------------------------------------------
# /predict — validation errors
# ---------------------------------------------------------------------------

def test_predict_rejects_invalid_categorical_value(client):
    """Pydantic's Literal validation must reject a value outside the
    allowed set (e.g. a typo or wrong case) with 422, before the request
    ever reaches the model."""
    bad_customer = {**VALID_CUSTOMER, "Contract": "month-to-month"}  # wrong case
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_rejects_missing_required_field(client):
    """A request missing a required field must be rejected with 422,
    not silently processed with a default or null value."""
    incomplete_customer = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    response = client.post("/predict", json=incomplete_customer)
    assert response.status_code == 422


def test_predict_rejects_negative_monthly_charges(client):
    """MonthlyCharges has a ge=0 constraint; a negative value must be
    rejected rather than passed through to the model."""
    bad_customer = {**VALID_CUSTOMER, "MonthlyCharges": -10.0}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422
