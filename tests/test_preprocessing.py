"""
Tests for the preprocessing functions defined in src/api/main.py.

These functions (engineer_features, encode_and_align) MUST reproduce the
exact transformations applied in notebooks/data_prep.ipynb. If they drift
even slightly, the model receives data shaped differently than what it was
trained on, and predictions become unreliable without any visible error.
These tests exist specifically to catch that kind of silent drift.

Run with:
    pytest tests/test_preprocessing.py -v
(run from the repo root so the src.api.main import resolves correctly)
"""

import pandas as pd
import pytest

from src.api import main as api_main


@pytest.fixture(scope="module", autouse=True)
def _load_artifacts_once():
    """The model/encoder/feature_columns are normally loaded by FastAPI's
    startup event, which never fires here since we import functions
    directly instead of running the server. Calling load_artifacts()
    explicitly reproduces that startup step so encode_and_align has real
    artifacts to work with instead of None."""
    api_main.load_artifacts()


SAMPLE_CUSTOMER = {
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


def make_raw_df(overrides=None):
    """Builds a one-row raw dataframe, same shape the API receives before
    any transformation, with optional field overrides for edge cases."""
    data = {**SAMPLE_CUSTOMER, **(overrides or {})}
    return pd.DataFrame([data])


# ---------------------------------------------------------------------------
# engineer_features
# ---------------------------------------------------------------------------

def test_engineer_features_adds_expected_columns():
    """The engineered dataframe must contain every derived column the model
    was trained on, not just the raw input columns."""
    raw_df = make_raw_df()
    result = api_main.engineer_features(raw_df)

    expected_new_cols = {
        "tenure_bucket", "num_services", "avg_charge_per_service",
        "is_month_to_month", "has_internet",
    }
    assert expected_new_cols.issubset(set(result.columns))


def test_num_services_counts_yes_only():
    """num_services must count only the six add-on service columns that are
    'Yes', ignoring 'No' and 'No internet service' values."""
    raw_df = make_raw_df({
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No internet service",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
    })
    result = api_main.engineer_features(raw_df)
    assert result.loc[0, "num_services"] == 3


def test_is_month_to_month_flag():
    """is_month_to_month must be 1 for month-to-month contracts and 0 for
    one/two-year contracts, since it is expected to be the strongest churn
    signal per the EDA takeaways."""
    monthly = api_main.engineer_features(make_raw_df({"Contract": "Month-to-month"}))
    yearly = api_main.engineer_features(make_raw_df({"Contract": "Two year"}))

    assert monthly.loc[0, "is_month_to_month"] == 1
    assert yearly.loc[0, "is_month_to_month"] == 0


def test_total_charges_missing_value_is_handled():
    """New customers (tenure=0) sometimes have an empty/invalid TotalCharges
    in the raw Telco dataset. engineer_features must coerce this to 0
    instead of raising or propagating NaN into the model."""
    raw_df = make_raw_df({"tenure": 0, "TotalCharges": ""})
    result = api_main.engineer_features(raw_df)
    assert result.loc[0, "TotalCharges"] == 0


# ---------------------------------------------------------------------------
# encode_and_align
# ---------------------------------------------------------------------------

def test_encode_and_align_matches_training_columns():
    """The final encoded dataframe must have exactly the columns the model
    was trained on, in the same order — this is what makes model.predict
    safe to call on a single new customer."""
    raw_df = make_raw_df()
    engineered = api_main.engineer_features(raw_df)
    encoded = api_main.encode_and_align(engineered)

    assert list(encoded.columns) == list(api_main.feature_columns)


def test_encode_and_align_single_row_output_shape():
    """One input customer must produce exactly one output row, with no
    extra or missing rows introduced by the encoding step."""
    raw_df = make_raw_df()
    engineered = api_main.engineer_features(raw_df)
    encoded = api_main.encode_and_align(engineered)

    assert encoded.shape[0] == 1
    assert encoded.shape[1] == len(api_main.feature_columns)