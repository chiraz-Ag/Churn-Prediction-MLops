"""
monitoring/populate_test_logs.py

Sends many randomized, VALID /predict requests to your running API, so that
logs/predictions.csv has enough rows (default: 50) for drift_report.py to
produce a statistically meaningful report, without clicking through Swagger
by hand 30+ times.

Prerequisite: the API must already be running in another terminal:
    uvicorn src.api.main:app --reload

Run from the repo root (in a SEPARATE terminal from the one running uvicorn):
    python monitoring/populate_test_logs.py
"""

import random

import requests

API_URL = "http://127.0.0.1:8000/predict"
NUM_REQUESTS = 50

# Valid values per field, matching the Literal constraints in CustomerInput
# (src/api/main.py). Internet-dependent fields are generated together below
# to respect the "No internet service" <-> InternetService="No" rule.
FIELD_OPTIONS = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

INTERNET_SERVICE_OPTIONS = ["DSL", "Fiber optic", "No"]
INTERNET_DEPENDENT_FIELDS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def random_customer() -> dict:
    customer = {field: random.choice(options) for field, options in FIELD_OPTIONS.items()}

    internet_service = random.choice(INTERNET_SERVICE_OPTIONS)
    customer["InternetService"] = internet_service
    if internet_service == "No":
        for field in INTERNET_DEPENDENT_FIELDS:
            customer[field] = "No internet service"
    else:
        for field in INTERNET_DEPENDENT_FIELDS:
            customer[field] = random.choice(["Yes", "No"])

    customer["tenure"] = random.randint(0, 72)
    customer["MonthlyCharges"] = round(random.uniform(18.0, 120.0), 2)
    customer["TotalCharges"] = round(customer["MonthlyCharges"] * max(customer["tenure"], 1), 2)

    return customer


def main():
    successes = 0
    failures = 0

    for i in range(NUM_REQUESTS):
        payload = random_customer()
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            if response.status_code == 200:
                successes += 1
            else:
                failures += 1
                print(f"Request {i + 1}: HTTP {response.status_code} — {response.text}")
        except requests.exceptions.ConnectionError:
            print(
                f"\nCould not connect to {API_URL}. "
                "Is the API running? Start it with:\n"
                "    uvicorn src.api.main:app --reload"
            )
            return

    print(f"\nDone: {successes} successful, {failures} failed, out of {NUM_REQUESTS} requests.")
    print("Check logs/predictions.csv — it should now have many more rows.")


if __name__ == "__main__":
    main()