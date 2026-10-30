# ChurnGuard AI — Customer Churn Prediction with MLOps

A churn prediction system for telecom customers, built as an end-to-end MLOps portfolio project: from data exploration to a served, monitored API with automated CI/CD.

Dataset: [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

---

## Features

- **Exploratory Data Analysis** — data cleaning, duplicate removal, outlier handling, statistical tests (Chi², t-test)
- **Feature engineering** — tenure buckets, service counts, derived ratios, OneHotEncoder fitted on train data only
- **Model serving** — FastAPI REST API with Pydantic request/response validation
- **Automated testing** — pytest suite for the API and preprocessing logic
- **Containerization** — multi-stage, non-root Dockerfile with healthcheck
- **CI/CD** — GitHub Actions pipeline (lint → test → build)
- **Cloud deployment** — hosted on Render
- **Monitoring & drift detection** — prediction logging + automated data drift reports with Evidently AI
- **Interactive dashboard** — Streamlit dashboard for live monitoring

---

## Project structure

```
Churn-Prediction-MLops/
├── notebooks/
│   ├── eda_churnguard_v2.ipynb       # Data cleaning, EDA, statistical tests
│   └── data_prep.ipynb               # Feature engineering, encoding, train/test split
├── src/
│   └── api/
│       └── main.py                   # FastAPI app: /health, /predict, prediction logging
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
├── monitoring/
│   ├── generate_reference.py         # Builds the raw reference dataset for drift detection
│   ├── drift_report.py               # Compares live predictions vs. reference with Evidently
│   ├── populate_test_logs.py         # Sends randomized requests to populate logs for testing
│   └── dashboard.py                  # Streamlit monitoring dashboard
├── models/                           # Trained model + preprocessing artifacts (gitignored)
├── data/                             # Raw & processed datasets (gitignored)
├── logs/                             # Live prediction logs (gitignored, created at runtime)
├── Dockerfile
├── .github/workflows/ci.yml          # Lint, test, build pipeline
└── requirements.txt
```

---

## Getting started

### 1. Clone and set up the environment

```bash
git clone https://github.com/chiraz-Ag/Churn-Prediction-MLops.git
cd Churn-Prediction-MLops
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

### 2. Prepare the data and train a model

Run, in order:
1. `notebooks/eda_churnguard_v2.ipynb` — produces `data/processed/telco_churn_cleaned.csv`
2. `notebooks/data_prep.ipynb` — produces the train/test splits and the encoder/feature artifacts in `models/`
3. Your training notebook/script — produces `models/best_model.joblib`

### 3. Run the API locally

```bash
uvicorn src.api.main:app --reload
```

- Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: `GET /health`
- Prediction: `POST /predict`

Example request body:
```json
{
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
  "TotalCharges": 29.85
}
```

Every successful call to `/predict` is also logged to `logs/predictions.csv` (raw input + prediction + probability + UTC timestamp), which feeds the monitoring pipeline below.

### 4. Run the tests

```bash
pytest
```

---

## Monitoring & drift detection

This project tracks whether live traffic still resembles the data the model was trained on, using [Evidently AI](https://www.evidentlyai.com/).

```bash
# 1. Build the reference dataset (the historical customer population)
python monitoring/generate_reference.py

# 2. (Optional) populate logs/predictions.csv with sample traffic for testing
python monitoring/populate_test_logs.py

# 3. Generate a drift report comparing logs/predictions.csv to the reference
python monitoring/drift_report.py
# -> writes monitoring/reports/drift_report.html
```

### Live dashboard

```bash
streamlit run monitoring/dashboard.py
```

Shows prediction volume, churn rate, probability distribution, and lets you regenerate the drift report from the browser.

---

## CI/CD

Every push triggers a GitHub Actions pipeline (`.github/workflows/ci.yml`):

1. **Lint** — `ruff`
2. **Test** — `pytest`
3. **Build** — `docker build` (image build only, not pushed to a registry)

## Deployment

The API is containerized and deployed on [Render](https://render.com).

- **Live API**: https://churnguard-api-fj06.onrender.com
- **Health check**: https://churnguard-api-fj06.onrender.com/health

> If the service is hosted on a free Render plan, the first request after a period of inactivity may take a while to respond while the container wakes up.

---

## Tech stack

Python · FastAPI · scikit-learn · pandas · Evidently AI · Streamlit · Docker · GitHub Actions · pytest · ruff

---

## Contributors

Built as a two-person portfolio project.
