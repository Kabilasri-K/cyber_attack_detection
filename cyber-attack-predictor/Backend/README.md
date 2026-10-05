# Backend

This folder contains the **ML engine, database layer, utilities, and tests** for the Cyber Attack Predictor application.

## Structure

```
Backend/
├── db.py                # SQLAlchemy database setup (SQLite: users, predictions, alerts)
├── __init__.py
├── src/
│   ├── predict.py       # Unified prediction pipeline (XGBoost + Isolation Forest)
│   ├── train.py         # Model training and evaluation pipeline
│   ├── preprocessing.py # Data cleaning and preprocessing
│   ├── features.py      # Feature engineering pipeline
│   ├── risk_engine.py   # Dynamic risk scoring and alert thresholding
│   ├── explain.py       # SHAP-based model explainability
│   └── __init__.py
├── scripts/
│   ├── seed_admin.py    # Seeds initial admin user into the database
│   └── __init__.py
└── tests/
    ├── test_predict.py      # Unit/integration tests for prediction pipeline
    ├── make_sample_data.py  # Test data generation utilities
    ├── run_eda_cells.py     # EDA notebook runner
    ├── validate_notebook.py # Notebook validation script
    └── __init__.py
```

## Key Components

| Module | Purpose |
|---|---|
| `db.py` | SQLite ORM models (User, Predictions, Alerts) and CRUD helpers |
| `src/predict.py` | Main `predict(df)` interface coordinating all ML inference |
| `src/train.py` | XGBoost classifier + Isolation Forest training pipeline |
| `src/preprocessing.py` | Raw CSV → clean Parquet pipeline |
| `src/features.py` | Feature engineering: rate, directional, TCP flags, packet size, duration |
| `src/risk_engine.py` | Risk score calculation, threat level mapping, alert thresholds |
| `src/explain.py` | SHAP-based natural language feature attribution explanations |
| `scripts/seed_admin.py` | CLI tool to bootstrap admin credentials |

## Running Training

From the project root (`cyber-attack-predictor/`):

```bash
python -m Backend.src.preprocessing   # Step 1: Clean raw CSVs
python -m Backend.src.features         # Step 2: Engineer features
python -m Backend.src.train            # Step 3: Train models
```

## Running Tests

```bash
pytest Backend/tests/test_predict.py
```

## Seeding Admin User

```bash
ADMIN_PASSWORD="YourSecurePassword!" python Backend/scripts/seed_admin.py
```
