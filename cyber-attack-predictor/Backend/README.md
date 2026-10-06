# Backend (FastAPI & PostgreSQL / MySQL ML Engine)

This folder contains the **FastAPI REST API, PostgreSQL / MySQL database layer (SQLAlchemy), ML inference engine, utilities, and tests** for the Cyber Attack Predictor application matching Section 8 of the project specification.

## Structure

```
Backend/
├── api.py               # Production FastAPI REST API (/predict, /alerts, /stats, /docs)
├── db.py                # Multi-engine SQLAlchemy ORM (PostgreSQL, MySQL, SQLite)
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
    ├── test_api.py          # Unit & integration tests for FastAPI REST API
    ├── test_predict.py      # Unit/integration tests for prediction pipeline
    ├── make_sample_data.py  # Test data generation utilities
    └── __init__.py
```

## Key Components

| Module | Technology | Purpose |
|---|---|---|
| `api.py` | **FastAPI** + Pydantic + Uvicorn | Production REST API with auto-generated Swagger UI (`/docs`) |
| `db.py` | **PostgreSQL / MySQL / SQLite** (SQLAlchemy) | ORM models (`User`, `Prediction`, `Alert`) with dynamic connection pooling |
| `src/predict.py` | **XGBoost** + **Isolation Forest** | Unified inference interface coordinating classification & zero-day anomaly detection |
| `src/risk_engine.py` | Python / NumPy | Dynamic composite risk score calculation (0–100 scale: Low, Med, High, Critical) |
| `src/explain.py` | **SHAP** TreeExplainer | Natural-language top feature drivers for SOC analysts |

## Database Configuration (PostgreSQL / MySQL)

By default, the backend automatically uses SQLite if no database credentials are provided. To connect to PostgreSQL or MySQL, configure `DATABASE_URL` or environment variables:

### PostgreSQL
```bash
export DATABASE_URL="postgresql+psycopg2://postgres:password@localhost:5432/cyber_threat"
```

### MySQL
```bash
export DATABASE_URL="mysql+pymysql://root:password@localhost:3306/cyber_threat"
```

## Launching the FastAPI Backend

From the project root (`cyber-attack-predictor/`):

```bash
uvicorn Backend.api:app --host 0.0.0.0 --port 8000 --reload
```

- Interactive OpenAPI Swagger UI: `http://localhost:8000/docs`
- Redoc Documentation: `http://localhost:8000/redoc`
- Standalone HTML/CSS/JS Frontend Client: `http://localhost:8000/ui`

## Running Tests

```bash
pytest tests/ -v
```
