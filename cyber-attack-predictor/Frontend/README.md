# Frontend

This folder contains the **Streamlit UI layer** of the Cyber Attack Predictor application.

> **Note:** This is a Streamlit application, so "Frontend" refers to the Python-based UI pages — not HTML/CSS/JS.

## Structure

```
Frontend/
├── Home.py              # Main dashboard page (entry point for Streamlit)
├── auth.py              # Authentication & session management (login UI + logic)
├── __init__.py
├── .streamlit/
│   └── config.toml      # Streamlit theme configuration
└── pages/
    ├── 1_Alerts.py      # Security Alerts Management & Triage page
    ├── 2_Trends.py      # Threat Trends & Historical Analytics page
    └── 3_History.py     # Prediction Audit Log & History page
```

## Running the Frontend

From the project root (`cyber-attack-predictor/`):

```bash
streamlit run Frontend/Home.py
```

## Dependencies

- Imports authentication from `Frontend.auth`
- Imports database functions from `Backend.db`
- Imports ML logic from `Backend.src.*`
