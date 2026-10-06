"""
api.py
======
Production-grade FastAPI REST API for the Cyber Attack Prediction & Early Warning System.

Satisfies Section 8 of the Project Specification:
- Backend: FastAPI
- Database: PostgreSQL / MySQL / SQLite (configured via SQLAlchemy in db.py)
- Endpoints:
  - GET  /health                 : Health check & system status
  - POST /api/v1/predict         : Single flow prediction & risk scoring
  - POST /api/v1/predict/batch   : Batch flow prediction
  - GET  /api/v1/predictions     : Audit log of past predictions
  - GET  /api/v1/alerts          : Early warning priority queue
  - PATCH /api/v1/alerts/{id}    : SOC triage & status update
  - GET  /api/v1/stats           : Aggregate threat statistics
  - POST /api/v1/auth/login      : User authentication
  - POST /api/v1/auth/register   : User registration
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Database & Prediction imports
from app.db import (
    DB_URL,
    Alert,
    Prediction,
    User,
    create_user,
    get_alerts,
    get_predictions,
    get_user_by_username,
    init_db,
    list_users,
    log_alert,
    log_prediction,
    seed_demo_telemetry,
    update_alert_status,
)
from src.predict import ModelRegistry, predict

from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cyber_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to initialize database and cache models."""
    logger.info("Initializing database schema...")
    init_db()
    seed_demo_telemetry()
    logger.info("Database initialized. Pre-caching ML models...")
    try:
        ModelRegistry.load()
        logger.info("ML Models (XGBoost & Isolation Forest) pre-cached successfully.")
    except Exception as e:
        logger.warning("Could not pre-load models during startup: %s", e)
    yield


# Initialize FastAPI App
app = FastAPI(
    title="🛡️ Cyber Attack Prediction & Early Warning REST API",
    description="""
High-throughput REST API for real-time network flow telemetry classification,
zero-day anomaly detection (Isolation Forest), attack classification (XGBoost),
dynamic risk calculation (0-100), and explainable AI (SHAP).
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS) for external frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ------------------------------------------------------------------------------

class FlowTelemetryInput(BaseModel):
    """Network flow telemetry features for inference."""
    features: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature dictionary containing network flow measurements (e.g., pkt_rate, byte_rate, syn_flag_ratio).",
        json_schema_extra={
            "example": {
                "pkt_rate": 850.5,
                "byte_rate": 142000.0,
                "fwd_pkt_rate": 810.0,
                "bwd_pkt_rate": 40.5,
                "fwd_bwd_pkt_ratio": 20.0,
                "syn_flag_ratio": 0.92,
                "ack_flag_ratio": 0.05,
                "rst_flag_ratio": 0.01,
            }
        },
    )
    source_ip: Optional[str] = Field("192.168.1.105", description="Optional source IP / sensor identifier")
    alert_threshold: Optional[float] = Field(60.0, description="Risk threshold for early warning alert trigger (0-100)")


class BatchFlowTelemetryInput(BaseModel):
    """Batch flow telemetry container."""
    flows: List[Dict[str, float]] = Field(..., description="List of flow telemetry feature dictionaries")
    alert_threshold: Optional[float] = Field(60.0, description="Alert trigger threshold (0-100)")


class PredictionResult(BaseModel):
    """Inference and threat assessment response."""
    risk_score: float = Field(..., description="Composite dynamic threat score from 0.0 to 100.0")
    threat_level: str = Field(..., description="Triage tier: Low (0-25), Medium (26-50), High (51-75), Critical (76-100)")
    attack_probability: float = Field(..., description="Probability of an active attack from 0.0 to 1.0")
    anomaly_score: float = Field(..., description="Normalized zero-day anomaly score from 0.0 to 1.0")
    predicted_attack_type: str = Field(..., description="Classified attack category (e.g. BENIGN, DoS Hulk, PortScan)")
    top_indicators: List[str] = Field(default_factory=list, description="Top SHAP feature drivers in plain English")
    alert: bool = Field(..., description="True if early warning condition met")
    priority: str = Field(..., description="Investigation priority: Low, Medium, High, Critical")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp")
    database_id: Optional[int] = Field(None, description="Persisted record ID in PostgreSQL / MySQL database")


class AlertItem(BaseModel):
    """Security alert record."""
    id: int
    timestamp: Optional[str]
    source: str
    risk_score: float
    threat_level: str
    attack_probability: float
    status: str
    resolved_by: Optional[str] = None
    notes: Optional[str] = None


class AlertStatusUpdate(BaseModel):
    """Payload to triage or update an alert."""
    status: str = Field(..., description="Target status: 'Open', 'Investigating', or 'Resolved'")
    resolved_by: Optional[str] = Field(None, description="Analyst username taking action")
    notes: Optional[str] = Field(None, description="Investigation notes or triage justification")


class AuthRequest(BaseModel):
    """User credentials."""
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6)


class UserResponse(BaseModel):
    """User representation."""
    id: int
    username: str
    role: str
    created_at: Optional[str]



@app.get("/", tags=["General"])
def root() -> Dict[str, Any]:
    """Root metadata endpoint with navigation links."""
    return {
        "title": "Cyber Attack Prediction & Early Warning System REST API",
        "version": "2.0.0",
        "status": "online",
        "documentation": "/docs",
        "alternative_docs": "/redoc",
        "web_dashboard": "http://localhost:8501",
        "database_backend": "PostgreSQL / MySQL / SQLite (SQLAlchemy)",
        "active_db_url": DB_URL.split("@")[-1] if "@" in DB_URL else DB_URL,
    }


@app.get("/health", tags=["General"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint monitoring models and database connectivity."""
    db_status = "connected"
    try:
        users = list_users()
        user_count = len(users)
    except Exception as e:
        db_status = f"error: {str(e)}"
        user_count = 0

    model_status = "loaded" if ModelRegistry._xgb_model is not None else "ready_on_demand"

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "database": {
            "status": db_status,
            "engine": DB_URL.split("://")[0] if "://" in DB_URL else "unknown",
            "users_count": user_count,
        },
        "ml_models": {
            "status": model_status,
            "supervised": "XGBoost Attack Classifier",
            "unsupervised": "Isolation Forest Anomaly Detector",
            "explainability": "SHAP TreeExplainer",
        },
    }


# ------------------------------------------------------------------------------
# Core Prediction Endpoints
# ------------------------------------------------------------------------------

@app.post("/api/v1/predict", response_model=PredictionResult, tags=["Predictions"])
def predict_flow(payload: FlowTelemetryInput) -> PredictionResult:
    """
    Evaluate a single network telemetry flow record.
    
    Coordinates:
    1. Multi-class XGBoost attack prediction
    2. Unsupervised Isolation Forest anomaly detection
    3. Dynamic Risk Engine calculation (0-100)
    4. SHAP explainability feature attribution
    5. Automatic persistence into PostgreSQL / MySQL / SQLite database
    """
    try:
        predictions = predict(payload.features, alert_threshold=payload.alert_threshold or 60.0)
        if not predictions:
            raise HTTPException(status_code=400, detail="Inference returned empty results.")

        res = predictions[0]

        # Persist prediction in the database
        logged_pred = log_prediction(res, username="api_client")

        # If alert condition is triggered, log to early warning queue
        if res.get("alert"):
            log_alert(
                risk_score=res["risk_score"],
                threat_level=res["threat_level"],
                attack_probability=res["attack_probability"],
                source=payload.source_ip or "Network Flow Sensor",
                status="Open",
                notes=", ".join(res.get("top_indicators", [])) or "Alert triggered by API threshold",
            )

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return PredictionResult(
            risk_score=res["risk_score"],
            threat_level=res["threat_level"],
            attack_probability=res["attack_probability"],
            anomaly_score=res["anomaly_score"],
            predicted_attack_type=res["predicted_attack_type"],
            top_indicators=res.get("top_indicators", []),
            alert=res["alert"],
            priority=res["priority"],
            timestamp=now_iso,
            database_id=logged_pred.id if logged_pred else None,
        )
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifacts not found: {str(fnf)}. Please run training first.",
        )
    except Exception as e:
        logger.exception("Inference error occurred")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/predict/batch", tags=["Predictions"])
def predict_batch(payload: BatchFlowTelemetryInput) -> Dict[str, Any]:
    """Run bulk inference across multiple network flow records."""
    if not payload.flows:
        raise HTTPException(status_code=400, detail="Empty flows payload provided.")

    try:
        results = predict(payload.flows, alert_threshold=payload.alert_threshold or 60.0)

        # Bulk log to database
        alert_count = 0
        for r in results:
            log_prediction(r, username="api_batch_client")
            if r.get("alert"):
                alert_count += 1
                log_alert(
                    risk_score=r["risk_score"],
                    threat_level=r["threat_level"],
                    attack_probability=r["attack_probability"],
                    source="Batch Ingestion Pipeline",
                    status="Open",
                    notes="Batch trigger: " + (", ".join(r.get("top_indicators", []))),
                )

        return {
            "total_processed": len(results),
            "alerts_triggered": alert_count,
            "average_risk_score": round(sum(r["risk_score"] for r in results) / len(results), 2),
            "results": results,
        }
    except Exception as e:
        logger.exception("Batch prediction failure")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# Alert & Telemetry Management Endpoints
# ------------------------------------------------------------------------------

@app.get("/api/v1/alerts", response_model=List[AlertItem], tags=["Alerts"])
def list_alerts(
    threat_level: Optional[str] = Query(None, description="Filter by threat level: Low, Medium, High, Critical"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status: Open, Investigating, Resolved"),
    limit: int = Query(50, ge=1, le=500),
) -> List[AlertItem]:
    """Retrieve security alerts sorted chronologically descending."""
    raw_alerts = get_alerts(threat_level=threat_level, status=status_filter, limit=limit)
    return [
        AlertItem(
            id=a["id"],
            timestamp=a["timestamp"],
            source=a["source"],
            risk_score=a["risk_score"],
            threat_level=a["threat_level"],
            attack_probability=a["attack_probability"],
            status=a["status"],
            resolved_by=a.get("resolved_by"),
            notes=a.get("notes"),
        )
        for a in raw_alerts
    ]


@app.patch("/api/v1/alerts/{alert_id}", tags=["Alerts"])
def patch_alert(alert_id: int, payload: AlertStatusUpdate) -> Dict[str, Any]:
    """Update an alert's status and investigation notes."""
    valid_statuses = {"Open", "Investigating", "Resolved"}
    if payload.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{payload.status}'. Must be one of: {sorted(list(valid_statuses))}",
        )

    updated = update_alert_status(
        alert_id=alert_id,
        new_status=payload.status,
        resolved_by=payload.resolved_by,
        notes=payload.notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Alert with ID {alert_id} not found.")

    return {"message": "Alert status updated successfully", "alert_id": alert_id, "new_status": payload.status}


@app.get("/api/v1/predictions", tags=["Audit Log"])
def list_predictions(
    threat_level: Optional[str] = Query(None, description="Filter by threat level"),
    attack_type: Optional[str] = Query(None, description="Filter by predicted attack type"),
    limit: int = Query(100, ge=1, le=1000),
) -> List[Dict[str, Any]]:
    """Retrieve historical prediction audit trail records from the database."""
    return get_predictions(threat_level=threat_level, attack_type=attack_type, limit=limit)


@app.get("/api/v1/stats", tags=["SOC Metrics"])
def get_soc_statistics() -> Dict[str, Any]:
    """Retrieve high-level SOC security operations statistics."""
    all_preds = get_predictions(limit=1000)
    all_alerts = get_alerts(limit=500)

    total_predictions = len(all_preds)
    avg_risk = (
        round(sum(p["risk_score"] for p in all_preds) / total_predictions, 1)
        if total_predictions > 0
        else 0.0
    )

    threat_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    attack_counts: Dict[str, int] = {}
    for p in all_preds:
        lvl = p.get("threat_level", "Low")
        threat_counts[lvl] = threat_counts.get(lvl, 0) + 1
        atk = p.get("predicted_attack_type", "BENIGN")
        attack_counts[atk] = attack_counts.get(atk, 0) + 1

    open_alerts = sum(1 for a in all_alerts if a.get("status") == "Open")

    return {
        "total_flows_analyzed": total_predictions,
        "average_risk_score": avg_risk,
        "open_alerts_count": open_alerts,
        "threat_level_distribution": threat_counts,
        "attack_type_distribution": attack_counts,
        "active_database_engine": DB_URL.split("://")[0] if "://" in DB_URL else "sqlite",
    }


# ------------------------------------------------------------------------------
# Authentication Endpoints (bcrypt & SQLAlchemy)
# ------------------------------------------------------------------------------

@app.post("/api/v1/auth/register", response_model=UserResponse, tags=["Authentication"])
def register(payload: AuthRequest) -> UserResponse:
    """Register a new analyst or security operator."""
    import bcrypt

    existing = get_user_by_username(payload.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username is already registered.")

    salt = bcrypt.gensalt(rounds=12)
    pwd_hash = bcrypt.hashpw(payload.password.encode("utf-8"), salt).decode("utf-8")
    user = create_user(username=payload.username, password_hash=pwd_hash, role="analyst")

    return UserResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        created_at=user.created_at.isoformat() if user.created_at else None,
    )


@app.post("/api/v1/auth/login", tags=["Authentication"])
def login(payload: AuthRequest) -> Dict[str, Any]:
    """Authenticate user credentials."""
    import bcrypt

    user = get_user_by_username(payload.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    if not bcrypt.checkpw(payload.password.encode("utf-8"), user.password_hash.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    return {
        "status": "authenticated",
        "user": user.to_dict(),
        "token_type": "bearer",
        "session": "active",
    }


# ------------------------------------------------------------------------------
# Mount HTML / CSS / JS Static Frontend Client
# ------------------------------------------------------------------------------

STATIC_DIR = PROJECT_ROOT / "Frontend" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/ui", response_class=HTMLResponse, tags=["Frontend Client"])
    def serve_frontend_ui():
        """Serve the standalone HTML / CSS / JavaScript SOC client."""
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h2>Frontend UI under construction.</h2>")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("Backend.api:app", host=host, port=port, reload=True)
