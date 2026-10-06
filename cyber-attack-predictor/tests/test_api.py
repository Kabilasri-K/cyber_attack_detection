"""
test_api.py
===========
Unit and integration tests for the FastAPI REST API backend.
Tests endpoints:
- GET /
- GET /health
- POST /api/v1/predict
- POST /api/v1/predict/batch
- GET /api/v1/alerts
- GET /api/v1/stats
"""

from fastapi.testclient import TestClient
import pytest
from Backend.api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "database" in data
    assert "ml_models" in data


def test_predict_single_flow():
    payload = {
        "features": {
            "pkt_rate": 100.0,
            "byte_rate": 50000.0,
            "fwd_pkt_rate": 50.0,
            "bwd_pkt_rate": 50.0,
            "fwd_bwd_pkt_ratio": 1.0,
            "syn_flag_ratio": 0.1,
            "ack_flag_ratio": 0.9,
            "rst_flag_ratio": 0.0,
        },
        "source_ip": "10.0.0.15",
        "alert_threshold": 60.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "threat_level" in data
    assert data["threat_level"] in ("Low", "Medium", "High", "Critical")
    assert 0.0 <= data["risk_score"] <= 100.0
    assert 0.0 <= data["attack_probability"] <= 1.0
    assert isinstance(data["top_indicators"], list)


def test_predict_batch_flows():
    payload = {
        "flows": [
            {"pkt_rate": 20.0, "byte_rate": 5000.0, "syn_flag_ratio": 0.0},
            {"pkt_rate": 1500.0, "byte_rate": 800000.0, "syn_flag_ratio": 0.95},
        ],
        "alert_threshold": 50.0,
    }
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert "average_risk_score" in data
    assert len(data["results"]) == 2


def test_list_alerts():
    response = client.get("/api/v1/alerts?limit=5")
    assert response.status_code == 200
    alerts = response.json()
    assert isinstance(alerts, list)


def test_soc_statistics():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_flows_analyzed" in data
    assert "threat_level_distribution" in data
    assert "open_alerts_count" in data
