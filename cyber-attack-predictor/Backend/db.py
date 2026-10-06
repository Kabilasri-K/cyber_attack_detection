"""
db.py
=====
SQLAlchemy database setup and schema definitions supporting:
- PostgreSQL (via psycopg2)
- MySQL (via pymysql)
- SQLite (local development / fallback)

Tables:
- users: id, username, password_hash, role
- predictions: id, timestamp, risk_score, threat_level, attack_probability, anomaly_score, ...
- alerts: id, timestamp, source, risk_score, threat_level, attack_probability, status
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

logger = logging.getLogger("cyber_db")

# ---------------------------------------------------------------------------
# Database Configuration (PostgreSQL / MySQL / SQLite)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_PATH = DB_DIR / "cyber_threat.db"
SQLITE_DEFAULT_URL = f"sqlite:///{SQLITE_PATH.as_posix()}"


def build_database_url() -> str:
    """
    Build database URL supporting:
    1. DATABASE_URL environment variable (PostgreSQL, MySQL, or SQLite)
    2. Explicit DB_TYPE (postgresql, mysql) with DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
    3. Safe local fallback to SQLite
    """
    raw_url = os.getenv("DATABASE_URL")
    if raw_url:
        if raw_url.startswith("postgres://"):
            return raw_url.replace("postgres://", "postgresql+psycopg2://", 1)
        if raw_url.startswith("postgresql://") and "+psycopg2" not in raw_url:
            return raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        if raw_url.startswith("mysql://") and "+pymysql" not in raw_url:
            return raw_url.replace("mysql://", "mysql+pymysql://", 1)
        return raw_url

    db_type = os.getenv("DB_TYPE", "").strip().lower()
    if db_type in ("postgresql", "postgres"):
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "postgres")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("DB_NAME", "cyber_threat")
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"

    if db_type == "mysql":
        user = os.getenv("DB_USER", "root")
        password = os.getenv("DB_PASSWORD", "")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "3306")
        name = os.getenv("DB_NAME", "cyber_threat")
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}"

    return SQLITE_DEFAULT_URL


def init_engine():
    target_url = build_database_url()
    is_sqlite = target_url.startswith("sqlite")

    if not is_sqlite:
        try:
            eng = create_engine(target_url, pool_pre_ping=True, echo=False)
            with eng.connect():
                logger.info("Successfully connected to primary database: %s", target_url.split("@")[-1])
            return eng, target_url
        except Exception as err:
            logger.warning(
                "Primary database (%s) unavailable: %s. Falling back to local SQLite.",
                target_url.split("@")[-1],
                err,
            )

    eng = create_engine(SQLITE_DEFAULT_URL, connect_args={"check_same_thread": False}, echo=False)
    return eng, SQLITE_DEFAULT_URL


engine, DB_URL = init_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



# ---------------------------------------------------------------------------
# Declarative Base & Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


class User(Base):
    """Users table for role-based authentication (admin, analyst)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="analyst", nullable=False)  # "admin" | "analyst"
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Prediction(Base):
    """Predictions table logging flow risk evaluations."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False, index=True
    )
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    threat_level: Mapped[str] = mapped_column(String(32), nullable=False)
    attack_probability: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_attack_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    top_indicators: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    alert: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    priority: Mapped[str] = mapped_column(String(32), default="Low", nullable=False)
    analyzed_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "risk_score": self.risk_score,
            "threat_level": self.threat_level,
            "attack_probability": self.attack_probability,
            "anomaly_score": self.anomaly_score,
            "predicted_attack_type": self.predicted_attack_type,
            "top_indicators": self.top_indicators,
            "alert": self.alert,
            "priority": self.priority,
            "analyzed_by": self.analyzed_by,
        }


class Alert(Base):
    """Alerts table recording critical and high risk early warning triggers."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(String(128), default="Network Flow Sensor", nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    threat_level: Mapped[str] = mapped_column(String(32), nullable=False)
    attack_probability: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="Open", nullable=False)  # "Open" | "Investigating" | "Resolved"
    resolved_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
            "risk_score": self.risk_score,
            "threat_level": self.threat_level,
            "attack_probability": self.attack_probability,
            "status": self.status,
            "resolved_by": self.resolved_by,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Database Management Helper Functions
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables in the SQLite database and run backward-compatible column checks."""
    Base.metadata.create_all(bind=engine)
    try:
        with engine.connect() as conn:
            cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(alerts)").fetchall()]
            if cols and "attack_probability" not in cols:
                conn.exec_driver_sql("ALTER TABLE alerts ADD COLUMN attack_probability FLOAT DEFAULT 0.5")
                conn.commit()
    except Exception:
        pass


def get_db():
    """Context manager / generator yielding a scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user_by_username(username: str) -> Optional[User]:
    """Retrieve user entity by unique username."""
    with SessionLocal() as session:
        stmt = select(User).where(User.username == username.strip())
        return session.scalars(stmt).first()


def create_user(username: str, password_hash: str, role: str = "analyst") -> User:
    """Create and persist a new user."""
    with SessionLocal() as session:
        user = User(
            username=username.strip(),
            password_hash=password_hash,
            role=role.strip().lower(),
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def list_users() -> List[Dict[str, Any]]:
    """Return list of all registered users."""
    with SessionLocal() as session:
        stmt = select(User).order_by(User.id.asc())
        users = session.scalars(stmt).all()
        return [u.to_dict() for u in users]


def log_prediction(pred_data: Dict[str, Any], username: Optional[str] = None) -> Prediction:
    """Record a prediction in the database."""
    with SessionLocal() as session:
        indicators = pred_data.get("top_indicators", [])
        top_str = "\n".join(indicators) if isinstance(indicators, list) else str(indicators)

        prediction = Prediction(
            risk_score=float(pred_data.get("risk_score", 0.0)),
            threat_level=str(pred_data.get("threat_level", "Low")),
            attack_probability=float(pred_data.get("attack_probability", 0.0)),
            anomaly_score=float(pred_data.get("anomaly_score", 0.0)),
            predicted_attack_type=pred_data.get("predicted_attack_type", "BENIGN"),
            top_indicators=top_str,
            alert=bool(pred_data.get("alert", False)),
            priority=str(pred_data.get("priority", "Low")),
            analyzed_by=username,
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)
        return prediction


def log_alert(
    risk_score: float,
    threat_level: str,
    attack_probability: float = 0.5,
    source: str = "Network Flow Sensor",
    status: str = "Open",
    notes: Optional[str] = None,
) -> Alert:
    """Record an alert in the database."""
    with SessionLocal() as session:
        alert = Alert(
            risk_score=float(risk_score),
            threat_level=threat_level,
            attack_probability=float(attack_probability),
            source=source,
            status=status,
            notes=notes,
        )
        session.add(alert)
        session.commit()
        session.refresh(alert)
        return alert


def get_recent_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve the most recent security alerts."""
    with SessionLocal() as session:
        stmt = select(Alert).order_by(Alert.timestamp.desc()).limit(limit)
        alerts = session.scalars(stmt).all()
        return [a.to_dict() for a in alerts]


def get_alerts(
    threat_level: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Retrieve filtered alerts from the database."""
    with SessionLocal() as session:
        stmt = select(Alert)
        if threat_level and threat_level != "All":
            stmt = stmt.where(Alert.threat_level == threat_level)
        if status and status != "All":
            stmt = stmt.where(Alert.status == status)

        stmt = stmt.order_by(Alert.timestamp.desc()).limit(limit)
        alerts = session.scalars(stmt).all()
        return [a.to_dict() for a in alerts]


def update_alert_status(
    alert_id: int,
    new_status: str,
    resolved_by: Optional[str] = None,
    notes: Optional[str] = None,
) -> bool:
    """Update status and resolution information for an alert."""
    with SessionLocal() as session:
        alert = session.get(Alert, alert_id)
        if not alert:
            return False
        alert.status = new_status
        if resolved_by:
            alert.resolved_by = resolved_by
        if notes:
            alert.notes = notes
        session.commit()
        return True


def get_predictions(
    threat_level: Optional[str] = None,
    attack_type: Optional[str] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Retrieve prediction audit log records from the database."""
    with SessionLocal() as session:
        stmt = select(Prediction)
        if threat_level and threat_level != "All":
            stmt = stmt.where(Prediction.threat_level == threat_level)
        if attack_type and attack_type != "All":
            stmt = stmt.where(Prediction.predicted_attack_type == attack_type)

        stmt = stmt.order_by(Prediction.timestamp.desc()).limit(limit)
        predictions = session.scalars(stmt).all()
        return [p.to_dict() for p in predictions]


def seed_demo_telemetry() -> None:
    """Seed initial realistic history and alert telemetry if tables are empty."""
    import random
    with SessionLocal() as session:
        count = session.query(Prediction).count()
        if count >= 10:
            return

        now = datetime.datetime.utcnow()
        scenarios = [
            ("DoS Hulk", 88.5, "Critical", 0.94, 0.82, True, "Critical", "Volumetric packet flood detected."),
            ("PortScan", 68.2, "High", 0.76, 0.58, True, "High", "Closed port reset scan observed."),
            ("BENIGN", 12.4, "Low", 0.04, 0.11, False, "Low", "Standard HTTPS browsing."),
            ("DoS Hulk", 91.0, "Critical", 0.96, 0.85, True, "Critical", "Extreme SYN buffer saturation."),
            ("BENIGN", 18.0, "Low", 0.06, 0.15, False, "Low", "Normal DNS resolution query."),
            ("PortScan", 72.0, "High", 0.79, 0.62, True, "High", "Sequential port sweep probe."),
            ("BENIGN", 8.5, "Low", 0.02, 0.09, False, "Low", "Normal background telemetry."),
            ("DDoS Low", 48.0, "Medium", 0.42, 0.60, False, "Medium", "Elevated forward packet rate."),
        ]

        for i in range(24):
            sc = random.choice(scenarios)
            ts = now - datetime.timedelta(hours=24 - i, minutes=random.randint(0, 50))
            pred = Prediction(
                timestamp=ts,
                risk_score=sc[1] + random.uniform(-3, 3),
                threat_level=sc[2],
                attack_probability=sc[3],
                anomaly_score=sc[4],
                predicted_attack_type=sc[0],
                top_indicators=sc[7],
                alert=sc[5],
                priority=sc[6],
                analyzed_by="system_sensor",
            )
            session.add(pred)

            if sc[5]:
                alert = Alert(
                    timestamp=ts,
                    source=f"Sensor-{random.randint(1, 4)}:192.168.1.{random.randint(10, 80)}",
                    risk_score=sc[1],
                    threat_level=sc[2],
                    attack_probability=sc[3],
                    status=random.choice(["Open", "Investigating", "Resolved"]),
                    notes=sc[7],
                )
                session.add(alert)

        session.commit()
