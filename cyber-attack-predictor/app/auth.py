"""
auth.py
=======
Authentication and Session Management module for the Streamlit dashboard.

Features:
- bcrypt password hashing and verification
- Role-based authorization: 'admin' and 'analyst'
- Persistent Streamlit session state management
- Reusable require_auth() decorator/function to protect all pages
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bcrypt
import streamlit as st
from app.db import get_user_by_username, init_db


# ---------------------------------------------------------------------------
# Password Hashing & Verification (bcrypt)
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """Hash a plain text password using bcrypt with automatic salt generation."""
    salt = bcrypt.gensalt(rounds=12)
    hashed_bytes = bcrypt.hashpw(plain_password.strip().encode("utf-8"), salt)
    return hashed_bytes.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify whether a plain text password matches a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.strip().encode("utf-8"),
            hashed_password.strip().encode("utf-8"),
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Session State Helpers
# ---------------------------------------------------------------------------

def init_auth_session() -> None:
    """Initialize authentication session state keys if not already present."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None


def is_authenticated() -> bool:
    """Check if a valid user is currently logged into the session."""
    init_auth_session()
    return bool(st.session_state.get("authenticated", False))


def get_current_user() -> Optional[Dict[str, Any]]:
    """Retrieve the current logged-in user profile dictionary."""
    init_auth_session()
    return st.session_state.get("user")


def is_admin() -> bool:
    """Check if the currently authenticated user holds the 'admin' role."""
    user = get_current_user()
    return bool(user and user.get("role", "").lower() == "admin")


def login_user(username: str, role: str) -> None:
    """Set authentication session state on successful credential validation."""
    st.session_state["authenticated"] = True
    st.session_state["user"] = {
        "username": username,
        "role": role,
    }


def logout_user() -> None:
    """Clear session state and terminate the current user session."""
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    st.rerun()


# ---------------------------------------------------------------------------
# Streamlit UI: Login Page
# ---------------------------------------------------------------------------

def render_login_page() -> None:
    """Render a modern dark-theme cybersecurity login form."""
    st.markdown(
        """
        <style>
        .login-card {
            max-width: 440px;
            margin: 40px auto 20px auto;
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 14px;
            padding: 32px 36px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
        }
        .login-title {
            text-align: center;
            font-size: 1.6rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 6px;
        }
        .login-subtitle {
            text-align: center;
            font-size: 0.9rem;
            color: #94a3b8;
            margin-bottom: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, col_center, _ = st.columns([1, 1.3, 1])

    with col_center:
        st.markdown(
            """
            <div class="login-card">
                <div style="text-align: center; font-size: 2.8rem; margin-bottom: 8px;">🛡️</div>
                <div class="login-title">SOC Security Access</div>
                <div class="login-subtitle">Cyber Attack Prediction & Early Warning System</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            st.markdown("##### 🔐 Authenticate Credentials")
            username = st.text_input("Username", placeholder="e.g. admin or analyst")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            submit_btn = st.form_submit_button(
                "Access Secure Portal", use_container_width=True, type="primary"
            )

            if submit_btn:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = get_user_by_username(username)
                    if user and verify_password(password, user.password_hash):
                        login_user(user.username, user.role)
                        st.success(f"Welcome back, {user.username}! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid username or password. Access Denied.")

        st.markdown(
            """
            <div style="text-align: center; margin-top: 18px; color: #64748b; font-size: 0.8rem;">
                Authorized personnel only. Sessions are monitored and audited.<br>
                Roles: <strong>admin</strong> | <strong>analyst</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Streamlit UI: Authenticated Sidebar Header
# ---------------------------------------------------------------------------

def render_auth_sidebar() -> None:
    """Render user profile and logout controls in the sidebar."""
    user = get_current_user()
    if not user:
        return

    with st.sidebar:
        role = user.get("role", "analyst").upper()
        role_color = "#ef4444" if role == "ADMIN" else "#06b6d4"

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 10px; padding: 12px 14px; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Operator</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">👤 {user['username']}</div>
                    </div>
                    <div>
                        <span style="background: rgba(255,255,255,0.08); color: {role_color}; border: 1px solid {role_color}88; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 800;">
                            {role}
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🚪 Log Out", use_container_width=True):
            logout_user()


# ---------------------------------------------------------------------------
# Master Protection Function
# ---------------------------------------------------------------------------

def require_auth() -> Dict[str, Any]:
    """Protect the calling page.

    - Initializes database tables.
    - If the user is NOT authenticated, renders the login page and terminates execution.
    - If the user IS authenticated, renders sidebar profile and returns current user info.
    """
    init_db()
    init_auth_session()

    if not is_authenticated():
        render_login_page()
        st.stop()

    render_auth_sidebar()
    return get_current_user() or {}
