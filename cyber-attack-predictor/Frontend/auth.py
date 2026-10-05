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
from Backend.db import get_user_by_username, init_db, create_user


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
    """Render a clean cybersecurity login/register form with dark theme forced."""
    if "auth_view" not in st.session_state:
        st.session_state["auth_view"] = "login"

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Force all text to be light so it's always visible */
        p, label, span, div, h1, h2, h3, h4, h5, h6,
        .stTextInput label, .stSelectbox label, .stCaption {
            color: #e2e8f0 !important;
        }

        .stTextInput > div > div > input {
            background-color: #1e293b !important;
            color: #f1f5f9 !important;
            border: 1px solid #334155 !important;
        }
        .stTextInput > div > div > input::placeholder {
            color: #64748b !important;
        }

        .auth-header {
            text-align: center;
            padding: 1.5rem 0 1rem 0;
        }
        .auth-header h1 {
            font-size: 1.7rem;
            font-weight: 800;
            color: #f1f5f9 !important;
            margin: 8px 0 4px 0;
        }
        .auth-header p {
            color: #94a3b8 !important;
            font-size: 0.88rem;
        }
        .panel-label {
            font-size: 1.05rem;
            font-weight: 700;
            color: #f1f5f9 !important;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid rgba(79,70,229,0.6);
            margin-bottom: 0.6rem;
        }
        .auth-footer {
            text-align: center;
            margin-top: 20px;
            color: #475569 !important;
            font-size: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        """
        <div class="auth-header">
            <div style="font-size:3rem;">🛡️</div>
            <h1>SOC Security Access</h1>
            <p>Cyber Attack Prediction &amp; Early Warning System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    view = st.session_state["auth_view"]

    _, col, _ = st.columns([1, 1.5, 1])

    with col:

        # ── SIGN IN VIEW ──────────────────────────────────────────────────────
        if view == "login":
            st.markdown(
                "<div style='color:#94a3b8; font-size:0.85rem; margin-bottom:0.8rem;'>"
                "Haven't got an account? Click <strong>Create account</strong> below ↓</div>",
                unsafe_allow_html=True,
            )

            with st.form("auth_login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                sign_in_btn = st.form_submit_button("✅ Sign In", use_container_width=True, type="primary")

                if sign_in_btn:
                    if not username or not password:
                        st.error("Please enter your username and password.")
                    else:
                        user = get_user_by_username(username)
                        if user and verify_password(password, user.password_hash):
                            login_user(user.username, user.role)
                            st.success(f"Welcome back, **{user.username}**! 🎉")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password.")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔑 Forgot password?", use_container_width=True, key="btn_forgot"):
                    st.session_state["auth_view"] = "forgot"
                    st.rerun()
            with c2:
                if st.button("🆕 Create account", use_container_width=True, key="btn_register"):
                    st.session_state["auth_view"] = "register"
                    st.rerun()

        # ── CREATE ACCOUNT VIEW ───────────────────────────────────────────────
        elif view == "register":
            st.markdown('<div class="panel-label">🆕 Create Account</div>', unsafe_allow_html=True)
            st.markdown(
                "<div style='color:#94a3b8;font-size:0.84rem;margin-bottom:0.8rem;'>"
                "Fill in the details below to get started.</div>",
                unsafe_allow_html=True,
            )

            with st.form("auth_register_form", clear_on_submit=True):
                full_name        = st.text_input("Full Name", placeholder="e.g. John Doe")
                new_username     = st.text_input("Create Username", placeholder="Choose a unique username")
                new_password     = st.text_input("Create Password", type="password", placeholder="Min 8 chars, 1 letter + 1 number")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat your password")
                role_choice      = st.selectbox("Select Role", ["analyst", "admin"])
                reg_btn = st.form_submit_button("🚀 Create Account", use_container_width=True, type="primary")

                if reg_btn:
                    if not full_name or not new_username or not new_password:
                        st.error("Please fill in all fields.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    elif len(new_password) < 8:
                        st.error("Password must be at least 8 characters.")
                    else:
                        existing = get_user_by_username(new_username)
                        if existing:
                            st.error("⚠️ That username is already registered. Please sign in.")
                        else:
                            create_user(new_username, hash_password(new_password), role_choice)
                            st.success("✅ Account created! You can now sign in.")
                            st.session_state["auth_view"] = "login"
                            st.rerun()

            if st.button("← Back to Sign In", use_container_width=True, key="btn_back_login"):
                st.session_state["auth_view"] = "login"
                st.rerun()

        # ── FORGOT PASSWORD VIEW ──────────────────────────────────────────────
        elif view == "forgot":
            st.markdown('<div class="panel-label">🔑 Forgot Password</div>', unsafe_allow_html=True)
            st.info("Enter your username and a reset link will be generated.")

            with st.form("auth_forgot_form", clear_on_submit=True):
                reset_user = st.text_input("Username", placeholder="Enter your username")
                forgot_btn = st.form_submit_button("Send Reset Link", use_container_width=True, type="primary")

                if forgot_btn:
                    if not reset_user:
                        st.error("Please enter your username.")
                    else:
                        # Always same message for security
                        st.success("✅ If that username exists, a reset link has been sent.")
                        user = get_user_by_username(reset_user)
                        if user:
                            import secrets, datetime
                            token  = secrets.token_urlsafe(32)
                            expiry = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
                            print(f"\n{'='*55}")
                            print(f"[DEV] Reset link for: {reset_user}")
                            print(f"Token  : {token}")
                            print(f"Expires: {expiry.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                            print(f"Link   : http://localhost:8501/reset?token={token}")
                            print(f"{'='*55}\n")

            if st.button("← Back to Sign In", use_container_width=True, key="btn_back_forgot"):
                st.session_state["auth_view"] = "login"
                st.rerun()

    # Footer
    st.markdown(
        """
        <div class="auth-footer">
            🔒 Authorized personnel only — sessions are monitored and audited.<br>
            Roles: <strong style="color:#94a3b8">admin</strong> &nbsp;|&nbsp;
            <strong style="color:#94a3b8">analyst</strong>
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
