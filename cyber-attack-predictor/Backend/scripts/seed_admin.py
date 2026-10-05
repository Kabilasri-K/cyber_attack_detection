"""
seed_admin.py
=============
Seeds an initial administrator user into the SQLite database.

Security:
- Never hardcodes passwords in the source code.
- Accepts password via ADMIN_PASSWORD environment variable or --password CLI argument.
- Falls back to interactive prompt (getpass) or generates a cryptographically secure random password.
- Uses bcrypt with automatic salting to store only the password hash.

Usage Examples:
    # 1. Via Environment Variable (Recommended for CI/CD)
    ADMIN_PASSWORD="MySecureAdminPass2026!" python scripts/seed_admin.py

    # 2. Via CLI argument
    python scripts/seed_admin.py --username admin --password "MySecureAdminPass2026!"

    # 3. Auto-generated secure random password
    python scripts/seed_admin.py --username admin
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Frontend.auth import hash_password
from Backend.db import User, SessionLocal, get_user_by_username, init_db


def seed_admin_user(
    username: str = "admin",
    password: str | None = None,
    role: str = "admin",
) -> None:
    """Create or update an administrative user with a bcrypt-hashed password."""
    # Ensure database schema exists
    init_db()

    # 1. Resolve password securely (no hardcoded credentials)
    resolved_password = password or os.environ.get("ADMIN_PASSWORD")

    generated = False
    if not resolved_password:
        if sys.stdin.isatty():
            import getpass
            try:
                resolved_password = getpass.getpass(f"Enter password for '{username}': ")
            except Exception:
                resolved_password = None

        if not resolved_password:
            # Fall back to generating a cryptographically secure random password
            resolved_password = secrets.token_urlsafe(16)
            generated = True

    # 2. Hash the password with bcrypt
    pw_hash = hash_password(resolved_password)

    # 3. Persist to database
    with SessionLocal() as session:
        existing_user = session.query(User).filter(User.username == username.strip()).first()
        if existing_user:
            print(f"[*] User '{username}' already exists. Updating credentials and role...")
            existing_user.password_hash = pw_hash
            existing_user.role = role.strip().lower()
            session.commit()
            action = "UPDATED"
        else:
            new_user = User(
                username=username.strip(),
                password_hash=pw_hash,
                role=role.strip().lower(),
            )
            session.add(new_user)
            session.commit()
            action = "CREATED"

    print("=" * 60)
    print(f" [SUCCESS] Admin user '{username}' {action} successfully!")
    print(f" Role     : {role}")
    print(f" Database : {PROJECT_ROOT / 'data' / 'cyber_threat.db'}")
    if generated:
        print("-" * 60)
        print(f" [!] AUTO-GENERATED TEMPORARY PASSWORD : {resolved_password}")
        print(" [!] Please record this password safely or change it upon login.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed administrative user in SQLite database.")
    parser.add_argument("--username", default="admin", help="Admin username (default: admin)")
    parser.add_argument("--password", default=None, help="Admin password (reads from ADMIN_PASSWORD env if not set)")
    parser.add_argument("--role", default="admin", choices=["admin", "analyst"], help="User role (default: admin)")

    args = parser.parse_args()
    seed_admin_user(username=args.username, password=args.password, role=args.role)
