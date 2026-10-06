"""
api.py
======
Root entry point for the FastAPI REST API.
Delegates to Backend.api:app.
"""

from Backend.api import app

if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=True)
