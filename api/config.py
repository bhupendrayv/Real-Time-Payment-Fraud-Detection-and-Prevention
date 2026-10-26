"""API configuration and environment variable parsing."""
from __future__ import annotations

import os
from pathlib import Path

# Paths
API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "fraudshield.db"

# Database URL: default to local SQLite with thread check disabled for SQLite
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")

# CORS Configuration
CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]

# Host and Port
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# Model Identification
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "xgb-v1")

# Rate Limiting (per client IP)
RATE_LIMIT_PER_MINUTE: str = os.getenv("RATE_LIMIT_PER_MINUTE", "120/minute")
