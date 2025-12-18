from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env if it exists (we'll add it in Block B)
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Centralized configuration.
    We read env vars once so the rest of the codebase doesn't call os.getenv everywhere.
    """
    database_url: str = os.getenv("DATABASE_URL", "")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")


settings = Settings()