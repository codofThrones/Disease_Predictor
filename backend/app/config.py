"""
Diagnosis AI — Application Configuration
Loads environment variables and provides typed settings.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env from backend root
_backend_dir = Path(__file__).resolve().parent.parent
load_dotenv(_backend_dir / ".env")


class Settings(BaseSettings):
    SECRET_KEY: str = "diagnosis-ai-default-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    DATABASE_URL: str = "sqlite:///./diagnosisai.db"
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    ARTIFACTS_DIR: str = "saved_artifacts"
    REPORTS_DIR: str = "generated_reports"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def artifacts_path(self) -> Path:
        return _backend_dir / self.ARTIFACTS_DIR

    @property
    def reports_path(self) -> Path:
        path = _backend_dir / self.REPORTS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path
    @property
    def database_url(self) -> str:
        if self.DATABASE_URL.startswith("sqlite:///./"):
            filename = self.DATABASE_URL.replace("sqlite:///./", "")
            return f"sqlite:///{_backend_dir / filename}"
        return self.DATABASE_URL
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
