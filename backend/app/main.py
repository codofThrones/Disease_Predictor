"""
Diagnosis AI — FastAPI Main Application
Entry point for the backend server.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from ml.loader import preload_essentials, get_available_models, get_disease_classes, get_symptom_schema
from schemas.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("diagnosis_ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # ── Startup ──
    logger.info("=" * 60)
    logger.info("  Diagnosis AI — Starting up...")
    logger.info("=" * 60)

    # Initialize database tables
    init_db()
    logger.info("Database initialized.")

    # Preload lightweight ML artifacts
    preload_essentials()
    logger.info("ML artifacts preloaded.")

    logger.info("=" * 60)
    logger.info("  Diagnosis AI — Ready to serve!")
    logger.info("=" * 60)

    yield

    # ── Shutdown ──
    logger.info("Diagnosis AI — Shutting down...")


# ── Create App ──
app = FastAPI(
    title="Diagnosis AI",
    description="AI-Powered Diagnostic Support System",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ──
from routes.auth import router as auth_router
from routes.predict import router as predict_router
from routes.report import router as report_router

app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(report_router)


# ── Health Check ──
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check endpoint."""
    return HealthResponse(
        status="healthy",
        app_name="Diagnosis AI",
        version="1.0.0",
        models_available=get_available_models(),
        total_symptoms=len(get_symptom_schema()),
        total_diseases=len(get_disease_classes()),
    )


@app.get("/", tags=["System"])
def root():
    """Root endpoint."""
    return {
        "app": "Diagnosis AI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
