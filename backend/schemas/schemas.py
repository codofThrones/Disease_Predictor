"""
Diagnosis AI — Pydantic Schemas
Request/response models for all API endpoints.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime


# ── Auth Schemas ──────────────────────────────────────────

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ── Symptom Input Schemas ─────────────────────────────────

class SymptomInput(BaseModel):
    """Single symptom entry from the frontend."""
    present: int = Field(0, ge=0, le=1, description="0=absent, 1=present")
    severity: int = Field(0, ge=0, le=3, description="0-3 severity scale")
    detail: Optional[str] = Field(None, description="Categorical detail string")
    location: Optional[str] = Field(None, description="Categorical location string")


class PredictionRequest(BaseModel):
    """Full prediction request from frontend."""
    age: int = Field(..., ge=0, le=120)
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    symptoms: dict[str, SymptomInput] = Field(
        ..., description="Map of symptom_name -> SymptomInput"
    )
    model_name: str = Field(
        "random_forest",
        description="Model to use: random_forest, xgboost, decision_tree, knn, logistic_regression, naive_bayes"
    )


# ── Prediction Response Schemas ───────────────────────────

class DiseaseMetadata(BaseModel):
    severity: str = "unknown"
    specialist: str = "General Medicine"
    emergency: bool = False
    recommendations: list[str] = []


class TopPrediction(BaseModel):
    disease: str
    probability: float
    metadata: Optional[DiseaseMetadata] = None


class PredictionResponse(BaseModel):
    id: int
    predicted_disease: str
    confidence: float
    top_predictions: list[TopPrediction]
    model_used: str
    metadata: Optional[DiseaseMetadata] = None
    recommendation: str
    timestamp: datetime

    class Config:
        from_attributes = True


class PredictionHistoryItem(BaseModel):
    id: int
    predicted_disease: str
    confidence: float
    model_used: str
    timestamp: datetime

    class Config:
        from_attributes = True


# ── Report Schemas ────────────────────────────────────────

class ReportRequest(BaseModel):
    prediction_id: int


class ReportResponse(BaseModel):
    id: int
    prediction_id: int
    status: str
    file_path: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ── Schema / Health Schemas ───────────────────────────────

class SymptomSchemaField(BaseModel):
    has_present: bool
    has_severity: bool
    has_detail: bool
    has_location: bool


class HealthResponse(BaseModel):
    status: str = "healthy"
    app_name: str = "Diagnosis AI"
    version: str = "1.0.0"
    models_available: list[str] = []
    total_symptoms: int = 0
    total_diseases: int = 0
