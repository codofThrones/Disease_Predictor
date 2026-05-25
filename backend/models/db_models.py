"""
Diagnosis AI — Database ORM Models
SQLAlchemy models for users, predictions, and reports.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Text, Boolean,
    DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    input_json = Column(JSON, nullable=False)
    model_used = Column(String(50), nullable=False, default="random_forest")
    predicted_disease = Column(String(200), nullable=False)
    confidence = Column(Float, nullable=False)
    top_predictions_json = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="predictions")
    reports = relationship("Report", back_populates="prediction", cascade="all, delete-orphan")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, index=True)
    file_path = Column(String(500), nullable=True)
    status = Column(String(20), default="pending")  # pending, generating, completed, failed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    prediction = relationship("Prediction", back_populates="reports")
