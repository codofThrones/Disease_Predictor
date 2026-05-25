"""
Diagnosis AI — Prediction Routes
Symptom submission, disease prediction, and history endpoints.
"""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from auth.auth import get_current_user, get_optional_user
from models.db_models import User, Prediction
from schemas.schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionHistoryItem,
    TopPrediction,
    DiseaseMetadata,
)
from ml.inference import predict as ml_predict
from ml.loader import get_symptom_schema, get_available_models, get_disease_classes

logger = logging.getLogger("diagnosis_ai.routes.predict")

router = APIRouter(prefix="/api", tags=["Predictions"])


@router.get("/schema")
def get_schema():
    """Return the symptom schema for the frontend to build dynamic forms."""
    return {
        "symptoms": get_symptom_schema(),
        "models": get_available_models(),
        "diseases": get_disease_classes(),
    }


@router.post("/predict", response_model=PredictionResponse)
def create_prediction(
    payload: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    """
    Accept structured symptom input, run ML inference, store result.
    Works for both authenticated and unauthenticated users.
    Authenticated users get predictions saved to their history.
    """
    # Convert Pydantic symptom models to dicts
    symptoms_dict = {
        name: {
            "present": s.present,
            "severity": s.severity,
            "detail": s.detail,
            "location": s.location,
        }
        for name, s in payload.symptoms.items()
    }

    # Log the prediction input
    username = current_user.username if current_user else "anonymous"
    logger.info(
        f"Prediction request: user={username}, "
        f"model={payload.model_name}, "
        f"active_symptoms={sum(1 for s in symptoms_dict.values() if s['present'] == 1)}"
    )

    # Run inference
    try:
        result = ml_predict(
            age=payload.age,
            gender=payload.gender,
            symptoms=symptoms_dict,
            model_name=payload.model_name,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )

    # Build response common fields
    top_predictions = [
        TopPrediction(
            disease=tp["disease"],
            probability=tp["probability"],
            metadata=DiseaseMetadata(**tp["metadata"]) if tp.get("metadata") else None,
        )
        for tp in result["top_predictions"]
    ]
    primary_meta = result.get("metadata")

    # Store prediction in database only if user is authenticated
    if current_user:
        prediction = Prediction(
            user_id=current_user.id,
            input_json={
                "age": payload.age,
                "gender": payload.gender,
                "symptoms": symptoms_dict,
                "model_name": payload.model_name,
            },
            model_used=payload.model_name,
            predicted_disease=result["predicted_disease"],
            confidence=result["confidence"],
            top_predictions_json=result["top_predictions"],
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        return PredictionResponse(
            id=prediction.id,
            predicted_disease=result["predicted_disease"],
            confidence=result["confidence"],
            top_predictions=top_predictions,
            model_used=result["model_used"],
            metadata=DiseaseMetadata(**primary_meta) if primary_meta else None,
            recommendation=result["recommendation"],
            timestamp=prediction.timestamp,
        )
    else:
        # Anonymous prediction — not persisted
        from datetime import datetime, timezone
        return PredictionResponse(
            id=0,
            predicted_disease=result["predicted_disease"],
            confidence=result["confidence"],
            top_predictions=top_predictions,
            model_used=result["model_used"],
            metadata=DiseaseMetadata(**primary_meta) if primary_meta else None,
            recommendation=result["recommendation"],
            timestamp=datetime.now(timezone.utc),
        )


@router.get("/predictions", response_model=list[PredictionHistoryItem])
def get_predictions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the current user's prediction history."""
    predictions = (
        db.query(Prediction)
        .filter(Prediction.user_id == current_user.id)
        .order_by(Prediction.timestamp.desc())
        .limit(50)
        .all()
    )
    return [PredictionHistoryItem.model_validate(p) for p in predictions]


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
def get_prediction_detail(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single prediction's full details."""
    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.user_id == current_user.id)
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found",
        )

    # Reconstruct top predictions from stored JSON
    top_predictions = []
    if prediction.top_predictions_json:
        for tp in prediction.top_predictions_json:
            top_predictions.append(
                TopPrediction(
                    disease=tp["disease"],
                    probability=tp["probability"],
                    metadata=DiseaseMetadata(**tp["metadata"]) if tp.get("metadata") else None,
                )
            )

    # Determine recommendation from metadata
    meta_raw = top_predictions[0].metadata if top_predictions else None
    if meta_raw and (meta_raw.emergency or meta_raw.severity == "severe"):
        recommendation = "⚠️ Seek immediate medical attention."
    elif prediction.confidence >= 0.7:
        recommendation = "Consult a physician for proper diagnosis and treatment plan."
    else:
        recommendation = "Please consult a healthcare professional if symptoms persist."

    return PredictionResponse(
        id=prediction.id,
        predicted_disease=prediction.predicted_disease,
        confidence=prediction.confidence,
        top_predictions=top_predictions,
        model_used=prediction.model_used,
        metadata=meta_raw,
        recommendation=recommendation,
        timestamp=prediction.timestamp,
    )
