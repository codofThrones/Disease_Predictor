"""
Diagnosis AI — ML Inference Engine
Runs prediction using loaded models and returns structured results.
"""

import logging
import numpy as np
from typing import Any

from ml.loader import get_model, get_label_encoder, get_disease_metadata
from ml.preprocessing import build_feature_vector

logger = logging.getLogger("diagnosis_ai.inference")


def predict(
    age: int,
    gender: int,
    symptoms: dict[str, dict[str, Any]],
    model_name: str = "random_forest",
    top_k: int = 5,
) -> dict:
    """
    Run disease prediction.
    
    Args:
        age: Patient age
        gender: 0=Female, 1=Male  
        symptoms: Structured symptom input
        model_name: Which model to use
        top_k: Number of top predictions to return
    
    Returns:
        Dict with predicted_disease, confidence, top_predictions, recommendation
    """
    # Build feature vector
    feature_vector = build_feature_vector(age, gender, symptoms)

    # Load model and encoders
    model = get_model(model_name)
    label_encoder = get_label_encoder()
    disease_metadata = get_disease_metadata()

    # Get prediction
    predicted_class = model.predict(feature_vector)[0]
    predicted_disease = label_encoder.inverse_transform([predicted_class])[0]

    # Get probabilities (if model supports it)
    top_predictions = []
    confidence = 0.0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_vector)[0]
        
        # Get top-k indices sorted by probability (descending)
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        for idx in top_indices:
            disease_name = label_encoder.inverse_transform([idx])[0]
            prob = float(probabilities[idx])
            
            # Get metadata for this disease
            meta = disease_metadata.get(disease_name, {})
            
            top_predictions.append({
                "disease": disease_name,
                "probability": round(prob, 4),
                "metadata": {
                    "severity": meta.get("severity", "unknown"),
                    "specialist": meta.get("specialist", "General Medicine"),
                    "emergency": meta.get("emergency", False),
                    "recommendations": meta.get("recommendations", []),
                }
            })
        
        # Confidence is the probability of the top prediction
        confidence = float(probabilities[predicted_class])
    else:
        # For models without predict_proba, confidence = 1.0 for the prediction
        confidence = 1.0
        meta = disease_metadata.get(predicted_disease, {})
        top_predictions.append({
            "disease": predicted_disease,
            "probability": 1.0,
            "metadata": {
                "severity": meta.get("severity", "unknown"),
                "specialist": meta.get("specialist", "General Medicine"),
                "emergency": meta.get("emergency", False),
                "recommendations": meta.get("recommendations", []),
            }
        })

    # Get primary disease metadata
    primary_meta = disease_metadata.get(predicted_disease, {})
    is_emergency = primary_meta.get("emergency", False)
    severity = primary_meta.get("severity", "moderate")

    # Generate recommendation
    if is_emergency or severity == "severe":
        recommendation = "⚠️ Seek immediate medical attention. This condition may require urgent care."
    elif confidence >= 0.7:
        recommendation = "Consult a physician for proper diagnosis and treatment plan."
    else:
        recommendation = (
            "The confidence level is moderate. Please consult a healthcare professional "
            "for a thorough evaluation if symptoms persist."
        )

    result = {
        "predicted_disease": predicted_disease,
        "confidence": round(confidence, 4),
        "top_predictions": top_predictions,
        "model_used": model_name,
        "metadata": {
            "severity": primary_meta.get("severity", "unknown"),
            "specialist": primary_meta.get("specialist", "General Medicine"),
            "emergency": primary_meta.get("emergency", False),
            "recommendations": primary_meta.get("recommendations", []),
        },
        "recommendation": recommendation,
    }

    logger.info(
        f"Prediction complete: {predicted_disease} "
        f"(confidence={confidence:.4f}, model={model_name})"
    )

    return result
