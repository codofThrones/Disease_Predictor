"""
Diagnosis AI — Feature Preprocessing
Converts structured symptom input into an ML-ready feature vector.

CRITICAL: Must preserve exact training-time feature order and encoding logic.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any

from ml.loader import get_feature_columns, get_string_encoders, get_symptom_schema

logger = logging.getLogger("diagnosis_ai.preprocessing")


def _safe_encode(encoder, value: str) -> int:
    """
    Safely encode a string value using a LabelEncoder.
    Returns 0 if the value is unknown (not seen during training).
    """
    if value is None:
        return 0
    try:
        return int(encoder.transform([value])[0])
    except (ValueError, KeyError):
        logger.warning(f"Unknown encoder value '{value}', defaulting to 0")
        return 0


def build_feature_vector(
    age: int,
    gender: int,
    symptoms: dict[str, dict[str, Any]],
) -> np.ndarray:
    """
    Build a feature vector from structured symptom input.
    
    Args:
        age: Patient age (integer)
        gender: 0=Female, 1=Male
        symptoms: Dict of symptom_name -> {present, severity, detail, location}
    
    Returns:
        1D numpy array aligned exactly with feature_columns order.
    """
    feature_columns = get_feature_columns()
    string_encoders = get_string_encoders()
    symptom_schema = get_symptom_schema()

    # Initialize all features to 0
    feature_map: dict[str, float] = {col: 0.0 for col in feature_columns}

    # ── Set demographic features ──
    if "GENDER" in feature_map:
        feature_map["GENDER"] = float(gender)
    if "AGE_END" in feature_map:
        feature_map["AGE_END"] = float(age)

    # ── Set symptom features ──
    for symptom_name, symptom_data in symptoms.items():
        # Skip symptoms not in schema
        if symptom_name not in symptom_schema:
            logger.warning(f"Symptom '{symptom_name}' not in schema, skipping.")
            continue

        schema = symptom_schema[symptom_name]

        # Present flag
        present_col = f"{symptom_name}_present"
        if present_col in feature_map and schema.get("has_present", False):
            feature_map[present_col] = float(symptom_data.get("present", 0))

        # Severity (0-3)
        severity_col = f"{symptom_name}_severity"
        if severity_col in feature_map and schema.get("has_severity", False):
            feature_map[severity_col] = float(symptom_data.get("severity", 0))

        # Detail (categorical string → encoded int)
        detail_col = f"{symptom_name}_detail"
        if detail_col in feature_map and schema.get("has_detail", False):
            detail_value = symptom_data.get("detail", None)
            if detail_value and detail_col in string_encoders:
                feature_map[detail_col] = float(
                    _safe_encode(string_encoders[detail_col], detail_value)
                )

        # Location (categorical string → encoded int)
        location_col = f"{symptom_name}_location"
        if location_col in feature_map and schema.get("has_location", False):
            location_value = symptom_data.get("location", None)
            if location_value and location_col in string_encoders:
                feature_map[location_col] = float(
                    _safe_encode(string_encoders[location_col], location_value)
                )

    # ── Build final vector in exact column order ──
    feature_vector = pd.DataFrame(
    [[feature_map.get(col, 0.0) for col in feature_columns]],
    columns=feature_columns
)
    logger.info(
        f"Feature vector built: shape={feature_vector.shape}, "
        f"non-zero={np.count_nonzero(feature_vector)}"
    )
    logger.info(f"Non-zero count: {np.count_nonzero(feature_vector)}")
    logger.info(f"Active symptoms received: {list(symptoms.keys())}")
    logger.info(f"Sample feature map: { {k:v for k,v in feature_map.items() if v != 0.0} }")

    return feature_vector


def get_active_symptoms(symptoms: dict[str, dict[str, Any]]) -> list[str]:
    """Return list of symptom names that are marked as present."""
    return [
        name for name, data in symptoms.items()
        if data.get("present", 0) == 1
    ]
