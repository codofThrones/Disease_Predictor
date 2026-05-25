"""
Diagnosis AI — ML Model Loader
Lazy-loads trained models and artifacts from saved_artifacts/.
Models are cached in memory after first load to avoid repeated disk I/O.
"""

import json
import pickle
import joblib
import logging
from pathlib import Path
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger("diagnosis_ai.loader")

# ── In-memory cache ──────────────────────────────────────

_models_cache: dict[str, Any] = {}
_label_encoder = None
_string_encoders = None
_feature_columns = None
_symptom_schema = None
_disease_metadata = None
_feature_importances = None

# Supported model names → file mapping
MODEL_FILES = {
    "random_forest": "random_forest.pkl",
    "xgboost": "xgboost.pkl",
    "decision_tree": "decision_tree.pkl",
    "knn": "knn.pkl",
    "logistic_regression": "logistic_regression.pkl",
    "naive_bayes": "naive_bayes.pkl",
}


def _artifacts_dir() -> Path:
    """Get the absolute path to saved_artifacts/."""
    return settings.artifacts_path


def _load_pickle(filename: str) -> Any:
    """Load a pickle file from artifacts directory."""
    filepath = _artifacts_dir() / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    return joblib.load(filepath)


def _load_json(filename: str) -> dict:
    """Load a JSON file from artifacts directory."""
    filepath = _artifacts_dir() / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Public Loaders ────────────────────────────────────────

def get_model(model_name: str) -> Any:
    """
    Lazy-load and cache a trained ML model.
    Returns the sklearn/xgboost model object.
    """
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_FILES.keys())}")

    if model_name not in _models_cache:
        logger.info(f"Loading model: {model_name}...")
        _models_cache[model_name] = _load_pickle(MODEL_FILES[model_name])
        logger.info(f"Model '{model_name}' loaded successfully.")

    return _models_cache[model_name]


def get_label_encoder():
    """Load the disease label encoder."""
    global _label_encoder
    if _label_encoder is None:
        logger.info("Loading label encoder...")
        _label_encoder = _load_pickle("label_encoder.pkl")
    return _label_encoder


def get_string_encoders() -> dict:
    """Load string encoders for categorical symptom fields."""
    global _string_encoders
    if _string_encoders is None:
        logger.info("Loading string encoders...")
        _string_encoders = _load_pickle("string_encoders.pkl")
    return _string_encoders


def get_feature_columns() -> list[str]:
    """Load the ordered list of feature column names used during training."""
    global _feature_columns
    if _feature_columns is None:
        logger.info("Loading feature columns...")
        _feature_columns = _load_pickle("feature_columns.pkl")
    return _feature_columns


def get_symptom_schema() -> dict:
    """Load symptom schema defining which fields each symptom has."""
    global _symptom_schema
    if _symptom_schema is None:
        logger.info("Loading symptom schema...")
        _symptom_schema = _load_json("symptom_schema.json")
    return _symptom_schema


def get_disease_metadata() -> dict:
    """Load disease metadata (severity, specialist, recommendations)."""
    global _disease_metadata
    if _disease_metadata is None:
        logger.info("Loading disease metadata...")
        _disease_metadata = _load_json("disease_metadata.json")
    return _disease_metadata


def get_feature_importances() -> dict:
    """Load feature importance scores per model."""
    global _feature_importances
    if _feature_importances is None:
        logger.info("Loading feature importances...")
        _feature_importances = _load_json("feature_importances.json")
    return _feature_importances


def get_available_models() -> list[str]:
    """Return list of model names that have files on disk."""
    available = []
    for name, filename in MODEL_FILES.items():
        if (_artifacts_dir() / filename).exists():
            available.append(name)
    return available


def get_disease_classes() -> list[str]:
    """Return all disease class names from the label encoder."""
    le = get_label_encoder()
    return list(le.classes_)


def preload_essentials():
    """
    Preload lightweight artifacts at startup.
    Does NOT load large model files — those are lazy-loaded on first prediction.
    """
    logger.info("Preloading essential artifacts...")
    get_feature_columns()
    get_label_encoder()
    get_string_encoders()
    get_symptom_schema()
    get_disease_metadata()
    logger.info(
        f"Essentials loaded: {len(get_feature_columns())} features, "
        f"{len(get_disease_classes())} diseases, "
        f"{len(get_symptom_schema())} symptoms"
    )
