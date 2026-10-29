"""Inference on validated records using a locally generated model artifact."""

from pathlib import Path

import joblib
import numpy as np

from .schema import DISCLAIMER, FEATURES, SCHEMA_VERSION, validate_record


def load_model(path: Path) -> dict:
    # joblib/pickle can execute code. Load only artifacts you trained yourself.
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run cardioalert train first.")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Incompatible model schema; retrain with this version of CardioAlert")
    if artifact.get("features") != FEATURES:
        raise ValueError("Model feature order does not match the input schema")
    return artifact


def predict_record(artifact: dict, record: dict) -> dict:
    if artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("features") != FEATURES:
        raise ValueError("Incompatible model schema")
    frame = validate_record(record)
    model = artifact["pipeline"]
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError("Expected binary model classes [0, 1]")
    score = float(model.predict_proba(frame)[0, classes.index(1)])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model returned an invalid score")
    threshold = float(artifact["threshold"])
    return {
        "model_score": score,
        "threshold": threshold,
        "classification_at_threshold": int(score >= threshold),
        "missing_features_imputed": [name for name in FEATURES if record[name] is None],
        "experiment": artifact["experiment"],
        "disclaimer": DISCLAIMER,
    }
