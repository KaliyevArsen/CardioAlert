from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cardioalert.schema import FEATURES, SCHEMA_VERSION
from cardioalert.training import make_pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def record():
    return json.loads((ROOT / "examples/fictional_complete.json").read_text())


@pytest.fixture
def training_frame():
    """Fictional software-test fixture; never used as evidence of medical performance."""
    rng = np.random.default_rng(123)
    count = 80
    frame = pd.DataFrame({
        "age": rng.integers(30, 78, count), "sex": rng.integers(0, 2, count),
        "cp": rng.integers(1, 5, count), "trestbps": rng.integers(100, 180, count),
        "chol": rng.integers(150, 350, count), "fbs": rng.integers(0, 2, count),
        "restecg": rng.integers(0, 3, count), "thalach": rng.integers(90, 190, count),
        "exang": rng.integers(0, 2, count), "oldpeak": rng.uniform(0, 4, count),
        "slope": rng.integers(1, 4, count), "ca": rng.integers(0, 4, count),
        "thal": rng.choice([3, 6, 7], count),
    })
    frame["target"] = (frame["age"] >= 53).astype(int)
    return frame


@pytest.fixture
def artifact(training_frame):
    model = make_pipeline(LogisticRegression(max_iter=2000))
    model.fit(training_frame[FEATURES], training_frame["target"])
    return {"schema_version": SCHEMA_VERSION, "features": FEATURES,
            "pipeline": model, "threshold": 0.5, "experiment": "fictional_test_fixture"}
