import copy
import json

import joblib
import numpy as np
import pytest

from cardioalert.predict import load_model, predict_record
from cardioalert.schema import FEATURES, validate_record


@pytest.mark.parametrize("name,values", [("cp", [1, 2, 3, 4]), ("slope", [1, 2, 3]), ("thal", [3, 6, 7])])
def test_original_uci_codes_are_preserved(record, name, values):
    for value in values:
        record[name] = value
        assert validate_record(record).iloc[0][name] == value


@pytest.mark.parametrize("name,value", [
    ("cp", 0), ("cp", 5), ("thal", 0), ("thal", 1), ("thal", 2), ("slope", 0),
    ("age", 17), ("age", 54.5), ("ca", 4), ("chol", 0), ("chol", -1),
    ("oldpeak", float("nan")), ("oldpeak", float("inf")), ("sex", True), ("age", "54"),
])
def test_bad_inputs_fail_before_inference(record, name, value):
    record[name] = value
    with pytest.raises(ValueError):
        validate_record(record)


def test_missing_key_is_not_silently_imputed(record):
    del record["thal"]
    with pytest.raises(ValueError, match="missing"):
        validate_record(record)


def test_target_or_extra_fields_cannot_enter_prediction(record):
    record["target"] = 1
    with pytest.raises(ValueError, match="extra"):
        validate_record(record)


def test_null_is_explicit_and_limited(record, artifact):
    for name in ["ca", "thal", "chol"]:
        record[name] = None
    result = predict_record(artifact, record)
    assert set(result["missing_features_imputed"]) == {"ca", "thal", "chol"}
    record["oldpeak"] = None
    with pytest.raises(ValueError, match="At most three"):
        predict_record(artifact, record)


def test_feature_order_independent_of_json_order(record, artifact):
    reverse = dict(reversed(list(record.items())))
    assert list(validate_record(reverse).columns) == FEATURES
    assert predict_record(artifact, record) == predict_record(artifact, reverse)


def test_saved_model_roundtrip_and_schema_rejection(tmp_path, record, artifact):
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    assert predict_record(load_model(path), record) == predict_record(artifact, record)
    artifact["schema_version"] = "different-coding"
    joblib.dump(artifact, path)
    with pytest.raises(ValueError, match="schema"):
        load_model(path)


def test_demo_output_is_json_finite_and_explicitly_non_diagnostic(record, artifact):
    result = predict_record(artifact, record)
    json.dumps(result, allow_nan=False)
    assert 0 <= result["model_score"] <= 1
    assert "not a diagnosis" in result["disclaimer"]
    assert "risk_probability" not in result


def test_nonfinite_prediction_rejected(record, artifact):
    class BrokenModel:
        classes_ = np.array([0, 1])
        def predict_proba(self, frame):
            return np.array([[float("nan"), float("nan")]])
    artifact = copy.copy(artifact)
    artifact["pipeline"] = BrokenModel()
    with pytest.raises(ValueError, match="invalid score"):
        predict_record(artifact, record)
