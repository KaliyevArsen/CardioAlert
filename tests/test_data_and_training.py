import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cardioalert.data import load_data
from cardioalert.evaluation import summarize
from cardioalert.predict import load_model
from cardioalert.schema import FEATURES
from cardioalert.training import split_data, train


def write_fixture(tmp_path, frame):
    path = tmp_path / "processed.cleveland.data"
    frame[FEATURES + ["target"]].to_csv(path, header=False, index=False, na_rep="?")
    return path


def test_missing_target_is_not_treated_as_healthy(tmp_path, training_frame):
    training_frame.loc[0, "target"] = np.nan
    write_fixture(tmp_path, training_frame)
    with pytest.raises(ValueError, match="target"):
        load_data(tmp_path, ("cleveland",))


def test_source_codes_provenance_and_duplicate_handling(tmp_path, training_frame):
    training_frame.loc[0, "chol"] = 0
    frame = pd.concat([training_frame, training_frame.iloc[[1]]], ignore_index=True)
    path = write_fixture(tmp_path, frame)
    loaded, metadata = load_data(tmp_path, ("cleveland",))
    assert len(loaded) == 80
    assert metadata["duplicate_feature_rows_removed"] == 1
    assert np.isnan(loaded.loc[0, "chol"])
    assert metadata["nonpositive_measurements_set_missing"]["chol"] == 1
    assert loaded.loc[0, "record_id"] == "cleveland:1"
    assert metadata["file_sha256"]["cleveland"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_300_holdout_is_exact_deterministic_and_disjoint():
    frame = pd.DataFrame({"target": np.tile([0, 1], 459)})
    train_ids, test_ids = split_data(frame, 300)
    again_train, again_test = split_data(frame, 300)
    assert len(train_ids) == 618 and len(test_ids) == 300
    assert not set(train_ids) & set(test_ids)
    assert set(train_ids) | set(test_ids) == set(range(918))
    assert np.array_equal(train_ids, again_train) and np.array_equal(test_ids, again_test)


def test_metric_confusion_orientation_and_threshold():
    report = summarize([0, 0, 1, 1], [0.1, 0.8, 0.5, 0.9], bootstrap=20)
    assert report["confusion_matrix"] == [[1, 1], [0, 2]]
    assert report["sensitivity"] == 1 and report["specificity"] == 0.5
    assert report["accuracy"] == 0.75


def test_end_to_end_training_report_matches_saved_model(tmp_path, training_frame):
    write_fixture(tmp_path, training_frame)
    output = tmp_path / "run"
    report = train(tmp_path, output, search="smoke", jobs=1, cv_folds=3)
    assert report["status"] == "smoke_test_only"
    assert report["historical_pilot"] is False
    assert report["selection"]["candidate_count"] == 1
    assert not set(report["split"]["train_record_ids"]) & set(report["split"]["test_record_ids"])
    data, _ = load_data(tmp_path, ("cleveland",))
    data = data.set_index("record_id")
    predictions = pd.read_csv(output / "holdout_predictions.csv")
    pipeline = load_model(output / "model.joblib")["pipeline"]
    scores = pipeline.predict_proba(data.loc[predictions.record_id, FEATURES])[:, 1]
    np.testing.assert_allclose(scores, predictions.xgboost_score)
    expected_mean = data.loc[report["split"]["train_record_ids"], FEATURES].mean().to_numpy()
    np.testing.assert_allclose(pipeline["imputer"].statistics_, expected_mean)
    assert json.loads((output / "metrics.json").read_text())["xgboost"]["n"] == 16
    assert (output / "evaluation.svg").stat().st_size > 0
    with pytest.raises(ValueError, match="not empty"):
        train(tmp_path, output, search="smoke", jobs=1, cv_folds=3)


def test_original_files_are_preserved_byte_for_byte():
    root = Path(__file__).resolve().parents[1]
    files = {"legacy_bot.py": "9407f794309b922560a3a74548c0a0ea3048d4e3",
             "legacy_training.ipynb": "49983fcb3d91305bd5b58e90985d8a5331c2627d"}
    for name, expected_git_blob in files.items():
        raw = (root / "archive" / name).read_bytes()
        assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected_git_blob
