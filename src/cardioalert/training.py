"""Fixed protocols: training-only model selection, then one holdout evaluation."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data import SITES, load_data
from .evaluation import summarize
from .schema import FEATURES, SCHEMA_VERSION

SEED = 42
THRESHOLD = 0.5
PARAM_GRID = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [2, 3, 4],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__reg_alpha": [0, 0.1, 0.5],
    "clf__reg_lambda": [1, 5, 10],
}


def make_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("clf", classifier),
    ])


def split_data(frame: pd.DataFrame, test_size: int | float):
    train_idx, test_idx = train_test_split(
        np.arange(len(frame)), test_size=test_size, stratify=frame["target"], random_state=SEED,
    )
    if set(train_idx) & set(test_idx):
        raise AssertionError("Train/test overlap")
    return train_idx, test_idx


def save_figures(y, probabilities, baseline, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y, probabilities, name="XGBoost", ax=axes[0])
    RocCurveDisplay.from_predictions(y, baseline, name="Logistic baseline", ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.7)
    axes[0].set_title("Held-out discrimination")
    for name, scores in [("XGBoost", probabilities), ("Logistic baseline", baseline)]:
        observed, predicted = calibration_curve(y, scores, n_bins=5, strategy="quantile")
        axes[1].plot(predicted, observed, marker="o", label=name)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.7)
    axes[1].set(xlabel="Mean model score", ylabel="Observed positive fraction",
                title="Calibration diagnostic (no recalibration)")
    axes[1].legend()
    fig.tight_layout()
    figure_path = output / "evaluation.svg"
    fig.savefig(figure_path, metadata={"Date": None})
    figure_path.write_text(
        "\n".join(line.rstrip() for line in figure_path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def train(data_dir: Path, output: Path, protocol: str = "cleveland", search: str = "full",
          jobs: int = 2, cv_folds: int = 10) -> dict:
    if protocol not in {"cleveland", "multisite300"}:
        raise ValueError("Unknown protocol")
    if search not in {"full", "smoke"}:
        raise ValueError("Unknown search mode")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Use a new run directory.")
    sites = ("cleveland",) if protocol == "cleveland" else SITES
    frame, data_metadata = load_data(data_dir, sites)
    train_idx, test_idx = split_data(frame, 0.2 if protocol == "cleveland" else 300)
    X_train, X_test = frame.iloc[train_idx][FEATURES], frame.iloc[test_idx][FEATURES]
    y_train, y_test = frame.iloc[train_idx]["target"], frame.iloc[test_idx]["target"]
    if y_train.value_counts().min() < cv_folds:
        raise ValueError("Not enough examples of each class for requested CV folds")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=False)
    model = make_pipeline(XGBClassifier(eval_metric="logloss", random_state=SEED,
                                        n_jobs=1, tree_method="hist"))
    grid = PARAM_GRID if search == "full" else {
        "clf__n_estimators": [30], "clf__max_depth": [2], "clf__learning_rate": [0.1],
    }
    searcher = GridSearchCV(model, grid, scoring="roc_auc", cv=cv, n_jobs=jobs,
                            error_score="raise", return_train_score=False)
    print(f"{protocol}: train={len(train_idx)}, test={len(test_idx)}, search={search}", flush=True)
    searcher.fit(X_train, y_train)
    fitted = searcher.best_estimator_
    baseline = make_pipeline(LogisticRegression(max_iter=2000, random_state=SEED))
    baseline.fit(X_train, y_train)
    # Neither the holdout labels nor its scores are used for selection or tuning.
    scores = fitted.predict_proba(X_test)[:, 1]
    baseline_scores = baseline.predict_proba(X_test)[:, 1]
    run_metadata = {
        "report_schema": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": protocol,
        "status": "reconstructed_research_experiment" if search == "full" else "smoke_test_only",
        "historical_pilot": False,
        "dataset": data_metadata,
        "split": {
            "seed": SEED, "stratify": "binary target", "train_n": len(train_idx),
            "test_n": len(test_idx), "train_record_ids": frame.iloc[train_idx]["record_id"].tolist(),
            "test_record_ids": frame.iloc[test_idx]["record_id"].tolist(),
            "train_site_counts": frame.iloc[train_idx]["site"].value_counts().to_dict(),
            "test_site_counts": frame.iloc[test_idx]["site"].value_counts().to_dict(),
        },
        "selection": {"mode": search, "cv_folds": cv_folds, "cv_shuffle": False,
                      "candidate_count": len(searcher.cv_results_["params"]),
                      "scoring": "roc_auc", "best_cv_auc": float(searcher.best_score_),
                      "best_parameters": searcher.best_params_, "threshold_fixed_before_fit": THRESHOLD},
        "xgboost": summarize(y_test, scores, THRESHOLD),
        "logistic_baseline": summarize(y_test, baseline_scores, THRESHOLD),
        "environment": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib", "matplotlib"]
        }},
        "limitations": [
            "Retrospective public data; not prospective clinical validation.",
            "Model scores are not validated individual risk probabilities.",
            "Bootstrap intervals describe resampling of this holdout, not deployment uncertainty.",
            "Multisite random split is not external-site validation; site/missingness differences may confound scores.",
            "Historical identifiers are unavailable; exact duplicate feature vectors were removed before splitting.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    artifact = {"schema_version": SCHEMA_VERSION, "features": FEATURES,
                "threshold": THRESHOLD, "experiment": protocol, "pipeline": fitted}
    joblib.dump(artifact, output / "model.joblib")
    run_metadata["model_sha256"] = hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest()
    (output / "metrics.json").write_text(json.dumps(run_metadata, indent=2, allow_nan=False) + "\n")
    pd.DataFrame({"record_id": frame.iloc[test_idx]["record_id"].to_numpy(),
                  "target": y_test.to_numpy(), "xgboost_score": scores,
                  "logistic_score": baseline_scores}).to_csv(output / "holdout_predictions.csv", index=False)
    pd.DataFrame(searcher.cv_results_).to_csv(output / "cv_results.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "importance": fitted["clf"].feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    save_figures(y_test, scores, baseline_scores, output)
    print(json.dumps({"output": str(output), "xgboost": run_metadata["xgboost"],
                      "baseline": run_metadata["logistic_baseline"]}, indent=2), flush=True)
    return run_metadata
