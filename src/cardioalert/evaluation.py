"""Metrics and deterministic uncertainty intervals, without threshold tuning."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)


def summarize(y_true, scores, threshold: float = 0.5, bootstrap: int = 1000) -> dict:
    y = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    rng = np.random.default_rng(20261029)
    aucs = []
    for _ in range(bootstrap):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) == 2:
            aucs.append(roc_auc_score(y[index], scores[index]))
    return {
        "n": int(len(y)), "positive_count": int(y.sum()), "threshold": threshold,
        "roc_auc": float(roc_auc_score(y, scores)),
        "roc_auc_bootstrap_95_ci": np.quantile(aucs, [0.025, 0.975]).tolist() if aucs else None,
        "bootstrap_resamples": bootstrap,
        "accuracy": float(accuracy_score(y, predictions)),
        "sensitivity": float(recall_score(y, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
