"""Repeated session-aware cross-validation of the linear probe."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.splits import random_group_folds


def linear_probe():
    """The linear probe used throughout the project (notebook 04)."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))


def per_class_f1_cv(X, y, groups, n_repeats=10, n_folds=5):
    """Per-class F1 of the linear probe, one row per repeat of session-aware k-fold CV.

    The predictions of the folds are pooled (each clip predicted once per repeat) before scoring.
    """
    y = np.asarray(y)
    classes = sorted(set(y))
    rows = []
    for repeat in range(n_repeats):
        pred = np.empty_like(y)
        for train_idx, test_idx in random_group_folds(y, groups, n_folds, seed=repeat):
            pred[test_idx] = linear_probe().fit(X[train_idx], y[train_idx]).predict(X[test_idx])
        rows.append(f1_score(y, pred, labels=classes, average=None))
    return pd.DataFrame(rows, columns=classes)