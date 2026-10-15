import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.churn_baseline import (
    load_dataset,
    split_dataset,
)
from scripts.churn_random_search import (
    build_cv,
    build_search,
)


def test_cv_uses_only_training_customers():
    X, y, _ = load_dataset()
    X_train, X_test, y_train, _ = split_dataset(X, y)

    assert X_train.index.intersection(X_test.index).empty

    folds = list(build_cv().split(X_train, y_train))
    assert len(folds) == 5

    for training_idx, validation_idx in folds:
        assert not set(training_idx).intersection(
            validation_idx
        )

        training_ids = set(
            X_train.iloc[training_idx].index
        )
        validation_ids = set(
            X_train.iloc[validation_idx].index
        )

        assert not training_ids.intersection(validation_ids)
        assert not validation_ids.intersection(set(X_test.index))


def test_search_configuration():
    X, y, _ = load_dataset()
    X_train, _, _, _ = split_dataset(X, y)

    search = build_search(X_train)

    assert search.n_iter == 16
    assert search.refit is True
    assert search.n_jobs == 1
    assert search.cv.n_splits == 5
    assert "preprocess" in search.estimator.named_steps


def test_saved_search_results():
    report = json.loads(
        (ROOT / "reports/random_search_summary.json").read_text()
    )

    results = pd.read_csv(
        ROOT / "reports/random_search_results.csv"
    )

    model = joblib.load(
        ROOT / "models/churn_random_search.joblib"
    )

    assert len(results) == 16
    assert report["holdout_used_in_search"] is False
    assert report["holdout_evaluations_this_stage"] == 0
    assert np.isfinite(report["random_cv_f2_mean"])
    assert "classifier" in model.named_steps
