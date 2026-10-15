import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.churn_baseline import (
    load_dataset,
    split_dataset,
)
from scripts.churn_grid_search import build_search


def test_grid_contiene_ganador_aleatorio():
    X, y, _ = load_dataset()
    X_train, _, _, _ = split_dataset(X, y)

    search = build_search(X_train)
    summary = json.loads(
        (ROOT / "reports/random_search_summary.json").read_text()
    )

    candidates = list(ParameterGrid(search.param_grid))

    assert len(candidates) == 24
    assert summary["best_parameters"] in candidates
    assert search.cv.n_splits == 5


def test_grid_no_usa_holdout():
    X, y, _ = load_dataset()
    X_train, X_test, y_train, _ = split_dataset(X, y)

    search = build_search(X_train)

    assert X_train.index.intersection(X_test.index).empty

    for tr, va in search.cv.split(X_train, y_train):
        assert not set(tr).intersection(set(va))
        assert set(X_train.iloc[va].index).isdisjoint(
            set(X_test.index)
        )


def test_resultados_grid_guardados():
    summary = json.loads(
        (ROOT / "reports/grid_search_summary.json").read_text()
    )

    results = pd.read_csv(
        ROOT / "reports/grid_search_results.csv"
    )

    model = joblib.load(
        ROOT / "models/churn_grid_search.joblib"
    )

    assert len(results) == 24
    assert summary["holdout_used_in_search"] is False
    assert summary["holdout_evaluations_this_stage"] == 0

    assert np.isclose(
        summary["randomized_winner_reproduced_f2"],
        summary["randomized_cv_f2_mean"],
        atol=0.000001,
    )

    assert (
        summary["grid_best_cv_f2_mean"]
        >= summary["randomized_cv_f2_mean"] - 0.000001
    )

    assert "classifier" in model.named_steps
