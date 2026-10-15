"""RandomizedSearchCV de StreamLoop: solo datos de train."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.churn_baseline import (
    SEED,
    build_pipeline,
    load_dataset,
    split_dataset,
)

SCORER = make_scorer(
    fbeta_score,
    beta=2,
    zero_division=0,
)


def build_cv():
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )


def build_search(X_train):
    pipeline = build_pipeline(X_train)

    # Todas las opciones pertenecen al clasificador.
    # El preprocesamiento sigue dentro del Pipeline.
    parameters = {
        "classifier__n_estimators": [
            80, 120, 180
        ],
        "classifier__max_depth": [
            None, 6, 10, 16
        ],
        "classifier__min_samples_leaf": [
            1, 2, 4, 8
        ],
        "classifier__min_samples_split": [
            2, 5, 10
        ],
        "classifier__class_weight": [
            None, "balanced", "balanced_subsample"
        ],
        "classifier__max_features": [
            "sqrt", 0.75
        ],
    }

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=16,
        scoring=SCORER,
        cv=build_cv(),
        refit=True,
        random_state=SEED,
        n_jobs=1,
        return_train_score=True,
        error_score="raise",
        verbose=1,
    )


def main():
    X, y, missing = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    assert len(X_train) == 5634
    assert len(X_test) == 1409
    assert X_train.index.intersection(X_test.index).empty

    print("=== BASELINE: VALIDACION SOLO EN TRAIN ===")

    baseline = build_pipeline(X_train)

    baseline_scores = cross_val_score(
        clone(baseline),
        X_train,
        y_train,
        scoring=SCORER,
        cv=build_cv(),
        n_jobs=1,
        error_score="raise",
    )

    print(
        "Baseline CV F2:",
        f"{baseline_scores.mean():.6f} +/- "
        f"{baseline_scores.std(ddof=1):.6f}"
    )

    print("=== RANDOMIZEDSEARCHCV ===")

    search = build_search(X_train)
    search.fit(X_train, y_train)

    # X_test e y_test permanecen sin utilizar.
    results = pd.DataFrame(search.cv_results_)

    fields = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "params",
    ]

    ranked = results[fields].sort_values(
        ["rank_test_score", "std_test_score"]
    ).reset_index(drop=True)

    assert len(ranked) == 16
    assert np.isfinite(
        results["mean_test_score"]
    ).all()

    best_idx = search.best_index_
    best_std = float(
        results.iloc[best_idx]["std_test_score"]
    )

    summary = {
        "algorithm": "RandomForestClassifier",
        "method": "RandomizedSearchCV",
        "metric": "F2",
        "reason": (
            "El costo de no detectar churn justifica "
            "priorizar recall sin ignorar precision."
        ),
        "train_rows": int(len(X_train)),
        "holdout_test_rows": int(len(X_test)),
        "cv_folds": 5,
        "candidate_count": 16,
        "baseline_cv_f2_mean": float(
            baseline_scores.mean()
        ),
        "baseline_cv_f2_std": float(
            baseline_scores.std(ddof=1)
        ),
        "random_cv_f2_mean": float(
            search.best_score_
        ),
        "random_cv_f2_std": best_std,
        "best_parameters": search.best_params_,
        "missing_total_charges": missing,
        "holdout_used_in_search": False,
        "holdout_evaluations_this_stage": 0,
    }

    reports = ROOT / "reports"
    models = ROOT / "models"

    reports.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)

    ranked.to_csv(
        reports / "random_search_results.csv",
        index=False,
    )

    (reports / "random_search_summary.json").write_text(
        json.dumps(
            summary, indent=2, ensure_ascii=False
        ) + "\n",
        encoding="utf-8",
    )

    joblib.dump(
        search.best_estimator_,
        models / "churn_random_search.joblib",
    )

    print("\n=== RESULTADOS DE LA BUSQUEDA ===")
    print(
        "Baseline CV F2:",
        f"{summary['baseline_cv_f2_mean']:.6f}"
    )
    print(
        "Mejor CV F2:",
        f"{summary['random_cv_f2_mean']:.6f}"
    )
    print(
        "Desv. CV:",
        f"{summary['random_cv_f2_std']:.6f}"
    )
    print("Mejores parametros:")

    for key, value in search.best_params_.items():
        print(f"  {key}: {value}")

    print("\nPASS: 16 candidatos evaluados")
    print("PASS: 5 folds estratificados")
    print("PASS: test externo sin utilizar")
    print("PASS: resultados y modelo guardados")


if __name__ == "__main__":
    main()
