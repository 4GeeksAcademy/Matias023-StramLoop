"""GridSearchCV StreamLoop: optimizacion solo en train."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.churn_baseline import (
    build_pipeline,
    load_dataset,
    split_dataset,
)
from scripts.churn_random_search import (
    SCORER,
    build_cv,
)


def build_search(X_train):
    parameters = {
        "classifier__n_estimators": [80],
        "classifier__max_depth": [5, 6, 8],
        "classifier__min_samples_leaf": [6, 8],
        "classifier__min_samples_split": [8, 10],
        "classifier__max_features": ["sqrt"],
        "classifier__class_weight": [
            "balanced",
            "balanced_subsample",
        ],
    }

    assert len(ParameterGrid(parameters)) == 24

    return GridSearchCV(
        estimator=build_pipeline(X_train),
        param_grid=parameters,
        scoring=SCORER,
        cv=build_cv(),
        n_jobs=1,
        refit=True,
        return_train_score=True,
        error_score="raise",
        verbose=1,
    )


def main():
    X, y, _ = load_dataset()
    X_train, X_test, y_train, _ = split_dataset(X, y)

    assert len(X_train) == 5634
    assert len(X_test) == 1409
    assert X_train.index.intersection(X_test.index).empty

    random_summary = json.loads(
        (ROOT / "reports/random_search_summary.json").read_text()
    )

    print("=== GRIDSEARCHCV: SOLO TRAIN ===")

    search = build_search(X_train)
    search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_)

    assert len(results) == 24
    assert np.isfinite(results["mean_test_score"]).all()

    # Confirmar que la mejor configuracion aleatoria
    # tambien fue evaluada en la busqueda exhaustiva.
    winner = random_summary["best_parameters"]

    matches = [
        i for i, params in enumerate(
            search.cv_results_["params"]
        )
        if params == winner
    ]

    assert len(matches) == 1

    reproduced_f2 = float(
        results.iloc[matches[0]]["mean_test_score"]
    )

    assert np.isclose(
        reproduced_f2,
        random_summary["random_cv_f2_mean"],
        atol=0.000001,
    ), "El resultado aleatorio no se reprodujo"

    ranked = results[[
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "params",
    ]].sort_values(
        ["rank_test_score", "std_test_score"]
    ).reset_index(drop=True)

    best_mean = float(search.best_score_)

    # Candidato de menor dispersion a una distancia
    # maxima de 0.005 del mejor F2 medio.
    near_best = results.loc[
        results["mean_test_score"] >= best_mean - 0.005
    ]

    stable = near_best.sort_values(
        ["std_test_score", "mean_test_score"],
        ascending=[True, False],
    ).iloc[0]

    summary = {
        "method": "GridSearchCV",
        "primary_metric": "F2",
        "cv_folds": 5,
        "candidates": 24,
        "train_rows": len(X_train),
        "holdout_rows": len(X_test),
        "baseline_cv_f2_mean": random_summary[
            "baseline_cv_f2_mean"
        ],
        "randomized_cv_f2_mean": random_summary[
            "random_cv_f2_mean"
        ],
        "randomized_cv_f2_std": random_summary[
            "random_cv_f2_std"
        ],
        "randomized_winner_reproduced_f2": reproduced_f2,
        "grid_best_cv_f2_mean": best_mean,
        "grid_best_cv_f2_std": float(
            results.iloc[search.best_index_]["std_test_score"]
        ),
        "grid_best_parameters": search.best_params_,
        "stability_tolerance_f2": 0.005,
        "stable_candidate_cv_f2_mean": float(
            stable["mean_test_score"]
        ),
        "stable_candidate_cv_f2_std": float(
            stable["std_test_score"]
        ),
        "stable_candidate_parameters": stable["params"],
        "holdout_used_in_search": False,
        "holdout_evaluations_this_stage": 0,
        "final_model_selected": False,
    }

    reports = ROOT / "reports"
    models = ROOT / "models"

    ranked.to_csv(
        reports / "grid_search_results.csv",
        index=False,
    )

    (reports / "grid_search_summary.json").write_text(
        json.dumps(
            summary, indent=2, ensure_ascii=False
        ) + "\n",
        encoding="utf-8",
    )

    joblib.dump(
        search.best_estimator_,
        models / "churn_grid_search.joblib",
    )

    print("\n=== COMPARACION F2 CV ===")
    print(
        "Baseline:",
        f"{summary['baseline_cv_f2_mean']:.6f}"
    )
    print(
        "RandomizedSearchCV:",
        f"{summary['randomized_cv_f2_mean']:.6f}"
    )
    print(
        "GridSearchCV:",
        f"{summary['grid_best_cv_f2_mean']:.6f}"
    )
    print(
        "Desviacion Grid:",
        f"{summary['grid_best_cv_f2_std']:.6f}"
    )

    print("\n=== MEJORES PARAMETROS GRID ===")
    for key, value in search.best_params_.items():
        print(f"{key}: {value}")

    print("\n=== CANDIDATO MAS ESTABLE ===")
    print(
        "F2 medio:",
        f"{summary['stable_candidate_cv_f2_mean']:.6f}"
    )
    print(
        "Desviacion:",
        f"{summary['stable_candidate_cv_f2_std']:.6f}"
    )
    print(summary["stable_candidate_parameters"])

    print("\nPASS: 24 combinaciones")
    print("PASS: 5 folds estratificados")
    print("PASS: candidato aleatorio reproducido")
    print("PASS: holdout sin evaluar")


if __name__ == "__main__":
    main()
