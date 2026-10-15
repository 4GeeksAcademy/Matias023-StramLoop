"""Seleccion final y unica evaluacion optimizada de StreamLoop."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.churn_baseline import (
    load_dataset,
    split_dataset,
)


def main():
    output = ROOT / "reports/final_model_selection.json"

    if output.exists():
        raise RuntimeError(
            "El conjunto de prueba final ya fue evaluado. "
            "No repetir la evaluacion."
        )

    baseline = json.loads(
        (ROOT / "reports/baseline_metrics.json").read_text()
    )

    random = json.loads(
        (ROOT / "reports/random_search_summary.json").read_text()
    )

    grid = json.loads(
        (ROOT / "reports/grid_search_summary.json").read_text()
    )

    assert baseline["test_evaluations"] == 1
    assert random["holdout_used_in_search"] is False
    assert grid["holdout_used_in_search"] is False

    assert np.isclose(
        random["random_cv_f2_mean"],
        grid["grid_best_cv_f2_mean"],
        atol=0.000001,
    )

    assert np.isclose(
        grid["grid_best_cv_f2_mean"],
        grid["stable_candidate_cv_f2_mean"],
        atol=0.000001,
    )

    # Modelo previamente seleccionado mediante CV.
    model = joblib.load(
        ROOT / "models/churn_grid_search.joblib"
    )

    selected_params = grid["grid_best_parameters"]

    for key, expected in selected_params.items():
        actual = model.get_params()[key]
        assert actual == expected, (
            f"Parametro diferente: {key}"
        )

    X, y, _ = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    assert len(X_train) == 5634
    assert len(X_test) == 1409
    assert X_train.index.intersection(X_test.index).empty

    print("=== EVALUACION FINAL DEL HOLDOUT ===")

    # Evaluacion unica del modelo optimizado.
    predicted = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    assert len(predicted) == len(probability) == 1409
    assert np.isfinite(probability).all()

    tn, fp, fn, tp = confusion_matrix(
        y_test, predicted, labels=[0, 1]
    ).ravel()

    final = {
        "accuracy": float(
            accuracy_score(y_test, predicted)
        ),
        "precision": float(
            precision_score(
                y_test, predicted, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predicted, zero_division=0
            )
        ),
        "f2_score": float(
            fbeta_score(
                y_test, predicted,
                beta=2,
                zero_division=0,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_test, probability
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    original = baseline["test_metrics"]

    differences = {
        "f2_delta": float(
            final["f2_score"] - original["f2_score"]
        ),
        "recall_delta": float(
            final["recall"] - original["recall"]
        ),
        "precision_delta": float(
            final["precision"] - original["precision"]
        ),
        "false_negative_reduction": int(
            original["false_negative"]
            - final["false_negative"]
        ),
        "false_positive_change": int(
            final["false_positive"]
            - original["false_positive"]
        ),
    }

    report = {
        "project": "StreamLoop Churn Tuning",
        "final_model": "GridSearchCV RandomForestClassifier",
        "primary_metric": "F2",
        "selection_basis": (
            "F2 medio y estabilidad en 5 folds "
            "sobre entrenamiento. Grid igualo "
            "al ganador de RandomizedSearchCV."
        ),
        "holdout_rows": len(X_test),
        "baseline_holdout_evaluations": 1,
        "optimized_holdout_evaluations": 1,
        "total_documented_holdout_evaluations": 2,
        "baseline_cv_f2": random["baseline_cv_f2_mean"],
        "randomized_cv_f2": random["random_cv_f2_mean"],
        "grid_cv_f2": grid["grid_best_cv_f2_mean"],
        "grid_cv_std": grid["grid_best_cv_f2_std"],
        "selected_parameters": selected_params,
        "baseline_test_metrics": original,
        "final_test_metrics": final,
        "differences": differences,
        "evaluation_policy": (
            "Baseline y modelo optimizado evaluados "
            "por separado. Ninguna busqueda utiliza test."
        ),
    }

    predictions = pd.DataFrame({
        "row_index": X_test.index,
        "actual_churn": y_test.to_numpy(),
        "predicted_churn": predicted,
        "probability_churn": probability,
    })

    # Guardar las predicciones para documentar
    # sin volver a ejecutar el modelo en test.
    predictions.to_csv(
        ROOT / "reports/final_predictions.csv",
        index=False,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n=== COMPARACION TEST ===")
    print(
        f"{'Metrica':<20}"
        f"{'Baseline':>12}"
        f"{'Optimizado':>14}"
    )

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f2_score",
        "average_precision",
        "false_positive",
        "false_negative",
        "true_positive",
        "true_negative",
    ]:
        print(
            f"{key:<20}"
            f"{original[key]:>12.4f}"
            f"{final[key]:>14.4f}"
        )

    print("\n=== DIFERENCIAS ===")
    for key, value in differences.items():
        print(f"{key}: {value}")

    print("\nPASS: modelo seleccionado por CV")
    print("PASS: una evaluacion final optimizada")
    print("PASS: resultados guardados")
    print("PASS: test excluido de las busquedas")


if __name__ == "__main__":
    main()
