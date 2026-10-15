import json
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_informe_con_metricas_correctas():
    report = (ROOT / "tuning_report.md").read_text()

    for text in [
        "RandomizedSearchCV",
        "GridSearchCV",
        "F2-score",
        "Limitaciones",
        "evaluacion",
    ]:
        assert text.lower() in report.lower()

    final = json.loads(
        (ROOT / "reports/final_model_selection.json").read_text()
    )

    baseline = final["baseline_test_metrics"]
    optimized = final["final_test_metrics"]

    assert optimized["f2_score"] > baseline["f2_score"]
    assert optimized["recall"] > baseline["recall"]
    assert optimized["false_negative"] < baseline["false_negative"]
    assert optimized["false_positive"] > baseline["false_positive"]


def test_notebook_valido_con_graficos():
    nb = nbformat.read(
        ROOT / "notebooks/streamloop_churn_tuning.ipynb",
        as_version=4,
    )
    nbformat.validate(nb)

    code_cells = [
        c for c in nb.cells if c.cell_type == "code"
    ]
    assert len(code_cells) >= 5
    assert all(
        c.execution_count is not None
        for c in code_cells
    )

    images = [
        output
        for cell in code_cells
        for output in cell.outputs
        if output.output_type == "display_data"
        and "image/png" in output.data
    ]
    assert len(images) >= 2


def test_resultados_finales_coherentes():
    result = json.loads(
        (ROOT / "reports/final_model_selection.json").read_text()
    )

    assert result["holdout_rows"] == 1409
    assert result["total_documented_holdout_evaluations"] == 2

    b = result["baseline_test_metrics"]
    f = result["final_test_metrics"]

    for metrics in (b, f):
        assert sum(
            metrics[key] for key in [
                "true_positive", "true_negative",
                "false_positive", "false_negative"
            ]
        ) == 1409

    assert np.isclose(
        result["differences"]["f2_delta"],
        f["f2_score"] - b["f2_score"],
    )
