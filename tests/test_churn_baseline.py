from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.churn_baseline import (
    load_dataset,
    split_dataset,
    build_pipeline,
)


def test_split_reproducible_sin_mezclar_clientes():
    X, y, _ = load_dataset()

    a, b, ya, yb = split_dataset(X, y)
    c, d, yc, yd = split_dataset(X, y)

    assert len(a) == 5634
    assert len(b) == 1409
    assert a.index.intersection(b.index).empty

    assert a.index.equals(c.index)
    assert b.index.equals(d.index)
    assert ya.equals(yc)
    assert yb.equals(yd)


def test_preprocesamiento_esta_dentro_del_pipeline():
    X, y, _ = load_dataset()
    X_train, _, y_train, _ = split_dataset(X, y)

    pipeline = build_pipeline(X_train)

    assert "preprocess" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    pipeline.fit(
        X_train.iloc[:500],
        y_train.iloc[:500]
    )

    predictions = pipeline.predict(
        X_train.iloc[500:510]
    )

    assert len(predictions) == 10
    assert set(predictions).issubset({0, 1})
