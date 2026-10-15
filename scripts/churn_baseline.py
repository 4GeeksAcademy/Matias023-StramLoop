"""StreamLoop: modelo baseline de cancelacion de clientes."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/raw/Telco-Customer-Churn.csv"
SEED = 42


def load_dataset():
    df = pd.read_csv(DATASET)

    required = {"customerID", "TotalCharges", "Churn"}
    assert required.issubset(df.columns)
    assert df.shape == (7043, 21)
    assert df["customerID"].is_unique
    assert df["Churn"].isin(["Yes", "No"]).all()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    missing = int(df["TotalCharges"].isna().sum())

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    return X, y, missing


def split_dataset(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )


def build_pipeline(X):
    numeric = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numerical_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    model = RandomForestClassifier(
        random_state=SEED,
        n_jobs=2,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", model),
    ])


def evaluate(model, X_test, y_test):
    predicted = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test, predicted, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(
            accuracy_score(y_test, predicted)
        ),
        "precision": float(
            precision_score(y_test, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predicted, zero_division=0)
        ),
        "f2_score": float(
            fbeta_score(
                y_test, predicted, beta=2, zero_division=0
            )
        ),
        "average_precision": float(
            average_precision_score(y_test, probability)
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main():
    X, y, missing = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    assert X_train.index.intersection(X_test.index).empty

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)

    result = {
        "model": "RandomForestClassifier",
        "version": "baseline",
        "random_state": SEED,
        "primary_metric": "f2_score",
        "metric_reason": (
            "Priorizar deteccion de clientes que cancelan, "
            "sin ignorar falsos positivos."
        ),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "missing_total_charges": missing,
        "test_metrics": metrics,
        "test_evaluations": 1,
    }

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    joblib.dump(
        pipeline,
        ROOT / "models/churn_baseline.joblib",
    )

    (ROOT / "reports/baseline_metrics.json").write_text(
        json.dumps(
            result, indent=2, ensure_ascii=False
        ) + "\n",
        encoding="utf-8",
    )

    print("=== RESULTADOS BASELINE ===")
    print("Train:", len(X_train))
    print("Test:", len(X_test))
    print("TotalCharges faltantes:", missing)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    assert np.isfinite(list(metrics.values())).all()

    print("PASS: baseline guardado")


if __name__ == "__main__":
    main()
