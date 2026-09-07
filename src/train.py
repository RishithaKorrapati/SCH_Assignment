"""Train on a Gold file — only if both gates passed.

Two models (LogisticRegression and RandomForest) share the same
preprocessing, a stratified train/test split, and the same metrics.
The winner is the higher ROC-AUC. If either gate failed, print a
reason and skip training instead of fitting on that day's data.
"""

from dataclasses import dataclass
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ingest import _as_day, _zone


TARGET = "Churn"
ID_COL = "customerID"


@dataclass
class TrainResult:
    trained: bool
    skipped: bool
    reason: str
    selected_model: str = None
    metrics: dict = None
    model_path: str = None
    metrics_path: str = None


def _skip(reason):
    print(f"TRAINING SKIPPED: {reason}")
    return TrainResult(trained=False, skipped=True, reason=reason)


def _preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def _scores(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train(day, config, *, input_passed, output_passed):
    """Fit two models on Gold for `day`. Only runs if both gates passed."""
    day_str = _as_day(day)

    if not input_passed:
        return _skip(
            f"{day_str}: input gate failed, so training will not run"
        )
    if not output_passed:
        return _skip(
            f"{day_str}: output gate failed, so training will not run"
        )

    gold_path = _zone(config, "gold_dir") / f"{day_str}.csv"
    if not gold_path.exists():
        return _skip(f"{day_str}: no Gold file at {gold_path}")

    df = pd.read_csv(gold_path)
    min_rows = config["training"]["min_gold_rows"]
    if len(df) < min_rows:
        return _skip(
            f"{day_str}: Gold has {len(df)} rows, need at least {min_rows}"
        )

    y = df[TARGET]
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    numeric_cols = [
        c for c in config["output_gate"]["numeric_columns"] if c in X.columns
    ]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    tcfg = config["training"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=tcfg["test_size"],
        random_state=tcfg["random_state"],
        stratify=y,
    )

    lr_cfg = tcfg["logistic_regression"]
    rf_cfg = tcfg["random_forest"]
    seed = tcfg["random_state"]
    candidates = {
        "LogisticRegression": Pipeline(
            [
                ("prep", _preprocessor(numeric_cols, categorical_cols)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=lr_cfg["max_iter"], random_state=seed
                    ),
                ),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            [
                ("prep", _preprocessor(numeric_cols, categorical_cols)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=rf_cfg["n_estimators"],
                        max_depth=rf_cfg["max_depth"],
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }

    metric_name = tcfg["selection_metric"]
    candidate_metrics = {}
    fitted = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        candidate_metrics[name] = _scores(y_test, y_pred, y_proba)
        fitted[name] = pipe

    winner = max(candidate_metrics, key=lambda n: candidate_metrics[n][metric_name])

    models_dir = _zone(config, "models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{day_str}.joblib"
    metrics_path = models_dir / f"{day_str}_metrics.json"
    joblib.dump(fitted[winner], model_path)

    payload = {
        "day": day_str,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "selected_model": winner,
        "selection_metric": metric_name,
        "candidates": candidate_metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(
        f"TRAINED {day_str}: selected {winner} "
        f"({metric_name}={candidate_metrics[winner][metric_name]:.4f}) "
        f"-> {model_path.name}"
    )
    return TrainResult(
        trained=True,
        skipped=False,
        reason="",
        selected_model=winner,
        metrics=payload,
        model_path=str(model_path),
        metrics_path=str(metrics_path),
    )
