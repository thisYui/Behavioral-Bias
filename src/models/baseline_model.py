"""
src/models/baseline_model.py
─────────────────────────────────────────────────────────────────────────────
Baseline models — train sạch, không có bias, dùng làm điểm so sánh.

Hỗ trợ:
  - Logistic Regression   (linear, well-calibrated)
  - Random Forest         (ensemble, moderate overfit risk)
  - Gradient Boosting     (powerful, overfit risk cao)
  - Decision Tree         (interpretable, dễ overfit)

Tất cả models đều:
  1. Wrapped trong sklearn Pipeline (scaler + model)
  2. Có cross-validation built-in
  3. Trả về ModelResult — gói gọn model + metrics + predictions

Public API:
    build_model(model_type, task, **kwargs) → Pipeline
    train_model(pipeline, X_train, y_train) → ModelResult
    train_baseline(tt_bundle, model_type) → ModelResult  ← main entry point
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

ModelType = Literal["logistic_regression", "random_forest", "gradient_boosting", "decision_tree"]


# ─────────────────────────────────────────────────────────────────────────────
# Output Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """
    Kết quả sau khi train model — chứa mọi thứ cần để evaluate và so sánh.
    """
    model_type: str
    pipeline: Pipeline

    # Predictions trên test set
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None  # None nếu model không hỗ trợ predict_proba

    # Cross-validation scores (trên train set)
    cv_scores: dict = field(default_factory=dict)

    # Test set metrics (tính sau trong evaluation.py)
    test_metrics: dict = field(default_factory=dict)

    # Metadata
    train_size: int = 0
    test_size: int = 0
    n_features: int = 0
    train_time_sec: float = 0.0
    label: str = ""          # tên hiển thị (vd: "Baseline RF", "Biased LR")

    def __repr__(self) -> str:
        acc = self.test_metrics.get("accuracy", "N/A")
        auc = self.test_metrics.get("roc_auc", "N/A")
        return (
            f"ModelResult(type='{self.model_type}', "
            f"label='{self.label}', "
            f"accuracy={acc}, roc_auc={auc})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model Configs
# ─────────────────────────────────────────────────────────────────────────────

# Default hyperparameters cho từng model type
# Được thiết kế để tránh overfit — đây là BASELINE sạch
DEFAULT_CONFIGS: dict[ModelType, dict] = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
    },
    "decision_tree": {
        "max_depth": 6,
        "min_samples_leaf": 10,
        "random_state": 42,
    },
}

MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "random_forest":       RandomForestClassifier,
    "gradient_boosting":   GradientBoostingClassifier,
    "decision_tree":       DecisionTreeClassifier,
}


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    model_type: ModelType = "logistic_regression",
    scale_features: bool = True,
    **override_params,
) -> Pipeline:
    """
    Tạo sklearn Pipeline: [StandardScaler →] Model.

    Parameters
    ----------
    model_type : str
        Loại model muốn tạo
    scale_features : bool
        True → thêm StandardScaler vào pipeline
        False → bỏ qua (khi data đã được scale từ preprocess.py)
    **override_params :
        Override bất kỳ default hyperparameter nào

    Returns
    -------
    sklearn Pipeline (chưa fit)

    Examples
    --------
    >>> pipe = build_model("random_forest", max_depth=5, n_estimators=200)
    """
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"model_type phải là một trong: {list(MODEL_CLASSES)}")

    # Merge default config với override
    config = {**DEFAULT_CONFIGS[model_type], **override_params}
    estimator = MODEL_CLASSES[model_type](**config)

    steps = []
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))

    pipeline = Pipeline(steps)

    logger.debug("Built %s pipeline (scale=%s, params=%s)", model_type, scale_features, config)
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    metrics: list[str] | None = None,
) -> dict:
    """
    Chạy stratified k-fold cross-validation trên train set.

    Parameters
    ----------
    metrics : list[str], optional
        Metrics cần tính. Default: accuracy, f1, roc_auc, precision, recall

    Returns
    -------
    dict với mean và std của từng metric
    """
    if metrics is None:
        metrics = ["accuracy", "f1", "roc_auc", "precision", "recall"]

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=metrics,
        return_train_score=True,
        n_jobs=-1,
    )

    summary = {}
    for metric in metrics:
        val_key   = f"test_{metric}"
        train_key = f"train_{metric}"
        summary[metric] = {
            "val_mean":   round(float(cv_results[val_key].mean()), 4),
            "val_std":    round(float(cv_results[val_key].std()), 4),
            "train_mean": round(float(cv_results[train_key].mean()), 4),
            "train_std":  round(float(cv_results[train_key].std()), 4),
            "overfit_gap": round(
                float(cv_results[train_key].mean() - cv_results[val_key].mean()), 4
            ),
        }

    logger.info(
        "CV (%d-fold) — AUC: %.4f±%.4f, Acc: %.4f±%.4f",
        cv_folds,
        summary["roc_auc"]["val_mean"], summary["roc_auc"]["val_std"],
        summary["accuracy"]["val_mean"], summary["accuracy"]["val_std"],
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "model",
    label: str = "",
    run_cv: bool = True,
    cv_folds: int = 5,
) -> ModelResult:
    """
    Fit pipeline và generate predictions trên test set.

    Parameters
    ----------
    label : str
        Tên hiển thị cho model (vd: "Baseline LR", "Survivorship Biased RF")
    run_cv : bool
        Có chạy cross-validation không (tốn thêm thời gian nhưng cho thêm insight)

    Returns
    -------
    ModelResult với đầy đủ predictions và CV scores
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

    logger.info("Training %s (%s)...", model_type, label or "no label")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            pass

    # Test metrics
    test_metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
    }
    if y_proba is not None:
        test_metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)

    # Cross-validation
    cv_scores = {}
    if run_cv:
        cv_scores = run_cross_validation(pipeline, X_train, y_train, cv_folds=cv_folds)

    logger.info(
        "Done in %.2fs — test acc=%.4f, auc=%s",
        train_time, test_metrics["accuracy"],
        test_metrics.get("roc_auc", "N/A"),
    )

    return ModelResult(
        model_type=model_type,
        pipeline=pipeline,
        y_true=y_test.values,
        y_pred=y_pred,
        y_proba=y_proba,
        cv_scores=cv_scores,
        test_metrics=test_metrics,
        train_size=len(X_train),
        test_size=len(X_test),
        n_features=X_train.shape[1],
        train_time_sec=round(train_time, 3),
        label=label or model_type,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(
    tt_bundle,                              # TrainTestBundle từ preprocess.py
    model_type: ModelType = "logistic_regression",
    label: str = "",
    run_cv: bool = True,
    scale_features: bool = False,           # False vì TrainTestBundle đã scale
    **model_kwargs,
) -> ModelResult:
    """
    Entry point duy nhất — nhận TrainTestBundle, trả về ModelResult.

    Parameters
    ----------
    tt_bundle : TrainTestBundle
        Output từ preprocess.preprocess()
    model_type : str
        Loại model baseline muốn train
    scale_features : bool
        False nếu dùng data từ preprocess.py (đã scale sẵn)
    **model_kwargs :
        Override hyperparameters

    Examples
    --------
    >>> from src.data.load_data import load_dataset
    >>> from src.data.preprocess import preprocess
    >>> from src.models.baseline_model import train_baseline
    >>>
    >>> bundle = load_dataset("synthetic_clf")
    >>> tt = preprocess(bundle)
    >>> result = train_baseline(tt, model_type="random_forest")
    >>> print(result)
    """
    pipeline = build_model(
        model_type=model_type,
        scale_features=scale_features,
        **model_kwargs,
    )

    _label = label or f"Baseline {model_type.replace('_', ' ').title()}"

    return train_model(
        pipeline=pipeline,
        X_train=tt_bundle.X_train,
        y_train=tt_bundle.y_train,
        X_test=tt_bundle.X_test,
        y_test=tt_bundle.y_test,
        model_type=model_type,
        label=_label,
        run_cv=run_cv,
    )


def train_all_baselines(
    tt_bundle,
    model_types: list[ModelType] | None = None,
    run_cv: bool = True,
) -> dict[str, ModelResult]:
    """
    Train tất cả baseline models cùng lúc — dùng để so sánh.

    Returns
    -------
    dict: model_type → ModelResult
    """
    if model_types is None:
        model_types = list(MODEL_CLASSES.keys())

    results = {}
    for mt in model_types:
        results[mt] = train_baseline(tt_bundle, model_type=mt, run_cv=run_cv)

    logger.info(
        "Trained %d baselines: %s",
        len(results),
        {k: v.test_metrics.get("roc_auc") for k, v in results.items()},
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data.load_data import load_dataset
    from src.data.preprocess import preprocess

    bundle = load_dataset("synthetic_clf", n_samples=1_000)
    tt = preprocess(bundle)

    # Train single baseline
    result = train_baseline(tt, model_type="logistic_regression")
    print(result)
    print("Test metrics :", result.test_metrics)
    print("CV AUC       :", result.cv_scores.get("roc_auc", {}).get("val_mean"))

    # Train all baselines
    all_results = train_all_baselines(tt, run_cv=False)
    for name, res in all_results.items():
        print(f"  {name:25s} → AUC={res.test_metrics.get('roc_auc')}, Acc={res.test_metrics['accuracy']}")