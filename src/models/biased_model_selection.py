"""
src/models/biased_model_selection.py
─────────────────────────────────────────────────────────────────────────────
Simulate các bias xảy ra trong quá trình MODEL SELECTION — không phải data.

Đây là layer thứ hai của bias: ngay cả khi data sạch, data scientist
vẫn có thể inject bias qua cách chọn và evaluate models.

Các bias được simulate:
  1. Threshold Fishing     — chọn decision threshold SAU khi thấy kết quả
  2. Metric Cherry-Picking — chọn metric nào báo cáo kết quả tốt nhất
  3. Test Set Reuse        — tune model trên test set nhiều lần (leakage)
  4. Hyperparameter Fishing— thử nhiều configs, report config tốt nhất
  5. Selective Reporting   — chỉ báo cáo models tốt, giấu models tệ

Public API:
    fish_decision_threshold(result, objective)
    cherry_pick_metric(result)
    simulate_test_set_reuse(tt_bundle, n_rounds)
    fish_hyperparameters(tt_bundle, model_type, param_grid, n_trials)
    simulate_biased_selection(tt_bundle, strategy, **kwargs) → SelectionBiasResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.baseline_model import ModelResult, build_model, train_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SelectionBiasResult:
    """Kết quả sau khi áp dụng model selection bias."""
    strategy: str
    baseline_result: ModelResult
    bias_params: dict = field(default_factory=dict)
    bias_report: dict = field(default_factory=dict)

    # Biased metrics sau khi apply strategy
    reported_metrics: dict = field(default_factory=dict)
    honest_metrics: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SelectionBiasResult(strategy='{self.strategy}', "
            f"reported={self.reported_metrics}, "
            f"honest={self.honest_metrics})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute all metrics at once
# ─────────────────────────────────────────────────────────────────────────────

def _all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict:
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
        metrics["avg_precision"] = round(average_precision_score(y_true, y_proba), 4)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Threshold Fishing
# ─────────────────────────────────────────────────────────────────────────────

def fish_decision_threshold(
    result: ModelResult,
    objective: Literal["f1", "precision", "recall", "accuracy"] = "f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict]:
    """
    Tìm decision threshold tốt nhất trên TEST SET — data leakage cổ điển.

    Vấn đề: threshold nên được chọn trên validation set riêng, không phải test.
    Bias: báo cáo metric tại threshold tốt nhất như thể đó là kết quả thật.

    Parameters
    ----------
    objective : str
        Metric muốn maximize khi chọn threshold
    thresholds : array, optional
        Danh sách thresholds cần thử. Default: 0.05 đến 0.95

    Returns
    -------
    best_threshold, report
    """
    if result.y_proba is None:
        raise ValueError("Model không có predict_proba — không thể fish threshold.")

    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.02)

    y_true  = result.y_true
    y_proba = result.y_proba

    # Default threshold (0.5) — honest baseline
    default_pred = (y_proba >= 0.5).astype(int)
    honest_metrics = _all_metrics(y_true, default_pred, y_proba)

    # Thử tất cả thresholds
    threshold_results = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        m = _all_metrics(y_true, y_pred_t, None)  # y_proba metrics không đổi
        m["threshold"] = round(float(t), 3)
        threshold_results.append(m)

    # Chọn threshold tốt nhất theo objective
    best = max(threshold_results, key=lambda x: x.get(objective, 0))
    best_threshold = best["threshold"]
    best_pred = (y_proba >= best_threshold).astype(int)
    reported_metrics = _all_metrics(y_true, best_pred, y_proba)

    report = {
        "strategy": "threshold_fishing",
        "objective": objective,
        "n_thresholds_tried": len(thresholds),
        "default_threshold": 0.5,
        "best_threshold": best_threshold,
        "honest_metrics": honest_metrics,
        "reported_metrics": reported_metrics,
        "gain": {
            k: round(reported_metrics.get(k, 0) - honest_metrics.get(k, 0), 4)
            for k in ["accuracy", "f1", "precision", "recall"]
        },
        "interpretation": (
            f"Chọn threshold={best_threshold} trên test set: "
            f"{objective}={reported_metrics.get(objective, 'N/A')} "
            f"thay vì honest {objective}={honest_metrics.get(objective, 'N/A')}"
        ),
        "all_threshold_scores": threshold_results,
    }

    logger.info(
        "Threshold fishing (%s): default=0.5→%.4f, best=%.2f→%.4f (gain=+%.4f)",
        objective,
        honest_metrics.get(objective, 0),
        best_threshold,
        reported_metrics.get(objective, 0),
        reported_metrics.get(objective, 0) - honest_metrics.get(objective, 0),
    )
    return best_threshold, report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Metric Cherry-Picking
# ─────────────────────────────────────────────────────────────────────────────

def cherry_pick_metric(
    result: ModelResult,
    report_top_n: int = 1,
) -> dict:
    """
    Tính tất cả metrics, chỉ báo cáo metric nào tốt nhất.

    Ví dụ: model có AUC=0.75 (meh) nhưng Precision=0.91 (trông đẹp),
    data scientist chỉ report Precision.

    Parameters
    ----------
    report_top_n : int
        Số metrics được "chọn" để báo cáo
    """
    y_true  = result.y_true
    y_pred  = result.y_pred
    y_proba = result.y_proba

    all_metrics = _all_metrics(y_true, y_pred, y_proba)

    # Rank metrics từ cao xuống thấp
    ranked = sorted(all_metrics.items(), key=lambda x: x[1], reverse=True)

    reported = dict(ranked[:report_top_n])
    hidden   = dict(ranked[report_top_n:])

    # "Honest" = trung bình tất cả metrics
    honest_avg = round(np.mean(list(all_metrics.values())), 4)
    reported_avg = round(np.mean(list(reported.values())), 4)

    report = {
        "strategy": "metric_cherry_picking",
        "all_metrics": all_metrics,
        "ranked_metrics": ranked,
        "reported_metrics": reported,
        "hidden_metrics": hidden,
        "report_top_n": report_top_n,
        "reported_avg": reported_avg,
        "honest_avg": honest_avg,
        "inflation": round(reported_avg - honest_avg, 4),
        "interpretation": (
            f"Báo cáo top-{report_top_n} metric ({reported}) "
            f"— impression inflation: {reported_avg:.4f} vs honest avg {honest_avg:.4f}"
        ),
    }

    logger.info(
        "Cherry-picked metric: reported=%s (avg=%.4f), hidden=%s (honest_avg=%.4f)",
        list(reported.keys()), reported_avg,
        list(hidden.keys()), honest_avg,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Test Set Reuse
# ─────────────────────────────────────────────────────────────────────────────

def simulate_test_set_reuse(
    tt_bundle,
    model_type: str = "logistic_regression",
    n_rounds: int = 10,
    adjust_regularization: bool = True,
) -> dict:
    """
    Simulate việc tune model nhiều lần trên CÙNG test set.

    Mỗi round: adjust một hyperparameter nhỏ, đo trên test set,
    giữ config nào test tốt nhất → inflated test performance.

    Parameters
    ----------
    n_rounds : int
        Số lần "refine" model trên test set
    adjust_regularization : bool
        Simulate điều chỉnh regularization strength mỗi round
    """
    rng = np.random.default_rng(seed=0)
    X_train = tt_bundle.X_train
    y_train = tt_bundle.y_train
    X_test  = tt_bundle.X_test
    y_test  = tt_bundle.y_test

    # Round 0: baseline train (honest — không nhìn test)
    pipe0 = build_model(model_type, scale_features=False)
    r0 = train_model(pipe0, X_train, y_train, X_test, y_test,
                     model_type=model_type, label="Round 0 (honest)", run_cv=False)
    honest_auc = r0.test_metrics.get("roc_auc", r0.test_metrics["accuracy"])

    round_results = [{"round": 0, "config": "default", "test_auc": honest_auc}]
    best_auc = honest_auc
    best_round = 0

    # Simulate n_rounds "peeking" at test set
    C_values = np.logspace(-2, 2, n_rounds)   # try nhiều regularization values

    for i, C in enumerate(C_values, start=1):
        kwargs = {"C": C} if model_type == "logistic_regression" else {
            "max_depth": int(rng.integers(3, 12)),
            "n_estimators": int(rng.integers(50, 300)),
        }
        pipe_i = build_model(model_type, scale_features=False, **kwargs)
        r_i = train_model(pipe_i, X_train, y_train, X_test, y_test,
                          model_type=model_type, label=f"Round {i}", run_cv=False)
        auc_i = r_i.test_metrics.get("roc_auc", r_i.test_metrics["accuracy"])
        round_results.append({"round": i, "config": kwargs, "test_auc": round(auc_i, 4)})

        if auc_i > best_auc:
            best_auc = auc_i
            best_round = i

    all_aucs = [r["test_auc"] for r in round_results]

    report = {
        "strategy": "test_set_reuse",
        "n_rounds": n_rounds,
        "honest_auc_round0": round(honest_auc, 4),
        "reported_auc_best": round(best_auc, 4),
        "inflation": round(best_auc - honest_auc, 4),
        "best_round": best_round,
        "all_round_aucs": all_aucs,
        "mean_auc_across_rounds": round(float(np.mean(all_aucs)), 4),
        "interpretation": (
            f"Test set reuse {n_rounds} rounds: "
            f"honest AUC={honest_auc:.4f} → reported AUC={best_auc:.4f} "
            f"(+{best_auc - honest_auc:.4f} inflation)"
        ),
    }

    logger.info(
        "Test set reuse (%d rounds): AUC %.4f→%.4f (+%.4f)",
        n_rounds, honest_auc, best_auc, best_auc - honest_auc,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: Hyperparameter Fishing
# ─────────────────────────────────────────────────────────────────────────────

def fish_hyperparameters(
    tt_bundle,
    model_type: str = "random_forest",
    n_random_trials: int = 30,
    use_validation: bool = False,   # False = fish trên test (biased), True = honest
    random_state: int = 0,
) -> dict:
    """
    Random search hyperparameters — báo cáo kết quả tốt nhất mà không
    điều chỉnh cho multiple comparisons.

    Phân biệt:
    - use_validation=False: tune trực tiếp trên test → BIASED
    - use_validation=True : tune trên separate val set → HONEST

    Parameters
    ----------
    n_random_trials : int
        Số configs ngẫu nhiên thử
    use_validation : bool
        False = simulate biased workflow (fish trên test set)
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed=random_state)

    X_train_full = tt_bundle.X_train
    y_train_full = tt_bundle.y_train
    X_test = tt_bundle.X_test
    y_test = tt_bundle.y_test

    if use_validation:
        # Honest: tách validation từ train
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
        X_eval, y_eval = X_val, y_val
        label = "validation (honest)"
    else:
        # Biased: dùng thẳng test set để "validate"
        X_tr, y_tr = X_train_full, y_train_full
        X_eval, y_eval = X_test, y_test
        label = "test set (biased)"

    # Random hyperparameter configs
    configs = []
    for _ in range(n_random_trials):
        if model_type in ("random_forest", "gradient_boosting"):
            cfg = {
                "n_estimators": int(rng.integers(50, 400)),
                "max_depth": int(rng.integers(2, 15)),
            }
        else:
            cfg = {"C": float(rng.choice(np.logspace(-3, 3, 50)))}
        configs.append(cfg)

    trial_results = []
    for i, cfg in enumerate(configs):
        pipe = build_model(model_type, scale_features=False, **cfg)
        r = train_model(pipe, X_tr, y_tr, X_eval, y_eval,
                        model_type=model_type, label=f"trial_{i}", run_cv=False)
        auc = r.test_metrics.get("roc_auc", r.test_metrics["accuracy"])
        trial_results.append({"trial": i, "config": cfg, "eval_auc": round(auc, 4)})

    # Best config evaluated on actual test set
    best = max(trial_results, key=lambda x: x["eval_auc"])
    best_pipe = build_model(model_type, scale_features=False, **best["config"])
    best_result = train_model(best_pipe, X_train_full, y_train_full, X_test, y_test,
                              model_type=model_type, label="best_config", run_cv=False)
    actual_test_auc = best_result.test_metrics.get("roc_auc", best_result.test_metrics["accuracy"])

    # Baseline (default params) on test
    base_pipe = build_model(model_type, scale_features=False)
    base_result = train_model(base_pipe, X_train_full, y_train_full, X_test, y_test,
                              model_type=model_type, label="default", run_cv=False)
    baseline_auc = base_result.test_metrics.get("roc_auc", base_result.test_metrics["accuracy"])

    all_eval_aucs = [t["eval_auc"] for t in trial_results]

    report = {
        "strategy": "hyperparameter_fishing",
        "model_type": model_type,
        "n_trials": n_random_trials,
        "eval_set": label,
        "biased": not use_validation,
        "best_config": best["config"],
        "best_eval_auc": round(best["eval_auc"], 4),
        "actual_test_auc": round(actual_test_auc, 4),
        "baseline_test_auc": round(baseline_auc, 4),
        "eval_inflation": round(best["eval_auc"] - actual_test_auc, 4),
        "vs_baseline": round(actual_test_auc - baseline_auc, 4),
        "mean_eval_auc": round(float(np.mean(all_eval_aucs)), 4),
        "max_eval_auc": round(float(np.max(all_eval_aucs)), 4),
        "all_trial_aucs": all_eval_aucs,
        "interpretation": (
            f"{'Biased' if not use_validation else 'Honest'} search ({n_random_trials} trials): "
            f"reported eval AUC={best['eval_auc']:.4f} → actual test AUC={actual_test_auc:.4f} "
            f"(eval inflation={best['eval_auc']-actual_test_auc:+.4f})"
        ),
    }

    logger.info(
        "HP fishing (%s, %d trials): eval AUC=%.4f, actual test AUC=%.4f",
        label, n_random_trials, best["eval_auc"], actual_test_auc,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 5: Selective Reporting
# ─────────────────────────────────────────────────────────────────────────────

def selective_reporting(
    tt_bundle,
    model_types: list[str] | None = None,
    report_top_n: int = 1,
) -> dict:
    """
    Train nhiều models, chỉ báo cáo model(s) tốt nhất — giấu kết quả tệ.

    Ví dụ: thử 6 models, chỉ publish kết quả model tốt nhất.
    Người đọc nghĩ model tốt hơn thực tế vì không thấy các lần thử thất bại.

    Parameters
    ----------
    report_top_n : int
        Số models được báo cáo
    """
    if model_types is None:
        model_types = ["logistic_regression", "random_forest",
                       "gradient_boosting", "decision_tree"]

    X_train = tt_bundle.X_train
    y_train = tt_bundle.y_train
    X_test  = tt_bundle.X_test
    y_test  = tt_bundle.y_test

    all_results = {}
    for mt in model_types:
        pipe = build_model(mt, scale_features=False)
        r = train_model(pipe, X_train, y_train, X_test, y_test,
                        model_type=mt, label=mt, run_cv=False)
        auc = r.test_metrics.get("roc_auc", r.test_metrics["accuracy"])
        all_results[mt] = {"auc": round(auc, 4), "metrics": r.test_metrics}

    ranked = sorted(all_results.items(), key=lambda x: x[1]["auc"], reverse=True)
    reported = dict(ranked[:report_top_n])
    suppressed = dict(ranked[report_top_n:])

    all_aucs = [v["auc"] for v in all_results.values()]
    honest_avg = round(float(np.mean(all_aucs)), 4)
    reported_avg = round(float(np.mean([v["auc"] for v in reported.values()])), 4)

    report = {
        "strategy": "selective_reporting",
        "n_models_trained": len(model_types),
        "report_top_n": report_top_n,
        "all_model_aucs": {k: v["auc"] for k, v in all_results.items()},
        "reported_models": {k: v["auc"] for k, v in reported.items()},
        "suppressed_models": {k: v["auc"] for k, v in suppressed.items()},
        "reported_avg_auc": reported_avg,
        "honest_avg_auc": honest_avg,
        "inflation": round(reported_avg - honest_avg, 4),
        "interpretation": (
            f"Trained {len(model_types)} models, reported top {report_top_n}: "
            f"AUC={reported_avg:.4f} vs honest avg={honest_avg:.4f} "
            f"(+{reported_avg - honest_avg:.4f})"
        ),
    }

    logger.info(
        "Selective reporting (%d models, top %d): reported=%.4f, honest_avg=%.4f",
        len(model_types), report_top_n, reported_avg, honest_avg,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main Simulation Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def simulate_biased_selection(
    tt_bundle,
    strategy: Literal[
        "threshold_fishing",
        "metric_cherry_picking",
        "test_set_reuse",
        "hyperparameter_fishing",
        "selective_reporting",
    ] = "threshold_fishing",
    model_type: str = "logistic_regression",
    **kwargs,
) -> SelectionBiasResult:
    """
    Entry point duy nhất cho model selection bias simulation.

    Tự động train baseline model trước, sau đó apply selection bias.

    Examples
    --------
    >>> from src.data.load_data import load_dataset
    >>> from src.data.preprocess import preprocess
    >>> from src.models.biased_model_selection import simulate_biased_selection
    >>>
    >>> bundle = load_dataset("synthetic_clf")
    >>> tt = preprocess(bundle)
    >>> result = simulate_biased_selection(tt, strategy="threshold_fishing")
    >>> print(result.bias_report["interpretation"])
    """
    from src.models.baseline_model import train_baseline

    # Train baseline model trước
    baseline = train_baseline(tt_bundle, model_type=model_type, run_cv=False)
    honest_metrics = baseline.test_metrics.copy()

    result = SelectionBiasResult(
        strategy=strategy,
        baseline_result=baseline,
        bias_params={"model_type": model_type, **kwargs},
        honest_metrics=honest_metrics,
    )

    # ── Strategy dispatch ─────────────────────────────────────────────────────
    if strategy == "threshold_fishing":
        _, report = fish_decision_threshold(baseline, **kwargs)
        result.reported_metrics = report["reported_metrics"]
        result.bias_report = report

    elif strategy == "metric_cherry_picking":
        report = cherry_pick_metric(baseline, **kwargs)
        result.reported_metrics = report["reported_metrics"]
        result.bias_report = report

    elif strategy == "test_set_reuse":
        report = simulate_test_set_reuse(tt_bundle, model_type=model_type, **kwargs)
        result.reported_metrics = {"roc_auc": report["reported_auc_best"]}
        result.bias_report = report

    elif strategy == "hyperparameter_fishing":
        report = fish_hyperparameters(tt_bundle, model_type=model_type, **kwargs)
        result.reported_metrics = {"roc_auc": report["best_eval_auc"]}
        result.bias_report = report

    elif strategy == "selective_reporting":
        report = selective_reporting(tt_bundle, **kwargs)
        result.reported_metrics = report["reported_models"]
        result.bias_report = report

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data.load_data import load_dataset
    from src.data.preprocess import preprocess

    bundle = load_dataset("synthetic_clf", n_samples=1_000)
    tt = preprocess(bundle)

    strategies = [
        ("threshold_fishing",      {}),
        ("metric_cherry_picking",  {"report_top_n": 1}),
        ("test_set_reuse",         {"n_rounds": 10}),
        ("hyperparameter_fishing", {"n_random_trials": 15}),
        ("selective_reporting",    {"report_top_n": 1}),
    ]

    for strat, kw in strategies:
        result = simulate_biased_selection(tt, strategy=strat, **kw)
        print(f"\n[{strat}]")
        print(" ", result.bias_report.get("interpretation", ""))