"""
src/simulation/overconfidence_bias.py
─────────────────────────────────────────────────────────────────────────────
Simulation Overconfidence Bias trong Data Science workflow.

Overconfidence bias xảy ra khi model (hoặc data scientist) quá tự tin
vào predictions — underestimate uncertainty và overestimate accuracy.

Biểu hiện cụ thể:
  1. Probability calibration kém: model nói 90% confident nhưng thực tế chỉ đúng 60%
  2. Prediction interval quá hẹp: CI 95% thực tế chỉ cover 60% cases
  3. Overfitting được báo cáo như genuine performance
  4. Metric hacking: báo cáo metric tốt nhất trong nhiều lần thử
  5. Threshold manipulation: chọn decision threshold sau khi thấy kết quả

Public API:
    inflate_probabilities(y_proba, inflation_factor)
    narrow_prediction_intervals(y_pred, intervals, squeeze_factor)
    simulate_metric_hacking(X, y, n_trials, metric)
    inject_overfit_confidence(model, X_train, X_test)
    simulate_overconfidence_bias(bundle, strategy, **kwargs) → OverconfidenceResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from src.data.load_data import DataBundle
from src.simulation.confirmation_bias import BiasedBundle

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output Container (mở rộng BiasedBundle cho regression/probability output)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OverconfidenceResult:
    """Kết quả simulation overconfidence — chứa cả metrics so sánh."""
    original_bundle: DataBundle
    bias_type: str = "overconfidence_bias"
    bias_strategy: str = ""
    bias_params: dict = field(default_factory=dict)
    bias_report: dict = field(default_factory=dict)

    # Probability outputs (cho calibration analysis)
    y_true: np.ndarray | None = None
    y_proba_original: np.ndarray | None = None
    y_proba_biased: np.ndarray | None = None

    # Interval outputs (cho regression)
    y_pred: np.ndarray | None = None
    intervals_original: np.ndarray | None = None  # shape (n, 2): [lower, upper]
    intervals_biased: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"OverconfidenceResult(strategy='{self.bias_strategy}')"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Calibration Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Đo lường mức độ calibration của probability predictions.

    Returns ECE (Expected Calibration Error), MCE (Maximum), Brier Score.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")

    # ECE: weighted average |predicted_prob - actual_freq|
    bin_sizes = np.histogram(y_proba, bins=n_bins, range=(0, 1))[0]
    ece = float(np.sum(np.abs(prob_true - prob_pred) * bin_sizes) / len(y_true))

    # MCE: max calibration error
    mce = float(np.max(np.abs(prob_true - prob_pred)))

    # Brier score (lower = better calibrated)
    brier = float(brier_score_loss(y_true, y_proba))

    # Overconfidence index: mean predicted - mean actual (>0 = overconfident)
    overconf_idx = float(np.mean(y_proba) - np.mean(y_true))

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "overconfidence_index": round(overconf_idx, 4),
        "mean_predicted_prob": round(float(np.mean(y_proba)), 4),
        "mean_actual_rate": round(float(np.mean(y_true)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Probability Inflation
# ─────────────────────────────────────────────────────────────────────────────

def inflate_probabilities(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    inflation_factor: float = 1.4,
    clip: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Inflate predicted probabilities — simulate model quá tự tin.

    Ví dụ: model logistic regression raw thường well-calibrated,
    nhưng random forest / boosting thường overconfident ở extremes.

    Parameters
    ----------
    inflation_factor : float
        Hệ số inflate xác suất về phía 0 hoặc 1 (>1 = overconfident)
    clip : bool
        Clip kết quả về [0, 1]
    """
    # Đẩy probabilities về phía 0.5 bằng cách stretch về extremes
    y_proba_biased = (y_proba - 0.5) * inflation_factor + 0.5
    if clip:
        y_proba_biased = np.clip(y_proba_biased, 0.0, 1.0)

    orig_metrics = compute_calibration_metrics(y_true, y_proba)
    bias_metrics = compute_calibration_metrics(y_true, y_proba_biased)

    report = {
        "strategy": "inflate_probabilities",
        "inflation_factor": inflation_factor,
        "original_calibration": orig_metrics,
        "biased_calibration": bias_metrics,
        "ece_increase": round(bias_metrics["ece"] - orig_metrics["ece"], 4),
        "overconfidence_increase": round(
            abs(bias_metrics["overconfidence_index"]) - abs(orig_metrics["overconfidence_index"]), 4
        ),
    }

    logger.info(
        "Probability inflation x%.1f: ECE %.4f→%.4f",
        inflation_factor, orig_metrics["ece"], bias_metrics["ece"],
    )
    return y_proba_biased, report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Narrow Prediction Intervals
# ─────────────────────────────────────────────────────────────────────────────

def narrow_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence_level: float = 0.95,
    squeeze_factor: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Tạo prediction intervals thực tế vs intervals bị "squeeze" (quá tự tin).

    Simulate: data scientist báo cáo 95% CI nhưng thực ra chỉ là 60% CI.

    Parameters
    ----------
    confidence_level : float
        Confidence level muốn đạt (0.95 = 95% CI)
    squeeze_factor : float
        Hệ số thu hẹp interval (0 < squeeze < 1, nhỏ hơn = hẹp hơn)

    Returns
    -------
    intervals_true : ndarray shape (n, 2) — intervals đúng
    intervals_biased : ndarray shape (n, 2) — intervals bị squeeze
    report : dict
    """
    residuals = y_true - y_pred
    std_resid = np.std(residuals)

    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence_level, 1.960)

    # Correct intervals
    margin_true = z * std_resid
    intervals_true = np.column_stack([
        y_pred - margin_true,
        y_pred + margin_true,
    ])

    # Biased (squeezed) intervals
    margin_biased = z * std_resid * squeeze_factor
    intervals_biased = np.column_stack([
        y_pred - margin_biased,
        y_pred + margin_biased,
    ])

    # Coverage rates
    def coverage(intervals: np.ndarray) -> float:
        inside = (y_true >= intervals[:, 0]) & (y_true <= intervals[:, 1])
        return float(inside.mean())

    cov_true   = coverage(intervals_true)
    cov_biased = coverage(intervals_biased)

    report = {
        "strategy": "narrow_prediction_intervals",
        "claimed_confidence_level": confidence_level,
        "true_coverage": round(cov_true, 4),
        "biased_coverage": round(cov_biased, 4),
        "coverage_gap": round(confidence_level - cov_biased, 4),
        "mean_interval_width_true": round(float(np.mean(intervals_true[:, 1] - intervals_true[:, 0])), 4),
        "mean_interval_width_biased": round(float(np.mean(intervals_biased[:, 1] - intervals_biased[:, 0])), 4),
        "squeeze_factor": squeeze_factor,
        "interpretation": (
            f"Model tuyên bố {confidence_level*100:.0f}% CI nhưng thực tế "
            f"chỉ cover {cov_biased*100:.1f}% cases."
        ),
    }

    logger.info(
        "Interval squeeze x%.1f: claimed %.0f%% CI, actual coverage %.1f%%",
        squeeze_factor, confidence_level * 100, cov_biased * 100,
    )
    return intervals_true, intervals_biased, report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Metric Hacking (p-hacking equivalent)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_metric_hacking(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 20,
    metric: Literal["accuracy", "f1", "roc_auc"] = "roc_auc",
    random_state_start: int = 0,
) -> dict:
    """
    Simulate "metric hacking": chạy nhiều lần với random seeds khác nhau,
    chỉ báo cáo kết quả tốt nhất — inflate perceived model performance.

    Ví dụ thực tế:
    - Chạy 20 lần cross-validation với seeds khác nhau, report run tốt nhất
    - Thử nhiều models, chỉ report model tốt nhất mà không điều chỉnh cho multiple testing

    Parameters
    ----------
    n_trials : int
        Số lần thử (càng nhiều, bias càng lớn)
    metric : str
        Metric được hack
    """
    scoring_map = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    scoring = scoring_map[metric]

    all_scores = []
    model = LogisticRegression(max_iter=1000)

    for i in range(n_trials):
        seed = random_state_start + i
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring,
                                  n_jobs=-1)
        all_scores.append({
            "trial": i,
            "seed": seed,
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "scores": scores.tolist(),
        })

    scores_means = [t["mean"] for t in all_scores]

    best_trial   = max(all_scores, key=lambda t: t["mean"])
    worst_trial  = min(all_scores, key=lambda t: t["mean"])
    honest_score = np.mean(scores_means)   # trung bình tất cả trials

    report = {
        "strategy": "simulate_metric_hacking",
        "metric": metric,
        "n_trials": n_trials,
        "honest_score": round(honest_score, 4),         # should report
        "reported_score": round(best_trial["mean"], 4), # actually reported
        "worst_score": round(worst_trial["mean"], 4),
        "score_inflation": round(best_trial["mean"] - honest_score, 4),
        "inflation_pct": round(100 * (best_trial["mean"] - honest_score) / honest_score, 2),
        "all_scores": scores_means,
        "best_trial": best_trial,
        "interpretation": (
            f"Báo cáo {metric}={best_trial['mean']:.4f} thay vì "
            f"honest estimate {honest_score:.4f} — "
            f"inflate {100*(best_trial['mean']-honest_score)/honest_score:.1f}%"
        ),
    }

    logger.info(
        "Metric hacking (%d trials): honest=%.4f, reported=%.4f (+%.1f%%)",
        n_trials, honest_score, best_trial["mean"],
        100 * (best_trial["mean"] - honest_score) / honest_score,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: Overfit Confidence (train vs test gap)
# ─────────────────────────────────────────────────────────────────────────────

def inject_overfit_confidence(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    overfit_n_estimators: int = 500,
    overfit_max_depth: int | None = None,  # None = unlimited = overfit
) -> dict:
    """
    Simulate overconfidence bằng cách dùng model overfit:
    Train accuracy rất cao nhưng test accuracy thấp hơn nhiều.
    Data scientist chỉ báo cáo train accuracy.

    Parameters
    ----------
    overfit_n_estimators : int
        Số trees — nhiều hơn → dễ overfit hơn
    overfit_max_depth : int, optional
        None = unlimited depth = overfit hoàn toàn
    """
    # Baseline model (regularized)
    baseline = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    baseline.fit(X_train, y_train)

    # Overfit model
    overfit = RandomForestClassifier(
        n_estimators=overfit_n_estimators,
        max_depth=overfit_max_depth,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    overfit.fit(X_train, y_train)

    def metrics(model, X, y) -> dict:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        return {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, y_prob), 4),
        }

    baseline_train = metrics(baseline, X_train, y_train)
    baseline_test  = metrics(baseline, X_test, y_test)
    overfit_train  = metrics(overfit, X_train, y_train)
    overfit_test   = metrics(overfit, X_test, y_test)

    report = {
        "strategy": "inject_overfit_confidence",
        "baseline": {"train": baseline_train, "test": baseline_test},
        "overfit_model": {"train": overfit_train, "test": overfit_test},
        "generalization_gap": {
            metric: round(overfit_train[metric] - overfit_test[metric], 4)
            for metric in ["accuracy", "f1", "roc_auc"]
        },
        "reported_vs_real": {
            metric: {
                "reported (train)": overfit_train[metric],
                "real (test)": overfit_test[metric],
                "inflation": round(overfit_train[metric] - overfit_test[metric], 4),
            }
            for metric in ["accuracy", "roc_auc"]
        },
        "interpretation": (
            f"Overfit model báo cáo train AUC={overfit_train['roc_auc']:.4f} "
            f"nhưng test AUC chỉ {overfit_test['roc_auc']:.4f} "
            f"(gap={overfit_train['roc_auc']-overfit_test['roc_auc']:.4f})."
        ),
    }

    logger.info(
        "Overfit model — train AUC=%.4f vs test AUC=%.4f (gap=%.4f)",
        overfit_train["roc_auc"], overfit_test["roc_auc"],
        overfit_train["roc_auc"] - overfit_test["roc_auc"],
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main Simulation Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def simulate_overconfidence_bias(
    bundle: DataBundle,
    strategy: Literal[
        "inflate_probabilities",
        "narrow_prediction_intervals",
        "simulate_metric_hacking",
        "inject_overfit_confidence",
    ] = "inflate_probabilities",
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs,
) -> OverconfidenceResult:
    """
    Entry point duy nhất cho overconfidence bias simulation.

    Examples
    --------
    >>> bundle = load_dataset("synthetic_clf")
    >>> result = simulate_overconfidence_bias(bundle, strategy="inflate_probabilities")
    >>> print(result.bias_report["interpretation"])
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = bundle.X, bundle.y

    # Train simple model để lấy probabilities
    X_arr = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, y, test_size=test_size, stratify=y if y.nunique() <= 20 else None,
        random_state=random_state,
    )

    base_model = LogisticRegression(max_iter=1000, random_state=random_state)
    base_model.fit(X_tr, y_tr)
    y_proba = base_model.predict_proba(X_te)[:, 1]
    y_pred  = base_model.predict(X_te)
    y_true  = y_te.values

    result = OverconfidenceResult(original_bundle=bundle, bias_strategy=strategy)

    # ── Strategy dispatch ─────────────────────────────────────────────────────
    if strategy == "inflate_probabilities":
        inflation_factor = kwargs.get("inflation_factor", 1.5)
        y_proba_b, report = inflate_probabilities(y_true, y_proba, inflation_factor=inflation_factor)
        result.y_true = y_true
        result.y_proba_original = y_proba
        result.y_proba_biased = y_proba_b
        result.bias_report = report

    elif strategy == "narrow_prediction_intervals":
        # Dùng regression-style: y_pred là probability, y_true là labels
        squeeze = kwargs.get("squeeze_factor", 0.35)
        conf    = kwargs.get("confidence_level", 0.95)
        intervals_t, intervals_b, report = narrow_prediction_intervals(
            y_true.astype(float), y_proba, confidence_level=conf, squeeze_factor=squeeze
        )
        result.y_true = y_true
        result.y_pred = y_proba
        result.intervals_original = intervals_t
        result.intervals_biased   = intervals_b
        result.bias_report = report

    elif strategy == "simulate_metric_hacking":
        X_df = pd.DataFrame(X_tr, columns=X.columns)
        report = simulate_metric_hacking(
            X_df, pd.Series(y_tr),
            n_trials=kwargs.get("n_trials", 20),
            metric=kwargs.get("metric", "roc_auc"),
        )
        result.bias_report = report

    elif strategy == "inject_overfit_confidence":
        X_train_df = pd.DataFrame(X_tr, columns=X.columns)
        X_test_df  = pd.DataFrame(X_te, columns=X.columns)
        report = inject_overfit_confidence(
            X_train_df, pd.Series(y_tr),
            X_test_df,  pd.Series(y_te),
            **kwargs,
        )
        result.bias_report = report

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    result.bias_params = {"strategy": strategy, "test_size": test_size, **kwargs}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data.load_data import load_dataset

    bundle = load_dataset("synthetic_clf", n_samples=2_000, class_imbalance=0.35)

    strategies = [
        ("inflate_probabilities",      {"inflation_factor": 1.6}),
        ("narrow_prediction_intervals",{"squeeze_factor": 0.3}),
        ("simulate_metric_hacking",    {"n_trials": 15}),
        ("inject_overfit_confidence",  {}),
    ]

    for strat, kw in strategies:
        result = simulate_overconfidence_bias(bundle, strategy=strat, **kw)
        print(result)
        interp = result.bias_report.get("interpretation") or result.bias_report.get("strategy")
        print(f"  → {interp}")
        print()