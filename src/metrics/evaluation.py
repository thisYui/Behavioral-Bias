"""
src/metrics/evaluation.py
─────────────────────────────────────────────────────────────────────────────
Đo lường và so sánh tác động của bias lên model performance.

Module này là "thước đo trung tâm" của project — nhận output từ tất cả
các simulation modules và trả về báo cáo so sánh baseline vs biased.

Chức năng chính:
  1. Classification metrics (accuracy, f1, auc, calibration...)
  2. Regression metrics (mae, rmse, coverage...)
  3. Bias impact quantification (delta, relative change, severity)
  4. Statistical significance testing (bias có thực sự meaningful không?)
  5. Full comparison report: baseline vs biased side-by-side

Public API:
    compute_classification_metrics(y_true, y_pred, y_proba) → dict
    compute_calibration_metrics(y_true, y_proba)            → dict
    compute_bias_impact(baseline_metrics, biased_metrics)   → BiasImpact
    compare_baseline_vs_biased(baseline_result, biased_bundle) → ComparisonReport
    evaluate_all_biases(bundle, bias_configs)               → list[ComparisonReport]
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import permutation_test_score
from scipy import stats

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Severity Thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Mức độ nghiêm trọng của bias dựa trên relative change (%)
SEVERITY_THRESHOLDS = {
    "negligible": 1.0,    # < 1%   — không đáng kể
    "minor":      3.0,    # 1–3%   — nhỏ
    "moderate":   7.0,    # 3–7%   — trung bình
    "severe":     15.0,   # 7–15%  — nghiêm trọng
    # > 15%                         — critical
}


def _severity_label(relative_change_pct: float) -> str:
    abs_change = abs(relative_change_pct)
    if abs_change < SEVERITY_THRESHOLDS["negligible"]:
        return "negligible"
    elif abs_change < SEVERITY_THRESHOLDS["minor"]:
        return "minor"
    elif abs_change < SEVERITY_THRESHOLDS["moderate"]:
        return "moderate"
    elif abs_change < SEVERITY_THRESHOLDS["severe"]:
        return "severe"
    else:
        return "critical"


# ─────────────────────────────────────────────────────────────────────────────
# Output Containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BiasImpact:
    """
    Tác động của bias lên từng metric:
      - delta          : biased_metric - baseline_metric
      - relative_pct   : delta / baseline * 100
      - severity       : negligible / minor / moderate / severe / critical
      - direction      : "inflated" nếu biased > baseline, "deflated" nếu ngược lại
    """
    metric_name: str
    baseline_value: float
    biased_value: float
    delta: float
    relative_pct: float
    severity: str
    direction: str  # "inflated" | "deflated" | "unchanged"

    def __repr__(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        return (
            f"BiasImpact({self.metric_name}: "
            f"{self.baseline_value:.4f} → {self.biased_value:.4f} "
            f"[{sign}{self.delta:.4f}, {sign}{self.relative_pct:.1f}%, {self.severity}])"
        )


@dataclass
class ComparisonReport:
    """
    Báo cáo đầy đủ so sánh baseline vs biased model.
    Dùng trong visualization và experiments.
    """
    bias_type: str
    bias_strategy: str
    bias_params: dict = field(default_factory=dict)

    # Raw metrics
    baseline_metrics: dict = field(default_factory=dict)
    biased_metrics: dict = field(default_factory=dict)

    # Impact analysis
    impacts: list[BiasImpact] = field(default_factory=list)
    overall_severity: str = "negligible"
    worst_impact: BiasImpact | None = None

    # Confusion matrices
    baseline_cm: np.ndarray | None = None
    biased_cm: np.ndarray | None = None

    # Calibration
    baseline_calibration: dict = field(default_factory=dict)
    biased_calibration: dict = field(default_factory=dict)

    # Statistical test
    stat_test: dict = field(default_factory=dict)

    # Summary
    summary: str = ""

    def __repr__(self) -> str:
        return (
            f"ComparisonReport("
            f"bias='{self.bias_type}/{self.bias_strategy}', "
            f"severity='{self.overall_severity}', "
            f"worst='{self.worst_impact}')"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export impacts thành DataFrame để dễ visualize."""
        rows = []
        for imp in self.impacts:
            rows.append({
                "metric": imp.metric_name,
                "baseline": imp.baseline_value,
                "biased": imp.biased_value,
                "delta": imp.delta,
                "relative_pct": imp.relative_pct,
                "severity": imp.severity,
                "direction": imp.direction,
                "bias_type": self.bias_type,
                "bias_strategy": self.bias_strategy,
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Classification Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    prefix: str = "",
) -> dict:
    """
    Tính đầy đủ classification metrics.

    Parameters
    ----------
    prefix : str
        Prefix cho tên metric (vd: "baseline_" hoặc "biased_")

    Returns
    -------
    dict với các metrics: accuracy, f1, precision, recall, mcc,
                          roc_auc, avg_precision, log_loss (nếu có proba)
    """
    p = prefix

    metrics = {
        f"{p}accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        f"{p}f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        f"{p}precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        f"{p}recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        f"{p}mcc":       round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }

    if y_proba is not None:
        try:
            metrics[f"{p}roc_auc"]       = round(float(roc_auc_score(y_true, y_proba)), 4)
            metrics[f"{p}avg_precision"] = round(float(average_precision_score(y_true, y_proba)), 4)
            metrics[f"{p}log_loss"]      = round(float(log_loss(y_true, y_proba)), 4)
            metrics[f"{p}brier_score"]   = round(float(brier_score_loss(y_true, y_proba)), 4)
        except Exception as e:
            logger.warning("Lỗi tính proba metrics: %s", e)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 2. Calibration Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Đo lường chất lượng probability calibration.

    Metrics:
    - ECE  (Expected Calibration Error) — weighted average calibration gap
    - MCE  (Maximum Calibration Error)  — worst-case gap
    - Brier Score                       — proper scoring rule
    - Overconfidence Index              — mean predicted - mean actual (>0 = overconfident)
    - Reliability curve data            — cho visualization
    """
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy="uniform"
        )
    except ValueError:
        logger.warning("Không đủ samples cho calibration curve.")
        return {}

    # ECE: average |predicted - actual| weighted by bin size
    # calibration_curve chỉ trả về bins có data → align bằng counts_nonzero
    counts, bin_edges = np.histogram(y_proba, bins=n_bins, range=(0, 1))
    counts_nonzero = counts[counts > 0]

    if len(counts_nonzero) == 0:
        return {}

    n = min(len(prob_true), len(counts_nonzero))
    ece = float(np.sum(np.abs(prob_true[:n] - prob_pred[:n]) * counts_nonzero[:n]) / len(y_true))
    mce = float(np.max(np.abs(prob_true - prob_pred)))
    overconf_idx = float(np.mean(y_proba) - np.mean(y_true))

    return {
        "ece":                  round(ece, 4),
        "mce":                  round(mce, 4),
        "brier_score":          round(float(brier_score_loss(y_true, y_proba)), 4),
        "overconfidence_index": round(overconf_idx, 4),
        "mean_predicted_prob":  round(float(np.mean(y_proba)), 4),
        "mean_actual_rate":     round(float(np.mean(y_true)), 4),
        "reliability_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bias Impact Quantification
# ─────────────────────────────────────────────────────────────────────────────

def compute_bias_impact(
    baseline_metrics: dict,
    biased_metrics: dict,
    higher_is_better: dict[str, bool] | None = None,
) -> list[BiasImpact]:
    """
    Tính tác động của bias cho từng metric chung giữa baseline và biased.

    Parameters
    ----------
    higher_is_better : dict, optional
        Chỉ định hướng "tốt hơn" cho từng metric.
        Default: accuracy/f1/auc/precision/recall → higher is better
                 log_loss/brier/ece → lower is better

    Returns
    -------
    list[BiasImpact] sorted by |relative_pct| descending
    """
    if higher_is_better is None:
        higher_is_better = {
            "accuracy": True, "f1": True, "precision": True,
            "recall": True, "roc_auc": True, "avg_precision": True, "mcc": True,
            "log_loss": False, "brier_score": False, "ece": False, "mce": False,
        }

    impacts = []
    # Chỉ xét metrics có trong cả hai
    common_metrics = set(baseline_metrics.keys()) & set(biased_metrics.keys())

    for metric in common_metrics:
        baseline_val = float(baseline_metrics[metric])
        biased_val   = float(biased_metrics[metric])
        delta        = biased_val - baseline_val
        rel_pct      = (delta / baseline_val * 100) if baseline_val != 0 else 0.0

        # Direction: "inflated" nếu biased > baseline theo hướng tốt hơn
        hib = higher_is_better.get(metric, True)
        if abs(delta) < 1e-6:
            direction = "unchanged"
        elif (hib and delta > 0) or (not hib and delta < 0):
            direction = "inflated"   # biased metric trông tốt hơn thực tế
        else:
            direction = "deflated"   # biased metric tệ hơn baseline

        impacts.append(BiasImpact(
            metric_name=metric,
            baseline_value=round(baseline_val, 4),
            biased_value=round(biased_val, 4),
            delta=round(delta, 4),
            relative_pct=round(rel_pct, 2),
            severity=_severity_label(rel_pct),
            direction=direction,
        ))

    # Sort theo |relative_pct| để dễ thấy metric bị ảnh hưởng nhất
    impacts.sort(key=lambda x: abs(x.relative_pct), reverse=True)
    return impacts


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statistical Significance Test
# ─────────────────────────────────────────────────────────────────────────────

def test_statistical_significance(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_biased: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Kiểm tra liệu sự khác biệt giữa baseline và biased có ý nghĩa thống kê.

    Dùng McNemar's test — phù hợp để so sánh 2 classifiers trên cùng test set.

    Parameters
    ----------
    alpha : float
        Significance level (default 0.05)

    Returns
    -------
    dict với p-value, statistic, is_significant, interpretation
    """
    # McNemar's test: so sánh disagreements giữa 2 classifiers
    correct_baseline = (y_pred_baseline == y_true)
    correct_biased   = (y_pred_biased   == y_true)

    # Contingency table
    b = np.sum(correct_baseline & ~correct_biased)   # baseline đúng, biased sai
    c = np.sum(~correct_baseline & correct_biased)   # baseline sai, biased đúng

    if b + c == 0:
        return {
            "test": "mcnemar",
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "alpha": alpha,
            "interpretation": "Hai models có predictions giống nhau hoàn toàn.",
        }

    # McNemar statistic với continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(stats.chi2.sf(statistic, df=1))
    is_sig = p_value < alpha

    # Effect size (Cohen's g)
    p_hat = b / (b + c)
    cohens_g = abs(p_hat - 0.5)

    return {
        "test": "mcnemar",
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 4),
        "is_significant": is_sig,
        "alpha": alpha,
        "contingency": {"b": int(b), "c": int(c)},
        "cohens_g": round(cohens_g, 4),
        "effect_size": "small" if cohens_g < 0.1 else "medium" if cohens_g < 0.3 else "large",
        "interpretation": (
            f"{'Có' if is_sig else 'Không có'} sự khác biệt có ý nghĩa thống kê "
            f"(p={p_value:.4f}, α={alpha}). "
            f"Effect size: {cohens_g:.4f} ({'small' if cohens_g < 0.1 else 'medium' if cohens_g < 0.3 else 'large'})."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Comparison Function
# ─────────────────────────────────────────────────────────────────────────────

def compare_baseline_vs_biased(
    baseline_result,           # ModelResult từ baseline_model.py
    biased_result,             # ModelResult train trên biased data
    bias_type: str = "unknown",
    bias_strategy: str = "unknown",
    bias_params: dict | None = None,
) -> ComparisonReport:
    """
    So sánh đầy đủ baseline vs biased model — trả về ComparisonReport.

    Parameters
    ----------
    baseline_result : ModelResult
        Kết quả từ train_baseline() — model sạch
    biased_result : ModelResult
        Kết quả từ train model trên biased data
    bias_type : str
        Loại bias (vd: "confirmation_bias", "survivorship_bias")
    bias_strategy : str
        Strategy cụ thể (vd: "cherry_pick_features")

    Examples
    --------
    >>> report = compare_baseline_vs_biased(baseline, biased, "survivorship_bias", "remove_failures")
    >>> print(report)
    >>> df = report.to_dataframe()
    """
    # ── Metrics ───────────────────────────────────────────────────────────────
    baseline_metrics = compute_classification_metrics(
        baseline_result.y_true,
        baseline_result.y_pred,
        baseline_result.y_proba,
    )
    biased_metrics = compute_classification_metrics(
        biased_result.y_true,
        biased_result.y_pred,
        biased_result.y_proba,
    )

    # ── Calibration ───────────────────────────────────────────────────────────
    baseline_cal, biased_cal = {}, {}
    if baseline_result.y_proba is not None:
        baseline_cal = compute_calibration_metrics(baseline_result.y_true, baseline_result.y_proba)
    if biased_result.y_proba is not None:
        biased_cal = compute_calibration_metrics(biased_result.y_true, biased_result.y_proba)

    # ── Confusion Matrices ────────────────────────────────────────────────────
    baseline_cm = confusion_matrix(baseline_result.y_true, baseline_result.y_pred)
    biased_cm   = confusion_matrix(biased_result.y_true,   biased_result.y_pred)

    # ── Bias Impact ───────────────────────────────────────────────────────────
    impacts = compute_bias_impact(baseline_metrics, biased_metrics)
    worst   = impacts[0] if impacts else None

    # Overall severity = severity của metric bị ảnh hưởng nhất
    overall_severity = worst.severity if worst else "negligible"

    # ── Statistical Test ──────────────────────────────────────────────────────
    # Chỉ test nếu cùng size test set
    stat_test = {}
    if len(baseline_result.y_true) == len(biased_result.y_true):
        stat_test = test_statistical_significance(
            baseline_result.y_true,
            baseline_result.y_pred,
            biased_result.y_pred,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    if worst:
        sign = "+" if worst.delta >= 0 else ""
        summary = (
            f"[{bias_type}/{bias_strategy}] "
            f"Severity: {overall_severity.upper()} | "
            f"Worst metric: {worst.metric_name} "
            f"({worst.baseline_value:.4f} → {worst.biased_value:.4f}, "
            f"{sign}{worst.relative_pct:.1f}%, {worst.direction})"
        )
    else:
        summary = f"[{bias_type}/{bias_strategy}] No measurable impact."

    logger.info(summary)

    return ComparisonReport(
        bias_type=bias_type,
        bias_strategy=bias_strategy,
        bias_params=bias_params or {},
        baseline_metrics=baseline_metrics,
        biased_metrics=biased_metrics,
        impacts=impacts,
        overall_severity=overall_severity,
        worst_impact=worst,
        baseline_cm=baseline_cm,
        biased_cm=biased_cm,
        baseline_calibration=baseline_cal,
        biased_calibration=biased_cal,
        stat_test=stat_test,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Batch Evaluation — chạy tất cả bias configs cùng lúc
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_biases(
    bundle,                            # DataBundle gốc
    bias_configs: list[dict],          # list of {bias_type, strategy, kwargs}
    model_type: str = "logistic_regression",
    preprocess_kwargs: dict | None = None,
) -> list[ComparisonReport]:
    """
    Chạy toàn bộ bias simulation pipeline và trả về list ComparisonReport.

    Workflow cho mỗi bias config:
      1. Preprocess data gốc → TrainTestBundle (baseline)
      2. Apply bias simulation → BiasedBundle
      3. Preprocess biased data → TrainTestBundle (biased)
      4. Train baseline model và biased model
      5. compare_baseline_vs_biased → ComparisonReport

    Parameters
    ----------
    bias_configs : list[dict]
        Mỗi dict có keys: bias_type, strategy, kwargs (optional)
        Ví dụ:
        [
            {"bias_type": "confirmation",  "strategy": "cherry_pick_features", "kwargs": {"keep_top_n": 5}},
            {"bias_type": "survivorship",  "strategy": "remove_failures",      "kwargs": {"removal_rate": 0.85}},
            {"bias_type": "overconfidence","strategy": "inflate_probabilities", "kwargs": {}},
        ]

    Returns
    -------
    list[ComparisonReport] — sorted by overall severity
    """
    from src.data.preprocess import preprocess
    from src.models.baseline_model import train_baseline
    from src.simulation.confirmation_bias import simulate_confirmation_bias
    from src.simulation.survivorship_bias import simulate_survivorship_bias
    from src.simulation.overconfidence_bias import simulate_overconfidence_bias

    pp_kwargs = preprocess_kwargs or {}

    # Train baseline một lần duy nhất
    tt_baseline = preprocess(bundle, **pp_kwargs)
    baseline_result = train_baseline(tt_baseline, model_type=model_type, run_cv=False)

    reports = []
    bias_simulators = {
        "confirmation":   simulate_confirmation_bias,
        "survivorship":   simulate_survivorship_bias,
    }

    for cfg in bias_configs:
        bias_type = cfg["bias_type"]
        strategy  = cfg["strategy"]
        kwargs    = cfg.get("kwargs", {})

        logger.info("Evaluating bias: %s / %s", bias_type, strategy)

        try:
            if bias_type == "overconfidence":
                # Overconfidence bias: không cần re-train — dùng direct output
                from src.simulation.overconfidence_bias import simulate_overconfidence_bias
                oc_result = simulate_overconfidence_bias(bundle, strategy=strategy, **kwargs)

                # Wrap vào fake ModelResult để reuse compare function
                from src.models.baseline_model import ModelResult
                from sklearn.pipeline import Pipeline

                dummy_pipe = baseline_result.pipeline  # reuse pipeline object

                if oc_result.y_proba_biased is not None:
                    y_pred_b = (oc_result.y_proba_biased >= 0.5).astype(int)
                    biased_mr = ModelResult(
                        model_type=model_type,
                        pipeline=dummy_pipe,
                        y_true=oc_result.y_true,
                        y_pred=y_pred_b,
                        y_proba=oc_result.y_proba_biased,
                        label=f"Biased ({strategy})",
                        train_size=baseline_result.train_size,
                        test_size=len(oc_result.y_true),
                        n_features=baseline_result.n_features,
                    )
                    # Baseline cũng dùng cùng test subset
                    baseline_mr_sub = ModelResult(
                        model_type=model_type,
                        pipeline=dummy_pipe,
                        y_true=oc_result.y_true,
                        y_pred=(oc_result.y_proba_original >= 0.5).astype(int),
                        y_proba=oc_result.y_proba_original,
                        label="Baseline",
                        train_size=baseline_result.train_size,
                        test_size=len(oc_result.y_true),
                        n_features=baseline_result.n_features,
                    )
                    report = compare_baseline_vs_biased(
                        baseline_mr_sub, biased_mr,
                        bias_type=bias_type, bias_strategy=strategy, bias_params=kwargs,
                    )
                else:
                    logger.warning("Overconfidence strategy '%s' không có y_proba_biased — skip.", strategy)
                    continue

            elif bias_type in bias_simulators:
                # Confirmation / Survivorship: re-train model trên biased data
                biased_bundle_obj = bias_simulators[bias_type](bundle, strategy=strategy, **kwargs)
                tt_biased = preprocess(biased_bundle_obj.biased, **pp_kwargs)
                biased_result = train_baseline(tt_biased, model_type=model_type, run_cv=False,
                                               label=f"Biased ({strategy})")
                report = compare_baseline_vs_biased(
                    baseline_result, biased_result,
                    bias_type=bias_type, bias_strategy=strategy, bias_params=kwargs,
                )
            else:
                logger.warning("Không biết bias_type='%s' — skip.", bias_type)
                continue

            reports.append(report)

        except Exception as e:
            logger.error("Lỗi khi evaluate %s/%s: %s", bias_type, strategy, e, exc_info=True)

    # Sort theo severity
    severity_order = {"critical": 0, "severe": 1, "moderate": 2, "minor": 3, "negligible": 4}
    reports.sort(key=lambda r: severity_order.get(r.overall_severity, 5))

    logger.info(
        "Evaluated %d bias configs — severity distribution: %s",
        len(reports),
        {r.bias_strategy: r.overall_severity for r in reports},
    )
    return reports


# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_table(reports: list[ComparisonReport]) -> pd.DataFrame:
    """
    Tổng hợp tất cả ComparisonReport thành DataFrame dễ đọc.
    Dùng trực tiếp trong notebooks và visualization.

    Columns: bias_type, strategy, severity, metric, baseline, biased, delta, rel_pct, direction
    """
    rows = []
    for report in reports:
        for impact in report.impacts:
            rows.append({
                "bias_type":    report.bias_type,
                "strategy":     report.bias_strategy,
                "severity":     report.overall_severity,
                "metric":       impact.metric_name,
                "baseline":     impact.baseline_value,
                "biased":       impact.biased_value,
                "delta":        impact.delta,
                "rel_pct":      impact.relative_pct,
                "direction":    impact.direction,
                "stat_sig":     report.stat_test.get("is_significant", None),
                "p_value":      report.stat_test.get("p_value", None),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        severity_cat = pd.CategoricalDtype(
            ["critical", "severe", "moderate", "minor", "negligible"], ordered=True
        )
        df["severity"] = df["severity"].astype(severity_cat)
        df = df.sort_values(["severity", "rel_pct"], ascending=[True, False])
    return df


def print_summary(reports: list[ComparisonReport], top_n: int = 5) -> None:
    """In summary table ra console — dùng trong experiments."""
    df = build_summary_table(reports)
    if df.empty:
        print("Không có reports.")
        return

    print("\n" + "═" * 80)
    print("  BIAS IMPACT SUMMARY")
    print("═" * 80)

    for report in reports:
        print(f"\n  {report.summary}")
        if report.stat_test:
            print(f"  Statistical test: {report.stat_test.get('interpretation', '')}")

    print("\n" + "─" * 80)
    print("  TOP IMPACTED METRICS (all biases combined)")
    print("─" * 80)

    top = df[df["direction"] == "inflated"].head(top_n)
    if top.empty:
        top = df.head(top_n)

    for _, row in top.iterrows():
        sign = "+" if row["delta"] >= 0 else ""
        print(
            f"  [{row['severity']:10s}] {row['bias_type']:20s} | "
            f"{row['metric']:15s}: {row['baseline']:.4f} → {row['biased']:.4f} "
            f"({sign}{row['rel_pct']:.1f}%)"
        )
    print("═" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data.load_data import load_dataset
    from src.data.preprocess import preprocess
    from src.models.baseline_model import train_baseline
    from src.simulation.survivorship_bias import simulate_survivorship_bias

    bundle = load_dataset("synthetic_clf", n_samples=2_000, class_imbalance=0.3)

    # Single comparison
    tt = preprocess(bundle)
    baseline = train_baseline(tt, model_type="logistic_regression", run_cv=False)

    biased_bundle = simulate_survivorship_bias(bundle, strategy="remove_failures", removal_rate=0.85)
    tt_b = preprocess(biased_bundle.biased)
    biased = train_baseline(tt_b, model_type="logistic_regression", run_cv=False,
                            label="Biased (survivorship)")

    report = compare_baseline_vs_biased(baseline, biased,
                                        bias_type="survivorship_bias",
                                        bias_strategy="remove_failures")
    print(report)
    for imp in report.impacts[:4]:
        print(" ", imp)

    # Batch evaluation
    print("\n--- Batch Evaluation ---")
    bias_configs = [
        {"bias_type": "confirmation",  "strategy": "cherry_pick_features",     "kwargs": {"keep_top_n": 5}},
        {"bias_type": "survivorship",  "strategy": "remove_failures",          "kwargs": {"removal_rate": 0.85}},
        {"bias_type": "overconfidence","strategy": "inflate_probabilities",     "kwargs": {"inflation_factor": 1.5}},
    ]
    reports = evaluate_all_biases(bundle, bias_configs)
    print_summary(reports)

    df = build_summary_table(reports)
    print(df[["bias_type", "strategy", "metric", "baseline", "biased", "rel_pct", "severity"]].head(10))