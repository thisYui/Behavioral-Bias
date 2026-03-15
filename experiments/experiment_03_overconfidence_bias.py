"""
experiments/experiment_03_overconfidence_bias.py
─────────────────────────────────────────────────────────────────────────────
Experiment 03: Overconfidence Bias

Mục tiêu:
  Chứng minh rằng overconfidence bias — model / data scientist quá tự tin
  vào predictions — làm cho reported performance cao hơn thực tế,
  và model fail khi gặp edge cases và out-of-distribution data.

Các scenarios được test:
  A. Probability Inflation   — model nói 90% confident, thực tế chỉ 60%
  B. Narrow Intervals        — báo cáo 95% CI nhưng thực tế chỉ cover 60%
  C. Metric Hacking          — chạy 20 trials, chỉ report best run
  D. Overfit Confidence      — báo cáo train AUC, bỏ qua test AUC
  E. Inflation Strength Sweep— test inflation_factor từ 1.0 đến 2.5

Kết quả được lưu vào:
  experiments/results/experiment_03/
    ├── metrics_summary.csv
    ├── 01_baseline_calibration.png
    ├── 02_calibration_comparison_scenario_a.png
    ├── 03_metric_hacking_scenario_c.png
    ├── 04_overfit_gap_scenario_d.png
    ├── 05_inflation_sweep.png
    ├── 06_severity_heatmap.png
    ├── 07_bias_dashboard.png
    └── experiment_03_report.txt
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.load_data import load_dataset
from src.data.preprocess import preprocess
from src.models.baseline_model import train_baseline, build_model, train_model, ModelResult
from src.simulation.overconfidence_bias import (
    simulate_overconfidence_bias,
    inflate_probabilities,
    compute_calibration_metrics,
)
from src.metrics.evaluation import (
    compare_baseline_vs_biased,
    compute_classification_metrics,
    compute_calibration_metrics as eval_calibration,
    build_summary_table,
    print_summary,
    ComparisonReport,
)
from src.visualization.plots import (
    set_style,
    plot_metric_comparison,
    PALETTE,
)
from src.visualization.bias_charts import (
    plot_bias_impact_bars,
    plot_calibration_comparison,
    plot_severity_heatmap,
    plot_metric_hacking_distribution,
    plot_bias_dashboard,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_ID   = "experiment_03"
EXPERIMENT_NAME = "Overconfidence Bias"
RESULTS_DIR     = Path("experiments/results") / EXPERIMENT_ID
RANDOM_STATE    = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
set_style()

SCENARIOS = [
    {
        "id":          "A",
        "name":        "Probability Inflation",
        "strategy":    "inflate_probabilities",
        "kwargs":      {"inflation_factor": 1.6},
        "description": "Inflate predicted probabilities 1.6x về phía extremes (0/1).",
    },
    {
        "id":          "B",
        "name":        "Narrow Prediction Intervals",
        "strategy":    "narrow_prediction_intervals",
        "kwargs":      {"squeeze_factor": 0.3, "confidence_level": 0.95},
        "description": "Báo cáo 95% CI nhưng squeeze interval xuống 30% width thật.",
    },
    {
        "id":          "C",
        "name":        "Metric Hacking",
        "strategy":    "simulate_metric_hacking",
        "kwargs":      {"n_trials": 25, "metric": "roc_auc"},
        "description": "Chạy 25 trials với seeds khác nhau, chỉ báo cáo trial tốt nhất.",
    },
    {
        "id":          "D",
        "name":        "Overfit Confidence",
        "strategy":    "inject_overfit_confidence",
        "kwargs":      {"overfit_n_estimators": 500, "overfit_max_depth": None},
        "description": "Dùng fully overfit RF, báo cáo train AUC thay vì test AUC.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Results dir: %s", RESULTS_DIR.resolve())


def save_fig(fig: plt.Figure, filename: str) -> Path:
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _make_fake_report(
    strategy: str,
    honest_metrics: dict,
    reported_metrics: dict,
    bias_data: dict | None = None,
) -> ComparisonReport:
    """
    Tạo ComparisonReport từ honest vs reported metrics
    (dùng cho overconfidence scenarios không re-train model).
    """
    from src.metrics.evaluation import compute_bias_impact, _severity_label, BiasImpact

    impacts = compute_bias_impact(honest_metrics, reported_metrics)
    worst   = impacts[0] if impacts else None
    overall = worst.severity if worst else "negligible"

    return ComparisonReport(
        bias_type="overconfidence_bias",
        bias_strategy=strategy,
        baseline_metrics=honest_metrics,
        biased_metrics=reported_metrics,
        impacts=impacts,
        overall_severity=overall,
        worst_impact=worst,
        summary=(
            f"[overconfidence_bias/{strategy}] "
            f"Severity: {overall.upper()} | "
            + (f"Worst: {worst.metric_name} {worst.relative_pct:+.1f}%" if worst else "")
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load & Baseline Calibration
# ─────────────────────────────────────────────────────────────────────────────

def step_01_load_and_baseline(
    n_samples: int = 3_000,
    class_imbalance: float = 0.3,
    model_type: str = "logistic_regression",
):
    logger.info("=" * 60)
    logger.info("STEP 1: Load data & establish baseline calibration")
    logger.info("=" * 60)

    bundle   = load_dataset("synthetic_clf", n_samples=n_samples,
                             class_imbalance=class_imbalance, random_state=RANDOM_STATE)
    tt       = preprocess(bundle, random_state=RANDOM_STATE)
    baseline = train_baseline(tt, model_type=model_type, label="Baseline", run_cv=True)

    logger.info("Baseline metrics: %s", baseline.test_metrics)

    # Baseline calibration plot
    if baseline.y_proba is not None:
        cal = eval_calibration(baseline.y_true, baseline.y_proba)
        logger.info("Baseline calibration — ECE=%.4f, OCI=%+.4f",
                    cal.get("ece", 0), cal.get("overconfidence_index", 0))

        # Quick reliability diagram
        fig, ax = plt.subplots(figsize=(6, 6))
        rc = cal.get("reliability_curve", {})
        if rc:
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect")
            ax.plot(rc["prob_pred"], rc["prob_true"], "o-",
                    color=PALETTE["baseline"], lw=2, label="Baseline LR")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title(f"Baseline Calibration\nECE={cal['ece']:.4f}, "
                         f"OCI={cal['overconfidence_index']:+.4f}")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            save_fig(fig, "01_baseline_calibration.png")

    return bundle, tt, baseline


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def step_02_run_scenarios(
    bundle,
    baseline,
) -> list[dict]:
    logger.info("=" * 60)
    logger.info("STEP 2: Run overconfidence bias scenarios")
    logger.info("=" * 60)

    results = []

    for scenario in SCENARIOS:
        logger.info("\n--- Scenario %s: %s ---", scenario["id"], scenario["name"])
        logger.info("Description: %s", scenario["description"])

        oc_result = simulate_overconfidence_bias(
            bundle, strategy=scenario["strategy"],
            random_state=RANDOM_STATE,
            **scenario["kwargs"],
        )
        logger.info("Strategy '%s' complete.", scenario["strategy"])

        br = oc_result.bias_report

        # ── Build ComparisonReport per strategy ─────────────────────────────
        if scenario["strategy"] == "inflate_probabilities":
            # Metrics từ honest probabilities vs inflated
            orig_m = br.get("original_calibration", {})
            bias_m = br.get("biased_calibration",   {})
            honest   = {k: orig_m[k] for k in ["ece", "brier_score"] if k in orig_m}
            reported = {k: bias_m[k] for k in ["ece", "brier_score"] if k in bias_m}
            # Thêm classification metrics
            if oc_result.y_true is not None and oc_result.y_proba_original is not None:
                y_pred_orig = (oc_result.y_proba_original >= 0.5).astype(int)
                y_pred_bias = (oc_result.y_proba_biased  >= 0.5).astype(int)
                honest.update(compute_classification_metrics(oc_result.y_true, y_pred_orig,
                                                             oc_result.y_proba_original))
                reported.update(compute_classification_metrics(oc_result.y_true, y_pred_bias,
                                                               oc_result.y_proba_biased))
            report = _make_fake_report(scenario["strategy"], honest, reported, br)

        elif scenario["strategy"] == "narrow_prediction_intervals":
            honest   = {"coverage": br.get("true_coverage",   0),
                        "interval_width": br.get("mean_interval_width_true",  0)}
            reported = {"coverage": br.get("biased_coverage", 0),
                        "interval_width": br.get("mean_interval_width_biased", 0)}
            report = _make_fake_report(scenario["strategy"], honest, reported, br)

        elif scenario["strategy"] == "simulate_metric_hacking":
            metric = scenario["kwargs"].get("metric", "roc_auc")
            honest   = {metric: br.get("honest_score",   0)}
            reported = {metric: br.get("reported_score", 0)}
            report = _make_fake_report(scenario["strategy"], honest, reported, br)

        elif scenario["strategy"] == "inject_overfit_confidence":
            baseline_m = br.get("baseline",    {})
            overfit_m  = br.get("overfit_model", {})
            honest   = overfit_m.get("test",  {})   # real test performance
            reported = overfit_m.get("train", {})   # what they claim (train)
            report = _make_fake_report(scenario["strategy"], honest, reported, br)

        else:
            report = _make_fake_report(scenario["strategy"], {}, {}, br)

        logger.info("Report: %s", report.summary)

        results.append({
            "scenario":  scenario,
            "oc_result": oc_result,
            "report":    report,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Calibration Comparison (Scenario A)
# ─────────────────────────────────────────────────────────────────────────────

def step_03_calibration_charts(results: list[dict]) -> None:
    logger.info("=" * 60)
    logger.info("STEP 3: Calibration comparison charts")
    logger.info("=" * 60)

    scenario_a = next(
        (r for r in results if r["scenario"]["strategy"] == "inflate_probabilities"), None
    )
    if scenario_a is None:
        return

    oc  = scenario_a["oc_result"]
    br  = oc.bias_report

    if oc.y_true is None or oc.y_proba_original is None:
        return

    # Compute calibration for both
    orig_cal  = eval_calibration(oc.y_true, oc.y_proba_original)
    biased_cal = eval_calibration(oc.y_true, oc.y_proba_biased)

    # Fake report to reuse plot_calibration_comparison
    fake_report          = scenario_a["report"]
    fake_report.baseline_calibration = orig_cal
    fake_report.biased_calibration   = biased_cal

    fig, _ = plot_calibration_comparison(
        fake_report,
        title="Calibration: Honest Probabilities vs Inflated (Overconfidence)",
    )
    save_fig(fig, "02_calibration_comparison_scenario_a.png")

    logger.info(
        "Calibration — Honest ECE=%.4f, Inflated ECE=%.4f",
        orig_cal.get("ece", 0), biased_cal.get("ece", 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Metric Hacking Chart (Scenario C)
# ─────────────────────────────────────────────────────────────────────────────

def step_04_metric_hacking_chart(results: list[dict]) -> None:
    logger.info("=" * 60)
    logger.info("STEP 4: Metric hacking distribution chart")
    logger.info("=" * 60)

    scenario_c = next(
        (r for r in results if r["scenario"]["strategy"] == "simulate_metric_hacking"), None
    )
    if scenario_c is None:
        return

    br = scenario_c["oc_result"].bias_report
    fig, _ = plot_metric_hacking_distribution(
        br,
        title=f"Metric Hacking: {br.get('n_trials', '?')} Trials, "
              f"Only Best Reported",
    )
    save_fig(fig, "03_metric_hacking_scenario_c.png")

    logger.info(
        "Metric hacking — honest=%.4f, reported=%.4f, inflation=+%.2f%%",
        br.get("honest_score", 0),
        br.get("reported_score", 0),
        br.get("inflation_pct", 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Overfit Gap Chart (Scenario D)
# ─────────────────────────────────────────────────────────────────────────────

def step_05_overfit_gap_chart(results: list[dict]) -> None:
    logger.info("=" * 60)
    logger.info("STEP 5: Overfit confidence gap chart")
    logger.info("=" * 60)

    scenario_d = next(
        (r for r in results if r["scenario"]["strategy"] == "inject_overfit_confidence"), None
    )
    if scenario_d is None:
        return

    br = scenario_d["oc_result"].bias_report

    baseline_m = br.get("baseline",    {})
    overfit_m  = br.get("overfit_model", {})
    if not baseline_m or not overfit_m:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics   = [m for m in ["accuracy", "f1", "roc_auc"] if m in baseline_m.get("train", {})]

    # Left: train vs test for overfit model
    ax = axes[0]
    x  = np.arange(len(metrics))
    w  = 0.25

    train_vals   = [overfit_m["train"].get(m, 0)   for m in metrics]
    test_vals    = [overfit_m["test"].get(m, 0)    for m in metrics]
    bl_test_vals = [baseline_m["test"].get(m, 0)   for m in metrics]

    bars_train = ax.bar(x - w, train_vals, w,
                        color=PALETTE["biased"], alpha=0.85,
                        label="Overfit — Train (REPORTED)")
    bars_test  = ax.bar(x,     test_vals,  w,
                        color=PALETTE["warning"], alpha=0.85,
                        label="Overfit — Test (real)")
    bars_bl    = ax.bar(x + w, bl_test_vals, w,
                        color=PALETTE["baseline"], alpha=0.85,
                        label="Baseline — Test")

    for bar, v in zip(list(bars_train) + list(bars_test) + list(bars_bl),
                      train_vals + test_vals + bl_test_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Overfit Confidence: Train vs Test\n(Reported vs Real Performance)")
    ax.legend(fontsize=8)

    # Right: generalization gap per metric
    ax2 = axes[1]
    gaps = br.get("generalization_gap", {})
    if gaps:
        gap_metrics = list(gaps.keys())
        gap_vals    = list(gaps.values())
        colors_gap  = [PALETTE["biased"] if g > 0 else PALETTE["positive"] for g in gap_vals]
        ax2.bar(range(len(gap_metrics)), gap_vals, color=colors_gap, alpha=0.85,
                edgecolor="white")
        ax2.axhline(0, color=PALETTE["neutral"], lw=1, linestyle="--")
        for i, v in enumerate(gap_vals):
            ax2.text(i, v + (0.001 if v >= 0 else -0.004),
                     f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                     fontsize=9, fontweight="semibold",
                     color=PALETTE["biased"] if v > 0 else PALETTE["positive"])
        ax2.set_xticks(range(len(gap_metrics)))
        ax2.set_xticklabels(gap_metrics, rotation=15)
        ax2.set_ylabel("Generalization Gap (Train − Test)")
        ax2.set_title("Overfitting Gap per Metric\n(+ve = Overconfident)")

    fig.suptitle("Overfit Confidence Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "04_overfit_gap_scenario_d.png")

    interp = br.get("interpretation", "")
    logger.info("Overfit insight: %s", interp)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Inflation Strength Sweep
# ─────────────────────────────────────────────────────────────────────────────

def step_06_inflation_sweep(
    bundle,
    inflation_factors: list[float] | None = None,
) -> None:
    logger.info("=" * 60)
    logger.info("STEP 6: Inflation factor sensitivity sweep")
    logger.info("=" * 60)

    if inflation_factors is None:
        inflation_factors = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]

    records = []

    # Train one model to get base probabilities
    tt = preprocess(bundle, random_state=RANDOM_STATE)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X_arr = StandardScaler().fit_transform(bundle.X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, bundle.y, test_size=0.2,
        stratify=bundle.y, random_state=RANDOM_STATE,
    )
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    y_proba_base = model.predict_proba(X_te)[:, 1]
    y_true = y_te.values

    for factor in inflation_factors:
        y_proba_inf, _ = inflate_probabilities(y_true, y_proba_base, inflation_factor=factor)
        cal = compute_calibration_metrics(y_true, y_proba_inf)
        records.append({
            "inflation_factor": factor,
            "ece":              cal.get("ece", 0),
            "brier_score":      cal.get("brier_score", 0),
            "overconf_index":   cal.get("overconfidence_index", 0),
            "mean_pred_prob":   cal.get("mean_predicted_prob", 0),
        })
        logger.info("  factor=%.1f → ECE=%.4f, OCI=%+.4f",
                    factor, cal.get("ece", 0), cal.get("overconfidence_index", 0))

    df_sweep = pd.DataFrame(records)
    df_sweep.to_csv(RESULTS_DIR / "inflation_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df_sweep["inflation_factor"], df_sweep["ece"],
                 "o-", color=PALETTE["biased"], lw=2, label="ECE ↑ = worse calibration")
    axes[0].plot(df_sweep["inflation_factor"], df_sweep["brier_score"],
                 "s-", color=PALETTE["warning"], lw=2, label="Brier Score ↑ = worse")
    axes[0].axvline(1.0, color=PALETTE["neutral"], lw=1, linestyle="--",
                    alpha=0.5, label="No inflation (x1.0)")
    axes[0].set_xlabel("Inflation Factor")
    axes[0].set_ylabel("Calibration Error")
    axes[0].set_title("Calibration Degradation vs Inflation Factor")
    axes[0].legend(fontsize=8)

    axes[1].plot(df_sweep["inflation_factor"], df_sweep["overconf_index"],
                 "^-", color=PALETTE["biased"], lw=2,
                 label="Overconfidence Index (mean pred − mean actual)")
    axes[1].axhline(0, color=PALETTE["neutral"], lw=1, linestyle="--")
    axes[1].set_xlabel("Inflation Factor")
    axes[1].set_ylabel("Overconfidence Index")
    axes[1].set_title("Overconfidence Index vs Inflation Factor\n(>0 = Model is overconfident)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Probability Inflation Sensitivity Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "05_inflation_sweep.png")


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Aggregate Analysis
# ─────────────────────────────────────────────────────────────────────────────

def step_07_aggregate_analysis(results: list[dict]) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 7: Aggregate analysis")
    logger.info("=" * 60)

    reports = [r["report"] for r in results]

    fig, _ = plot_severity_heatmap(
        reports,
        title="Overconfidence Bias — Severity Heatmap",
        metrics=["accuracy", "f1", "roc_auc", "ece", "brier_score", "coverage"],
    )
    save_fig(fig, "06_severity_heatmap.png")

    df = build_summary_table(reports)
    df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    print_summary(reports)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def step_08_dashboard(results: list[dict]) -> None:
    reports = [r["report"] for r in results]
    fig = plot_bias_dashboard(
        reports,
        title=f"Experiment 03: {EXPERIMENT_NAME} — Impact Dashboard",
        metrics=["accuracy", "f1", "roc_auc", "ece", "brier_score"],
    )
    save_fig(fig, "07_bias_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Write Report
# ─────────────────────────────────────────────────────────────────────────────

def step_09_write_report(
    results: list[dict],
    df_summary: pd.DataFrame,
    baseline,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        f"  EXPERIMENT 03: {EXPERIMENT_NAME.upper()}",
        f"  Generated: {ts}",
        "=" * 70,
        "",
        "KEY INSIGHT",
        "-" * 40,
        "  Overconfidence bias causes reported metrics to significantly exceed",
        "  actual real-world performance. A model claiming 95% confidence",
        "  may only be correct 60% of the time. Metric hacking amplifies",
        "  this further by selecting the best random outcome from many trials.",
        "",
        "BASELINE MODEL",
        "-" * 40,
        f"  Model type   : {baseline.model_type}",
        f"  Test metrics : {baseline.test_metrics}",
        "",
        "SCENARIO RESULTS",
        "-" * 40,
    ]

    for r in results:
        sc     = r["scenario"]
        report = r["report"]
        br     = r["oc_result"].bias_report
        wi     = report.worst_impact

        lines += [
            "",
            f"  Scenario {sc['id']}: {sc['name']}",
            f"  Strategy      : {sc['strategy']}",
            f"  Description   : {sc['description']}",
            f"  Severity      : {report.overall_severity.upper()}",
            f"  Summary       : {report.summary}",
        ]

        if sc["strategy"] == "simulate_metric_hacking":
            lines.append(
                f"  Honest score  : {br.get('honest_score', 'N/A'):.4f}  →  "
                f"Reported: {br.get('reported_score', 'N/A'):.4f}  "
                f"(+{br.get('inflation_pct', 0):.1f}%)"
            )
        elif sc["strategy"] == "inject_overfit_confidence":
            interp = br.get("interpretation", "")
            lines.append(f"  Insight       : {interp}")
        elif sc["strategy"] == "narrow_prediction_intervals":
            lines.append(
                f"  Claimed CI    : {br.get('claimed_confidence_level', 0):.0%}  →  "
                f"Actual coverage: {br.get('biased_coverage', 0):.1%}  "
                f"(gap={br.get('coverage_gap', 0):.1%})"
            )

        if wi:
            lines.append(
                f"  Worst metric  : {wi.metric_name} "
                f"{wi.baseline_value:.4f} → {wi.biased_value:.4f} "
                f"({wi.relative_pct:+.1f}%)"
            )

    lines += ["", "=" * 70, ""]
    (RESULTS_DIR / "experiment_03_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    logger.info("Report written.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    n_samples: int = 3_000,
    class_imbalance: float = 0.3,
    model_type: str = "logistic_regression",
) -> dict:
    """
    Run toàn bộ Experiment 03.

    Returns dict chứa bundle, baseline, results, df_summary.
    """
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  EXPERIMENT 03: OVERCONFIDENCE BIAS      ║")
    logger.info("╚══════════════════════════════════════════╝")

    setup()

    bundle, tt, baseline = step_01_load_and_baseline(
        n_samples=n_samples,
        class_imbalance=class_imbalance,
        model_type=model_type,
    )
    results = step_02_run_scenarios(bundle, baseline)
    step_03_calibration_charts(results)
    step_04_metric_hacking_chart(results)
    step_05_overfit_gap_chart(results)
    step_06_inflation_sweep(bundle)
    df_summary = step_07_aggregate_analysis(results)
    step_08_dashboard(results)
    step_09_write_report(results, df_summary, baseline)

    logger.info("✓ Experiment 03 complete. Results: %s", RESULTS_DIR.resolve())

    return {
        "bundle":     bundle,
        "baseline":   baseline,
        "results":    results,
        "df_summary": df_summary,
    }


if __name__ == "__main__":
    run()