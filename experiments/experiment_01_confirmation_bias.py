"""
experiments/experiment_01_confirmation_bias.py
─────────────────────────────────────────────────────────────────────────────
Experiment 01: Confirmation Bias

Mục tiêu:
  Chứng minh rằng confirmation bias trong data selection và feature engineering
  dẫn đến model performance bị inflate — model trông tốt hơn thực tế.

Các scenarios được test:
  A. Cherry-pick features theo F-score (chỉ giữ top N features)
  B. Filter confirming samples (loại bỏ samples không khớp hypothesis)
  C. Biased feature correlation (chỉ giữ features có positive correlation)
  D. Select confirming subgroup (chỉ evaluate trên subgroup "đẹp")

Kết quả được lưu vào:
  experiments/results/experiment_01/
    ├── metrics_summary.csv
    ├── bias_impact_bars_[strategy].png
    ├── severity_heatmap.png
    └── experiment_01_report.txt
"""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ── Project imports ────────────────────────────────────────────────────────────
from src.data.load_data import load_dataset
from src.data.preprocess import preprocess
from src.models.baseline_model import train_baseline, train_all_baselines
from src.simulation.confirmation_bias import simulate_confirmation_bias
from src.metrics.evaluation import (
    compare_baseline_vs_biased,
    evaluate_all_biases,
    build_summary_table,
    print_summary,
)
from src.visualization.plots import (
    set_style,
    plot_metric_comparison,
    plot_class_distribution,
    plot_feature_importance,
)
from src.visualization.bias_charts import (
    plot_bias_impact_bars,
    plot_severity_heatmap,
    plot_bias_dashboard,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_ID   = "experiment_01"
EXPERIMENT_NAME = "Confirmation Bias"
RESULTS_DIR     = Path("experiments/results") / EXPERIMENT_ID
RANDOM_STATE    = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

set_style()

# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id":          "A",
        "name":        "Feature Cherry-Picking",
        "bias_type":   "confirmation",
        "strategy":    "cherry_pick_features",
        "kwargs":      {"keep_top_n": 5, "task": "classification"},
        "description": "Chỉ giữ 5 features có F-score cao nhất, thêm noise features.",
    },
    {
        "id":          "B",
        "name":        "Sample Filtering",
        "bias_type":   "confirmation",
        "strategy":    "filter_confirming_samples",
        "kwargs":      {"remove_class": 0, "remove_fraction": 0.5},
        "description": "Loại 50% samples class 0 — giả vờ chúng là 'outliers'.",
    },
    {
        "id":          "C",
        "name":        "Correlation Direction Bias",
        "bias_type":   "confirmation",
        "strategy":    "biased_feature_correlation",
        "kwargs":      {"min_corr": 0.05, "direction": "positive"},
        "description": "Chỉ giữ features có positive correlation với target.",
    },
    {
        "id":          "D",
        "name":        "Subgroup Selection",
        "bias_type":   "confirmation",
        "strategy":    "select_confirming_subgroup",
        "kwargs":      {"percentile_range": (40.0, 100.0)},
        "description": "Chỉ evaluate trên subgroup có feature_00 > percentile 40.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Setup
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


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load & Baseline
# ─────────────────────────────────────────────────────────────────────────────

def step_01_load_and_baseline(
    n_samples: int = 3_000,
    class_imbalance: float = 0.35,
    model_type: str = "logistic_regression",
):
    logger.info("=" * 60)
    logger.info("STEP 1: Load data & train baseline")
    logger.info("=" * 60)

    bundle = load_dataset(
        "synthetic_clf",
        n_samples=n_samples,
        class_imbalance=class_imbalance,
        random_state=RANDOM_STATE,
    )
    logger.info("Dataset: %s", bundle)

    tt = preprocess(bundle, random_state=RANDOM_STATE)
    baseline = train_baseline(
        tt, model_type=model_type,
        label="Baseline (Clean)", run_cv=True,
    )
    logger.info("Baseline: %s", baseline.test_metrics)

    # Plot: class distribution
    fig, _ = plot_class_distribution(
        {"Original": bundle.y},
        title="Dataset Class Distribution",
    )
    save_fig(fig, "01_class_distribution.png")

    return bundle, tt, baseline


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run All Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def step_02_run_scenarios(
    bundle,
    baseline,
    model_type: str = "logistic_regression",
) -> list[dict]:
    logger.info("=" * 60)
    logger.info("STEP 2: Run confirmation bias scenarios")
    logger.info("=" * 60)

    results = []

    for scenario in SCENARIOS:
        logger.info(
            "\n--- Scenario %s: %s ---", scenario["id"], scenario["name"]
        )
        logger.info("Description: %s", scenario["description"])

        # Apply confirmation bias
        kwargs = scenario["kwargs"].copy()

        # Scenario D cần feature name cụ thể
        if scenario["strategy"] == "select_confirming_subgroup":
            kwargs["feature"] = bundle.X.columns[0]

        biased_bundle = simulate_confirmation_bias(
            bundle, strategy=scenario["strategy"], **kwargs
        )
        logger.info("BiasedBundle: %s", biased_bundle)
        logger.info("Bias report keys: %s", list(biased_bundle.bias_report.keys()))

        # Train model trên biased data
        tt_biased  = preprocess(biased_bundle.biased, random_state=RANDOM_STATE)
        biased_model = train_baseline(
            tt_biased, model_type=model_type,
            label=f"Biased ({scenario['name']})", run_cv=False,
        )

        # Compare
        report = compare_baseline_vs_biased(
            baseline, biased_model,
            bias_type="confirmation_bias",
            bias_strategy=scenario["strategy"],
            bias_params=kwargs,
        )
        logger.info("Report: %s", report.summary)

        # Plot: bias impact bars
        fig, _ = plot_bias_impact_bars(
            report,
            title=f"Scenario {scenario['id']}: {scenario['name']}",
        )
        save_fig(fig, f"02_impact_bars_scenario_{scenario['id'].lower()}.png")

        # Plot: class distribution shift (nếu sample size thay đổi)
        if len(biased_bundle.biased.y) != len(bundle.y):
            fig, _ = plot_class_distribution(
                {
                    "Original": bundle.y,
                    f"After {scenario['name']}": biased_bundle.biased.y,
                },
                title=f"Class Distribution Shift — {scenario['name']}",
            )
            save_fig(fig, f"02_class_dist_scenario_{scenario['id'].lower()}.png")

        results.append({
            "scenario": scenario,
            "biased_bundle": biased_bundle,
            "biased_model": biased_model,
            "report": report,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Feature Importance Analysis
# ─────────────────────────────────────────────────────────────────────────────

def step_03_feature_importance(results: list[dict], bundle) -> None:
    logger.info("=" * 60)
    logger.info("STEP 3: Feature importance analysis")
    logger.info("=" * 60)

    # Scenario A: cherry-pick features — show injected noise features
    scenario_a = next(
        (r for r in results if r["scenario"]["strategy"] == "cherry_pick_features"), None
    )
    if scenario_a is None:
        return

    biased_model = scenario_a["biased_model"]
    pipeline     = biased_model.pipeline
    feature_names = scenario_a["biased_bundle"].biased.X.columns.tolist()

    # Lấy importances từ model
    model_step = pipeline.named_steps.get("model")
    if hasattr(model_step, "coef_"):
        importances = np.abs(model_step.coef_[0])
    elif hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
    else:
        logger.warning("Model không có feature importances — skip step 3.")
        return

    # Highlight noise features
    noise_features = [f for f in feature_names if "noise_confirm" in f]

    if len(importances) == len(feature_names):
        fig, _ = plot_feature_importance(
            feature_names=feature_names,
            importances=importances,
            top_n=min(20, len(feature_names)),
            title="Feature Importance (Cherry-Pick Scenario)\nRed = Injected Noise Features",
            highlight_cols=noise_features,
        )
        save_fig(fig, "03_feature_importance_scenario_a.png")
        logger.info(
            "Noise features in top importances: %s",
            [f for f in noise_features if f in feature_names]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Multi-Model Comparison
# ─────────────────────────────────────────────────────────────────────────────

def step_04_model_comparison(bundle, results: list[dict]) -> None:
    logger.info("=" * 60)
    logger.info("STEP 4: Multi-model comparison")
    logger.info("=" * 60)

    # Lấy report từ scenario có bias lớn nhất (Scenario B - sample filtering)
    scenario_b = next(
        (r for r in results if r["scenario"]["strategy"] == "filter_confirming_samples"), None
    )
    if scenario_b is None:
        return

    biased_bundle = scenario_b["biased_bundle"]

    model_types = ["logistic_regression", "random_forest", "decision_tree"]
    metrics_all = {}

    # Baseline metrics
    tt_clean = preprocess(bundle, random_state=RANDOM_STATE)
    for mt in model_types:
        mr = train_baseline(tt_clean, model_type=mt, run_cv=False)
        metrics_all[f"Baseline {mt.split('_')[0].title()}"] = mr.test_metrics

    # Biased metrics
    tt_biased = preprocess(biased_bundle.biased, random_state=RANDOM_STATE)
    for mt in model_types:
        mr = train_baseline(tt_biased, model_type=mt, run_cv=False)
        metrics_all[f"Biased {mt.split('_')[0].title()}"] = mr.test_metrics

    fig, _ = plot_metric_comparison(
        metrics_all,
        title="Baseline vs Biased: Across Multiple Model Types\n(Confirmation Bias — Sample Filtering)",
    )
    save_fig(fig, "04_multi_model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Aggregate Analysis
# ─────────────────────────────────────────────────────────────────────────────

def step_05_aggregate_analysis(results: list[dict]) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 5: Aggregate analysis & severity heatmap")
    logger.info("=" * 60)

    reports = [r["report"] for r in results]

    # Severity heatmap
    fig, _ = plot_severity_heatmap(
        reports,
        title=f"Confirmation Bias — Severity Heatmap\n(All {len(reports)} Scenarios)",
    )
    save_fig(fig, "05_severity_heatmap.png")

    # Summary table
    df = build_summary_table(reports)
    csv_path = RESULTS_DIR / "metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved summary CSV: %s", csv_path)

    print_summary(reports)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def step_06_dashboard(results: list[dict]) -> None:
    logger.info("=" * 60)
    logger.info("STEP 6: Generate master dashboard")
    logger.info("=" * 60)

    reports = [r["report"] for r in results]
    fig = plot_bias_dashboard(
        reports,
        title=f"Experiment 01: {EXPERIMENT_NAME} — Impact Dashboard",
    )
    save_fig(fig, "06_bias_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Write Report
# ─────────────────────────────────────────────────────────────────────────────

def step_07_write_report(
    results: list[dict],
    df_summary: pd.DataFrame,
    baseline,
) -> None:
    logger.info("=" * 60)
    logger.info("STEP 7: Write text report")
    logger.info("=" * 60)

    report_path = RESULTS_DIR / "experiment_01_report.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 70,
        f"  EXPERIMENT 01: {EXPERIMENT_NAME.upper()}",
        f"  Generated: {ts}",
        "=" * 70,
        "",
        "BASELINE MODEL",
        "-" * 40,
        f"  Model type    : {baseline.model_type}",
        f"  Train size    : {baseline.train_size}",
        f"  Test size     : {baseline.test_size}",
        f"  Test metrics  : {baseline.test_metrics}",
        "",
        "SCENARIO RESULTS",
        "-" * 40,
    ]

    for r in results:
        sc     = r["scenario"]
        report = r["report"]
        wi     = report.worst_impact
        lines += [
            "",
            f"  Scenario {sc['id']}: {sc['name']}",
            f"  Strategy   : {sc['strategy']}",
            f"  Description: {sc['description']}",
            f"  Severity   : {report.overall_severity.upper()}",
            f"  Summary    : {report.summary}",
        ]
        if wi:
            lines += [
                f"  Worst metric: {wi.metric_name} "
                f"{wi.baseline_value:.4f} → {wi.biased_value:.4f} "
                f"({wi.relative_pct:+.1f}%, {wi.direction})",
            ]
        if report.stat_test:
            lines.append(
                f"  Stat test  : {report.stat_test.get('interpretation', '')}"
            )

    lines += [
        "",
        "TOP IMPACTED METRICS",
        "-" * 40,
    ]

    if not df_summary.empty:
        top = df_summary[df_summary["direction"] == "inflated"].head(8)
        for _, row in top.iterrows():
            lines.append(
                f"  [{row['severity']:10s}] {row['strategy']:35s} | "
                f"{row['metric']:15s}: {row['baseline']:.4f} → {row['biased']:.4f} "
                f"({row['rel_pct']:+.1f}%)"
            )

    lines += ["", "=" * 70, ""]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written: %s", report_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    n_samples: int = 3_000,
    class_imbalance: float = 0.35,
    model_type: str = "logistic_regression",
) -> dict:
    """
    Run toàn bộ Experiment 01.

    Parameters
    ----------
    n_samples : int
        Số samples synthetic data
    class_imbalance : float
        Tỉ lệ minority class (0.0–0.5)
    model_type : str
        Model dùng để train và so sánh

    Returns
    -------
    dict chứa bundle, baseline, results, df_summary
    """
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  EXPERIMENT 01: CONFIRMATION BIAS        ║")
    logger.info("╚══════════════════════════════════════════╝")

    setup()

    bundle, tt, baseline = step_01_load_and_baseline(
        n_samples=n_samples,
        class_imbalance=class_imbalance,
        model_type=model_type,
    )
    results    = step_02_run_scenarios(bundle, baseline, model_type=model_type)
    step_03_feature_importance(results, bundle)
    step_04_model_comparison(bundle, results)
    df_summary = step_05_aggregate_analysis(results)
    step_06_dashboard(results)
    step_07_write_report(results, df_summary, baseline)

    logger.info("✓ Experiment 01 complete. Results: %s", RESULTS_DIR.resolve())

    return {
        "bundle":     bundle,
        "baseline":   baseline,
        "results":    results,
        "df_summary": df_summary,
    }


if __name__ == "__main__":
    run()