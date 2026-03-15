"""
experiments/experiment_02_survivorship_bias.py
─────────────────────────────────────────────────────────────────────────────
Experiment 02: Survivorship Bias

Mục tiêu:
  Chứng minh rằng survivorship bias — chỉ nhìn vào "survivors" —
  dẫn đến model học một distribution khác hoàn toàn với reality,
  và khi deploy trên real data (có cả failures), model fail.

Các scenarios được test:
  A. Remove Failures      — loại 85% failures khỏi training set
  B. Historical Filter    — chỉ train trên entities vượt ngưỡng performance
  C. Selection Filter     — simulate data collection bias (chỉ record actives)
  D. Look-ahead Bias      — inject thông tin từ tương lai (data leakage)
  E. Removal Rate Sweep   — test removal_rate từ 0% đến 95% (sensitivity analysis)

Kết quả được lưu vào:
  experiments/results/experiment_02/
    ├── metrics_summary.csv
    ├── 01_class_distribution.png
    ├── 02_impact_bars_scenario_[a-d].png
    ├── 03_survivorship_funnel_[a-c].png
    ├── 04_removal_rate_sweep.png
    ├── 05_severity_heatmap.png
    ├── 06_bias_dashboard.png
    └── experiment_02_report.txt
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
from src.models.baseline_model import train_baseline
from src.simulation.survivorship_bias import simulate_survivorship_bias
from src.metrics.evaluation import (
    compare_baseline_vs_biased,
    build_summary_table,
    print_summary,
)
from src.visualization.plots import (
    set_style,
    plot_class_distribution,
    plot_roc_curve,
    plot_metric_comparison,
)
from src.visualization.bias_charts import (
    plot_bias_impact_bars,
    plot_severity_heatmap,
    plot_survivorship_funnel,
    plot_bias_dashboard,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_ID   = "experiment_02"
EXPERIMENT_NAME = "Survivorship Bias"
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
        "name":        "Remove Failures",
        "strategy":    "remove_failures",
        "kwargs":      {"failure_class": 0, "removal_rate": 0.85},
        "description": "Loại 85% failures — chỉ train trên survivors.",
    },
    {
        "id":          "B",
        "name":        "Historical Performance Filter",
        "strategy":    "apply_historical_filter",
        "kwargs":      {"cutoff_percentile": 25.0, "keep_above": True},
        "description": "Chỉ dùng data của entities vượt p25 performance threshold.",
    },
    {
        "id":          "C",
        "name":        "Selection Filter (Data Collection Bias)",
        "strategy":    "simulate_selection_filter",
        "kwargs":      {"keep_above": True},
        "description": "Simulate chỉ record data từ active users — drop-offs bị mất.",
    },
    {
        "id":          "D",
        "name":        "Look-ahead Bias (Data Leakage)",
        "strategy":    "inject_lookahead_bias",
        "kwargs":      {"n_future_features": 3, "leakage_strength": 0.8},
        "description": "Inject 3 features chứa thông tin tương lai vào training data.",
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
    n_samples: int = 4_000,
    class_imbalance: float = 0.3,
    model_type: str = "random_forest",
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

    tt       = preprocess(bundle, random_state=RANDOM_STATE)
    baseline = train_baseline(
        tt, model_type=model_type,
        label="Baseline (Full Data)", run_cv=True,
    )
    logger.info("Baseline metrics: %s", baseline.test_metrics)

    fig, _ = plot_class_distribution(
        {"Full Dataset": bundle.y},
        title="Original Class Distribution (Before Survivorship Bias)",
    )
    save_fig(fig, "01_class_distribution_original.png")

    return bundle, tt, baseline


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def step_02_run_scenarios(
    bundle,
    baseline,
    model_type: str = "random_forest",
) -> list[dict]:
    logger.info("=" * 60)
    logger.info("STEP 2: Run survivorship bias scenarios")
    logger.info("=" * 60)

    results = []

    for scenario in SCENARIOS:
        logger.info("\n--- Scenario %s: %s ---", scenario["id"], scenario["name"])
        logger.info("Description: %s", scenario["description"])

        kwargs = scenario["kwargs"].copy()
        if scenario["strategy"] == "simulate_selection_filter":
            kwargs["filter_feature"] = bundle.X.columns[0]

        biased_bundle = simulate_survivorship_bias(
            bundle, strategy=scenario["strategy"], **kwargs
        )
        logger.info("BiasedBundle: %s", biased_bundle)

        # Train model trên biased data
        tt_biased    = preprocess(biased_bundle.biased, random_state=RANDOM_STATE)
        biased_model = train_baseline(
            tt_biased, model_type=model_type,
            label=f"Biased ({scenario['name']})", run_cv=False,
        )

        # Evaluate biased model trên CLEAN test set của baseline
        # Đây là key insight: model biased fail trên real-world data
        from src.models.baseline_model import ModelResult, build_model, train_model
        pipeline_retrain = build_model(model_type, scale_features=False)
        real_world_result = train_model(
            pipeline_retrain,
            tt_biased.X_train, tt_biased.y_train,
            preprocess(bundle, random_state=RANDOM_STATE).X_test,
            preprocess(bundle, random_state=RANDOM_STATE).y_test,
            model_type=model_type,
            label=f"Biased model on REAL data ({scenario['name']})",
            run_cv=False,
        )

        # Compare biased vs baseline (cùng test set)
        report = compare_baseline_vs_biased(
            baseline, biased_model,
            bias_type="survivorship_bias",
            bias_strategy=scenario["strategy"],
            bias_params=kwargs,
        )
        logger.info("Impact report: %s", report.summary)

        # Plot: impact bars
        fig, _ = plot_bias_impact_bars(
            report,
            title=f"Scenario {scenario['id']}: {scenario['name']}",
        )
        save_fig(fig, f"02_impact_bars_scenario_{scenario['id'].lower()}.png")

        # Plot: survivorship funnel (chỉ cho A, B, C)
        if scenario["strategy"] in ("remove_failures", "apply_historical_filter",
                                    "simulate_selection_filter"):
            fig, _ = plot_survivorship_funnel(
                biased_bundle.bias_report,
                title=f"Survivorship Funnel — {scenario['name']}",
            )
            save_fig(fig, f"03_funnel_scenario_{scenario['id'].lower()}.png")

        # Plot: class distribution shift
        if len(biased_bundle.biased.y) != len(bundle.y):
            fig, _ = plot_class_distribution(
                {
                    "Original (All Data)": bundle.y,
                    f"After {scenario['name']}": biased_bundle.biased.y,
                },
                title=f"Distribution Shift — {scenario['name']}",
            )
            save_fig(fig, f"02b_dist_shift_scenario_{scenario['id'].lower()}.png")

        results.append({
            "scenario":         scenario,
            "biased_bundle":    biased_bundle,
            "biased_model":     biased_model,
            "real_world_result": real_world_result,
            "report":           report,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: ROC Comparison — Biased vs Baseline vs Real-World
# ─────────────────────────────────────────────────────────────────────────────

def step_03_roc_comparison(results: list[dict], baseline) -> None:
    logger.info("=" * 60)
    logger.info("STEP 3: ROC curve comparison")
    logger.info("=" * 60)

    # Dùng Scenario A (remove_failures) — bias rõ ràng nhất
    scenario_a = next(
        (r for r in results if r["scenario"]["strategy"] == "remove_failures"), None
    )
    if scenario_a is None:
        return

    biased_model      = scenario_a["biased_model"]
    real_world_result = scenario_a["real_world_result"]

    y_true_dict, y_proba_dict, colors_dict = {}, {}, {}

    if baseline.y_proba is not None:
        y_true_dict["Baseline (Clean)"]        = baseline.y_true
        y_proba_dict["Baseline (Clean)"]       = baseline.y_proba

    if biased_model.y_proba is not None:
        y_true_dict["Biased (Survivorship)"]   = biased_model.y_true
        y_proba_dict["Biased (Survivorship)"]  = biased_model.y_proba

    if real_world_result.y_proba is not None:
        y_true_dict["Biased on Real Data"]     = real_world_result.y_true
        y_proba_dict["Biased on Real Data"]    = real_world_result.y_proba

    if len(y_true_dict) >= 2:
        from src.visualization.plots import PALETTE
        colors = {
            "Baseline (Clean)":       PALETTE["baseline"],
            "Biased (Survivorship)":  PALETTE["biased"],
            "Biased on Real Data":    PALETTE["warning"],
        }
        fig, _ = plot_roc_curve(
            y_true_dict, y_proba_dict,
            colors=colors,
            title="ROC Comparison: Baseline vs Survivorship Biased vs Real-World",
        )
        save_fig(fig, "03_roc_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Removal Rate Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def step_04_removal_rate_sweep(
    bundle,
    baseline,
    model_type: str = "random_forest",
    removal_rates: list[float] | None = None,
) -> None:
    logger.info("=" * 60)
    logger.info("STEP 4: Removal rate sensitivity analysis")
    logger.info("=" * 60)

    if removal_rates is None:
        removal_rates = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95]

    sweep_records = []

    for rate in removal_rates:
        if rate == 0.0:
            # No bias
            tt_clean = preprocess(bundle, random_state=RANDOM_STATE)
            mr = train_baseline(tt_clean, model_type=model_type, run_cv=False)
        else:
            biased = simulate_survivorship_bias(
                bundle, strategy="remove_failures",
                failure_class=0, removal_rate=rate,
            )
            tt_b = preprocess(biased.biased, random_state=RANDOM_STATE)
            mr   = train_baseline(tt_b, model_type=model_type, run_cv=False)

        record = {
            "removal_rate": rate,
            "n_train":      mr.train_size,
            **mr.test_metrics,
        }
        sweep_records.append(record)
        logger.info("  rate=%.0f%%: acc=%.4f, auc=%s",
                    rate * 100,
                    mr.test_metrics["accuracy"],
                    mr.test_metrics.get("roc_auc", "N/A"))

    df_sweep = pd.DataFrame(sweep_records)

    # Save CSV
    csv_path = RESULTS_DIR / "removal_rate_sweep.csv"
    df_sweep.to_csv(csv_path, index=False)

    # Plot
    from src.visualization.plots import PALETTE
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics_plot = [m for m in ["accuracy", "f1", "roc_auc"] if m in df_sweep.columns]
    colors_list  = [PALETTE["baseline"], PALETTE["biased"], PALETTE["positive"]]

    for metric, color in zip(metrics_plot, colors_list):
        axes[0].plot(
            df_sweep["removal_rate"] * 100,
            df_sweep[metric],
            "o-", lw=2, color=color, label=metric, markersize=5,
        )

    # Annotate zero-bias point
    baseline_auc = df_sweep[df_sweep["removal_rate"] == 0.0]["roc_auc"].values
    if len(baseline_auc):
        axes[0].axhline(
            baseline_auc[0], color=PALETTE["neutral"],
            lw=1, linestyle="--", alpha=0.5, label="Baseline (no bias)",
        )

    axes[0].set_xlabel("Failure Removal Rate (%)")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Metric Drift vs Removal Rate\n(Survivorship Bias Severity)")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(-2, 100)

    # Right: training size reduction
    axes[1].bar(
        df_sweep["removal_rate"] * 100,
        df_sweep["n_train"],
        width=6,
        color=PALETTE["baseline"], alpha=0.75, edgecolor="white",
    )
    axes[1].set_xlabel("Failure Removal Rate (%)")
    axes[1].set_ylabel("Training Samples Remaining")
    axes[1].set_title("Training Set Size vs Removal Rate")
    axes[1].set_xlim(-5, 100)

    fig.suptitle("Survivorship Bias Sensitivity Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "04_removal_rate_sweep.png")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Aggregate Analysis
# ─────────────────────────────────────────────────────────────────────────────

def step_05_aggregate_analysis(results: list[dict]) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 5: Aggregate analysis")
    logger.info("=" * 60)

    reports = [r["report"] for r in results]

    fig, _ = plot_severity_heatmap(
        reports,
        title="Survivorship Bias — Severity Heatmap",
    )
    save_fig(fig, "05_severity_heatmap.png")

    df = build_summary_table(reports)
    df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

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
        title=f"Experiment 02: {EXPERIMENT_NAME} — Impact Dashboard",
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
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 70,
        f"  EXPERIMENT 02: {EXPERIMENT_NAME.upper()}",
        f"  Generated: {ts}",
        "=" * 70,
        "",
        "KEY INSIGHT",
        "-" * 40,
        "  Survivorship bias causes a model trained on 'survivors' to",
        "  appear highly accurate — but fail when deployed on real-world",
        "  data that includes failures. This is the classic look-only-at-",
        "  winners fallacy.",
        "",
        "BASELINE MODEL",
        "-" * 40,
        f"  Model type   : {baseline.model_type}",
        f"  Train size   : {baseline.train_size}",
        f"  Test metrics : {baseline.test_metrics}",
        "",
        "SCENARIO RESULTS",
        "-" * 40,
    ]

    for r in results:
        sc     = r["scenario"]
        report = r["report"]
        rw     = r["real_world_result"]
        wi     = report.worst_impact

        lines += [
            "",
            f"  Scenario {sc['id']}: {sc['name']}",
            f"  Strategy      : {sc['strategy']}",
            f"  Description   : {sc['description']}",
            f"  Severity      : {report.overall_severity.upper()}",
            f"  Summary       : {report.summary}",
            f"  Real-world AUC: {rw.test_metrics.get('roc_auc', 'N/A')} "
            f"(vs biased {r['biased_model'].test_metrics.get('roc_auc', 'N/A')})",
        ]
        if wi:
            lines.append(
                f"  Worst metric  : {wi.metric_name} "
                f"{wi.baseline_value:.4f} → {wi.biased_value:.4f} "
                f"({wi.relative_pct:+.1f}%)"
            )

    lines += ["", "=" * 70, ""]
    (RESULTS_DIR / "experiment_02_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    logger.info("Report written.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    n_samples: int = 4_000,
    class_imbalance: float = 0.3,
    model_type: str = "random_forest",
) -> dict:
    """
    Run toàn bộ Experiment 02.

    Returns dict chứa bundle, baseline, results, df_summary.
    """
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  EXPERIMENT 02: SURVIVORSHIP BIAS        ║")
    logger.info("╚══════════════════════════════════════════╝")

    setup()

    bundle, tt, baseline = step_01_load_and_baseline(
        n_samples=n_samples,
        class_imbalance=class_imbalance,
        model_type=model_type,
    )
    results = step_02_run_scenarios(bundle, baseline, model_type=model_type)
    step_03_roc_comparison(results, baseline)
    step_04_removal_rate_sweep(bundle, baseline, model_type=model_type)
    df_summary = step_05_aggregate_analysis(results)
    step_06_dashboard(results)
    step_07_write_report(results, df_summary, baseline)

    logger.info("✓ Experiment 02 complete. Results: %s", RESULTS_DIR.resolve())

    return {
        "bundle":     bundle,
        "baseline":   baseline,
        "results":    results,
        "df_summary": df_summary,
    }


if __name__ == "__main__":
    run()