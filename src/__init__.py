"""
src/__init__.py
─────────────────────────────────────────────────────────────────────────────
Single entry point cho toàn bộ behavioral-bias-ds package.

Usage trong notebooks và experiments:
    from src import load_dataset, preprocess
    from src import simulate_confirmation_bias, simulate_survivorship_bias
    from src import train_baseline, compare_baseline_vs_biased
    from src import plot_bias_dashboard
"""

# ── Data ──────────────────────────────────────────────────────────────────────
from src.data.load_data import load_dataset, DataBundle
from src.data.preprocess import preprocess, TrainTestBundle

# ── Simulation ────────────────────────────────────────────────────────────────
from src.simulation.confirmation_bias import (
    simulate_confirmation_bias,
    BiasedBundle,
)
from src.simulation.survivorship_bias import simulate_survivorship_bias
from src.simulation.overconfidence_bias import (
    simulate_overconfidence_bias,
    OverconfidenceResult,
)

# ── Models ────────────────────────────────────────────────────────────────────
from src.models.baseline_model import (
    build_model,
    train_model,
    train_baseline,
    train_all_baselines,
    ModelResult,
)
from src.models.biased_model_selection import (
    simulate_biased_selection,
    SelectionBiasResult,
    fish_decision_threshold,
    cherry_pick_metric,
    simulate_test_set_reuse,
    fish_hyperparameters,
    selective_reporting,
)

# ── Metrics ───────────────────────────────────────────────────────────────────
from src.metrics.evaluation import (
    compute_classification_metrics,
    compute_calibration_metrics,
    compute_bias_impact,
    compare_baseline_vs_biased,
    evaluate_all_biases,
    build_summary_table,
    print_summary,
    BiasImpact,
    ComparisonReport,
)

# ── Visualization ─────────────────────────────────────────────────────────────
from src.visualization.plots import (
    set_style,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_feature_importance,
    plot_class_distribution,
    plot_metric_comparison,
    plot_learning_curve,
    plot_correlation_heatmap,
    plot_distribution_comparison,
)
from src.visualization.bias_charts import (
    plot_bias_impact_bars,
    plot_calibration_comparison,
    plot_severity_heatmap,
    plot_threshold_landscape,
    plot_survivorship_funnel,
    plot_metric_hacking_distribution,
    plot_bias_dashboard,
)

__all__ = [
    # data
    "load_dataset", "DataBundle",
    "preprocess", "TrainTestBundle",
    # simulation
    "simulate_confirmation_bias", "BiasedBundle",
    "simulate_survivorship_bias",
    "simulate_overconfidence_bias", "OverconfidenceResult",
    # models
    "build_model", "train_model", "train_baseline", "train_all_baselines", "ModelResult",
    "simulate_biased_selection", "SelectionBiasResult", "fish_decision_threshold",
    "cherry_pick_metric", "simulate_test_set_reuse","fish_hyperparameters",
    "selective_reporting",
    # metrics
    "compute_classification_metrics", "compute_calibration_metrics",
    "compute_bias_impact", "compare_baseline_vs_biased",
    "evaluate_all_biases", "build_summary_table", "print_summary",
    "BiasImpact", "ComparisonReport",
    # visualization — general
    "set_style",
    "plot_confusion_matrix", "plot_roc_curve", "plot_pr_curve",
    "plot_feature_importance", "plot_class_distribution",
    "plot_metric_comparison", "plot_learning_curve",
    "plot_correlation_heatmap", "plot_distribution_comparison",
    # visualization — bias
    "plot_bias_impact_bars", "plot_calibration_comparison",
    "plot_severity_heatmap", "plot_threshold_landscape",
    "plot_survivorship_funnel", "plot_metric_hacking_distribution",
    "plot_bias_dashboard",
]