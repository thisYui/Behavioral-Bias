"""
src/visualization/bias_charts.py
─────────────────────────────────────────────────────────────────────────────
Specialized charts cho behavioral bias analysis.

Mỗi function được thiết kế để trực quan hóa MỘT khía cạnh cụ thể của bias:
  - Tác động bias lên metrics (before/after)
  - Calibration degradation
  - Severity heatmap
  - Threshold fishing landscape
  - Survivorship funnel
  - Overconfidence reliability diagram
  - Full bias dashboard (tổng hợp)

Import từ plots.py để dùng chung PALETTE và style.

Public API:
    plot_bias_impact_bars(report)
    plot_calibration_comparison(report)
    plot_severity_heatmap(reports)
    plot_threshold_landscape(threshold_report)
    plot_survivorship_funnel(survivorship_report)
    plot_overconfidence_reliability(oc_result)
    plot_metric_hacking_distribution(hacking_report)
    plot_bias_dashboard(reports)           ← main summary figure
"""

from __future__ import annotations

import warnings
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from src.visualization.plots import (
    PALETTE, SEVERITY_COLORS, FIGSIZE_DEFAULTS, set_style, _finalize,
    plot_confusion_matrix, plot_metric_comparison,
)

warnings.filterwarnings("ignore")
set_style()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bias Impact — Before / After Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_bias_impact_bars(
    report,                      # ComparisonReport từ evaluation.py
    metrics: list[str] | None = None,
    title: str | None = None,
    figsize: tuple = (11, 5),
) -> tuple[Figure, Axes]:
    """
    Grouped bar chart: baseline vs biased, annotate delta và severity.

    Đây là chart chính để show "bias làm thay đổi metrics bao nhiêu".
    """
    if metrics is None:
        metrics = ["accuracy", "f1", "roc_auc", "precision", "recall"]

    available = set(report.baseline_metrics.keys()) & set(report.biased_metrics.keys())
    metrics = [m for m in metrics if m in available]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(metrics))
    width = 0.32

    baseline_vals = [report.baseline_metrics.get(m, 0) for m in metrics]
    biased_vals   = [report.biased_metrics.get(m, 0)   for m in metrics]

    bars_b = ax.bar(x - width / 2, baseline_vals, width,
                    label="Baseline (clean data)",
                    color=PALETTE["baseline"], alpha=0.88,
                    edgecolor="white", linewidth=0.8)
    bars_d = ax.bar(x + width / 2, biased_vals, width,
                    label=f"Biased ({report.bias_strategy})",
                    color=PALETTE["biased"], alpha=0.88,
                    edgecolor="white", linewidth=0.8)

    # Value labels
    for bar, v in zip(bars_b, baseline_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                color=PALETTE["text_muted"])

    for bar, v, bv in zip(bars_d, biased_vals, baseline_vals):
        delta = v - bv
        sign  = "+" if delta >= 0 else ""
        color = PALETTE["biased"] if delta > 0 else PALETTE["positive"]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                color=PALETTE["text_muted"])
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                f"({sign}{delta:.3f})", ha="center", va="bottom", fontsize=7,
                color=color, fontweight="semibold")

    # Severity badges
    impact_map = {imp.metric_name: imp for imp in report.impacts}
    for i, metric in enumerate(metrics):
        imp = impact_map.get(metric)
        if imp and imp.severity not in ("negligible",):
            ax.text(x[i], -0.06, imp.severity,
                    ha="center", va="top", fontsize=6.5, transform=ax.transData,
                    color=SEVERITY_COLORS.get(imp.severity, PALETTE["neutral"]),
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=SEVERITY_COLORS.get(imp.severity, "#eee") + "22",
                              edgecolor=SEVERITY_COLORS.get(imp.severity, "#eee"),
                              linewidth=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.18)
    ax.set_title(title or f"Bias Impact: {report.bias_type} / {report.bias_strategy}")
    ax.legend(loc="lower right", fontsize=9)

    # Severity annotation box
    if report.worst_impact:
        wi = report.worst_impact
        ax.text(0.02, 0.97,
                f"Worst: {wi.metric_name} {wi.relative_pct:+.1f}%  |  "
                f"Severity: {report.overall_severity.upper()}",
                transform=ax.transAxes, fontsize=8,
                va="top", color=SEVERITY_COLORS.get(report.overall_severity, PALETTE["text"]),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=SEVERITY_COLORS.get(report.overall_severity, "#ccc"),
                          linewidth=1.2))

    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 2. Calibration Comparison — Reliability Diagram
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_comparison(
    report,                       # ComparisonReport
    title: str = "Calibration: Baseline vs Biased",
    figsize: tuple = (10, 5),
) -> tuple[Figure, list[Axes]]:
    """
    Reliability diagram so sánh calibration baseline vs biased.
    Đặc biệt hữu ích cho overconfidence bias visualization.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def _draw_reliability(ax, cal_data: dict, label: str, color: str) -> None:
        if not cal_data or "reliability_curve" not in cal_data:
            ax.text(0.5, 0.5, "No calibration data", ha="center", va="center",
                    transform=ax.transAxes, color=PALETTE["text_muted"])
            ax.set_title(label)
            return

        prob_true = cal_data["reliability_curve"]["prob_true"]
        prob_pred = cal_data["reliability_curve"]["prob_pred"]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect calibration")

        # Model calibration
        ax.plot(prob_pred, prob_true, "o-", color=color, lw=2, markersize=6,
                alpha=0.9, label=f"{label}")
        ax.fill_between(prob_pred, prob_pred, prob_true,
                        alpha=0.15, color=color, label="Calibration gap")

        # Histogram of predictions (secondary axis)
        ax2 = ax.twinx()
        ax2.hist(prob_pred, bins=10, range=(0, 1), color=color,
                 alpha=0.15, edgecolor="none")
        ax2.set_ylabel("Count", color=PALETTE["text_muted"], fontsize=8)
        ax2.tick_params(axis="y", labelcolor=PALETTE["text_muted"], labelsize=7)
        ax2.set_ylim(0, ax2.get_ylim()[1] * 4)  # push histogram to bottom

        # Metrics annotation
        ece  = cal_data.get("ece", 0)
        oci  = cal_data.get("overconfidence_index", 0)
        ax.text(0.05, 0.92, f"ECE = {ece:.4f}\nOCI = {oci:+.4f}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=PALETTE["border"], linewidth=0.8))

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(label, fontsize=11)
        ax.legend(loc="upper left", fontsize=8)

    _draw_reliability(axes[0], report.baseline_calibration,
                      "Baseline", PALETTE["baseline"])
    _draw_reliability(axes[1], report.biased_calibration,
                      f"Biased ({report.bias_strategy})", PALETTE["biased"])

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, list(axes)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Severity Heatmap — tổng quan tất cả biases × metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_severity_heatmap(
    reports: list,                 # list[ComparisonReport]
    metrics: list[str] | None = None,
    title: str = "Bias Severity Heatmap",
    figsize: tuple | None = None,
) -> tuple[Figure, Axes]:
    """
    Heatmap: rows = bias strategies, cols = metrics, values = |relative_pct|.
    Color intensity = mức độ tác động của bias.

    Chart tổng quan nhất của toàn project.
    """
    if metrics is None:
        metrics = ["accuracy", "f1", "roc_auc", "precision", "recall", "brier_score"]

    # Build matrix
    strategies = [f"{r.bias_type}\n{r.bias_strategy}" for r in reports]
    matrix = np.zeros((len(reports), len(metrics)))
    direction_matrix = np.full((len(reports), len(metrics)), "")

    for i, report in enumerate(reports):
        impact_map = {imp.metric_name: imp for imp in report.impacts}
        for j, metric in enumerate(metrics):
            imp = impact_map.get(metric)
            if imp:
                matrix[i, j] = abs(imp.relative_pct)
                direction_matrix[i, j] = "▲" if imp.direction == "inflated" else "▼"

    if figsize is None:
        figsize = (max(9, len(metrics) * 1.3), max(4, len(reports) * 0.9 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    # Custom colormap: white → amber → red
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "bias_severity",
        ["#F8FAFC", "#FEF3C7", "#FCA5A5", "#DC2626", "#7C3AED"],
        N=256,
    )

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=20)

    # Annotations: value + direction arrow
    for i in range(len(reports)):
        for j in range(len(metrics)):
            val = matrix[i, j]
            arr = direction_matrix[i, j]
            text_color = "white" if val > 12 else PALETTE["text"]
            ax.text(j, i, f"{arr}{val:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color=text_color, fontweight="semibold")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(reports)))
    ax.set_yticklabels(strategies, fontsize=8)
    ax.set_title(title, pad=16, fontsize=13, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("|Relative Change %|", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Severity legend
    legend_patches = [
        mpatches.Patch(color=c, label=sev.title())
        for sev, c in [
            ("negligible", "#F8FAFC"), ("minor", "#FEF3C7"),
            ("moderate", "#FCA5A5"), ("severe", "#DC2626"), ("critical", "#7C3AED")
        ]
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              bbox_to_anchor=(1.12, 1), fontsize=8, title="Severity")

    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 4. Threshold Fishing Landscape
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_landscape(
    threshold_report: dict,
    title: str = "Decision Threshold Fishing",
    figsize: tuple = (10, 5),
) -> tuple[Figure, list[Axes]]:
    """
    Visualize tác động của việc fish threshold trên test set.
    Shows: metric score theo threshold + highlight khoảng "honest" vs "biased".
    """
    all_scores = threshold_report.get("all_threshold_scores", [])
    if not all_scores:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No threshold data", ha="center", va="center")
        return fig, [ax]

    df = pd.DataFrame(all_scores)
    metrics_cols = [c for c in ["accuracy", "f1", "precision", "recall"] if c in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Left: metric lines vs threshold ────────────────────────────────────
    ax = axes[0]
    metric_colors = {
        "accuracy":  PALETTE["baseline"],
        "f1":        PALETTE["biased"],
        "precision": PALETTE["positive"],
        "recall":    PALETTE["warning"],
    }

    for metric in metrics_cols:
        ax.plot(df["threshold"], df[metric], lw=2,
                color=metric_colors.get(metric, PALETTE["neutral"]),
                label=metric, alpha=0.85)

    # Highlight best threshold
    best_t  = threshold_report.get("best_threshold", 0.5)
    default = 0.5
    obj     = threshold_report.get("objective", "f1")

    ax.axvline(default, color=PALETTE["baseline"], lw=1.5, linestyle="--",
               alpha=0.6, label=f"Default (0.5)")
    ax.axvline(best_t, color=PALETTE["biased"], lw=2, linestyle=":",
               alpha=0.9, label=f"Best ({best_t:.2f})")

    # Shaded "fishing zone"
    ax.axvspan(min(df["threshold"]), max(df["threshold"]),
               alpha=0.04, color=PALETTE["biased"],
               label="Fishing zone (all tried)")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metric vs Threshold")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])

    # ── Right: honest vs reported comparison ─────────────────────────────
    ax2 = axes[1]
    honest   = threshold_report.get("honest_metrics", {})
    reported = threshold_report.get("reported_metrics", {})
    shared   = [m for m in metrics_cols if m in honest and m in reported]

    x = np.arange(len(shared))
    w = 0.35
    bars_h = ax2.bar(x - w / 2, [honest[m]   for m in shared], w,
                     color=PALETTE["baseline"], alpha=0.85, label="Honest (threshold=0.5)")
    bars_r = ax2.bar(x + w / 2, [reported[m] for m in shared], w,
                     color=PALETTE["biased"],   alpha=0.85, label=f"Reported (threshold={best_t:.2f})")

    for bar, m in zip(bars_h, shared):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{honest[m]:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, m in zip(bars_r, shared):
        gain = reported[m] - honest[m]
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{reported[m]:.3f}", ha="center", va="bottom", fontsize=8)
        if abs(gain) > 0.001:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                     f"(+{gain:.3f})", ha="center", va="bottom", fontsize=7,
                     color=PALETTE["biased"], fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(shared, rotation=15, ha="right")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1.15)
    ax2.set_title(f"Honest vs Reported ({obj} optimized)")
    ax2.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, list(axes)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Survivorship Funnel
# ─────────────────────────────────────────────────────────────────────────────

def plot_survivorship_funnel(
    survivorship_report: dict,
    title: str = "Survivorship Bias Funnel",
    figsize: tuple = (9, 5),
) -> tuple[Figure, list[Axes]]:
    """
    Visualize quá trình loại bỏ failures: original → survivors.
    Shows class distribution shift và sample reduction.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Left: funnel / bar showing samples removed ────────────────────────
    ax = axes[0]
    orig_n   = survivorship_report.get("original_total",  survivorship_report.get("original_n", 0))
    biased_n = survivorship_report.get("biased_total",    survivorship_report.get("filtered_n",  survivorship_report.get("survived_n", 0)))
    removed  = orig_n - biased_n

    labels = ["Original Dataset", "Removed\n(Failures)", "Survived\n(Biased Training Set)"]
    values = [orig_n, removed, biased_n]
    colors = [PALETTE["baseline"], PALETTE["biased"], PALETTE["positive"]]

    bars = ax.barh(labels, values, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + orig_n * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:,} ({v/orig_n:.1%})", va="center", fontsize=9,
                color=PALETTE["text_muted"])

    ax.set_xlim(0, orig_n * 1.3)
    ax.set_xlabel("Sample Count")
    ax.set_title("Sample Retention")
    ax.invert_yaxis()

    # ── Right: class distribution shift ───────────────────────────────────
    ax2 = axes[1]

    orig_dist  = survivorship_report.get("original_class_distribution",
                 survivorship_report.get("original_class_dist",
                 survivorship_report.get("class_dist_before", {})))
    biased_dist = survivorship_report.get("biased_class_distribution",
                  survivorship_report.get("biased_class_dist",
                  survivorship_report.get("class_dist_after", {})))

    if orig_dist and biased_dist:
        classes = sorted(set(list(orig_dist.keys()) + list(biased_dist.keys())))
        x = np.arange(len(classes))
        w = 0.35

        orig_vals  = [orig_dist.get(c, 0)  for c in classes]
        biased_vals = [biased_dist.get(c, 0) for c in classes]

        # Normalize to proportions
        orig_sum   = sum(orig_vals)
        biased_sum = sum(biased_vals)
        orig_vals_n  = [v / orig_sum  if orig_sum  else 0 for v in orig_vals]
        biased_vals_n = [v / biased_sum if biased_sum else 0 for v in biased_vals]

        bars_o = ax2.bar(x - w / 2, orig_vals_n, w,
                         color=PALETTE["baseline"], alpha=0.85, label="Original")
        bars_b = ax2.bar(x + w / 2, biased_vals_n, w,
                         color=PALETTE["biased"], alpha=0.85, label="After Survivorship")

        for bar, v in zip(bars_o, orig_vals_n):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.1%}", ha="center", fontsize=8)
        for bar, v in zip(bars_b, biased_vals_n):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.1%}", ha="center", fontsize=8)

        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Class {c}" for c in classes])
        ax2.set_ylabel("Proportion")
        ax2.set_ylim(0, 1.15)
        ax2.set_title("Class Distribution Shift")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Distribution data unavailable",
                 ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, list(axes)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Metric Hacking Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_hacking_distribution(
    hacking_report: dict,
    title: str = "Metric Hacking: Score Distribution Across Trials",
    figsize: tuple = (10, 5),
) -> tuple[Figure, list[Axes]]:
    """
    Histogram của scores qua tất cả trials + highlight honest vs reported.
    Cho thấy tại sao reporting best-of-N inflate perceived performance.
    """
    all_scores = hacking_report.get("all_scores", [])
    if not all_scores:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No trial data", ha="center", va="center")
        return fig, [ax]

    honest   = hacking_report.get("honest_score", np.mean(all_scores))
    reported = hacking_report.get("reported_score", max(all_scores))
    metric   = hacking_report.get("metric", "score")
    n_trials = hacking_report.get("n_trials", len(all_scores))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Left: histogram ───────────────────────────────────────────────────
    ax = axes[0]
    ax.hist(all_scores, bins=min(15, n_trials // 2 + 2),
            color=PALETTE["neutral"], alpha=0.6, edgecolor="white",
            label=f"All {n_trials} trials")

    ax.axvline(honest,   color=PALETTE["baseline"], lw=2.5, linestyle="-",
               label=f"Honest (mean) = {honest:.4f}")
    ax.axvline(reported, color=PALETTE["biased"],   lw=2.5, linestyle="--",
               label=f"Reported (best) = {reported:.4f}")

    # Shade the inflation region
    ax.axvspan(honest, reported, alpha=0.12, color=PALETTE["biased"],
               label=f"Inflation = +{reported-honest:.4f}")

    ax.set_xlabel(f"{metric} Score")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {n_trials} Trial Scores")
    ax.legend(fontsize=8)

    # ── Right: inflation breakdown ─────────────────────────────────────────
    ax2 = axes[1]
    categories = ["Honest\n(should report)", "Inflated\nGain", "Reported\n(actually reported)"]
    base = [honest, 0, 0]
    gain = [0, reported - honest, 0]
    reported_bar = [0, 0, reported]

    x = np.arange(len(categories))
    ax2.bar(x, [honest, reported - honest, reported],
            color=[PALETTE["baseline"], PALETTE["biased"], PALETTE["warning"]],
            alpha=0.85, edgecolor="white", linewidth=0.8,
            label=["Honest", "Inflation", "Reported"])

    for xi, (cat, val) in enumerate(zip(categories, [honest, reported - honest, reported])):
        sign = "+" if xi == 1 else ""
        ax2.text(xi, val + 0.001, f"{sign}{val:.4f}",
                 ha="center", va="bottom", fontsize=9, fontweight="semibold")

    inflation_pct = hacking_report.get("inflation_pct", 0)
    ax2.text(0.5, 0.97,
             f"Inflation: +{inflation_pct:.2f}%\n({n_trials} trials, only 1 reported)",
             transform=ax2.transAxes, ha="center", va="top", fontsize=9,
             color=PALETTE["biased"], fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor=PALETTE["biased"], linewidth=1))

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_ylabel(f"{metric} Score")
    ax2.set_ylim(min(all_scores) * 0.97, reported * 1.08)
    ax2.set_title("Honest vs Reported Score")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, list(axes)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full Bias Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_bias_dashboard(
    reports: list,                  # list[ComparisonReport]
    title: str = "Behavioral Bias Impact Dashboard",
    figsize: tuple = (16, 12),
    metrics: list[str] | None = None,
) -> Figure:
    """
    Master dashboard tổng hợp tác động của tất cả biases.

    Layout:
    ┌──────────────────────────┬──────────────────┐
    │  Severity Heatmap (top)  │  Overall Stats   │
    ├───────────┬──────────────┼──────────────────┤
    │ Impact    │ Impact       │  Worst Metric    │
    │ Report[0] │ Report[1]   │  per Bias Type   │
    ├───────────┴──────────────┴──────────────────┤
    │  Impact delta line chart (bottom)           │
    └─────────────────────────────────────────────┘
    """
    if metrics is None:
        metrics = ["accuracy", "f1", "roc_auc", "precision", "recall"]

    fig = plt.figure(figsize=figsize, facecolor=PALETTE["background"])
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.45, wspace=0.35,
                            top=0.92, bottom=0.06, left=0.07, right=0.97)

    # ── Row 0: Severity Heatmap (full width) ─────────────────────────────
    ax_heat = fig.add_subplot(gs[0, :])
    if reports:
        strategies = [f"{r.bias_type}\n{r.bias_strategy}" for r in reports]
        avail = set()
        for r in reports:
            avail |= set(r.baseline_metrics.keys())
        m_cols = [m for m in metrics if m in avail]

        matrix = np.zeros((len(reports), len(m_cols)))
        for i, report in enumerate(reports):
            imp_map = {imp.metric_name: imp for imp in report.impacts}
            for j, metric in enumerate(m_cols):
                imp = imp_map.get(metric)
                if imp:
                    matrix[i, j] = abs(imp.relative_pct)

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "dash_cmap", ["#EFF6FF", "#FEF3C7", "#FCA5A5", "#DC2626"], N=256
        )
        im = ax_heat.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=15)
        ax_heat.set_xticks(range(len(m_cols)))
        ax_heat.set_xticklabels(m_cols, fontsize=9, rotation=20, ha="right")
        ax_heat.set_yticks(range(len(reports)))
        ax_heat.set_yticklabels(strategies, fontsize=8)
        ax_heat.set_title("Bias Impact Heatmap (|Relative Change %|)",
                          fontsize=11, fontweight="semibold", pad=10)

        for i in range(len(reports)):
            for j in range(len(m_cols)):
                val = matrix[i, j]
                tc  = "white" if val > 10 else PALETTE["text"]
                ax_heat.text(j, i, f"{val:.1f}%", ha="center", va="center",
                             fontsize=8, color=tc, fontweight="semibold")

        cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical",
                            shrink=0.85, pad=0.01)
        cbar.set_label("|Δ%|", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # ── Row 1: Individual Impact Charts (first 2 reports) ─────────────────
    for idx in range(min(2, len(reports))):
        ax_imp = fig.add_subplot(gs[1, idx])
        report = reports[idx]
        avail  = set(report.baseline_metrics.keys()) & set(report.biased_metrics.keys())
        m_plot = [m for m in metrics if m in avail]

        baseline_vals = [report.baseline_metrics.get(m, 0) for m in m_plot]
        biased_vals   = [report.biased_metrics.get(m, 0)   for m in m_plot]

        x = np.arange(len(m_plot))
        w = 0.35
        ax_imp.bar(x - w / 2, baseline_vals, w,
                   color=PALETTE["baseline"], alpha=0.85, label="Baseline")
        ax_imp.bar(x + w / 2, biased_vals, w,
                   color=PALETTE["biased"], alpha=0.85, label="Biased")

        ax_imp.set_xticks(x)
        ax_imp.set_xticklabels(m_plot, rotation=25, ha="right", fontsize=7)
        ax_imp.set_ylim(0, 1.12)
        ax_imp.set_title(f"{report.bias_strategy}\n[{report.overall_severity.upper()}]",
                         fontsize=9, fontweight="semibold")
        ax_imp.legend(fontsize=7, loc="lower right")
        ax_imp.tick_params(labelsize=7)

    # ── Row 1 Col 2: Severity Summary Donut ───────────────────────────────
    ax_donut = fig.add_subplot(gs[1, 2])
    if reports:
        from collections import Counter
        sev_counts = Counter(r.overall_severity for r in reports)
        sev_order  = ["critical", "severe", "moderate", "minor", "negligible"]
        labels_d   = [s for s in sev_order if s in sev_counts]
        sizes_d    = [sev_counts[s] for s in labels_d]
        colors_d   = [SEVERITY_COLORS[s] for s in labels_d]

        wedges, texts, autotexts = ax_donut.pie(
            sizes_d, labels=labels_d, colors=colors_d,
            autopct="%1.0f%%", startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
            textprops={"fontsize": 8},
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_fontweight("bold")

        ax_donut.set_title("Severity Distribution\nAcross All Biases",
                           fontsize=9, fontweight="semibold")

    # ── Row 2: Delta line chart ────────────────────────────────────────────
    ax_line = fig.add_subplot(gs[2, :])
    if reports:
        avail_all = set()
        for r in reports:
            avail_all |= {imp.metric_name for imp in r.impacts}
        m_line = [m for m in metrics if m in avail_all]

        for i, report in enumerate(reports):
            imp_map = {imp.metric_name: imp for imp in report.impacts}
            deltas  = [imp_map[m].relative_pct if m in imp_map else 0 for m in m_line]
            color   = list(SEVERITY_COLORS.values())[i % len(SEVERITY_COLORS)]
            label   = f"{report.bias_strategy} [{report.overall_severity}]"
            ax_line.plot(range(len(m_line)), deltas, "o-", lw=2, markersize=6,
                         color=color, alpha=0.85, label=label)

        ax_line.axhline(0, color=PALETTE["text_muted"], lw=1, linestyle="--", alpha=0.4)

        # Shade positive (inflated) region
        y_max = ax_line.get_ylim()[1] if ax_line.get_ylim()[1] > 0 else 5
        ax_line.axhspan(0, max(5, y_max), alpha=0.04, color=PALETTE["biased"])
        ax_line.axhspan(min(-5, ax_line.get_ylim()[0]), 0, alpha=0.04, color=PALETTE["positive"])

        ax_line.set_xticks(range(len(m_line)))
        ax_line.set_xticklabels(m_line, fontsize=9)
        ax_line.set_ylabel("Relative Change (%)")
        ax_line.set_title("Metric Bias Impact: Relative Change % by Strategy", fontsize=10)
        ax_line.legend(fontsize=8, loc="upper right", ncol=min(3, len(reports)))

    fig.suptitle(title, fontsize=15, fontweight="bold", color=PALETTE["text"], y=0.97)
    return fig