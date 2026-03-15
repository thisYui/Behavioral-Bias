"""
src/visualization/plots.py
─────────────────────────────────────────────────────────────────────────────
General-purpose plotting utilities dùng chung cho toàn project.

Tất cả functions trả về (fig, ax) hoặc (fig, axes) để caller có thể
customize thêm hoặc save theo ý muốn.

Palette và style được định nghĩa tập trung ở đây — import từ bias_charts.py
để đảm bảo visual consistency.

Public API:
    set_style()                         — apply global matplotlib style
    plot_confusion_matrix(cm, ...)
    plot_roc_curve(y_true, y_proba, ...)
    plot_precision_recall_curve(...)
    plot_feature_importance(model, ...)
    plot_class_distribution(y, ...)
    plot_metric_comparison(metrics_dict, ...)
    plot_learning_curve(pipeline, X, y, ...)
"""

from __future__ import annotations

import warnings
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import learning_curve

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Design System
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "baseline":   "#2563EB",   # blue  — clean, trustworthy
    "biased":     "#DC2626",   # red   — danger, inflated
    "neutral":    "#64748B",   # slate — secondary info
    "positive":   "#16A34A",   # green — good outcome
    "warning":    "#D97706",   # amber — caution
    "background": "#F8FAFC",   # near-white
    "surface":    "#FFFFFF",
    "border":     "#E2E8F0",
    "text":       "#0F172A",
    "text_muted": "#94A3B8",
}

SEVERITY_COLORS = {
    "negligible": "#94A3B8",
    "minor":      "#60A5FA",
    "moderate":   "#F59E0B",
    "severe":     "#EF4444",
    "critical":   "#7C3AED",
}

FIGSIZE_DEFAULTS = {
    "small":  (6, 4),
    "medium": (9, 6),
    "large":  (12, 8),
    "wide":   (14, 5),
    "square": (6, 6),
}


def set_style() -> None:
    """Apply global matplotlib style cho toàn project."""
    plt.rcParams.update({
        "figure.facecolor":      PALETTE["background"],
        "axes.facecolor":        PALETTE["surface"],
        "axes.edgecolor":        PALETTE["border"],
        "axes.labelcolor":       PALETTE["text"],
        "axes.labelsize":        11,
        "axes.titlesize":        13,
        "axes.titleweight":      "semibold",
        "axes.titlepad":         12,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.grid":             True,
        "grid.color":            PALETTE["border"],
        "grid.linewidth":        0.6,
        "grid.alpha":            0.8,
        "xtick.color":           PALETTE["text_muted"],
        "ytick.color":           PALETTE["text_muted"],
        "xtick.labelsize":       9,
        "ytick.labelsize":       9,
        "legend.fontsize":       9,
        "legend.framealpha":     0.9,
        "legend.edgecolor":      PALETTE["border"],
        "font.family":           "DejaVu Sans",
        "figure.dpi":            120,
        "savefig.dpi":           150,
        "savefig.bbox":          "tight",
        "savefig.facecolor":     PALETTE["background"],
    })


# Apply on import
set_style()


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _finalize(fig: Figure, title: str = "", tight: bool = True) -> Figure:
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold",
                     color=PALETTE["text"], y=1.02)
    if tight:
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    ax: Axes | None = None,
    cmap: str = "Blues",
) -> tuple[Figure, Axes]:
    """
    Vẽ confusion matrix với annotation % và count.

    Parameters
    ----------
    normalize : bool
        True → hiển thị % theo row (recall per class)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULTS["square"])
    else:
        fig = ax.get_figure()

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    if normalize:
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        annot = np.array([
            [f"{cm_pct[i, j]:.1%}\n({cm[i, j]})" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])
        vmin, vmax = 0, 1
    else:
        annot = cm.astype(str)
        cm_pct = cm
        vmin, vmax = None, None

    sns.heatmap(
        cm_pct, annot=annot, fmt="s",
        xticklabels=labels, yticklabels=labels,
        cmap=cmap, linewidths=0.5, linecolor=PALETTE["border"],
        vmin=vmin, vmax=vmax,
        annot_kws={"size": 10, "weight": "semibold"},
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", labelpad=8)
    ax.set_ylabel("True Label", labelpad=8)
    ax.set_title(title, pad=12)
    ax.tick_params(left=False, bottom=False)

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROC Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true_dict: dict[str, np.ndarray],
    y_proba_dict: dict[str, np.ndarray],
    colors: dict[str, str] | None = None,
    title: str = "ROC Curve",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Vẽ nhiều ROC curves cùng lúc (baseline vs biased).

    Parameters
    ----------
    y_true_dict : dict label → y_true array
    y_proba_dict : dict label → y_proba array
    colors : dict label → hex color (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULTS["square"])
    else:
        fig = ax.get_figure()

    default_colors = [PALETTE["baseline"], PALETTE["biased"],
                      PALETTE["positive"], PALETTE["warning"]]

    for i, (label, y_true) in enumerate(y_true_dict.items()):
        y_proba = y_proba_dict[label]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        color = (colors or {}).get(label, default_colors[i % len(default_colors)])
        lw = 2.5 if i == 0 else 1.8

        ax.plot(fpr, tpr, color=color, lw=lw, alpha=0.9,
                label=f"{label} (AUC = {roc_auc:.4f})")

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1,
            linestyle="--", alpha=0.5, label="Random (AUC = 0.50)")

    ax.fill_between([0, 1], [0, 1], alpha=0.03, color=PALETTE["neutral"])
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 3. Precision-Recall Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curve(
    y_true_dict: dict[str, np.ndarray],
    y_proba_dict: dict[str, np.ndarray],
    colors: dict[str, str] | None = None,
    title: str = "Precision-Recall Curve",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Vẽ Precision-Recall curves cho nhiều models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULTS["square"])
    else:
        fig = ax.get_figure()

    default_colors = [PALETTE["baseline"], PALETTE["biased"],
                      PALETTE["positive"], PALETTE["warning"]]

    for i, (label, y_true) in enumerate(y_true_dict.items()):
        y_proba = y_proba_dict[label]
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        color = (colors or {}).get(label, default_colors[i % len(default_colors)])

        ax.plot(recall, precision, color=color, lw=2.2 if i == 0 else 1.8,
                alpha=0.9, label=f"{label} (AP = {ap:.4f})")

    # Baseline: random classifier
    pos_rate = list(y_true_dict.values())[0].mean()
    ax.axhline(pos_rate, color=PALETTE["neutral"], lw=1, linestyle="--",
               alpha=0.5, label=f"Random (AP = {pos_rate:.2f})")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance",
    highlight_cols: list[str] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Horizontal bar chart cho feature importance.

    Parameters
    ----------
    highlight_cols : list[str]
        Tô màu đỏ các features bị inject bias (noise features, leaked features...)
    """
    if ax is None:
        h = min(0.35 * top_n + 1, 12)
        fig, ax = plt.subplots(figsize=(9, h))
    else:
        fig = ax.get_figure()

    # Sort và lấy top N
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    # Colors: đỏ cho highlighted features
    highlight_set = set(highlight_cols or [])
    colors = [PALETTE["biased"] if n in highlight_set else PALETTE["baseline"]
              for n in names]

    bars = ax.barh(range(len(names)), vals[::-1], color=colors[::-1],
                   height=0.6, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Importance Score")
    ax.set_title(title)

    # Legend nếu có highlighted
    if highlight_set:
        patches = [
            mpatches.Patch(color=PALETTE["baseline"], label="Normal feature"),
            mpatches.Patch(color=PALETTE["biased"],   label="Bias-injected feature"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=8)

    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 5. Class Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    y_dict: dict[str, pd.Series | np.ndarray],
    title: str = "Class Distribution",
    normalize: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    So sánh class distribution trước và sau khi apply bias.

    Parameters
    ----------
    y_dict : dict label → y array
        Ví dụ: {"Original": y_orig, "After Survivorship Bias": y_biased}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULTS["medium"])
    else:
        fig = ax.get_figure()

    n_datasets = len(y_dict)
    n_classes  = len(np.unique(list(y_dict.values())[0]))
    x = np.arange(n_classes)
    width = 0.7 / n_datasets

    bar_colors = [PALETTE["baseline"], PALETTE["biased"],
                  PALETTE["positive"], PALETTE["warning"]]

    for i, (label, y) in enumerate(y_dict.items()):
        y_arr = np.array(y)
        classes, counts = np.unique(y_arr, return_counts=True)
        vals = counts / counts.sum() if normalize else counts
        offset = (i - n_datasets / 2 + 0.5) * width

        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=label, color=bar_colors[i % len(bar_colors)],
                      alpha=0.85, edgecolor="white", linewidth=0.8)

        # Value labels
        for bar, v in zip(bars, vals):
            label_text = f"{v:.1%}" if normalize else str(int(v))
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    label_text, ha="center", va="bottom", fontsize=8,
                    color=PALETTE["text_muted"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.1 if normalize else None)

    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 6. Metric Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_comparison(
    metrics_dict: dict[str, dict],
    metrics_to_plot: list[str] | None = None,
    title: str = "Model Metrics Comparison",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Grouped bar chart so sánh metrics của nhiều models.

    Parameters
    ----------
    metrics_dict : dict model_label → metrics_dict
        Ví dụ: {"Baseline": {"accuracy": 0.85, "roc_auc": 0.91},
                "Biased":   {"accuracy": 0.92, "roc_auc": 0.94}}
    metrics_to_plot : list[str], optional
        Metrics muốn hiển thị. Default: accuracy, f1, roc_auc, precision, recall
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["accuracy", "f1", "roc_auc", "precision", "recall"]

    # Filter only available metrics
    available = set()
    for m in metrics_dict.values():
        available |= set(m.keys())
    metrics_to_plot = [m for m in metrics_to_plot if m in available]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(9, len(metrics_to_plot) * 1.5), 5))
    else:
        fig = ax.get_figure()

    models = list(metrics_dict.keys())
    n_models = len(models)
    x = np.arange(len(metrics_to_plot))
    width = 0.7 / n_models

    bar_colors = [PALETTE["baseline"], PALETTE["biased"],
                  PALETTE["positive"], PALETTE["warning"]]

    for i, model_label in enumerate(models):
        m = metrics_dict[model_label]
        vals = [m.get(metric, 0) for metric in metrics_to_plot]
        offset = (i - n_models / 2 + 0.5) * width

        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=model_label,
                      color=bar_colors[i % len(bar_colors)],
                      alpha=0.85, edgecolor="white", linewidth=0.8)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=PALETTE["text_muted"])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title(title)
    ax.legend(loc="lower right")

    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 7. Learning Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curve(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    title: str = "Learning Curve",
    cv: int = 5,
    scoring: str = "roc_auc",
    train_sizes: np.ndarray | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Vẽ learning curve — detect overfitting và underfitting.
    Đặc biệt hữu ích để visualize tác động của survivorship bias.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULTS["medium"])
    else:
        fig = ax.get_figure()

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    train_sz, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        cv=cv, scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=-1,
        shuffle=True, random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    ax.plot(train_sz, train_mean, "o-", color=PALETTE["baseline"],
            lw=2, label="Train score", markersize=5)
    ax.fill_between(train_sz,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.12, color=PALETTE["baseline"])

    ax.plot(train_sz, val_mean, "s-", color=PALETTE["biased"],
            lw=2, label="Validation score", markersize=5)
    ax.fill_between(train_sz,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.12, color=PALETTE["biased"])

    # Annotate gap at final point
    final_gap = train_mean[-1] - val_mean[-1]
    if abs(final_gap) > 0.005:
        ax.annotate(
            f"Gap: {final_gap:+.3f}",
            xy=(train_sz[-1], (train_mean[-1] + val_mean[-1]) / 2),
            xytext=(-60, 0), textcoords="offset points",
            fontsize=8, color=PALETTE["warning"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["warning"], lw=1.2),
        )

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(scoring.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=max(0, min(val_mean) - 0.1))

    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 8. Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    top_n_features: int = 20,
    title: str = "Feature Correlation Heatmap",
    figsize: tuple | None = None,
) -> tuple[Figure, Axes]:
    """
    Correlation heatmap — phát hiện multicollinearity và bias-injected features.

    Parameters
    ----------
    y : pd.Series, optional
        Nếu cung cấp, thêm cột correlation với target
    top_n_features : int
        Chỉ hiển thị top N features (theo variance)
    """
    if figsize is None:
        n = min(top_n_features, X.shape[1])
        figsize = (n * 0.55 + 1, n * 0.5 + 1)

    fig, ax = plt.subplots(figsize=figsize)

    # Chọn top N features theo variance
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(top_n_features).index.tolist()
    X_sub = X[top_features]

    if y is not None:
        df_plot = X_sub.copy()
        df_plot["[target]"] = y.values
    else:
        df_plot = X_sub

    corr = df_plot.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask

    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.3, linecolor=PALETTE["border"],
        annot=corr.shape[0] <= 15,
        annot_kws={"size": 6},
        fmt=".2f",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        square=True,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 9. Distribution Comparison (KDE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribution_comparison(
    data_dict: dict[str, np.ndarray | pd.Series],
    feature_name: str = "Feature",
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    KDE plot so sánh phân phối của một feature trước/sau bias.
    Dùng để visualize distribution shift do survivorship/selection bias.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULTS["medium"])
    else:
        fig = ax.get_figure()

    colors = [PALETTE["baseline"], PALETTE["biased"],
              PALETTE["positive"], PALETTE["warning"]]

    for i, (label, data) in enumerate(data_dict.items()):
        arr = np.array(data).flatten()
        color = colors[i % len(colors)]
        sns.kdeplot(arr, ax=ax, color=color, lw=2.2,
                    label=f"{label} (μ={arr.mean():.3f}, σ={arr.std():.3f})",
                    fill=True, alpha=0.1)
        ax.axvline(arr.mean(), color=color, lw=1, linestyle="--", alpha=0.6)

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.set_title(title or f"Distribution: {feature_name}")
    ax.legend()

    fig.tight_layout()
    return fig, ax