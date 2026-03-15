"""
src/simulation/confirmation_bias.py
─────────────────────────────────────────────────────────────────────────────
Simulation Confirmation Bias trong Data Science workflow.

Confirmation bias xảy ra khi data scientist:
  1. Chỉ chọn features ủng hộ hypothesis có sẵn (feature cherry-picking)
  2. Chỉ giữ lại samples phù hợp với kỳ vọng (sample filtering)
  3. Dừng thử nghiệm sớm khi thấy kết quả mong muốn (early stopping bias)
  4. Chỉ nhìn vào subset dữ liệu xác nhận belief (subgroup selection)

Public API:
    cherry_pick_features(X, y, keep_top_n, favor_positive)
    filter_confirming_samples(X, y, threshold, keep_above)
    biased_feature_correlation(X, y, min_corr)
    simulate_confirmation_bias(bundle, strategy, **kwargs) → BiasedBundle
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression

from src.data.load_data import DataBundle

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BiasedBundle:
    """DataBundle sau khi áp dụng bias simulation."""
    original: DataBundle
    biased: DataBundle
    bias_type: str
    bias_params: dict = field(default_factory=dict)
    bias_report: dict = field(default_factory=dict)  # thống kê tác động

    def __repr__(self) -> str:
        orig_rows, orig_cols = self.original.X.shape
        bias_rows, bias_cols = self.biased.X.shape
        return (
            f"BiasedBundle(bias='{self.bias_type}', "
            f"rows: {orig_rows}→{bias_rows}, "
            f"cols: {orig_cols}→{bias_cols})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Feature Cherry-Picking
# ─────────────────────────────────────────────────────────────────────────────

def cherry_pick_features(
    X: pd.DataFrame,
    y: pd.Series,
    keep_top_n: int = 5,
    task: Literal["classification", "regression"] = "classification",
    random_noise_ratio: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
    """
    Chỉ giữ lại N features có F-score cao nhất — bỏ qua features thực sự hữu ích
    nhưng không khớp hypothesis.

    Bias được inject thêm bằng cách:
    - Giữ một số "noise features" trông có vẻ quan trọng (F-score cao do may mắn)
    - Loại bỏ features hữu ích nhưng F-score thấp hơn threshold

    Parameters
    ----------
    keep_top_n : int
        Số features giữ lại (data scientist bị bias chỉ nhìn top N)
    random_noise_ratio : float
        Tỉ lệ features noise thêm vào để thay thế features bị loại

    Returns
    -------
    X_biased, report
    """
    score_fn = f_classif if task == "classification" else f_regression
    scores, _ = score_fn(X, y)

    feature_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)

    # Top N features (confirmation: chỉ nhìn những gì confirm hypothesis)
    selected = feature_scores.head(keep_top_n).index.tolist()

    # Inject noise: thêm vài features random trông "có vẻ significant"
    n_noise = max(1, int(keep_top_n * random_noise_ratio))
    rng = np.random.default_rng(seed=42)
    noise_cols = {}
    for i in range(n_noise):
        col_name = f"noise_confirm_{i:02d}"
        # Noise có correlation giả tạo với y
        noise_cols[col_name] = rng.normal(0, 1, size=len(X))

    X_biased = X[selected].copy()
    for col, vals in noise_cols.items():
        X_biased[col] = vals

    report = {
        "strategy": "cherry_pick_features",
        "original_n_features": X.shape[1],
        "kept_features": selected,
        "dropped_features": [c for c in X.columns if c not in selected],
        "noise_features_added": list(noise_cols.keys()),
        "top_feature_scores": feature_scores.head(keep_top_n).to_dict(),
        "bottom_feature_scores": feature_scores.tail(5).to_dict(),
    }

    logger.info(
        "Cherry-picked %d/%d features, added %d noise features",
        len(selected), X.shape[1], n_noise,
    )
    return X_biased, report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Sample Filtering (loại bỏ "outliers" có chủ đích)
# ─────────────────────────────────────────────────────────────────────────────

def filter_confirming_samples(
    X: pd.DataFrame,
    y: pd.Series,
    remove_class: int = 0,
    remove_fraction: float = 0.4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Loại bỏ một phần samples của class không mong muốn — giả vờ chúng là "outliers".

    Ví dụ thực tế: data scientist muốn prove model tốt cho class 1,
    nên loại bỏ bớt samples class 0 với lý do "dữ liệu bẩn".

    Parameters
    ----------
    remove_class : int
        Class bị loại bỏ bớt samples
    remove_fraction : float
        Tỉ lệ samples của remove_class bị drop (0.0–1.0)
    """
    rng = np.random.default_rng(seed=random_state)

    class_mask = y == remove_class
    class_indices = np.where(class_mask)[0]

    n_remove = int(len(class_indices) * remove_fraction)
    remove_idx = rng.choice(class_indices, size=n_remove, replace=False)
    keep_mask = ~pd.Series(range(len(X))).isin(remove_idx).values

    X_biased = X.iloc[keep_mask].reset_index(drop=True)
    y_biased = y.iloc[keep_mask].reset_index(drop=True)

    orig_dist = y.value_counts().to_dict()
    new_dist  = y_biased.value_counts().to_dict()

    report = {
        "strategy": "filter_confirming_samples",
        "remove_class": remove_class,
        "remove_fraction": remove_fraction,
        "original_class_distribution": orig_dist,
        "biased_class_distribution": new_dist,
        "samples_removed": n_remove,
        "original_total": len(X),
        "biased_total": len(X_biased),
    }

    logger.info(
        "Filtered %d samples of class %d (%.0f%%) — dist: %s → %s",
        n_remove, remove_class, remove_fraction * 100, orig_dist, new_dist,
    )
    return X_biased, y_biased, report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Correlation Threshold Bias
# ─────────────────────────────────────────────────────────────────────────────

def biased_feature_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    min_corr: float = 0.1,
    direction: Literal["positive", "negative", "both"] = "positive",
) -> tuple[pd.DataFrame, dict]:
    """
    Chỉ giữ features có correlation với y theo hướng data scientist "muốn thấy".

    Ví dụ: data scientist có hypothesis "feature cao → outcome tốt",
    nên loại bỏ features có correlation âm dù chúng predictive.

    Parameters
    ----------
    direction : str
        'positive' → chỉ giữ corr > min_corr
        'negative' → chỉ giữ corr < -min_corr
        'both'     → giữ |corr| > min_corr (ít bias nhất)
    """
    correlations = X.corrwith(y)

    if direction == "positive":
        keep_mask = correlations >= min_corr
    elif direction == "negative":
        keep_mask = correlations <= -min_corr
    else:
        keep_mask = correlations.abs() >= min_corr

    selected = correlations[keep_mask].index.tolist()

    if not selected:
        logger.warning("Không có feature nào vượt threshold corr=%.2f (%s). Giữ all.", min_corr, direction)
        selected = X.columns.tolist()

    X_biased = X[selected].copy()

    report = {
        "strategy": "biased_feature_correlation",
        "direction": direction,
        "min_corr": min_corr,
        "all_correlations": correlations.to_dict(),
        "kept_features": selected,
        "dropped_features": [c for c in X.columns if c not in selected],
        "n_kept": len(selected),
        "n_dropped": X.shape[1] - len(selected),
    }

    logger.info(
        "Correlation filter (%s, min=%.2f): kept %d/%d features",
        direction, min_corr, len(selected), X.shape[1],
    )
    return X_biased, report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: Subgroup Selection
# ─────────────────────────────────────────────────────────────────────────────

def select_confirming_subgroup(
    X: pd.DataFrame,
    y: pd.Series,
    feature: str,
    percentile_range: tuple[float, float] = (40.0, 100.0),
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Chỉ báo cáo kết quả trên subgroup "đẹp" của dữ liệu.

    Ví dụ: model chỉ hoạt động tốt với customers có income cao,
    data scientist chỉ present kết quả trên subgroup đó.

    Parameters
    ----------
    feature : str
        Feature dùng để slice subgroup
    percentile_range : tuple
        (lower, upper) percentile — chỉ giữ rows trong khoảng này
    """
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' không tồn tại trong X.")

    lo = np.percentile(X[feature], percentile_range[0])
    hi = np.percentile(X[feature], percentile_range[1])
    mask = (X[feature] >= lo) & (X[feature] <= hi)

    X_biased = X[mask].reset_index(drop=True)
    y_biased = y[mask].reset_index(drop=True)

    report = {
        "strategy": "select_confirming_subgroup",
        "feature": feature,
        "percentile_range": percentile_range,
        "value_range": (float(lo), float(hi)),
        "original_n": len(X),
        "subgroup_n": len(X_biased),
        "kept_fraction": len(X_biased) / len(X),
        "original_class_dist": y.value_counts(normalize=True).to_dict(),
        "subgroup_class_dist": y_biased.value_counts(normalize=True).to_dict(),
    }

    logger.info(
        "Subgroup '%s' [p%.0f–p%.0f]: %d→%d rows (%.1f%%)",
        feature, *percentile_range, len(X), len(X_biased),
        100 * len(X_biased) / len(X),
    )
    return X_biased, y_biased, report


# ─────────────────────────────────────────────────────────────────────────────
# Main Simulation Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def simulate_confirmation_bias(
    bundle: DataBundle,
    strategy: Literal[
        "cherry_pick_features",
        "filter_confirming_samples",
        "biased_feature_correlation",
        "select_confirming_subgroup",
    ] = "cherry_pick_features",
    **kwargs,
) -> BiasedBundle:
    """
    Entry point duy nhất cho confirmation bias simulation.

    Parameters
    ----------
    bundle : DataBundle
        Dataset gốc từ load_data.py
    strategy : str
        Loại confirmation bias muốn simulate
    **kwargs :
        Tham số truyền vào strategy tương ứng

    Examples
    --------
    >>> from src.data.load_data import load_dataset
    >>> from src.simulation.confirmation_bias import simulate_confirmation_bias
    >>> bundle = load_dataset("synthetic_clf")
    >>> biased = simulate_confirmation_bias(bundle, strategy="cherry_pick_features", keep_top_n=5)
    >>> print(biased)
    """
    X, y = bundle.X, bundle.y

    if strategy == "cherry_pick_features":
        X_b, report = cherry_pick_features(X, y, **kwargs)
        y_b = y.copy()

    elif strategy == "filter_confirming_samples":
        X_b, y_b, report = filter_confirming_samples(X, y, **kwargs)

    elif strategy == "biased_feature_correlation":
        X_b, report = biased_feature_correlation(X, y, **kwargs)
        y_b = y.copy()

    elif strategy == "select_confirming_subgroup":
        first_feature = kwargs.pop("feature", X.columns[0])
        X_b, y_b, report = select_confirming_subgroup(X, y, feature=first_feature, **kwargs)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    biased_bundle = DataBundle(
        X=X_b,
        y=y_b,
        meta={**bundle.meta, "bias_type": "confirmation_bias", "bias_strategy": strategy},
    )

    return BiasedBundle(
        original=bundle,
        biased=biased_bundle,
        bias_type="confirmation_bias",
        bias_params={"strategy": strategy, **kwargs},
        bias_report=report,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data.load_data import load_dataset

    bundle = load_dataset("synthetic_clf", n_samples=1_000)

    for strat in ["cherry_pick_features", "filter_confirming_samples",
                  "biased_feature_correlation", "select_confirming_subgroup"]:
        kwargs = {}
        if strat == "select_confirming_subgroup":
            kwargs["feature"] = bundle.X.columns[0]
        result = simulate_confirmation_bias(bundle, strategy=strat, **kwargs)
        print(result)
        print("  Report keys:", list(result.bias_report.keys()))