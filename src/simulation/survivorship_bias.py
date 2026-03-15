"""
src/simulation/survivorship_bias.py
─────────────────────────────────────────────────────────────────────────────
Simulation Survivorship Bias trong Data Science workflow.

Survivorship bias xảy ra khi chỉ nhìn vào "survivors" — những cases
đã vượt qua một filter nào đó — và bỏ qua toàn bộ cases thất bại.

Ví dụ kinh điển:
  - Phân tích chỉ dùng công ty còn tồn tại (bỏ qua công ty đã phá sản)
  - Model chỉ train trên khách hàng còn active (bỏ qua churned customers)
  - Backtesting trading strategy chỉ trên stocks còn niêm yết
  - Đánh giá model chỉ trên users đã complete flow (bỏ qua drop-offs)

Public API:
    remove_failures(X, y, failure_class, removal_rate)
    apply_historical_filter(X, y, time_col, cutoff_percentile)
    simulate_selection_filter(X, y, feature, threshold, keep_above)
    simulate_survivorship_bias(bundle, strategy, **kwargs) → BiasedBundle
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.data.load_data import DataBundle
from src.simulation.confirmation_bias import BiasedBundle

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Remove Failures Directly
# ─────────────────────────────────────────────────────────────────────────────

def remove_failures(
    X: pd.DataFrame,
    y: pd.Series,
    failure_class: int = 0,
    removal_rate: float = 0.85,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Loại bỏ phần lớn samples của "failure class" — chỉ giữ lại survivors.

    Ví dụ: trong credit scoring, chỉ train trên khách hàng đã trả được nợ
    (class 1), loại bỏ hầu hết khách hàng default (class 0).

    Parameters
    ----------
    failure_class : int
        Class đại diện cho "failures" (thường là class 0)
    removal_rate : float
        Tỉ lệ failures bị loại (0.5–1.0 để tạo bias rõ ràng)
    """
    rng = np.random.default_rng(seed=random_state)

    failure_idx = np.where(y == failure_class)[0]
    n_remove = int(len(failure_idx) * removal_rate)
    removed_idx = rng.choice(failure_idx, size=n_remove, replace=False)

    keep_mask = ~np.isin(np.arange(len(X)), removed_idx)
    X_b = X.iloc[keep_mask].reset_index(drop=True)
    y_b = y.iloc[keep_mask].reset_index(drop=True)

    orig_dist = y.value_counts().to_dict()
    new_dist  = y_b.value_counts().to_dict()

    # Tính imbalance ratio
    orig_ratio = orig_dist.get(1, 0) / max(orig_dist.get(0, 1), 1)
    new_ratio  = new_dist.get(1, 0) / max(new_dist.get(0, 1), 1)

    report = {
        "strategy": "remove_failures",
        "failure_class": failure_class,
        "removal_rate": removal_rate,
        "n_removed": n_remove,
        "original_total": len(X),
        "biased_total": len(X_b),
        "original_class_dist": orig_dist,
        "biased_class_dist": new_dist,
        "original_positive_ratio": round(orig_ratio, 4),
        "biased_positive_ratio": round(new_ratio, 4),
        "bias_magnitude": round(new_ratio - orig_ratio, 4),
    }

    logger.info(
        "Removed %d failures (%.0f%%) — ratio %.2f→%.2f",
        n_remove, removal_rate * 100, orig_ratio, new_ratio,
    )
    return X_b, y_b, report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Historical Survivorship Filter
# ─────────────────────────────────────────────────────────────────────────────

def apply_historical_filter(
    X: pd.DataFrame,
    y: pd.Series,
    performance_feature: str | None = None,
    cutoff_percentile: float = 30.0,
    keep_above: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Simulate survivorship bias kiểu "historical data":
    Chỉ giữ lại entities vượt qua một ngưỡng performance.

    Ví dụ: Phân tích hedge funds — chỉ dùng data của funds còn tồn tại
    (funds có return > threshold), bỏ qua funds đã đóng cửa.

    Parameters
    ----------
    performance_feature : str, optional
        Feature đại diện cho "performance". Nếu None, dùng feature đầu tiên.
    cutoff_percentile : float
        Ngưỡng percentile để "survive" (vd: 30 = chỉ giữ top/bottom 70%)
    keep_above : bool
        True → giữ rows trên ngưỡng (survivors có performance cao)
        False → giữ rows dưới ngưỡng
    """
    feat = performance_feature or X.columns[0]
    if feat not in X.columns:
        raise ValueError(f"Feature '{feat}' không tồn tại.")

    cutoff_value = np.percentile(X[feat], cutoff_percentile)

    if keep_above:
        mask = X[feat] >= cutoff_value
    else:
        mask = X[feat] <= cutoff_value

    X_b = X[mask].reset_index(drop=True)
    y_b = y[mask].reset_index(drop=True)

    report = {
        "strategy": "apply_historical_filter",
        "performance_feature": feat,
        "cutoff_percentile": cutoff_percentile,
        "cutoff_value": float(cutoff_value),
        "keep_above": keep_above,
        "original_n": len(X),
        "survived_n": int(mask.sum()),
        "eliminated_n": int((~mask).sum()),
        "survival_rate": float(mask.mean()),
        "original_class_dist": y.value_counts(normalize=True).to_dict(),
        "survived_class_dist": y_b.value_counts(normalize=True).to_dict(),
        "feature_mean_before": float(X[feat].mean()),
        "feature_mean_after": float(X_b[feat].mean()),
    }

    logger.info(
        "Historical filter on '%s' (p%.0f=%+.2f, keep_above=%s): %d→%d rows",
        feat, cutoff_percentile, cutoff_value, keep_above, len(X), len(X_b),
    )
    return X_b, y_b, report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Selection Filter (simulate data collection bias)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_selection_filter(
    X: pd.DataFrame,
    y: pd.Series,
    filter_feature: str | None = None,
    threshold: float | None = None,
    keep_above: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Simulate bias trong quá trình thu thập dữ liệu:
    Chỉ record được data của entities đáp ứng một điều kiện nào đó.

    Ví dụ:
    - Chỉ survey khách hàng mua hàng (bỏ qua người không mua)
    - Chỉ collect data từ users active (bỏ qua churned users)
    - Chỉ test product trên users đã sign up (bỏ qua users bỏ đi trước)

    Parameters
    ----------
    filter_feature : str, optional
        Feature dùng làm điều kiện filter. Nếu None, dùng feature đầu tiên.
    threshold : float, optional
        Ngưỡng filter. Nếu None, dùng median.
    keep_above : bool
        True → giữ rows có feature > threshold (survivors)
    """
    rng = np.random.default_rng(seed=random_state)

    feat = filter_feature or X.columns[0]
    thresh = threshold if threshold is not None else float(X[feat].median())

    # Hard filter: loại bỏ hoàn toàn
    if keep_above:
        hard_mask = X[feat] >= thresh
    else:
        hard_mask = X[feat] <= thresh

    # Soft noise: một số borderline cases bị miss (simulate imperfect collection)
    borderline_range = X[feat].std() * 0.2
    borderline = (X[feat] - thresh).abs() <= borderline_range
    noise_drop = rng.random(size=len(X)) < 0.3  # 30% borderline bị drop ngẫu nhiên

    final_mask = hard_mask & ~(borderline & noise_drop)

    X_b = X[final_mask].reset_index(drop=True)
    y_b = y[final_mask].reset_index(drop=True)

    report = {
        "strategy": "simulate_selection_filter",
        "filter_feature": feat,
        "threshold": float(thresh),
        "keep_above": keep_above,
        "original_n": len(X),
        "filtered_n": len(X_b),
        "hard_filtered": int((~hard_mask).sum()),
        "noise_dropped": int((borderline & noise_drop & hard_mask).sum()),
        "survival_rate": round(len(X_b) / len(X), 4),
        "feature_stats_before": {
            "mean": float(X[feat].mean()),
            "std": float(X[feat].std()),
            "median": float(X[feat].median()),
        },
        "feature_stats_after": {
            "mean": float(X_b[feat].mean()),
            "std": float(X_b[feat].std()),
            "median": float(X_b[feat].median()),
        },
        "class_dist_before": y.value_counts(normalize=True).to_dict(),
        "class_dist_after": y_b.value_counts(normalize=True).to_dict(),
    }

    logger.info(
        "Selection filter '%s'>%.2f: %d→%d rows (%.1f%% survived)",
        feat, thresh, len(X), len(X_b), 100 * len(X_b) / len(X),
    )
    return X_b, y_b, report


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: Temporal Survivorship (simulate look-ahead bias)
# ─────────────────────────────────────────────────────────────────────────────

def inject_lookahead_bias(
    X: pd.DataFrame,
    y: pd.Series,
    n_future_features: int = 3,
    leakage_strength: float = 0.7,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Inject look-ahead bias: thêm features chứa thông tin từ tương lai.

    Ví dụ trong finance: dùng closing price của ngày T để predict
    movement của ngày T (thay vì T-1) — thông tin không có tại thời điểm predict.

    Parameters
    ----------
    n_future_features : int
        Số features "tương lai" được inject
    leakage_strength : float
        Mức độ correlation giữa future feature và target (0–1)
    """
    rng = np.random.default_rng(seed=random_state)

    X_b = X.copy()
    injected_cols = []

    for i in range(n_future_features):
        col_name = f"future_leak_{i:02d}"
        # Feature có correlation cao với y → simulate information leakage
        noise = rng.normal(0, 1, size=len(y))
        leaked_feature = leakage_strength * (y.values * 2 - 1) + (1 - leakage_strength) * noise
        X_b[col_name] = leaked_feature
        injected_cols.append(col_name)

    report = {
        "strategy": "inject_lookahead_bias",
        "n_future_features": n_future_features,
        "leakage_strength": leakage_strength,
        "injected_columns": injected_cols,
        "warning": (
            "Features này chứa thông tin target — model sẽ overfit "
            "và fail hoàn toàn trên real-world data."
        ),
    }

    logger.warning(
        "⚠ Injected %d look-ahead features (leakage=%.1f) — FOR SIMULATION ONLY",
        n_future_features, leakage_strength,
    )
    return X_b, y.copy(), report


# ─────────────────────────────────────────────────────────────────────────────
# Main Simulation Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def simulate_survivorship_bias(
    bundle: DataBundle,
    strategy: Literal[
        "remove_failures",
        "apply_historical_filter",
        "simulate_selection_filter",
        "inject_lookahead_bias",
    ] = "remove_failures",
    **kwargs,
) -> BiasedBundle:
    """
    Entry point duy nhất cho survivorship bias simulation.

    Examples
    --------
    >>> bundle = load_dataset("synthetic_clf")
    >>> biased = simulate_survivorship_bias(bundle, strategy="remove_failures", removal_rate=0.9)
    >>> print(biased)
    >>> print(biased.bias_report)
    """
    X, y = bundle.X, bundle.y

    if strategy == "remove_failures":
        X_b, y_b, report = remove_failures(X, y, **kwargs)

    elif strategy == "apply_historical_filter":
        X_b, y_b, report = apply_historical_filter(X, y, **kwargs)

    elif strategy == "simulate_selection_filter":
        X_b, y_b, report = simulate_selection_filter(X, y, **kwargs)

    elif strategy == "inject_lookahead_bias":
        X_b, y_b, report = inject_lookahead_bias(X, y, **kwargs)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    biased_bundle = DataBundle(
        X=X_b,
        y=y_b,
        meta={**bundle.meta, "bias_type": "survivorship_bias", "bias_strategy": strategy},
    )

    return BiasedBundle(
        original=bundle,
        biased=biased_bundle,
        bias_type="survivorship_bias",
        bias_params={"strategy": strategy, **kwargs},
        bias_report=report,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data.load_data import load_dataset

    bundle = load_dataset("synthetic_clf", n_samples=2_000, class_imbalance=0.3)

    strategies = [
        ("remove_failures",          {"removal_rate": 0.85}),
        ("apply_historical_filter",  {"cutoff_percentile": 25.0}),
        ("simulate_selection_filter",{}),
        ("inject_lookahead_bias",    {"n_future_features": 2, "leakage_strength": 0.8}),
    ]

    for strat, kw in strategies:
        result = simulate_survivorship_bias(bundle, strategy=strat, **kw)
        print(result)
        orig_n = result.bias_report.get("original_total") or result.bias_report.get("original_n")
        biased_n = result.bias_report.get("biased_total") or result.bias_report.get("filtered_n")
        if orig_n and biased_n:
            print(f"  Rows: {orig_n} → {biased_n}")
        print()