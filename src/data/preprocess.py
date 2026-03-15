"""
src/data/preprocess.py
─────────────────────────────────────────────────────────────────────────────
Pipeline tiền xử lý dữ liệu: clean → encode → scale → split.

Thiết kế theo sklearn Pipeline để:
  - Không bị data leakage (fit chỉ trên train, transform trên test)
  - Dễ tái sử dụng trong các bias simulation experiments
  - Reproducible qua random_state

Public API:
    build_preprocessor(X)         → sklearn ColumnTransformer
    split_data(bundle)             → TrainTestBundle
    preprocess(bundle)             → TrainTestBundle (end-to-end)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.data.load_data import DataBundle

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainTestBundle:
    """Kết quả sau khi split + transform."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer  # đã fit trên train
    feature_names: list[str]
    meta: dict

    def __repr__(self) -> str:
        return (
            f"TrainTestBundle("
            f"train={len(self.X_train)}, test={len(self.X_test)}, "
            f"features={len(self.feature_names)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Column Type Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_column_types(
    X: pd.DataFrame,
    cat_threshold: int = 20,
) -> tuple[list[str], list[str], list[str]]:
    """
    Tự động phân loại cột thành: numeric, low-cardinality categorical, high-cardinality.

    Parameters
    ----------
    cat_threshold : int
        Số unique values tối đa để coi là low-cardinality (dùng OneHot).
        Vượt quá → OrdinalEncoder (tránh feature explosion).

    Returns
    -------
    numeric_cols, onehot_cols, ordinal_cols
    """
    numeric_cols, onehot_cols, ordinal_cols = [], [], []

    for col in X.columns:
        dtype = X[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            n_unique = X[col].nunique()
            if n_unique <= cat_threshold:
                onehot_cols.append(col)
            else:
                ordinal_cols.append(col)
        else:
            # datetime, bool, etc. → treat as numeric sau khi cast
            numeric_cols.append(col)

    logger.debug(
        "Column types — numeric: %d, onehot: %d, ordinal: %d",
        len(numeric_cols), len(onehot_cols), len(ordinal_cols),
    )
    return numeric_cols, onehot_cols, ordinal_cols


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessor(
    X: pd.DataFrame,
    numeric_strategy: str = "mean",     # chiến lược impute numeric: mean | median | most_frequent
    scale_numeric: bool = True,
    cat_threshold: int = 20,
) -> ColumnTransformer:
    """
    Xây dựng sklearn ColumnTransformer phù hợp với schema của X.

    Pipeline cho mỗi loại cột:
    - Numeric  : Impute (mean/median) → StandardScaler (tùy chọn)
    - Low-card : Impute (most_frequent) → OneHotEncoder
    - High-card: Impute (most_frequent) → OrdinalEncoder

    Notes
    -----
    Chưa fit — gọi .fit_transform(X_train) sau.
    """
    numeric_cols, onehot_cols, ordinal_cols = detect_column_types(X, cat_threshold)

    # ── Numeric pipeline ──────────────────────────────────────────────────────
    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_strategy))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    # ── Categorical (low-card) pipeline ───────────────────────────────────────
    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop="if_binary",       # tránh multicollinearity cho binary cols
        )),
    ])

    # ── Categorical (high-card) pipeline ─────────────────────────────────────
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    # ── Compose ───────────────────────────────────────────────────────────────
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if onehot_cols:
        transformers.append(("onehot", onehot_pipeline, onehot_cols))
    if ordinal_cols:
        transformers.append(("ordinal", ordinal_pipeline, ordinal_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",       # drop cột không được khai báo
        verbose_feature_names_out=True,
    )

    logger.info(
        "Built preprocessor — %d numeric, %d onehot, %d ordinal cols",
        len(numeric_cols), len(onehot_cols), len(ordinal_cols),
    )
    return preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Target Encoder
# ─────────────────────────────────────────────────────────────────────────────

def encode_target(y: pd.Series) -> tuple[pd.Series, Optional[LabelEncoder]]:
    """
    Encode target nếu là string/categorical → integer.
    Trả về (y_encoded, encoder) — encoder = None nếu y đã là numeric.
    """
    if pd.api.types.is_numeric_dtype(y):
        return y, None

    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y), name=y.name, index=y.index)
    logger.info("Encoded target: %s → %s", dict(zip(le.classes_, le.transform(le.classes_))), "")
    return y_encoded, le


# ─────────────────────────────────────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    bundle: DataBundle,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train/test split với option stratify (cho classification).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    y = bundle.y

    strat = y if stratify and y.nunique() <= 20 else None
    if stratify and strat is None:
        logger.warning("Stratify bị tắt: target có >20 unique values (regression?).")

    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X, y,
        test_size=test_size,
        stratify=strat,
        random_state=random_state,
    )

    logger.info(
        "Split: train=%d (%.0f%%), test=%d (%.0f%%)",
        len(X_train), 100 * (1 - test_size),
        len(X_test), 100 * test_size,
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    bundle: DataBundle,
    test_size: float = 0.2,
    stratify: bool = True,
    numeric_strategy: str = "mean",
    scale_numeric: bool = True,
    cat_threshold: int = 20,
    random_state: int = 42,
) -> TrainTestBundle:
    """
    Pipeline hoàn chỉnh: DataBundle → TrainTestBundle.

    Thứ tự:
      1. Encode target (nếu cần)
      2. Train/test split  ← split TRƯỚC khi fit preprocessor (chống leakage)
      3. Fit preprocessor trên X_train
      4. Transform X_train và X_test
      5. Trả về TrainTestBundle

    Examples
    --------
    >>> from src.data.load_data import load_dataset
    >>> from src.data.preprocess import preprocess
    >>> bundle = load_dataset("synthetic_clf")
    >>> tt = preprocess(bundle)
    >>> tt.X_train.shape
    """
    # 1. Encode target
    y_encoded, target_encoder = encode_target(bundle.y)
    bundle_clean = DataBundle(X=bundle.X, y=y_encoded, meta=bundle.meta)

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(
        bundle_clean, test_size=test_size, stratify=stratify, random_state=random_state
    )

    # 3. Build + fit preprocessor trên TRAIN only
    preprocessor = build_preprocessor(
        X_train,
        numeric_strategy=numeric_strategy,
        scale_numeric=scale_numeric,
        cat_threshold=cat_threshold,
    )
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr  = preprocessor.transform(X_test)

    # 4. Lấy tên features sau transform
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_train_arr.shape[1])]

    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
    X_test_df  = pd.DataFrame(X_test_arr,  columns=feature_names, index=X_test.index)

    # 5. Meta
    meta = {
        **bundle.meta,
        "test_size": test_size,
        "n_train": len(X_train_df),
        "n_test": len(X_test_df),
        "n_features_out": len(feature_names),
        "target_encoder": target_encoder,
        "numeric_strategy": numeric_strategy,
        "scale_numeric": scale_numeric,
        "random_state": random_state,
    }

    logger.info(
        "Preprocessing done → %d train / %d test / %d features",
        len(X_train_df), len(X_test_df), len(feature_names),
    )
    return TrainTestBundle(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        preprocessor=preprocessor,
        feature_names=feature_names,
        meta=meta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data.load_data import load_dataset

    bundle = load_dataset("synthetic_clf", n_samples=1_000, class_imbalance=0.25)
    tt = preprocess(bundle)

    print(tt)
    print("X_train shape :", tt.X_train.shape)
    print("y_train dist  :", tt.y_train.value_counts().to_dict())
    print("Features[:5]  :", tt.feature_names[:5])