"""
src/data/load_data.py
─────────────────────────────────────────────────────────────────────────────
Load datasets cho behavioral bias experiments.

Hỗ trợ 3 nguồn:
  1. Synthetic data  — sklearn.make_classification / make_regression
  2. UCI Repository  — dùng ucimlrepo
  3. Kaggle          — dùng Kaggle API (cần kaggle.json)

Tất cả hàm đều trả về:
    X : pd.DataFrame  — features
    y : pd.Series     — target
    meta : dict       — thông tin dataset (tên, nguồn, số dòng/cột...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataBundle:
    """Gói gọn X, y và metadata — truyền giữa các bước pipeline."""
    X: pd.DataFrame
    y: pd.Series
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"DataBundle(rows={len(self.X)}, features={self.X.shape[1]}, "
            f"source='{self.meta.get('source', 'unknown')}')"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Data
# ─────────────────────────────────────────────────────────────────────────────

def load_synthetic_classification(
    n_samples: int = 2_000,
    n_features: int = 20,
    n_informative: int = 8,
    n_redundant: int = 4,
    class_imbalance: float = 0.3,   # tỉ lệ class minority (0–0.5)
    random_state: int = 42,
) -> DataBundle:
    """
    Tạo dataset phân loại tổng hợp.

    Parameters
    ----------
    class_imbalance : float
        Tỉ lệ class 1 (minority). Dùng để demo survivorship bias sau này.
    """
    weights = [1 - class_imbalance, class_imbalance]

    X_arr, y_arr = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        weights=weights,
        flip_y=0.05,            # thêm noise nhẹ cho thực tế hơn
        random_state=random_state,
    )

    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="target")

    meta = {
        "source": "synthetic_classification",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_informative": n_informative,
        "class_imbalance": class_imbalance,
        "class_distribution": y.value_counts().to_dict(),
        "random_state": random_state,
    }

    logger.info("Loaded synthetic classification: %d rows, %d features", n_samples, n_features)
    return DataBundle(X=X, y=y, meta=meta)


def load_synthetic_regression(
    n_samples: int = 2_000,
    n_features: int = 15,
    n_informative: int = 6,
    noise: float = 0.1,
    random_state: int = 42,
) -> DataBundle:
    """
    Tạo dataset hồi quy tổng hợp.
    Dùng để demo overconfidence bias (model quá tự tin vào prediction interval).
    """
    X_arr, y_arr = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
    )

    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="target")

    meta = {
        "source": "synthetic_regression",
        "n_samples": n_samples,
        "n_features": n_features,
        "noise": noise,
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "random_state": random_state,
    }

    logger.info("Loaded synthetic regression: %d rows, %d features", n_samples, n_features)
    return DataBundle(X=X, y=y, meta=meta)


# ─────────────────────────────────────────────────────────────────────────────
# 2. UCI Repository
# ─────────────────────────────────────────────────────────────────────────────

# Map tên thân thiện → UCI dataset ID
UCI_DATASETS: dict[str, int] = {
    "adult":        2,    # income prediction — tốt cho confirmation bias
    "heart_disease": 45,  # binary classification
    "wine":         186,  # multi-class
    "credit":       144,  # credit approval — tốt cho survivorship bias
}


def load_uci(
    name: str = "adult",
    target_col: str | None = None,
) -> DataBundle:
    """
    Load dataset từ UCI ML Repository.

    Parameters
    ----------
    name : str
        Tên dataset (xem UCI_DATASETS) hoặc truyền thẳng UCI ID (int as str).
    target_col : str, optional
        Tên cột target. Nếu None, tự động lấy cột cuối.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError("Chạy: pip install ucimlrepo")

    dataset_id = UCI_DATASETS.get(name, int(name))
    logger.info("Fetching UCI dataset id=%d ...", dataset_id)

    repo = fetch_ucirepo(id=dataset_id)
    X: pd.DataFrame = repo.data.features
    y_raw = repo.data.targets

    # Flatten target nếu là DataFrame nhiều cột
    if isinstance(y_raw, pd.DataFrame):
        col = target_col or y_raw.columns[-1]
        y = y_raw[col].squeeze()
    else:
        y = y_raw.squeeze()
    y.name = "target"

    meta = {
        "source": f"uci_{name}",
        "uci_id": dataset_id,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "uci_metadata": repo.metadata if hasattr(repo, "metadata") else {},
    }

    logger.info("Loaded UCI '%s': %d rows, %d features", name, len(X), X.shape[1])
    return DataBundle(X=X, y=y, meta=meta)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Kaggle
# ─────────────────────────────────────────────────────────────────────────────

def load_kaggle(
    competition_or_dataset: str,
    file_name: str,
    target_col: str,
    download_dir: str = "data/raw",
    is_competition: bool = True,
) -> DataBundle:
    """
    Download và load CSV từ Kaggle.

    Yêu cầu: file ~/.kaggle/kaggle.json hoặc biến môi trường
    KAGGLE_USERNAME và KAGGLE_KEY.

    Parameters
    ----------
    competition_or_dataset : str
        Slug của competition (vd: 'titanic') hoặc dataset (vd: 'user/dataset').
    file_name : str
        Tên file CSV cần download (vd: 'train.csv').
    target_col : str
        Tên cột target trong CSV.
    """
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError("Chạy: pip install kaggle")

    import os
    import subprocess

    os.makedirs(download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, file_name)

    if not os.path.exists(file_path):
        logger.info("Downloading '%s' from Kaggle...", file_name)
        if is_competition:
            cmd = ["kaggle", "competitions", "download", "-c", competition_or_dataset,
                   "-f", file_name, "-p", download_dir, "--unzip"]
        else:
            cmd = ["kaggle", "datasets", "download", "-d", competition_or_dataset,
                   "-f", file_name, "-p", download_dir, "--unzip"]
        subprocess.run(cmd, check=True)

    df = pd.read_csv(file_path)
    y = df.pop(target_col).rename("target")
    X = df

    meta = {
        "source": f"kaggle_{competition_or_dataset}",
        "file": file_name,
        "n_samples": len(X),
        "n_features": X.shape[1],
    }

    logger.info("Loaded Kaggle '%s/%s': %d rows", competition_or_dataset, file_name, len(X))
    return DataBundle(X=X, y=y, meta=meta)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    source: Literal["synthetic_clf", "synthetic_reg", "uci", "kaggle"] = "synthetic_clf",
    **kwargs,
) -> DataBundle:
    """
    Entry point duy nhất — dùng trong notebooks và experiments.

    Examples
    --------
    >>> bundle = load_dataset("synthetic_clf", n_samples=3000)
    >>> bundle = load_dataset("uci", name="adult")
    >>> bundle = load_dataset("kaggle", competition_or_dataset="titanic",
    ...                       file_name="train.csv", target_col="Survived")
    """
    loaders = {
        "synthetic_clf": load_synthetic_classification,
        "synthetic_reg": load_synthetic_regression,
        "uci":           load_uci,
        "kaggle":        load_kaggle,
    }
    if source not in loaders:
        raise ValueError(f"source phải là một trong: {list(loaders)}")
    return loaders[source](**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    bundle = load_dataset("synthetic_clf", n_samples=1_000, class_imbalance=0.2)
    print(bundle)
    print(bundle.X.head(3))
    print("Class distribution:", bundle.meta["class_distribution"])