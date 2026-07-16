"""Shared helpers: seeding, metrics IO, and evaluation/plotting utilities.

Separated to ensure baseline and transformer use the exact same evaluation functions,
making comparisons fair (apples-to-apples).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np

import config


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = config.SEED) -> None:
    """Fix every RNG we touch (Python, NumPy, Torch, HF)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # Torch might not be installed when running baseline only
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# metrics.json IO (machine-readable, updates per key)
# --------------------------------------------------------------------------- #
def read_metrics() -> dict:
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def update_metrics(key: str, value: dict) -> None:
    """Read existing metrics.json and overwrite only the specified key."""
    data = read_metrics()
    data[key] = value
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Metrics & plots (macro-F1 is the main metric due to imbalanced data)
# --------------------------------------------------------------------------- #
def compute_metric_dict(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def classification_report_dict(
    y_true: Sequence[int], y_pred: Sequence[int], label_names: Sequence[str]
) -> dict:
    from sklearn.metrics import classification_report

    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=list(label_names),
        output_dict=True,
        zero_division=0,
    )


def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
    title: str,
    out_path: Path,
) -> Path:
    """Plot and save confusion matrix (row-normalized = recall per class)."""
    import matplotlib

    matplotlib.use("Agg")  # headless (Colab / CI)
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    n = len(label_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=cm,            # Show raw numbers
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={"label": "row-normalized (recall)"},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
