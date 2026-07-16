"""Phase 5 — Evaluation & head-to-head comparison (on the same test set).

Outputs: per-class classification report, confusion matrix for each model,
comparison table (accuracy / macro-F1 / weighted-F1), and comparison charts — saved to reports/

run:  python -m src.evaluate
(requires running `python -m src.baseline` and `python -m src.train` first)
"""
from __future__ import annotations

import logging

import numpy as np

import config
from src.data import get_label_info, load_splits
from src.utils import (
    classification_report_dict,
    compute_metric_dict,
    read_metrics,
    save_confusion_matrix,
    set_seed,
    update_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def load_baseline():
    if not config.BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline not found at {config.BASELINE_PATH} — run `python -m src.baseline` first"
        )
    import joblib

    from src.baseline import thai_tokenizer  # noqa: F401  -- allows thai_tokenizer to be imported for unpickling
    import sys
    sys.modules["__main__"].thai_tokenizer = thai_tokenizer # Fix __main__ reference

    return joblib.load(config.BASELINE_PATH)


def predict_transformer(texts, model_dir: str, batch_size: int = 64) -> np.ndarray:
    """Run batch inference. Always specify max_length."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            enc = tok(
                batch,
                truncation=True,
                max_length=config.MAX_LEN,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return np.array(preds)


# --------------------------------------------------------------------------- #
# Per-model evaluation
# --------------------------------------------------------------------------- #
def evaluate_one(display_name: str, y_true, y_pred, names, key: str) -> dict:
    from sklearn.metrics import classification_report

    metrics = compute_metric_dict(y_true, y_pred)
    report = classification_report_dict(y_true, y_pred, names)
    save_confusion_matrix(
        y_true,
        y_pred,
        names,
        title=f"Confusion matrix — {display_name} (test)",
        out_path=config.FIGURES_DIR / f"confusion_matrix_{key}.png",
    )
    update_metrics(key, {"display_name": display_name, "test": metrics, "classification_report": report})

    print(f"\n=== {display_name} — test classification report ===")
    print(classification_report(y_true, y_pred, target_names=names, zero_division=0))
    return {"display_name": display_name, **metrics}


# --------------------------------------------------------------------------- #
# Comparison artifacts
# --------------------------------------------------------------------------- #
def build_comparison_md(results: dict) -> str:
    lines = [
        "| Model | Accuracy | Macro-F1 | Weighted-F1 |",
        "|-------|----------|----------|-------------|",
    ]
    for key in ("baseline", "transformer"):
        if key in results:
            r = results[key]
            lines.append(
                f"| {r['display_name']} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} | {r['f1_weighted']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def plot_comparison(results: dict, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_keys = ["accuracy", "f1_macro", "f1_weighted"]
    models = [k for k in ("baseline", "transformer") if k in results]
    if len(models) < 2:
        return  # Require 2 models for comparison

    x = np.arange(len(metric_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, key in enumerate(models):
        vals = [results[key][m] for m in metric_keys]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=results[key]["display_name"])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro-F1", "Weighted-F1"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Baseline vs WangchanBERTa (test set)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    set_seed(config.SEED)
    ds = load_splits()
    info = get_label_info(ds)
    names = info["names"]

    y_test = np.array(ds["test"][config.LABEL_COL])
    X_test = ds["test"][config.TEXT_COL]

    results: dict[str, dict] = {}

    # ---- baseline ----
    pipe = load_baseline()
    base_pred = pipe.predict(X_test)
    results["baseline"] = evaluate_one("TF-IDF + LogReg (baseline)", y_test, base_pred, names, "baseline")

    # ---- transformer ----
    if (config.BEST_MODEL_DIR / "config.json").exists():
        tr_pred = predict_transformer(X_test, str(config.BEST_MODEL_DIR))
        results["transformer"] = evaluate_one(
            "WangchanBERTa (fine-tuned)", y_test, tr_pred, names, "transformer"
        )
    else:
        logger.warning(
            "Transformer model not found at %s — run `python -m src.train` first to complete comparison",
            config.BEST_MODEL_DIR,
        )

    # ---- comparison ----
    table = build_comparison_md(results)
    print("\n=== Comparison ===\n" + table)
    (config.REPORTS_DIR / "comparison.md").write_text(table, encoding="utf-8")
    plot_comparison(results, config.FIGURES_DIR / "model_comparison.png")
    update_metrics("comparison", {k: {m: results[k][m] for m in ("accuracy", "f1_macro", "f1_weighted")} for k in results})

    logger.info("Saved all results to %s", config.METRICS_PATH)
    return read_metrics()


if __name__ == "__main__":
    main()
