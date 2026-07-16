"""Phase 6 — Error analysis.

Review examples where the transformer made incorrect predictions, 
categorize error types, and summarize potential improvements.
Outputs: reports/error_analysis.md + reports/misclassified.csv

run:  python -m src.error_analysis   (requires a trained model from src.train)
"""
from __future__ import annotations

import collections
import logging
import re

import numpy as np
import pandas as pd

import config
from src.data import get_label_info, load_splits
from src.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

N_EXAMPLES = 20

_LATIN = re.compile(r"[A-Za-z]")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF\U00002190-\U000021FF]"
)


def predict_with_proba(texts, model_dir: str, batch_size: int = 64):
    """Return (preds, confidence) where confidence is max softmax probability."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            enc = tok(
                batch, truncation=True, max_length=config.MAX_LEN, padding=True, return_tensors="pt"
            ).to(device)
            probs = model(**enc).logits.softmax(dim=-1)
            conf, idx = probs.max(dim=-1)
            preds.extend(idx.cpu().numpy().tolist())
            confs.extend(conf.cpu().numpy().tolist())
    return np.array(preds), np.array(confs)


def _md_safe(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split()).replace("|", "/")
    return text[:limit] + ("…" if len(text) > limit else "")


def main():
    set_seed(config.SEED)
    if not (config.BEST_MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Model not found at {config.BEST_MODEL_DIR} — run `python -m src.train` first"
        )

    ds = load_splits()
    info = get_label_info(ds)
    id2label = info["id2label"]

    texts = list(ds["test"][config.TEXT_COL])
    y_true = np.array(ds["test"][config.LABEL_COL])
    y_pred, conf = predict_with_proba(texts, str(config.BEST_MODEL_DIR))

    df = pd.DataFrame(
        {
            "text": texts,
            "true": [id2label[i] for i in y_true],
            "pred": [id2label[i] for i in y_pred],
            "confidence": conf.round(3),
        }
    )
    df["correct"] = y_true == y_pred
    errors = df[~df["correct"]].copy()
    err_rate = len(errors) / len(df) * 100
    logger.info("Errors: %d / %d (%.1f%%)", len(errors), len(df), err_rate)

    errors.to_csv(config.REPORTS_DIR / "misclassified.csv", index=False, encoding="utf-8-sig")

    # ---- Group confusion pairs (true -> pred) ----
    pairs = (
        errors.groupby(["true", "pred"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )

    # ---- Heuristic observations ----
    errors["has_latin"] = errors["text"].apply(lambda t: bool(_LATIN.search(str(t))))
    errors["has_emoji"] = errors["text"].apply(lambda t: bool(_EMOJI.search(str(t))))

    # Minority class
    train_counts = collections.Counter(ds["train"][config.LABEL_COL])
    minority_name = id2label[min(train_counts, key=train_counts.get)]
    n_minority_missed = int((errors["true"] == minority_name).sum())

    # Top confusion pair
    top_pair = pairs.iloc[0] if len(pairs) else None

    # ---- Select top "confident but wrong" examples ----
    sample = errors.sort_values("confidence", ascending=False).head(N_EXAMPLES)

    # ---- Write report ----
    lines = []
    lines.append("# Error Analysis — WangchanBERTa (test set)\n")
    top_pair_txt = (
        f"{top_pair['true']} → {top_pair['pred']} ({top_pair['count']} times)" if top_pair is not None else "—"
    )
    lines.append(
        f"- Overall error rate: **{err_rate:.1f}%** ({len(errors)}/{len(df)})\n"
        f"- Errors with English text (code-mixing): **{int(errors['has_latin'].sum())}**\n"
        f"- Errors with emoji: **{int(errors['has_emoji'].sum())}**\n"
        f"- Most confused pair (true → pred): **{top_pair_txt}**\n"
        f"- Minority class `{minority_name}` misclassified: **{n_minority_missed}**\n"
    )

    lines.append("\n## Top confusion pairs (true → pred)\n")
    lines.append("| True | Pred | Count |")
    lines.append("|------|------|-------|")
    for _, r in pairs.head(8).iterrows():
        lines.append(f"| {r['true']} | {r['pred']} | {r['count']} |")

    lines.append(f"\n## Top 'Confident but Wrong' examples (Top {N_EXAMPLES} by confidence)\n")
    lines.append("| Text | True | Pred | Conf |")
    lines.append("|------|------|------|------|")
    for _, r in sample.iterrows():
        lines.append(f"| {_md_safe(r['text'])} | {r['true']} | {r['pred']} | {r['confidence']:.2f} |")

    lines.append(
        "\n## Summary of Patterns Found\n"
        f"1. **Most confused pair is {top_pair_txt}** — Boundary between these 2 classes is thin\n"
        f"2. **Minority class `{minority_name}` is misclassified**: Insufficient training data, model hasn't seen enough examples\n"
        "3. **code-mixing / emoji / sarcasm**: Context where Thai subwords fail to fully capture sentiment\n"
        "\n## Future Improvements\n"
        f"- **Data augmentation** for minority class `{minority_name}` (back-translation / templating) to solve data scarcity\n"
        "- Group emojis as specific tokens / add emoji sentiment features\n"
        "- Try larger models (e.g., larger wangchanberta) or ensemble with baseline\n"
        "- Collect more sarcasm data or perform targeted hard-example mining\n"
    )

    out = config.REPORTS_DIR / "error_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved -> %s and %s", out, config.REPORTS_DIR / "misclassified.csv")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
