"""Phase 3 — Baseline: TF-IDF + Logistic Regression (PyThaiNLP word tokenizer).

run:  python -m src.baseline
"""
from __future__ import annotations

import logging

import joblib
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

import config
from src.data import get_label_info, load_splits
from src.utils import (
    classification_report_dict,
    compute_metric_dict,
    save_confusion_matrix,
    set_seed,
    update_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def thai_tokenizer(text: str):
    """Tokenize Thai text using the `newmm` engine.

    Note: Must be a top-level function (not lambda) so joblib can pickle 
    and reload it properly in evaluate.py or the demo.
    """
    return word_tokenize(text, engine="newmm")


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=thai_tokenizer,
                    token_pattern=None,   # Using a custom tokenizer, disable default pattern
                    ngram_range=(1, 2),   # Unigrams and bigrams
                    min_df=3,             # Ignore terms that appear in less than 3 documents
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",  # Compensate for class imbalance
                    random_state=config.SEED,
                ),
            ),
        ]
    )


def main() -> dict:
    set_seed(config.SEED)
    ds = load_splits()
    info = get_label_info(ds)
    names = info["names"]

    X_train, y_train = ds["train"][config.TEXT_COL], ds["train"][config.LABEL_COL]
    X_test, y_test = ds["test"][config.TEXT_COL], ds["test"][config.LABEL_COL]

    pipe = build_pipeline()
    logger.info("Fitting TF-IDF + LogisticRegression on %d train docs ...", len(X_train))
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics = compute_metric_dict(y_test, preds)
    report = classification_report_dict(y_test, preds, names)
    logger.info("Baseline test metrics: %s", metrics)

    # Save artifacts
    joblib.dump(pipe, config.BASELINE_PATH)
    logger.info("Saved baseline model -> %s", config.BASELINE_PATH)

    cm_path = save_confusion_matrix(
        y_test,
        preds,
        names,
        title="Confusion matrix — TF-IDF + LogReg (test)",
        out_path=config.FIGURES_DIR / "confusion_matrix_baseline.png",
    )
    logger.info("Saved confusion matrix -> %s", cm_path)

    update_metrics(
        "baseline",
        {
            "model": "tfidf_ngram12_logreg_balanced",
            "test": metrics,
            "classification_report": report,
        },
    )

    print("\n=== Baseline (TF-IDF + LogReg) — test classification report ===")
    print(classification_report(y_test, preds, target_names=names, zero_division=0))
    return metrics


if __name__ == "__main__":
    main()
