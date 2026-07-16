"""Phase 4 — Fine-tune WangchanBERTa (4-class Thai sentiment).

Model: airesearch/wangchanberta-base-att-spm-uncased
       (RoBERTa-based, SentencePiece, vocab 25k, uncased)

run (requires GPU; Google Colab T4 recommended):  python -m src.train
"""
from __future__ import annotations

import inspect
import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import config
from src.data import get_label_info, load_splits
from src.utils import set_seed, update_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(config.MODEL_NAME)
    except Exception as exc:  # pragma: no cover
        logger.warning("AutoTokenizer failed (%s) -> fallback to CamembertTokenizer", exc)
        from transformers import CamembertTokenizer

        tok = CamembertTokenizer.from_pretrained(config.MODEL_NAME)
        tok.additional_special_tokens = ["<s>NOTUSED", "</s>NOTUSED", "<_>"]
        return tok


# --------------------------------------------------------------------------- #
# Metrics — macro-F1 is the main metric (imbalanced data)
# --------------------------------------------------------------------------- #
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# --------------------------------------------------------------------------- #
# Weighted-loss Trainer
# --------------------------------------------------------------------------- #
class WeightedTrainer(Trainer):
    """Trainer using weighted CrossEntropy.
    
    The `question` class is a minority. Without class weighting, the model might 
    ignore this class, causing a drop in macro-F1. Weighted loss forces the model 
    to pay attention to minority classes.
    """

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    # **kwargs supports newer transformers signature (e.g., num_items_in_batch)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# --------------------------------------------------------------------------- #
# TrainingArguments
# --------------------------------------------------------------------------- #
def build_training_args() -> TrainingArguments:
    use_fp16 = torch.cuda.is_available()
    if not use_fp16:
        logger.warning("GPU not found -> fp16=False. Training will be slow (recommend Colab T4)")

    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"

    kwargs = dict(
        output_dir=str(config.OUTPUT_DIR),
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=50,
        report_to="none",
        seed=config.SEED,
    )
    kwargs[eval_key] = "epoch"  # Must match save_strategy for load_best_model_at_end
    return TrainingArguments(**kwargs)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    set_seed(config.SEED)
    ds = load_splits()
    info = get_label_info(ds)
    num_labels = info["num_classes"]

    tokenizer = load_tokenizer()

    def tokenize_fn(batch):
        return tokenizer(batch[config.TEXT_COL], truncation=True, max_length=config.MAX_LEN)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=[config.TEXT_COL])
    # Trainer/loss expects column named "labels"
    tokenized = tokenized.rename_column(config.LABEL_COL, "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # dynamic padding

    # Calculate class weights from training labels
    train_labels = np.array(ds["train"][config.LABEL_COL])
    weights = compute_class_weight("balanced", classes=np.arange(num_labels), y=train_labels)
    class_weights = torch.tensor(weights, dtype=torch.float)
    logger.info("Class weights: %s", dict(zip(info["names"], weights.round(3))))

    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=num_labels,
        id2label=info["id2label"],   # Useful for inference returning real label names
        label2id=info["label2id"],
    )

    args = build_training_args()

    # forward-compat: transformers >=4.46 uses processing_class instead of tokenizer
    trainer_params = inspect.signature(Trainer.__init__).parameters
    tok_kw = "processing_class" if "processing_class" in trainer_params else "tokenizer"

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
    )
    trainer_kwargs[tok_kw] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    logger.info("Starting training ...")
    trainer.train()

    trainer.save_model(str(config.BEST_MODEL_DIR))
    tokenizer.save_pretrained(str(config.BEST_MODEL_DIR))
    logger.info("Saved best model -> %s", config.BEST_MODEL_DIR)

    # Evaluate on validation (test evaluation is done in src/evaluate.py)
    val_metrics = trainer.evaluate()
    clean = {k.replace("eval_", ""): float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))}
    update_metrics("transformer_val", clean)
    logger.info("Validation metrics: %s", clean)


if __name__ == "__main__":
    main()
