"""Central configuration (paths, dataset/model names, hyperparameters, seed).

All modules import from this single source of truth to ensure the baseline and transformer models 
use consistent values, making the results reproducible.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Create necessary directories (idempotent)
for _d in (DATA_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
DATASET_NAME = "pythainlp/wisesight_sentiment"
TEXT_COL = "texts"
LABEL_COL = "category"

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
MAX_LEN = 256
OUTPUT_DIR = MODELS_DIR / "thai-sentiment"
BEST_MODEL_DIR = OUTPUT_DIR / "best"
BASELINE_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"

# --------------------------------------------------------------------------- #
# Hyperparameters (fix seed everywhere for reproducibility)
# --------------------------------------------------------------------------- #
SEED = 42
LEARNING_RATE = 2e-5    # Optimal learning rate for fine-tuning BERT typically ranges from 2e-5 to 5e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2

# --------------------------------------------------------------------------- #
# Split (used only if dataset lacks a validation split)
# --------------------------------------------------------------------------- #
VAL_SIZE = 0.1   # Proportion of train data to split for validation (stratified)
