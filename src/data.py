"""Phase 1-2 — Data acquisition, light cleaning, and reproducible splitting."""
from __future__ import annotations

import logging

from datasets import ClassLabel, DatasetDict, load_dataset

import config
from src.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Optional text normalization
try:
    from pythainlp.util import normalize as _pythai_normalize
except Exception:  # pragma: no cover
    _pythai_normalize = None

_VALIDATION_ALIASES = ("validation", "valid", "dev")


# --------------------------------------------------------------------------- #
# Cleaning (Light cleaning - WangchanBERTa handles raw Thai well)
# --------------------------------------------------------------------------- #
def clean_text(text: str, normalize: bool = True) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())  # Collapse multiple whitespaces
    if normalize and _pythai_normalize is not None:
        text = _pythai_normalize(text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #
def _validate_schema(ds: DatasetDict) -> None:
    split = next(iter(ds.values()))
    cols = set(split.column_names)
    missing = {config.TEXT_COL, config.LABEL_COL} - cols
    if missing:
        raise KeyError(
            f"Expected columns missing: {missing}. Found: {sorted(cols)}. "
            "Check column names in config.py."
        )
    if not isinstance(split.features[config.LABEL_COL], ClassLabel):
        raise TypeError(
            f"Column '{config.LABEL_COL}' is not a ClassLabel — cannot stratify or map labels."
        )


def _normalize_split_names(ds: DatasetDict) -> DatasetDict:
    """Rename validation split to 'validation' if it exists under an alias."""
    for alias in _VALIDATION_ALIASES:
        if alias in ds and alias != "validation":
            ds["validation"] = ds.pop(alias)
            logger.info("Renamed split '%s' -> 'validation'", alias)
            break
    return ds


def _ensure_validation_split(ds: DatasetDict, seed: int) -> DatasetDict:
    """If there is no validation split, create a stratified split from the train set."""
    ds = _normalize_split_names(ds)
    if "validation" in ds:
        return ds
    if "train" not in ds:
        raise KeyError("No 'train' split available to create validation split.")
    logger.info(
        "No validation split found -> Creating from train (stratified, val_size=%.2f, seed=%d)",
        config.VAL_SIZE,
        seed,
    )
    split = ds["train"].train_test_split(
        test_size=config.VAL_SIZE,
        stratify_by_column=config.LABEL_COL,  # Preserve class distribution
        seed=seed,
    )
    ds["train"] = split["train"]
    ds["validation"] = split["test"]
    return ds


def _save_splits(ds: DatasetDict) -> None:
    for name, split in ds.items():
        out = config.DATA_DIR / f"{name}.parquet"
        split.to_parquet(str(out))
        logger.info("Saved %s split (%d rows) -> %s", name, split.num_rows, out)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_splits(normalize: bool = True, save: bool = True, seed: int = config.SEED) -> DatasetDict:
    """Load, clean, and optionally create a validation split. Returns a deterministic DatasetDict.

    Guarantees the presence of train / validation / test splits.
    """
    set_seed(seed)
    ds = load_dataset(config.DATASET_NAME)
    _validate_schema(ds)

    ds = ds.map(
        lambda ex: {config.TEXT_COL: clean_text(ex[config.TEXT_COL], normalize)},
        desc="cleaning text",
    )
    ds = ds.filter(lambda ex: len(ex[config.TEXT_COL]) > 0, desc="drop empties")

    ds = _ensure_validation_split(ds, seed=seed)

    if "test" not in ds:
        raise KeyError(
            f"No test split found in dataset (found: {list(ds.keys())}). "
            "The plan assumes the dataset already has a test split."
        )

    if save:
        _save_splits(ds)
    return ds


def get_label_info(ds: DatasetDict) -> dict:
    """Extract label mapping dynamically from the ClassLabel feature."""
    feat: ClassLabel = ds["train"].features[config.LABEL_COL]
    names = list(feat.names)
    return {
        "names": names,
        "num_classes": feat.num_classes,
        "id2label": {i: n for i, n in enumerate(names)},
        "label2id": {n: i for i, n in enumerate(names)},
    }


if __name__ == "__main__":
    dsd = load_splits()
    info = get_label_info(dsd)
    print("\n=== Splits ===")
    for name, split in dsd.items():
        print(f"  {name:12s}: {split.num_rows:,} rows")
    print("\n=== Labels (ordered by index) ===")
    print(" ", info["id2label"])
    print("\n=== Class distribution (train) ===")
    import collections

    counts = collections.Counter(dsd["train"][config.LABEL_COL])
    for idx, name in info["id2label"].items():
        print(f"  {name:10s}: {counts.get(idx, 0):,}")
