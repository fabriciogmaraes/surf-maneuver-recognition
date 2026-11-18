"""Load cached encoder features aligned with data/splits.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.inventory import PROJECT_ROOT

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"


def load_features(splits: pd.DataFrame, name: str = "vjepa2_vitl_t8") -> np.ndarray:
    """Features of every clip in `splits`, in the same row order: (n_clips, time, dim)."""
    saved = np.load(FEATURES_DIR / f"{name}.npz")
    by_clip = dict(zip(saved["clips"], saved["features"]))
    missing = [c for c in splits["clip"] if c not in by_clip]
    if missing:
        raise KeyError(f"{len(missing)} clips have no cached features, e.g. {missing[0]}")
    return np.stack([by_clip[c] for c in splits["clip"]])