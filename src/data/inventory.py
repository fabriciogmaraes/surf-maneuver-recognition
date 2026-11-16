"""Build a table with one row per clip of the Surf Maneuver Recognition dataset."""
import re
from pathlib import Path

import pandas as pd

# src/data/inventory.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "data-split"


def name_pattern(name: str) -> str:
    """Group file names by naming convention (a proxy for the clip's source)."""
    if "imbriqu" in name:
        return "sequence_underscore" if "_" in name else "sequence_space"
    return "other"


def clip_number(name: str) -> int | None:
    """Trailing clip number after a space or underscore; None if absent (e.g. "360.mp4")."""
    match = re.search(r"[ _](\d+)\.mp4$", name)
    return int(match.group(1)) if match else None


def build_clip_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per clip: file, split, maneuver, path, naming pattern, source and clip number.

    The label comes from the folder name: data-split/<split>/<maneuver>/<clip>.mp4
    """
    rows = []
    for video in sorted(data_dir.glob("*/*/*.mp4")):
        rows.append({
            "file": video.name,
            "split": video.parts[-3],
            "maneuver": video.parts[-2],
            "path": video,
        })

    df = pd.DataFrame(rows)
    df["name_pattern"] = df["file"].map(name_pattern)
    # Source: the naming pattern, or the file prefix for "other" files ("Roller_29.mp4" -> "roller")
    df["source"] = df["name_pattern"].where(
        df["name_pattern"] != "other",
        df["file"].str.split(r"[_.]", regex=True).str[0].str.lower(),
    )
    df["clip_number"] = df["file"].map(clip_number)
    return df