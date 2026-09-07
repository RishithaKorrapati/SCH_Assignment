"""Read Bronze; write Silver and Gold. Call the writers only after both gates pass."""

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.utils.config import ROOT


def _as_day(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _zone(config, key):
    return ROOT / config["paths"][key]


def load_bronze(day, config):
    """Read one raw daily drop. TotalCharges stays text so blanks survive."""
    path = _zone(config, "bronze_dir") / f"{_as_day(day)}.csv"
    return pd.read_csv(path, dtype={"TotalCharges": str})


def store_silver(cleaned_df, day, config):
    """Write one cleaned day. This is the trusted daily fact table."""
    silver_dir = _zone(config, "silver_dir")
    silver_dir.mkdir(parents=True, exist_ok=True)
    path = silver_dir / f"{_as_day(day)}.csv"
    cleaned_df.to_csv(path, index=False)
    return path


def materialize_gold(day, config):
    """Concatenate every Silver file with date <= today and write Gold to disk.

    Gold is a snapshot of the training corpus *as of this day*, not a
    dataframe that disappears when the process exits. Phase 6 trains on
    this file.
    """
    day_str = _as_day(day)
    silver_dir = _zone(config, "silver_dir")
    gold_dir = _zone(config, "gold_dir")
    gold_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in sorted(silver_dir.glob("*.csv")):
        if path.stem <= day_str:
            frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError(
            f"No Silver files on or before {day_str} in {silver_dir}"
        )

    gold = pd.concat(frames, ignore_index=True)
    out = gold_dir / f"{day_str}.csv"
    gold.to_csv(out, index=False)
    return gold, out
