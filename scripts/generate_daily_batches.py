"""Simulate daily Bronze files. Assumption:

The IBM Telco CSV has no date column, so we cannot replay a real calendar.
We shuffle the full dump with a fixed seed, split it into ~10 slices, and
write each slice as data/bronze/YYYY-MM-DD.csv.

Exactly one day is deliberately broken (duplicate IDs, a null required
field, an out-of-range bill, a bad Churn label, and too few rows) so the
input gate has a real failure to catch.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  


SOURCE_CSV = ROOT / "data" / "source" / "telco_customer_churn.csv"


def _split_evenly(df, n_days):
    """Slice a shuffled frame into n_days contiguous batches (sizes differ by at most 1)."""
    n = len(df)
    base, remainder = divmod(n, n_days)
    batches = []
    start = 0
    for i in range(n_days):
        size = base + (1 if i < remainder else 0)
        batches.append(df.iloc[start : start + size].copy())
        start += size
    return batches


def _corrupt(batch, cfg):
    """Inject every hard-fail condition the input gate is supposed to catch."""
    min_rows = cfg["input_gate"]["min_row_count"]
    max_charge = cfg["input_gate"]["monthly_charges_max"]

    bad = batch.iloc[: min_rows // 2].copy()  # shrink below the hard floor
    # Duplicate customerID: two customers, one identity.
    bad.iloc[1, bad.columns.get_loc("customerID")] = bad.iloc[0]["customerID"]
    # Null a required field (gender is in required_non_null_columns).
    bad.iloc[2, bad.columns.get_loc("gender")] = None
    # Out-of-range bill: just past the real-world ceiling in config.
    bad.iloc[3, bad.columns.get_loc("MonthlyCharges")] = max_charge + 1
    # Garbage label: not in {Yes, No}.
    bad.iloc[4, bad.columns.get_loc("Churn")] = "maybe"
    return bad


def main():
    cfg = load_config()
    sim = cfg["simulation"]
    bronze_dir = ROOT / cfg["paths"]["bronze_dir"]
    bronze_dir.mkdir(parents=True, exist_ok=True)

    for stale in bronze_dir.glob("*.csv"):
        stale.unlink()

    # Keep TotalCharges as text so the 11 blanks survive the split.
    df = pd.read_csv(SOURCE_CSV, dtype={"TotalCharges": str})
    df = df.sample(frac=1, random_state=sim["shuffle_seed"]).reset_index(drop=True)

    batches = _split_evenly(df, sim["n_days"])
    start = datetime.strptime(sim["start_date"], "%Y-%m-%d").date()
    corrupt_idx = sim["corrupt_day_index"]

    print(f"{'date':<12} {'rows':>6}  status")
    for i, batch in enumerate(batches):
        day = start + timedelta(days=i)
        if i == corrupt_idx:
            batch = _corrupt(batch, cfg)
            status = "CORRUPTED"
        else:
            status = "ok"
        out = bronze_dir / f"{day.isoformat()}.csv"
        batch.to_csv(out, index=False)
        print(f"{day.isoformat():<12} {len(batch):>6}  {status}")

    corrupt_day = start + timedelta(days=corrupt_idx)
    print(
        f"\nWrote {sim['n_days']} files to {bronze_dir}. "
        f"Corrupted {corrupt_day.isoformat()} "
        f"(duplicates, null gender, MonthlyCharges>{cfg['input_gate']['monthly_charges_max']}, "
        f"Churn='maybe', rows<{cfg['input_gate']['min_row_count']})."
    )


if __name__ == "__main__":
    main()
