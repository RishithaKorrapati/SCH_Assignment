from pathlib import Path

import pandas as pd

from src.pipeline import run_day
from src.utils.config import ROOT


def _write_bronze(config, day, df):
    path = Path(config["paths"]["bronze_dir"]) / f"{day}.csv"
    df.to_csv(path, index=False)
    return path

def test_corrupt_day_quarantined_skips_training_clean_day_trains(
    isolated_config, config
):
    source = pd.read_csv(
        ROOT / "data" / "source" / "telco_customer_churn.csv",
        dtype={"TotalCharges": str},
    )
    n_good = config["training"]["min_gold_rows"] + 50
    good = source.head(n_good).copy()
    bad = source.iloc[n_good : n_good + 50].copy()
    bad.iloc[0, bad.columns.get_loc("Churn")] = "maybe"

    _write_bronze(isolated_config, "2026-03-01", good)
    _write_bronze(isolated_config, "2026-03-02", bad)

    good_row, counts = run_day("2026-03-01", isolated_config, previous_row_counts=[])
    bad_row, _ = run_day("2026-03-02", isolated_config, previous_row_counts=counts)

    assert good_row["input"].startswith("PASS")
    assert good_row["output"].startswith("PASS")
    assert good_row["training"].startswith("TRAINED")

    assert bad_row["input"] == "FAIL"
    assert bad_row["output"] == "SKIP"
    assert bad_row["training"].startswith("SKIPPED")
    assert "input gate failed" in bad_row["training"]

    quarantine = Path(isolated_config["paths"]["quarantine_dir"]) / "2026-03-02.csv"
    assert quarantine.exists()
    assert not (Path(isolated_config["paths"]["silver_dir"]) / "2026-03-02.csv").exists()
    assert not (Path(isolated_config["paths"]["models_dir"]) / "2026-03-02.joblib").exists()

    assert (Path(isolated_config["paths"]["silver_dir"]) / "2026-03-01.csv").exists()
    assert (Path(isolated_config["paths"]["gold_dir"]) / "2026-03-01.csv").exists()
    assert (Path(isolated_config["paths"]["models_dir"]) / "2026-03-01.joblib").exists()
