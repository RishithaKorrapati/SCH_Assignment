from pathlib import Path
import copy
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


def failure_codes(result):
    return [item.split(":")[0] for item in result.hard_failures]


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def valid_batch(config):
    """A real, trusted slice — large enough to clear the row-count floor."""
    path = ROOT / "data" / "source" / "telco_customer_churn.csv"
    df = pd.read_csv(path, dtype={"TotalCharges": str})
    n = config["input_gate"]["min_row_count"] + 20
    return df.head(n).copy()


@pytest.fixture
def isolated_config(tmp_path, config):
    """Point every zone at a temp dir so tests do not touch the real run."""
    cfg = copy.deepcopy(config)
    cfg["paths"] = {
        "bronze_dir": str(tmp_path / "bronze"),
        "quarantine_dir": str(tmp_path / "bronze" / "quarantine"),
        "silver_dir": str(tmp_path / "silver"),
        "gold_dir": str(tmp_path / "gold"),
        "reports_dir": str(tmp_path / "reports"),
        "models_dir": str(tmp_path / "models"),
    }
    for path in cfg["paths"].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return cfg
