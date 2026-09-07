from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    config_path = Path(path) if path else ROOT / "config.yaml"
    with config_path.open() as f:
        return yaml.safe_load(f)
