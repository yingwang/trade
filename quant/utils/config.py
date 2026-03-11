"""Configuration loader."""

import yaml
from pathlib import Path


def load_config(path: str = None) -> dict:
    """Load configuration from YAML file."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config.yaml"
    else:
        path = Path(path)

    with open(path) as f:
        return yaml.safe_load(f)
