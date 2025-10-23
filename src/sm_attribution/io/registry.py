from __future__ import annotations
import os
import yaml
from typing import Dict

class Registry:
    """
    Thin wrapper around configs/data_registry.yml.
    Refuses to guess paths. Raises FileNotFoundError or KeyError loudly.
    """
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        with open(yaml_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.paths = self.cfg.get("paths", {})
        if not self.paths:
            raise KeyError("Missing 'paths' in data_registry.yml")

        self.models_raw = self.cfg.get("models", {})
        self.processed = (self.cfg.get("processed", {}) or {}).get("models_1m", {})

    def get_model_raw(self, model: str, scenario: str) -> str:
        try:
            path = self.models_raw[model][scenario]
        except KeyError as e:
            raise KeyError(f"Missing registry key for model={model}, scenario={scenario}") from e
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw file not found: {path}")
        return path

    def get_model_processed(self, model: str, scenario: str) -> str:
        try:
            path = self.processed[model][scenario]
        except KeyError as e:
            raise KeyError(f"Missing processed path in registry for model={model}, scenario={scenario}") from e
        # ensure parent exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def list_models(self) -> Dict[str, Dict[str, str]]:
        return self.models_raw

    def scenarios(self) -> tuple[str, ...]:
        # canonical set used throughout code
        return (
            "obsclim_histsoc",
            "counterclim_histsoc",
            "obsclim_1901soc",
            "counterclim_1901soc",
        )