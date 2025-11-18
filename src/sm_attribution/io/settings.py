from __future__ import annotations
import yaml
from pathlib import Path


class Settings:
    def __init__(self, path: str | None = None):
        """
        Load global settings from settings.yml.

        If `path` is None, resolve configs/settings.yml relative to the
        project root (three parents up from this file, same logic as
        io.registry.default_registry).
        """
        if path is None:
            # src/sm_attribution/io/settings.py -> project root
            root = Path(__file__).resolve().parents[3]
            path = root / "configs" / "settings.yml"
        else:
            path = Path(path)
            # If a relative path is given, treat it as relative to project root
            if not path.is_absolute():
                root = Path(__file__).resolve().parents[3]
                path = root / path

        if not path.exists():
            raise FileNotFoundError(f"Settings file not found: {path}")

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        self.depth_target_m = float(cfg.get("depth_target_m", 1.0))
        self.time_resolution = cfg.get("time_resolution", "monthly")
        self.target_calendar = cfg.get("target_calendar", "proleptic_gregorian")
        self.ssi = cfg.get("ssi", {})
        self.grid = cfg.get("grid", None)


# simple singleton-style accessor (import where needed)
_settings: Settings | None = None


def get_settings(path: str | None = None) -> Settings:
    """
    Return a singleton Settings instance.

    Normally called as get_settings() with no arguments.
    If called with a custom `path` the first time, that path is used
    to initialize the singleton.
    """
    global _settings
    if _settings is None:
        _settings = Settings(path)
    return _settings