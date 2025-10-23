from __future__ import annotations
import yaml

class Settings:
    def __init__(self, path: str = "configs/settings.yml"):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        self.depth_target_m = float(cfg.get("depth_target_m", 1.0))
        self.time_resolution = cfg.get("time_resolution", "monthly")
        self.target_calendar = cfg.get("target_calendar", "proleptic_gregorian")
        self.ssi = cfg.get("ssi", {})
        self.grid = cfg.get("grid", None)

# simple singleton-style accessor (import where needed)
_settings = None
def get_settings(path: str = "configs/settings.yml") -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings(path)
    return _settings