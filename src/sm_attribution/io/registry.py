#src/sm_attribution/io/registry.py

from __future__ import annotations
import os
import yaml
from typing import Dict
from pathlib import Path


def project_root() -> Path:
    """
    Return the absolute path to the sm_attribution project root.

    Resolution order
    ----------------
    1. ``SM_ATTR_ROOT`` environment variable (if set and the directory exists).
    2. Auto-detect from this file's location:
       ``src/sm_attribution/io/registry.py`` → parents[3] = project root.
    """
    env = os.environ.get("SM_ATTR_ROOT")
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(
            f"SM_ATTR_ROOT={env!r} does not exist or is not a directory"
        )
    return Path(__file__).resolve().parents[3]


class Registry:
    """
    Thin wrapper around configs/data_registry.yml.
    Refuses to guess paths. Raises FileNotFoundError or KeyError loudly.

    All relative paths in the YAML file are resolved to absolute paths
    at load time using the project root directory.
    """
    def __init__(self, yaml_path: str, root: Path | None = None):
        self.yaml_path = yaml_path
        self._project_root = root or project_root()

        with open(yaml_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.paths = self.cfg.get("paths", {})
        if not self.paths:
            raise KeyError("Missing 'paths' in data_registry.yml")

        # Resolve all relative paths to absolute using project root
        self._resolve_paths()

        self.models_raw = self.cfg.get("models", {})
        self.processed = (self.cfg.get("processed", {}) or {}).get("models_1m", {})

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_paths(self) -> None:
        """Resolve relative paths in the loaded config using the project root.

        Template strings (containing ``{``) and already-absolute paths are
        left untouched.  Only plain relative paths are prepended with
        ``self._project_root``.
        """
        root = str(self._project_root)

        def _resolve(p: str) -> str:
            if not isinstance(p, str) or "{" in p or os.path.isabs(p):
                return p
            return os.path.join(root, p)

        # 1. paths: section
        for key in self.paths:
            if isinstance(self.paths[key], str):
                self.paths[key] = _resolve(self.paths[key])

        # 2. models: section (scenario → path; skip "variable")
        for model_dict in self.cfg.get("models", {}).values():
            if not isinstance(model_dict, dict):
                continue
            for k in list(model_dict.keys()):
                if k == "variable" or not isinstance(model_dict[k], str):
                    continue
                model_dict[k] = _resolve(model_dict[k])

        # 3. model_ancils: section (arbitrarily nested dicts of paths)
        def _resolve_nested(d: dict) -> None:
            for k in list(d.keys()):
                v = d[k]
                if isinstance(v, dict):
                    _resolve_nested(v)
                elif isinstance(v, str):
                    d[k] = _resolve(v)

        if "model_ancils" in self.cfg:
            _resolve_nested(self.cfg["model_ancils"])

        # 4. observations: section (only the "path" key per dataset)
        for dataset in self.cfg.get("observations", {}).values():
            if isinstance(dataset, dict) and "path" in dataset:
                dataset["path"] = _resolve(dataset["path"])

        # 5. processed: section (models_1m and observed_1m)
        for section_key in ("models_1m", "observed_1m"):
            section = (self.cfg.get("processed") or {}).get(section_key, {})
            if not isinstance(section, dict):
                continue
            for key, val in section.items():
                if isinstance(val, dict):
                    for k2 in list(val.keys()):
                        if isinstance(val[k2], str):
                            val[k2] = _resolve(val[k2])
                elif isinstance(val, str):
                    section[key] = _resolve(val)

    # ------------------------------------------------------------------
    # Model accessors
    # ------------------------------------------------------------------

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
    
    # Model ancillary resources for depth homogenization
    def get_model_ancil(self, *keys: str) -> str:
        """
        Return an ancillary path for a model, e.g.,
        get_ancil("watergap2-2e", "landcover").
        """
        node = self.cfg.get("model_ancils", {})
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                raise KeyError(f"Missing model ancils {'/'.join(keys)} in data_registry.yml")
            node = node[k]
        if not isinstance(node, str):
            raise KeyError(f"Ancillary path at {'/'.join(keys)} is not a string")
        path = os.path.expanduser(os.path.expandvars(node))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ancillary file not found: {path}")
        return path

    # ------------------------------------------------------------------
    # OBSERVATIONS
    # ------------------------------------------------------------------
    def get_obs_raw(self, dataset: str) -> str:
        """Return path pattern for raw observation dataset (from observations section)."""
        try:
            path = self.cfg["observations"][dataset]["path"]
        except KeyError as e:
            raise KeyError(f"Missing raw observation key in registry for dataset={dataset}") from e
        return path

    def get_obs_processed(self, dataset: str) -> str:
        """Return path for processed observation dataset (from processed.observed_1m section)."""
        try:
            path = self.cfg["processed"]["observed_1m"][dataset]
        except KeyError as e:
            raise KeyError(f"Missing processed observation key in registry for dataset={dataset}") from e
        # ensure parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    # ------------------------------------------------------------------
    # ADDITIONAL PROPERTIES
    # ------------------------------------------------------------------

    def get_common_period(self) -> tuple[str, str]:
        if "common_period" not in self.cfg:
            raise KeyError("Missing `common_period` in data_registry.yml")
        p = self.cfg["common_period"]
        return p["start"], p["end"]

    def get_obs_period(self, obs_key: str) -> tuple[str, str]:
        p = self.cfg["obs_periods"][obs_key]
        return p["start"], p["end"]

    def get_model_period(self) -> tuple[str, str]:
        p = self.cfg["model_period"]
        return p["start"], p["end"]

    def get_obs_ssi_method(self, obs_key: str) -> str:
        return (self.cfg.get("obs_ssi_method", {}) or {}).get(obs_key, "standard")

    # ------------------------------------------------------------------
    # Period helpers (shared by scripts and viz modules)
    # ------------------------------------------------------------------

    def resolve_ref_period(self, obskey: str, period_mode: str) -> tuple[str, str]:
        """
        Return (ref_start, ref_end) for an obs dataset given a period mode.

        * ``"common"`` – uses ``common_period`` from the registry
          (currently 2003-01 … 2019-12).
        * ``"maxspan"`` – intersection of the obs availability window
          with the model availability window.
        """
        if period_mode == "common":
            return self.get_common_period()

        if period_mode == "maxspan":
            o0, o1 = self.get_obs_period(obskey)
            m0, m1 = self.get_model_period()
            start = max(o0, m0)
            end = min(o1, m1)
            if start > end:
                raise ValueError(
                    f"Empty clipped window for {obskey}: "
                    f"{o0}..{o1} clipped to {m0}..{m1}"
                )
            return start, end

        raise ValueError(f"Unknown period_mode: {period_mode}")

    @staticmethod
    def corr_start_from_ref(ref_start: str, scale_months: int) -> str:
        """
        Compute the correlation-period start by offsetting *ref_start*
        forward by *scale_months* months (to avoid edge effects from
        the SSI rolling window).

        Parameters
        ----------
        ref_start : str
            ``"YYYY-MM"`` reference-period start.
        scale_months : int
            SSI accumulation scale, e.g. 3.

        Returns
        -------
        str
            ``"YYYY-MM"`` correlation-period start.
        """
        yr, mo = (int(x) for x in ref_start.split("-"))
        mo += scale_months
        while mo > 12:
            mo -= 12
            yr += 1
        return f"{yr:04d}-{mo:02d}"

    @property
    def cfg_dict(self) -> dict:
        """Return the full loaded YAML dict (for templates, etc.)."""
        return self.cfg

def default_registry() -> "Registry":
    """
    Construct a Registry by locating the project root and using configs/data_registry.yml.
    Works regardless of working directory (e.g., from notebooks).
    """
    root = project_root()
    yaml_path = root / "configs" / "data_registry.yml"
    return Registry(str(yaml_path), root=root)