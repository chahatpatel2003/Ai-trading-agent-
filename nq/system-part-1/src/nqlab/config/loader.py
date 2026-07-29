"""Layered configuration loader.

Merge order (later overrides earlier):

    base.yaml -> instruments/nq.yaml -> roll/volume.yaml
    -> profiles/<profile>.yaml -> explicit overrides -> environment

The result is validated into an immutable :class:`AppConfig`.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from nqlab.config.models import AppConfig, TopstepRules

_ENV_PREFIX = "NQLAB_"


def find_config_dir(start: Path | None = None) -> Path:
    """Locate the ``config/`` directory.

    Resolution order: ``$NQLAB_CONFIG_DIR`` -> nearest ancestor of ``start``/cwd
    containing ``config/base.yaml`` -> repo root inferred from this file.
    """
    env_dir = os.environ.get("NQLAB_CONFIG_DIR")
    if env_dir:
        return Path(env_dir).expanduser()

    here = (start or Path.cwd()).resolve()
    for parent in (here, *here.parents):
        if (parent / "config" / "base.yaml").exists():
            return parent / "config"

    # Fallback: <repo>/config relative to this file (src/nqlab/config/loader.py).
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "config"
    if (candidate / "base.yaml").exists():
        return candidate

    raise FileNotFoundError(
        "Could not locate a config/ directory. Set NQLAB_CONFIG_DIR or run from "
        "within the project tree."
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (dicts merge, else replace)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_env(config: dict[str, Any]) -> dict[str, Any]:
    """Apply a small, explicit set of environment overrides for common knobs."""
    env_map: dict[str, tuple[str, ...]] = {
        f"{_ENV_PREFIX}DATA_ROOT": ("data", "nq_data_root"),
        f"{_ENV_PREFIX}CURATED_ROOT": ("data", "curated_root"),
        f"{_ENV_PREFIX}CATALOG_PATH": ("data", "catalog_path"),
        f"{_ENV_PREFIX}ARTIFACTS_ROOT": ("data", "artifacts_root"),
        f"{_ENV_PREFIX}LOG_LEVEL": ("logging", "level"),
        f"{_ENV_PREFIX}LOG_FORMAT": ("logging", "format"),
    }
    result = copy.deepcopy(config)
    for env_name, path in env_map.items():
        value = os.environ.get(env_name)
        if value is None:
            continue
        cursor = result
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = value
    return result


def load_config(
    profile: str = "dev",
    *,
    config_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
    apply_env: bool = True,
) -> AppConfig:
    """Build the resolved :class:`AppConfig` for ``profile``."""
    cdir = config_dir or find_config_dir()

    merged: dict[str, Any] = {}
    layers = [
        cdir / "base.yaml",
        cdir / "instruments" / "nq.yaml",
        cdir / "roll" / "volume.yaml",
        cdir / "sessions" / "cme.yaml",
        cdir / "backtest" / "default.yaml",
        cdir / "validation" / "default.yaml",
        cdir / "profiles" / f"{profile}.yaml",
    ]
    for layer in layers:
        merged = _deep_merge(merged, _load_yaml(layer))

    if overrides:
        merged = _deep_merge(merged, overrides)
    if apply_env:
        merged = _apply_env(merged)

    merged["profile"] = profile
    return AppConfig.model_validate(merged)


def load_topstep_rules(name: str, *, config_dir: Path | None = None) -> TopstepRules:
    """Load a funded-account rule set from ``config/risk/<name>.yaml``."""
    cdir = config_dir or find_config_dir()
    path = cdir / "risk" / f"{name}.yaml"
    data = _load_yaml(path)
    if not data:
        raise FileNotFoundError(f"No Topstep rule set found: {path}")
    payload = data.get("topstep", data)
    return TopstepRules.model_validate(payload)
