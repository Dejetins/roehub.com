"""
Composition helpers for indicators API module.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from trading.contexts.indicators.adapters.outbound.registry import YamlIndicatorRegistry
from trading.contexts.indicators.domain.definitions import all_defs

_ENV_NAME_KEY = "ROEHUB_ENV"
_CONFIG_PATH_KEY = "ROEHUB_INDICATORS_CONFIG"
_ALLOWED_ENVS = ("dev", "prod", "test")


def build_indicators_registry(*, environ: Mapping[str, str]) -> YamlIndicatorRegistry:
    """
    Build fail-fast indicators registry from environment-aware YAML config.

    Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md

    Args:
        environ: Process environment mapping.
    Returns:
        YamlIndicatorRegistry: Ready-to-use merged registry adapter.
    Assumptions:
        Config path resolves to `configs/<env>/indicators.yaml` unless overridden.
    Raises:
        FileNotFoundError: If YAML file does not exist.
        ValueError: If environment/config is invalid or YAML validation fails.
    Side Effects:
        Reads defaults YAML from filesystem.
    """
    config_path = _resolve_indicators_config_path(environ=environ)
    return YamlIndicatorRegistry.from_yaml(
        defs=all_defs(),
        config_path=config_path,
    )


def _resolve_indicators_config_path(*, environ: Mapping[str, str]) -> Path:
    """
    Resolve indicators defaults config path from environment.

    Args:
        environ: Process environment mapping.
    Returns:
        Path: Path to `indicators.yaml`.
    Assumptions:
        Override env var has priority over derived `configs/<env>/...` path.
    Raises:
        ValueError: If env name is unsupported or override path is blank.
    Side Effects:
        None.
    """
    override = environ.get(_CONFIG_PATH_KEY, "").strip()
    if override:
        return Path(override)

    env_name = _resolve_env_name(environ=environ)
    return Path("configs") / env_name / "indicators.yaml"


def _resolve_env_name(*, environ: Mapping[str, str]) -> str:
    """
    Resolve runtime environment name for indicators config selection.

    Args:
        environ: Process environment mapping.
    Returns:
        str: One of `dev`, `prod`, `test`.
    Assumptions:
        Missing env variable defaults to `dev`.
    Raises:
        ValueError: If env value is not in allowed list.
    Side Effects:
        None.
    """
    raw_env = environ.get(_ENV_NAME_KEY, "dev").strip().lower()
    if raw_env not in _ALLOWED_ENVS:
        raise ValueError(
            f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}, got {raw_env!r}"
        )
    return raw_env
