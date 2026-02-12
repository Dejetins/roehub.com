"""
Runtime config loader for indicators Numba compute engine.

Docs: docs/architecture/indicators/indicators-compute-engine-core.md
Related: trading.contexts.indicators.adapters.outbound.compute_numba.engine,
  trading.contexts.indicators.adapters.outbound.compute_numba.warmup
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

_ENV_NAME_KEY = "ROEHUB_ENV"
_CONFIG_PATH_KEY = "ROEHUB_INDICATORS_CONFIG"
_ALLOWED_ENVS = ("dev", "prod", "test")

_THREADS_ENV_KEYS = ("ROEHUB_NUMBA_NUM_THREADS", "NUMBA_NUM_THREADS")
_CACHE_DIR_ENV_KEYS = ("ROEHUB_NUMBA_CACHE_DIR", "NUMBA_CACHE_DIR")
_MAX_TOTAL_ENV_KEYS = ("ROEHUB_MAX_COMPUTE_BYTES_TOTAL", "MAX_COMPUTE_BYTES_TOTAL")
_MAX_VARIANTS_ENV_KEYS = ("ROEHUB_MAX_VARIANTS_PER_COMPUTE", "MAX_VARIANTS_PER_COMPUTE")

_DEFAULT_NUMBA_NUM_THREADS = max(1, min(os.cpu_count() or 1, 16))
_DEFAULT_NUMBA_CACHE_DIR = Path(".cache/numba")
_DEFAULT_MAX_COMPUTE_BYTES_TOTAL = 5 * 1024**3
_DEFAULT_MAX_VARIANTS_PER_COMPUTE = 600_000


@dataclass(frozen=True, slots=True)
class IndicatorsComputeNumbaConfig:
    """
    Immutable runtime config for indicators CPU/Numba compute engine.

    Docs: docs/architecture/indicators/indicators-compute-engine-core.md
    Related: trading.contexts.indicators.adapters.outbound.compute_numba.engine,
      trading.contexts.indicators.adapters.outbound.compute_numba.warmup
    """

    numba_num_threads: int = _DEFAULT_NUMBA_NUM_THREADS
    numba_cache_dir: Path = _DEFAULT_NUMBA_CACHE_DIR
    max_compute_bytes_total: int = _DEFAULT_MAX_COMPUTE_BYTES_TOTAL
    max_variants_per_compute: int = _DEFAULT_MAX_VARIANTS_PER_COMPUTE

    def __post_init__(self) -> None:
        """
        Validate numba runtime config invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Thread count and total budget are positive integers.
        Raises:
            ValueError: If any value violates required bounds.
        Side Effects:
            Normalizes cache directory path to `Path`.
        """
        if self.numba_num_threads <= 0:
            raise ValueError(
                "numba_num_threads must be > 0, "
                f"got {self.numba_num_threads}"
            )
        if self.max_compute_bytes_total <= 0:
            raise ValueError(
                "max_compute_bytes_total must be > 0, "
                f"got {self.max_compute_bytes_total}"
            )
        if self.max_variants_per_compute <= 0:
            raise ValueError(
                "max_variants_per_compute must be > 0, "
                f"got {self.max_variants_per_compute}"
            )
        if not str(self.numba_cache_dir).strip():
            raise ValueError("numba_cache_dir must be a non-empty path")
        object.__setattr__(self, "numba_cache_dir", Path(self.numba_cache_dir))


def load_indicators_compute_numba_config(
    *,
    environ: Mapping[str, str],
) -> IndicatorsComputeNumbaConfig:
    """
    Load indicators Numba runtime config from YAML and env overrides.

    Args:
        environ: Environment mapping used to resolve env and override values.
    Returns:
        IndicatorsComputeNumbaConfig: Validated runtime settings.
    Assumptions:
        Optional `compute.numba` section lives in indicators YAML.
    Raises:
        FileNotFoundError: If indicators YAML path does not exist.
        ValueError: If YAML or environment values are invalid.
    Side Effects:
        Reads one YAML file from disk.
    """
    config_path = _resolve_indicators_config_path(environ=environ)
    file_payload = _load_optional_numba_payload(path=config_path)

    numba_num_threads = _resolve_int_setting(
        environ=environ,
        env_keys=_THREADS_ENV_KEYS,
        payload=file_payload,
        payload_key="numba_num_threads",
        default=_DEFAULT_NUMBA_NUM_THREADS,
    )
    numba_cache_dir = _resolve_path_setting(
        environ=environ,
        env_keys=_CACHE_DIR_ENV_KEYS,
        payload=file_payload,
        payload_key="numba_cache_dir",
        default=_DEFAULT_NUMBA_CACHE_DIR,
    )
    max_compute_bytes_total = _resolve_int_setting(
        environ=environ,
        env_keys=_MAX_TOTAL_ENV_KEYS,
        payload=file_payload,
        payload_key="max_compute_bytes_total",
        default=_DEFAULT_MAX_COMPUTE_BYTES_TOTAL,
    )
    max_variants_per_compute = _resolve_int_setting(
        environ=environ,
        env_keys=_MAX_VARIANTS_ENV_KEYS,
        payload=file_payload,
        payload_key="max_variants_per_compute",
        default=_DEFAULT_MAX_VARIANTS_PER_COMPUTE,
    )

    return IndicatorsComputeNumbaConfig(
        numba_num_threads=numba_num_threads,
        numba_cache_dir=numba_cache_dir,
        max_compute_bytes_total=max_compute_bytes_total,
        max_variants_per_compute=max_variants_per_compute,
    )


def _resolve_indicators_config_path(*, environ: Mapping[str, str]) -> Path:
    """
    Resolve indicators YAML path using explicit override or `ROEHUB_ENV`.

    Args:
        environ: Environment mapping.
    Returns:
        Path: Indicators YAML path.
    Assumptions:
        `ROEHUB_INDICATORS_CONFIG` has priority over env-derived path.
    Raises:
        ValueError: If env value is invalid.
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
    Resolve normalized runtime environment name.

    Args:
        environ: Environment mapping.
    Returns:
        str: One of `dev`, `prod`, `test`.
    Assumptions:
        Missing env falls back to `dev`.
    Raises:
        ValueError: If value is outside allowed set.
    Side Effects:
        None.
    """
    raw_env = environ.get(_ENV_NAME_KEY, "dev").strip().lower()
    if raw_env not in _ALLOWED_ENVS:
        raise ValueError(
            f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}, got {raw_env!r}"
        )
    return raw_env


def _load_optional_numba_payload(*, path: Path) -> Mapping[str, Any]:
    """
    Load optional `compute.numba` mapping from indicators YAML.

    Args:
        path: Indicators config path.
    Returns:
        Mapping[str, Any]: Optional `compute.numba` mapping, or empty mapping.
    Assumptions:
        Unknown keys are ignored by this loader.
    Raises:
        FileNotFoundError: If YAML path does not exist.
        ValueError: If YAML structure is invalid.
    Side Effects:
        Reads one UTF-8 file from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"indicators compute config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("indicators config must be a mapping at top-level")

    compute_map = raw.get("compute")
    if compute_map is None:
        return {}
    if not isinstance(compute_map, dict):
        raise ValueError("compute section must be a mapping")

    numba_map = compute_map.get("numba")
    if numba_map is None:
        return {}
    if not isinstance(numba_map, dict):
        raise ValueError("compute.numba section must be a mapping")
    return numba_map


def _resolve_int_setting(
    *,
    environ: Mapping[str, str],
    env_keys: tuple[str, ...],
    payload: Mapping[str, Any],
    payload_key: str,
    default: int,
) -> int:
    """
    Resolve integer setting from env -> payload -> default precedence.

    Args:
        environ: Environment mapping.
        env_keys: Candidate env variable names by priority.
        payload: Parsed YAML subsection.
        payload_key: YAML key name.
        default: Fallback default value.
    Returns:
        int: Resolved integer value.
    Assumptions:
        String env values use base-10 integer format.
    Raises:
        ValueError: If provided value cannot be parsed as positive int.
    Side Effects:
        None.
    """
    for env_key in env_keys:
        raw = environ.get(env_key, "").strip()
        if raw:
            value = _parse_positive_int(raw, key=env_key)
            return value

    payload_value = payload.get(payload_key)
    if payload_value is None:
        return default

    if isinstance(payload_value, bool) or not isinstance(payload_value, int):
        raise ValueError(
            f"expected int for compute.numba.{payload_key}, "
            f"got {type(payload_value).__name__}"
        )
    if payload_value <= 0:
        raise ValueError(
            f"compute.numba.{payload_key} must be > 0, got {payload_value}"
        )
    return payload_value


def _resolve_path_setting(
    *,
    environ: Mapping[str, str],
    env_keys: tuple[str, ...],
    payload: Mapping[str, Any],
    payload_key: str,
    default: Path,
) -> Path:
    """
    Resolve path setting from env -> payload -> default precedence.

    Args:
        environ: Environment mapping.
        env_keys: Candidate env variable names by priority.
        payload: Parsed YAML subsection.
        payload_key: YAML key name.
        default: Fallback default path.
    Returns:
        Path: Resolved non-empty path.
    Assumptions:
        Relative paths are allowed and resolved by caller context.
    Raises:
        ValueError: If provided path is blank or non-string in YAML.
    Side Effects:
        None.
    """
    for env_key in env_keys:
        raw = environ.get(env_key, "").strip()
        if raw:
            return Path(raw)

    payload_value = payload.get(payload_key)
    if payload_value is None:
        return default
    if not isinstance(payload_value, str):
        raise ValueError(
            f"expected string for compute.numba.{payload_key}, "
            f"got {type(payload_value).__name__}"
        )
    normalized = payload_value.strip()
    if not normalized:
        raise ValueError(f"compute.numba.{payload_key} must be non-empty")
    return Path(normalized)


def _parse_positive_int(raw: str, *, key: str) -> int:
    """
    Parse positive integer from environment string.

    Args:
        raw: Raw env string.
        key: Env key name for diagnostics.
    Returns:
        int: Parsed positive integer.
    Assumptions:
        Input value is stripped before parsing.
    Raises:
        ValueError: If value is not a positive integer.
    Side Effects:
        None.
    """
    try:
        parsed = int(raw, 10)
    except ValueError as error:
        raise ValueError(f"{key} must be int, got {raw!r}") from error
    if parsed <= 0:
        raise ValueError(f"{key} must be > 0, got {parsed}")
    return parsed


__all__ = [
    "IndicatorsComputeNumbaConfig",
    "load_indicators_compute_numba_config",
]
