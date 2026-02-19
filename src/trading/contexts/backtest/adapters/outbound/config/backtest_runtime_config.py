from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

_ENV_NAME_KEY = "ROEHUB_ENV"
_BACKTEST_CONFIG_PATH_KEY = "ROEHUB_BACKTEST_CONFIG"
_ALLOWED_ENVS = ("dev", "prod", "test")

_WARMUP_BARS_DEFAULT = 200
_TOP_K_DEFAULT = 300
_PRESELECT_DEFAULT = 20000


@dataclass(frozen=True, slots=True)
class BacktestRuntimeConfig:
    """
    Backtest runtime config v1 loaded from `configs/<env>/backtest.yaml`.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - configs/dev/backtest.yaml
      - configs/test/backtest.yaml
      - configs/prod/backtest.yaml
    """

    version: int
    warmup_bars_default: int = _WARMUP_BARS_DEFAULT
    top_k_default: int = _TOP_K_DEFAULT
    preselect_default: int = _PRESELECT_DEFAULT

    def __post_init__(self) -> None:
        """
        Validate runtime config invariants for fail-fast startup behavior.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Version remains fixed to `1` for BKT-EPIC-01 contract.
        Raises:
            ValueError: If version is not 1 or any integer defaults are non-positive.
        Side Effects:
            None.
        """
        if self.version != 1:
            raise ValueError(f"backtest config version must be 1, got {self.version!r}")
        if self.warmup_bars_default <= 0:
            raise ValueError("backtest.warmup_bars_default must be > 0")
        if self.top_k_default <= 0:
            raise ValueError("backtest.top_k_default must be > 0")
        if self.preselect_default <= 0:
            raise ValueError("backtest.preselect_default must be > 0")


def resolve_backtest_config_path(
    *,
    environ: Mapping[str, str],
) -> Path:
    """
    Resolve runtime config path using env override precedence contract.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - configs/dev/backtest.yaml
      - configs/test/backtest.yaml
      - configs/prod/backtest.yaml

    Args:
        environ: Runtime environment mapping.
    Returns:
        Path: Resolved `backtest.yaml` path.
    Assumptions:
        Precedence is `ROEHUB_BACKTEST_CONFIG` > `configs/<ROEHUB_ENV>/backtest.yaml`.
    Raises:
        ValueError: If `ROEHUB_ENV` value is unsupported.
    Side Effects:
        None.
    """
    override_path = environ.get(_BACKTEST_CONFIG_PATH_KEY, "").strip()
    if override_path:
        return Path(override_path)

    env_name = _resolve_env_name(environ=environ)
    return Path("configs") / env_name / "backtest.yaml"


def load_backtest_runtime_config(path: str | Path) -> BacktestRuntimeConfig:
    """
    Load and validate source-of-truth Backtest runtime YAML configuration.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - configs/dev/backtest.yaml
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - apps/api/wiring/modules

    Args:
        path: Path to `backtest.yaml`.
    Returns:
        BacktestRuntimeConfig: Parsed validated config object.
    Assumptions:
        Missing scalar keys fallback to documented defaults.
    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If YAML shape or values are invalid.
    Side Effects:
        Reads one UTF-8 YAML file from filesystem.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"backtest config not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("backtest config must be mapping at top-level")

    version = _get_int(payload, "version", required=True)
    backtest_map = _get_mapping(payload, "backtest", required=False)

    warmup_bars_default = _get_int_with_default(
        backtest_map,
        "warmup_bars_default",
        default=_WARMUP_BARS_DEFAULT,
    )
    top_k_default = _get_int_with_default(
        backtest_map,
        "top_k_default",
        default=_TOP_K_DEFAULT,
    )
    preselect_default = _get_int_with_default(
        backtest_map,
        "preselect_default",
        default=_PRESELECT_DEFAULT,
    )

    return BacktestRuntimeConfig(
        version=version,
        warmup_bars_default=warmup_bars_default,
        top_k_default=top_k_default,
        preselect_default=preselect_default,
    )


def _resolve_env_name(*, environ: Mapping[str, str]) -> str:
    """
    Resolve normalized runtime environment name for fallback path generation.

    Args:
        environ: Runtime environment mapping.
    Returns:
        str: One of `dev`, `prod`, or `test`.
    Assumptions:
        Missing `ROEHUB_ENV` defaults to `dev`.
    Raises:
        ValueError: If runtime env value is unsupported.
    Side Effects:
        None.
    """
    raw_env_name = environ.get(_ENV_NAME_KEY, "dev").strip().lower()
    if raw_env_name not in _ALLOWED_ENVS:
        raise ValueError(
            f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}, got {raw_env_name!r}"
        )
    return raw_env_name


def _get_mapping(data: Mapping[str, Any], key: str, *, required: bool) -> Mapping[str, Any]:
    """
    Read nested mapping from YAML payload.

    Args:
        data: Source mapping.
        key: Mapping key.
        required: Whether key is mandatory.
    Returns:
        Mapping[str, Any]: Nested mapping or empty mapping.
    Assumptions:
        Optional missing mapping sections are represented as empty mapping.
    Raises:
        ValueError: If required key missing or value is not mapping.
    Side Effects:
        None.
    """
    value = data.get(key)
    if value is None:
        if required:
            raise ValueError(f"missing required key: {key}")
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"expected mapping at key '{key}', got {type(value).__name__}")
    return value


def _get_int(data: Mapping[str, Any], key: str, *, required: bool) -> int:
    """
    Read integer value from payload while rejecting bools.

    Args:
        data: Source mapping.
        key: Integer key name.
        required: Whether key is mandatory.
    Returns:
        int: Parsed integer value.
    Assumptions:
        Bool values are rejected despite inheriting from `int`.
    Raises:
        ValueError: If missing required key or value type is invalid.
    Side Effects:
        None.
    """
    value = data.get(key)
    if value is None:
        if required:
            raise ValueError(f"missing required key: {key}")
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected int at key '{key}', got {type(value).__name__}")
    return value


def _get_int_with_default(data: Mapping[str, Any], key: str, *, default: int) -> int:
    """
    Read optional integer with explicit fallback default.

    Args:
        data: Source mapping.
        key: Integer key name.
        default: Fallback value for absent key.
    Returns:
        int: Parsed integer.
    Assumptions:
        Fallback defaults are already validated by dataclass constructor.
    Raises:
        ValueError: If provided value type is invalid.
    Side Effects:
        None.
    """
    if key not in data:
        return default
    return _get_int(data, key, required=True)


__all__ = [
    "BacktestRuntimeConfig",
    "load_backtest_runtime_config",
    "resolve_backtest_config_path",
]

