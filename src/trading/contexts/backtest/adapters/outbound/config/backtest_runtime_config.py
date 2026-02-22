from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

_ENV_NAME_KEY = "ROEHUB_ENV"
_BACKTEST_CONFIG_PATH_KEY = "ROEHUB_BACKTEST_CONFIG"
_ALLOWED_ENVS = ("dev", "prod", "test")

_WARMUP_BARS_DEFAULT = 200
_TOP_K_DEFAULT = 300
_PRESELECT_DEFAULT = 20000
_TOP_TRADES_N_DEFAULT = 3

_INIT_CASH_QUOTE_DEFAULT = 10000.0
_FIXED_QUOTE_DEFAULT = 100.0
_SAFE_PROFIT_PERCENT_DEFAULT = 30.0
_SLIPPAGE_PCT_DEFAULT = 0.01
_FEE_PCT_DEFAULT_BY_MARKET_ID = {
    1: 0.075,
    2: 0.1,
    3: 0.075,
    4: 0.1,
}


@dataclass(frozen=True, slots=True)
class BacktestExecutionRuntimeConfig:
    """
    Runtime defaults for execution engine v1 loaded from `backtest.execution` section.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - configs/dev/backtest.yaml
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
    """

    init_cash_quote_default: float = _INIT_CASH_QUOTE_DEFAULT
    fixed_quote_default: float = _FIXED_QUOTE_DEFAULT
    safe_profit_percent_default: float = _SAFE_PROFIT_PERCENT_DEFAULT
    slippage_pct_default: float = _SLIPPAGE_PCT_DEFAULT
    fee_pct_default_by_market_id: Mapping[int, float] = field(
        default_factory=lambda: MappingProxyType(dict(_FEE_PCT_DEFAULT_BY_MARKET_ID))
    )

    def __post_init__(self) -> None:
        """
        Validate runtime execution defaults and normalize fee mapping immutability.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Percent fields use human percent units (`0.1 == 0.1%`).
        Raises:
            ValueError: If one scalar value or fee mapping entry is invalid.
        Side Effects:
            Replaces fee mapping with immutable key-sorted mapping proxy.
        """
        if self.init_cash_quote_default <= 0.0:
            raise ValueError("backtest.execution.init_cash_quote_default must be > 0")
        if self.fixed_quote_default <= 0.0:
            raise ValueError("backtest.execution.fixed_quote_default must be > 0")
        if self.safe_profit_percent_default < 0.0 or self.safe_profit_percent_default > 100.0:
            raise ValueError(
                "backtest.execution.safe_profit_percent_default must be in [0, 100]"
            )
        if self.slippage_pct_default < 0.0:
            raise ValueError("backtest.execution.slippage_pct_default must be >= 0")

        normalized_fee_map: dict[int, float] = {}
        for raw_market_id in sorted(self.fee_pct_default_by_market_id.keys()):
            market_id = int(raw_market_id)
            fee_pct = float(self.fee_pct_default_by_market_id[raw_market_id])
            if market_id <= 0:
                raise ValueError(
                    "backtest.execution.fee_pct_default_by_market_id keys must be > 0"
                )
            if fee_pct < 0.0:
                raise ValueError(
                    "backtest.execution.fee_pct_default_by_market_id values must be >= 0"
                )
            normalized_fee_map[market_id] = fee_pct

        if len(normalized_fee_map) == 0:
            raise ValueError("backtest.execution.fee_pct_default_by_market_id must be non-empty")

        object.__setattr__(
            self,
            "fee_pct_default_by_market_id",
            MappingProxyType(normalized_fee_map),
        )


@dataclass(frozen=True, slots=True)
class BacktestReportingRuntimeConfig:
    """
    Runtime defaults for reporting v1 loaded from `backtest.reporting` section.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - configs/dev/backtest.yaml
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    top_trades_n_default: int = _TOP_TRADES_N_DEFAULT

    def __post_init__(self) -> None:
        """
        Validate reporting defaults with fail-fast startup semantics.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Trades payload is returned only for top-ranked variants to limit response size.
        Raises:
            ValueError: If `top_trades_n_default` is non-positive.
        Side Effects:
            None.
        """
        if self.top_trades_n_default <= 0:
            raise ValueError("backtest.reporting.top_trades_n_default must be > 0")


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
    execution: BacktestExecutionRuntimeConfig = field(
        default_factory=BacktestExecutionRuntimeConfig
    )
    reporting: BacktestReportingRuntimeConfig = field(
        default_factory=BacktestReportingRuntimeConfig
    )

    def __post_init__(self) -> None:
        """
        Validate runtime config invariants for fail-fast startup behavior.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Version remains fixed to `1` for BKT contracts.
        Raises:
            ValueError: If version is not 1 or integer defaults are non-positive.
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
        if self.execution is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest.execution section must be configured")
        if self.reporting is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest.reporting section must be configured")


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
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
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
    execution_map = _get_mapping(backtest_map, "execution", required=False)
    reporting_map = _get_mapping(backtest_map, "reporting", required=False)

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

    execution = BacktestExecutionRuntimeConfig(
        init_cash_quote_default=_get_float_with_default(
            execution_map,
            "init_cash_quote_default",
            default=_INIT_CASH_QUOTE_DEFAULT,
        ),
        fixed_quote_default=_get_float_with_default(
            execution_map,
            "fixed_quote_default",
            default=_FIXED_QUOTE_DEFAULT,
        ),
        safe_profit_percent_default=_get_float_with_default(
            execution_map,
            "safe_profit_percent_default",
            default=_SAFE_PROFIT_PERCENT_DEFAULT,
        ),
        slippage_pct_default=_get_float_with_default(
            execution_map,
            "slippage_pct_default",
            default=_SLIPPAGE_PCT_DEFAULT,
        ),
        fee_pct_default_by_market_id=_parse_market_fee_defaults(
            data=_get_mapping(execution_map, "fee_pct_default_by_market_id", required=False)
        ),
    )
    reporting = BacktestReportingRuntimeConfig(
        top_trades_n_default=_get_int_with_default(
            reporting_map,
            "top_trades_n_default",
            default=_TOP_TRADES_N_DEFAULT,
        ),
    )

    return BacktestRuntimeConfig(
        version=version,
        warmup_bars_default=warmup_bars_default,
        top_k_default=top_k_default,
        preselect_default=preselect_default,
        execution=execution,
        reporting=reporting,
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
        Fallback defaults are validated by dataclass constructor.
    Raises:
        ValueError: If provided value type is invalid.
    Side Effects:
        None.
    """
    if key not in data:
        return default
    return _get_int(data, key, required=True)


def _get_float(data: Mapping[str, Any], key: str, *, required: bool) -> float:
    """
    Read float-compatible numeric value from payload while rejecting bools.

    Args:
        data: Source mapping.
        key: Numeric key name.
        required: Whether key is mandatory.
    Returns:
        float: Parsed floating-point value.
    Assumptions:
        Integer values are accepted and converted to float.
    Raises:
        ValueError: If required value is missing or type is invalid.
    Side Effects:
        None.
    """
    value = data.get(key)
    if value is None:
        if required:
            raise ValueError(f"missing required key: {key}")
        return 0.0
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"expected float at key '{key}', got {type(value).__name__}")
    return float(value)


def _get_float_with_default(data: Mapping[str, Any], key: str, *, default: float) -> float:
    """
    Read optional numeric value with explicit fallback default.

    Args:
        data: Source mapping.
        key: Numeric key name.
        default: Fallback value for absent key.
    Returns:
        float: Parsed floating-point value.
    Assumptions:
        Fallback defaults are validated by dataclass constructor.
    Raises:
        ValueError: If provided value type is invalid.
    Side Effects:
        None.
    """
    if key not in data:
        return default
    return _get_float(data, key, required=True)


def _parse_market_fee_defaults(*, data: Mapping[str, Any]) -> Mapping[int, float]:
    """
    Parse optional market-fee mapping from YAML execution section.

    Args:
        data: Raw mapping payload from YAML.
    Returns:
        Mapping[int, float]: Parsed immutable market-fee mapping.
    Assumptions:
        Empty payload falls back to fixed v1 defaults.
    Raises:
        ValueError: If key/value cannot be parsed into valid market id / fee pair.
    Side Effects:
        None.
    """
    if len(data) == 0:
        return MappingProxyType(dict(_FEE_PCT_DEFAULT_BY_MARKET_ID))

    normalized: dict[int, float] = {}
    for raw_key in sorted(data.keys(), key=lambda item: str(item).strip()):
        raw_value = data[raw_key]
        key_literal = str(raw_key).strip()
        if not key_literal:
            raise ValueError("fee_pct_default_by_market_id keys must be non-empty")
        try:
            market_id = int(key_literal)
        except ValueError as error:
            raise ValueError(
                "fee_pct_default_by_market_id keys must be integer literals"
            ) from error

        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ValueError("fee_pct_default_by_market_id values must be numeric")
        normalized[market_id] = float(raw_value)
    return MappingProxyType(normalized)


__all__ = [
    "BacktestExecutionRuntimeConfig",
    "BacktestReportingRuntimeConfig",
    "BacktestRuntimeConfig",
    "load_backtest_runtime_config",
    "resolve_backtest_config_path",
]
