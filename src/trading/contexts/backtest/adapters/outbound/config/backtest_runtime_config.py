from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import UUID

import yaml

from trading.contexts.backtest.application.dto import (
    BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1,
    BACKTEST_RANKING_SECONDARY_METRIC_DEFAULT_V1,
    normalize_backtest_ranking_metric_literal,
)
from trading.contexts.indicators.application.services.grid_builder import (
    MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    MAX_VARIANTS_PER_COMPUTE_DEFAULT,
)

_ENV_NAME_KEY = "ROEHUB_ENV"
_BACKTEST_CONFIG_PATH_KEY = "ROEHUB_BACKTEST_CONFIG"
_ALLOWED_ENVS = ("dev", "prod", "test")

_WARMUP_BARS_DEFAULT = 200
_TOP_K_DEFAULT = 300
_PRESELECT_DEFAULT = 20000
_TOP_TRADES_N_DEFAULT = 3
_PRIMARY_METRIC_DEFAULT = BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1
_SECONDARY_METRIC_DEFAULT = BACKTEST_RANKING_SECONDARY_METRIC_DEFAULT_V1

_INIT_CASH_QUOTE_DEFAULT = 10000.0
_FIXED_QUOTE_DEFAULT = 100.0
_SAFE_PROFIT_PERCENT_DEFAULT = 30.0
_SLIPPAGE_PCT_DEFAULT = 0.01
_MAX_NUMBA_THREADS_DEFAULT = max(1, os.cpu_count() or 1)
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
class BacktestRankingRuntimeConfig:
    """
    Runtime defaults for ranking contract loaded from `backtest.ranking` section.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - configs/dev/backtest.yaml
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/dto/backtest_runtime_defaults.py
    """

    primary_metric_default: str = _PRIMARY_METRIC_DEFAULT
    secondary_metric_default: str | None = _SECONDARY_METRIC_DEFAULT

    def __post_init__(self) -> None:
        """
        Validate ranking defaults and normalize metric identifiers.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Metric identifiers follow lowercase snake_case literals from fixed v1 list.
        Raises:
            ValueError: If one ranking metric literal is invalid or duplicated.
        Side Effects:
            Normalizes metric literals to lowercase snake_case.
        """
        normalized_primary_metric = normalize_backtest_ranking_metric_literal(
            metric=self.primary_metric_default,
            field_path="backtest.ranking.primary_metric_default",
        )
        object.__setattr__(self, "primary_metric_default", normalized_primary_metric)

        if self.secondary_metric_default is None:
            return

        normalized_secondary_metric = normalize_backtest_ranking_metric_literal(
            metric=self.secondary_metric_default,
            field_path="backtest.ranking.secondary_metric_default",
        )
        if normalized_secondary_metric == normalized_primary_metric:
            raise ValueError(
                "backtest.ranking.secondary_metric_default must be different from "
                "backtest.ranking.primary_metric_default"
            )
        object.__setattr__(self, "secondary_metric_default", normalized_secondary_metric)


@dataclass(frozen=True, slots=True)
class BacktestGuardsRuntimeConfig:
    """
    Runtime guard limits for staged backtest compute loaded from `backtest.guards`.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
    Related:
      - configs/dev/backtest.yaml
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - apps/api/wiring/modules/backtest.py
    """

    max_variants_per_compute: int = MAX_VARIANTS_PER_COMPUTE_DEFAULT
    max_compute_bytes_total: int = MAX_COMPUTE_BYTES_TOTAL_DEFAULT

    def __post_init__(self) -> None:
        """
        Validate deterministic staged compute guard values.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Guard values are strict-positive integers in variants/bytes units.
        Raises:
            ValueError: If one guard value is non-positive.
        Side Effects:
            None.
        """
        if self.max_variants_per_compute <= 0:
            raise ValueError("backtest.guards.max_variants_per_compute must be > 0")
        if self.max_compute_bytes_total <= 0:
            raise ValueError("backtest.guards.max_compute_bytes_total must be > 0")


@dataclass(frozen=True, slots=True)
class BacktestCpuRuntimeConfig:
    """
    Runtime CPU settings for backtest compute loaded from `backtest.cpu`.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/runbooks/indicators-numba-cache-and-threads.md
    Related:
      - configs/dev/backtest.yaml
      - src/trading/contexts/backtest/application/services/numba_runtime_v1.py
      - apps/api/wiring/modules/backtest.py
    """

    max_numba_threads: int = _MAX_NUMBA_THREADS_DEFAULT

    def __post_init__(self) -> None:
        """
        Validate backtest CPU settings with fail-fast startup semantics.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Numba thread cap must stay strictly positive.
        Raises:
            ValueError: If `max_numba_threads` is non-positive.
        Side Effects:
            None.
        """
        if self.max_numba_threads <= 0:
            raise ValueError("backtest.cpu.max_numba_threads must be > 0")


@dataclass(frozen=True, slots=True)
class BacktestSyncRuntimeConfig:
    """
    Runtime sync settings loaded from strict required `backtest.sync.*` YAML section.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
    Related:
      - configs/dev/backtest.yaml
      - apps/api/routes/backtests.py
      - apps/api/wiring/modules/backtest.py
    """

    sync_deadline_seconds: float

    def __post_init__(self) -> None:
        """
        Validate sync-route deadline settings with fail-fast startup semantics.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Sync deadline is configured in seconds and must be strictly positive.
        Raises:
            ValueError: If `sync_deadline_seconds` is non-positive.
        Side Effects:
            None.
        """
        if self.sync_deadline_seconds <= 0.0:
            raise ValueError("backtest.sync.sync_deadline_seconds must be > 0")


@dataclass(frozen=True, slots=True)
class BacktestJobsRuntimeConfig:
    """
    Runtime jobs settings loaded from strict required `backtest.jobs.*` YAML section.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/milestone-5-epics-v1.md
    Related:
      - configs/dev/backtest.yaml
      - apps/worker/backtest_job_runner/main/main.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
    """

    enabled: bool
    top_k_persisted_default: int
    max_active_jobs_per_user: int
    claim_poll_seconds: float
    lease_seconds: int
    heartbeat_seconds: int
    parallel_workers: int
    snapshot_seconds: int | None = None
    snapshot_variants_step: int | None = None

    def __post_init__(self) -> None:
        """
        Validate strict jobs runtime fields with fail-fast startup behavior.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Required keys are always present in YAML and validated by loader.
        Raises:
            ValueError: If one field violates deterministic bounds.
        Side Effects:
            None.
        """
        if self.top_k_persisted_default <= 0:
            raise ValueError("backtest.jobs.top_k_persisted_default must be > 0")
        if self.max_active_jobs_per_user <= 0:
            raise ValueError("backtest.jobs.max_active_jobs_per_user must be > 0")
        if self.claim_poll_seconds <= 0:
            raise ValueError("backtest.jobs.claim_poll_seconds must be > 0")
        if self.lease_seconds <= 0:
            raise ValueError("backtest.jobs.lease_seconds must be > 0")
        if self.heartbeat_seconds <= 0:
            raise ValueError("backtest.jobs.heartbeat_seconds must be > 0")
        if self.parallel_workers <= 0:
            raise ValueError("backtest.jobs.parallel_workers must be > 0")

        if self.snapshot_seconds is not None and self.snapshot_seconds <= 0:
            raise ValueError("backtest.jobs.snapshot_seconds must be > 0 when provided")
        if self.snapshot_variants_step is not None and self.snapshot_variants_step <= 0:
            raise ValueError(
                "backtest.jobs.snapshot_variants_step must be > 0 when provided"
            )


@dataclass(frozen=True, slots=True)
class BacktestRuntimeConfig:
    """
    Backtest runtime config v1 loaded from `configs/<env>/backtest.yaml`.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - configs/dev/backtest.yaml
      - configs/test/backtest.yaml
      - configs/prod/backtest.yaml
    """

    version: int
    jobs: BacktestJobsRuntimeConfig
    sync: BacktestSyncRuntimeConfig
    warmup_bars_default: int = _WARMUP_BARS_DEFAULT
    top_k_default: int = _TOP_K_DEFAULT
    preselect_default: int = _PRESELECT_DEFAULT
    ranking: BacktestRankingRuntimeConfig = field(
        default_factory=BacktestRankingRuntimeConfig
    )
    execution: BacktestExecutionRuntimeConfig = field(
        default_factory=BacktestExecutionRuntimeConfig
    )
    reporting: BacktestReportingRuntimeConfig = field(
        default_factory=BacktestReportingRuntimeConfig
    )
    guards: BacktestGuardsRuntimeConfig = field(default_factory=BacktestGuardsRuntimeConfig)
    cpu: BacktestCpuRuntimeConfig = field(default_factory=BacktestCpuRuntimeConfig)

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
        if self.ranking is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest.ranking section must be configured")
        if self.execution is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest.execution section must be configured")
        if self.reporting is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest.reporting section must be configured")
        if self.guards is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest.guards section must be configured")
        if self.cpu is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest.cpu section must be configured")
        if self.jobs is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest.jobs section must be configured")
        if self.sync is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest.sync section must be configured")



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
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - configs/dev/backtest.yaml
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - apps/api/wiring/modules/backtest.py

    Args:
        path: Path to `backtest.yaml`.
    Returns:
        BacktestRuntimeConfig: Parsed validated config object.
    Assumptions:
        Missing non-required scalar keys fallback to documented defaults.
        `backtest.jobs.*` and `backtest.sync.sync_deadline_seconds` are strict-required.
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
    ranking_map = _get_mapping(backtest_map, "ranking", required=False)
    execution_map = _get_mapping(backtest_map, "execution", required=False)
    reporting_map = _get_mapping(backtest_map, "reporting", required=False)
    guards_map = _get_mapping(backtest_map, "guards", required=False)
    cpu_map = _get_mapping(backtest_map, "cpu", required=False)
    jobs_map = _get_mapping(backtest_map, "jobs", required=True)
    sync_map = _get_mapping(backtest_map, "sync", required=True)

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
    ranking = BacktestRankingRuntimeConfig(
        primary_metric_default=_get_str_with_default(
            ranking_map,
            "primary_metric_default",
            default=_PRIMARY_METRIC_DEFAULT,
        ),
        secondary_metric_default=_get_optional_str_with_default(
            ranking_map,
            "secondary_metric_default",
            default=_SECONDARY_METRIC_DEFAULT,
        ),
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
    guards = BacktestGuardsRuntimeConfig(
        max_variants_per_compute=_get_int_with_default(
            guards_map,
            "max_variants_per_compute",
            default=MAX_VARIANTS_PER_COMPUTE_DEFAULT,
        ),
        max_compute_bytes_total=_get_int_with_default(
            guards_map,
            "max_compute_bytes_total",
            default=MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
        ),
    )
    cpu = BacktestCpuRuntimeConfig(
        max_numba_threads=_get_int_with_default(
            cpu_map,
            "max_numba_threads",
            default=_MAX_NUMBA_THREADS_DEFAULT,
        )
    )
    jobs = BacktestJobsRuntimeConfig(
        enabled=_get_bool(jobs_map, "enabled", required=True),
        top_k_persisted_default=_get_int(jobs_map, "top_k_persisted_default", required=True),
        max_active_jobs_per_user=_get_int(jobs_map, "max_active_jobs_per_user", required=True),
        claim_poll_seconds=_get_float(jobs_map, "claim_poll_seconds", required=True),
        lease_seconds=_get_int(jobs_map, "lease_seconds", required=True),
        heartbeat_seconds=_get_int(jobs_map, "heartbeat_seconds", required=True),
        parallel_workers=_get_int(jobs_map, "parallel_workers", required=True),
        snapshot_seconds=_get_optional_int(jobs_map, "snapshot_seconds"),
        snapshot_variants_step=_get_optional_int(jobs_map, "snapshot_variants_step"),
    )
    sync = BacktestSyncRuntimeConfig(
        sync_deadline_seconds=_get_float(sync_map, "sync_deadline_seconds", required=True)
    )

    return BacktestRuntimeConfig(
        version=version,
        jobs=jobs,
        sync=sync,
        warmup_bars_default=warmup_bars_default,
        top_k_default=top_k_default,
        preselect_default=preselect_default,
        ranking=ranking,
        execution=execution,
        reporting=reporting,
        guards=guards,
        cpu=cpu,
    )



def build_backtest_runtime_config_hash(*, config: BacktestRuntimeConfig) -> str:
    """
    Build deterministic runtime hash from result-affecting Backtest runtime sections only.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
      - configs/dev/backtest.yaml
      - apps/api/dto/backtests.py

    Args:
        config: Parsed runtime config object.
    Returns:
        str: Canonical SHA-256 hash string.
    Assumptions:
        Hash includes result-affecting defaults (ranking/execution/reporting/persisted top-k)
        and excludes operational-only knobs.
    Raises:
        TypeError: If payload normalization fails for unsupported node type.
    Side Effects:
        None.
    """
    payload = {
        "backtest": {
            "warmup_bars_default": config.warmup_bars_default,
            "top_k_default": config.top_k_default,
            "preselect_default": config.preselect_default,
            "ranking": {
                "primary_metric_default": config.ranking.primary_metric_default,
                "secondary_metric_default": config.ranking.secondary_metric_default,
            },
            "execution": {
                "init_cash_quote_default": config.execution.init_cash_quote_default,
                "fixed_quote_default": config.execution.fixed_quote_default,
                "safe_profit_percent_default": config.execution.safe_profit_percent_default,
                "slippage_pct_default": config.execution.slippage_pct_default,
                "fee_pct_default_by_market_id": {
                    str(market_id): config.execution.fee_pct_default_by_market_id[market_id]
                    for market_id in sorted(config.execution.fee_pct_default_by_market_id.keys())
                },
            },
            "reporting": {
                "top_trades_n_default": config.reporting.top_trades_n_default,
            },
            "jobs": {
                "top_k_persisted_default": config.jobs.top_k_persisted_default,
            },
        }
    }
    canonical_json = json.dumps(
        _normalize_json_value(value=payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()



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



def _get_bool(data: Mapping[str, Any], key: str, *, required: bool) -> bool:
    """
    Read boolean value from payload.

    Args:
        data: Source mapping.
        key: Boolean key name.
        required: Whether key is mandatory.
    Returns:
        bool: Parsed boolean value.
    Assumptions:
        Bool type is strict and does not coerce integers.
    Raises:
        ValueError: If required key missing or value has invalid type.
    Side Effects:
        None.
    """
    value = data.get(key)
    if value is None:
        if required:
            raise ValueError(f"missing required key: {key}")
        return False
    if not isinstance(value, bool):
        raise ValueError(f"expected bool at key '{key}', got {type(value).__name__}")
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



def _get_optional_int(data: Mapping[str, Any], key: str) -> int | None:
    """
    Read optional integer value from payload while rejecting bools.

    Args:
        data: Source mapping.
        key: Integer key name.
    Returns:
        int | None: Parsed integer value or `None` when key is absent.
    Assumptions:
        Optional field absence means no runtime override is configured.
    Raises:
        ValueError: If provided value type is invalid.
    Side Effects:
        None.
    """
    if key not in data:
        return None
    return _get_int(data, key, required=True)



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



def _get_str(data: Mapping[str, Any], key: str, *, required: bool) -> str:
    """
    Read string literal value from payload.

    Args:
        data: Source mapping.
        key: String key name.
        required: Whether key is mandatory.
    Returns:
        str: Parsed string value.
    Assumptions:
        Value is used as metric identifier literal and stripped by downstream validators.
    Raises:
        ValueError: If required key is missing or value type is invalid.
    Side Effects:
        None.
    """
    value = data.get(key)
    if value is None:
        if required:
            raise ValueError(f"missing required key: {key}")
        return ""
    if not isinstance(value, str):
        raise ValueError(f"expected str at key '{key}', got {type(value).__name__}")
    return value


def _get_str_with_default(data: Mapping[str, Any], key: str, *, default: str) -> str:
    """
    Read optional string literal with explicit fallback default.

    Args:
        data: Source mapping.
        key: String key name.
        default: Fallback value for absent key.
    Returns:
        str: Parsed string literal.
    Assumptions:
        Fallback default is validated by downstream ranking runtime config object.
    Raises:
        ValueError: If provided value type is invalid.
    Side Effects:
        None.
    """
    if key not in data:
        return default
    return _get_str(data, key, required=True)


def _get_optional_str_with_default(
    data: Mapping[str, Any],
    key: str,
    *,
    default: str | None,
) -> str | None:
    """
    Read optional string literal with explicit fallback and `null` support.

    Args:
        data: Source mapping.
        key: String key name.
        default: Fallback value when key is absent.
    Returns:
        str | None: Parsed string or `None`.
    Assumptions:
        YAML `null` value maps to Python `None` and means disabled secondary metric.
    Raises:
        ValueError: If provided non-null value type is invalid.
    Side Effects:
        None.
    """
    if key not in data:
        return default
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"expected str at key '{key}', got {type(value).__name__}")
    return value


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



def _normalize_json_value(*, value: Any) -> Any:
    """
    Normalize arbitrary payload node into deterministic JSON-serializable value.

    Args:
        value: Arbitrary payload node.
    Returns:
        Any: JSON-serializable normalized value.
    Assumptions:
        Mapping keys are converted to strings and sorted recursively.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda key: str(key)):
            normalized[str(raw_key)] = _normalize_json_value(value=value[raw_key])
        return normalized

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_value(value=item) for item in value]

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    return value


__all__ = [
    "BacktestCpuRuntimeConfig",
    "BacktestExecutionRuntimeConfig",
    "BacktestGuardsRuntimeConfig",
    "BacktestJobsRuntimeConfig",
    "BacktestRankingRuntimeConfig",
    "BacktestReportingRuntimeConfig",
    "BacktestRuntimeConfig",
    "BacktestSyncRuntimeConfig",
    "build_backtest_runtime_config_hash",
    "load_backtest_runtime_config",
    "resolve_backtest_config_path",
]
