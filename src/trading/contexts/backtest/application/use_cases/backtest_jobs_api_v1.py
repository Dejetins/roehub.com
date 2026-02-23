from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Mapping
from uuid import UUID, uuid4

from trading.contexts.backtest.application.dto import RunBacktestRequest, RunBacktestTemplate
from trading.contexts.backtest.application.ports import (
    BacktestJobListQuery,
    BacktestJobRepository,
    BacktestJobResultsRepository,
    BacktestStrategyReader,
    BacktestStrategySnapshot,
    CurrentUser,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobMode,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.errors import (
    BacktestForbiddenError,
    BacktestNotFoundError,
    BacktestValidationError,
)
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor, ExecutionParamsV1

from .errors import backtest_job_forbidden, backtest_job_not_found, validation_error

NowProvider = Callable[[], datetime]
JobIdFactory = Callable[[], UUID]


@dataclass(frozen=True, slots=True)
class CreateBacktestJobCommand:
    """
    Command payload for EPIC-11 `POST /backtests/jobs` job creation orchestration.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/routes/backtest_jobs.py
      - apps/api/dto/backtest_jobs.py
    """

    run_request: RunBacktestRequest
    request_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        """
        Validate command payload invariants required for deterministic request snapshots.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `request_payload` is derived from strict API DTO with `extra=forbid`.
        Raises:
            ValueError: If request objects are missing.
        Side Effects:
            None.
        """
        if self.run_request is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateBacktestJobCommand.run_request is required")
        if self.request_payload is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateBacktestJobCommand.request_payload is required")


@dataclass(frozen=True, slots=True)
class _ResolvedJobCreationContext:
    """
    Internal resolved EPIC-11 create context used for deterministic snapshot/hash creation.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/routes/backtest_jobs.py
      - apps/api/dto/backtest_jobs.py
    """

    mode: BacktestJobMode
    template: RunBacktestTemplate
    warmup_bars: int
    top_k: int
    preselect: int
    top_trades_n: int
    spec_hash: str | None
    spec_payload_json: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class BacktestJobTopReadResult:
    """
    Top-read payload for EPIC-11 `/backtests/jobs/{job_id}/top` use-case contract.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/dto/backtest_jobs.py
      - apps/api/routes/backtest_jobs.py
    """

    job: BacktestJob
    rows: tuple[BacktestJobTopVariant, ...]


class CreateBacktestJobUseCase:
    """
    Create queued Backtest job with owner/quota/validation checks and canonical hashes.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - apps/api/routes/backtest_jobs.py
    """

    def __init__(
        self,
        *,
        job_repository: BacktestJobRepository,
        strategy_reader: BacktestStrategyReader,
        top_k_persisted_default: int,
        max_active_jobs_per_user: int,
        warmup_bars_default: int,
        top_k_default: int,
        preselect_default: int,
        top_trades_n_default: int,
        init_cash_quote_default: float,
        fixed_quote_default: float,
        safe_profit_percent_default: float,
        slippage_pct_default: float,
        fee_pct_default_by_market_id: Mapping[int, float],
        backtest_runtime_config_hash: str,
        now_provider: NowProvider | None = None,
        job_id_factory: JobIdFactory | None = None,
    ) -> None:
        """
        Initialize create-job use-case dependencies and deterministic validation settings.

        Args:
            job_repository: Jobs repository port for create/quota operations.
            strategy_reader: Saved strategy snapshot reader for saved-mode ownership checks.
            top_k_persisted_default: Persisted top-k cap from runtime config.
            max_active_jobs_per_user: Active jobs quota for one user.
            warmup_bars_default: Runtime default for warmup bars.
            top_k_default: Runtime default for top-k.
            preselect_default: Runtime default for preselect.
            top_trades_n_default: Runtime default for top-trades payload cap.
            init_cash_quote_default: Runtime default for execution init cash.
            fixed_quote_default: Runtime default for fixed quote sizing.
            safe_profit_percent_default: Runtime default for profit lock percent.
            slippage_pct_default: Runtime default for slippage percent.
            fee_pct_default_by_market_id: Runtime default fee mapping by market id.
            backtest_runtime_config_hash: Precomputed result-affecting runtime hash.
            now_provider: Optional UTC clock provider.
            job_id_factory: Optional deterministic job-id factory for tests.
        Returns:
            None.
        Assumptions:
            Constructor validates invariants and does not perform IO.
        Raises:
            ValueError: If one dependency or scalar invariant is invalid.
        Side Effects:
            None.
        """
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateBacktestJobUseCase requires job_repository")
        if strategy_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateBacktestJobUseCase requires strategy_reader")
        if top_k_persisted_default <= 0:
            raise ValueError("top_k_persisted_default must be > 0")
        if max_active_jobs_per_user <= 0:
            raise ValueError("max_active_jobs_per_user must be > 0")
        if warmup_bars_default <= 0:
            raise ValueError("warmup_bars_default must be > 0")
        if top_k_default <= 0:
            raise ValueError("top_k_default must be > 0")
        if preselect_default <= 0:
            raise ValueError("preselect_default must be > 0")
        if top_trades_n_default <= 0:
            raise ValueError("top_trades_n_default must be > 0")
        if top_k_default > top_k_persisted_default:
            raise ValueError("top_k_default must be <= top_k_persisted_default")
        if init_cash_quote_default <= 0.0:
            raise ValueError("init_cash_quote_default must be > 0")
        if fixed_quote_default <= 0.0:
            raise ValueError("fixed_quote_default must be > 0")
        if safe_profit_percent_default < 0.0 or safe_profit_percent_default > 100.0:
            raise ValueError("safe_profit_percent_default must be in [0, 100]")
        if slippage_pct_default < 0.0:
            raise ValueError("slippage_pct_default must be >= 0")
        if len(backtest_runtime_config_hash.strip()) != 64:
            raise ValueError("backtest_runtime_config_hash must be 64 lowercase hex chars")

        self._job_repository = job_repository
        self._strategy_reader = strategy_reader
        self._top_k_persisted_default = top_k_persisted_default
        self._max_active_jobs_per_user = max_active_jobs_per_user
        self._warmup_bars_default = warmup_bars_default
        self._top_k_default = top_k_default
        self._preselect_default = preselect_default
        self._top_trades_n_default = top_trades_n_default
        self._init_cash_quote_default = init_cash_quote_default
        self._fixed_quote_default = fixed_quote_default
        self._safe_profit_percent_default = safe_profit_percent_default
        self._slippage_pct_default = slippage_pct_default
        self._fee_pct_default_by_market_id = _normalize_fee_defaults(
            values=fee_pct_default_by_market_id
        )
        self._backtest_runtime_config_hash = backtest_runtime_config_hash.strip().lower()
        self._now = now_provider or _utc_now
        self._job_id_factory = job_id_factory or uuid4

    def execute(
        self,
        *,
        command: CreateBacktestJobCommand,
        current_user: CurrentUser,
    ) -> BacktestJob:
        """
        Validate and persist queued Backtest job snapshot for authenticated owner.

        Args:
            command: Create command with parsed run request and strict API payload snapshot.
            current_user: Authenticated owner identity.
        Returns:
            BacktestJob: Persisted queued job snapshot.
        Assumptions:
            Use-case is responsible for owner/quota checks and canonical hash creation.
        Raises:
            RoehubError: Canonical `validation_error|forbidden|not_found` via mapped helpers.
            ValueError: If deterministic payload normalization fails unexpectedly.
        Side Effects:
            Reads strategy snapshot and writes one row to `backtest_jobs`.
        """
        if command is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateBacktestJobUseCase.execute requires command")
        if current_user is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateBacktestJobUseCase.execute requires current_user")

        active_jobs_total = self._job_repository.count_active_for_user(
            user_id=current_user.user_id,
        )
        if active_jobs_total >= self._max_active_jobs_per_user:
            raise validation_error(
                message=(
                    "Active backtest jobs quota exceeded"
                    f" (max={self._max_active_jobs_per_user})"
                ),
                errors=(
                    {
                        "path": "body",
                        "code": "quota_exceeded",
                        "message": (
                            "active jobs limit reached"
                            f" ({active_jobs_total}/{self._max_active_jobs_per_user})"
                        ),
                    },
                ),
            )

        resolved = self._resolve_create_context(
            command=command,
            current_user=current_user,
        )
        effective_execution_payload = self._resolve_effective_execution_payload(
            template=resolved.template,
        )
        request_json = self._build_request_json_payload(
            request_payload=command.request_payload,
            run_request=command.run_request,
            resolved=resolved,
            effective_execution_payload=effective_execution_payload,
        )

        job = BacktestJob.create_queued(
            job_id=self._job_id_factory(),
            user_id=current_user.user_id,
            mode=resolved.mode,
            created_at=self._now(),
            request_json=request_json,
            request_hash=_build_sha256_from_payload(payload=request_json),
            spec_hash=resolved.spec_hash,
            spec_payload_json=resolved.spec_payload_json,
            engine_params_hash=_build_sha256_from_payload(
                payload={
                    "direction_mode": resolved.template.direction_mode,
                    "sizing_mode": resolved.template.sizing_mode,
                    "execution": effective_execution_payload,
                }
            ),
            backtest_runtime_config_hash=self._backtest_runtime_config_hash,
        )
        return self._job_repository.create(job=job)

    def _resolve_create_context(
        self,
        *,
        command: CreateBacktestJobCommand,
        current_user: CurrentUser,
    ) -> _ResolvedJobCreationContext:
        """
        Resolve mode/template/defaults and saved snapshot invariants for create flow.

        Args:
            command: Create command payload.
            current_user: Authenticated owner identity.
        Returns:
            _ResolvedJobCreationContext: Resolved deterministic create context.
        Assumptions:
            Request mode (`saved|template`) was validated in `RunBacktestRequest`.
        Raises:
            BacktestValidationError: If one EPIC-11 validation invariant is violated.
            BacktestNotFoundError: If saved strategy snapshot is missing or deleted.
            BacktestForbiddenError: If saved strategy owner mismatches current user.
        Side Effects:
            Reads saved strategy snapshot in `saved` mode.
        """
        run_request = command.run_request
        warmup_bars = _resolve_positive_override(
            value=run_request.warmup_bars,
            default=self._warmup_bars_default,
            field_path="body.warmup_bars",
        )
        top_k = _resolve_positive_override(
            value=run_request.top_k,
            default=self._top_k_default,
            field_path="body.top_k",
        )
        preselect = _resolve_positive_override(
            value=run_request.preselect,
            default=self._preselect_default,
            field_path="body.preselect",
        )
        top_trades_n = _resolve_positive_override(
            value=run_request.top_trades_n,
            default=self._top_trades_n_default,
            field_path="body.top_trades_n",
        )

        if top_k > self._top_k_persisted_default:
            raise validation_error(
                message=(
                    "Backtest jobs request top_k must be <= "
                    "backtest.jobs.top_k_persisted_default"
                ),
                errors=(
                    {
                        "path": "body.top_k",
                        "code": "max_value",
                        "message": (
                            f"top_k must be <= {self._top_k_persisted_default}"
                        ),
                    },
                ),
            )

        if run_request.top_trades_n is not None and top_trades_n > top_k:
            raise validation_error(
                message="Backtest jobs request top_trades_n must be <= top_k",
                errors=(
                    {
                        "path": "body.top_trades_n",
                        "code": "max_value",
                        "message": "top_trades_n must be <= top_k",
                    },
                ),
            )
        if top_trades_n > top_k:
            top_trades_n = top_k

        if run_request.strategy_id is None:
            if run_request.template is None:
                raise BacktestValidationError(
                    "RunBacktestRequest.template is required for template mode"
                )
            return _ResolvedJobCreationContext(
                mode="template",
                template=run_request.template,
                warmup_bars=warmup_bars,
                top_k=top_k,
                preselect=preselect,
                top_trades_n=top_trades_n,
                spec_hash=None,
                spec_payload_json=None,
            )

        snapshot = self._strategy_reader.load_any(strategy_id=run_request.strategy_id)
        base_template = _template_from_snapshot(
            strategy_id=run_request.strategy_id,
            snapshot=snapshot,
            current_user=current_user,
        )
        resolved_template = _apply_saved_overrides(
            base_template=base_template,
            overrides=run_request.overrides,
        )
        spec_payload = dict(snapshot.spec_payload or {}) if snapshot is not None else {}
        if len(spec_payload) == 0:
            raise BacktestValidationError("saved mode job requires non-empty strategy spec payload")

        return _ResolvedJobCreationContext(
            mode="saved",
            template=resolved_template,
            warmup_bars=warmup_bars,
            top_k=top_k,
            preselect=preselect,
            top_trades_n=top_trades_n,
            spec_hash=_build_sha256_from_payload(payload=spec_payload),
            spec_payload_json=MappingProxyType(spec_payload),
        )

    def _resolve_effective_execution_payload(
        self,
        *,
        template: RunBacktestTemplate,
    ) -> Mapping[str, float]:
        """
        Resolve full effective execution parameters from template overrides and runtime defaults.

        Args:
            template: Effective template resolved for create flow.
        Returns:
            Mapping[str, float]: Canonical execution mapping used by hashes and snapshots.
        Assumptions:
            Unknown market id without fee default is treated as validation failure.
        Raises:
            BacktestValidationError: If fee default for template market id is missing.
            ValueError: If execution scalar shape is invalid.
        Side Effects:
            None.
        """
        execution_values = template.execution_params or {}
        market_id = template.instrument_id.market_id.value
        if market_id not in self._fee_pct_default_by_market_id:
            raise validation_error(
                message="Backtest jobs request market_id does not have fee default",
                errors=(
                    {
                        "path": "body.template.instrument_id.market_id",
                        "code": "unsupported",
                        "message": f"unsupported market_id: {market_id}",
                    },
                ),
            )

        execution_params = ExecutionParamsV1(
            direction_mode=template.direction_mode,
            sizing_mode=template.sizing_mode,
            init_cash_quote=_resolve_number(
                values=execution_values,
                primary_key="init_cash_quote",
                secondary_key="init_cash",
                default=self._init_cash_quote_default,
            ),
            fixed_quote=_resolve_number(
                values=execution_values,
                primary_key="fixed_quote",
                secondary_key="",
                default=self._fixed_quote_default,
            ),
            safe_profit_percent=_resolve_number(
                values=execution_values,
                primary_key="safe_profit_percent",
                secondary_key="",
                default=self._safe_profit_percent_default,
            ),
            fee_pct=_resolve_number(
                values=execution_values,
                primary_key="fee_pct",
                secondary_key="market_fee_pct",
                default=self._fee_pct_default_by_market_id[market_id],
            ),
            slippage_pct=_resolve_number(
                values=execution_values,
                primary_key="slippage_pct",
                secondary_key="",
                default=self._slippage_pct_default,
            ),
        )
        return MappingProxyType(
            {
                "init_cash_quote": execution_params.init_cash_quote,
                "fixed_quote": execution_params.fixed_quote,
                "safe_profit_percent": execution_params.safe_profit_percent,
                "fee_pct": execution_params.fee_pct,
                "slippage_pct": execution_params.slippage_pct,
            }
        )

    def _build_request_json_payload(
        self,
        *,
        request_payload: Mapping[str, Any],
        run_request: RunBacktestRequest,
        resolved: _ResolvedJobCreationContext,
        effective_execution_payload: Mapping[str, float],
    ) -> Mapping[str, Any]:
        """
        Build worker-compatible request snapshot payload with effective scalar defaults.

        Args:
            request_payload: Strict API request mapping payload.
            run_request: Parsed application request DTO.
            resolved: Resolved create context.
            effective_execution_payload: Full execution mapping resolved with defaults.
        Returns:
            Mapping[str, Any]: Deterministic snapshot payload persisted in `request_json`.
        Assumptions:
            Snapshot transport shape stays compatible with worker decoder API DTO.
        Raises:
            BacktestValidationError: If required template payload is missing in template mode.
            ValueError: If payload normalization fails.
        Side Effects:
            None.
        """
        normalized_payload = _normalize_json_mapping(values=request_payload)
        normalized_payload["warmup_bars"] = resolved.warmup_bars
        normalized_payload["top_k"] = resolved.top_k
        normalized_payload["preselect"] = resolved.preselect
        normalized_payload["top_trades_n"] = resolved.top_trades_n

        if resolved.mode == "template":
            raw_template = normalized_payload.get("template")
            if not isinstance(raw_template, Mapping):
                raise BacktestValidationError("template mode request requires template payload")
            template_payload = dict(raw_template)
            template_payload["direction_mode"] = resolved.template.direction_mode
            template_payload["sizing_mode"] = resolved.template.sizing_mode
            template_payload["execution"] = {
                key: effective_execution_payload[key]
                for key in sorted(effective_execution_payload.keys())
            }
            normalized_payload["template"] = template_payload
            normalized_payload.pop("overrides", None)
            return MappingProxyType(normalized_payload)

        overrides_payload: dict[str, Any] = {}
        raw_overrides = normalized_payload.get("overrides")
        if isinstance(raw_overrides, Mapping):
            overrides_payload = dict(raw_overrides)
        overrides_payload["direction_mode"] = resolved.template.direction_mode
        overrides_payload["sizing_mode"] = resolved.template.sizing_mode
        overrides_payload["execution"] = {
            key: effective_execution_payload[key]
            for key in sorted(effective_execution_payload.keys())
        }
        normalized_payload["overrides"] = overrides_payload

        if run_request.strategy_id is not None:
            normalized_payload["strategy_id"] = str(run_request.strategy_id)
        normalized_payload.pop("template", None)
        return MappingProxyType(normalized_payload)


class GetBacktestJobStatusUseCase:
    """
    Read one owner-scoped Backtest job status snapshot with explicit 403/404 semantics.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - apps/api/routes/backtest_jobs.py
      - apps/api/dto/backtest_jobs.py
    """

    def __init__(self, *, job_repository: BacktestJobRepository) -> None:
        """
        Initialize status use-case with jobs repository dependency.

        Args:
            job_repository: Jobs repository port.
        Returns:
            None.
        Assumptions:
            Repository supports unscoped read for explicit owner checks.
        Raises:
            ValueError: If dependency is missing.
        Side Effects:
            None.
        """
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("GetBacktestJobStatusUseCase requires job_repository")
        self._job_repository = job_repository

    def execute(self, *, job_id: UUID, current_user: CurrentUser) -> BacktestJob:
        """
        Read owner job status snapshot with explicit foreign/not-found branching.

        Args:
            job_id: Requested job identifier.
            current_user: Authenticated owner identity.
        Returns:
            BacktestJob: Owner job snapshot.
        Assumptions:
            Existing foreign resource must map to `403 forbidden`.
        Raises:
            RoehubError: Canonical `forbidden` or `not_found` for ownership checks.
        Side Effects:
            Reads one job row from storage.
        """
        return _require_owner_job(
            job_repository=self._job_repository,
            job_id=job_id,
            current_user=current_user,
        )


class GetBacktestJobTopUseCase:
    """
    Read owner job top snapshot rows with deterministic limit/state policy validation.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - apps/api/routes/backtest_jobs.py
      - apps/api/dto/backtest_jobs.py
    """

    def __init__(
        self,
        *,
        job_repository: BacktestJobRepository,
        results_repository: BacktestJobResultsRepository,
        top_k_persisted_default: int,
    ) -> None:
        """
        Initialize top-read use-case dependencies and persisted limit policy.

        Args:
            job_repository: Jobs repository port.
            results_repository: Results repository port.
            top_k_persisted_default: Persisted top cap runtime setting.
        Returns:
            None.
        Assumptions:
            Top read limit must not exceed persisted cap.
        Raises:
            ValueError: If dependency or limit invariant is invalid.
        Side Effects:
            None.
        """
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("GetBacktestJobTopUseCase requires job_repository")
        if results_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("GetBacktestJobTopUseCase requires results_repository")
        if top_k_persisted_default <= 0:
            raise ValueError("top_k_persisted_default must be > 0")
        self._job_repository = job_repository
        self._results_repository = results_repository
        self._top_k_persisted_default = top_k_persisted_default

    def execute(
        self,
        *,
        job_id: UUID,
        current_user: CurrentUser,
        limit: int | None,
    ) -> BacktestJobTopReadResult:
        """
        Read owner top snapshot rows with deterministic limit validation.

        Args:
            job_id: Requested job identifier.
            current_user: Authenticated owner identity.
            limit: Optional top rows limit.
        Returns:
            BacktestJobTopReadResult: Owner job and persisted rows payload.
        Assumptions:
            Row ordering in repository is deterministic (`rank ASC, variant_key ASC`).
        Raises:
            RoehubError: Canonical `forbidden|not_found|validation_error` errors.
        Side Effects:
            Reads one job row and zero or more top snapshot rows from storage.
        """
        resolved_limit = self._resolve_limit(limit=limit)
        owner_job = _require_owner_job(
            job_repository=self._job_repository,
            job_id=job_id,
            current_user=current_user,
        )
        rows = self._results_repository.list_top_variants(job_id=job_id, limit=resolved_limit)
        return BacktestJobTopReadResult(job=owner_job, rows=rows)

    def _resolve_limit(self, *, limit: int | None) -> int:
        """
        Resolve `/top` limit against persisted cap with deterministic validation errors.

        Args:
            limit: Optional query limit.
        Returns:
            int: Effective positive limit within persisted cap.
        Assumptions:
            Missing limit falls back to `top_k_persisted_default`.
        Raises:
            RoehubError: Canonical `validation_error` when limit is out of bounds.
        Side Effects:
            None.
        """
        if limit is None:
            return self._top_k_persisted_default
        if limit <= 0:
            raise validation_error(
                message="Top rows limit must be > 0",
                errors=(
                    {
                        "path": "query.limit",
                        "code": "greater_than",
                        "message": "limit must be > 0",
                    },
                ),
            )
        if limit > self._top_k_persisted_default:
            raise validation_error(
                message=(
                    "Top rows limit must be <= backtest.jobs.top_k_persisted_default"
                ),
                errors=(
                    {
                        "path": "query.limit",
                        "code": "max_value",
                        "message": (
                            f"limit must be <= {self._top_k_persisted_default}"
                        ),
                    },
                ),
            )
        return limit


class ListBacktestJobsUseCase:
    """
    List owner Backtest jobs using deterministic keyset pagination contract.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - src/trading/contexts/backtest/domain/value_objects/backtest_job_cursor.py
      - apps/api/routes/backtest_jobs.py
    """

    def __init__(self, *, job_repository: BacktestJobRepository) -> None:
        """
        Initialize list use-case with jobs repository dependency.

        Args:
            job_repository: Jobs repository port.
        Returns:
            None.
        Assumptions:
            Repository provides keyset page payload for `(created_at DESC, job_id DESC)`.
        Raises:
            ValueError: If dependency is missing.
        Side Effects:
            None.
        """
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("ListBacktestJobsUseCase requires job_repository")
        self._job_repository = job_repository

    def execute(
        self,
        *,
        current_user: CurrentUser,
        state: BacktestJobState | None,
        limit: int,
        cursor: BacktestJobListCursor | None,
    ):
        """
        Read owner jobs page using deterministic keyset list query contract.

        Args:
            current_user: Authenticated owner identity.
            state: Optional state filter literal.
            limit: Requested page size.
            cursor: Optional keyset cursor.
        Returns:
            BacktestJobListPage: Repository page payload with deterministic `next_cursor`.
        Assumptions:
            Limit constraints are validated by `BacktestJobListQuery`.
        Raises:
            ValueError: If query shape is invalid.
        Side Effects:
            Reads one jobs page from storage.
        """
        if current_user is None:  # type: ignore[truthy-bool]
            raise ValueError("ListBacktestJobsUseCase.execute requires current_user")

        query = BacktestJobListQuery(
            user_id=current_user.user_id,
            state=state,
            limit=limit,
            cursor=cursor,
        )
        return self._job_repository.list_for_user(query=query)


class CancelBacktestJobUseCase:
    """
    Request owner job cancel and return updated idempotent status payload.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - apps/api/routes/backtest_jobs.py
      - apps/api/dto/backtest_jobs.py
    """

    def __init__(
        self,
        *,
        job_repository: BacktestJobRepository,
        now_provider: NowProvider | None = None,
    ) -> None:
        """
        Initialize cancel use-case with repository and optional deterministic clock.

        Args:
            job_repository: Jobs repository port.
            now_provider: Optional UTC clock provider.
        Returns:
            None.
        Assumptions:
            Repository cancel operation is idempotent for terminal job states.
        Raises:
            ValueError: If dependency is missing.
        Side Effects:
            None.
        """
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("CancelBacktestJobUseCase requires job_repository")
        self._job_repository = job_repository
        self._now = now_provider or _utc_now

    def execute(self, *, job_id: UUID, current_user: CurrentUser) -> BacktestJob:
        """
        Request cancel for owner job and return current deterministic status snapshot.

        Args:
            job_id: Requested job identifier.
            current_user: Authenticated owner identity.
        Returns:
            BacktestJob: Updated (or already-terminal) owner job snapshot.
        Assumptions:
            Existing foreign job must map to `403`, missing job to `404`.
        Raises:
            RoehubError: Canonical `forbidden` or `not_found` for owner checks.
        Side Effects:
            Writes cancel marker/state transition for owner active jobs.
        """
        _require_owner_job(
            job_repository=self._job_repository,
            job_id=job_id,
            current_user=current_user,
        )
        cancelled = self._job_repository.cancel(
            job_id=job_id,
            user_id=current_user.user_id,
            cancel_requested_at=self._now(),
        )
        if cancelled is None:
            raise backtest_job_not_found(job_id=job_id)
        return cancelled



def _require_owner_job(
    *,
    job_repository: BacktestJobRepository,
    job_id: UUID,
    current_user: CurrentUser,
) -> BacktestJob:
    """
    Read job by id and enforce explicit owner policy (`403` foreign, `404` missing).

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/roadmap/milestone-5-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - apps/api/routes/backtest_jobs.py

    Args:
        job_repository: Jobs repository port.
        job_id: Requested job identifier.
        current_user: Authenticated owner identity.
    Returns:
        BacktestJob: Owner job snapshot.
    Assumptions:
        Access policy intentionally reads without owner SQL filters first.
    Raises:
        RoehubError: Canonical `not_found` for missing row and `forbidden` for foreign owner.
    Side Effects:
        Reads one job row from storage.
    """
    job = job_repository.get(job_id=job_id)
    if job is None:
        raise backtest_job_not_found(job_id=job_id)
    if job.user_id != current_user.user_id:
        raise backtest_job_forbidden(job_id=job_id)
    return job



def _template_from_snapshot(
    *,
    strategy_id: UUID,
    snapshot: BacktestStrategySnapshot | None,
    current_user: CurrentUser,
) -> RunBacktestTemplate:
    """
    Convert saved strategy snapshot into run template with deterministic ownership checks.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-api-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py

    Args:
        strategy_id: Requested strategy identifier.
        snapshot: Loaded saved strategy snapshot.
        current_user: Authenticated owner identity.
    Returns:
        RunBacktestTemplate: Template resolved from saved strategy snapshot.
    Assumptions:
        Missing and deleted snapshots are mapped to one `not_found` contract.
    Raises:
        BacktestNotFoundError: If snapshot is missing or deleted.
        BacktestForbiddenError: If snapshot owner mismatches current user.
    Side Effects:
        None.
    """
    if snapshot is None or snapshot.is_deleted:
        raise BacktestNotFoundError(strategy_id=strategy_id)
    if snapshot.user_id != current_user.user_id:
        raise BacktestForbiddenError(strategy_id=strategy_id)

    return RunBacktestTemplate(
        instrument_id=snapshot.instrument_id,
        timeframe=snapshot.timeframe,
        indicator_grids=snapshot.indicator_grids,
        indicator_selections=snapshot.indicator_selections,
        signal_grids=snapshot.signal_grids,
        risk_grid=snapshot.risk_grid,
        direction_mode=snapshot.direction_mode,
        sizing_mode=snapshot.sizing_mode,
        risk_params=snapshot.risk_params,
        execution_params=snapshot.execution_params,
    )



def _apply_saved_overrides(
    *,
    base_template: RunBacktestTemplate,
    overrides,
) -> RunBacktestTemplate:
    """
    Apply saved-mode overrides over base snapshot template deterministically.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-api-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py

    Args:
        base_template: Template resolved from saved strategy snapshot.
        overrides: Optional saved-mode overrides DTO.
    Returns:
        RunBacktestTemplate: Effective template with overrides applied.
    Assumptions:
        `overrides` follows `RunBacktestSavedOverrides` constructor invariants.
    Raises:
        ValueError: If merged values violate template invariants.
    Side Effects:
        None.
    """
    if overrides is None:
        return base_template

    direction_mode = (
        overrides.direction_mode
        if overrides.direction_mode is not None
        else base_template.direction_mode
    )
    sizing_mode = (
        overrides.sizing_mode
        if overrides.sizing_mode is not None
        else base_template.sizing_mode
    )
    signal_grids = _merge_signal_grids(
        base=base_template.signal_grids or {},
        updates=overrides.signal_grids or {},
    )
    risk_params = _merge_scalar_mappings(
        base=base_template.risk_params or {},
        updates=overrides.risk_params or {},
    )
    execution_params = _merge_scalar_mappings(
        base=base_template.execution_params or {},
        updates=overrides.execution_params or {},
    )
    risk_grid = overrides.risk_grid if overrides.risk_grid is not None else base_template.risk_grid

    return RunBacktestTemplate(
        instrument_id=base_template.instrument_id,
        timeframe=base_template.timeframe,
        indicator_grids=base_template.indicator_grids,
        indicator_selections=base_template.indicator_selections,
        signal_grids=signal_grids,
        risk_grid=risk_grid,
        direction_mode=direction_mode,
        sizing_mode=sizing_mode,
        risk_params=risk_params,
        execution_params=execution_params,
    )



def _resolve_positive_override(
    *,
    value: int | None,
    default: int,
    field_path: str,
) -> int:
    """
    Resolve optional positive integer override against runtime default.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtest_jobs.py

    Args:
        value: Optional override value.
        default: Runtime default value.
        field_path: Request path used for deterministic validation details.
    Returns:
        int: Effective positive integer value.
    Assumptions:
        Runtime defaults are validated before use-case invocation.
    Raises:
        RoehubError: Canonical `validation_error` when override is non-positive.
    Side Effects:
        None.
    """
    if value is None:
        return default
    if value <= 0:
        raise validation_error(
            message="Backtest jobs request override values must be > 0",
            errors=(
                {
                    "path": field_path,
                    "code": "greater_than",
                    "message": "value must be > 0",
                },
            ),
        )
    return value



def _resolve_number(
    *,
    values: Mapping[str, Any],
    primary_key: str,
    secondary_key: str,
    default: float,
) -> float:
    """
    Resolve numeric override by primary/secondary key with deterministic fallback.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/backtest/backtest-jobs-api-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/dto/backtests.py

    Args:
        values: Scalar mapping with optional numeric overrides.
        primary_key: Preferred key name.
        secondary_key: Optional backward-compatible key alias.
        default: Runtime default numeric value.
    Returns:
        float: Resolved numeric value.
    Assumptions:
        Boolean values are rejected despite inheriting from `int`.
    Raises:
        ValueError: If override value is not numeric.
    Side Effects:
        None.
    """
    candidate: Any = None
    if primary_key in values:
        candidate = values[primary_key]
    elif secondary_key and secondary_key in values:
        candidate = values[secondary_key]
    if candidate is None:
        return float(default)
    if isinstance(candidate, bool) or not isinstance(candidate, int | float):
        raise ValueError(f"execution field '{primary_key}' must be numeric")
    return float(candidate)



def _merge_scalar_mappings(
    *,
    base: Mapping[str, int | float | str | bool | None],
    updates: Mapping[str, int | float | str | bool | None],
) -> Mapping[str, int | float | str | bool | None]:
    """
    Merge scalar mappings deterministically with update precedence and sorted keys.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py

    Args:
        base: Base scalar mapping.
        updates: Override scalar mapping.
    Returns:
        Mapping[str, int | float | str | bool | None]: Immutable merged mapping.
    Assumptions:
        Scalar keys are non-empty after normalization.
    Raises:
        ValueError: If one key is blank.
    Side Effects:
        None.
    """
    merged: dict[str, int | float | str | bool | None] = {}
    for raw_key in sorted(base.keys(), key=lambda key: str(key).strip()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("saved-mode scalar override key must be non-empty")
        merged[key] = base[raw_key]
    for raw_key in sorted(updates.keys(), key=lambda key: str(key).strip()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("saved-mode scalar override key must be non-empty")
        merged[key] = updates[raw_key]
    return MappingProxyType(merged)



def _merge_signal_grids(
    *,
    base: Mapping[str, Mapping[str, Any]],
    updates: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    """
    Merge nested signal-grid mappings deterministically by indicator and parameter keys.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py

    Args:
        base: Base signal-grid mapping.
        updates: Override signal-grid mapping.
    Returns:
        Mapping[str, Mapping[str, Any]]: Immutable merged nested mapping.
    Assumptions:
        Mapping values are validated upstream by DTO constructors.
    Raises:
        ValueError: If one indicator id or parameter key is blank.
    Side Effects:
        None.
    """
    merged: dict[str, dict[str, Any]] = {}

    for raw_indicator_id in sorted(base.keys(), key=lambda key: str(key).strip()):
        indicator_id = str(raw_indicator_id).strip()
        if not indicator_id:
            raise ValueError("saved-mode signal override indicator_id must be non-empty")
        merged[indicator_id] = {}
        for raw_param_name in sorted(
            base[raw_indicator_id].keys(),
            key=lambda key: str(key).strip(),
        ):
            param_name = str(raw_param_name).strip()
            if not param_name:
                raise ValueError("saved-mode signal override param_name must be non-empty")
            merged[indicator_id][param_name] = base[raw_indicator_id][raw_param_name]

    for raw_indicator_id in sorted(updates.keys(), key=lambda key: str(key).strip()):
        indicator_id = str(raw_indicator_id).strip()
        if not indicator_id:
            raise ValueError("saved-mode signal override indicator_id must be non-empty")
        if indicator_id not in merged:
            merged[indicator_id] = {}
        for raw_param_name in sorted(
            updates[raw_indicator_id].keys(),
            key=lambda key: str(key).strip(),
        ):
            param_name = str(raw_param_name).strip()
            if not param_name:
                raise ValueError("saved-mode signal override param_name must be non-empty")
            merged[indicator_id][param_name] = updates[raw_indicator_id][raw_param_name]

    immutable_nested: dict[str, Mapping[str, Any]] = {}
    for indicator_id in sorted(merged.keys()):
        immutable_nested[indicator_id] = MappingProxyType(dict(merged[indicator_id]))
    return MappingProxyType(immutable_nested)



def _normalize_fee_defaults(*, values: Mapping[int, float]) -> Mapping[int, float]:
    """
    Normalize fee defaults mapping to immutable deterministic `market_id -> fee_pct` payload.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-api-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py

    Args:
        values: Runtime fee defaults mapping.
    Returns:
        Mapping[int, float]: Immutable normalized mapping.
    Assumptions:
        Mapping contains at least one market id.
    Raises:
        ValueError: If one key/value is invalid.
    Side Effects:
        None.
    """
    normalized: dict[int, float] = {}
    for raw_market_id in sorted(values.keys()):
        market_id = int(raw_market_id)
        fee_pct = float(values[raw_market_id])
        if market_id <= 0:
            raise ValueError("fee_pct_default_by_market_id keys must be > 0")
        if fee_pct < 0.0:
            raise ValueError("fee_pct_default_by_market_id values must be >= 0")
        normalized[market_id] = fee_pct
    if len(normalized) == 0:
        raise ValueError("fee_pct_default_by_market_id must be non-empty")
    return MappingProxyType(normalized)



def _normalize_json_mapping(*, values: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize mapping payload into deterministic JSON-compatible dictionary.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py

    Args:
        values: Raw mapping payload.
    Returns:
        dict[str, Any]: Normalized key-sorted JSON-compatible payload.
    Assumptions:
        Unknown scalar objects are stringified for deterministic persistence.
    Raises:
        ValueError: If normalized payload cannot be encoded as JSON object.
    Side Effects:
        None.
    """
    normalized = _normalize_json_value(value=dict(values))
    if not isinstance(normalized, Mapping):
        raise ValueError("request payload must be JSON object")
    try:
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError as error:
        raise ValueError("request payload must be JSON-serializable") from error
    return dict(normalized)



def _normalize_json_value(*, value: Any) -> Any:
    """
    Normalize arbitrary value into deterministic JSON-compatible structure.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py

    Args:
        value: Arbitrary JSON-like node.
    Returns:
        Any: Deterministic mapping/list/scalar node.
    Assumptions:
        Datetime and UUID values are encoded as ISO/str literals.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda item: str(item)):
            normalized_mapping[str(raw_key)] = _normalize_json_value(value=value[raw_key])
        return normalized_mapping

    if isinstance(value, list | tuple):
        return [_normalize_json_value(value=item) for item in value]

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)



def _build_sha256_from_payload(*, payload: Mapping[str, Any]) -> str:
    """
    Build deterministic SHA-256 hash from canonical JSON representation.

    Docs:
      - docs/architecture/backtest/backtest-jobs-api-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py

    Args:
        payload: JSON-compatible mapping payload.
    Returns:
        str: Lowercase SHA-256 hex hash string.
    Assumptions:
        Canonical JSON uses sorted keys and compact separators.
    Raises:
        TypeError: If payload contains unsupported non-JSON values.
    Side Effects:
        None.
    """
    import hashlib

    canonical_json = json.dumps(
        _normalize_json_value(value=payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()



def _utc_now() -> datetime:
    """
    Return timezone-aware UTC timestamp for deterministic job lifecycle writes.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - apps/api/routes/backtest_jobs.py

    Args:
        None.
    Returns:
        datetime: Current UTC datetime.
    Assumptions:
        Caller stores timestamps in UTC across API/worker/storage layers.
    Raises:
        None.
    Side Effects:
        None.
    """
    return datetime.now(timezone.utc)


__all__ = [
    "BacktestJobTopReadResult",
    "CancelBacktestJobUseCase",
    "CreateBacktestJobCommand",
    "CreateBacktestJobUseCase",
    "GetBacktestJobStatusUseCase",
    "GetBacktestJobTopUseCase",
    "ListBacktestJobsUseCase",
]
