from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Protocol, cast
from uuid import UUID, uuid4

from trading.contexts.backtest.application.dto import (
    BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1,
    BacktestReportV1,
    RunBacktestRequest,
    RunBacktestResponse,
)
from trading.contexts.backtest.application.ports import BacktestJobRepository, CurrentUser
from trading.contexts.backtest.application.services import (
    BacktestJobTopVariantCandidateV1,
    build_finalized_snapshot_rows,
    validate_execution_profile_mode_v2,
)
from trading.contexts.backtest.application.services.run_control_v1 import BacktestRunControlV1
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobMode,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, Timeframe

from .backtest_jobs_api_v1 import (
    CreateBacktestJobCommand,
    _build_request_hash_from_request_json,
    _build_sha256_from_payload,
    _normalize_json_mapping,
)

NowProvider = Callable[[], datetime]
RunIdFactory = Callable[[], UUID]
_SYNC_INLINE_LOCKED_BY = "sync-inline"
_AUTO_FALLBACK_ELIGIBLE_ERRORS = frozenset(
    {
        "max_compute_bytes_total_exceeded",
        "max_variants_per_compute_exceeded",
        "background_auto_required",
    }
)
_SYNC_INLINE_REDESIGNED_EXECUTION_PROFILE_MODE = "hybrid_conservative"


class BacktestRunsApiUseCase(Protocol):
    """
    Structural contract for `/backtests` sync execution and lazy report API orchestration.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - apps/api/routes/backtests.py
    """

    def execute(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        request_payload: Mapping[str, Any] | None = None,
        run_control: BacktestRunControlV1 | None = None,
    ) -> RunBacktestResponse:
        """
        Execute one sync backtest flow with optional canonical request payload snapshot.

        Args:
            request: Parsed application request DTO.
            current_user: Authenticated owner identity.
            request_payload: Optional strict API payload snapshot for persisted-run orchestrators.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            RunBacktestResponse: Deterministic sync response DTO.
        Assumptions:
            Implementations may ignore `request_payload` when persistence is not required.
        Raises:
            Exception: Domain/application errors are implementation-specific.
        Side Effects:
            May execute sync compute and optionally persist run metadata.
        """
        ...

    def build_variant_report(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        variant_payload: Any,
        include_trades: bool = False,
        run_control: BacktestRunControlV1 | None = None,
    ) -> BacktestReportV1:
        """
        Build one lazy single-variant report for `/backtests/variant-report`.

        Args:
            request: Parsed application request DTO.
            current_user: Authenticated owner identity.
            variant_payload: Explicit variant payload selected for on-demand detail generation.
            include_trades: Include-trades flag for report generation.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            BacktestReportV1: Deterministic report payload.
        Assumptions:
            Report generation reuses the same runtime semantics as sync execution.
        Raises:
            Exception: Domain/application errors are implementation-specific.
        Side Effects:
            Executes report generation logic and may perform compute IO.
        """
        ...


class BacktestRunPreflightUseCase(Protocol):
    """
    Structural contract for deterministic staged-budget preflight without execution.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - apps/api/wiring/modules/backtest.py
    """

    def preflight(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        run_control: BacktestRunControlV1 | None = None,
    ) -> None:
        """
        Validate the request against canonical staged guard budgets.

        Args:
            request: Parsed application request DTO.
            current_user: Authenticated owner identity.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            None.
        Assumptions:
            Preflight uses the same runtime contract as the corresponding launch path.
        Raises:
            Exception: Domain/application errors are implementation-specific.
        Side Effects:
            May read runtime inputs required for deterministic guard evaluation.
        """
        ...


class BacktestBackgroundJobCreateUseCase(Protocol):
    """
    Structural contract for creating queued persisted runs for background execution.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - apps/api/wiring/modules/backtest.py
    """

    def execute(
        self,
        *,
        command: CreateBacktestJobCommand,
        current_user: CurrentUser,
    ) -> BacktestJob:
        """
        Persist one queued background run snapshot.

        Args:
            command: Canonical queued background-run create command.
            current_user: Authenticated owner identity.
        Returns:
            BacktestJob: Persisted queued run snapshot.
        Assumptions:
            Persistence remains deterministic and summary-only.
        Raises:
            Exception: Domain/application errors are implementation-specific.
        Side Effects:
            Writes one row into the unified persisted-run storage family.
        """
        ...


class LaunchBacktestRunWithAutoFallbackUseCase:
    """
    Launch `POST /backtests` with deterministic `sync_inline -> background_auto` fallback.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/routes/backtests.py
    """

    def __init__(
        self,
        *,
        sync_inline_use_case: BacktestRunsApiUseCase,
        background_preflight_use_case: BacktestRunPreflightUseCase,
        background_create_use_case: BacktestBackgroundJobCreateUseCase,
        engine_version: str,
    ) -> None:
        """
        Initialize deterministic `/backtests` launch orchestration dependencies.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
          - apps/api/wiring/modules/backtest.py
          - apps/api/routes/backtests.py
        Args:
            sync_inline_use_case: Persisted sync-inline launch path using sync half-budgets.
            background_preflight_use_case: Full-budget guard preflight without persistence.
            background_create_use_case: Queued background-run creator over unified storage.
            engine_version: Stable engine/runtime literal exposed in API launch responses.
        Returns:
            None.
        Assumptions:
            Background preflight and background create share the same canonical request contract.
        Raises:
            ValueError: If dependencies or engine version are invalid.
        Side Effects:
            None.
        """
        if sync_inline_use_case is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "LaunchBacktestRunWithAutoFallbackUseCase requires sync_inline_use_case"
            )
        if background_preflight_use_case is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "LaunchBacktestRunWithAutoFallbackUseCase requires background_preflight_use_case"
            )
        if background_create_use_case is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "LaunchBacktestRunWithAutoFallbackUseCase requires background_create_use_case"
            )
        normalized_engine_version = engine_version.strip()
        if not normalized_engine_version:
            raise ValueError(
                "LaunchBacktestRunWithAutoFallbackUseCase requires engine_version"
            )

        self._sync_inline_use_case = sync_inline_use_case
        self._background_preflight_use_case = background_preflight_use_case
        self._background_create_use_case = background_create_use_case
        self._engine_version = normalized_engine_version

    def execute(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        request_payload: Mapping[str, Any] | None = None,
        run_control: BacktestRunControlV1 | None = None,
    ) -> RunBacktestResponse:
        """
        Launch sync-inline first, then fallback deterministically to `background_auto`.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
          - apps/api/routes/backtests.py
        Args:
            request: Parsed application request DTO.
            current_user: Authenticated owner identity.
            request_payload: Strict API payload snapshot used for persistence-compatible flows.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            RunBacktestResponse: Sync-inline success or explicit queued `background_auto` launch.
        Assumptions:
            Only canonical staged guard overflow errors are eligible for fallback.
        Raises:
            RoehubError: Canonical validation/not-found/forbidden/conflict errors.
            ValueError: If required inputs are missing.
        Side Effects:
            May execute sync compute, run full-budget preflight, and persist one queued run row.
        """
        if request is None:  # type: ignore[truthy-bool]
            raise ValueError("LaunchBacktestRunWithAutoFallbackUseCase.execute requires request")
        if current_user is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "LaunchBacktestRunWithAutoFallbackUseCase.execute requires current_user"
            )
        if request_payload is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "LaunchBacktestRunWithAutoFallbackUseCase.execute requires request_payload"
            )

        background_execution_profile_mode: str | None = None
        try:
            return self._sync_inline_use_case.execute(
                request=request,
                current_user=current_user,
                request_payload=request_payload,
                run_control=run_control,
            )
        except RoehubError as error:
            if not _is_auto_fallback_eligible_error(error=error):
                raise
            background_execution_profile_mode = _execution_profile_mode_from_error(
                error=error
            )

        self._background_preflight_use_case.preflight(
            request=request,
            current_user=current_user,
            run_control=run_control,
        )
        created_run = self._background_create_use_case.execute(
            command=CreateBacktestJobCommand(
                run_request=request,
                request_payload=request_payload,
                execution_mode="background_auto",
                execution_profile_mode=background_execution_profile_mode,
            ),
            current_user=current_user,
        )
        return _build_background_auto_launch_response(
            request=request,
            created_run=created_run,
            engine_version=self._engine_version,
        )

    def build_variant_report(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        variant_payload: Any,
        include_trades: bool = False,
        run_control: BacktestRunControlV1 | None = None,
    ) -> BacktestReportV1:
        """
        Delegate lazy variant-report generation to the sync-inline report path.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/backtest-runs-history-v2.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - apps/api/routes/backtests.py
        Args:
            request: Parsed application request DTO.
            current_user: Authenticated owner identity.
            variant_payload: Explicit variant payload selected for lazy detail/report fetch.
            include_trades: Include-trades flag for report generation.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            BacktestReportV1: Deterministic lazy report payload.
        Assumptions:
            Auto-fallback launch semantics do not change lazy detail generation contract.
        Raises:
            RoehubError: Propagates canonical errors from the delegated sync report path.
        Side Effects:
            Delegates to the existing report builder path without touching persisted storage.
        """
        return self._sync_inline_use_case.build_variant_report(
            request=request,
            current_user=current_user,
            variant_payload=variant_payload,
            include_trades=include_trades,
            run_control=run_control,
        )


class CreateAndRunBacktestSyncInlineUseCase:
    """
    Create-and-execute persisted sync-inline Backtest flow over the unified jobs storage family.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - apps/api/routes/backtests.py
    """

    def __init__(
        self,
        *,
        run_use_case: BacktestRunsApiUseCase,
        job_repository: BacktestJobRepository,
        backtest_runtime_config_hash: str,
        engine_version: str,
        now_provider: NowProvider | None = None,
        run_id_factory: RunIdFactory | None = None,
    ) -> None:
        """
        Initialize sync-inline persisted-run orchestrator dependencies.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - apps/api/wiring/modules/backtest.py
        Args:
            run_use_case: Existing sync backtest use-case performing internal preflight + execute.
            job_repository: Persisted-run repository over the unified jobs storage family.
            backtest_runtime_config_hash: Canonical result-affecting runtime config hash.
            engine_version: Stable engine/runtime literal exposed in sync responses.
            now_provider: Optional UTC clock provider for deterministic tests.
            run_id_factory: Optional persisted run id factory for deterministic tests.
        Returns:
            None.
        Assumptions:
            Runtime config hash and engine-version literal are validated on startup wiring.
        Raises:
            ValueError: If one dependency or invariant is invalid.
        Side Effects:
            None.
        """
        if run_use_case is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateAndRunBacktestSyncInlineUseCase requires run_use_case")
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateAndRunBacktestSyncInlineUseCase requires job_repository")
        normalized_engine_version = engine_version.strip()
        if not normalized_engine_version:
            raise ValueError("CreateAndRunBacktestSyncInlineUseCase requires engine_version")
        normalized_runtime_hash = backtest_runtime_config_hash.strip().lower()
        if len(normalized_runtime_hash) != 64:
            raise ValueError(
                "CreateAndRunBacktestSyncInlineUseCase requires 64-char "
                "backtest_runtime_config_hash"
            )

        self._run_use_case = run_use_case
        self._job_repository = job_repository
        self._backtest_runtime_config_hash = normalized_runtime_hash
        self._engine_version = normalized_engine_version
        self._now = now_provider or _utc_now
        self._run_id_factory = run_id_factory or uuid4

    def execute(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        request_payload: Mapping[str, Any] | None = None,
        run_control: BacktestRunControlV1 | None = None,
    ) -> RunBacktestResponse:
        """
        Execute sync run inline, then persist final run row and summary-only top rows atomically.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
          - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
          - apps/api/routes/backtests.py
        Args:
            request: Parsed application request DTO.
            current_user: Authenticated owner identity.
            request_payload:
                Strict API payload snapshot used to persist canonical `request_json` after
                storage-only metadata is stripped. The sync-inline wrapper also injects the
                server-owned internal `execution_profile_mode=hybrid_conservative` marker into
                the runtime payload so `POST /backtests` runs through the redesigned
                prefilter-first sync engine without changing the public transport contract.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            RunBacktestResponse: Sync response enriched with persisted run identity metadata.
        Assumptions:
            Internal preflight remains delegated to the existing sync `RunBacktestUseCase`, while
            the effective sync execution profile stays internal-only and excluded from request-hash
            semantics.
        Raises:
            BacktestValidationError: If persisted metadata cannot be built deterministically.
            RoehubError: Propagates canonical validation/not-found/forbidden failures
                from the inner sync use-case.
        Side Effects:
            Executes sync backtest compute and writes one terminal run row plus summary-only top
            rows into the unified Postgres storage family.
        """
        if request is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateAndRunBacktestSyncInlineUseCase.execute requires request")
        if current_user is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "CreateAndRunBacktestSyncInlineUseCase.execute requires current_user"
            )
        if request_payload is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "CreateAndRunBacktestSyncInlineUseCase.execute requires request_payload"
            )

        created_at = self._now()
        effective_request_payload = _with_sync_inline_redesigned_engine_request_payload(
            request_payload=request_payload
        )
        base_response = self._run_use_case.execute(
            request=request,
            current_user=current_user,
            request_payload=effective_request_payload,
            run_control=run_control,
        )
        finished_at = self._now()

        artifact_pin = _artifact_pin_from_response(response=base_response)
        request_json = _build_request_json_payload(
            request=request,
            request_payload=effective_request_payload,
            response=base_response,
        )
        engine_params_hash = _build_engine_params_hash(response=base_response)
        run = _build_terminal_sync_inline_run(
            run_id=self._run_id_factory(),
            created_at=created_at,
            finished_at=finished_at,
            current_user=current_user,
            request=request,
            response=base_response,
            request_json=request_json,
            artifact_pin=artifact_pin,
            backtest_runtime_config_hash=self._backtest_runtime_config_hash,
            engine_params_hash=engine_params_hash,
        )
        persisted_rows = _build_persisted_top_rows(
            job_id=run.job_id,
            persisted_at=finished_at,
            response=base_response,
        )
        persisted_run = self._job_repository.create_with_top_variants(
            job=run,
            top_variants=persisted_rows,
        )

        return replace(
            base_response,
            run_id=persisted_run.job_id,
            state=persisted_run.state,
            execution_mode=persisted_run.execution_mode,
            engine_version=self._engine_version,
            artifact_slot=artifact_pin.artifact_slot,
            artifact_slot_generation=artifact_pin.artifact_slot_generation,
            artifact_asof_date=artifact_pin.artifact_asof_date,
            artifact_manifest_hash=artifact_pin.artifact_manifest_hash,
            spec_hash=persisted_run.spec_hash,
            spec_payload_json=persisted_run.spec_payload_json,
            engine_params_hash=persisted_run.engine_params_hash,
        )

    def build_variant_report(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        variant_payload: Any,
        include_trades: bool = False,
        run_control: BacktestRunControlV1 | None = None,
    ) -> BacktestReportV1:
        """
        Delegate lazy single-variant report generation to the existing sync use-case.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - apps/api/routes/backtests.py
        Args:
            request: Parsed application request DTO.
            current_user: Authenticated owner identity.
            variant_payload: Explicit variant payload selected for lazy detail/report fetch.
            include_trades: Include-trades flag for report generation.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            BacktestReportV1: Deterministic lazy report payload.
        Assumptions:
            Persisted sync-inline cutover does not change lazy single-variant detail semantics.
        Raises:
            RoehubError: Propagates canonical errors from the inner sync use-case.
        Side Effects:
            Delegates to the existing report builder path without touching persisted storage.
        """
        return self._run_use_case.build_variant_report(
            request=request,
            current_user=current_user,
            variant_payload=variant_payload,
            include_trades=include_trades,
            run_control=run_control,
        )


def _is_auto_fallback_eligible_error(*, error: RoehubError) -> bool:
    """
    Classify deterministic sync launch errors eligible for `background_auto` routing.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - apps/api/routes/backtests.py
    Args:
        error: Canonical API/domain error raised by the sync-inline launch path.
    Returns:
        bool: `True` only for structured guard-overflow or sync-launch-budget routing errors.
    Assumptions:
        Structured launch details use the stable `details.error` literal contract.
    Raises:
        None.
    Side Effects:
        None.
    """
    if error.code != "validation_error" or error.details is None:
        return False
    detail_error = error.details.get("error")
    if not isinstance(detail_error, str):
        return False
    return detail_error in _AUTO_FALLBACK_ELIGIBLE_ERRORS


def _execution_profile_mode_from_error(*, error: RoehubError) -> str | None:
    """
    Read additive effective execution-profile hint from one auto-fallback-eligible error.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
    Args:
        error: Canonical planner/launch error considered for `background_auto` routing.
    Returns:
        str | None: Normalized execution-profile literal, or `None` when the error carries no
            additive profile hint.
    Assumptions:
        Heavy-but-valid sync launch rejections and staged guard overflows may include
        `details.execution_profile_mode` to keep persisted progress semantics truthful.
    Raises:
        None.
    Side Effects:
        None.
    """
    if error.details is None:
        return None
    raw_mode = error.details.get("execution_profile_mode")
    if not isinstance(raw_mode, str) or not raw_mode.strip():
        return None
    try:
        return validate_execution_profile_mode_v2(value=raw_mode)
    except ValueError:
        return None


def _with_sync_inline_redesigned_engine_request_payload(
    *,
    request_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """
    Force persisted sync-inline launches onto the redesigned prefilter-first engine profile.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-engine-vnext.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - apps/api/routes/backtests.py
    Args:
        request_payload: Strict API request snapshot captured before sync launch orchestration.
    Returns:
        Mapping[str, Any]: Canonical payload copy with internal
            `execution_profile_mode=hybrid_conservative`.
    Assumptions:
        `POST /backtests` keeps the same public request/response shape; the Milestone F1 sync
        cutover is represented only by additive internal metadata that stays excluded from
        canonical request-hash semantics.
    Raises:
        ValueError: Propagated if the configured internal execution-profile literal is invalid.
    Side Effects:
        None.
    """
    normalized_payload = dict(_normalize_json_mapping(values=request_payload))
    normalized_payload["execution_profile_mode"] = validate_execution_profile_mode_v2(
        value=_SYNC_INLINE_REDESIGNED_EXECUTION_PROFILE_MODE
    )
    return normalized_payload


def _build_background_auto_launch_response(
    *,
    request: RunBacktestRequest,
    created_run: BacktestJob,
    engine_version: str,
) -> RunBacktestResponse:
    """
    Convert queued `background_auto` run snapshot into `/backtests` launch response DTO.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/dto/backtests.py
    Args:
        request: Original parsed application request DTO.
        created_run: Persisted queued run snapshot created for fallback.
        engine_version: Stable engine/runtime literal exposed by API responses.
    Returns:
        RunBacktestResponse: Explicit queued background launch response with empty variants.
    Assumptions:
        Auto-fallback launch is summary-only and does not compute top variants inline.
    Raises:
        BacktestValidationError: If persisted queued metadata is incomplete.
    Side Effects:
        None.
    """
    artifact_pin = _require_job_artifact_pin(created_run=created_run)
    if created_run.execution_mode != "background_auto":
        raise BacktestValidationError(
            "background_auto launch response requires background_auto execution_mode"
        )
    return RunBacktestResponse(
        mode=request.mode,
        instrument_id=InstrumentId(
            market_id=MarketId(_require_job_market_id(created_run=created_run)),
            symbol=Symbol(_require_job_symbol(created_run=created_run)),
        ),
        timeframe=Timeframe(_require_job_timeframe(created_run=created_run)),
        strategy_id=request.strategy_id,
        top_k=_require_positive_int_request_json(
            request_json=created_run.request_json,
            field_name="top_k",
        ),
        preselect=_require_positive_int_request_json(
            request_json=created_run.request_json,
            field_name="preselect",
        ),
        variants=tuple(),
        total_indicator_compute_calls=0,
        run_id=created_run.job_id,
        state=created_run.state,
        execution_mode=created_run.execution_mode,
        execution_profile_mode=_require_persisted_execution_profile_mode(
            created_run=created_run
        ),
        engine_version=engine_version,
        artifact_slot=artifact_pin.artifact_slot,
        artifact_slot_generation=artifact_pin.artifact_slot_generation,
        artifact_asof_date=artifact_pin.artifact_asof_date,
        artifact_manifest_hash=artifact_pin.artifact_manifest_hash,
        spec_hash=created_run.spec_hash,
        spec_payload_json=created_run.spec_payload_json,
        engine_params_hash=created_run.engine_params_hash,
    )


def _require_job_artifact_pin(*, created_run: BacktestJob) -> BacktestJobArtifactPin:
    """
    Require pinned artifact metadata on one queued or terminal persisted run snapshot.

    Args:
        created_run: Persisted run snapshot.
    Returns:
        BacktestJobArtifactPin: Immutable artifact pin metadata.
    Assumptions:
        All persisted runs for `/backtests` carry slot-pinning identity.
    Raises:
        BacktestValidationError: If pin metadata is absent.
    Side Effects:
        None.
    """
    if created_run.artifact_pin is None:
        raise BacktestValidationError("persisted run requires artifact pin metadata")
    return created_run.artifact_pin


def _require_positive_int_request_json(
    *,
    request_json: Mapping[str, Any],
    field_name: str,
) -> int:
    """
    Read one positive integer field from persisted canonical `request_json`.

    Args:
        request_json: Persisted canonical request payload.
        field_name: Required positive integer field name.
    Returns:
        int: Positive integer field value.
    Assumptions:
        Background launch response reuses defaults already materialized into `request_json`.
    Raises:
        BacktestValidationError: If field is missing or not a positive integer.
    Side Effects:
        None.
    """
    raw_value = request_json.get(field_name)
    if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
        raise BacktestValidationError(
            f"persisted run request_json requires positive integer field {field_name!r}"
        )
    return raw_value


def _require_persisted_execution_profile_mode(
    *,
    created_run: BacktestJob,
) -> str:
    """
    Read one persisted effective execution-profile mode from additive run metadata.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
      - apps/api/dto/backtests.py
    Args:
        created_run: Persisted run snapshot from unified jobs storage.
    Returns:
        str: Normalized execution-profile mode literal.
    Assumptions:
        New rows store read-model profile metadata in additive fields first; queued launch rows
        may still rely on the additive hint field, while historical rows may require explicit
        `request_json.execution_profile_mode` fallback.
    Raises:
        BacktestValidationError: If the field is missing or invalid.
    Side Effects:
        None.
    """
    raw_mode = created_run.effective_execution_profile_mode
    if raw_mode is None:
        raw_mode = created_run.execution_profile_mode_hint
    if raw_mode is None:
        legacy_mode = created_run.request_json.get("execution_profile_mode")
        raw_mode = legacy_mode if isinstance(legacy_mode, str) else None
    if not isinstance(raw_mode, str) or not raw_mode.strip():
        raise BacktestValidationError(
            "persisted run metadata requires additive execution-profile fields"
        )
    try:
        return validate_execution_profile_mode_v2(value=raw_mode)
    except ValueError as error:
        raise BacktestValidationError(
            "persisted run metadata requires valid additive execution-profile fields"
        ) from error


def _require_job_market_id(*, created_run: BacktestJob) -> int:
    """
    Require canonical market id on persisted run metadata.

    Args:
        created_run: Persisted run snapshot.
    Returns:
        int: Positive market id.
    Assumptions:
        Unified persisted-run metadata fills market id for all launch branches.
    Raises:
        BacktestValidationError: If market id is absent or invalid.
    Side Effects:
        None.
    """
    market_id = created_run.market_id
    if market_id is None or market_id <= 0:
        raise BacktestValidationError("persisted run requires positive market_id metadata")
    return market_id


def _require_job_symbol(*, created_run: BacktestJob) -> str:
    """
    Require canonical symbol on persisted run metadata.

    Args:
        created_run: Persisted run snapshot.
    Returns:
        str: Non-empty symbol literal.
    Assumptions:
        Unified persisted-run metadata fills symbol for all launch branches.
    Raises:
        BacktestValidationError: If symbol is absent or blank.
    Side Effects:
        None.
    """
    symbol = created_run.symbol
    if symbol is None or not symbol.strip():
        raise BacktestValidationError("persisted run requires symbol metadata")
    return symbol


def _require_job_timeframe(*, created_run: BacktestJob) -> str:
    """
    Require canonical timeframe on persisted run metadata.

    Args:
        created_run: Persisted run snapshot.
    Returns:
        str: Non-empty timeframe literal.
    Assumptions:
        Unified persisted-run metadata fills timeframe for all launch branches.
    Raises:
        BacktestValidationError: If timeframe is absent or blank.
    Side Effects:
        None.
    """
    timeframe = created_run.timeframe
    if timeframe is None or not timeframe.strip():
        raise BacktestValidationError("persisted run requires timeframe metadata")
    return timeframe


def _artifact_pin_from_response(*, response: RunBacktestResponse) -> BacktestJobArtifactPin:
    """
    Convert sync run artifact metadata into the immutable persisted-run pin DTO.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    Args:
        response: Completed sync response carrying slot-pinned artifact metadata.
    Returns:
        BacktestJobArtifactPin: Immutable artifact pin reused for persistence and API response.
    Assumptions:
        The inner sync run already executed against this exact pinned artifact identity.
    Raises:
        BacktestValidationError: If one artifact metadata field is missing from the sync response.
    Side Effects:
        None.
    """
    if (
        response.artifact_slot is None
        or response.artifact_slot_generation is None
        or response.artifact_asof_date is None
        or response.artifact_manifest_hash is None
    ):
        raise BacktestValidationError(
            "sync_inline persisted run requires slot-pinned artifact metadata"
        )
    return BacktestJobArtifactPin(
        artifact_slot=response.artifact_slot,
        artifact_slot_generation=response.artifact_slot_generation,
        artifact_manifest_hash=response.artifact_manifest_hash,
        artifact_asof_date=response.artifact_asof_date,
    )


def _build_request_json_payload(
    *,
    request: RunBacktestRequest,
    request_payload: Mapping[str, Any],
    response: RunBacktestResponse,
) -> Mapping[str, Any]:
    """
    Build canonical persisted `request_json` payload for one completed sync-inline run.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    Args:
        request: Parsed application request DTO.
        request_payload: Strict API request payload snapshot.
        response: Completed sync response carrying resolved runtime defaults.
    Returns:
        Mapping[str, Any]: Deterministic JSON-compatible request snapshot for persistence.
    Assumptions:
        Sync-inline persistence keeps the same canonical request shape as the jobs storage family,
        while live execution-profile metadata is stored additively outside `request_json`.
    Raises:
        BacktestValidationError: If template/saved mode payload cannot be reconstructed.
    Side Effects:
        None.
    """
    normalized_payload = _normalize_json_mapping(values=request_payload)
    normalized_payload.pop("execution_profile_mode", None)
    normalized_payload.pop("execution_profile_mode_hint", None)
    normalized_payload.pop("effective_execution_profile_mode", None)
    normalized_payload["top_k"] = response.top_k
    normalized_payload["preselect"] = response.preselect

    direction_mode = _require_direction_mode(response=response)
    sizing_mode = _require_sizing_mode(response=response)
    execution_payload = _sorted_execution_payload(response=response)

    if request.mode == "template":
        raw_template = normalized_payload.get("template")
        if not isinstance(raw_template, Mapping):
            raise BacktestValidationError("template mode sync_inline persistence requires template")
        template_payload = dict(raw_template)
        template_payload["direction_mode"] = direction_mode
        template_payload["sizing_mode"] = sizing_mode
        template_payload["execution"] = execution_payload
        normalized_payload["template"] = template_payload
        normalized_payload.pop("overrides", None)
        return normalized_payload

    overrides_payload: dict[str, Any] = {}
    raw_overrides = normalized_payload.get("overrides")
    if isinstance(raw_overrides, Mapping):
        overrides_payload = dict(raw_overrides)
    overrides_payload["direction_mode"] = direction_mode
    overrides_payload["sizing_mode"] = sizing_mode
    overrides_payload["execution"] = execution_payload
    normalized_payload["overrides"] = overrides_payload
    if request.strategy_id is not None:
        normalized_payload["strategy_id"] = str(request.strategy_id)
    normalized_payload.pop("template", None)
    return normalized_payload


def _require_response_execution_profile_mode(*, response: RunBacktestResponse) -> str:
    """
    Require effective exact execution profile mode on completed sync `/backtests` responses.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    Args:
        response: Completed sync response carrying resolved runtime metadata.
    Returns:
        str: Normalized execution-profile mode literal.
    Assumptions:
        Profile-aware exact classification always resolves one effective profile before
        persisted-run sync storage is written.
    Raises:
        BacktestValidationError: If the sync response lacks a valid effective profile literal.
    Side Effects:
        None.
    """
    raw_mode = response.execution_profile_mode
    if raw_mode is None or not raw_mode.strip():
        raise BacktestValidationError(
            "sync_inline persisted run requires execution_profile_mode"
        )
    try:
        return validate_execution_profile_mode_v2(value=raw_mode)
    except ValueError as error:
        raise BacktestValidationError(
            "sync_inline persisted run requires valid execution_profile_mode"
        ) from error


def _resolve_sync_inline_persisted_execution_profile_metadata(
    *,
    response: RunBacktestResponse,
) -> tuple[str, str]:
    """
    Build additive sync-inline execution-profile metadata persisted outside `request_json`.

    Args:
        response: Completed sync response carrying resolved runtime metadata.
    Returns:
        tuple[str, str]: Additive
            `(execution_profile_mode_hint, effective_execution_profile_mode)` metadata tuple.
    Assumptions:
        The sync-inline path completes before persistence, so both launch-time and effective
        profile metadata are known deterministically from the finished response.
    Raises:
        BacktestValidationError: If the response lacks a valid execution-profile mode.
    Side Effects:
        None.
    """
    execution_profile_mode = _require_response_execution_profile_mode(response=response)
    return (execution_profile_mode, execution_profile_mode)


def _build_engine_params_hash(*, response: RunBacktestResponse) -> str:
    """
    Build deterministic engine-params hash from resolved sync execution payload.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - apps/api/dto/backtests.py
    Args:
        response: Completed sync response with resolved execution metadata.
    Returns:
        str: Canonical SHA-256 hash of effective direction/sizing/execution settings.
    Assumptions:
        Response carries the same effective execution payload that was used during the sync run.
    Raises:
        BacktestValidationError: If response lacks effective execution metadata.
    Side Effects:
        None.
    """
    return _build_sha256_from_payload(
        payload={
            "direction_mode": _require_direction_mode(response=response),
            "sizing_mode": _require_sizing_mode(response=response),
            "execution": _sorted_execution_payload(response=response),
        }
    )


def _build_terminal_sync_inline_run(
    *,
    run_id: UUID,
    created_at: datetime,
    finished_at: datetime,
    current_user: CurrentUser,
    request: RunBacktestRequest,
    response: RunBacktestResponse,
    request_json: Mapping[str, Any],
    artifact_pin: BacktestJobArtifactPin,
    backtest_runtime_config_hash: str,
    engine_params_hash: str,
) -> BacktestJob:
    """
    Build final succeeded sync-inline run aggregate snapshot for one persisted sync response.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
    Args:
        run_id: Persisted run identifier.
        created_at: Sync run create timestamp captured before compute started.
        finished_at: Terminal timestamp captured after sync compute completed.
        current_user: Authenticated owner identity.
        request: Parsed application request DTO.
        response: Completed sync response carrying resolved runtime metadata.
        request_json: Canonical request snapshot payload for storage.
        artifact_pin: Immutable artifact pin used by the sync execution.
        backtest_runtime_config_hash: Canonical result-affecting runtime hash.
        engine_params_hash: Canonical effective execution hash.
    Returns:
        BacktestJob: Terminal succeeded run aggregate ready for persistence.
    Assumptions:
        Sync-inline persistence records one logical attempt represented by
        `queued -> running -> succeeded` domain transitions in memory.
    Raises:
        BacktestValidationError: If saved-mode response lacks required persisted spec metadata.
    Side Effects:
        None.
    """
    ranking_primary_metric = (
        request.ranking.primary_metric
        if request.ranking is not None
        else BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1
    )
    (
        execution_profile_mode_hint,
        effective_execution_profile_mode,
    ) = _resolve_sync_inline_persisted_execution_profile_metadata(response=response)
    spec_hash = response.spec_hash if request.mode == "saved" else None
    spec_payload_json = response.spec_payload_json if request.mode == "saved" else None
    if request.mode == "saved" and (spec_hash is None or spec_payload_json is None):
        raise BacktestValidationError(
            "saved mode sync_inline persistence requires spec snapshot metadata"
        )

    queued = BacktestJob.create_queued(
        job_id=run_id,
        user_id=current_user.user_id,
        mode=cast(BacktestJobMode, request.mode),
        created_at=created_at,
        request_json=request_json,
        request_hash=_build_request_hash_from_request_json(payload=request_json),
        spec_hash=spec_hash,
        spec_payload_json=spec_payload_json,
        engine_params_hash=engine_params_hash,
        backtest_runtime_config_hash=backtest_runtime_config_hash,
        artifact_pin=artifact_pin,
        execution_mode="sync_inline",
        execution_profile_mode_hint=execution_profile_mode_hint,
        effective_execution_profile_mode=effective_execution_profile_mode,
        market_id=response.instrument_id.market_id.value,
        symbol=str(response.instrument_id.symbol),
        timeframe=str(response.timeframe),
        requested_top_n=response.top_k,
        ranking_primary_metric=ranking_primary_metric,
        ranking_secondary_metric=None,
    )
    claimed = queued.claim(
        changed_at=created_at,
        locked_by=_SYNC_INLINE_LOCKED_BY,
        lease_expires_at=created_at + timedelta(seconds=1),
    )
    return claimed.finish(next_state="succeeded", changed_at=finished_at)


def _build_persisted_top_rows(
    *,
    job_id: UUID,
    persisted_at: datetime,
    response: RunBacktestResponse,
) -> tuple[BacktestJobTopVariant, ...]:
    """
    Convert sync response variants into summary-only persisted top rows via shared worker mappers.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
    Args:
        job_id: Persisted run identifier.
        persisted_at: Final persistence timestamp in UTC.
        response: Completed sync response with ranked variants.
    Returns:
        tuple[BacktestJobTopVariant, ...]: Summary-only persisted top rows ordered by rank.
    Assumptions:
        Response variants are already ordered by the final deterministic ranking contract.
    Raises:
        BacktestValidationError: If one ranked variant lacks explicit payload data.
    Side Effects:
        None.
    """
    candidates: list[BacktestJobTopVariantCandidateV1] = []
    for variant in response.variants:
        if variant.payload is None:  # pragma: no cover - guarded by DTO invariant
            raise BacktestValidationError("sync_inline persisted top rows require variant payload")
        candidates.append(
            BacktestJobTopVariantCandidateV1(
                variant_index=variant.variant_index,
                variant_key=variant.variant_key,
                indicator_variant_key=variant.indicator_variant_key,
                total_return_pct=variant.total_return_pct,
                indicator_selections=variant.payload.indicator_selections,
                signal_params=cast(
                    Mapping[str, Mapping[str, Any]],
                    variant.payload.signal_params or {},
                ),
                risk_params=cast(Mapping[str, Any], variant.payload.risk_params or {}),
                summary_metrics_json=variant.summary_metrics_json,
                best_tp_pct=variant.best_tp_pct,
                best_sl_pct=variant.best_sl_pct,
            )
        )
    return build_finalized_snapshot_rows(
        job_id=job_id,
        now=persisted_at,
        ranked_candidates=tuple(candidates),
        direction_mode=_require_direction_mode(response=response),
        sizing_mode=_require_sizing_mode(response=response),
        execution_params=_sorted_execution_payload(response=response),
        reports_by_variant_key={},
        trades_by_variant_key={},
    )


def _require_direction_mode(*, response: RunBacktestResponse) -> str:
    """
    Require deterministic effective direction mode on the completed sync response.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    Args:
        response: Completed sync response.
    Returns:
        str: Effective direction mode literal.
    Assumptions:
        `RunBacktestUseCase` always resolves and returns one effective direction mode.
    Raises:
        BacktestValidationError: If direction mode metadata is unexpectedly absent.
    Side Effects:
        None.
    """
    if response.direction_mode is None:
        raise BacktestValidationError("sync_inline persisted run requires direction_mode")
    return response.direction_mode


def _require_sizing_mode(*, response: RunBacktestResponse) -> str:
    """
    Require deterministic effective sizing mode on the completed sync response.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    Args:
        response: Completed sync response.
    Returns:
        str: Effective sizing mode literal.
    Assumptions:
        `RunBacktestUseCase` always resolves and returns one effective sizing mode.
    Raises:
        BacktestValidationError: If sizing mode metadata is unexpectedly absent.
    Side Effects:
        None.
    """
    if response.sizing_mode is None:
        raise BacktestValidationError("sync_inline persisted run requires sizing_mode")
    return response.sizing_mode


def _sorted_execution_payload(*, response: RunBacktestResponse) -> dict[str, Any]:
    """
    Require and normalize effective execution params from the completed sync response.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    Args:
        response: Completed sync response.
    Returns:
        dict[str, Any]: Deterministic execution mapping sorted by key.
    Assumptions:
        Effective execution params are resolved before compute and returned by the sync use-case.
    Raises:
        BacktestValidationError: If execution metadata is unexpectedly absent.
    Side Effects:
        None.
    """
    if response.execution_params is None:
        raise BacktestValidationError("sync_inline persisted run requires execution_params")
    return {key: response.execution_params[key] for key in sorted(response.execution_params.keys())}


def _utc_now() -> datetime:
    """
    Return timezone-aware UTC timestamp for persisted sync-inline lifecycle writes.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    Args:
        None.
    Returns:
        datetime: Current UTC-aware timestamp.
    Assumptions:
        Persisted sync-inline rows use UTC timestamps consistently with the jobs storage family.
    Raises:
        None.
    Side Effects:
        None.
    """
    return datetime.now(timezone.utc)


__all__ = [
    "BacktestRunsApiUseCase",
    "CreateAndRunBacktestSyncInlineUseCase",
]
