from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from uuid import UUID

from trading.contexts.backtest.application.dto import (
    BacktestReportV1,
    BacktestVariantPayloadV1,
    RunBacktestRequest,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobRepository,
    BacktestJobRequestDecoder,
    BacktestJobResultsRepository,
    CurrentUser,
)
from trading.contexts.backtest.application.services import (
    ArtifactPinnedIdentityV2,
    ArtifactSlotPinnedRuntimeContextV2,
    BacktestArtifactSlotResolverV2,
    ExecutionProfileModeLiteralV2,
    ExecutionProfilesCatalogV2,
    artifact_coordinates_from_market_id_v2,
    default_execution_profiles_catalog_v2,
    validate_execution_profile_mode_v2,
)
from trading.contexts.backtest.application.services.run_control_v1 import BacktestRunControlV1
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor

from .backtest_jobs_api_v1 import ListBacktestJobsUseCase
from .errors import (
    backtest_run_forbidden,
    backtest_run_not_found,
    validation_error,
)
from .run_backtest import RunBacktestUseCase
from .run_backtest_job_runner_v1 import (
    apply_saved_overrides_v1,
    build_template_from_saved_spec_payload_v1,
)

NowProvider = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class BacktestRunProgressSnapshot:
    """
    Additive persisted-run progress read model for public status/history responses.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
      - apps/api/dto/backtest_runs.py
      - apps/api/routes/backtest_runs.py
    """

    execution_profile_mode: ExecutionProfileModeLiteralV2
    progress_percent: int
    eta_seconds: int | None


class BacktestRunProgressSnapshotBuilder:
    """
    Project persisted run counters onto additive progress/ETA fields for public runs read paths.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - apps/api/dto/backtest_runs.py
    """

    def __init__(
        self,
        *,
        execution_profiles: ExecutionProfilesCatalogV2 | None = None,
        now_provider: NowProvider | None = None,
    ) -> None:
        """
        Store the execution-profile catalog and clock used for conservative ETA projection.

        Args:
            execution_profiles: Optional startup-validated execution-profile catalog.
            now_provider: Optional UTC clock provider for deterministic tests/read models.
        Returns:
            None.
        Assumptions:
            Current A2 contract keeps persisted run profile resolution additive and may fall back
            to the configured default profile when the run snapshot has no explicit profile field.
        Raises:
            ValueError: If the resolved execution-profile catalog is missing.
        Side Effects:
            None.
        """
        resolved_execution_profiles = (
            execution_profiles or default_execution_profiles_catalog_v2()
        )
        if resolved_execution_profiles is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "BacktestRunProgressSnapshotBuilder requires execution_profiles"
            )
        self._execution_profiles = resolved_execution_profiles
        self._now = now_provider or _utc_now

    def build(self, *, run: BacktestJob) -> BacktestRunProgressSnapshot:
        """
        Build additive `progress_percent/eta_seconds/execution_profile_mode` for one run.

        Args:
            run: Persisted run aggregate read from unified storage.
        Returns:
            BacktestRunProgressSnapshot: Additive progress read model for public API/UI.
        Assumptions:
            ETA relies only on current-run timestamps and weighted progress; benchmark-history
            fallbacks stay out of scope for this milestone.
        Raises:
            None.
        Side Effects:
            None.
        """
        execution_profile_mode = self._resolve_execution_profile_mode(run=run)
        stage_weights = self._execution_profiles.profile_for_mode(
            mode=execution_profile_mode
        ).progress_weights
        return BacktestRunProgressSnapshot(
            execution_profile_mode=execution_profile_mode,
            progress_percent=run.progress_percent(stage_weights=stage_weights),
            eta_seconds=run.eta_seconds(stage_weights=stage_weights, now=self._now()),
        )

    def _resolve_execution_profile_mode(
        self,
        *,
        run: BacktestJob,
    ) -> ExecutionProfileModeLiteralV2:
        """
        Resolve the effective execution-profile mode for a persisted run status projection.

        Args:
            run: Persisted run aggregate read from unified storage.
        Returns:
            ExecutionProfileModeLiteralV2: Stable execution-profile mode literal for UI rendering.
        Assumptions:
            Profile-aware launch now persists effective profile selection in `request_json` for
            `/backtests` runs, while older rows or unrelated legacy rows may still fall back to
            the configured default exact profile.
        Raises:
            None.
        Side Effects:
            None.
        """
        raw_mode = run.request_json.get("execution_profile_mode")
        if isinstance(raw_mode, str) and raw_mode.strip():
            try:
                normalized_mode = validate_execution_profile_mode_v2(value=raw_mode)
                self._execution_profiles.profile_for_mode(mode=normalized_mode)
                return normalized_mode
            except ValueError:
                pass
        return self._execution_profiles.default_mode


@dataclass(frozen=True, slots=True)
class BacktestRunTopReadResult:
    """
    Public runs `/top` payload over persisted summary-only rows.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
      - apps/api/dto/backtest_runs.py
      - apps/api/routes/backtest_runs.py
    """

    job: BacktestJob
    rows: tuple[BacktestJobTopVariant, ...]


class BuildBacktestRunVariantReportUseCase:
    """
    Recompute one persisted run variant detail from stored request and pinned artifact context.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - apps/api/routes/backtest_runs.py
    """

    def __init__(
        self,
        *,
        job_repository: BacktestJobRepository,
        request_decoder: BacktestJobRequestDecoder,
        run_use_case: RunBacktestUseCase,
        artifact_slot_resolver: BacktestArtifactSlotResolverV2,
    ) -> None:
        """
        Initialize persisted-run single-variant detail use-case dependencies.

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/ports/backtest_job_request_decoder.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - apps/api/wiring/modules/backtest.py
        Args:
            job_repository: Persisted-run repository over unified jobs storage.
            request_decoder: Decoder for canonical persisted `request_json`.
            run_use_case: Backtest use-case exposing one-variant report build path.
            artifact_slot_resolver: Shared slot-pinned artifact resolver used for reproducibility.
        Returns:
            None.
        Assumptions:
            Startup wiring already validated artifact configs and resolver dependencies fail-fast.
        Raises:
            ValueError: If one dependency is missing.
        Side Effects:
            None.
        """
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("BuildBacktestRunVariantReportUseCase requires job_repository")
        if request_decoder is None:  # type: ignore[truthy-bool]
            raise ValueError("BuildBacktestRunVariantReportUseCase requires request_decoder")
        if run_use_case is None:  # type: ignore[truthy-bool]
            raise ValueError("BuildBacktestRunVariantReportUseCase requires run_use_case")
        if artifact_slot_resolver is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "BuildBacktestRunVariantReportUseCase requires artifact_slot_resolver"
            )
        self._job_repository = job_repository
        self._request_decoder = request_decoder
        self._run_use_case = run_use_case
        self._artifact_slot_resolver = artifact_slot_resolver

    def execute(
        self,
        *,
        run_id: UUID,
        current_user: CurrentUser,
        variant_payload: BacktestVariantPayloadV1,
        include_trades: bool = False,
        run_control: BacktestRunControlV1 | None = None,
    ) -> BacktestReportV1:
        """
        Recompute exactly one selected persisted-run variant detail lazily.

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - apps/api/routes/backtest_runs.py
        Args:
            run_id: Persisted public run identifier.
            current_user: Authenticated owner identity.
            variant_payload: Explicit selected variant payload from summary row/UI selection.
            include_trades: Whether report payload should include trades.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            BacktestReportV1: Deterministic report payload for one variant only.
        Assumptions:
            Existing foreign run must map to `403 forbidden`, missing run to `404 not_found`.
        Raises:
            RoehubError: Canonical `forbidden`, `not_found`, or `validation_error` failures.
        Side Effects:
            Reads persisted run row, loads pinned artifact metadata, and executes one report build.
        """
        owner_run = _require_owner_run(
            job_repository=self._job_repository,
            run_id=run_id,
            current_user=current_user,
        )
        request = self._request_decoder.decode(payload=owner_run.request_json)
        resolved_request = self._resolve_report_request(run=owner_run, request=request)
        artifact_context = self._resolve_pinned_artifact_context(
            run=owner_run,
            request=resolved_request,
        )
        if resolved_request.template is None:  # pragma: no cover - guarded above
            raise BacktestValidationError(
                "Persisted run detail request requires resolved template payload"
            )
        return self._run_use_case.build_variant_report_for_template(
            requested_time_range=resolved_request.time_range,
            template=resolved_request.template,
            warmup_bars=resolved_request.warmup_bars,
            variant_payload=variant_payload,
            include_trades=include_trades,
            run_control=run_control,
            artifact_context=artifact_context,
            template_root_path="persisted_run.template",
        )

    def _resolve_report_request(
        self,
        *,
        run: BacktestJob,
        request: RunBacktestRequest,
    ) -> RunBacktestRequest:
        """
        Rebuild original request semantics from persisted run storage snapshot only.

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
          - src/trading/contexts/backtest/domain/entities/backtest_job.py
        Args:
            run: Owner persisted run row.
            request: Decoded canonical `request_json` payload.
        Returns:
            RunBacktestRequest: Request rebuilt for one-variant lazy detail execution.
        Assumptions:
            Saved-mode persisted runs reconstruct template from `spec_payload_json + overrides`.
        Raises:
            BacktestValidationError: If persisted run snapshot is incomplete or inconsistent.
        Side Effects:
            None.
        """
        if request.template is not None:
            return request
        if run.mode != "saved":
            raise BacktestValidationError(
                "Persisted run request_json must include template payload for template mode"
            )
        if run.spec_payload_json is None:
            raise BacktestValidationError(
                "Persisted saved run requires spec_payload_json for lazy variant detail"
            )
        template = apply_saved_overrides_v1(
            base_template=build_template_from_saved_spec_payload_v1(
                spec_payload=run.spec_payload_json
            ),
            overrides=request.overrides,
        )
        return RunBacktestRequest(
            time_range=request.time_range,
            strategy_id=None,
            template=template,
            overrides=None,
            warmup_bars=request.warmup_bars,
            top_k=request.top_k,
            preselect=request.preselect,
            top_trades_n=request.top_trades_n,
            ranking=request.ranking,
        )

    def _resolve_pinned_artifact_context(
        self,
        *,
        run: BacktestJob,
        request: RunBacktestRequest,
    ) -> ArtifactSlotPinnedRuntimeContextV2:
        """
        Resolve immutable slot-pinned runtime context from persisted run artifact identity.

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - src/trading/contexts/backtest/domain/entities/backtest_job.py
        Args:
            run: Owner persisted run row carrying pinned artifact metadata.
            request: Reconstructed run request with effective template payload.
        Returns:
            ArtifactSlotPinnedRuntimeContextV2: Immutable pinned artifact context.
        Assumptions:
            Lazy detail must not read live `current.yaml`; it must reuse persisted run pin fields.
        Raises:
            BacktestValidationError: If persisted artifact metadata is missing or inconsistent.
        Side Effects:
            Reads one explicit slot manifest from disk.
        """
        if run.artifact_pin is None:
            raise BacktestValidationError(
                "Persisted run requires slot-pinned artifact metadata for lazy variant detail"
            )
        if request.template is None:  # pragma: no cover - guarded by caller
            raise BacktestValidationError(
                "Persisted run detail request requires resolved template payload"
            )
        try:
            coordinates = artifact_coordinates_from_market_id_v2(
                market_id=request.template.instrument_id.market_id.value,
                symbol=str(request.template.instrument_id.symbol),
            )
            return self._artifact_slot_resolver.resolve_pinned_context(
                coordinates,
                ArtifactPinnedIdentityV2(
                    artifact_slot=run.artifact_pin.artifact_slot,
                    slot_generation=run.artifact_pin.artifact_slot_generation,
                    artifact_asof_date=run.artifact_pin.artifact_asof_date,
                    artifact_manifest_hash=run.artifact_pin.artifact_manifest_hash,
                ),
            )
        except ValueError as error:
            raise BacktestValidationError(
                "Persisted run artifact pin violates shared slot-pinned context contract: "
                f"{error}"
            ) from error
        except FileNotFoundError as error:
            raise BacktestValidationError(
                "Pinned backtest artifacts are unavailable for persisted run instrument"
            ) from error


class GetBacktestRunStatusUseCase:
    """
    Read one owner-scoped persisted run snapshot with explicit `403` vs `404` semantics.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - apps/api/routes/backtest_runs.py
      - apps/api/dto/backtest_runs.py
    """

    def __init__(self, *, job_repository: BacktestJobRepository) -> None:
        """
        Initialize status use-case with persisted-run repository dependency.

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
          - apps/api/routes/backtest_runs.py
        Args:
            job_repository: Persisted-run repository port over unified jobs storage.
        Returns:
            None.
        Assumptions:
            Repository supports unscoped reads for explicit owner policy checks.
        Raises:
            ValueError: If dependency is missing.
        Side Effects:
            None.
        """
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("GetBacktestRunStatusUseCase requires job_repository")
        self._job_repository = job_repository

    def execute(self, *, run_id: UUID, current_user: CurrentUser) -> BacktestJob:
        """
        Read owner run snapshot with public `run_id` error vocabulary.

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - apps/api/routes/backtest_runs.py
          - apps/api/common/errors.py
        Args:
            run_id: Requested persisted run identifier.
            current_user: Authenticated owner identity.
        Returns:
            BacktestJob: Owner run snapshot backed by unified storage.
        Assumptions:
            Existing foreign run must map to `403 forbidden`.
        Raises:
            RoehubError: Canonical `forbidden` or `not_found` for owner checks.
        Side Effects:
            Reads one run row from storage.
        """
        return _require_owner_run(
            job_repository=self._job_repository,
            run_id=run_id,
            current_user=current_user,
        )


class GetBacktestRunTopUseCase:
    """
    Read owner persisted run summary rows with deterministic limit validation.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - apps/api/routes/backtest_runs.py
      - apps/api/dto/backtest_runs.py
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

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
          - apps/api/routes/backtest_runs.py
        Args:
            job_repository: Persisted-run repository port.
            results_repository: Summary top-rows repository port.
            top_k_persisted_default: Persisted summary-row cap from runtime config.
        Returns:
            None.
        Assumptions:
            Public runs `/top` shares the same persisted cap as legacy jobs endpoints.
        Raises:
            ValueError: If dependency or limit invariant is invalid.
        Side Effects:
            None.
        """
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("GetBacktestRunTopUseCase requires job_repository")
        if results_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("GetBacktestRunTopUseCase requires results_repository")
        if top_k_persisted_default <= 0:
            raise ValueError("top_k_persisted_default must be > 0")
        self._job_repository = job_repository
        self._results_repository = results_repository
        self._top_k_persisted_default = top_k_persisted_default

    def execute(
        self,
        *,
        run_id: UUID,
        current_user: CurrentUser,
        limit: int | None,
    ) -> BacktestRunTopReadResult:
        """
        Read owner summary-only top rows for one persisted run.

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
          - apps/api/routes/backtest_runs.py
        Args:
            run_id: Requested persisted run identifier.
            current_user: Authenticated owner identity.
            limit: Optional summary rows limit.
        Returns:
            BacktestRunTopReadResult: Owner run and deterministically ordered summary rows.
        Assumptions:
            Row ordering is fixed to `rank ASC, variant_key ASC`.
        Raises:
            RoehubError: Canonical `forbidden|not_found|validation_error` errors.
        Side Effects:
            Reads one run row and zero or more summary rows from storage.
        """
        resolved_limit = self._resolve_limit(limit=limit)
        owner_run = _require_owner_run(
            job_repository=self._job_repository,
            run_id=run_id,
            current_user=current_user,
        )
        rows = self._results_repository.list_top_variants(job_id=run_id, limit=resolved_limit)
        return BacktestRunTopReadResult(job=owner_run, rows=rows)

    def _resolve_limit(self, *, limit: int | None) -> int:
        """
        Resolve public `/runs/{run_id}/top` limit against persisted summary-row cap.

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - apps/api/routes/backtest_runs.py
          - apps/api/common/errors.py
        Args:
            limit: Optional query limit.
        Returns:
            int: Effective positive limit within persisted cap.
        Assumptions:
            Missing limit falls back to `backtest.jobs.top_k_persisted_default`.
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
                        "message": f"limit must be <= {self._top_k_persisted_default}",
                    },
                ),
            )
        return limit


class ListBacktestRunsUseCase:
    """
    List owner persisted runs using deterministic keyset pagination contract.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - apps/api/routes/backtest_runs.py
      - apps/api/dto/backtest_runs.py
    """

    def __init__(self, *, job_repository: BacktestJobRepository) -> None:
        """
        Initialize list use-case with persisted-run repository dependency.

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
          - apps/api/routes/backtest_runs.py
        Args:
            job_repository: Persisted-run repository port.
        Returns:
            None.
        Assumptions:
            Public runs list reuses the same keyset ordering and cursor semantics as jobs.
        Raises:
            ValueError: If dependency is missing.
        Side Effects:
            None.
        """
        self._delegate = ListBacktestJobsUseCase(job_repository=job_repository)

    def execute(
        self,
        *,
        current_user: CurrentUser,
        state: BacktestJobState | None,
        limit: int,
        cursor: BacktestJobListCursor | None,
    ) -> BacktestJobListPage:
        """
        Read owner persisted runs page using shared keyset repository query semantics.

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
          - apps/api/routes/backtest_runs.py
        Args:
            current_user: Authenticated owner identity.
            state: Optional state filter literal.
            limit: Requested page size.
            cursor: Optional opaque keyset cursor value object.
        Returns:
            BacktestJobListPage: Deterministic page payload backed by unified storage.
        Assumptions:
            Ordering stays fixed to `created_at DESC, job_id DESC`.
        Raises:
            ValueError: If query shape is invalid.
        Side Effects:
            Reads one runs page from storage.
        """
        return self._delegate.execute(
            current_user=current_user,
            state=state,
            limit=limit,
            cursor=cursor,
        )


class CancelBacktestRunUseCase:
    """
    Request owner run cancel and return updated idempotent status payload.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - apps/api/routes/backtest_runs.py
      - apps/api/dto/backtest_runs.py
    """

    def __init__(
        self,
        *,
        job_repository: BacktestJobRepository,
        now_provider: NowProvider | None = None,
    ) -> None:
        """
        Initialize cancel use-case with repository and optional deterministic clock.

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
          - apps/api/routes/backtest_runs.py
        Args:
            job_repository: Persisted-run repository port.
            now_provider: Optional UTC clock provider.
        Returns:
            None.
        Assumptions:
            Cancel is idempotent for terminal persisted runs.
        Raises:
            ValueError: If dependency is missing.
        Side Effects:
            None.
        """
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("CancelBacktestRunUseCase requires job_repository")
        self._job_repository = job_repository
        self._now = now_provider or _utc_now

    def execute(self, *, run_id: UUID, current_user: CurrentUser) -> BacktestJob:
        """
        Request cancel for owner persisted run and return current deterministic snapshot.

        Docs:
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
          - apps/api/routes/backtest_runs.py
        Args:
            run_id: Requested persisted run identifier.
            current_user: Authenticated owner identity.
        Returns:
            BacktestJob: Updated owner run snapshot.
        Assumptions:
            Existing foreign run must map to `403`, missing run to `404`.
        Raises:
            RoehubError: Canonical `forbidden` or `not_found` for owner checks.
        Side Effects:
            Writes cancel marker or terminal-state no-op for the owner run.
        """
        _require_owner_run(
            job_repository=self._job_repository,
            run_id=run_id,
            current_user=current_user,
        )
        cancelled = self._job_repository.cancel(
            job_id=run_id,
            user_id=current_user.user_id,
            cancel_requested_at=self._now(),
        )
        if cancelled is None:
            raise backtest_run_not_found(run_id=run_id)
        return cancelled


def _require_owner_run(
    *,
    job_repository: BacktestJobRepository,
    run_id: UUID,
    current_user: CurrentUser,
) -> BacktestJob:
    """
    Read persisted run by id and enforce explicit public owner policy.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - apps/api/routes/backtest_runs.py
    Args:
        job_repository: Persisted-run repository port over unified storage.
        run_id: Requested persisted run identifier.
        current_user: Authenticated owner identity.
    Returns:
        BacktestJob: Owner run snapshot.
    Assumptions:
        Access policy intentionally reads without owner SQL filters first.
    Raises:
        RoehubError: Canonical `not_found` for missing row and `forbidden` for foreign owner.
    Side Effects:
        Reads one run row from storage.
    """
    run = job_repository.get(job_id=run_id)
    if run is None:
        raise backtest_run_not_found(run_id=run_id)
    if run.user_id != current_user.user_id:
        raise backtest_run_forbidden(run_id=run_id)
    return run


def _utc_now() -> datetime:
    """
    Return UTC-aware current timestamp for persisted run lifecycle mutations.

    Docs:
      - docs/architecture/backtest/backtest-runs-history-v2.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
      - apps/api/routes/backtest_runs.py
      - src/trading/platform/time/system_clock.py
    Args:
        None.
    Returns:
        datetime: Current timezone-aware UTC timestamp.
    Assumptions:
        Caller needs wall-clock time only for idempotent cancel markers in tests/runtime.
    Raises:
        None.
    Side Effects:
        Reads system clock.
    """
    return datetime.now(timezone.utc)


__all__ = [
    "BacktestRunTopReadResult",
    "BuildBacktestRunVariantReportUseCase",
    "CancelBacktestRunUseCase",
    "GetBacktestRunStatusUseCase",
    "GetBacktestRunTopUseCase",
    "ListBacktestRunsUseCase",
]
