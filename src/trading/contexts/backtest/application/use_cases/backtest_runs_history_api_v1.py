from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from uuid import UUID

from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobRepository,
    BacktestJobResultsRepository,
    CurrentUser,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor

from .backtest_jobs_api_v1 import ListBacktestJobsUseCase
from .errors import (
    backtest_run_forbidden,
    backtest_run_not_found,
    validation_error,
)

NowProvider = Callable[[], datetime]


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
    "CancelBacktestRunUseCase",
    "GetBacktestRunStatusUseCase",
    "GetBacktestRunTopUseCase",
    "ListBacktestRunsUseCase",
]
