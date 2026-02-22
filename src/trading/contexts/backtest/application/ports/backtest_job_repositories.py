from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobErrorPayload,
    BacktestJobStage,
    BacktestJobStageAShortlist,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class BacktestJobListQuery:
    """
    Deterministic keyset list query parameters for user-scoped Backtest jobs API reads.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - apps/api/routes/backtests.py
    """

    user_id: UserId
    limit: int = 50
    state: BacktestJobState | None = None
    cursor: BacktestJobListCursor | None = None

    def __post_init__(self) -> None:
        """
        Validate keyset list query invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Query ordering is fixed to `(created_at DESC, job_id DESC)`.
        Raises:
            ValueError: If limit is out of range.
        Side Effects:
            None.
        """
        if self.user_id is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestJobListQuery.user_id is required")
        if self.limit <= 0:
            raise ValueError("BacktestJobListQuery.limit must be > 0")
        if self.limit > 250:
            raise ValueError("BacktestJobListQuery.limit must be <= 250")


@dataclass(frozen=True, slots=True)
class BacktestJobListPage:
    """
    Deterministic keyset page payload for Backtest jobs list repository contract.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - apps/api/routes/backtests.py
    """

    items: tuple[BacktestJob, ...]
    next_cursor: BacktestJobListCursor | None


class BacktestJobRepository(Protocol):
    """
    Backtest job core storage port for create/get/list/cancel/quota operations.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/milestone-5-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - alembic/versions/20260222_0003_backtest_jobs_v1.py
    """

    def create(self, *, job: BacktestJob) -> BacktestJob:
        """
        Persist new job row and return stored immutable snapshot.

        Args:
            job: Prepared queued Backtest job aggregate.
        Returns:
            BacktestJob: Persisted immutable row projection.
        Assumptions:
            Saved/template invariants are already validated by domain aggregate.
        Raises:
            ValueError: If storage write fails.
        Side Effects:
            Writes one row in `backtest_jobs` table.
        """
        ...

    def get(self, *, job_id: UUID, user_id: UserId | None = None) -> BacktestJob | None:
        """
        Load job snapshot by id with optional owner filter.

        Args:
            job_id: Job identifier.
            user_id: Optional owner filter for user-scoped read paths.
        Returns:
            BacktestJob | None: Job snapshot or `None` when not found.
        Assumptions:
            Owner checks are explicit and deterministic at use-case/API layer.
        Raises:
            ValueError: If row mapping fails.
        Side Effects:
            Reads one row from `backtest_jobs` table.
        """
        ...

    def list_for_user(self, *, query: BacktestJobListQuery) -> BacktestJobListPage:
        """
        List user jobs using deterministic keyset ordering and cursor semantics.

        Args:
            query: User-scoped keyset query payload.
        Returns:
            BacktestJobListPage: Deterministic page payload.
        Assumptions:
            SQL ordering is fixed to `created_at DESC, job_id DESC`.
        Raises:
            ValueError: If storage read or row mapping fails.
        Side Effects:
            Reads zero or more rows from `backtest_jobs` table.
        """
        ...

    def cancel(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        cancel_requested_at: datetime,
    ) -> BacktestJob | None:
        """
        Request cancel for owner job with deterministic queued/running semantics.

        Args:
            job_id: Job identifier.
            user_id: Job owner identifier.
            cancel_requested_at: Cancel timestamp in UTC.
        Returns:
            BacktestJob | None: Updated job snapshot or `None` when job is not found.
        Assumptions:
            `queued` jobs are cancelled immediately; `running` jobs get cancel-request mark.
        Raises:
            ValueError: If storage write/read fails.
        Side Effects:
            Updates one row in `backtest_jobs` table.
        """
        ...

    def count_active_for_user(self, *, user_id: UserId) -> int:
        """
        Count owner active jobs (`queued + running`) for per-user quota checks.

        Args:
            user_id: Job owner identifier.
        Returns:
            int: Number of active jobs.
        Assumptions:
            Active states are fixed by Backtest jobs storage contract.
        Raises:
            ValueError: If storage read fails.
        Side Effects:
            Reads aggregate count from `backtest_jobs` table.
        """
        ...


class BacktestJobLeaseRepository(Protocol):
    """
    Backtest job lease port for claim/heartbeat/progress/finish worker operations.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/milestone-5-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_lease_repository.py
      - alembic/versions/20260222_0003_backtest_jobs_v1.py
    """

    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestJob | None:
        """
        Claim one job using FIFO queue order and SKIP LOCKED reclaim semantics.

        Args:
            now: Claim timestamp in UTC.
            locked_by: Worker owner identity.
            lease_seconds: Lease TTL in seconds.
        Returns:
            BacktestJob | None: Claimed running job snapshot or `None` when queue is empty.
        Assumptions:
            Claim prefers oldest `queued` jobs before expired `running` reclaim candidates.
        Raises:
            ValueError: If storage write/read fails.
        Side Effects:
            Updates one row lease fields in `backtest_jobs` table.
        """
        ...

    def heartbeat(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestJob | None:
        """
        Extend active lease for running job under owner-matched conditional write.

        Args:
            job_id: Job identifier.
            now: Heartbeat timestamp in UTC.
            locked_by: Expected worker owner identity.
            lease_seconds: Lease extension TTL in seconds.
        Returns:
            BacktestJob | None: Updated running job snapshot or `None` when lease is lost.
        Assumptions:
            Conditional write is guarded by `(job_id, locked_by, lease_expires_at > now)`.
        Raises:
            ValueError: If storage write/read fails.
        Side Effects:
            Updates one row lease fields in `backtest_jobs` table.
        """
        ...

    def update_progress(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        stage: BacktestJobStage,
        processed_units: int,
        total_units: int,
    ) -> BacktestJob | None:
        """
        Persist running progress fields guarded by active lease owner predicate.

        Args:
            job_id: Job identifier.
            now: Progress timestamp in UTC.
            locked_by: Expected worker owner identity.
            stage: Current stage literal.
            processed_units: Processed stage units.
            total_units: Total stage units.
        Returns:
            BacktestJob | None: Updated running job snapshot or `None` when lease is lost.
        Assumptions:
            Worker writes must be conditional on active lease to avoid split-brain updates.
        Raises:
            ValueError: If storage write/read fails.
        Side Effects:
            Updates one row progress fields in `backtest_jobs` table.
        """
        ...

    def finish(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        next_state: BacktestJobState,
        last_error: str | None = None,
        last_error_json: BacktestJobErrorPayload | None = None,
    ) -> BacktestJob | None:
        """
        Transition running job to terminal state with lease-owner conditional guard.

        Args:
            job_id: Job identifier.
            now: Finish timestamp in UTC.
            locked_by: Expected worker owner identity.
            next_state: Target terminal state (`succeeded|failed|cancelled`).
            last_error: Short failure text for failed state.
            last_error_json: RoehubError-like failure payload for failed state.
        Returns:
            BacktestJob | None: Updated terminal job snapshot or `None` when lease is lost.
        Assumptions:
            `queued -> failed` is forbidden by domain contract and must never be persisted.
        Raises:
            ValueError: If storage write/read fails.
        Side Effects:
            Updates one row state and terminal fields in `backtest_jobs` table.
        """
        ...


class BacktestJobResultsRepository(Protocol):
    """
    Backtest job results port for top-k snapshots and Stage-A shortlist persistence.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
      - alembic/versions/20260222_0003_backtest_jobs_v1.py
    """

    def replace_top_variants_snapshot(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        rows: tuple[BacktestJobTopVariant, ...],
    ) -> bool:
        """
        Replace whole best-so-far top-k snapshot in one transaction-like SQL statement.

        Args:
            job_id: Job identifier.
            now: Snapshot timestamp in UTC.
            locked_by: Expected worker owner identity.
            rows: Full ranked rows payload replacing existing snapshot.
        Returns:
            bool: `True` when lease-guarded replace is applied; `False` when lease is lost.
        Assumptions:
            Snapshot write uses delete+insert contract for deterministic full replacement.
        Raises:
            ValueError: If storage write fails.
        Side Effects:
            Replaces rows in `backtest_job_top_variants` table.
        """
        ...

    def list_top_variants(self, *, job_id: UUID, limit: int) -> tuple[BacktestJobTopVariant, ...]:
        """
        Load persisted ranked top variants ordered deterministically by rank and key.

        Args:
            job_id: Job identifier.
            limit: Max rows to read.
        Returns:
            tuple[BacktestJobTopVariant, ...]: Deterministic ranked rows.
        Assumptions:
            SQL ordering is explicit and stable for equal rank values.
        Raises:
            ValueError: If storage read or row mapping fails.
        Side Effects:
            Reads rows from `backtest_job_top_variants` table.
        """
        ...

    def save_stage_a_shortlist(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        shortlist: BacktestJobStageAShortlist,
    ) -> bool:
        """
        Persist Stage-A shortlist with lease-owner guarded upsert semantics.

        Args:
            job_id: Job identifier.
            now: Upsert timestamp in UTC.
            locked_by: Expected worker owner identity.
            shortlist: Stage-A shortlist payload.
        Returns:
            bool: `True` when upsert is applied; `False` when lease is lost.
        Assumptions:
            Stage-A payload uses deterministic ordered indexes array.
        Raises:
            ValueError: If storage write fails.
        Side Effects:
            Upserts one row in `backtest_job_stage_a_shortlist` table.
        """
        ...

    def get_stage_a_shortlist(self, *, job_id: UUID) -> BacktestJobStageAShortlist | None:
        """
        Load persisted Stage-A shortlist payload by job id.

        Args:
            job_id: Job identifier.
        Returns:
            BacktestJobStageAShortlist | None: Snapshot payload or `None`.
        Assumptions:
            Row key is one-to-one with `backtest_jobs.job_id`.
        Raises:
            ValueError: If row mapping fails.
        Side Effects:
            Reads one row from `backtest_job_stage_a_shortlist` table.
        """
        ...


__all__ = [
    "BacktestJobLeaseRepository",
    "BacktestJobListPage",
    "BacktestJobListQuery",
    "BacktestJobRepository",
    "BacktestJobResultsRepository",
]
