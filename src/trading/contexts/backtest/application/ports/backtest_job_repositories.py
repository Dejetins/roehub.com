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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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

    def create_with_top_variants(
        self,
        *,
        job: BacktestJob,
        top_variants: tuple[BacktestJobTopVariant, ...],
        stage_a_shortlist: BacktestJobStageAShortlist | None = None,
    ) -> BacktestJob:
        """
        Persist one terminal run row plus deterministic summary-only top rows atomically.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
          - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
            backtest_job_repository.py
          - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
        Args:
            job: Prepared terminal persisted-run aggregate.
            top_variants: Summary-only top rows ordered by `rank ASC, variant_key ASC`.
            stage_a_shortlist:
                Optional internal Stage A shortlist snapshot reused for
                `exact_no_risk_parity` sync persistence.
        Returns:
            BacktestJob: Persisted immutable job row projection.
        Assumptions:
            Sync-inline cutover writes final state, summary-only top rows, and optional internal
            shortlist state via the existing jobs table family.
        Raises:
            ValueError: If storage write fails or row mapping breaks.
        Side Effects:
            Writes one row in `backtest_jobs` and zero or more rows in
            `backtest_job_top_variants`, plus at most one internal
            `backtest_job_stage_a_shortlist` row.
        """
        ...

    def find_by_idempotency_key(
        self,
        *,
        user_id: UserId,
        idempotency_key_hash: str,
        created_after: datetime,
    ) -> BacktestJob | None:
        """
        Find durable owner job snapshot for one v1 idempotency key hash inside its TTL.

        Args:
            user_id: Job owner identifier.
            idempotency_key_hash: SHA-256 hash of the public `Idempotency-Key` value.
            created_after: Lower bound for TTL-compatible replay lookup.
        Returns:
            BacktestJob | None: Existing job snapshot or `None`.
        Assumptions:
            New v1 rows persist idempotency metadata under
            `request_json.idempotency.key_hash`.
        Raises:
            ValueError: If storage read fails.
        Side Effects:
            Reads at most one row from `backtest_jobs`.
        """
        ...

    def claim_for_inline_execution(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        now: datetime,
        locked_by: str,
        lease_expires_at: datetime,
    ) -> BacktestJob | None:
        """
        Claim one just-created queued job for sync-inline execution.

        Args:
            job_id: Job identifier.
            user_id: Owner identifier.
            now: Claim timestamp.
            locked_by: Inline worker owner literal.
            lease_expires_at: Lease expiry timestamp.
        Returns:
            BacktestJob | None: Running snapshot or `None` if state/owner changed.
        Assumptions:
            Sync-inline v1 still persists the running lifecycle state before compute.
        Raises:
            ValueError: If storage write/read fails.
        Side Effects:
            Updates one row from `queued` to `running`.
        """
        ...

    def finish_with_top_variants(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        now: datetime,
        locked_by: str,
        next_state: BacktestJobState,
        top_variants: tuple[BacktestJobTopVariant, ...],
        last_error: str | None = None,
        last_error_json: BacktestJobErrorPayload | None = None,
    ) -> BacktestJob | None:
        """
        Finish a running inline job and persist summary-only top rows atomically.

        Args:
            job_id: Job identifier.
            user_id: Owner identifier.
            now: Terminal timestamp.
            locked_by: Expected inline worker owner literal.
            next_state: Terminal state.
            top_variants: Summary-only rows for succeeded jobs.
            last_error: Failure text when `next_state='failed'`.
            last_error_json: Failure payload when `next_state='failed'`.
        Returns:
            BacktestJob | None: Terminal job snapshot or `None` when lease/owner changed.
        Assumptions:
            Storage `BacktestJobTopVariant.variant_key` remains SHA-only; public readable
            key is carried inside `payload_json.public_variant_key`.
        Raises:
            ValueError: If storage write/read fails.
        Side Effects:
            Updates one job row and inserts zero or more top rows.
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

    def list_top_variants(
        self,
        *,
        job_id: UUID,
        limit: int | None = None,
    ) -> tuple[BacktestJobTopVariant, ...]:
        """
        List persisted summary-only top rows for one job ordered by rank.

        Args:
            job_id: Parent job identifier.
            limit: Optional positive cap for preview reads.
        Returns:
            tuple[BacktestJobTopVariant, ...]: Rows sorted by `rank ASC`.
        Assumptions:
            Ownership has already been checked by the use-case layer.
        Raises:
            ValueError: If row mapping fails.
        Side Effects:
            Reads rows from `backtest_job_top_variants`.
        """
        ...

    def get_top_variant_by_public_key(
        self,
        *,
        job_id: UUID,
        public_variant_key: str,
    ) -> BacktestJobTopVariant | None:
        """
        Resolve one public readable variant key to a persisted top row inside one job.

        Args:
            job_id: Parent job identifier.
            public_variant_key: Public route key, not the raw storage SHA.
        Returns:
            BacktestJobTopVariant | None: Row or `None`.
        Assumptions:
            Direct public lookup by raw storage SHA is intentionally not supported.
        Raises:
            ValueError: If row mapping fails.
        Side Effects:
            Reads at most one top row.
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
            `queued` jobs are cancelled immediately; `running` jobs keep state=`running` and
            preserve the first persisted `cancel_requested_at` marker until worker finalization.
        Raises:
            ValueError: If storage write/read fails.
        Side Effects:
            Updates one row in `backtest_jobs` table.
        """
        ...

    def delete_terminal(self, *, job_id: UUID, user_id: UserId) -> bool:
        """
        Hard-delete one owner terminal job row and let dependent rows cascade.

        Args:
            job_id: Job identifier.
            user_id: Job owner identifier.
        Returns:
            bool: True when a terminal owner job was deleted.
        Assumptions:
            Active jobs are never hard-deleted through this port; callers must cancel first.
        Raises:
            ValueError: If storage write fails.
        Side Effects:
            Deletes one row from `backtest_jobs`; dependent rows cascade by schema.
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

    def count_created_for_user_since(
        self,
        *,
        user_id: UserId,
        created_after: datetime,
    ) -> int:
        """
        Count owner job creates inside one admission-rate window.

        Args:
            user_id: Job owner identifier.
            created_after: Inclusive lower bound for the create window.
        Returns:
            int: Number of jobs created by this owner in the window.
        Assumptions:
            Idempotency replays do not create new rows, so row count matches new admits.
        Raises:
            ValueError: If storage read fails.
        Side Effects:
            Reads aggregate count from `backtest_jobs`.
        """
        ...

    def count_active_global(self) -> int:
        """
        Count all active jobs (`queued + running`) for service-wide guardrails.

        Args:
            None.
        Returns:
            int: Active global job count.
        Assumptions:
            Active state set is fixed by Backtest jobs storage contract.
        Raises:
            ValueError: If storage read fails.
        Side Effects:
            Reads aggregate count from `backtest_jobs`.
        """
        ...

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        """
        Count active jobs pinning one previously published inactive-slot manifest identity.

        Args:
            market_id: Canonical market id for the symbol-root being published.
            symbol: Instrument symbol pinned by the active jobs.
            artifact_slot: Candidate inactive slot literal.
            artifact_manifest_hash: SHA-256 of the inactive slot `manifest.yaml`.
        Returns:
            int: Number of active jobs blocking rebuild/publish of this slot content.
        Assumptions:
            R8-03 blocking set is explicit: only `queued|running` rows with
            `execution_mode in ('background_auto', 'background_manual_legacy')` participate in
            inactive-slot publish guard, and saved/template requests store `(market_id, symbol)`
            in `request_json` or `spec_payload_json`.
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
      - docs/architecture/backtest/README.md
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


__all__ = [
    "BacktestJobLeaseRepository",
    "BacktestJobListPage",
    "BacktestJobListQuery",
    "BacktestJobRepository",
]
