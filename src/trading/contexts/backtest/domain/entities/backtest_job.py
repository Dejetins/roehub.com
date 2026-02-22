from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence, cast
from uuid import UUID

from trading.contexts.backtest.domain.errors import (
    BacktestJobLeaseError,
    BacktestJobTransitionError,
)
from trading.shared_kernel.primitives import UserId

BacktestJobMode = Literal["saved", "template"]
BacktestJobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
BacktestJobStage = Literal["stage_a", "stage_b", "finalizing"]

_ACTIVE_JOB_STATES: frozenset[str] = frozenset({"queued", "running"})
_TERMINAL_JOB_STATES: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})
_ALLOWED_JOB_STATE_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "cancelled"}),
    "running": frozenset({"succeeded", "failed", "cancelled"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}
_STAGE_ORDER: dict[str, int] = {
    "stage_a": 0,
    "stage_b": 1,
    "finalizing": 2,
}


@dataclass(frozen=True, slots=True)
class BacktestJobErrorPayload:
    """
    RoehubError-like payload persisted in `backtest_jobs.last_error_json`.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    Related:
      - src/trading/platform/errors/roehub_error.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
    """

    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate payload fields and normalize details into deterministic mapping.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `details` is JSON-compatible and must remain stable for persisted failures.
        Raises:
            BacktestJobTransitionError: If payload fields are blank or not serializable.
        Side Effects:
            Replaces `details` with immutable key-sorted mapping proxy.
        """
        normalized_code = self.code.strip()
        normalized_message = self.message.strip()
        if not normalized_code:
            raise BacktestJobTransitionError("BacktestJobErrorPayload.code must be non-empty")
        if not normalized_message:
            raise BacktestJobTransitionError("BacktestJobErrorPayload.message must be non-empty")

        normalized_details = _normalize_json_object(value=self.details)
        try:
            json.dumps(
                normalized_details,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        except TypeError as error:
            raise BacktestJobTransitionError(
                "BacktestJobErrorPayload.details must be JSON-serializable"
            ) from error

        object.__setattr__(self, "code", normalized_code)
        object.__setattr__(self, "message", normalized_message)
        object.__setattr__(self, "details", MappingProxyType(normalized_details))

    def to_mapping(self) -> Mapping[str, Any]:
        """
        Convert payload into deterministic mapping shape for JSONB persistence.

        Args:
            None.
        Returns:
            Mapping[str, Any]: Mapping with canonical `code/message/details` keys.
        Assumptions:
            Returned mapping is consumed by explicit SQL adapters as JSON payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class BacktestJob:
    """
    Immutable Backtest job aggregate with deterministic lifecycle and lease invariants.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/milestone-5-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - alembic/versions/20260222_0003_backtest_jobs_v1.py
    """

    job_id: UUID
    user_id: UserId
    mode: BacktestJobMode
    state: BacktestJobState
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    request_json: Mapping[str, Any] = field(default_factory=dict)
    request_hash: str = ""
    spec_hash: str | None = None
    spec_payload_json: Mapping[str, Any] | None = None
    engine_params_hash: str = ""
    backtest_runtime_config_hash: str = ""
    stage: BacktestJobStage = "stage_a"
    processed_units: int = 0
    total_units: int = 0
    progress_updated_at: datetime | None = None
    locked_by: str | None = None
    locked_at: datetime | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    attempt: int = 0
    last_error: str | None = None
    last_error_json: BacktestJobErrorPayload | None = None

    def __post_init__(self) -> None:
        """
        Validate lifecycle, stage, lease, and reproducibility invariants for persisted jobs.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            All timestamps are UTC-aware and hashes are canonical SHA-256 hex strings.
        Raises:
            BacktestJobTransitionError: If one invariant is violated.
        Side Effects:
            Normalizes textual fields and replaces mappings with immutable proxies.
        """
        if self.mode not in {"saved", "template"}:
            raise BacktestJobTransitionError(f"BacktestJob.mode is unsupported: {self.mode!r}")
        if self.state not in _ALLOWED_JOB_STATE_TRANSITIONS:
            raise BacktestJobTransitionError(f"BacktestJob.state is unsupported: {self.state!r}")
        if self.stage not in _STAGE_ORDER:
            raise BacktestJobTransitionError(f"BacktestJob.stage is unsupported: {self.stage!r}")

        if self.user_id is None:  # type: ignore[truthy-bool]
            raise BacktestJobTransitionError("BacktestJob.user_id is required")

        _ensure_utc_datetime(name="created_at", value=self.created_at)
        _ensure_utc_datetime(name="updated_at", value=self.updated_at)
        if self.updated_at < self.created_at:
            raise BacktestJobTransitionError("BacktestJob.updated_at cannot be before created_at")

        _ensure_optional_utc_datetime(name="started_at", value=self.started_at)
        _ensure_optional_utc_datetime(name="finished_at", value=self.finished_at)
        _ensure_optional_utc_datetime(
            name="cancel_requested_at",
            value=self.cancel_requested_at,
        )
        _ensure_optional_utc_datetime(
            name="progress_updated_at",
            value=self.progress_updated_at,
        )
        _ensure_optional_utc_datetime(name="locked_at", value=self.locked_at)
        _ensure_optional_utc_datetime(name="lease_expires_at", value=self.lease_expires_at)
        _ensure_optional_utc_datetime(name="heartbeat_at", value=self.heartbeat_at)

        if self.started_at is not None and self.started_at < self.created_at:
            raise BacktestJobTransitionError("BacktestJob.started_at cannot be before created_at")
        if self.finished_at is not None and self.started_at is not None:
            if self.finished_at < self.started_at:
                raise BacktestJobTransitionError(
                    "BacktestJob.finished_at cannot be before started_at"
                )
        if self.finished_at is not None and self.updated_at < self.finished_at:
            raise BacktestJobTransitionError("BacktestJob.updated_at cannot be before finished_at")

        if self.state in _TERMINAL_JOB_STATES and self.finished_at is None:
            raise BacktestJobTransitionError(
                "BacktestJob.finished_at must be set for terminal state"
            )
        if self.state in _ACTIVE_JOB_STATES and self.finished_at is not None:
            raise BacktestJobTransitionError(
                "BacktestJob.finished_at must be None for active state"
            )
        if self.state == "queued" and self.started_at is not None:
            raise BacktestJobTransitionError("BacktestJob.started_at must be None for queued state")
        if self.state == "running" and self.started_at is None:
            raise BacktestJobTransitionError("BacktestJob.started_at must be set for running state")

        if self.state == "running":
            if self.locked_by is None or not self.locked_by.strip():
                raise BacktestJobLeaseError("BacktestJob.locked_by must be set for running state")
            if self.locked_at is None:
                raise BacktestJobLeaseError("BacktestJob.locked_at must be set for running state")
            if self.lease_expires_at is None:
                raise BacktestJobLeaseError(
                    "BacktestJob.lease_expires_at must be set for running state"
                )
            if self.heartbeat_at is None:
                raise BacktestJobLeaseError(
                    "BacktestJob.heartbeat_at must be set for running state"
                )
            if self.lease_expires_at <= self.locked_at:
                raise BacktestJobLeaseError(
                    "BacktestJob.lease_expires_at must be after locked_at"
                )
        elif any(
            item is not None
            for item in (
                self.locked_by,
                self.locked_at,
                self.lease_expires_at,
                self.heartbeat_at,
            )
        ):
            raise BacktestJobLeaseError(
                "BacktestJob lease fields must be null outside running state"
            )

        if self.attempt < 0:
            raise BacktestJobTransitionError("BacktestJob.attempt must be >= 0")
        if self.processed_units < 0:
            raise BacktestJobTransitionError("BacktestJob.processed_units must be >= 0")
        if self.total_units < 0:
            raise BacktestJobTransitionError("BacktestJob.total_units must be >= 0")
        if self.total_units > 0 and self.processed_units > self.total_units:
            raise BacktestJobTransitionError(
                "BacktestJob.processed_units cannot exceed total_units"
            )

        normalized_request = _normalize_json_object(value=self.request_json)
        if len(normalized_request) == 0:
            raise BacktestJobTransitionError(
                "BacktestJob.request_json must be non-empty JSON object"
            )
        object.__setattr__(self, "request_json", MappingProxyType(normalized_request))

        if self.mode == "saved":
            if self.spec_hash is None or not self.spec_hash.strip():
                raise BacktestJobTransitionError("BacktestJob.spec_hash is required for saved mode")
            if self.spec_payload_json is None:
                raise BacktestJobTransitionError(
                    "BacktestJob.spec_payload_json is required for saved mode"
                )
            normalized_spec_payload = _normalize_json_object(value=self.spec_payload_json)
            if len(normalized_spec_payload) == 0:
                raise BacktestJobTransitionError(
                    "BacktestJob.spec_payload_json must be non-empty JSON object for saved mode"
                )
            object.__setattr__(
                self,
                "spec_payload_json",
                MappingProxyType(normalized_spec_payload),
            )
        else:
            if self.spec_hash is not None:
                raise BacktestJobTransitionError(
                    "BacktestJob.spec_hash must be None for template mode"
                )
            if self.spec_payload_json is not None:
                raise BacktestJobTransitionError(
                    "BacktestJob.spec_payload_json must be None for template mode"
                )

        normalized_mode = self.mode.strip().lower()
        normalized_state = self.state.strip().lower()
        normalized_stage = self.stage.strip().lower()
        object.__setattr__(self, "mode", cast(BacktestJobMode, normalized_mode))
        object.__setattr__(self, "state", cast(BacktestJobState, normalized_state))
        object.__setattr__(self, "stage", cast(BacktestJobStage, normalized_stage))

        _ensure_sha256_hex(name="request_hash", value=self.request_hash)
        _ensure_sha256_hex(name="engine_params_hash", value=self.engine_params_hash)
        _ensure_sha256_hex(
            name="backtest_runtime_config_hash",
            value=self.backtest_runtime_config_hash,
        )
        if self.spec_hash is not None:
            _ensure_sha256_hex(name="spec_hash", value=self.spec_hash)
            object.__setattr__(self, "spec_hash", self.spec_hash.strip().lower())

        if self.state == "failed":
            if self.last_error is None or not self.last_error.strip():
                raise BacktestJobTransitionError(
                    "BacktestJob.last_error must be set for failed state"
                )
            if self.last_error_json is None:
                raise BacktestJobTransitionError(
                    "BacktestJob.last_error_json must be set for failed state"
                )
            object.__setattr__(self, "last_error", self.last_error.strip())
        else:
            if self.last_error is not None:
                raise BacktestJobTransitionError(
                    "BacktestJob.last_error must be null outside failed state"
                )
            if self.last_error_json is not None:
                raise BacktestJobTransitionError(
                    "BacktestJob.last_error_json must be null outside failed state"
                )

        if self.locked_by is not None:
            object.__setattr__(self, "locked_by", self.locked_by.strip())

    @classmethod
    def create_queued(
        cls,
        *,
        job_id: UUID,
        user_id: UserId,
        mode: BacktestJobMode,
        created_at: datetime,
        request_json: Mapping[str, Any],
        request_hash: str,
        spec_hash: str | None,
        spec_payload_json: Mapping[str, Any] | None,
        engine_params_hash: str,
        backtest_runtime_config_hash: str,
    ) -> BacktestJob:
        """
        Build initial queued job snapshot with deterministic defaults.

        Args:
            job_id: Stable job identifier.
            user_id: Job owner identifier.
            mode: Job mode literal (`saved` or `template`).
            created_at: Creation timestamp in UTC.
            request_json: Canonical effective request payload.
            request_hash: Request payload SHA-256 hash.
            spec_hash: Saved strategy payload hash for saved mode.
            spec_payload_json: Saved strategy payload snapshot for saved mode.
            engine_params_hash: Effective execution settings hash.
            backtest_runtime_config_hash: Runtime result-affecting hash.
        Returns:
            BacktestJob: New queued job aggregate.
        Assumptions:
            Caller prepared canonical payload and hash values before persistence.
        Raises:
            BacktestJobTransitionError: If one invariant is invalid.
        Side Effects:
            None.
        """
        return cls(
            job_id=job_id,
            user_id=user_id,
            mode=mode,
            state="queued",
            created_at=created_at,
            updated_at=created_at,
            started_at=None,
            finished_at=None,
            cancel_requested_at=None,
            request_json=request_json,
            request_hash=request_hash,
            spec_hash=spec_hash,
            spec_payload_json=spec_payload_json,
            engine_params_hash=engine_params_hash,
            backtest_runtime_config_hash=backtest_runtime_config_hash,
            stage="stage_a",
            processed_units=0,
            total_units=0,
            progress_updated_at=None,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            heartbeat_at=None,
            attempt=0,
            last_error=None,
            last_error_json=None,
        )

    def is_active(self) -> bool:
        """
        Check whether job contributes to active-per-user quota (`queued` + `running`).

        Args:
            None.
        Returns:
            bool: `True` for active lifecycle states.
        Assumptions:
            Active set is fixed by Backtest Jobs v1 contract.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self.state in _ACTIVE_JOB_STATES

    def can_transition_to(self, *, next_state: BacktestJobState) -> bool:
        """
        Check whether state transition is allowed by deterministic lifecycle graph.

        Args:
            next_state: Target lifecycle state.
        Returns:
            bool: `True` when transition is valid.
        Assumptions:
            `queued -> failed` is forbidden by contract and absent in transition graph.
        Raises:
            None.
        Side Effects:
            None.
        """
        return next_state in _ALLOWED_JOB_STATE_TRANSITIONS[self.state]

    def claim(
        self,
        *,
        changed_at: datetime,
        locked_by: str,
        lease_expires_at: datetime,
    ) -> BacktestJob:
        """
        Claim queued or expired-running job and assign active lease owner.

        Args:
            changed_at: Claim timestamp in UTC.
            locked_by: Lease owner identity (`<hostname>-<pid>` style literal).
            lease_expires_at: Lease expiration timestamp in UTC.
        Returns:
            BacktestJob: Claimed running job snapshot.
        Assumptions:
            Reclaim path is represented as `running -> running` with incremented attempt.
        Raises:
            BacktestJobTransitionError: If state does not allow claim transition.
            BacktestJobLeaseError: If lease owner/timestamps are invalid.
        Side Effects:
            None.
        """
        normalized_locked_by = locked_by.strip()
        if not normalized_locked_by:
            raise BacktestJobLeaseError("BacktestJob.claim requires non-empty locked_by")

        _ensure_utc_datetime(name="changed_at", value=changed_at)
        _ensure_utc_datetime(name="lease_expires_at", value=lease_expires_at)
        if lease_expires_at <= changed_at:
            raise BacktestJobLeaseError(
                "BacktestJob.claim lease_expires_at must be after changed_at"
            )
        if changed_at < self.updated_at:
            raise BacktestJobTransitionError(
                "BacktestJob.claim changed_at cannot be before current updated_at"
            )
        if self.state not in {"queued", "running"}:
            raise BacktestJobTransitionError(
                f"BacktestJob.claim cannot claim state {self.state!r}"
            )

        started_at = self.started_at if self.started_at is not None else changed_at
        return BacktestJob(
            job_id=self.job_id,
            user_id=self.user_id,
            mode=self.mode,
            state="running",
            created_at=self.created_at,
            updated_at=changed_at,
            started_at=started_at,
            finished_at=None,
            cancel_requested_at=self.cancel_requested_at,
            request_json=self.request_json,
            request_hash=self.request_hash,
            spec_hash=self.spec_hash,
            spec_payload_json=self.spec_payload_json,
            engine_params_hash=self.engine_params_hash,
            backtest_runtime_config_hash=self.backtest_runtime_config_hash,
            stage=self.stage,
            processed_units=self.processed_units,
            total_units=self.total_units,
            progress_updated_at=self.progress_updated_at,
            locked_by=normalized_locked_by,
            locked_at=changed_at,
            lease_expires_at=lease_expires_at,
            heartbeat_at=changed_at,
            attempt=self.attempt + 1,
            last_error=None,
            last_error_json=None,
        )

    def renew_lease(
        self,
        *,
        changed_at: datetime,
        locked_by: str,
        lease_expires_at: datetime,
    ) -> BacktestJob:
        """
        Extend running job lease for the same owner.

        Args:
            changed_at: Heartbeat timestamp in UTC.
            locked_by: Expected active owner identifier.
            lease_expires_at: New lease expiration timestamp in UTC.
        Returns:
            BacktestJob: Updated running job snapshot.
        Assumptions:
            Heartbeat is accepted only when current owner still holds the lease.
        Raises:
            BacktestJobTransitionError: If state or timestamp ordering is invalid.
            BacktestJobLeaseError: If lease owner mismatches or new lease is invalid.
        Side Effects:
            None.
        """
        if self.state != "running":
            raise BacktestJobTransitionError(
                "BacktestJob.renew_lease is allowed only for running state"
            )
        if self.locked_by != locked_by.strip():
            raise BacktestJobLeaseError("BacktestJob.renew_lease locked_by mismatch")

        _ensure_utc_datetime(name="changed_at", value=changed_at)
        _ensure_utc_datetime(name="lease_expires_at", value=lease_expires_at)
        if changed_at < self.updated_at:
            raise BacktestJobTransitionError(
                "BacktestJob.renew_lease changed_at cannot be before current updated_at"
            )
        if lease_expires_at <= changed_at:
            raise BacktestJobLeaseError(
                "BacktestJob.renew_lease lease_expires_at must be after changed_at"
            )

        return BacktestJob(
            job_id=self.job_id,
            user_id=self.user_id,
            mode=self.mode,
            state=self.state,
            created_at=self.created_at,
            updated_at=changed_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            cancel_requested_at=self.cancel_requested_at,
            request_json=self.request_json,
            request_hash=self.request_hash,
            spec_hash=self.spec_hash,
            spec_payload_json=self.spec_payload_json,
            engine_params_hash=self.engine_params_hash,
            backtest_runtime_config_hash=self.backtest_runtime_config_hash,
            stage=self.stage,
            processed_units=self.processed_units,
            total_units=self.total_units,
            progress_updated_at=self.progress_updated_at,
            locked_by=self.locked_by,
            locked_at=self.locked_at,
            lease_expires_at=lease_expires_at,
            heartbeat_at=changed_at,
            attempt=self.attempt,
            last_error=self.last_error,
            last_error_json=self.last_error_json,
        )

    def update_progress(
        self,
        *,
        changed_at: datetime,
        stage: BacktestJobStage,
        processed_units: int,
        total_units: int,
    ) -> BacktestJob:
        """
        Update running progress counters with monotonic stage progression.

        Args:
            changed_at: Progress timestamp in UTC.
            stage: Progress stage literal.
            processed_units: Stage processed units counter.
            total_units: Stage total units counter.
        Returns:
            BacktestJob: Updated running job snapshot.
        Assumptions:
            Stage progression is monotonic (`stage_a -> stage_b -> finalizing`).
        Raises:
            BacktestJobTransitionError: If state, stage ordering, or counters are invalid.
        Side Effects:
            None.
        """
        if self.state != "running":
            raise BacktestJobTransitionError(
                "BacktestJob.update_progress is allowed only for running state"
            )
        if stage not in _STAGE_ORDER:
            raise BacktestJobTransitionError(f"BacktestJob.stage is unsupported: {stage!r}")

        _ensure_utc_datetime(name="changed_at", value=changed_at)
        if changed_at < self.updated_at:
            raise BacktestJobTransitionError(
                "BacktestJob.update_progress changed_at cannot be before current updated_at"
            )
        if processed_units < 0:
            raise BacktestJobTransitionError(
                "BacktestJob.update_progress processed_units must be >= 0"
            )
        if total_units < 0:
            raise BacktestJobTransitionError(
                "BacktestJob.update_progress total_units must be >= 0"
            )
        if total_units > 0 and processed_units > total_units:
            raise BacktestJobTransitionError(
                "BacktestJob.update_progress processed_units cannot exceed total_units"
            )

        if _STAGE_ORDER[stage] < _STAGE_ORDER[self.stage]:
            raise BacktestJobTransitionError(
                f"BacktestJob.update_progress cannot move stage backward: {self.stage} -> {stage}"
            )

        return BacktestJob(
            job_id=self.job_id,
            user_id=self.user_id,
            mode=self.mode,
            state=self.state,
            created_at=self.created_at,
            updated_at=changed_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            cancel_requested_at=self.cancel_requested_at,
            request_json=self.request_json,
            request_hash=self.request_hash,
            spec_hash=self.spec_hash,
            spec_payload_json=self.spec_payload_json,
            engine_params_hash=self.engine_params_hash,
            backtest_runtime_config_hash=self.backtest_runtime_config_hash,
            stage=stage,
            processed_units=processed_units,
            total_units=total_units,
            progress_updated_at=changed_at,
            locked_by=self.locked_by,
            locked_at=self.locked_at,
            lease_expires_at=self.lease_expires_at,
            heartbeat_at=self.heartbeat_at,
            attempt=self.attempt,
            last_error=None,
            last_error_json=None,
        )

    def request_cancel(self, *, changed_at: datetime) -> BacktestJob:
        """
        Apply cancel intent according to current state (`queued` immediate, `running` deferred).

        Args:
            changed_at: Cancel-request timestamp in UTC.
        Returns:
            BacktestJob: Updated snapshot.
        Assumptions:
            Cancel operation is idempotent for terminal states.
        Raises:
            BacktestJobTransitionError: If timestamp ordering is invalid.
        Side Effects:
            None.
        """
        _ensure_utc_datetime(name="changed_at", value=changed_at)
        if changed_at < self.updated_at:
            raise BacktestJobTransitionError(
                "BacktestJob.request_cancel changed_at cannot be before current updated_at"
            )

        if self.state == "queued":
            return BacktestJob(
                job_id=self.job_id,
                user_id=self.user_id,
                mode=self.mode,
                state="cancelled",
                created_at=self.created_at,
                updated_at=changed_at,
                started_at=None,
                finished_at=changed_at,
                cancel_requested_at=changed_at,
                request_json=self.request_json,
                request_hash=self.request_hash,
                spec_hash=self.spec_hash,
                spec_payload_json=self.spec_payload_json,
                engine_params_hash=self.engine_params_hash,
                backtest_runtime_config_hash=self.backtest_runtime_config_hash,
                stage=self.stage,
                processed_units=self.processed_units,
                total_units=self.total_units,
                progress_updated_at=self.progress_updated_at,
                locked_by=None,
                locked_at=None,
                lease_expires_at=None,
                heartbeat_at=None,
                attempt=self.attempt,
                last_error=None,
                last_error_json=None,
            )

        if self.state == "running":
            return BacktestJob(
                job_id=self.job_id,
                user_id=self.user_id,
                mode=self.mode,
                state=self.state,
                created_at=self.created_at,
                updated_at=changed_at,
                started_at=self.started_at,
                finished_at=self.finished_at,
                cancel_requested_at=changed_at,
                request_json=self.request_json,
                request_hash=self.request_hash,
                spec_hash=self.spec_hash,
                spec_payload_json=self.spec_payload_json,
                engine_params_hash=self.engine_params_hash,
                backtest_runtime_config_hash=self.backtest_runtime_config_hash,
                stage=self.stage,
                processed_units=self.processed_units,
                total_units=self.total_units,
                progress_updated_at=self.progress_updated_at,
                locked_by=self.locked_by,
                locked_at=self.locked_at,
                lease_expires_at=self.lease_expires_at,
                heartbeat_at=self.heartbeat_at,
                attempt=self.attempt,
                last_error=None,
                last_error_json=None,
            )

        return self

    def finish(
        self,
        *,
        next_state: BacktestJobState,
        changed_at: datetime,
        last_error: str | None = None,
        last_error_json: BacktestJobErrorPayload | None = None,
    ) -> BacktestJob:
        """
        Transition running job to terminal state with deterministic failure payload rules.

        Args:
            next_state: Target terminal state (`succeeded`, `failed`, or `cancelled`).
            changed_at: Terminal transition timestamp in UTC.
            last_error: Short failure message for `failed` state.
            last_error_json: RoehubError-like JSON payload for `failed` state.
        Returns:
            BacktestJob: Terminal job snapshot.
        Assumptions:
            `queued -> failed` transition is forbidden by lifecycle contract.
        Raises:
            BacktestJobTransitionError: If transition or payload invariants are invalid.
        Side Effects:
            None.
        """
        if next_state not in _TERMINAL_JOB_STATES:
            raise BacktestJobTransitionError(
                f"BacktestJob.finish requires terminal next_state, got {next_state!r}"
            )
        if not self.can_transition_to(next_state=next_state):
            raise BacktestJobTransitionError(
                f"BacktestJob invalid transition {self.state!r} -> {next_state!r}"
            )

        _ensure_utc_datetime(name="changed_at", value=changed_at)
        if changed_at < self.updated_at:
            raise BacktestJobTransitionError(
                "BacktestJob.finish changed_at cannot be before current updated_at"
            )

        normalized_last_error: str | None = None
        normalized_last_error_json: BacktestJobErrorPayload | None = None
        if next_state == "failed":
            if last_error is None or not last_error.strip():
                raise BacktestJobTransitionError(
                    "BacktestJob.finish failed transition requires last_error"
                )
            if last_error_json is None:
                raise BacktestJobTransitionError(
                    "BacktestJob.finish failed transition requires last_error_json"
                )
            normalized_last_error = last_error.strip()
            normalized_last_error_json = last_error_json

        next_stage: BacktestJobStage = self.stage
        if next_state == "succeeded":
            next_stage = "finalizing"

        return BacktestJob(
            job_id=self.job_id,
            user_id=self.user_id,
            mode=self.mode,
            state=cast(BacktestJobState, next_state),
            created_at=self.created_at,
            updated_at=changed_at,
            started_at=self.started_at,
            finished_at=changed_at,
            cancel_requested_at=self.cancel_requested_at,
            request_json=self.request_json,
            request_hash=self.request_hash,
            spec_hash=self.spec_hash,
            spec_payload_json=self.spec_payload_json,
            engine_params_hash=self.engine_params_hash,
            backtest_runtime_config_hash=self.backtest_runtime_config_hash,
            stage=next_stage,
            processed_units=self.processed_units,
            total_units=self.total_units,
            progress_updated_at=self.progress_updated_at,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            heartbeat_at=None,
            attempt=self.attempt,
            last_error=normalized_last_error,
            last_error_json=normalized_last_error_json,
        )



def is_backtest_job_state_active(*, state: BacktestJobState) -> bool:
    """
    Check whether job state is active in Backtest Jobs v1 lifecycle.

    Args:
        state: Job state literal.
    Returns:
        bool: `True` for `queued` and `running` states.
    Assumptions:
        Active states are fixed by active-jobs quota and worker claim contracts.
    Raises:
        None.
    Side Effects:
        None.
    """
    return state in _ACTIVE_JOB_STATES



def is_backtest_job_state_terminal(*, state: BacktestJobState) -> bool:
    """
    Check whether job state is terminal in Backtest Jobs v1 lifecycle.

    Args:
        state: Job state literal.
    Returns:
        bool: `True` for `succeeded`, `failed`, or `cancelled`.
    Assumptions:
        Terminal states cannot transition further.
    Raises:
        None.
    Side Effects:
        None.
    """
    return state in _TERMINAL_JOB_STATES



def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    """
    Validate timezone-aware UTC datetime field.

    Args:
        name: Field name for deterministic error messages.
        value: Datetime value to validate.
    Returns:
        None.
    Assumptions:
        Persisted timestamps in jobs storage are UTC.
    Raises:
        BacktestJobTransitionError: If datetime is naive or non-UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise BacktestJobTransitionError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise BacktestJobTransitionError(f"{name} must be UTC datetime")



def _ensure_optional_utc_datetime(*, name: str, value: datetime | None) -> None:
    """
    Validate optional datetime value when provided.

    Args:
        name: Field name for deterministic error messages.
        value: Optional datetime value.
    Returns:
        None.
    Assumptions:
        Missing value means optional field is intentionally unset.
    Raises:
        BacktestJobTransitionError: If provided datetime is naive or non-UTC.
    Side Effects:
        None.
    """
    if value is None:
        return
    _ensure_utc_datetime(name=name, value=value)



def _ensure_sha256_hex(*, name: str, value: str) -> None:
    """
    Validate SHA-256 hex string format used by reproducibility hashes.

    Args:
        name: Field name for deterministic error messages.
        value: Raw hash value.
    Returns:
        None.
    Assumptions:
        Hashes use lowercase 64-char hexadecimal representation.
    Raises:
        BacktestJobTransitionError: If hash format is invalid.
    Side Effects:
        None.
    """
    normalized = value.strip().lower()
    if len(normalized) != 64:
        raise BacktestJobTransitionError(f"{name} must be 64 lowercase hex chars")
    allowed = set("0123456789abcdef")
    if any(char not in allowed for char in normalized):
        raise BacktestJobTransitionError(f"{name} must be 64 lowercase hex chars")



def _normalize_json_object(*, value: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize mapping payload into deterministic JSON-compatible object.

    Args:
        value: Raw mapping payload.
    Returns:
        dict[str, Any]: Key-sorted normalized object payload.
    Assumptions:
        Mapping keys can be converted to strings without information loss.
    Raises:
        BacktestJobTransitionError: If normalized payload is not JSON object.
    Side Effects:
        None.
    """
    normalized = _normalize_json_value(value=dict(value))
    if not isinstance(normalized, Mapping):
        raise BacktestJobTransitionError("Expected JSON object payload")
    return dict(normalized)



def _normalize_json_value(*, value: Any) -> Any:
    """
    Normalize arbitrary JSON-like value into deterministic structure.

    Args:
        value: Raw JSON-like node.
    Returns:
        Any: Deterministic mapping/list/scalar value.
    Assumptions:
        Unknown non-JSON objects are stringified for deterministic persistence.
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

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_value(value=item) for item in value]

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


__all__ = [
    "BacktestJob",
    "BacktestJobErrorPayload",
    "BacktestJobMode",
    "BacktestJobStage",
    "BacktestJobState",
    "is_backtest_job_state_active",
    "is_backtest_job_state_terminal",
]
