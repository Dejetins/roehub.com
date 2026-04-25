from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from math import ceil
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
BacktestArtifactSlotLiteral = Literal["slot_a", "slot_b"]
BacktestJobExecutionMode = Literal[
    "sync_inline",
    "background_auto",
    "background_manual_legacy",
]

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
_ALLOWED_ARTIFACT_SLOTS: frozenset[str] = frozenset({"slot_a", "slot_b"})
_ALLOWED_EXECUTION_MODES: frozenset[str] = frozenset(
    {"sync_inline", "background_auto", "background_manual_legacy"}
)
_BACKGROUND_JOB_EXECUTION_MODES: frozenset[str] = frozenset(
    {"background_auto", "background_manual_legacy"}
)
_ALLOWED_RANKING_METRICS: frozenset[str] = frozenset(
    {
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "sharpe_trades",
        "win_rate_pct",
    }
)


@dataclass(frozen=True, slots=True)
class BacktestJobErrorPayload:
    """
    RoehubError-like payload persisted in `backtest_jobs.last_error_json`.

    Docs:
      - docs/architecture/backtest/README.md
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
class BacktestJobArtifactPin:
    """
    Immutable artifact-slot identity pinned to one queued/running Backtest job attempt.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - alembic/versions/20260326_0004_backtest_job_artifact_pin_v1.py
    """

    artifact_slot: BacktestArtifactSlotLiteral
    artifact_slot_generation: int
    artifact_manifest_hash: str
    artifact_asof_date: str

    def __post_init__(self) -> None:
        """
        Validate strict artifact-slot pin metadata persisted for reproducible background runs.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Pin metadata is captured once at job creation and must remain immutable afterward.
        Raises:
            BacktestJobTransitionError: If slot, generation, hash, or date literal is invalid.
        Side Effects:
            Normalizes slot/hash literals to canonical lowercase forms.
        """
        if not isinstance(self.artifact_slot, str):
            raise BacktestJobTransitionError(
                "BacktestJobArtifactPin.artifact_slot must be 'slot_a' or 'slot_b'"
            )
        if not isinstance(self.artifact_manifest_hash, str):
            raise BacktestJobTransitionError(
                "BacktestJobArtifactPin.artifact_manifest_hash must be 64 lowercase hex chars"
            )
        if not isinstance(self.artifact_asof_date, str):
            raise BacktestJobTransitionError(
                "BacktestJobArtifactPin.artifact_asof_date must be YYYY-MM-DD"
            )
        normalized_slot = self.artifact_slot.strip().lower()
        if normalized_slot not in _ALLOWED_ARTIFACT_SLOTS:
            raise BacktestJobTransitionError(
                "BacktestJobArtifactPin.artifact_slot must be 'slot_a' or 'slot_b'"
            )
        if self.artifact_slot_generation <= 0:
            raise BacktestJobTransitionError(
                "BacktestJobArtifactPin.artifact_slot_generation must be > 0"
            )
        _ensure_sha256_hex(
            name="artifact_manifest_hash",
            value=self.artifact_manifest_hash,
        )
        _ensure_strict_date_literal(
            name="artifact_asof_date",
            value=self.artifact_asof_date,
        )
        object.__setattr__(
            self,
            "artifact_slot",
            cast(BacktestArtifactSlotLiteral, normalized_slot),
        )
        object.__setattr__(
            self,
            "artifact_manifest_hash",
            self.artifact_manifest_hash.strip().lower(),
        )


@dataclass(frozen=True, slots=True)
class BacktestJobStageWeights:
    """
    Deterministic `stage_a/stage_b/finalizing` weights for persisted run progress read models.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
      - apps/api/dto/backtest_runs.py
    """

    stage_a: int
    stage_b: int
    finalizing: int

    def __post_init__(self) -> None:
        """
        Validate deterministic stage-weight invariants for `0..100%` progress projection.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Public progress percent is expressed in integer percent points summing to `100`.
        Raises:
            BacktestJobTransitionError: If one weight is non-positive or the total is not `100`.
        Side Effects:
            None.
        """
        for field_name, field_value in (
            ("stage_a", self.stage_a),
            ("stage_b", self.stage_b),
            ("finalizing", self.finalizing),
        ):
            if isinstance(field_value, bool) or not isinstance(field_value, int):
                raise BacktestJobTransitionError(
                    f"BacktestJobStageWeights.{field_name} must be integer"
                )
            if field_value <= 0:
                raise BacktestJobTransitionError(
                    f"BacktestJobStageWeights.{field_name} must be > 0"
                )
        if self.stage_a + self.stage_b + self.finalizing != 100:
            raise BacktestJobTransitionError(
                "BacktestJobStageWeights must sum to 100"
            )

    def completed_weight_before(self, *, stage: BacktestJobStage) -> int:
        """
        Return the cumulative completed weight before the requested stage starts.

        Args:
            stage: Target progress stage literal.
        Returns:
            int: Deterministic completed weight in percent points before `stage`.
        Assumptions:
            Stage ordering is fixed to `stage_a -> stage_b -> finalizing`.
        Raises:
            BacktestJobTransitionError: If stage literal is unsupported.
        Side Effects:
            None.
        """
        if stage == "stage_a":
            return 0
        if stage == "stage_b":
            return self.stage_a
        if stage == "finalizing":
            return self.stage_a + self.stage_b
        raise BacktestJobTransitionError(
            f"BacktestJobStageWeights stage is unsupported: {stage!r}"
        )

    def current_stage_weight(self, *, stage: BacktestJobStage) -> int:
        """
        Return the deterministic weight assigned to one active progress stage.

        Args:
            stage: Target progress stage literal.
        Returns:
            int: Weight in percent points for the current stage.
        Assumptions:
            Stage literals reuse the persisted run lifecycle vocabulary.
        Raises:
            BacktestJobTransitionError: If stage literal is unsupported.
        Side Effects:
            None.
        """
        if stage == "stage_a":
            return self.stage_a
        if stage == "stage_b":
            return self.stage_b
        if stage == "finalizing":
            return self.finalizing
        raise BacktestJobTransitionError(
            f"BacktestJobStageWeights stage is unsupported: {stage!r}"
        )


@dataclass(frozen=True, slots=True)
class BacktestJob:
    """
    Immutable Backtest job aggregate with deterministic lifecycle and lease invariants.

    Docs:
      - docs/architecture/backtest/README.md
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
    artifact_pin: BacktestJobArtifactPin | None = None
    execution_mode: BacktestJobExecutionMode | None = None
    execution_profile_mode_hint: str | None = None
    effective_execution_profile_mode: str | None = None
    market_id: int | None = None
    symbol: str | None = None
    timeframe: str | None = None
    requested_top_n: int | None = None
    ranking_primary_metric: str | None = None
    ranking_secondary_metric: str | None = None
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

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
          - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
            backtest_job_repository.py
          - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
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
        if self.artifact_pin is not None:
            object.__setattr__(
                self,
                "artifact_pin",
                BacktestJobArtifactPin(
                    artifact_slot=self.artifact_pin.artifact_slot,
                    artifact_slot_generation=self.artifact_pin.artifact_slot_generation,
                    artifact_manifest_hash=self.artifact_pin.artifact_manifest_hash,
                    artifact_asof_date=self.artifact_pin.artifact_asof_date,
                ),
            )
        if self.spec_hash is not None:
            _ensure_sha256_hex(name="spec_hash", value=self.spec_hash)
            object.__setattr__(self, "spec_hash", self.spec_hash.strip().lower())

        has_persisted_run_metadata = any(
            item is not None
            for item in (
                self.execution_mode,
                self.market_id,
                self.symbol,
                self.timeframe,
                self.requested_top_n,
                self.ranking_primary_metric,
                self.ranking_secondary_metric,
            )
        )
        has_execution_profile_metadata = any(
            item is not None
            for item in (
                self.execution_profile_mode_hint,
                self.effective_execution_profile_mode,
            )
        )
        if has_execution_profile_metadata and not has_persisted_run_metadata:
            raise BacktestJobTransitionError(
                "BacktestJob execution-profile metadata requires persisted run metadata"
            )
        if has_persisted_run_metadata:
            if (
                self.execution_mode is None
                or self.market_id is None
                or self.symbol is None
                or self.timeframe is None
                or self.requested_top_n is None
                or self.ranking_primary_metric is None
            ):
                raise BacktestJobTransitionError(
                    "BacktestJob persisted run metadata must be all set except "
                    "ranking_secondary_metric"
                )
            normalized_execution_mode = self.execution_mode.strip().lower()
            if normalized_execution_mode not in _ALLOWED_EXECUTION_MODES:
                raise BacktestJobTransitionError(
                    "BacktestJob.execution_mode must be one of "
                    f"{sorted(_ALLOWED_EXECUTION_MODES)}"
                )
            if isinstance(self.market_id, bool) or not isinstance(self.market_id, int):
                raise BacktestJobTransitionError("BacktestJob.market_id must be integer")
            if self.market_id <= 0:
                raise BacktestJobTransitionError("BacktestJob.market_id must be > 0")
            normalized_symbol = self.symbol.strip().upper()
            if not normalized_symbol:
                raise BacktestJobTransitionError("BacktestJob.symbol must be non-empty")
            normalized_timeframe = self.timeframe.strip().lower()
            if not normalized_timeframe:
                raise BacktestJobTransitionError("BacktestJob.timeframe must be non-empty")
            if isinstance(self.requested_top_n, bool) or not isinstance(self.requested_top_n, int):
                raise BacktestJobTransitionError("BacktestJob.requested_top_n must be integer")
            if self.requested_top_n <= 0:
                raise BacktestJobTransitionError("BacktestJob.requested_top_n must be > 0")
            normalized_primary_metric = _normalize_ranking_metric_literal(
                name="ranking_primary_metric",
                value=self.ranking_primary_metric,
            )
            normalized_secondary_metric = None
            if self.ranking_secondary_metric is not None:
                normalized_secondary_metric = _normalize_ranking_metric_literal(
                    name="ranking_secondary_metric",
                    value=self.ranking_secondary_metric,
                )
                if normalized_secondary_metric == normalized_primary_metric:
                    raise BacktestJobTransitionError(
                        "BacktestJob.ranking_secondary_metric must differ from "
                        "ranking_primary_metric"
                    )
            normalized_execution_profile_mode_hint = _normalize_optional_profile_mode_literal(
                name="execution_profile_mode_hint",
                value=self.execution_profile_mode_hint,
            )
            normalized_effective_execution_profile_mode = (
                _normalize_optional_profile_mode_literal(
                    name="effective_execution_profile_mode",
                    value=self.effective_execution_profile_mode,
                )
            )
            object.__setattr__(
                self,
                "execution_mode",
                cast(BacktestJobExecutionMode, normalized_execution_mode),
            )
            object.__setattr__(
                self,
                "execution_profile_mode_hint",
                normalized_execution_profile_mode_hint,
            )
            object.__setattr__(
                self,
                "effective_execution_profile_mode",
                normalized_effective_execution_profile_mode,
            )
            object.__setattr__(self, "market_id", self.market_id)
            object.__setattr__(self, "symbol", normalized_symbol)
            object.__setattr__(self, "timeframe", normalized_timeframe)
            object.__setattr__(self, "requested_top_n", self.requested_top_n)
            object.__setattr__(self, "ranking_primary_metric", normalized_primary_metric)
            object.__setattr__(self, "ranking_secondary_metric", normalized_secondary_metric)

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
        artifact_pin: BacktestJobArtifactPin | None = None,
        execution_mode: BacktestJobExecutionMode | None = None,
        execution_profile_mode_hint: str | None = None,
        effective_execution_profile_mode: str | None = None,
        market_id: int | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        requested_top_n: int | None = None,
        ranking_primary_metric: str | None = None,
        ranking_secondary_metric: str | None = None,
    ) -> BacktestJob:
        """
        Build initial queued job snapshot with deterministic defaults.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
          - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
            backtest_job_repository.py
          - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
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
            artifact_pin: Optional strict artifact-slot identity pinned at job creation time.
            execution_mode: Optional persisted-run execution mode literal for R7-01 storage.
            execution_profile_mode_hint:
                Optional launch-time metadata hint persisted outside `request_json`.
            effective_execution_profile_mode:
                Optional read-model execution-profile metadata persisted outside `request_json`.
            market_id: Optional denormalized instrument market id for history reads.
            symbol: Optional denormalized instrument symbol for history reads.
            timeframe: Optional denormalized timeframe literal for history reads.
            requested_top_n: Optional persisted summary rows request cap.
            ranking_primary_metric: Optional persisted primary ranking metric literal.
            ranking_secondary_metric: Optional persisted secondary ranking metric literal.
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
            artifact_pin=artifact_pin,
            execution_mode=execution_mode,
            execution_profile_mode_hint=execution_profile_mode_hint,
            effective_execution_profile_mode=effective_execution_profile_mode,
            market_id=market_id,
            symbol=symbol,
            timeframe=timeframe,
            requested_top_n=requested_top_n,
            ranking_primary_metric=ranking_primary_metric,
            ranking_secondary_metric=ranking_secondary_metric,
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

    def stage_progress_ratio(self) -> float:
        """
        Return the normalized completion ratio of the currently persisted stage counters.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/domain/entities/backtest_job.py
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - apps/api/dto/backtest_runs.py
        Args:
            None.
        Returns:
            float: Clamped ratio in `[0.0, 1.0]` for `processed_units / total_units`.
        Assumptions:
            Missing or zero `total_units` means current-stage completion is not observable yet.
        Raises:
            None.
        Side Effects:
            None.
        """
        if self.total_units <= 0:
            return 0.0
        return min(max(self.processed_units / self.total_units, 0.0), 1.0)

    def progress_percent(
        self,
        *,
        stage_weights: BacktestJobStageWeights,
    ) -> int:
        """
        Project persisted run counters onto a deterministic weighted `0..100%` progress scale.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/domain/entities/backtest_job.py
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - apps/api/dto/backtest_runs.py
        Args:
            stage_weights: Deterministic stage weights for the effective execution profile.
        Returns:
            int: Weighted integer progress percent in `[0, 100]`.
        Assumptions:
            Successful terminal runs are rendered as complete even if finalizing counters remain
            at `0/1` because storage persists the last in-flight snapshot before finishing.
        Raises:
            None.
        Side Effects:
            None.
        """
        if self.state == "succeeded":
            return 100
        weighted_value = stage_weights.completed_weight_before(
            stage=self.stage
        ) + (
            stage_weights.current_stage_weight(stage=self.stage) * self.stage_progress_ratio()
        )
        return int(round(min(max(weighted_value, 0.0), 100.0)))

    def eta_seconds(
        self,
        *,
        stage_weights: BacktestJobStageWeights,
        now: datetime,
    ) -> int | None:
        """
        Estimate remaining runtime from current-run progress only, or return `None`.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/domain/entities/backtest_job.py
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - apps/api/dto/backtest_runs.py
        Args:
            stage_weights: Deterministic stage weights for the effective execution profile.
            now: Read-model reference timestamp in UTC.
        Returns:
            int | None: Conservative remaining seconds estimate, or `None` when the signal is
                not defensible yet.
        Assumptions:
            ETA is published only for active running jobs with non-zero weighted progress derived
            from the current run itself; benchmark-history fallbacks are out of scope here.
        Raises:
            BacktestJobTransitionError: If `now` is not a UTC-aware timestamp.
        Side Effects:
            None.
        """
        _ensure_utc_datetime(name="now", value=now)
        if self.state != "running" or self.started_at is None or self.progress_updated_at is None:
            return None

        progress_percent = self.progress_percent(stage_weights=stage_weights)
        if progress_percent <= 0 or progress_percent >= 100:
            return None

        reference_time = now
        if self.progress_updated_at > reference_time:
            reference_time = self.progress_updated_at
        if reference_time <= self.started_at:
            return None

        elapsed_seconds = (reference_time - self.started_at).total_seconds()
        if elapsed_seconds < 1.0:
            return None

        remaining_seconds = elapsed_seconds * ((100 - progress_percent) / progress_percent)
        if remaining_seconds <= 0.0:
            return None
        return max(1, ceil(remaining_seconds))

    def claim(
        self,
        *,
        changed_at: datetime,
        locked_by: str,
        lease_expires_at: datetime,
    ) -> BacktestJob:
        """
        Claim queued or expired-running job and assign active lease owner.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
            backtest_job_lease_repository.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
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
        return replace(
            self,
            state="running",
            updated_at=changed_at,
            started_at=started_at,
            finished_at=None,
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

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
            backtest_job_lease_repository.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
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

        return replace(
            self,
            updated_at=changed_at,
            lease_expires_at=lease_expires_at,
            heartbeat_at=changed_at,
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

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
            backtest_job_lease_repository.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
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

        return replace(
            self,
            updated_at=changed_at,
            stage=stage,
            processed_units=processed_units,
            total_units=total_units,
            progress_updated_at=changed_at,
            last_error=None,
            last_error_json=None,
        )

    def request_cancel(self, *, changed_at: datetime) -> BacktestJob:
        """
        Apply cancel intent according to current state (`queued` immediate, `running` deferred).

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
            backtest_job_repository.py
          - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
        Args:
            changed_at: Cancel-request timestamp in UTC.
        Returns:
            BacktestJob: Updated snapshot.
        Assumptions:
            Running cancel remains best-effort, but the first `cancel_requested_at` marker is
            preserved for deterministic public lifecycle visibility and slot-safety checks.
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
            return replace(
                self,
                state="cancelled",
                updated_at=changed_at,
                started_at=None,
                finished_at=changed_at,
                cancel_requested_at=changed_at,
                locked_by=None,
                locked_at=None,
                lease_expires_at=None,
                heartbeat_at=None,
                last_error=None,
                last_error_json=None,
            )

        if self.state == "running":
            if self.cancel_requested_at is not None:
                return self
            return replace(
                self,
                updated_at=changed_at,
                cancel_requested_at=changed_at,
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

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
            backtest_job_lease_repository.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
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

        return replace(
            self,
            state=cast(BacktestJobState, next_state),
            updated_at=changed_at,
            finished_at=changed_at,
            stage=next_stage,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            heartbeat_at=None,
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


def _ensure_strict_date_literal(*, name: str, value: str) -> None:
    """
    Validate one strict `YYYY-MM-DD` literal used in persisted artifact pin metadata.

    Args:
        name: Field name for deterministic error messages.
        value: Candidate date literal.
    Returns:
        None.
    Assumptions:
        Persisted artifact pin metadata keeps date literals in canonical string form.
    Raises:
        BacktestJobTransitionError: If the date literal is blank or not valid ISO date.
    Side Effects:
        None.
    """
    normalized = value.strip()
    if len(normalized) != 10:
        raise BacktestJobTransitionError(f"{name} must be YYYY-MM-DD")
    try:
        date.fromisoformat(normalized)
    except ValueError as error:
        raise BacktestJobTransitionError(f"{name} must be YYYY-MM-DD") from error


def _normalize_ranking_metric_literal(*, name: str, value: str) -> str:
    """
    Normalize one persisted ranking metric literal against the fixed R6-04/R7-01 set.

    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
    Args:
        name: Field name used in deterministic error messages.
        value: Raw ranking metric literal.
    Returns:
        str: Lowercase normalized metric literal.
    Assumptions:
        Allowed literals are fixed by the approved ranking contract and stay additive-only here.
    Raises:
        BacktestJobTransitionError: If literal is blank or unsupported.
    Side Effects:
        None.
    """
    normalized = value.strip().lower()
    if not normalized:
        raise BacktestJobTransitionError(f"BacktestJob.{name} must be non-empty")
    if normalized not in _ALLOWED_RANKING_METRICS:
        raise BacktestJobTransitionError(
            f"BacktestJob.{name} must be one of {sorted(_ALLOWED_RANKING_METRICS)}"
        )
    return normalized


def _normalize_optional_profile_mode_literal(*, name: str, value: str | None) -> str | None:
    """
    Normalize one optional persisted execution-profile metadata literal.

    Args:
        name: Field name used in deterministic error messages.
        value: Optional raw execution-profile metadata literal.
    Returns:
        str | None: Lowercase normalized literal, or `None`.
    Assumptions:
        Validation of allowed execution-profile values happens in application/storage layers; the
        aggregate only enforces explicit non-empty normalized string semantics.
    Raises:
        BacktestJobTransitionError: If the provided literal is blank or not a string.
    Side Effects:
        None.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise BacktestJobTransitionError(f"BacktestJob.{name} must be string when set")
    normalized = value.strip().lower()
    if not normalized:
        raise BacktestJobTransitionError(f"BacktestJob.{name} must be non-empty when set")
    return normalized


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
    "BacktestArtifactSlotLiteral",
    "BacktestJob",
    "BacktestJobArtifactPin",
    "BacktestJobExecutionMode",
    "BacktestJobErrorPayload",
    "BacktestJobMode",
    "BacktestJobStage",
    "BacktestJobState",
    "is_backtest_job_state_active",
    "is_backtest_job_state_terminal",
]
