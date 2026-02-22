from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import UUID

from trading.contexts.backtest.domain.entities.backtest_job import BacktestJobState
from trading.contexts.backtest.domain.errors import BacktestJobTransitionError


@dataclass(frozen=True, slots=True)
class BacktestJobTopVariant:
    """
    Persisted ranked top-variant snapshot row for one Backtest job.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
      - alembic/versions/20260222_0003_backtest_jobs_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    job_id: UUID
    rank: int
    variant_key: str
    indicator_variant_key: str
    variant_index: int
    total_return_pct: float
    payload_json: Mapping[str, Any]
    report_table_md: str | None
    trades_json: tuple[Mapping[str, Any], ...] | None
    updated_at: datetime

    def __post_init__(self) -> None:
        """
        Validate top-variant row shape and normalize JSON payloads.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Variant keys are canonical lowercase SHA-256 hex literals.
        Raises:
            BacktestJobTransitionError: If one field violates storage contract.
        Side Effects:
            Replaces payload and trades items with immutable normalized structures.
        """
        if self.rank <= 0:
            raise BacktestJobTransitionError("BacktestJobTopVariant.rank must be > 0")
        if self.variant_index < 0:
            raise BacktestJobTransitionError("BacktestJobTopVariant.variant_index must be >= 0")

        normalized_variant_key = self.variant_key.strip().lower()
        normalized_indicator_key = self.indicator_variant_key.strip().lower()
        _ensure_sha256_key(name="variant_key", value=normalized_variant_key)
        _ensure_sha256_key(name="indicator_variant_key", value=normalized_indicator_key)

        if isinstance(self.total_return_pct, bool) or not isinstance(
            self.total_return_pct,
            int | float,
        ):
            raise BacktestJobTransitionError(
                "BacktestJobTopVariant.total_return_pct must be numeric"
            )

        payload = _normalize_json_object(value=self.payload_json)
        trades_payload: tuple[Mapping[str, Any], ...] | None = None
        if self.trades_json is not None:
            normalized_trades: list[Mapping[str, Any]] = []
            for item in self.trades_json:
                normalized_trades.append(MappingProxyType(_normalize_json_object(value=item)))
            trades_payload = tuple(normalized_trades)

        _ensure_utc_datetime(name="updated_at", value=self.updated_at)

        normalized_report = self.report_table_md
        if normalized_report is not None:
            normalized_report = normalized_report.strip()
            if not normalized_report:
                raise BacktestJobTransitionError(
                    "BacktestJobTopVariant.report_table_md cannot be blank"
                )

        object.__setattr__(self, "variant_key", normalized_variant_key)
        object.__setattr__(self, "indicator_variant_key", normalized_indicator_key)
        object.__setattr__(self, "total_return_pct", float(self.total_return_pct))
        object.__setattr__(self, "payload_json", MappingProxyType(payload))
        object.__setattr__(self, "trades_json", trades_payload)
        object.__setattr__(self, "report_table_md", normalized_report)


@dataclass(frozen=True, slots=True)
class BacktestJobStageAShortlist:
    """
    Persisted Stage-A shortlist projection used for deterministic restart/resume in worker.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
      - alembic/versions/20260222_0003_backtest_jobs_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
    """

    job_id: UUID
    stage_a_indexes: tuple[int, ...]
    stage_a_variants_total: int
    risk_total: int
    preselect_used: int
    updated_at: datetime

    def __post_init__(self) -> None:
        """
        Validate Stage-A shortlist payload shape and invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `stage_a_indexes` order is deterministic as produced by Stage-A ranking.
        Raises:
            BacktestJobTransitionError: If one invariant is violated.
        Side Effects:
            Normalizes indexes into immutable integer tuple.
        """
        if self.stage_a_variants_total <= 0:
            raise BacktestJobTransitionError(
                "BacktestJobStageAShortlist.stage_a_variants_total must be > 0"
            )
        if self.risk_total <= 0:
            raise BacktestJobTransitionError("BacktestJobStageAShortlist.risk_total must be > 0")
        if self.preselect_used <= 0:
            raise BacktestJobTransitionError(
                "BacktestJobStageAShortlist.preselect_used must be > 0"
            )

        normalized_indexes: list[int] = []
        if len(self.stage_a_indexes) == 0:
            raise BacktestJobTransitionError(
                "BacktestJobStageAShortlist.stage_a_indexes must be non-empty"
            )
        for raw_index in self.stage_a_indexes:
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise BacktestJobTransitionError(
                    "BacktestJobStageAShortlist.stage_a_indexes items must be integers"
                )
            if raw_index < 0:
                raise BacktestJobTransitionError(
                    "BacktestJobStageAShortlist.stage_a_indexes items must be >= 0"
                )
            normalized_indexes.append(raw_index)

        _ensure_utc_datetime(name="updated_at", value=self.updated_at)
        object.__setattr__(self, "stage_a_indexes", tuple(normalized_indexes))

    def to_json_array(self) -> list[int]:
        """
        Convert shortlist indexes into JSON array payload for SQL adapter writes.

        Args:
            None.
        Returns:
            list[int]: Deterministic ordered indexes list.
        Assumptions:
            Order is preserved from immutable `stage_a_indexes` tuple.
        Raises:
            None.
        Side Effects:
            None.
        """
        return list(self.stage_a_indexes)



def report_table_md_allowed_for_state(*, state: BacktestJobState) -> bool:
    """
    Check whether `report_table_md` is allowed for given job state.

    Args:
        state: Job lifecycle state.
    Returns:
        bool: `True` only for `succeeded` state.
    Assumptions:
        Non-succeeded jobs keep `report_table_md` as `NULL` in storage contract.
    Raises:
        None.
    Side Effects:
        None.
    """
    return state == "succeeded"



def _ensure_sha256_key(*, name: str, value: str) -> None:
    """
    Validate canonical SHA-256 hex key literal format.

    Args:
        name: Field name used in deterministic error messages.
        value: Candidate key literal.
    Returns:
        None.
    Assumptions:
        Keys are lowercase 64-char SHA-256 hex values.
    Raises:
        BacktestJobTransitionError: If key shape is invalid.
    Side Effects:
        None.
    """
    if len(value) != 64:
        raise BacktestJobTransitionError(
            f"BacktestJobTopVariant.{name} must be 64 lowercase hex chars"
        )
    allowed = set("0123456789abcdef")
    if any(char not in allowed for char in value):
        raise BacktestJobTransitionError(
            f"BacktestJobTopVariant.{name} must be 64 lowercase hex chars"
        )



def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    """
    Validate timezone-aware UTC datetime field.

    Args:
        name: Field name used in deterministic error messages.
        value: Datetime value.
    Returns:
        None.
    Assumptions:
        Persisted result timestamps are UTC-aware.
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



def _normalize_json_object(*, value: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize mapping payload into deterministic JSON-compatible object.

    Args:
        value: Raw object payload.
    Returns:
        dict[str, Any]: Key-sorted normalized JSON object.
    Assumptions:
        Mapping keys can be represented as strings.
    Raises:
        BacktestJobTransitionError: If payload cannot be normalized to JSON object.
    Side Effects:
        None.
    """
    normalized = _normalize_json_value(value=dict(value))
    if not isinstance(normalized, Mapping):
        raise BacktestJobTransitionError("Expected JSON object payload")
    try:
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError as error:
        raise BacktestJobTransitionError("Payload must be JSON-serializable") from error
    return dict(normalized)



def _normalize_json_value(*, value: Any) -> Any:
    """
    Normalize arbitrary JSON-like node into deterministic structure.

    Args:
        value: Raw JSON-like node.
    Returns:
        Any: Deterministic mapping/list/scalar value.
    Assumptions:
        Unknown non-JSON objects are stringified for stable persistence.
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
    "BacktestJobStageAShortlist",
    "BacktestJobTopVariant",
    "report_table_md_allowed_for_state",
]
