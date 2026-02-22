from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.backtest.domain.errors import BacktestValidationError


@dataclass(frozen=True, slots=True)
class BacktestJobListCursor:
    """
    Keyset pagination cursor for deterministic Backtest jobs list ordering.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - apps/api/routes/backtests.py
    """

    created_at: datetime
    job_id: UUID

    def __post_init__(self) -> None:
        """
        Validate cursor fields used in keyset pagination predicates.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Cursor follows ordering `(created_at DESC, job_id DESC)`.
        Raises:
            BacktestValidationError: If `created_at` is not UTC-aware.
        Side Effects:
            None.
        """
        _ensure_utc_datetime(name="created_at", value=self.created_at)

    def to_payload(self) -> Mapping[str, str]:
        """
        Convert cursor into transport-safe payload mapping.

        Args:
            None.
        Returns:
            Mapping[str, str]: Payload with `created_at` ISO string and `job_id` literal.
        Assumptions:
            Consumer serializes mapping as JSON for API cursor transport.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "created_at": self.created_at.isoformat(),
            "job_id": str(self.job_id),
        }

    @classmethod
    def from_payload(cls, *, payload: Mapping[str, Any]) -> BacktestJobListCursor:
        """
        Parse cursor from mapping payload and validate deterministic shape.

        Args:
            payload: Raw cursor mapping.
        Returns:
            BacktestJobListCursor: Parsed validated cursor object.
        Assumptions:
            Payload keys are exactly `created_at` and `job_id` with string values.
        Raises:
            BacktestValidationError: If payload cannot be parsed.
        Side Effects:
            None.
        """
        raw_created_at = payload.get("created_at")
        raw_job_id = payload.get("job_id")
        if not isinstance(raw_created_at, str) or not raw_created_at.strip():
            raise BacktestValidationError("cursor.created_at must be non-empty string")
        if not isinstance(raw_job_id, str) or not raw_job_id.strip():
            raise BacktestValidationError("cursor.job_id must be non-empty string")

        try:
            created_at = datetime.fromisoformat(raw_created_at.strip())
        except ValueError as error:
            raise BacktestValidationError("cursor.created_at must be ISO datetime") from error

        try:
            job_id = UUID(raw_job_id.strip())
        except ValueError as error:
            raise BacktestValidationError("cursor.job_id must be UUID string") from error

        return cls(created_at=created_at, job_id=job_id)



def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    """
    Validate timezone-aware UTC datetime field for pagination cursor.

    Args:
        name: Field name used in deterministic error message.
        value: Datetime value.
    Returns:
        None.
    Assumptions:
        Cursor timestamps are persisted and compared in UTC.
    Raises:
        BacktestValidationError: If datetime is naive or non-UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise BacktestValidationError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise BacktestValidationError(f"{name} must be UTC datetime")


__all__ = ["BacktestJobListCursor"]
