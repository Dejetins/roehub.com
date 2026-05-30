from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.strategy.domain.errors import StrategySpecValidationError
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class StrategyBacktestVariantProvenance:
    """
    Immutable provenance row linking a Strategy snapshot to its source backtest variant.
    """

    strategy_id: UUID
    user_id: UserId
    source_job_id: UUID
    source_variant_key: str
    source_variant_hash: str
    source_indicator_variant_hash: str | None
    backtest_request_hash: str
    backtest_result_config_hash: str
    strategy_spec_hash: str
    launch_request_hash: str
    idempotency_key_hash: str
    created_at: datetime
    metadata_json: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "source_variant_key",
            "source_variant_hash",
            "backtest_request_hash",
            "backtest_result_config_hash",
            "strategy_spec_hash",
            "launch_request_hash",
            "idempotency_key_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise StrategySpecValidationError(
                    f"StrategyBacktestVariantProvenance.{field_name} must be non-empty"
                )
        _ensure_utc_datetime(name="created_at", value=self.created_at)


def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise StrategySpecValidationError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise StrategySpecValidationError(f"{name} must be UTC datetime")
