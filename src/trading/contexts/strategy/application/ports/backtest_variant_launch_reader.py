from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from uuid import UUID

from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class BacktestVariantLaunchSnapshot:
    """
    Backtest-context data needed to create an immutable strategy from a variant.
    """

    job_id: UUID
    owner_user_id: UserId
    job_state: str
    request_hash: str
    result_config_hash: str
    market_id: int
    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    variant_key: str
    variant_hash: str
    indicator_variant_hash: str | None
    rank: int
    summary_metrics: Mapping[str, Any]
    canonical_variant_params: Mapping[str, Any]
    readable_params: Mapping[str, Any]


class BacktestVariantLaunchReader(Protocol):
    """
    ACL port resolving owner-scoped backtest variants for Strategy launch.
    """

    def get(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        variant_key: str,
    ) -> BacktestVariantLaunchSnapshot:
        """
        Return one owner-visible variant snapshot or raise a stable RoehubError.
        """
        ...
