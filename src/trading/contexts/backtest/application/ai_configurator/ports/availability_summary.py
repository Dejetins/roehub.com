from __future__ import annotations

from typing import Any, Mapping, Protocol


class BacktestAiAvailabilitySummaryRepository(Protocol):
    def load_availability_summary(self) -> Mapping[str, Any]:
        """
        Load publisher-owned `availability_summary.yaml` as a mapping.

        Implementations must not expose local filesystem paths to model-facing prompt
        context. The application layer treats the loaded YAML payload as the only
        source of truth for available symbols, exchange/market, timeframes, and periods.
        """
        ...


__all__ = ["BacktestAiAvailabilitySummaryRepository"]
