from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from trading.contexts.backtest.application.dto import BacktestRiskGridSpec
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.specifications import GridParamSpec, GridSpec
from trading.shared_kernel.primitives import InstrumentId, Timeframe, UserId


@dataclass(frozen=True, slots=True)
class BacktestStrategySnapshot:
    """
    Saved strategy projection consumed by backtest use-case via ACL port.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
    """

    strategy_id: UUID
    user_id: UserId
    is_deleted: bool
    instrument_id: InstrumentId
    timeframe: Timeframe
    indicator_grids: tuple[GridSpec, ...]
    indicator_selections: tuple[IndicatorVariantSelection, ...]
    signal_grids: dict[str, dict[str, GridParamSpec]] | None = None
    risk_grid: BacktestRiskGridSpec | None = None

    def __post_init__(self) -> None:
        """
        Validate minimal saved-strategy snapshot invariants for backtest v1.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Snapshot is loaded without owner filtering and must be checked in use-case.
        Raises:
            ValueError: If required fields are missing or indicator payloads are empty.
        Side Effects:
            None.
        """
        if self.user_id is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestStrategySnapshot.user_id is required")
        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestStrategySnapshot.instrument_id is required")
        if self.timeframe is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestStrategySnapshot.timeframe is required")
        if len(self.indicator_grids) == 0:
            raise ValueError("BacktestStrategySnapshot.indicator_grids must be non-empty")
        if len(self.indicator_selections) == 0:
            raise ValueError("BacktestStrategySnapshot.indicator_selections must be non-empty")


class BacktestStrategyReader(Protocol):
    """
    Backtest ACL port for loading one saved strategy by id without owner filtering.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
      - src/trading/contexts/backtest/application/ports/current_user.py
    """

    def load_any(self, *, strategy_id: UUID) -> BacktestStrategySnapshot | None:
        """
        Load strategy snapshot by id without owner filtering.

        Args:
            strategy_id: Saved strategy identifier.
        Returns:
            BacktestStrategySnapshot | None: Snapshot for explicit ownership checks or `None`.
        Assumptions:
            Adapter boundary handles translation from strategy storage model to backtest snapshot.
        Raises:
            ValueError: If adapter cannot map stored strategy payload deterministically.
        Side Effects:
            Reads saved strategy from outbound storage.
        """
        ...
