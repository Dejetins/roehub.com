from __future__ import annotations

from dataclasses import dataclass

from trading.contexts.backtest.domain.value_objects import BacktestVariantIdentity


@dataclass(frozen=True, slots=True)
class BacktestPositionPlaceholder:
    """
    Placeholder open-position projection for BKT-EPIC-01 skeleton boundaries.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_placeholders.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    direction: str
    quantity: float

    def __post_init__(self) -> None:
        """
        Validate minimal position placeholder invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Only `long` and `short` position directions are supported in v1 contracts.
        Raises:
            ValueError: If direction literal is unsupported or quantity is non-positive.
        Side Effects:
            Normalizes direction literal to lowercase.
        """
        normalized_direction = self.direction.strip().lower()
        object.__setattr__(self, "direction", normalized_direction)
        if normalized_direction not in {"long", "short"}:
            raise ValueError("BacktestPositionPlaceholder.direction must be long or short")
        if self.quantity <= 0:
            raise ValueError("BacktestPositionPlaceholder.quantity must be > 0")


@dataclass(frozen=True, slots=True)
class BacktestTradePlaceholder:
    """
    Placeholder trade projection for BKT-EPIC-01 skeleton boundaries.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_placeholders.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    trade_id: str
    opened_at_index: int
    closed_at_index: int | None = None

    def __post_init__(self) -> None:
        """
        Validate minimal trade placeholder invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Candle index references are deterministic non-negative integers.
        Raises:
            ValueError: If id is blank or index invariants are violated.
        Side Effects:
            Normalizes trade id by stripping spaces.
        """
        normalized_trade_id = self.trade_id.strip()
        object.__setattr__(self, "trade_id", normalized_trade_id)
        if not normalized_trade_id:
            raise ValueError("BacktestTradePlaceholder.trade_id must be non-empty")
        if self.opened_at_index < 0:
            raise ValueError("BacktestTradePlaceholder.opened_at_index must be >= 0")
        if self.closed_at_index is not None and self.closed_at_index < self.opened_at_index:
            raise ValueError(
                "BacktestTradePlaceholder.closed_at_index must be >= opened_at_index"
            )


@dataclass(frozen=True, slots=True)
class BacktestResultPlaceholder:
    """
    Placeholder result envelope for one variant in BKT-EPIC-01.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    variant: BacktestVariantIdentity
    positions: tuple[BacktestPositionPlaceholder, ...] = ()
    trades: tuple[BacktestTradePlaceholder, ...] = ()

    def __post_init__(self) -> None:
        """
        Validate minimal result placeholder shape.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Position/trade collections are immutable snapshots in deterministic order.
        Raises:
            ValueError: If `variant` identity is missing.
        Side Effects:
            None.
        """
        if self.variant is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestResultPlaceholder.variant is required")

