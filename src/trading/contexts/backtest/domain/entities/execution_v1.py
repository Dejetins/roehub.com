from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PositionV1:
    """
    Single open position snapshot for close-fill execution engine v1.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - src/trading/contexts/backtest/domain/entities/execution_v1.py
      - tests/unit/contexts/backtest/domain/entities/test_execution_v1_entities.py
    """

    trade_id: int
    direction: str
    qty_base: float
    entry_fill_price: float
    entry_bar_index: int
    entry_quote_amount: float
    entry_fee_quote: float

    def __post_init__(self) -> None:
        """
        Validate one open-position state snapshot.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Position direction is one of `long`/`short` literals.
        Raises:
            ValueError: If one numeric invariant is violated.
        Side Effects:
            Normalizes direction literal to lowercase stripped representation.
        """
        if self.trade_id <= 0:
            raise ValueError("PositionV1.trade_id must be > 0")
        normalized_direction = self.direction.strip().lower()
        object.__setattr__(self, "direction", normalized_direction)
        if normalized_direction not in {"long", "short"}:
            raise ValueError("PositionV1.direction must be long or short")
        if self.qty_base <= 0.0:
            raise ValueError("PositionV1.qty_base must be > 0")
        if self.entry_fill_price <= 0.0:
            raise ValueError("PositionV1.entry_fill_price must be > 0")
        if self.entry_bar_index < 0:
            raise ValueError("PositionV1.entry_bar_index must be >= 0")
        if self.entry_quote_amount <= 0.0:
            raise ValueError("PositionV1.entry_quote_amount must be > 0")
        if self.entry_fee_quote < 0.0:
            raise ValueError("PositionV1.entry_fee_quote must be >= 0")

    def unrealized_gross_pnl_quote(self, *, close_price: float) -> float:
        """
        Compute gross unrealized quote PnL at provided close price.

        Args:
            close_price: Close price used for mark-to-market.
        Returns:
            float: Gross unrealized quote PnL without entry/exit fees.
        Assumptions:
            Close price is positive finite scalar.
        Raises:
            ValueError: If close price is non-positive.
        Side Effects:
            None.
        """
        if close_price <= 0.0:
            raise ValueError("close_price must be > 0")

        current_notional = self.qty_base * close_price
        if self.direction == "long":
            return current_notional - self.entry_quote_amount
        return self.entry_quote_amount - current_notional


@dataclass(frozen=True, slots=True)
class TradeV1:
    """
    Closed trade snapshot emitted by close-fill execution engine v1.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - src/trading/contexts/backtest/domain/entities/execution_v1.py
      - tests/unit/contexts/backtest/application/services/test_execution_engine_v1.py
    """

    trade_id: int
    direction: str
    entry_bar_index: int
    exit_bar_index: int
    entry_fill_price: float
    exit_fill_price: float
    qty_base: float
    entry_quote_amount: float
    exit_quote_amount: float
    entry_fee_quote: float
    exit_fee_quote: float
    gross_pnl_quote: float
    net_pnl_quote: float
    locked_profit_quote: float
    exit_reason: str

    def __post_init__(self) -> None:
        """
        Validate closed trade snapshot invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Entry/exit prices already include slippage semantics.
        Raises:
            ValueError: If one numeric invariant or textual field is invalid.
        Side Effects:
            Normalizes `direction` and `exit_reason` literals.
        """
        if self.trade_id <= 0:
            raise ValueError("TradeV1.trade_id must be > 0")
        normalized_direction = self.direction.strip().lower()
        object.__setattr__(self, "direction", normalized_direction)
        if normalized_direction not in {"long", "short"}:
            raise ValueError("TradeV1.direction must be long or short")

        if self.entry_bar_index < 0:
            raise ValueError("TradeV1.entry_bar_index must be >= 0")
        if self.exit_bar_index < self.entry_bar_index:
            raise ValueError("TradeV1.exit_bar_index must be >= entry_bar_index")
        if self.entry_fill_price <= 0.0:
            raise ValueError("TradeV1.entry_fill_price must be > 0")
        if self.exit_fill_price <= 0.0:
            raise ValueError("TradeV1.exit_fill_price must be > 0")
        if self.qty_base <= 0.0:
            raise ValueError("TradeV1.qty_base must be > 0")
        if self.entry_quote_amount <= 0.0:
            raise ValueError("TradeV1.entry_quote_amount must be > 0")
        if self.exit_quote_amount <= 0.0:
            raise ValueError("TradeV1.exit_quote_amount must be > 0")
        if self.entry_fee_quote < 0.0:
            raise ValueError("TradeV1.entry_fee_quote must be >= 0")
        if self.exit_fee_quote < 0.0:
            raise ValueError("TradeV1.exit_fee_quote must be >= 0")
        if self.locked_profit_quote < 0.0:
            raise ValueError("TradeV1.locked_profit_quote must be >= 0")

        normalized_exit_reason = self.exit_reason.strip().lower()
        object.__setattr__(self, "exit_reason", normalized_exit_reason)
        if not normalized_exit_reason:
            raise ValueError("TradeV1.exit_reason must be non-empty")


@dataclass(frozen=True, slots=True)
class AccountStateV1:
    """
    Strategy-accounting balances for sizing/profit-lock policy in engine v1.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - src/trading/contexts/backtest/domain/entities/execution_v1.py
      - tests/unit/contexts/backtest/application/services/test_execution_engine_v1.py
    """

    available_quote: float
    safe_quote: float = 0.0

    def __post_init__(self) -> None:
        """
        Validate account-state scalar bounds.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `available_quote` may become negative temporarily in simplified v1 model.
        Raises:
            ValueError: If `safe_quote` is negative.
        Side Effects:
            None.
        """
        if self.safe_quote < 0.0:
            raise ValueError("AccountStateV1.safe_quote must be >= 0")

    def reserve_for_entry(
        self,
        *,
        quote_amount: float,
        entry_fee_quote: float,
    ) -> AccountStateV1:
        """
        Return new account state after reserving budget and fee on position entry.

        Args:
            quote_amount: Entry notional reserved for the position.
            entry_fee_quote: Entry commission charged on fill notional.
        Returns:
            AccountStateV1: Updated account state.
        Assumptions:
            Reserve operation is called only when opening new position.
        Raises:
            ValueError: If amount or fee is negative.
        Side Effects:
            None.
        """
        if quote_amount < 0.0:
            raise ValueError("quote_amount must be >= 0")
        if entry_fee_quote < 0.0:
            raise ValueError("entry_fee_quote must be >= 0")
        return AccountStateV1(
            available_quote=self.available_quote - quote_amount - entry_fee_quote,
            safe_quote=self.safe_quote,
        )

    def release_after_close(
        self,
        *,
        quote_amount: float,
        gross_pnl_quote: float,
        exit_fee_quote: float,
    ) -> AccountStateV1:
        """
        Return new account state after position close and exit fee settlement.

        Args:
            quote_amount: Entry notional released back to available quote.
            gross_pnl_quote: Gross quote PnL without fees.
            exit_fee_quote: Exit commission charged on close fill notional.
        Returns:
            AccountStateV1: Updated account state.
        Assumptions:
            Entry fee was already charged during `reserve_for_entry` operation.
        Raises:
            ValueError: If quote amount or exit fee is negative.
        Side Effects:
            None.
        """
        if quote_amount < 0.0:
            raise ValueError("quote_amount must be >= 0")
        if exit_fee_quote < 0.0:
            raise ValueError("exit_fee_quote must be >= 0")
        return AccountStateV1(
            available_quote=self.available_quote + quote_amount + gross_pnl_quote - exit_fee_quote,
            safe_quote=self.safe_quote,
        )

    def lock_profit(self, *, locked_quote: float) -> AccountStateV1:
        """
        Return new account state after profit-lock transfer to safe balance.

        Args:
            locked_quote: Quote amount moved from available balance to safe balance.
        Returns:
            AccountStateV1: Updated account state.
        Assumptions:
            Profit lock is applied only after profitable trade close.
        Raises:
            ValueError: If lock amount is negative.
        Side Effects:
            None.
        """
        if locked_quote < 0.0:
            raise ValueError("locked_quote must be >= 0")
        return AccountStateV1(
            available_quote=self.available_quote - locked_quote,
            safe_quote=self.safe_quote + locked_quote,
        )


@dataclass(frozen=True, slots=True)
class ExecutionOutcomeV1:
    """
    Deterministic close-fill engine output used for staged ranking and tests.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - tests/unit/contexts/backtest/application/services/test_execution_engine_v1.py
    """

    trades: tuple[TradeV1, ...]
    equity_end_quote: float
    available_quote: float
    safe_quote: float
    total_return_pct: float

    def __post_init__(self) -> None:
        """
        Validate deterministic execution-output scalar invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `total_return_pct` is computed from initial and final equity by engine.
        Raises:
            ValueError: If one scalar is not finite enough for deterministic consumers.
        Side Effects:
            None.
        """
        if self.safe_quote < 0.0:
            raise ValueError("ExecutionOutcomeV1.safe_quote must be >= 0")


__all__ = [
    "AccountStateV1",
    "ExecutionOutcomeV1",
    "PositionV1",
    "TradeV1",
]
