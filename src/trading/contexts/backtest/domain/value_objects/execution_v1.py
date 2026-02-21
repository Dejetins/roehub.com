from __future__ import annotations

from dataclasses import dataclass

_ALLOWED_DIRECTION_MODES = {"long-only", "short-only", "long-short"}
_ALLOWED_SIZING_MODES = {
    "all_in",
    "fixed_quote",
    "strategy_compound",
    "strategy_compound_profit_lock",
}


@dataclass(frozen=True, slots=True)
class ExecutionParamsV1:
    """
    Immutable execution parameters for close-fill backtest engine v1.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    direction_mode: str
    sizing_mode: str
    init_cash_quote: float
    fixed_quote: float
    safe_profit_percent: float
    fee_pct: float
    slippage_pct: float

    def __post_init__(self) -> None:
        """
        Validate execution-parameter invariants used by close-fill engine.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Percent parameters use human percent units (`1.0 == 1%`).
        Raises:
            ValueError: If one mode literal is unsupported or numeric bounds are invalid.
        Side Effects:
            Normalizes mode literals to lowercase stripped representation.
        """
        normalized_direction_mode = self.direction_mode.strip().lower()
        object.__setattr__(self, "direction_mode", normalized_direction_mode)
        if normalized_direction_mode not in _ALLOWED_DIRECTION_MODES:
            raise ValueError(
                "ExecutionParamsV1.direction_mode must be one of: "
                f"{sorted(_ALLOWED_DIRECTION_MODES)}"
            )

        normalized_sizing_mode = self.sizing_mode.strip().lower()
        object.__setattr__(self, "sizing_mode", normalized_sizing_mode)
        if normalized_sizing_mode not in _ALLOWED_SIZING_MODES:
            raise ValueError(
                "ExecutionParamsV1.sizing_mode must be one of: "
                f"{sorted(_ALLOWED_SIZING_MODES)}"
            )

        if self.init_cash_quote <= 0.0:
            raise ValueError("ExecutionParamsV1.init_cash_quote must be > 0")
        if self.fixed_quote <= 0.0:
            raise ValueError("ExecutionParamsV1.fixed_quote must be > 0")
        if self.safe_profit_percent < 0.0 or self.safe_profit_percent > 100.0:
            raise ValueError("ExecutionParamsV1.safe_profit_percent must be in [0, 100]")
        if self.fee_pct < 0.0:
            raise ValueError("ExecutionParamsV1.fee_pct must be >= 0")
        if self.slippage_pct < 0.0:
            raise ValueError("ExecutionParamsV1.slippage_pct must be >= 0")

    @property
    def fee_rate(self) -> float:
        """
        Return fee rate as decimal fraction.

        Args:
            None.
        Returns:
            float: Decimal fee rate (`0.001 == 0.1%`).
        Assumptions:
            `fee_pct` is expressed in human percent units.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self.fee_pct / 100.0

    @property
    def slippage_rate(self) -> float:
        """
        Return slippage rate as decimal fraction.

        Args:
            None.
        Returns:
            float: Decimal slippage rate (`0.0001 == 0.01%`).
        Assumptions:
            `slippage_pct` is expressed in human percent units.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self.slippage_pct / 100.0


@dataclass(frozen=True, slots=True)
class RiskParamsV1:
    """
    Immutable close-based SL/TP parameters for one backtest variant run.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    sl_enabled: bool
    sl_pct: float | None
    tp_enabled: bool
    tp_pct: float | None

    def __post_init__(self) -> None:
        """
        Validate risk-parameter invariants for close-based SL/TP engine checks.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Percent parameters use human percent units (`3.0 == 3%`).
        Raises:
            ValueError: If enabled SL/TP axis misses percentage value or value is invalid.
        Side Effects:
            Normalizes disabled axis percentages to `None`.
        """
        normalized_sl_pct: float | None = self.sl_pct
        normalized_tp_pct: float | None = self.tp_pct

        if self.sl_enabled:
            if normalized_sl_pct is None:
                raise ValueError("RiskParamsV1.sl_pct must be set when sl_enabled is true")
            if normalized_sl_pct < 0.0:
                raise ValueError("RiskParamsV1.sl_pct must be >= 0 when enabled")
        else:
            normalized_sl_pct = None

        if self.tp_enabled:
            if normalized_tp_pct is None:
                raise ValueError("RiskParamsV1.tp_pct must be set when tp_enabled is true")
            if normalized_tp_pct < 0.0:
                raise ValueError("RiskParamsV1.tp_pct must be >= 0 when enabled")
        else:
            normalized_tp_pct = None

        object.__setattr__(self, "sl_pct", normalized_sl_pct)
        object.__setattr__(self, "tp_pct", normalized_tp_pct)

    @property
    def sl_rate(self) -> float | None:
        """
        Return SL rate as decimal fraction when SL axis is enabled.

        Args:
            None.
        Returns:
            float | None: Decimal SL rate or `None` when disabled.
        Assumptions:
            `sl_pct` is expressed in human percent units.
        Raises:
            None.
        Side Effects:
            None.
        """
        if self.sl_pct is None:
            return None
        return self.sl_pct / 100.0

    @property
    def tp_rate(self) -> float | None:
        """
        Return TP rate as decimal fraction when TP axis is enabled.

        Args:
            None.
        Returns:
            float | None: Decimal TP rate or `None` when disabled.
        Assumptions:
            `tp_pct` is expressed in human percent units.
        Raises:
            None.
        Side Effects:
            None.
        """
        if self.tp_pct is None:
            return None
        return self.tp_pct / 100.0


__all__ = [
    "ExecutionParamsV1",
    "RiskParamsV1",
]
