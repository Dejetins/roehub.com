from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SignalV1(str, Enum):
    """
    Discrete per-bar signal literal for backtest signal-rules v1.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/domain/value_objects/signal_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py
    """

    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True, slots=True)
class IndicatorSignalsV1:
    """
    Deterministic signal series emitted for one indicator.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/domain/value_objects/signal_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py
    """

    indicator_id: str
    signals: np.ndarray

    def __post_init__(self) -> None:
        """
        Validate indicator id and normalize signal series values.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Input `signals` is a one-dimensional per-bar array.
        Raises:
            ValueError: If indicator id is blank, series shape is invalid, or value is unknown.
        Side Effects:
            Normalizes identifier to lowercase and signal array to unicode string dtype.
        """
        normalized_indicator_id = self.indicator_id.strip().lower()
        if not normalized_indicator_id:
            raise ValueError("IndicatorSignalsV1.indicator_id must be non-empty")
        object.__setattr__(self, "indicator_id", normalized_indicator_id)

        if self.signals.ndim != 1:
            raise ValueError("IndicatorSignalsV1.signals must be a 1D array")

        normalized_values: list[str] = []
        for value in self.signals.tolist():
            normalized_value = str(value).strip().upper()
            if normalized_value not in {
                SignalV1.LONG.value,
                SignalV1.SHORT.value,
                SignalV1.NEUTRAL.value,
            }:
                raise ValueError(
                    "IndicatorSignalsV1.signals values must be LONG, SHORT, or NEUTRAL"
                )
            normalized_values.append(normalized_value)
        object.__setattr__(
            self,
            "signals",
            np.asarray(normalized_values, dtype="U7"),
        )


@dataclass(frozen=True, slots=True)
class AggregatedSignalsV1:
    """
    Deterministic strategy-level aggregation payload for signals-from-indicators v1.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/domain/value_objects/signal_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py
    """

    per_indicator_signals: tuple[IndicatorSignalsV1, ...]
    final_signal: np.ndarray
    final_long: np.ndarray
    final_short: np.ndarray
    conflicting_signals: int

    def __post_init__(self) -> None:
        """
        Validate deterministic ordering and shape consistency of aggregated payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `final_*` arrays share one bar-aligned timeline.
        Raises:
            ValueError: If ordering, lengths, or conflicting counter invariants are broken.
        Side Effects:
            Normalizes final signal labels and boolean arrays to canonical dtypes.
        """
        indicator_ids = tuple(item.indicator_id for item in self.per_indicator_signals)
        if tuple(sorted(indicator_ids)) != indicator_ids:
            raise ValueError(
                "AggregatedSignalsV1.per_indicator_signals must be sorted by indicator_id"
            )

        if self.final_signal.ndim != 1:
            raise ValueError("AggregatedSignalsV1.final_signal must be a 1D array")
        if self.final_long.ndim != 1:
            raise ValueError("AggregatedSignalsV1.final_long must be a 1D array")
        if self.final_short.ndim != 1:
            raise ValueError("AggregatedSignalsV1.final_short must be a 1D array")

        timeline_length = int(self.final_signal.shape[0])
        if self.final_long.shape[0] != timeline_length:
            raise ValueError("AggregatedSignalsV1.final_long length must match final_signal")
        if self.final_short.shape[0] != timeline_length:
            raise ValueError("AggregatedSignalsV1.final_short length must match final_signal")
        if self.conflicting_signals < 0:
            raise ValueError("AggregatedSignalsV1.conflicting_signals must be >= 0")
        if self.conflicting_signals > timeline_length:
            raise ValueError(
                "AggregatedSignalsV1.conflicting_signals must be <= bars count in final_signal"
            )

        for item in self.per_indicator_signals:
            if item.signals.shape[0] != timeline_length:
                raise ValueError(
                    "AggregatedSignalsV1 indicator signal length must match final_signal"
                )

        normalized_final_signal: list[str] = []
        for value in self.final_signal.tolist():
            normalized_value = str(value).strip().upper()
            if normalized_value not in {
                SignalV1.LONG.value,
                SignalV1.SHORT.value,
                SignalV1.NEUTRAL.value,
            }:
                raise ValueError(
                    "AggregatedSignalsV1.final_signal values must be LONG, SHORT, or NEUTRAL"
                )
            normalized_final_signal.append(normalized_value)

        object.__setattr__(
            self,
            "final_signal",
            np.asarray(normalized_final_signal, dtype="U7"),
        )
        object.__setattr__(
            self,
            "final_long",
            np.asarray(self.final_long, dtype=np.bool_),
        )
        object.__setattr__(
            self,
            "final_short",
            np.asarray(self.final_short, dtype=np.bool_),
        )
