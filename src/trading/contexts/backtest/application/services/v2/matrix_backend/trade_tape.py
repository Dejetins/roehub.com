from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from trading.contexts.backtest.application.dto import BacktestPreparePoolsResult
from trading.contexts.backtest.application.services.v2.tp_sl_exact import (
    build_trade_list_15m_for_indicator_rows_slow,
)


@dataclass(frozen=True, slots=True)
class CandidateTradeTape:
    local_indices: tuple[int, ...]
    entry_abs: np.ndarray
    direction: np.ndarray
    signal_exit_abs: np.ndarray

    @property
    def trade_count(self) -> int:
        return int(self.entry_abs.shape[0])

    @property
    def long_trade_count(self) -> int:
        return int(np.count_nonzero(self.direction == np.int8(1)))

    @property
    def short_trade_count(self) -> int:
        return int(np.count_nonzero(self.direction == np.int8(-1)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "local_indices": list(self.local_indices),
            "trade_count": self.trade_count,
            "long_trade_count": self.long_trade_count,
            "short_trade_count": self.short_trade_count,
            "entry_abs_min": None if self.trade_count == 0 else int(np.min(self.entry_abs)),
            "entry_abs_max": None if self.trade_count == 0 else int(np.max(self.entry_abs)),
        }


@dataclass(frozen=True, slots=True)
class SparseTradeTapeExtraction:
    candidate_count: int
    tapes: tuple[CandidateTradeTape, ...]
    selected_entries: np.ndarray

    @property
    def trade_count(self) -> int:
        return sum(tape.trade_count for tape in self.tapes)

    @property
    def long_trade_count(self) -> int:
        return sum(tape.long_trade_count for tape in self.tapes)

    @property
    def short_trade_count(self) -> int:
        return sum(tape.short_trade_count for tape in self.tapes)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema": "backtest_tp_sl_sparse_trade_tape_v1",
            "candidate_count": self.candidate_count,
            "trade_count": self.trade_count,
            "long_trade_count": self.long_trade_count,
            "short_trade_count": self.short_trade_count,
            "selected_entry_count": int(self.selected_entries.shape[0]),
            "selected_entry_min": None
            if int(self.selected_entries.shape[0]) == 0
            else int(np.min(self.selected_entries)),
            "selected_entry_max": None
            if int(self.selected_entries.shape[0]) == 0
            else int(np.max(self.selected_entries)),
            "sample": [tape.as_mapping() for tape in self.tapes[:5]],
        }


def extract_selected_trade_tapes(
    *,
    prepared_result: BacktestPreparePoolsResult,
    selected_rows_by_indicator: Mapping[str, np.ndarray],
    direction_mode: str,
    max_candidates: int,
) -> SparseTradeTapeExtraction:
    if max_candidates <= 0:
        raise ValueError("max_candidates must be > 0")
    if not selected_rows_by_indicator:
        return SparseTradeTapeExtraction(
            candidate_count=0,
            tapes=(),
            selected_entries=np.empty(0, dtype=np.int32),
        )

    selected_size = min(
        int(np.asarray(rows).shape[0]) for rows in selected_rows_by_indicator.values()
    )
    candidate_count = min(selected_size, int(max_candidates))
    tapes: list[CandidateTradeTape] = []
    entries: list[np.ndarray] = []
    for candidate_idx in range(candidate_count):
        local_indices = tuple(
            int(np.asarray(selected_rows_by_indicator[indicator_id])[candidate_idx])
            for indicator_id in prepared_result.indicator_ids
        )
        entry_abs, direction, signal_exit_abs = build_trade_list_15m_for_indicator_rows_slow(
            prepared_result=prepared_result,
            local_indices=local_indices,
            direction_mode=direction_mode,
        )
        tape = CandidateTradeTape(
            local_indices=local_indices,
            entry_abs=np.ascontiguousarray(entry_abs, dtype=np.int32),
            direction=np.ascontiguousarray(direction, dtype=np.int8),
            signal_exit_abs=np.ascontiguousarray(signal_exit_abs, dtype=np.int32),
        )
        tapes.append(tape)
        if tape.trade_count > 0:
            entries.append(tape.entry_abs)

    selected_entries = (
        np.unique(np.concatenate(entries)).astype(np.int32, copy=False)
        if entries
        else np.empty(0, dtype=np.int32)
    )
    return SparseTradeTapeExtraction(
        candidate_count=candidate_count,
        tapes=tuple(tapes),
        selected_entries=np.ascontiguousarray(selected_entries, dtype=np.int32),
    )


__all__ = [
    "CandidateTradeTape",
    "SparseTradeTapeExtraction",
    "extract_selected_trade_tapes",
]
