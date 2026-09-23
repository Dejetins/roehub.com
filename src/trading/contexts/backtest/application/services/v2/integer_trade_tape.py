"""Bounded integer-only signal interval scan; financial arithmetic is not compiled here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numba as nb
import numpy as np

from trading.contexts.backtest.application.dto import BacktestPreparePoolsResult


class TradeTapeBuilder(Protocol):
    def __call__(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        local_indices: tuple[int, ...],
        direction_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@nb.njit(cache=True, fastmath=False)
def scan_integer_intervals(signal, start, stop, reversal, entries, directions, exits):
    """Write at most one interval per signal; caller owns all output buffers."""
    count = 0
    current_dir = 0
    current_entry = 0
    for index in range(signal.size):
        direction = int(signal[index])
        if direction == 0 and not (not reversal and current_dir != 0):
            continue
        entry = start + index + 1
        if entry >= stop:
            break
        if direction == 0:
            entries[count] = current_entry
            directions[count] = current_dir
            exits[count] = entry
            count += 1
            current_dir = 0
            current_entry = 0
            continue
        if current_dir == 0:
            current_dir = direction
            current_entry = entry
            continue
        if direction == current_dir:
            continue
        entries[count] = current_entry
        directions[count] = current_dir
        exits[count] = entry
        count += 1
        current_dir = direction
        current_entry = entry
    if current_dir != 0:
        entries[count] = current_entry
        directions[count] = current_dir
        exits[count] = stop
        count += 1
    return count


@dataclass(frozen=True, slots=True)
class IntegerTradeTapeBuilder:
    """Explicit call-local dependency. Unsupported inputs retain the baseline callable."""

    baseline: TradeTapeBuilder
    min_bars: int = 32
    max_bytes: int = 64 * 1024 * 1024

    def __call__(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        local_indices: tuple[int, ...],
        direction_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prepared = prepared_result
        pools = {pool.indicator_id: pool for pool in prepared.indicator_pools}
        rows = tuple(
            pools[name].trade_T[local_indices[pos]]
            for pos, name in enumerate(prepared.indicator_ids)
        )
        start = int(prepared.time_slice_start_15m)
        stop = int(prepared.time_slice_stop_15m)
        n = rows[0].size if rows else 0
        # 9n output + <=3n live consensus/mask buffers; conservative 16n envelope.
        supported = (
            prepared.timeframe == "15m"
            and n >= self.min_bars
            and 16 * n <= self.max_bytes
            and 0 <= start <= stop <= np.iinfo(np.int32).max
            and start + n + 1 <= np.iinfo(np.int64).max
            and direction_mode in ("long_only", "short", "long_short_reversal")
            and all(row.ndim == 1 and row.size == n and row.dtype == np.int8 for row in rows)
        )
        if not supported:
            return self.baseline(
                prepared_result=prepared,
                local_indices=local_indices,
                direction_mode=direction_mode,
            )
        try:
            signal = rows[0].copy()
            for row in rows[1:]:
                signal[row != signal] = np.int8(0)
            if direction_mode == "long_only":
                signal = (signal == np.int8(1)).astype(np.int8)
            elif direction_mode == "short":
                signal = -(signal == np.int8(-1)).astype(np.int8)
            entries = np.empty(n, dtype=np.int32)
            directions = np.empty(n, dtype=np.int8)
            exits = np.empty(n, dtype=np.int32)
        except MemoryError:
            return self.baseline(
                prepared_result=prepared,
                local_indices=local_indices,
                direction_mode=direction_mode,
            )
        count = scan_integer_intervals(
            signal,
            start,
            stop,
            direction_mode == "long_short_reversal",
            entries,
            directions,
            exits,
        )
        # Views retain only these bounded per-call arrays; no global/job cache.
        return entries[:count], directions[:count], exits[:count]
