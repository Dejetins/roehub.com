from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from trading.contexts.backtest.application.dto import PreparedIndicatorPool

DEFAULT_CONSENSUS_SIGNATURE_ENUMERATION_LIMIT = 10_000
_U64_MODULUS = 1 << 64


@dataclass(frozen=True, slots=True)
class IndicatorRowSignatureTelemetry:
    indicator_id: str
    rows_after_prefilter: int
    unique_rows_after_dedup: int
    duplicate_signal_row_ids: tuple[int, ...]
    unique_signal_row_ids: tuple[int, ...]
    row_signature_collision_count: int

    def as_mapping(self) -> dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "rows_after_prefilter": self.rows_after_prefilter,
            "unique_rows_after_dedup": self.unique_rows_after_dedup,
            "duplicate_signal_row_ids": list(self.duplicate_signal_row_ids),
            "unique_signal_row_ids": list(self.unique_signal_row_ids),
            "row_signature_collision_count": self.row_signature_collision_count,
        }


@dataclass(frozen=True, slots=True)
class RowSignatureTelemetry:
    rows_after_prefilter: int
    unique_rows_after_dedup: int
    duplicate_signal_row_ids: dict[str, tuple[int, ...]]
    unique_signal_row_ids: dict[str, tuple[int, ...]]
    row_signature_collision_count: int
    consensus_signature_count: int
    consensus_signature_mode: str
    consensus_signature_enumeration_limit: int
    candidate_upper_bound_after_row_dedup: int
    indicators: tuple[IndicatorRowSignatureTelemetry, ...]

    @property
    def duplicate_row_count(self) -> int:
        return self.rows_after_prefilter - self.unique_rows_after_dedup

    @property
    def duplicate_row_fraction(self) -> float:
        if self.rows_after_prefilter <= 0:
            return 0.0
        return float(self.duplicate_row_count) / float(self.rows_after_prefilter)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema": "backtest_row_signature_telemetry_v1",
            "rows_after_prefilter": self.rows_after_prefilter,
            "unique_rows_after_dedup": self.unique_rows_after_dedup,
            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_row_fraction": self.duplicate_row_fraction,
            "duplicate_signal_row_ids": {
                indicator_id: list(row_ids)
                for indicator_id, row_ids in self.duplicate_signal_row_ids.items()
            },
            "unique_signal_row_ids": {
                indicator_id: list(row_ids)
                for indicator_id, row_ids in self.unique_signal_row_ids.items()
            },
            "row_signature_collision_count": self.row_signature_collision_count,
            "consensus_signature_count": self.consensus_signature_count,
            "consensus_signature_mode": self.consensus_signature_mode,
            "consensus_signature_enumeration_limit": (
                self.consensus_signature_enumeration_limit
            ),
            "candidate_upper_bound_after_row_dedup": (
                self.candidate_upper_bound_after_row_dedup
            ),
            "collision_strategy": (
                "telemetry groups exact SHA-256 row-content signatures; the derived "
                "u64 sidecar-style signature is collision-checked only and must disable "
                "future dedup/cache if collisions are non-zero"
            ),
            "duplicate_mapping_semantics": (
                "duplicate_signal_row_ids lists original source row ids whose exact "
                "signal row content matches the first stable unique row for the same "
                "indicator; Stage 02 never removes or reorders those rows"
            ),
            "indicators": [indicator.as_mapping() for indicator in self.indicators],
        }


@dataclass(frozen=True, slots=True)
class _IndicatorSignatureState:
    telemetry: IndicatorRowSignatureTelemetry
    unique_trade_rows: tuple[np.ndarray, ...]


def build_row_signature_telemetry(
    indicator_pools: Sequence[PreparedIndicatorPool],
    *,
    consensus_signature_enumeration_limit: int = DEFAULT_CONSENSUS_SIGNATURE_ENUMERATION_LIMIT,
) -> RowSignatureTelemetry:
    """
    Compute shadow-only duplicate row and consensus-signature telemetry.

    The function does not mutate pools and does not return any structure used by
    candidate planning or scoring. Exact row identity is the SHA-256 digest of
    the row content plus dtype/shape metadata. The sidecar-style u64 digest is
    collision-checked but not used as the equality key.
    """

    if consensus_signature_enumeration_limit <= 0:
        raise ValueError("consensus_signature_enumeration_limit must be > 0")
    states = tuple(_indicator_signature_state(pool) for pool in indicator_pools)
    rows_after_prefilter = sum(state.telemetry.rows_after_prefilter for state in states)
    unique_rows_after_dedup = sum(
        state.telemetry.unique_rows_after_dedup for state in states
    )
    candidate_upper_bound = _product(
        state.telemetry.unique_rows_after_dedup for state in states
    )
    consensus_count, consensus_mode = _consensus_signature_count(
        states=states,
        upper_bound=candidate_upper_bound,
        enumeration_limit=consensus_signature_enumeration_limit,
    )
    return RowSignatureTelemetry(
        rows_after_prefilter=rows_after_prefilter,
        unique_rows_after_dedup=unique_rows_after_dedup,
        duplicate_signal_row_ids={
            state.telemetry.indicator_id: state.telemetry.duplicate_signal_row_ids
            for state in states
        },
        unique_signal_row_ids={
            state.telemetry.indicator_id: state.telemetry.unique_signal_row_ids
            for state in states
        },
        row_signature_collision_count=sum(
            state.telemetry.row_signature_collision_count for state in states
        ),
        consensus_signature_count=consensus_count,
        consensus_signature_mode=consensus_mode,
        consensus_signature_enumeration_limit=consensus_signature_enumeration_limit,
        candidate_upper_bound_after_row_dedup=candidate_upper_bound,
        indicators=tuple(state.telemetry for state in states),
    )


def _indicator_signature_state(pool: PreparedIndicatorPool) -> _IndicatorSignatureState:
    trade_t = np.ascontiguousarray(np.asarray(pool.trade_T, dtype=np.int8))
    row_ids = np.asarray(pool.row_ids, dtype=np.int32)
    if trade_t.ndim != 2 or int(trade_t.shape[0]) != int(row_ids.shape[0]):
        raise ValueError(f"row signature alignment mismatch for {pool.indicator_id!r}")

    first_position_by_full: dict[str, int] = {}
    first_full_by_u64: dict[int, str] = {}
    unique_positions: list[int] = []
    duplicate_row_ids: list[int] = []
    collision_count = 0

    for row_pos in range(int(trade_t.shape[0])):
        row = trade_t[row_pos]
        full_signature, u64_signature = _row_signatures(row)
        existing_full = first_full_by_u64.get(u64_signature)
        if existing_full is None:
            first_full_by_u64[u64_signature] = full_signature
        elif existing_full != full_signature:
            collision_count += 1

        if full_signature in first_position_by_full:
            duplicate_row_ids.append(int(row_ids[row_pos]))
            continue
        first_position_by_full[full_signature] = row_pos
        unique_positions.append(row_pos)

    unique_rows = tuple(
        np.ascontiguousarray(trade_t[position]) for position in unique_positions
    )
    unique_row_ids = tuple(int(row_ids[position]) for position in unique_positions)
    return _IndicatorSignatureState(
        telemetry=IndicatorRowSignatureTelemetry(
            indicator_id=str(pool.indicator_id),
            rows_after_prefilter=int(row_ids.shape[0]),
            unique_rows_after_dedup=len(unique_positions),
            duplicate_signal_row_ids=tuple(duplicate_row_ids),
            unique_signal_row_ids=unique_row_ids,
            row_signature_collision_count=collision_count,
        ),
        unique_trade_rows=unique_rows,
    )


def _row_signatures(row: np.ndarray) -> tuple[str, int]:
    row_i8 = np.ascontiguousarray(np.asarray(row, dtype=np.int8))
    payload = (
        b"dtype=int8;shape="
        + str(tuple(row_i8.shape)).encode("ascii")
        + b";"
        + row_i8.tobytes(order="C")
    )
    full = hashlib.sha256(payload).hexdigest()
    u64 = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return full, u64 % _U64_MODULUS


def _consensus_signature_count(
    *,
    states: Sequence[_IndicatorSignatureState],
    upper_bound: int,
    enumeration_limit: int,
) -> tuple[int, str]:
    if not states:
        return 0, "exact_empty"
    if upper_bound > enumeration_limit:
        return upper_bound, "upper_bound_unique_row_product"

    signatures: set[str] = set()
    unique_rows_by_indicator = tuple(state.unique_trade_rows for state in states)
    for rows in itertools.product(*unique_rows_by_indicator):
        consensus = _consensus_row(rows)
        signatures.add(_row_signatures(consensus)[0])
    return len(signatures), "exact_consensus_enumerated"


def _consensus_row(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty(0, dtype=np.int8)
    stacked = np.vstack([np.asarray(row, dtype=np.int8) for row in rows])
    positive = np.all(stacked == np.int8(1), axis=0)
    negative = np.all(stacked == np.int8(-1), axis=0)
    consensus = np.zeros(stacked.shape[1], dtype=np.int8)
    consensus[positive] = np.int8(1)
    consensus[negative] = np.int8(-1)
    return np.ascontiguousarray(consensus)


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


__all__ = [
    "DEFAULT_CONSENSUS_SIGNATURE_ENUMERATION_LIMIT",
    "IndicatorRowSignatureTelemetry",
    "RowSignatureTelemetry",
    "build_row_signature_telemetry",
]
