from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from heapq import heappush, heapreplace
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.backtest.domain.entities import BacktestJobTopVariant, TradeV1
from trading.contexts.backtest.domain.value_objects import BacktestVariantScalar
from trading.contexts.indicators.application.dto import IndicatorVariantSelection

FrontierSignatureV1 = tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class BacktestJobSnapshotCadenceV1:
    """
    Snapshot cadence policy for Stage-B persisted top variants updates.

    Docs:
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
    """

    snapshot_seconds: int | None
    snapshot_variants_step: int | None

    def __post_init__(self) -> None:
        """
        Validate cadence thresholds with strict positive invariants when configured.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Threshold values are optional and independently configurable.
        Raises:
            ValueError: If provided thresholds are non-positive.
        Side Effects:
            None.
        """
        if self.snapshot_seconds is not None and self.snapshot_seconds <= 0:
            raise ValueError("BacktestJobSnapshotCadenceV1.snapshot_seconds must be > 0")
        if self.snapshot_variants_step is not None and self.snapshot_variants_step <= 0:
            raise ValueError("BacktestJobSnapshotCadenceV1.snapshot_variants_step must be > 0")

    def should_persist(
        self,
        *,
        now: datetime,
        last_persist_at: datetime,
        processed_variants: int,
        last_persist_processed_variants: int,
    ) -> bool:
        """
        Apply EPIC-10 OR trigger semantics for Stage-B snapshot persistence.

        Args:
            now: Current UTC-aware timestamp.
            last_persist_at: Timestamp of previous persisted snapshot.
            processed_variants: Total processed Stage-B variants at current checkpoint.
            last_persist_processed_variants: Processed count captured at previous snapshot.
        Returns:
            bool: `True` when snapshot must be persisted now.
        Assumptions:
            Processed counters are monotonic inside one claim attempt.
        Raises:
            ValueError: If processed counters are invalid.
        Side Effects:
            None.
        """
        if processed_variants < 0:
            raise ValueError("processed_variants must be >= 0")
        if last_persist_processed_variants < 0:
            raise ValueError("last_persist_processed_variants must be >= 0")
        if processed_variants < last_persist_processed_variants:
            raise ValueError(
                "processed_variants must be >= last_persist_processed_variants"
            )

        elapsed_seconds = (now - last_persist_at).total_seconds()
        by_time = (
            self.snapshot_seconds is not None and elapsed_seconds >= float(self.snapshot_seconds)
        )
        by_step = (
            self.snapshot_variants_step is not None
            and (processed_variants - last_persist_processed_variants)
            >= self.snapshot_variants_step
        )
        return by_time or by_step


@dataclass(frozen=True, slots=True)
class BacktestJobTopVariantCandidateV1:
    """
    One deterministic Stage-B candidate retained in running bounded top-K buffer.

    Docs:
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
    """

    variant_index: int
    variant_key: str
    indicator_variant_key: str
    total_return_pct: float
    indicator_selections: tuple[IndicatorVariantSelection, ...]
    signal_params: Mapping[str, Mapping[str, BacktestVariantScalar]]
    risk_params: Mapping[str, BacktestVariantScalar]

    def __post_init__(self) -> None:
        """
        Validate deterministic candidate identity and normalize nested mappings.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Variant keys are canonical lowercase SHA-256 hex strings.
        Raises:
            ValueError: If one identity field is invalid.
        Side Effects:
            Replaces nested mappings with immutable sorted mapping proxies.
        """
        if self.variant_index < 0:
            raise ValueError("BacktestJobTopVariantCandidateV1.variant_index must be >= 0")
        if len(self.variant_key) != 64:
            raise ValueError("BacktestJobTopVariantCandidateV1.variant_key must be 64 hex chars")
        if len(self.indicator_variant_key) != 64:
            raise ValueError(
                "BacktestJobTopVariantCandidateV1.indicator_variant_key must be 64 hex chars"
            )

        object.__setattr__(
            self,
            "signal_params",
            _freeze_nested_scalar_mapping(values=self.signal_params),
        )
        object.__setattr__(
            self,
            "risk_params",
            MappingProxyType(_sorted_scalar_mapping(values=self.risk_params)),
        )
        object.__setattr__(self, "total_return_pct", float(self.total_return_pct))


class BacktestJobTopKBufferV1:
    """
    Bounded deterministic top-K buffer for streaming Stage-B candidate scoring.

    Docs:
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
    """

    def __init__(self, *, limit: int) -> None:
        """
        Initialize bounded top-K buffer with positive capacity.

        Docs:
          - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
          - tests/unit/contexts/backtest/application/services/test_job_runner_streaming_v1.py
        Args:
            limit: Maximum number of retained candidates.
        Returns:
            None.
        Assumptions:
            Capacity remains constant for one job attempt.
        Raises:
            ValueError: If `limit` is non-positive.
        Side Effects:
            Creates empty in-memory candidate buffer.
        """
        if limit <= 0:
            raise ValueError("BacktestJobTopKBufferV1.limit must be > 0")
        self._limit = limit
        self._heap: list[
            tuple[float, tuple[int, ...], BacktestJobTopVariantCandidateV1]
        ] = []

    def include(self, *, candidate: BacktestJobTopVariantCandidateV1) -> None:
        """
        Add one candidate if it belongs to current deterministic top-K frontier.

        Docs:
          - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
          - tests/unit/contexts/backtest/application/services/test_job_runner_streaming_v1.py
        Args:
            candidate: Scored Stage-B candidate.
        Returns:
            None.
        Assumptions:
            Ranking key is `total_return_pct DESC, variant_key ASC`.
        Raises:
            None.
        Side Effects:
            Mutates in-memory bounded buffer.
        """
        if len(self._heap) < self._limit:
            # HOT PATH: bounded heap keeps only current top-K frontier in memory.
            heappush(self._heap, _candidate_heap_entry(candidate=candidate))
            return

        worst = self._heap[0][2]
        if not _candidate_outranks(candidate=candidate, baseline=worst):
            return

        # HOT PATH: replace current worst retained candidate in O(log K).
        heapreplace(self._heap, _candidate_heap_entry(candidate=candidate))

    def ranked(self) -> tuple[BacktestJobTopVariantCandidateV1, ...]:
        """
        Return deterministic ranked candidates snapshot from buffer.

        Docs:
          - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
          - tests/unit/contexts/backtest/application/services/test_job_runner_streaming_v1.py
        Args:
            None.
        Returns:
            tuple[BacktestJobTopVariantCandidateV1, ...]: Ranked candidates.
        Assumptions:
            Internal heap keeps only bounded frontier; ranking is materialized on read.
        Raises:
            None.
        Side Effects:
            None.
        """
        return tuple(sorted((entry[2] for entry in self._heap), key=_candidate_rank_key))

    def __len__(self) -> int:
        """
        Return number of currently retained candidates.

        Docs:
          - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
          - tests/unit/contexts/backtest/application/services/test_job_runner_streaming_v1.py
        Args:
            None.
        Returns:
            int: Retained candidates count.
        Assumptions:
            Count is always in `[0, limit]`.
        Raises:
            None.
        Side Effects:
            None.
        """
        return len(self._heap)


def build_running_snapshot_rows(
    *,
    job_id: UUID,
    now: datetime,
    ranked_candidates: tuple[BacktestJobTopVariantCandidateV1, ...],
    direction_mode: str,
    sizing_mode: str,
    execution_params: Mapping[str, BacktestVariantScalar],
) -> tuple[BacktestJobTopVariant, ...]:
    """
    Build persisted running snapshot rows with `report_table_md` and trades set to null.

    Args:
        job_id: Job identifier.
        now: Snapshot timestamp in UTC.
        ranked_candidates: Ranked Stage-B candidates.
        direction_mode: Effective direction mode for variant payload.
        sizing_mode: Effective sizing mode for variant payload.
        execution_params: Effective execution parameters payload.
    Returns:
        tuple[BacktestJobTopVariant, ...]: Deterministic running snapshot rows.
    Assumptions:
        Candidates are already ordered by final deterministic ranking contract.
    Raises:
        ValueError: If one candidate cannot be converted into storage row.
    Side Effects:
        None.
    """
    rows: list[BacktestJobTopVariant] = []
    for rank, candidate in enumerate(ranked_candidates, start=1):
        rows.append(
            BacktestJobTopVariant(
                job_id=job_id,
                rank=rank,
                variant_key=candidate.variant_key,
                indicator_variant_key=candidate.indicator_variant_key,
                variant_index=candidate.variant_index,
                total_return_pct=candidate.total_return_pct,
                payload_json=build_variant_payload_json(
                    candidate=candidate,
                    direction_mode=direction_mode,
                    sizing_mode=sizing_mode,
                    execution_params=execution_params,
                ),
                report_table_md=None,
                trades_json=None,
                updated_at=now,
            )
        )
    return tuple(rows)


def build_frontier_signature(
    *,
    ranked_candidates: tuple[BacktestJobTopVariantCandidateV1, ...],
) -> FrontierSignatureV1:
    """
    Build deterministic Stage-B frontier signature from ranked candidates for snapshot gating.

    Docs:
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py
    Args:
        ranked_candidates: Deterministically ranked persisted Stage-B candidates.
    Returns:
        FrontierSignatureV1:
            Frontier signature built from ordered `(variant_key, total_return_pct)` entries.
    Assumptions:
        Candidate order is final deterministic ranking order used for persistence writes.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple(
        (candidate.variant_key, candidate.total_return_pct)
        for candidate in ranked_candidates
    )


def build_finalized_snapshot_rows(
    *,
    job_id: UUID,
    now: datetime,
    ranked_candidates: tuple[BacktestJobTopVariantCandidateV1, ...],
    direction_mode: str,
    sizing_mode: str,
    execution_params: Mapping[str, BacktestVariantScalar],
    reports_by_variant_key: Mapping[str, str],
    trades_by_variant_key: Mapping[str, tuple[TradeV1, ...] | None],
) -> tuple[BacktestJobTopVariant, ...]:
    """
    Build finalized succeeded snapshot rows with `report_table_md` and ranked trades policy.

    Args:
        job_id: Job identifier.
        now: Snapshot timestamp in UTC.
        ranked_candidates: Ranked persisted top candidates.
        direction_mode: Effective direction mode for payload.
        sizing_mode: Effective sizing mode for payload.
        execution_params: Effective execution parameters payload.
        reports_by_variant_key: Markdown report table by variant key.
        trades_by_variant_key: Optional trades tuple by variant key.
    Returns:
        tuple[BacktestJobTopVariant, ...]: Deterministic finalized rows.
    Assumptions:
        Report markdown is present for every persisted candidate in succeeded finalization.
    Raises:
        ValueError: If one candidate does not have required report markdown payload.
    Side Effects:
        None.
    """
    rows: list[BacktestJobTopVariant] = []
    for rank, candidate in enumerate(ranked_candidates, start=1):
        report_table_md = reports_by_variant_key.get(candidate.variant_key)
        if report_table_md is None:
            raise ValueError(
                "build_finalized_snapshot_rows requires report markdown for persisted candidate"
            )
        trades = trades_by_variant_key.get(candidate.variant_key)
        rows.append(
            BacktestJobTopVariant(
                job_id=job_id,
                rank=rank,
                variant_key=candidate.variant_key,
                indicator_variant_key=candidate.indicator_variant_key,
                variant_index=candidate.variant_index,
                total_return_pct=candidate.total_return_pct,
                payload_json=build_variant_payload_json(
                    candidate=candidate,
                    direction_mode=direction_mode,
                    sizing_mode=sizing_mode,
                    execution_params=execution_params,
                ),
                report_table_md=report_table_md,
                trades_json=build_trades_json_payload(trades=trades),
                updated_at=now,
            )
        )
    return tuple(rows)


def build_variant_payload_json(
    *,
    candidate: BacktestJobTopVariantCandidateV1,
    direction_mode: str,
    sizing_mode: str,
    execution_params: Mapping[str, BacktestVariantScalar],
) -> Mapping[str, Any]:
    """
    Build deterministic JSON payload for persisted top-variant `payload_json` column.

    Args:
        candidate: Ranked Stage-B candidate payload.
        direction_mode: Effective direction mode.
        sizing_mode: Effective sizing mode.
        execution_params: Effective execution scalar payload.
    Returns:
        Mapping[str, Any]: Deterministic JSON object payload.
    Assumptions:
        Payload shape mirrors API variant payload fields used by EPIC-11 reads.
    Raises:
        None.
    Side Effects:
        None.
    """
    indicator_selections = tuple(
        sorted(candidate.indicator_selections, key=lambda item: item.indicator_id)
    )
    return {
        "indicator_selections": [
            {
                "indicator_id": selection.indicator_id,
                "inputs": {
                    key: selection.inputs[key] for key in sorted(selection.inputs.keys())
                },
                "params": {
                    key: selection.params[key] for key in sorted(selection.params.keys())
                },
            }
            for selection in indicator_selections
        ],
        "signal_params": _to_plain_nested_scalar_mapping(values=candidate.signal_params),
        "risk_params": _to_plain_scalar_mapping(values=candidate.risk_params),
        "execution_params": _to_plain_scalar_mapping(values=execution_params),
        "direction_mode": direction_mode,
        "sizing_mode": sizing_mode,
    }


def build_trades_json_payload(
    *,
    trades: tuple[TradeV1, ...] | None,
) -> tuple[Mapping[str, Any], ...] | None:
    """
    Convert optional deterministic trades tuple into JSON-ready storage payload.

    Args:
        trades: Optional trades payload.
    Returns:
        tuple[Mapping[str, Any], ...] | None: JSON-ready trades payload or `None`.
    Assumptions:
        Trades are already ranked by deterministic `trade_id ASC`.
    Raises:
        None.
    Side Effects:
        None.
    """
    if trades is None:
        return None
    return tuple(
        {
            "trade_id": trade.trade_id,
            "direction": trade.direction,
            "entry_bar_index": trade.entry_bar_index,
            "exit_bar_index": trade.exit_bar_index,
            "entry_fill_price": trade.entry_fill_price,
            "exit_fill_price": trade.exit_fill_price,
            "qty_base": trade.qty_base,
            "entry_quote_amount": trade.entry_quote_amount,
            "exit_quote_amount": trade.exit_quote_amount,
            "entry_fee_quote": trade.entry_fee_quote,
            "exit_fee_quote": trade.exit_fee_quote,
            "gross_pnl_quote": trade.gross_pnl_quote,
            "net_pnl_quote": trade.net_pnl_quote,
            "locked_profit_quote": trade.locked_profit_quote,
            "exit_reason": trade.exit_reason,
        }
        for trade in trades
    )


def _candidate_rank_key(
    candidate: BacktestJobTopVariantCandidateV1,
) -> tuple[float, str]:
    """
    Build deterministic candidate rank key for sorting (`return desc`, `variant_key asc`).

    Args:
        candidate: Candidate payload.
    Returns:
        tuple[float, str]: Deterministic sort key.
    Assumptions:
        Lower tuple value means better rank (`-return` first, then key).
    Raises:
        None.
    Side Effects:
        None.
    """
    return (-candidate.total_return_pct, candidate.variant_key)


def _candidate_heap_entry(
    *,
    candidate: BacktestJobTopVariantCandidateV1,
) -> tuple[float, tuple[int, ...], BacktestJobTopVariantCandidateV1]:
    """
    Build heap entry where root is always the worst retained candidate.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
      - tests/unit/contexts/backtest/application/services/test_job_runner_streaming_v1.py
    Args:
        candidate: Stage-B scored candidate.
    Returns:
        tuple[float, tuple[int, ...], BacktestJobTopVariantCandidateV1]:
            Heap entry ordered by `total_return_pct ASC, variant_key DESC`.
    Assumptions:
        Root entry represents current deterministic worst candidate inside bounded top-K set.
    Raises:
        None.
    Side Effects:
        Allocates one tuple for each retained heap insertion/replacement.
    """
    return (
        candidate.total_return_pct,
        _descending_text_key(value=candidate.variant_key),
        candidate,
    )


def _candidate_outranks(
    *,
    candidate: BacktestJobTopVariantCandidateV1,
    baseline: BacktestJobTopVariantCandidateV1,
) -> bool:
    """
    Check whether candidate outranks baseline by deterministic Stage-B ranking key.

    Args:
        candidate: New candidate.
        baseline: Baseline candidate.
    Returns:
        bool: `True` when candidate has higher deterministic rank.
    Assumptions:
        Ranking key is fixed to `total_return_pct DESC, variant_key ASC`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _candidate_rank_key(candidate) < _candidate_rank_key(baseline)


def _descending_text_key(*, value: str) -> tuple[int, ...]:
    """
    Encode text into tuple comparable in reverse lexicographical order.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
      - tests/unit/contexts/backtest/application/services/test_job_runner_streaming_v1.py
    Args:
        value: Deterministic tie-break key.
    Returns:
        tuple[int, ...]:
            Comparable tuple where lexicographically larger strings become smaller values.
    Assumptions:
        Sentinel `0` handles prefix ordering and is greater than any negated code point.
    Raises:
        None.
    Side Effects:
        Allocates tuple used only for retained heap entries.
    """
    return (*(-ord(char) for char in value), 0)


def _sorted_scalar_mapping(
    *,
    values: Mapping[str, BacktestVariantScalar],
) -> dict[str, BacktestVariantScalar]:
    """
    Normalize scalar mapping into deterministic key-sorted plain mapping.

    Args:
        values: Scalar mapping payload.
    Returns:
        dict[str, BacktestVariantScalar]: Deterministic normalized mapping.
    Assumptions:
        Mapping keys can be represented as non-empty strings.
    Raises:
        ValueError: If key is blank after normalization.
    Side Effects:
        None.
    """
    normalized: dict[str, BacktestVariantScalar] = {}
    for raw_key in sorted(values.keys(), key=lambda item: str(item)):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("mapping keys must be non-empty")
        normalized[key] = values[raw_key]
    return normalized


def _freeze_nested_scalar_mapping(
    *,
    values: Mapping[str, Mapping[str, BacktestVariantScalar]],
) -> Mapping[str, Mapping[str, BacktestVariantScalar]]:
    """
    Freeze nested scalar mapping into deterministic immutable mapping proxies.

    Args:
        values: Nested mapping payload.
    Returns:
        Mapping[str, Mapping[str, BacktestVariantScalar]]: Frozen deterministic mapping.
    Assumptions:
        Nested mapping represents `indicator_id -> signal_param -> scalar`.
    Raises:
        ValueError: If one nested key is blank.
    Side Effects:
        None.
    """
    normalized: dict[str, Mapping[str, BacktestVariantScalar]] = {}
    for raw_indicator_id in sorted(values.keys(), key=lambda item: str(item)):
        indicator_id = str(raw_indicator_id).strip()
        if not indicator_id:
            raise ValueError("nested mapping first-level keys must be non-empty")
        normalized[indicator_id] = MappingProxyType(
            _sorted_scalar_mapping(values=values[raw_indicator_id])
        )
    return MappingProxyType(normalized)


def _to_plain_scalar_mapping(
    *,
    values: Mapping[str, BacktestVariantScalar],
) -> dict[str, BacktestVariantScalar]:
    """
    Convert scalar mapping into deterministic key-sorted plain dictionary.

    Args:
        values: Scalar mapping payload.
    Returns:
        dict[str, BacktestVariantScalar]: Deterministic plain mapping.
    Assumptions:
        Keys are already normalized by upstream DTO/domain constructors.
    Raises:
        ValueError: If one key is blank.
    Side Effects:
        None.
    """
    return _sorted_scalar_mapping(values=values)


def _to_plain_nested_scalar_mapping(
    *,
    values: Mapping[str, Mapping[str, BacktestVariantScalar]],
) -> dict[str, dict[str, BacktestVariantScalar]]:
    """
    Convert nested scalar mapping into deterministic plain dictionaries.

    Args:
        values: Nested mapping payload.
    Returns:
        dict[str, dict[str, BacktestVariantScalar]]: Deterministic plain nested mapping.
    Assumptions:
        Nested keys are normalized and represent deterministic signal payload identity.
    Raises:
        ValueError: If one nested key is blank.
    Side Effects:
        None.
    """
    normalized: dict[str, dict[str, BacktestVariantScalar]] = {}
    for raw_indicator_id in sorted(values.keys(), key=lambda item: str(item)):
        indicator_id = str(raw_indicator_id).strip()
        if not indicator_id:
            raise ValueError("nested mapping first-level keys must be non-empty")
        normalized[indicator_id] = _sorted_scalar_mapping(values=values[raw_indicator_id])
    return normalized


__all__ = [
    "FrontierSignatureV1",
    "BacktestJobSnapshotCadenceV1",
    "BacktestJobTopKBufferV1",
    "BacktestJobTopVariantCandidateV1",
    "build_frontier_signature",
    "build_finalized_snapshot_rows",
    "build_running_snapshot_rows",
    "build_trades_json_payload",
    "build_variant_payload_json",
]
