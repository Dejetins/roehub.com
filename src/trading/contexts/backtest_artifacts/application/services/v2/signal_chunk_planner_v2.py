"""Deterministic ChunkPlanner for artifact-only signal materialization."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import (
    ArtifactSignalChunkJobV2,
    ArtifactSignalChunkPlanningRequestV2,
    ChunkPlannerV2,
)


@dataclass(frozen=True, slots=True)
class DeterministicSignalChunkPlannerV2(ChunkPlannerV2):
    """
    Production ChunkPlanner implementation for bounded artifact signal execution.

    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    def plan(
        self,
        *,
        request: ArtifactSignalChunkPlanningRequestV2,
    ) -> tuple[ArtifactSignalChunkJobV2, ...]:
        """
        Plan contiguous deterministic row slices for one `(indicator_id, timeframe)` target.

        Args:
            request: Typed planner request including row count, timeline, and worker budget.
        Returns:
            tuple[ArtifactSignalChunkJobV2, ...]: Ordered non-overlapping chunk jobs.
        Assumptions:
            `estimated_bytes_per_row` already accounts for chunk-local scratch plus the writable
            row slice inside `signals/<tf>/<indicator_id>/signals.i8.npy`.
        Raises:
            ValueError: If the configured worker budget cannot fit the minimum chunk size.
        Side Effects:
            None.
        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        budget_cap_rows = request.worker_memory_budget_bytes // request.estimated_bytes_per_row
        effective_chunk_rows_min = min(request.signal_chunk_rows_min, request.variant_count)
        if budget_cap_rows < effective_chunk_rows_min:
            raise ValueError(
                "ChunkPlanner cannot fit the configured signal chunk rows into the worker "
                "budget: "
                f"indicator_id={request.indicator_id!r}, timeframe={request.timeframe!r}, "
                f"worker_memory_budget_bytes={request.worker_memory_budget_bytes!r}, "
                f"estimated_bytes_per_row={request.estimated_bytes_per_row!r}, "
                f"signal_chunk_rows_min={request.signal_chunk_rows_min!r}, "
                f"variant_count={request.variant_count!r}"
            )
        chunk_rows = min(
            request.signal_chunk_rows_max,
            request.variant_count,
            max(effective_chunk_rows_min, budget_cap_rows),
        )
        chunk_count = int(math.ceil(request.variant_count / chunk_rows))
        jobs: list[ArtifactSignalChunkJobV2] = []
        for chunk_index in range(chunk_count):
            row_start_inclusive = chunk_index * chunk_rows
            row_end_exclusive = min(request.variant_count, row_start_inclusive + chunk_rows)
            jobs.append(
                ArtifactSignalChunkJobV2(
                    indicator_id=request.indicator_id,
                    timeframe=request.timeframe,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    row_start_inclusive=row_start_inclusive,
                    row_end_exclusive=row_end_exclusive,
                    chunk_rows=row_end_exclusive - row_start_inclusive,
                )
            )
        return tuple(jobs)

