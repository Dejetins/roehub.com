from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, Mapping, cast
from uuid import uuid4

import numpy as np
import pytest

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCostEstimate,
    BacktestPreflightResult,
    BacktestPreparePoolsResult,
    PreparedExecutionMapping,
    PreparedIndicatorPool,
    PreparedIndicatorRowMetadata,
    PreparePoolsTiming,
)
from trading.contexts.backtest.application.services.v2.job_orchestration import (
    SAMPLE_WARMUP_STAGE_NAME,
    SERVICE_TOTAL_WITHOUT_WARMUP_STAGE_NAME,
    BacktestRuntimeJobOrchestrationService,
)
from trading.contexts.backtest.application.services.v2.prepare_pools import (
    build_signal_segments,
    row_metadata_order_hash,
)


def test_no_risk_job_runs_sample_warmup_before_measured_exact() -> None:
    prepared = _prepared_result(rows=3)
    exact = _ExactService()
    service = BacktestRuntimeJobOrchestrationService(
        prepare_pools=_PreparePools(prepared),
        combo_planning=_ComboPlanning(),
        no_risk_exact=exact,
        tp_sl_hit_times=_UnusedService(),
        tp_sl_exact=_UnusedService(),
        artifact_array_loader=_UnusedService(),
        top_result_assembly=cast(Any, _TopResultAssembly()),
    )

    result = service.execute(
        job_id=uuid4(),
        preflight=_preflight(top_n=50, risk_mode="none"),
        updated_at=datetime.now(UTC),
    )

    assert [call["top_n"] for call in exact.calls] == [1, 50]
    assert [call["rows"] for call in exact.calls] == [2, 3]
    assert result.stage_timings[SAMPLE_WARMUP_STAGE_NAME] >= 0.0
    assert result.stage_timings[SERVICE_TOTAL_WITHOUT_WARMUP_STAGE_NAME] == pytest.approx(
        0.1 + 0.2 + 0.3 + 0.4
    )
    counters = result.instrumentation_counters
    assert counters["artifact_load_ms"] is None
    assert counters["signals_pack_ms"] is None
    assert counters["row_signature_ms"] is not None
    assert counters["row_signature_ms"] >= 0.0
    assert counters["combo_iteration_ms"] == pytest.approx(200.0)
    assert counters["exact_scoring_ms"] == pytest.approx(300.0)
    assert counters["top_result_assembly_ms"] == pytest.approx(400.0)
    assert counters["rows_before_prefilter"] is None
    assert counters["rows_after_prefilter"] == 3
    assert counters["unique_rows_after_dedup"] == 1
    assert counters["duplicate_signal_row_ids"] == {"ma.dema": [1, 2]}
    assert counters["row_signature_collision_count"] == 0
    assert counters["consensus_signature_count"] == 1
    assert counters["consensus_signature_mode"] == "exact_consensus_enumerated"
    assert counters["candidate_upper_bound_after_row_dedup"] == 1
    assert counters["combo_count_planned"] == 3
    assert counters["candidates_after_proxy"] == 3
    assert counters["exact_candidates"] == 3
    assert counters["exact_candidates_per_sec"] == pytest.approx(10.0)
    assert counters["trade_cell_evals_per_sec"] is None


@dataclass(slots=True)
class _PreparePools:
    prepared: BacktestPreparePoolsResult

    def execute(self, **_: object) -> BacktestPreparePoolsResult:
        return self.prepared


class _ComboPlanning:
    def execute(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        normalized_request: Mapping[str, Any],
    ) -> SimpleNamespace:
        del normalized_request
        return SimpleNamespace(
            prepared_rows=int(prepared_result.indicator_pools[0].row_ids.shape[0]),
            telemetry=SimpleNamespace(
                as_mapping=lambda: {
                    "stage_timings": {"combo_iteration": 0.2},
                    "cartesian_combinations": 3,
                    "proxy_candidates_selected": 3,
                },
                stage_timings={"combo_iteration": 0.2},
                cartesian_combinations=3,
                proxy_candidates_selected=3,
            ),
        )


class _ExactService:
    def __init__(self) -> None:
        self.calls: list[dict[str, int]] = []

    def execute(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        combo_planning_result: SimpleNamespace,
        normalized_request: Mapping[str, Any],
    ) -> SimpleNamespace:
        rows = int(prepared_result.indicator_pools[0].row_ids.shape[0])
        assert rows == combo_planning_result.prepared_rows
        self.calls.append({"top_n": int(normalized_request["top_n"]), "rows": rows})
        return SimpleNamespace(
            top_results=(),
            telemetry=SimpleNamespace(
                as_mapping=lambda: {
                    "stage_timings": {"exact_scoring": 0.3},
                    "top_results_count": int(normalized_request["top_n"]),
                },
                stage_timings={"exact_scoring": 0.3},
                exact_candidates_evaluated=3,
            ),
            self_check=SimpleNamespace(as_mapping=lambda: {"status": "not_run"}),
            memory_cleanup_evidence=SimpleNamespace(as_mapping=lambda: {"pass": True}),
        )


class _TopResultAssembly:
    stage_timings = {"top_result_assembly": 0.4}
    top_variants: tuple[object, ...] = ()
    summary_hash = "summary"

    def assemble(self, **_: object) -> object:
        return self


class _UnusedService:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"unexpected service call: {name}")


def _preflight(*, top_n: int, risk_mode: str) -> BacktestPreflightResult:
    return BacktestPreflightResult(
        normalized_request={
            "coordinates": {
                "exchange": "binance",
                "market_type": "spot",
                "symbol": "BTCUSDT",
            },
            "risk": {"mode": risk_mode},
            "top_n": top_n,
        },
        request_hash="request",
        result_config_hash="result",
        artifact_metadata=BacktestArtifactMetadata(
            artifact_slot="current",
            artifact_slot_generation=1,
            artifact_manifest_hash="artifact",
            artifact_asof_date="2026-05-14",
            hit_times_manifest_hash="hit-times",
            published_at_utc="2026-05-14T00:00:00Z",
        ),
        cost_estimate=BacktestCostEstimate(
            indicator_rows=3,
            candidate_combinations=3,
            tp_sl_cells=0,
            cost_class="heavy",
        ),
    )


def _prepared_result(*, rows: int) -> BacktestPreparePoolsResult:
    pools = (_pool(indicator_id="ma.dema", rows=rows),)
    return BacktestPreparePoolsResult(
        timeframe="15m",
        indicator_ids=("ma.dema",),
        indicator_pools=pools,
        signal_returns_15m=np.zeros(4, dtype=np.float32),
        execution_mapping=PreparedExecutionMapping(
            signal_entry_exec_idx_15m=np.arange(4, dtype=np.int32),
            run_bar_open_1m_idx_15m=np.arange(4, dtype=np.int32),
            run_bar_close_1m_idx_15m=np.arange(4, dtype=np.int32),
            t_exec_limit_1m=4,
        ),
        time_slice_start_15m=0,
        time_slice_stop_15m=4,
        trade_T_length=4,
        eval_T_length=4,
        row_metadata_order_hash=row_metadata_order_hash(pools),
        timing=PreparePoolsTiming(
            stage_name="prepare_pools_total",
            wall_time_s=0.1,
            subsegments={"prepare_pools_core": 0.1},
        ),
        execution_open_1m=np.ones(4, dtype=np.float32),
        execution_close_1m=np.ones(4, dtype=np.float32),
    )


def _pool(*, indicator_id: str, rows: int) -> PreparedIndicatorPool:
    row_ids = np.arange(rows, dtype=np.int32)
    trade_t = np.ones((rows, 4), dtype=np.int8)
    change_count = np.zeros(rows, dtype=np.int32)
    return PreparedIndicatorPool(
        indicator_id=indicator_id,
        row_ids=row_ids,
        filtered_row_ids=row_ids.copy(),
        trade_T=trade_t,
        eval_T=trade_t.copy(),
        segments=build_signal_segments(trade_t, change_count=change_count),
        row_score=np.ones(rows, dtype=np.float32),
        score_adj=np.ones(rows, dtype=np.float32),
        nonzero=np.ones(rows, dtype=np.int32),
        proxy=np.ones(rows, dtype=np.float32),
        change_count=change_count,
        metadata=tuple(
            PreparedIndicatorRowMetadata(
                indicator_id=indicator_id,
                row_id=index,
                source="close",
                window=index + 5,
            )
            for index in range(rows)
        ),
    )
