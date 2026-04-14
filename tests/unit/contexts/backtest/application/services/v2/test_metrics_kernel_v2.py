from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trading.contexts.backtest.application.services.v2 import (
    StageACompactTradeV2,
    StageBBestCellReplayCaseV2,
    StageBHitTimesFixtureV2,
    StageBHitTimesSliceV2,
    StageBLevelFactorsV2,
    StageBReplayPayloadV2,
    StageBTradeExitV2,
    build_execution_outcome_from_replay_v2,
    compute_stage_b_metrics_v2,
    load_stage_b_golden_fixture_catalog_v2,
    replay_risk_cell_exact_v2,
    stage_b_metrics_to_ranking_payload_v2,
)
from trading.contexts.backtest.application.services.v2.metrics_kernel import (
    normalize_persisted_summary_metrics_v2,
)
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "stage_b_golden_fixtures_v2.json"


def test_compute_stage_b_metrics_v2_matches_best_cell_golden_fixture() -> None:
    """
    Verify exact replay metrics match the committed R5-03 best-cell fixture oracle.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stage B runtime metrics must preserve the golden oracle semantics for ranking and summary
        payloads over compact trades.
    Raises:
        AssertionError: If one deterministic Stage B metric drifts from the golden fixture.
    Side Effects:
        Reads the committed Stage B golden fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
    """
    case = _best_cell_case_v2()
    hit_times = _hit_times_slice_from_fixture_v2(
        hit_times=case.hit_times,
        level_factors=case.level_factors,
    )
    replay = replay_risk_cell_exact_v2(
        compact_trades=_compact_trades_from_best_cell_case_v2(case=case),
        hit_times=hit_times,
        exec_open=np.asarray(case.prices.exec_open, dtype=np.float64),
        exec_close=np.asarray(case.prices.exec_close, dtype=np.float64),
        tp_index=case.expected_result.best_tp_index,
        sl_index=case.expected_result.best_sl_index,
        close_on_end=case.close_on_end,
    )
    metrics = compute_stage_b_metrics_v2(
        replay=replay,
        fee_rate=float(case.fee_rate),
        bars_per_year_exec=float(case.bars_per_year_exec),
    )

    assert metrics.total_return_pct == pytest.approx(
        float(case.expected_result.metrics.total_return) * 100.0,
        rel=1e-9,
        abs=1e-9,
    )
    assert metrics.max_drawdown_pct == pytest.approx(
        float(case.expected_result.metrics.max_drawdown) * 100.0,
        rel=1e-9,
        abs=1e-9,
    )
    assert metrics.sharpe_trades == pytest.approx(
        float(case.expected_result.metrics.sharpe),
        rel=1e-9,
        abs=1e-9,
    )
    assert metrics.win_rate_pct == pytest.approx(
        float(case.expected_result.metrics.winrate) * 100.0,
        rel=1e-9,
        abs=1e-9,
    )
    assert metrics.avg_trade_ret_pct == pytest.approx(
        float(case.expected_result.metrics.avg_trade_return) * 100.0,
        rel=1e-9,
        abs=1e-9,
    )
    assert metrics.avg_trade_exec_bars == pytest.approx(
        float(case.expected_result.metrics.avg_trade_bars),
        rel=1e-9,
        abs=1e-9,
    )
    assert metrics.exposure_pct == pytest.approx(
        float(case.expected_result.metrics.exposure) * 100.0,
        rel=1e-9,
        abs=1e-9,
    )
    assert metrics.trade_count == case.expected_result.trade_count


def test_stage_b_metrics_to_ranking_payload_v2_exposes_stable_aliases() -> None:
    """
    Verify Stage B metrics export stable ranking aliases expected by staged runner contracts.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public ranking payload keeps v1 metric keys/literals stable while Stage B v2 lands
        additively.
    Raises:
        AssertionError: If required aliases drift or numeric values are rewritten.
    Side Effects:
        Reads the committed Stage B golden fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
    """
    case = _best_cell_case_v2()
    metrics = compute_stage_b_metrics_v2(
        replay=replay_risk_cell_exact_v2(
            compact_trades=_compact_trades_from_best_cell_case_v2(case=case),
            hit_times=_hit_times_slice_from_fixture_v2(
                hit_times=case.hit_times,
                level_factors=case.level_factors,
            ),
            exec_open=np.asarray(case.prices.exec_open, dtype=np.float64),
            exec_close=np.asarray(case.prices.exec_close, dtype=np.float64),
            tp_index=case.expected_result.best_tp_index,
            sl_index=case.expected_result.best_sl_index,
            close_on_end=case.close_on_end,
        ),
        fee_rate=float(case.fee_rate),
        bars_per_year_exec=float(case.bars_per_year_exec),
    )

    payload = stage_b_metrics_to_ranking_payload_v2(metrics=metrics)

    assert payload["total_return_pct"] == pytest.approx(metrics.total_return_pct)
    assert payload["Total Return [%]"] == pytest.approx(metrics.total_return_pct)
    assert payload["max_drawdown_pct"] == pytest.approx(metrics.max_drawdown_pct)
    assert payload["Max. Drawdown [%]"] == pytest.approx(metrics.max_drawdown_pct)
    assert payload["return_over_max_drawdown"] == pytest.approx(metrics.return_over_max_drawdown)
    assert payload["profit_factor"] == pytest.approx(metrics.profit_factor)


def test_normalize_persisted_summary_metrics_v2_drops_non_finite_values_only() -> None:
    """
    Verify persisted summary metrics drop `Infinity`/`NaN` while raw ranking semantics stay intact.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stage B ranking may legitimately use non-finite raw metrics, but persisted summary JSON
        must keep only finite numeric values.
    Raises:
        AssertionError: If raw ranking aliases are rewritten or persisted summary sanitization
            keeps non-finite values.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
    """
    replay = StageBReplayPayloadV2(
        tp_index=0,
        sl_index=0,
        sentinel_index=4,
        close_on_end=True,
        trade_exits=(
            StageBTradeExitV2(
                trade_index=0,
                entry_exec_idx=0,
                direction=1,
                sig_exit_exec_idx=1,
                exit_exec_idx=1,
                exit_reason="tp",
                gross_factor=1.10,
                closed=True,
            ),
        ),
    )

    metrics = compute_stage_b_metrics_v2(
        replay=replay,
        fee_rate=0.0,
        bars_per_year_exec=365.0,
    )
    payload = stage_b_metrics_to_ranking_payload_v2(metrics=metrics)
    summary = normalize_persisted_summary_metrics_v2(
        metrics={
            "profit_factor": payload["profit_factor"],
            "return_over_max_drawdown": payload["return_over_max_drawdown"],
            "sharpe_trades": float("nan"),
            "total_return_pct": payload["total_return_pct"],
        }
    )

    assert np.isinf(payload["profit_factor"])
    assert np.isinf(payload["return_over_max_drawdown"])
    assert "profit_factor" not in summary
    assert "return_over_max_drawdown" not in summary
    assert "sharpe_trades" not in summary
    assert summary["total_return_pct"] == pytest.approx(payload["total_return_pct"])


def test_build_execution_outcome_from_replay_v2_materializes_closed_stage_b_trades() -> None:
    """
    Verify exact Stage B replay can materialize details-compatible closed trades deterministically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Details/reporting flow consumes exact replay after ranking and should not redefine exit
        semantics already fixed by the replay payload.
    Raises:
        AssertionError: If trade bodies or top-level totals drift from the replay facts.
    Side Effects:
        Reads the committed Stage B golden fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
      - src/trading/contexts/backtest/application/services/reporting_service_v1.py
    """
    case = _best_cell_case_v2()
    hit_times = _hit_times_slice_from_fixture_v2(
        hit_times=case.hit_times,
        level_factors=case.level_factors,
    )
    replay = replay_risk_cell_exact_v2(
        compact_trades=_compact_trades_from_best_cell_case_v2(case=case),
        hit_times=hit_times,
        exec_open=np.asarray(case.prices.exec_open, dtype=np.float64),
        exec_close=np.asarray(case.prices.exec_close, dtype=np.float64),
        tp_index=case.expected_result.best_tp_index,
        sl_index=case.expected_result.best_sl_index,
        close_on_end=case.close_on_end,
    )
    metrics = compute_stage_b_metrics_v2(
        replay=replay,
        fee_rate=float(case.fee_rate),
        bars_per_year_exec=float(case.bars_per_year_exec),
    )

    outcome = build_execution_outcome_from_replay_v2(
        replay=replay,
        metrics=metrics,
        execution_params=ExecutionParamsV1(
            direction_mode="long-short",
            sizing_mode="all_in",
            init_cash_quote=1000.0,
            fixed_quote=100.0,
            safe_profit_percent=30.0,
            fee_pct=0.0,
            slippage_pct=0.0,
        ),
        exec_open=np.asarray(case.prices.exec_open, dtype=np.float64),
        exec_close=np.asarray(case.prices.exec_close, dtype=np.float64),
        tp_values=hit_times.tp_values,
        sl_values=hit_times.sl_values,
    )

    assert len(outcome.trades) == case.expected_result.trade_count
    assert tuple(trade.exit_reason for trade in outcome.trades) == ("tp", "tp")
    assert outcome.total_return_pct == pytest.approx(metrics.total_return_pct, rel=1e-9, abs=1e-9)
    assert outcome.equity_end_quote == pytest.approx(1562.5, rel=1e-9, abs=1e-9)
    assert outcome.available_quote == pytest.approx(1562.5, rel=1e-9, abs=1e-9)
    assert outcome.safe_quote == pytest.approx(0.0, rel=1e-9, abs=1e-9)


def _best_cell_case_v2() -> StageBBestCellReplayCaseV2:
    """
    Return the committed best-cell replay fixture used by Stage B metrics tests.

    Args:
        None.
    Returns:
        StageBBestCellReplayCaseV2: Typed golden replay case.
    Assumptions:
        The catalog contains exactly one best-cell replay case for R5-03 semantics.
    Raises:
        AssertionError: If the expected case is missing.
    Side Effects:
        Reads the committed Stage B golden fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py
    """
    catalog = load_stage_b_golden_fixture_catalog_v2(path=_FIXTURE_PATH)
    for case in catalog.cases:
        if isinstance(case, StageBBestCellReplayCaseV2):
            return case
    raise AssertionError("best-cell replay case is missing from the golden catalog")


def _hit_times_slice_from_fixture_v2(
    *,
    hit_times: StageBHitTimesFixtureV2,
    level_factors: StageBLevelFactorsV2,
) -> StageBHitTimesSliceV2:
    """
    Convert typed golden fixture hit-times into the runtime Stage B slice contract.

    Args:
        hit_times: Typed golden fixture hit-times tables.
        level_factors: Typed golden fixture gross factors used to derive TP/SL grid rates.
    Returns:
        StageBHitTimesSliceV2: Runtime-ready local hit-times slice.
    Assumptions:
        Golden fixtures use identical TP/SL level semantics for long and short directions.
    Raises:
        None.
    Side Effects:
        Allocates deterministic NumPy arrays for kernel inputs.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py
    """
    return StageBHitTimesSliceV2(
        tp_values=np.asarray(
            tuple(float(value) - 1.0 for value in level_factors.long_tp),
            dtype=np.float32,
        ),
        sl_values=np.asarray(
            tuple(1.0 - float(value) for value in level_factors.long_sl),
            dtype=np.float32,
        ),
        long_tp=np.asarray(hit_times.long_tp, dtype=np.int64),
        long_sl=np.asarray(hit_times.long_sl, dtype=np.int64),
        short_tp=np.asarray(hit_times.short_tp, dtype=np.int64),
        short_sl=np.asarray(hit_times.short_sl, dtype=np.int64),
        sentinel_index=hit_times.sentinel_index,
    )


def _compact_trades_from_best_cell_case_v2(
    *,
    case: StageBBestCellReplayCaseV2,
) -> tuple[StageACompactTradeV2, ...]:
    """
    Convert typed golden fixture compact trades into the runtime Stage A trade contract.

    Args:
        case: Typed best-cell replay fixture.
    Returns:
        tuple[StageACompactTradeV2, ...]: Runtime-ready compact trades.
    Assumptions:
        Fixture ordering is already deterministic and matches Stage A output ordering.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
      - tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py
    """
    return tuple(
        StageACompactTradeV2(
            entry_signal_idx=index,
            entry_exec_idx=trade.entry_exec,
            direction=trade.direction,
            sig_exit_signal_idx=None,
            sig_exit_exec_idx=trade.sig_exit_exec,
        )
        for index, trade in enumerate(case.compact_trades)
    )
