from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from trading.contexts.backtest_artifacts.application.services.v2.artifact_backed_stage_b_scorer_v2 import (  # noqa: E501
    BacktestArtifactBackedStageBScorerV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactHitTimesArraysV2,
    ArtifactHitTimesManifestDocumentV2,
    StageACompactTradeV2,
    StageBHitTimesSliceV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.metrics_kernel import (
    compute_stage_b_metrics_v2,
)
from trading.contexts.backtest_artifacts.application.services.v2.risk_exit_kernel_1m import (
    replay_best_risk_cell_exact_v2,
    replay_risk_cell_exact_v2,
    resolve_risk_trade_exit_1m_v2,
    run_reference_vs_fast_self_check_v2,
    search_risk_cells_total_return_fast_v2,
    slice_hit_times_to_execution_window_v2,
)
from trading.contexts.backtest_artifacts.application.services.v2.stage_b_golden_fixtures_v2 import (
    StageBBestCellReplayCaseV2,
    StageBHitTimesFixtureV2,
    StageBLevelFactorsV2,
    StageBTradeExitCaseV2,
    load_stage_b_best_cell_replay_reference_case_v2,
    load_stage_b_golden_fixture_catalog_v2,
)

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "stage_b_golden_fixtures_v2.json"


def test_resolve_risk_trade_exit_1m_v2_matches_golden_trade_exit_cases() -> None:
    """
    Verify exact one-trade exit resolution matches the committed R5-03 oracle cases.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Golden fixture trade-exit cases remain the semantic oracle for `signal exit wins on equal
        bar`, `SL wins TP tie`, `entry_exec + 1`, and `close_on_end = 1`.
    Raises:
        AssertionError: If runtime exit facts drift from the executable oracle catalog.
    Side Effects:
        Reads the committed Stage B golden fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    for case in _trade_exit_cases_v2():
        resolved = resolve_risk_trade_exit_1m_v2(
            trade_index=0,
            trade=StageACompactTradeV2(
                entry_signal_idx=0,
                entry_exec_idx=case.entry_exec,
                direction=case.direction,
                sig_exit_signal_idx=None,
                sig_exit_exec_idx=case.sig_exit_exec,
            ),
            hit_times=_hit_times_slice_from_fixture_v2(
                hit_times=case.hit_times,
                level_factors=case.level_factors,
            ),
            exec_open=np.asarray(case.prices.exec_open, dtype=np.float64),
            exec_close=np.asarray(case.prices.exec_close, dtype=np.float64),
            tp_index=case.tp_index,
            sl_index=case.sl_index,
            close_on_end=case.close_on_end,
        )

        assert resolved.exit_exec_idx == case.expected_exit.exit_exec
        assert resolved.exit_reason == case.expected_exit.exit_reason
        assert resolved.closed is case.expected_exit.closed
        assert resolved.gross_factor == pytest.approx(
            float(case.expected_exit.gross_factor),
            rel=1e-9,
            abs=1e-9,
        )


def test_resolve_risk_trade_exit_1m_v2_ignores_hits_on_entry_bar() -> None:
    """
    Verify TP/SL lookup starts strictly at `entry_exec + 1` instead of the entry bar itself.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `1m hit-times` tables remain same-bar-inclusive artifacts, so runtime must choose the
        lookup start explicitly.
    Raises:
        AssertionError: If a TP/SL hit authored on the entry bar is incorrectly consumed.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
    """
    hit_times = StageBHitTimesSliceV2(
        tp_values=np.asarray((0.25,), dtype=np.float32),
        sl_values=np.asarray((0.25,), dtype=np.float32),
        long_tp=np.asarray(((1, 4, 4, 4),), dtype=np.int64),
        long_sl=np.asarray(((1, 4, 4, 4),), dtype=np.int64),
        short_tp=np.asarray(((1, 4, 4, 4),), dtype=np.int64),
        short_sl=np.asarray(((1, 4, 4, 4),), dtype=np.int64),
        sentinel_index=4,
    )

    resolved = resolve_risk_trade_exit_1m_v2(
        trade_index=0,
        trade=StageACompactTradeV2(
            entry_signal_idx=0,
            entry_exec_idx=1,
            direction=1,
            sig_exit_signal_idx=None,
            sig_exit_exec_idx=4,
        ),
        hit_times=hit_times,
        exec_open=np.asarray((100.0, 100.0, 100.0, 100.0), dtype=np.float64),
        exec_close=np.asarray((100.0, 100.0, 100.0, 100.0), dtype=np.float64),
        tp_index=0,
        sl_index=0,
        close_on_end=False,
    )

    assert resolved.exit_reason == "unclosed"
    assert resolved.closed is False
    assert resolved.exit_exec_idx == 1


def test_search_risk_cells_total_return_fast_v2_matches_bruteforce_exact_replay() -> None:
    """
    Verify fast TP/SL search matches brute-force exact replay on the golden synthetic grid.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Fast search is allowed to skip full replay per cell only if the selected winner and the
        entire `total_return_pct` matrix remain identical to exact replay.
    Raises:
        AssertionError: If fast search drifts from brute-force exact replay on the fixture grid.
    Side Effects:
        Reads the committed Stage B golden fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
    """
    case = _best_cell_case_v2()
    compact_trades = _compact_trades_from_best_cell_case_v2(case=case)
    hit_times = _hit_times_slice_from_fixture_v2(
        hit_times=case.hit_times,
        level_factors=case.level_factors,
    )
    exec_open = np.asarray(case.prices.exec_open, dtype=np.float64)
    exec_close = np.asarray(case.prices.exec_close, dtype=np.float64)
    fast_result = search_risk_cells_total_return_fast_v2(
        compact_trades=compact_trades,
        hit_times=hit_times,
        exec_open=exec_open,
        exec_close=exec_close,
        fee_rate=float(case.fee_rate),
        close_on_end=case.close_on_end,
    )

    brute_force = np.empty_like(fast_result.total_return_pct, dtype=np.float64)
    for tp_index in range(hit_times.tp_values.shape[0]):
        for sl_index in range(hit_times.sl_values.shape[0]):
            replay = replay_risk_cell_exact_v2(
                compact_trades=compact_trades,
                hit_times=hit_times,
                exec_open=exec_open,
                exec_close=exec_close,
                tp_index=tp_index,
                sl_index=sl_index,
                close_on_end=case.close_on_end,
            )
            brute_force[tp_index, sl_index] = compute_stage_b_metrics_v2(
                replay=replay,
                fee_rate=float(case.fee_rate),
                bars_per_year_exec=float(case.bars_per_year_exec),
            ).total_return_pct

    np.testing.assert_allclose(
        fast_result.total_return_pct,
        brute_force,
        rtol=1e-9,
        atol=1e-9,
    )
    expected_flat = int(np.argmax(brute_force))
    expected_tp_index, expected_sl_index = np.unravel_index(expected_flat, brute_force.shape)
    assert fast_result.best_tp_index == int(expected_tp_index)
    assert fast_result.best_sl_index == int(expected_sl_index)
    assert fast_result.best_total_return_pct == pytest.approx(
        float(brute_force[expected_tp_index, expected_sl_index]),
        rel=1e-9,
        abs=1e-9,
    )


def test_run_reference_vs_fast_self_check_v2_validates_bounded_subset() -> None:
    """
    Verify the explicit reference-vs-fast self-check compares a bounded subset deterministically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The new self-check surface stays debug/test-only, uses exact replay as the slow
        reference, and keeps the bounded subset smaller than the full fixture on at least one
        axis.
    Raises:
        AssertionError: If the bounded subset diagnostics or deterministic parity drift.
    Side Effects:
        Reads the committed Stage B golden fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb
    """
    case = _best_cell_case_v2()
    compact_trades = _compact_trades_from_best_cell_case_v2(case=case)
    hit_times = _hit_times_slice_from_fixture_v2(
        hit_times=case.hit_times,
        level_factors=case.level_factors,
    )
    bounded_trade_count = max(1, len(compact_trades) - 1)
    bounded_tp_level_count = max(1, int(hit_times.tp_values.shape[0]) - 1)
    bounded_sl_level_count = max(1, int(hit_times.sl_values.shape[0]) - 1)

    self_check = run_reference_vs_fast_self_check_v2(
        compact_trades=compact_trades,
        hit_times=hit_times,
        exec_open=np.asarray(case.prices.exec_open, dtype=np.float64),
        exec_close=np.asarray(case.prices.exec_close, dtype=np.float64),
        fee_rate=float(case.fee_rate),
        max_trade_count=bounded_trade_count,
        max_tp_level_count=bounded_tp_level_count,
        max_sl_level_count=bounded_sl_level_count,
        close_on_end=case.close_on_end,
    )

    assert self_check.total_trade_count == len(compact_trades)
    assert self_check.bounded_trade_count == bounded_trade_count
    assert self_check.total_tp_level_count == int(hit_times.tp_values.shape[0])
    assert self_check.bounded_tp_level_count == bounded_tp_level_count
    assert self_check.total_sl_level_count == int(hit_times.sl_values.shape[0])
    assert self_check.bounded_sl_level_count == bounded_sl_level_count
    assert self_check.fast_result.best_tp_index == self_check.reference_best_tp_index
    assert self_check.fast_result.best_sl_index == self_check.reference_best_sl_index
    assert self_check.fast_result.best_total_return_pct == pytest.approx(
        self_check.reference_best_total_return_pct,
        rel=0.0,
        abs=1e-9,
    )
    assert self_check.max_abs_total_return_diff == pytest.approx(0.0, rel=0.0, abs=1e-9)
    np.testing.assert_allclose(
        self_check.fast_result.total_return_pct,
        self_check.reference_total_return_pct,
        rtol=0.0,
        atol=1e-9,
    )


def test_slice_hit_times_to_execution_window_v2_accepts_widened_artifact_grid() -> None:
    """
    Verify Stage B stays grid-agnostic when `hit_times/1m` artifacts publish wider TP/SL grids.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stage B must read `tp_values` and `sl_values` from artifact arrays/manifests and keep
        fast search plus exact replay deterministic after local window rebasing.
    Raises:
        AssertionError: If widened artifact grids drift during local slicing or fast search.
    Side Effects:
        Allocates widened synthetic `hit_times/1m` artifact arrays in memory only.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """
    tp_values = np.asarray((0.01, 0.02, 0.03, 0.04, 0.05), dtype=np.float32)
    sl_values = np.asarray((0.01, 0.02, 0.03, 0.04), dtype=np.float32)
    local_long_tp = np.asarray(
        (
            (0, 1, 2, 3),
            (1, 2, 3, 4),
            (1, 2, 4, 4),
            (2, 3, 4, 4),
            (4, 4, 4, 4),
        ),
        dtype=np.int64,
    )
    local_long_sl = np.asarray(
        (
            (0, 1, 2, 3),
            (1, 2, 3, 4),
            (2, 3, 4, 4),
            (4, 4, 4, 4),
        ),
        dtype=np.int64,
    )
    global_sentinel_index = 8
    exec_target_slice = slice(2, 6)
    local_sentinel_index = int(exec_target_slice.stop - exec_target_slice.start)
    global_long_tp = np.full(
        (tp_values.shape[0], global_sentinel_index),
        global_sentinel_index,
        dtype=np.uint32,
    )
    global_long_sl = np.full(
        (sl_values.shape[0], global_sentinel_index),
        global_sentinel_index,
        dtype=np.uint32,
    )
    for level_index in range(local_long_tp.shape[0]):
        for time_index in range(local_sentinel_index):
            local_hit = int(local_long_tp[level_index, time_index])
            global_long_tp[level_index, exec_target_slice.start + time_index] = (
                global_sentinel_index
                if local_hit == local_sentinel_index
                else exec_target_slice.start + local_hit
            )
    for level_index in range(local_long_sl.shape[0]):
        for time_index in range(local_sentinel_index):
            local_hit = int(local_long_sl[level_index, time_index])
            global_long_sl[level_index, exec_target_slice.start + time_index] = (
                global_sentinel_index
                if local_hit == local_sentinel_index
                else exec_target_slice.start + local_hit
            )
    artifact_hit_times = ArtifactHitTimesArraysV2(
        manifest=cast(
            ArtifactHitTimesManifestDocumentV2,
            SimpleNamespace(sentinel_index=global_sentinel_index),
        ),
        tp_values=tp_values,
        sl_values=sl_values,
        long_tp=global_long_tp,
        long_sl=global_long_sl,
        short_tp=global_long_tp,
        short_sl=global_long_sl,
    )

    hit_times = slice_hit_times_to_execution_window_v2(
        hit_times_arrays=artifact_hit_times,
        exec_target_slice=exec_target_slice,
    )

    np.testing.assert_array_equal(hit_times.tp_values, tp_values)
    np.testing.assert_array_equal(hit_times.sl_values, sl_values)
    np.testing.assert_array_equal(hit_times.long_tp, local_long_tp)
    np.testing.assert_array_equal(hit_times.long_sl, local_long_sl)
    np.testing.assert_array_equal(hit_times.short_tp, local_long_tp)
    np.testing.assert_array_equal(hit_times.short_sl, local_long_sl)
    assert hit_times.long_tp.shape == (5, 4)
    assert hit_times.long_sl.shape == (4, 4)
    assert hit_times.short_tp.shape == (5, 4)
    assert hit_times.short_sl.shape == (4, 4)

    compact_trades = (
        StageACompactTradeV2(
            entry_signal_idx=0,
            entry_exec_idx=0,
            direction=1,
            sig_exit_signal_idx=None,
            sig_exit_exec_idx=local_sentinel_index,
        ),
    )
    exec_open = np.asarray((100.0, 100.0, 100.0, 100.0), dtype=np.float64)
    exec_close = np.asarray((100.0, 100.0, 100.0, 100.0), dtype=np.float64)
    fast_result = search_risk_cells_total_return_fast_v2(
        compact_trades=compact_trades,
        hit_times=hit_times,
        exec_open=exec_open,
        exec_close=exec_close,
        fee_rate=0.0,
        close_on_end=False,
    )

    brute_force = np.empty_like(fast_result.total_return_pct, dtype=np.float64)
    for tp_index in range(hit_times.tp_values.shape[0]):
        for sl_index in range(hit_times.sl_values.shape[0]):
            replay = replay_risk_cell_exact_v2(
                compact_trades=compact_trades,
                hit_times=hit_times,
                exec_open=exec_open,
                exec_close=exec_close,
                tp_index=tp_index,
                sl_index=sl_index,
                close_on_end=False,
            )
            brute_force[tp_index, sl_index] = compute_stage_b_metrics_v2(
                replay=replay,
                fee_rate=0.0,
            ).total_return_pct

    assert fast_result.total_return_pct.shape == (5, 4)
    np.testing.assert_allclose(
        fast_result.total_return_pct,
        brute_force,
        rtol=1e-9,
        atol=1e-9,
    )
    expected_flat = int(np.argmax(brute_force))
    expected_tp_index, expected_sl_index = np.unravel_index(expected_flat, brute_force.shape)
    assert fast_result.best_tp_index == int(expected_tp_index)
    assert fast_result.best_sl_index == int(expected_sl_index)


def test_artifact_backed_stage_b_scorer_v2_resolves_widened_grid_risk_indexes() -> None:
    """
    Verify the artifact-backed Stage B scorer resolves TP/SL indexes from widened artifact grids.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Runtime risk payloads keep human-percent units while artifact `tp_values` and `sl_values`
        stay in decimal-rate form.
    Raises:
        AssertionError: If widened artifact-grid values cannot be matched by the scorer.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    scorer = object.__new__(BacktestArtifactBackedStageBScorerV2)
    scorer._local_hit_times = StageBHitTimesSliceV2(
        tp_values=np.asarray((0.01, 0.02, 0.03, 0.04, 0.05), dtype=np.float32),
        sl_values=np.asarray((0.01, 0.02, 0.03, 0.04), dtype=np.float32),
        long_tp=np.ones((5, 1), dtype=np.int64),
        long_sl=np.ones((4, 1), dtype=np.int64),
        short_tp=np.ones((5, 1), dtype=np.int64),
        short_sl=np.ones((4, 1), dtype=np.int64),
        sentinel_index=1,
    )

    tp_index, sl_index = scorer._resolve_risk_level_indexes_v2(
        risk_params={
            "tp_enabled": True,
            "tp_pct": 4.0,
            "sl_enabled": True,
            "sl_pct": 3.0,
        }
    )

    assert tp_index == 3
    assert sl_index == 2


def test_replay_best_risk_cell_exact_v2_matches_best_cell_golden_fixture() -> None:
    """
    Verify best-cell winner selection and exact replay metrics match the golden replay fixture.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The committed R5-03 best-cell case remains the locked semantic oracle for Stage B replay.
    Raises:
        AssertionError: If fast winner selection or exact replay metrics drift from the oracle.
    Side Effects:
        Reads the committed Stage B golden fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
    """
    case = _best_cell_case_v2()
    compact_trades = _compact_trades_from_best_cell_case_v2(case=case)
    hit_times = _hit_times_slice_from_fixture_v2(
        hit_times=case.hit_times,
        level_factors=case.level_factors,
    )
    exec_open = np.asarray(case.prices.exec_open, dtype=np.float64)
    exec_close = np.asarray(case.prices.exec_close, dtype=np.float64)

    fast_result, replay = replay_best_risk_cell_exact_v2(
        compact_trades=compact_trades,
        hit_times=hit_times,
        exec_open=exec_open,
        exec_close=exec_close,
        fee_rate=float(case.fee_rate),
        close_on_end=case.close_on_end,
    )
    metrics = compute_stage_b_metrics_v2(
        replay=replay,
        fee_rate=float(case.fee_rate),
        bars_per_year_exec=float(case.bars_per_year_exec),
    )

    assert fast_result.best_tp_index == case.expected_result.best_tp_index
    assert fast_result.best_sl_index == case.expected_result.best_sl_index
    assert replay.tp_index == case.expected_result.best_tp_index
    assert replay.sl_index == case.expected_result.best_sl_index
    assert metrics.trade_count == case.expected_result.trade_count
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
    assert metrics.sharpe_trades == pytest.approx(
        float(case.expected_result.metrics.sharpe),
        rel=1e-9,
        abs=1e-9,
    )


def test_resolve_risk_trade_exit_1m_v2_rejects_execution_length_drift() -> None:
    """
    Verify Stage B risk kernel fails fast when execution arrays drift from `sentinel_index`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Runtime must reject shape drift before hot-path ranking can consume invalid arrays.
    Raises:
        AssertionError: If mismatched execution arrays are accepted.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    hit_times = StageBHitTimesSliceV2(
        tp_values=np.asarray((0.25,), dtype=np.float32),
        sl_values=np.asarray((0.25,), dtype=np.float32),
        long_tp=np.asarray(((3, 3, 3),), dtype=np.int64),
        long_sl=np.asarray(((3, 3, 3),), dtype=np.int64),
        short_tp=np.asarray(((3, 3, 3),), dtype=np.int64),
        short_sl=np.asarray(((3, 3, 3),), dtype=np.int64),
        sentinel_index=3,
    )

    with pytest.raises(ValueError, match="exec_open length must match sentinel_index"):
        resolve_risk_trade_exit_1m_v2(
            trade_index=0,
            trade=StageACompactTradeV2(
                entry_signal_idx=0,
                entry_exec_idx=0,
                direction=1,
                sig_exit_signal_idx=None,
                sig_exit_exec_idx=3,
            ),
            hit_times=hit_times,
            exec_open=np.asarray((100.0, 100.0), dtype=np.float64),
            exec_close=np.asarray((100.0, 100.0, 100.0), dtype=np.float64),
            tp_index=0,
            sl_index=0,
        )


def _trade_exit_cases_v2() -> tuple[StageBTradeExitCaseV2, ...]:
    """
    Return typed trade-exit cases from the committed Stage B golden fixture catalog.

    Args:
        None.
    Returns:
        tuple[StageBTradeExitCaseV2, ...]: Ordered trade-exit cases from the golden catalog.
    Assumptions:
        Catalog validation already guarantees deterministic case ordering and typing.
    Raises:
        None.
    Side Effects:
        Reads the committed Stage B golden fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
    """
    catalog = load_stage_b_golden_fixture_catalog_v2(path=_FIXTURE_PATH)
    return tuple(case for case in catalog.cases if isinstance(case, StageBTradeExitCaseV2))


def _best_cell_case_v2() -> StageBBestCellReplayCaseV2:
    """
    Return the committed best-cell replay fixture used by Stage B kernel parity tests.

    Args:
        None.
    Returns:
        StageBBestCellReplayCaseV2: Typed golden replay case.
    Assumptions:
        The golden catalog contains exactly one best-cell replay case for R5-03 semantics.
    Raises:
        ValueError: If the expected best-cell replay case is missing or duplicated.
    Side Effects:
        Reads the committed Stage B golden fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
    """
    return load_stage_b_best_cell_replay_reference_case_v2(path=_FIXTURE_PATH)


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
        StageBHitTimesSliceV2: Runtime-ready hit-times slice with local `sentinel_index`.
    Assumptions:
        Golden fixtures use the same TP/SL level factors for long and short directions.
    Raises:
        None.
    Side Effects:
        Allocates deterministic NumPy arrays for runtime kernel inputs.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
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
        tuple[StageACompactTradeV2, ...]: Runtime-ready compact trades in fixture order.
    Assumptions:
        Golden fixtures model the Stage A output contract already, differing only in field names.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
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
