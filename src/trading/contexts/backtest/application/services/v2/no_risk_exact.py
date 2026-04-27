from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numba as nb
import numpy as np

from trading.contexts.backtest.application.dto import (
    BacktestComboChunk,
    BacktestComboPlanningResult,
    BacktestNoRiskExactConfig,
    BacktestNoRiskExactResult,
    BacktestNoRiskExactTelemetry,
    BacktestNoRiskPriceContext,
    BacktestPreparePoolsResult,
    BacktestProxyContext,
    BacktestProxyFilterResult,
    PreparedIndicatorPool,
)
from trading.contexts.backtest.application.services.v2.combo_planning import (
    EVENT_SEGMENTS_2_NO_RISK_BACKEND,
    EVENT_SEGMENTS_N_NO_RISK_BACKEND,
    STREAMING_2_NO_RISK_BACKEND,
    BacktestComboPlanningService,
    build_local_row_pools,
    iter_combo_chunks,
    make_combo_idx_matrix,
)

NO_RISK_IMPLEMENTATION_ID = "no_risk_exact_v1_iteration_4"
SERVICE_WARMUP_STAGE_NAME = "service_warmup"
SELF_CHECK_STAGE_NAME = "self_check"
EXACT_SCORING_STAGE_NAME = "exact_scoring"
HEAP_UPDATE_STAGE_NAME = "heap_update"
TOP_RESULT_PROXY_FILL_STAGE_NAME = "top_result_proxy_fill"
BENCHMARK_TOP_K_DEFAULT = 5
SAMPLE_WARMUP_TOP_K = 1
SELF_CHECK_RETURN_TOLERANCE = 1e-4
NEG_INF = np.float32(-1e30)

_DIRECTION_LONG_ONLY = np.int8(1)
_DIRECTION_LONG_SHORT_REVERSAL = np.int8(2)
_SIZING_ALL_IN = np.int8(0)
_SIZING_FIXED_QUOTE = np.int8(1)
_SIZING_FIXED_EQUITY_PCT = np.int8(2)
_SIZING_FIXED_EQUITY_PCT_MIN_QUOTE = np.int8(3)
_SIZING_FIXED_EQUITY_PCT_MAX_QUOTE = np.int8(4)


class BacktestNoRiskExactRejected(ValueError):
    """
    Deterministic internal rejection for unsupported no-risk exact scoring inputs.
    """


@dataclass(frozen=True, slots=True)
class _ExecutionSettings:
    direction_mode: str
    direction_mode_code: np.int8
    fee_rate: float
    slippage_rate: float
    initial_cash_quote: float
    bars_per_year_exec: float
    sizing_mode_code: np.int8
    sizing_value: float
    sizing_bound: float
    profit_lock_enabled: np.int8
    safe_profit_percent: float
    close_on_end: np.int8


@dataclass(slots=True)
class _NoRiskScoreArrays:
    total_return_pct: np.ndarray
    max_drawdown_pct: np.ndarray
    return_over_max_drawdown: np.ndarray
    profit_factor: np.ndarray
    trade_count: np.ndarray
    sharpe_trades: np.ndarray
    win_rate_pct: np.ndarray
    avg_trade_ret_pct: np.ndarray
    avg_trade_exec_bars: np.ndarray
    exposure_pct: np.ndarray

    @classmethod
    def empty(cls, size: int) -> _NoRiskScoreArrays:
        return cls(
            total_return_pct=np.empty(size, dtype=np.float64),
            max_drawdown_pct=np.empty(size, dtype=np.float64),
            return_over_max_drawdown=np.empty(size, dtype=np.float64),
            profit_factor=np.empty(size, dtype=np.float64),
            trade_count=np.empty(size, dtype=np.int32),
            sharpe_trades=np.empty(size, dtype=np.float64),
            win_rate_pct=np.empty(size, dtype=np.float64),
            avg_trade_ret_pct=np.empty(size, dtype=np.float64),
            avg_trade_exec_bars=np.empty(size, dtype=np.float64),
            exposure_pct=np.empty(size, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactScoringService:
    """
    Internal Iteration 4 service for no-risk exact scoring and notebook top-K.
    """

    config: BacktestNoRiskExactConfig = BacktestNoRiskExactConfig()
    combo_planning_service: BacktestComboPlanningService = field(
        default_factory=BacktestComboPlanningService
    )

    def execute(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        combo_planning_result: BacktestComboPlanningResult,
        normalized_request: Mapping[str, Any],
        price_context: BacktestNoRiskPriceContext,
        benchmark_top_k: int | None = None,
    ) -> BacktestNoRiskExactResult:
        _reject_unless_no_risk(
            normalized_request=normalized_request,
            combo_planning_result=combo_planning_result,
        )
        execution_settings = _execution_settings_from_normalized(
            normalized_request,
            bars_per_year_exec=self.config.bars_per_year_exec,
        )
        top_k = self.config.benchmark_top_k if benchmark_top_k is None else int(benchmark_top_k)
        if top_k <= 0:
            raise BacktestNoRiskExactRejected("benchmark_top_k must be > 0")

        stage_timings = _zero_stage_timings()
        heap: list[tuple[tuple[float, tuple[int, ...]], dict[str, Any]]] = []
        local_row_pools = build_local_row_pools(prepared_result=prepared_result)
        combo_iter = iter_combo_chunks(
            indicator_ids=prepared_result.indicator_ids,
            local_row_pools=local_row_pools,
            chunk_size=self.config.combo_chunk_size,
        )
        ranking_metric, ranking_direction = _ranking_from_normalized(normalized_request)

        self_check: Mapping[str, Any] = {
            "checked": 0,
            "passed": True,
            "exact_backend": combo_planning_result.backend.backend_id,
        }
        self_check_done = self.config.self_check_n == 0
        combo_chunks_processed = 0
        exact_candidates_evaluated = 0

        while True:
            try:
                combo_chunk = next(combo_iter)
            except StopIteration:
                break
            combo_chunks_processed += 1

            filter_result = self.combo_planning_service.proxy_filter(
                combo_chunk=combo_chunk,
                proxy_context=combo_planning_result.proxy_context,
            )
            if filter_result.selected_candidate_count == 0:
                continue
            exact_candidates_evaluated += filter_result.selected_candidate_count

            if not self_check_done:
                stage_start = time.perf_counter()
                self_check = run_fast_vs_reference_self_check_two(
                    filter_result=filter_result,
                    prepared_result=prepared_result,
                    combo_planning_result=combo_planning_result,
                    price_context=price_context,
                    execution_settings=execution_settings,
                    check_n=self.config.self_check_n,
                )
                stage_timings[SELF_CHECK_STAGE_NAME] += time.perf_counter() - stage_start
                self_check_done = True

            scores = _NoRiskScoreArrays.empty(filter_result.selected_candidate_count)
            stage_start = time.perf_counter()
            evaluate_no_risk_exact_chunk(
                filter_result=filter_result,
                prepared_result=prepared_result,
                combo_planning_result=combo_planning_result,
                price_context=price_context,
                execution_settings=execution_settings,
                scores=scores,
            )
            stage_timings[EXACT_SCORING_STAGE_NAME] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            heap_update(
                heap=heap,
                filter_result=filter_result,
                prepared_result=prepared_result,
                scores=scores,
                top_k=top_k,
                ranking_metric=ranking_metric,
                ranking_direction=ranking_direction,
            )
            stage_timings[HEAP_UPDATE_STAGE_NAME] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        top_results = top_result_proxy_fill(
            heap=heap,
            prepared_result=prepared_result,
            proxy_context=combo_planning_result.proxy_context,
        )
        stage_timings[TOP_RESULT_PROXY_FILL_STAGE_NAME] += time.perf_counter() - stage_start

        request_top_n = int(normalized_request.get("top_n", 100))
        return BacktestNoRiskExactResult(
            top_results=tuple(top_results),
            self_check=self_check,
            telemetry=BacktestNoRiskExactTelemetry(
                stage_timings=stage_timings,
                request_top_n=request_top_n,
                benchmark_top_k=top_k,
                top_results_count=len(top_results),
                heap_capacity=top_k,
                exact_backend_display_name=_backend_display_name(combo_planning_result),
                implementation_id=NO_RISK_IMPLEMENTATION_ID,
                exact_candidates_evaluated=exact_candidates_evaluated,
                combo_chunks_processed=combo_chunks_processed,
            ),
        )


def evaluate_no_risk_exact_chunk(
    *,
    filter_result: BacktestProxyFilterResult,
    prepared_result: BacktestPreparePoolsResult,
    combo_planning_result: BacktestComboPlanningResult,
    price_context: BacktestNoRiskPriceContext,
    execution_settings: _ExecutionSettings,
    scores: _NoRiskScoreArrays,
) -> None:
    """
    Dispatch one no-risk exact chunk through the selected backend strategy.
    """

    backend_id = combo_planning_result.backend.backend_id
    indicator_ids = tuple(prepared_result.indicator_ids)
    if backend_id == EVENT_SEGMENTS_2_NO_RISK_BACKEND:
        left_id, right_id = indicator_ids
        left_pool, right_pool = _ordered_pools(prepared_result)
        evaluate_no_risk_event_segments_two(
            filter_result.selected_rows_by_indicator[left_id],
            filter_result.selected_rows_by_indicator[right_id],
            left_pool.segments.starts,
            left_pool.segments.ends,
            left_pool.segments.values,
            left_pool.segments.counts,
            right_pool.segments.starts,
            right_pool.segments.ends,
            right_pool.segments.values,
            right_pool.segments.counts,
            prepared_result.execution_mapping.signal_entry_exec_idx_15m,
            price_context.execution_open_1m,
            price_context.execution_close_1m,
            np.int32(prepared_result.execution_mapping.t_exec_limit_1m),
            execution_settings.initial_cash_quote,
            execution_settings.sizing_mode_code,
            execution_settings.sizing_value,
            execution_settings.sizing_bound,
            execution_settings.fee_rate,
            execution_settings.slippage_rate,
            execution_settings.safe_profit_percent,
            execution_settings.profit_lock_enabled,
            execution_settings.bars_per_year_exec,
            execution_settings.close_on_end,
            execution_settings.direction_mode_code,
            scores.total_return_pct,
            scores.max_drawdown_pct,
            scores.return_over_max_drawdown,
            scores.profit_factor,
            scores.trade_count,
            scores.sharpe_trades,
            scores.win_rate_pct,
            scores.avg_trade_ret_pct,
            scores.avg_trade_exec_bars,
            scores.exposure_pct,
        )
        return

    if backend_id == STREAMING_2_NO_RISK_BACKEND:
        left_id, right_id = indicator_ids
        left_pool, right_pool = _ordered_pools(prepared_result)
        evaluate_no_risk_streaming_two(
            filter_result.selected_rows_by_indicator[left_id],
            filter_result.selected_rows_by_indicator[right_id],
            left_pool.trade_T,
            right_pool.trade_T,
            prepared_result.execution_mapping.signal_entry_exec_idx_15m,
            price_context.execution_open_1m,
            price_context.execution_close_1m,
            np.int32(prepared_result.execution_mapping.t_exec_limit_1m),
            execution_settings.initial_cash_quote,
            execution_settings.sizing_mode_code,
            execution_settings.sizing_value,
            execution_settings.sizing_bound,
            execution_settings.fee_rate,
            execution_settings.slippage_rate,
            execution_settings.safe_profit_percent,
            execution_settings.profit_lock_enabled,
            execution_settings.bars_per_year_exec,
            execution_settings.close_on_end,
            execution_settings.direction_mode_code,
            scores.total_return_pct,
            scores.max_drawdown_pct,
            scores.return_over_max_drawdown,
            scores.profit_factor,
            scores.trade_count,
            scores.sharpe_trades,
            scores.win_rate_pct,
            scores.avg_trade_ret_pct,
            scores.avg_trade_exec_bars,
            scores.exposure_pct,
        )
        return

    if backend_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND:
        exact_context = combo_planning_result.exact_context
        if (
            exact_context.starts is None
            or exact_context.ends is None
            or exact_context.values is None
            or exact_context.counts is None
        ):
            raise BacktestNoRiskExactRejected("event_segments_n_no_risk requires exact context")
        combo_idx_by_indicator = make_combo_idx_matrix(
            combo_chunk=_filter_result_as_combo_chunk(filter_result),
            indicator_ids=indicator_ids,
        )
        segment_pos_workspace = np.empty(
            (filter_result.selected_candidate_count, len(indicator_ids)),
            dtype=np.int32,
        )
        evaluate_no_risk_event_segments_n(
            combo_idx_by_indicator,
            exact_context.starts,
            exact_context.ends,
            exact_context.values,
            exact_context.counts,
            segment_pos_workspace,
            prepared_result.execution_mapping.signal_entry_exec_idx_15m,
            price_context.execution_open_1m,
            price_context.execution_close_1m,
            np.int32(prepared_result.execution_mapping.t_exec_limit_1m),
            execution_settings.initial_cash_quote,
            execution_settings.sizing_mode_code,
            execution_settings.sizing_value,
            execution_settings.sizing_bound,
            execution_settings.fee_rate,
            execution_settings.slippage_rate,
            execution_settings.safe_profit_percent,
            execution_settings.profit_lock_enabled,
            execution_settings.bars_per_year_exec,
            execution_settings.close_on_end,
            execution_settings.direction_mode_code,
            scores.total_return_pct,
            scores.max_drawdown_pct,
            scores.return_over_max_drawdown,
            scores.profit_factor,
            scores.trade_count,
            scores.sharpe_trades,
            scores.win_rate_pct,
            scores.avg_trade_ret_pct,
            scores.avg_trade_exec_bars,
            scores.exposure_pct,
        )
        return

    raise BacktestNoRiskExactRejected(f"Unsupported no-risk backend {backend_id!r}")


def heap_update(
    *,
    heap: list[tuple[tuple[float, tuple[int, ...]], dict[str, Any]]],
    filter_result: BacktestProxyFilterResult,
    prepared_result: BacktestPreparePoolsResult,
    scores: _NoRiskScoreArrays,
    top_k: int,
    ranking_metric: str,
    ranking_direction: str,
) -> None:
    """
    Notebook-compatible heap_update for compact in-memory top rows.
    """

    indicator_ids = tuple(prepared_result.indicator_ids)
    pools = _ordered_pools(prepared_result)
    confirm = filter_result.confirm
    proxy = filter_result.proxy
    proxy_pending = confirm is None or proxy is None
    for local_idx in range(filter_result.selected_candidate_count):
        selected_local_indices = tuple(
            int(filter_result.selected_rows_by_indicator[indicator_id][local_idx])
            for indicator_id in indicator_ids
        )
        orig_rows = tuple(
            int(pools[pos].row_ids[selected_local_indices[pos]])
            for pos in range(len(indicator_ids))
        )
        metric_value = _score_value(scores=scores, metric=ranking_metric, index=local_idx)
        rank_score = metric_value if ranking_direction == "desc" else -metric_value
        if proxy_pending:
            confirm_count = 0
            proxy_score = 0.0
        else:
            if confirm is None or proxy is None:
                raise AssertionError("active proxy metadata is missing")
            confirm_count = int(confirm[local_idx])
            proxy_score = float(proxy[local_idx])
        item: dict[str, Any] = {
            "total_return_pct": float(scores.total_return_pct[local_idx]),
            "confirm_count": confirm_count,
            "proxy_score": proxy_score,
            "trade_count": int(scores.trade_count[local_idx]),
            "max_drawdown_pct": float(scores.max_drawdown_pct[local_idx]),
            "return_over_max_drawdown": float(scores.return_over_max_drawdown[local_idx]),
            "profit_factor": float(scores.profit_factor[local_idx]),
            "sharpe_trades": float(scores.sharpe_trades[local_idx]),
            "win_rate_pct": float(scores.win_rate_pct[local_idx]),
            "avg_trade_ret_pct": float(scores.avg_trade_ret_pct[local_idx]),
            "avg_trade_exec_bars": float(scores.avg_trade_exec_bars[local_idx]),
            "exposure_pct": float(scores.exposure_pct[local_idx]),
            "_local_indices": selected_local_indices,
            "_proxy_pending": proxy_pending,
        }
        for pos, indicator_id in enumerate(indicator_ids):
            item[indicator_id] = pools[pos].metadata[selected_local_indices[pos]].as_mapping()
        heap_key = (rank_score, orig_rows)
        heap_item = (heap_key, item)
        if len(heap) < top_k:
            heapq.heappush(heap, heap_item)
        elif heap_key > heap[0][0]:
            heapq.heapreplace(heap, heap_item)


def top_result_proxy_fill(
    *,
    heap: list[tuple[tuple[float, tuple[int, ...]], dict[str, Any]]],
    prepared_result: BacktestPreparePoolsResult,
    proxy_context: BacktestProxyContext,
) -> list[dict[str, Any]]:
    """
    Notebook-compatible top_result_proxy_fill over final heap rows only.
    """

    indicator_ids = tuple(prepared_result.indicator_ids)
    pools = _ordered_pools(prepared_result)
    top_results: list[dict[str, Any]] = []
    for _, item in sorted(heap, key=lambda pair: pair[0], reverse=True):
        local_indices = item.pop("_local_indices")
        proxy_pending = bool(item.pop("_proxy_pending"))
        if proxy_pending:
            eval_rows = tuple(
                pools[pos].eval_T[int(local_indices[pos])]
                for pos in range(len(indicator_ids))
            )
            confirm_count, proxy_score = proxy_for_indicator_rows(
                eval_rows=eval_rows,
                ret_15m=prepared_result.signal_returns_15m,
                min_confirm=proxy_context.combo_min_confirm,
                fee_penalty_per_confirm=proxy_context.fee_penalty_per_confirm,
            )
            item["confirm_count"] = int(confirm_count)
            item["proxy_score"] = float(proxy_score)
        top_results.append(item)
    return top_results


def proxy_for_indicator_rows(
    *,
    eval_rows: tuple[np.ndarray, ...],
    ret_15m: np.ndarray,
    min_confirm: int,
    fee_penalty_per_confirm: np.float32,
) -> tuple[int, float]:
    """
    Notebook-equivalent confirm/proxy score for one final top row.
    """

    if not eval_rows:
        return 0, float(NEG_INF)
    consensus = np.asarray(eval_rows[0], dtype=np.int8).copy()
    for eval_row in eval_rows[1:]:
        consensus[consensus != np.asarray(eval_row, dtype=np.int8)] = np.int8(0)
    confirms = int(np.count_nonzero(consensus))
    if confirms < int(min_confirm):
        return confirms, float(NEG_INF)
    returns = np.asarray(ret_15m, dtype=np.float32)
    proxy = np.float32(0.0)
    if confirms:
        proxy += np.sum(returns[consensus == 1], dtype=np.float32)
        proxy -= np.sum(returns[consensus == -1], dtype=np.float32)
    proxy -= fee_penalty_per_confirm * np.float32(confirms)
    return confirms, float(proxy)


def run_fast_vs_reference_self_check_two(
    *,
    filter_result: BacktestProxyFilterResult,
    prepared_result: BacktestPreparePoolsResult,
    combo_planning_result: BacktestComboPlanningResult,
    price_context: BacktestNoRiskPriceContext,
    execution_settings: _ExecutionSettings,
    check_n: int,
    ret_tol: float = SELF_CHECK_RETURN_TOLERANCE,
) -> dict[str, Any]:
    """
    Compare the selected exact backend against a generic slow no-risk reference.
    """

    indicator_ids = tuple(prepared_result.indicator_ids)
    n_check = min(int(check_n), filter_result.selected_candidate_count)
    if n_check <= 0:
        return {
            "checked": 0,
            "passed": True,
            "exact_backend": combo_planning_result.backend.backend_id,
            "max_abs_exact_backend_ret_diff": 0.0,
        }

    subset = BacktestProxyFilterResult(
        indicator_ids=indicator_ids,
        selected_indexes=filter_result.selected_indexes[:n_check],
        selected_rows_by_indicator={
            indicator_id: filter_result.selected_rows_by_indicator[indicator_id][:n_check]
            for indicator_id in indicator_ids
        },
        input_candidate_count=n_check,
        valid_candidate_count=n_check,
        selected_candidate_count=n_check,
        confirm=None if filter_result.confirm is None else filter_result.confirm[:n_check],
        proxy=None if filter_result.proxy is None else filter_result.proxy[:n_check],
    )
    backend_scores = _NoRiskScoreArrays.empty(n_check)
    evaluate_no_risk_exact_chunk(
        filter_result=subset,
        prepared_result=prepared_result,
        combo_planning_result=combo_planning_result,
        price_context=price_context,
        execution_settings=execution_settings,
        scores=backend_scores,
    )

    reference_total_return_pct = np.empty(n_check, dtype=np.float64)
    reference_trade_count = np.empty(n_check, dtype=np.int32)
    for row_idx in range(n_check):
        local_indices = tuple(
            int(subset.selected_rows_by_indicator[indicator_id][row_idx])
            for indicator_id in indicator_ids
        )
        metrics = evaluate_no_risk_reference_rows_slow(
            indicator_ids=indicator_ids,
            indicator_pools=prepared_result.indicator_pools,
            local_indices=local_indices,
            execution_mapping=prepared_result.execution_mapping.signal_entry_exec_idx_15m,
            t_exec_limit_1m=prepared_result.execution_mapping.t_exec_limit_1m,
            exec_open_1m=price_context.execution_open_1m,
            exec_close_1m=price_context.execution_close_1m,
            execution_settings=execution_settings,
        )
        reference_total_return_pct[row_idx] = float(metrics[0])
        reference_trade_count[row_idx] = int(metrics[4])

    if not np.array_equal(reference_trade_count, backend_scores.trade_count):
        raise AssertionError(
            f"Exact backend {combo_planning_result.backend.backend_id!r} trade counts "
            "differ from generic slow reference."
        )
    max_abs_diff = float(
        np.max(np.abs(reference_total_return_pct - backend_scores.total_return_pct))
    )
    if max_abs_diff > ret_tol:
        raise AssertionError(
            f"Exact backend {combo_planning_result.backend.backend_id!r} total return "
            f"differs from generic slow reference by {max_abs_diff}, tolerance {ret_tol}."
        )
    return {
        "checked": n_check,
        "passed": True,
        "exact_backend": combo_planning_result.backend.backend_id,
        "direction_mode": execution_settings.direction_mode,
        "return_tolerance": ret_tol,
        "max_abs_exact_backend_ret_diff": max_abs_diff,
        "trade_count_equal": True,
    }


def build_trade_list_for_indicator_rows_slow(
    *,
    indicator_ids: tuple[str, ...],
    indicator_pools: Sequence[PreparedIndicatorPool],
    local_indices: tuple[int, ...],
    execution_mapping: np.ndarray,
    t_exec_limit_1m: int,
    direction_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slow generic no-risk trade-list reference for any supported arity.
    """

    pool_by_id = {pool.indicator_id: pool for pool in indicator_pools}
    rows = [
        pool_by_id[indicator_id].trade_T[local_indices[pos]]
        for pos, indicator_id in enumerate(indicator_ids)
    ]
    n_sig = int(rows[0].shape[0])
    entry_exec: list[int] = []
    directions: list[int] = []
    sig_exit_exec: list[int] = []
    current_dir = 0
    current_entry = 0
    for signal_idx in range(n_sig):
        raw_dir = int(rows[0][signal_idx])
        if raw_dir != 0:
            for row in rows[1:]:
                if int(row[signal_idx]) != raw_dir:
                    raw_dir = 0
                    break
        dirn = apply_direction_mode_py(raw_dir, direction_mode)
        if dirn == 0 and not (direction_mode == "long_only" and current_dir != 0):
            continue
        entry_idx = int(execution_mapping[signal_idx])
        if entry_idx >= int(t_exec_limit_1m):
            break
        if dirn == 0:
            entry_exec.append(current_entry)
            directions.append(current_dir)
            sig_exit_exec.append(entry_idx)
            current_dir = 0
            current_entry = 0
            continue
        if current_dir == 0:
            current_dir = dirn
            current_entry = entry_idx
            continue
        if dirn == current_dir:
            continue
        entry_exec.append(current_entry)
        directions.append(current_dir)
        sig_exit_exec.append(entry_idx)
        current_dir = dirn
        current_entry = entry_idx
    if current_dir != 0:
        entry_exec.append(current_entry)
        directions.append(current_dir)
        sig_exit_exec.append(int(t_exec_limit_1m))
    return (
        np.asarray(entry_exec, dtype=np.int32),
        np.asarray(directions, dtype=np.int8),
        np.asarray(sig_exit_exec, dtype=np.int32),
    )


def evaluate_no_risk_reference_rows_slow(
    *,
    indicator_ids: tuple[str, ...],
    indicator_pools: Sequence[PreparedIndicatorPool],
    local_indices: tuple[int, ...],
    execution_mapping: np.ndarray,
    t_exec_limit_1m: int,
    exec_open_1m: np.ndarray,
    exec_close_1m: np.ndarray,
    execution_settings: _ExecutionSettings,
) -> tuple[float, float, float, float, int, float, float, float, float, float]:
    entry_arr, dir_arr, exit_arr = build_trade_list_for_indicator_rows_slow(
        indicator_ids=indicator_ids,
        indicator_pools=indicator_pools,
        local_indices=local_indices,
        execution_mapping=execution_mapping,
        t_exec_limit_1m=t_exec_limit_1m,
        direction_mode=execution_settings.direction_mode,
    )
    if int(entry_arr.size) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    metrics = score_trade_list_no_risk(
        entry_arr,
        dir_arr,
        exit_arr,
        np.int32(entry_arr.size),
        exec_open_1m,
        exec_close_1m,
        np.int32(t_exec_limit_1m),
        execution_settings.initial_cash_quote,
        execution_settings.sizing_mode_code,
        execution_settings.sizing_value,
        execution_settings.sizing_bound,
        execution_settings.fee_rate,
        execution_settings.slippage_rate,
        execution_settings.safe_profit_percent,
        execution_settings.profit_lock_enabled,
        execution_settings.bars_per_year_exec,
        execution_settings.close_on_end,
    )
    return (
        float(metrics[0]),
        float(metrics[1]),
        float(metrics[2]),
        float(metrics[3]),
        int(metrics[4]),
        float(metrics[5]),
        float(metrics[6]),
        float(metrics[7]),
        float(metrics[8]),
        float(metrics[9]),
    )


def apply_direction_mode_py(raw_dir: int, direction_mode: str) -> int:
    if direction_mode == "long_only":
        return 1 if raw_dir == 1 else 0
    if direction_mode == "long_short_reversal":
        return int(raw_dir)
    raise BacktestNoRiskExactRejected(f"Unsupported direction_mode={direction_mode!r}")


@nb.njit(cache=True, inline="always")
def _consensus_dir2(left_value: np.int8, right_value: np.int8) -> np.int8:
    if left_value == 1 and right_value == 1:
        return np.int8(1)
    if left_value == -1 and right_value == -1:
        return np.int8(-1)
    return np.int8(0)


@nb.njit(cache=True, inline="always")
def _apply_direction_mode(raw_dir: np.int8, direction_mode: np.int8) -> np.int8:
    if direction_mode == _DIRECTION_LONG_ONLY:
        if raw_dir == 1:
            return np.int8(1)
        return np.int8(0)
    return raw_dir


@nb.njit(cache=True, inline="always")
def _resolve_quote_amount(
    available_quote: float,
    safe_quote: float,
    sizing_mode_code: np.int8,
    sizing_value: float,
    sizing_bound: float,
) -> float:
    equity = available_quote + safe_quote
    if sizing_mode_code == _SIZING_ALL_IN:
        quote_amount = available_quote
    elif sizing_mode_code == _SIZING_FIXED_QUOTE:
        quote_amount = sizing_value
    else:
        quote_amount = equity * (sizing_value / 100.0)
        if sizing_mode_code == _SIZING_FIXED_EQUITY_PCT_MIN_QUOTE:
            if quote_amount < sizing_bound:
                quote_amount = sizing_bound
        elif sizing_mode_code == _SIZING_FIXED_EQUITY_PCT_MAX_QUOTE:
            if quote_amount > sizing_bound:
                quote_amount = sizing_bound
    if quote_amount > available_quote:
        quote_amount = available_quote
    if quote_amount < 0.0:
        return 0.0
    return quote_amount


@nb.njit(cache=True, inline="always")
def _trade_sharpe_kernel(
    trade_count: np.int32,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    bars_per_year_exec: float,
    sentinel_index: np.int32,
) -> float:
    if trade_count <= 1:
        return 0.0
    mean_trade_return = sum_trade_return / float(trade_count)
    variance = (sum_trade_return_squared / float(trade_count)) - (
        mean_trade_return * mean_trade_return
    )
    if variance <= 0.0:
        return 0.0
    years = float(sentinel_index) / float(bars_per_year_exec)
    if years <= 0.0:
        years = 1.0
    trades_per_year = float(trade_count) / years
    return (mean_trade_return / math.sqrt(variance)) * math.sqrt(trades_per_year)


@nb.njit(cache=True, inline="always")
def _apply_no_risk_trade_to_state(
    entry_idx: np.int32,
    trade_direction: np.int8,
    exit_exec_idx: np.int32,
    exit_price_raw: float,
    exec_open_1m: np.ndarray,
    available_quote: float,
    safe_quote: float,
    equity: float,
    peak_equity: float,
    max_drawdown_pct: float,
    gross_profit_quote: float,
    gross_loss_quote: float,
    closed_trade_count: np.int32,
    win_count: np.int32,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    total_trade_return_pct: float,
    total_trade_exec_bars: float,
    exposure_bars: float,
    sizing_mode_code: np.int8,
    sizing_value: float,
    sizing_bound: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    profit_lock_enabled: np.int8,
):
    if available_quote <= 0.0:
        return (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
        )

    quote_amount = _resolve_quote_amount(
        available_quote,
        safe_quote,
        sizing_mode_code,
        sizing_value,
        sizing_bound,
    )
    if quote_amount <= 0.0:
        return (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
        )

    entry_price_raw = float(exec_open_1m[entry_idx])
    if trade_direction == 1:
        entry_fill_price = entry_price_raw * (1.0 + slippage_rate)
        exit_fill_price = exit_price_raw * (1.0 - slippage_rate)
    else:
        entry_fill_price = entry_price_raw * (1.0 - slippage_rate)
        exit_fill_price = exit_price_raw * (1.0 + slippage_rate)

    qty_base = quote_amount / entry_fill_price
    entry_fee_quote = quote_amount * fee_rate
    available_quote -= quote_amount + entry_fee_quote

    exit_quote_amount = qty_base * exit_fill_price
    exit_fee_quote = exit_quote_amount * fee_rate
    if trade_direction == 1:
        gross_pnl_quote = exit_quote_amount - quote_amount
    else:
        gross_pnl_quote = quote_amount - exit_quote_amount
    available_quote += quote_amount + gross_pnl_quote - exit_fee_quote
    net_pnl_quote = gross_pnl_quote - entry_fee_quote - exit_fee_quote

    if profit_lock_enabled == 1 and net_pnl_quote > 0.0:
        locked_profit_quote = net_pnl_quote * (safe_profit_percent / 100.0)
        available_quote -= locked_profit_quote
        safe_quote += locked_profit_quote

    equity = available_quote + safe_quote
    if equity > peak_equity:
        peak_equity = equity
    elif peak_equity > 0.0:
        drawdown_pct = ((peak_equity - equity) / peak_equity) * 100.0
        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct

    trade_return_pct = (net_pnl_quote / quote_amount) * 100.0
    trade_return = net_pnl_quote / quote_amount
    bars_held = float(exit_exec_idx - entry_idx)
    if bars_held < 0.0:
        bars_held = 0.0

    closed_trade_count += 1
    if net_pnl_quote > 0.0:
        win_count += 1
        gross_profit_quote += net_pnl_quote
    elif net_pnl_quote < 0.0:
        gross_loss_quote += abs(net_pnl_quote)
    sum_trade_return += trade_return
    sum_trade_return_squared += trade_return * trade_return
    total_trade_return_pct += trade_return_pct
    total_trade_exec_bars += bars_held
    exposure_bars += bars_held
    return (
        available_quote,
        safe_quote,
        equity,
        peak_equity,
        max_drawdown_pct,
        gross_profit_quote,
        gross_loss_quote,
        closed_trade_count,
        win_count,
        sum_trade_return,
        sum_trade_return_squared,
        total_trade_return_pct,
        total_trade_exec_bars,
        exposure_bars,
    )


@nb.njit(cache=True, inline="always")
def _write_metric_outputs(
    out_index: int,
    equity: float,
    init_cash_quote: float,
    max_drawdown_pct: float,
    gross_profit_quote: float,
    gross_loss_quote: float,
    closed_trade_count: np.int32,
    win_count: np.int32,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    total_trade_return_pct: float,
    total_trade_exec_bars: float,
    exposure_bars: float,
    bars_per_year_exec: float,
    t_exec_limit: np.int32,
    out_total_return_pct: np.ndarray,
    out_max_drawdown_pct: np.ndarray,
    out_return_over_max_drawdown: np.ndarray,
    out_profit_factor: np.ndarray,
    out_trade_count: np.ndarray,
    out_sharpe_trades: np.ndarray,
    out_win_rate_pct: np.ndarray,
    out_avg_trade_ret_pct: np.ndarray,
    out_avg_trade_exec_bars: np.ndarray,
    out_exposure_pct: np.ndarray,
) -> None:
    total_return_pct = ((equity / init_cash_quote) - 1.0) * 100.0
    if gross_loss_quote > 0.0:
        profit_factor = gross_profit_quote / gross_loss_quote
    elif gross_profit_quote > 0.0:
        profit_factor = np.inf
    else:
        profit_factor = 0.0

    if max_drawdown_pct > 0.0:
        return_over_max_drawdown = total_return_pct / max_drawdown_pct
    elif total_return_pct > 0.0:
        return_over_max_drawdown = np.inf
    else:
        return_over_max_drawdown = 0.0

    if closed_trade_count > 0:
        win_rate_pct = (float(win_count) / float(closed_trade_count)) * 100.0
        avg_trade_ret_pct = total_trade_return_pct / float(closed_trade_count)
        avg_trade_exec_bars = total_trade_exec_bars / float(closed_trade_count)
    else:
        win_rate_pct = 0.0
        avg_trade_ret_pct = 0.0
        avg_trade_exec_bars = 0.0

    exposure_pct = (exposure_bars / float(t_exec_limit)) * 100.0 if t_exec_limit > 0 else 0.0
    sharpe_trades = _trade_sharpe_kernel(
        closed_trade_count,
        sum_trade_return,
        sum_trade_return_squared,
        bars_per_year_exec,
        t_exec_limit,
    )
    out_total_return_pct[out_index] = total_return_pct
    out_max_drawdown_pct[out_index] = max_drawdown_pct
    out_return_over_max_drawdown[out_index] = return_over_max_drawdown
    out_profit_factor[out_index] = profit_factor
    out_trade_count[out_index] = closed_trade_count
    out_sharpe_trades[out_index] = sharpe_trades
    out_win_rate_pct[out_index] = win_rate_pct
    out_avg_trade_ret_pct[out_index] = avg_trade_ret_pct
    out_avg_trade_exec_bars[out_index] = avg_trade_exec_bars
    out_exposure_pct[out_index] = exposure_pct


@nb.njit(cache=True)
def score_trade_list_no_risk(
    entry_exec_idx: np.ndarray,
    dir_arr: np.ndarray,
    sig_exit_exec_idx: np.ndarray,
    n_trades: np.int32,
    exec_open_1m: np.ndarray,
    exec_close_1m: np.ndarray,
    t_exec_limit: np.int32,
    init_cash_quote: float,
    sizing_mode_code: np.int8,
    sizing_value: float,
    sizing_bound: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    profit_lock_enabled: np.int8,
    bars_per_year_exec: float,
    close_on_end: np.int8,
) -> tuple[float, float, float, float, np.int32, float, float, float, float, float]:
    available_quote = init_cash_quote
    safe_quote = 0.0
    equity = init_cash_quote
    peak_equity = equity
    max_drawdown_pct = 0.0
    gross_profit_quote = 0.0
    gross_loss_quote = 0.0
    closed_trade_count = np.int32(0)
    win_count = np.int32(0)
    sum_trade_return = 0.0
    sum_trade_return_squared = 0.0
    total_trade_return_pct = 0.0
    total_trade_exec_bars = 0.0
    exposure_bars = 0.0

    for trade_index in range(n_trades):
        entry_idx = np.int32(entry_exec_idx[trade_index])
        if entry_idx >= t_exec_limit:
            continue
        exit_idx = np.int32(sig_exit_exec_idx[trade_index])
        if exit_idx < t_exec_limit:
            exit_exec_idx = exit_idx
            exit_price_raw = float(exec_open_1m[exit_exec_idx])
        elif close_on_end == 1 and t_exec_limit > 0:
            exit_exec_idx = np.int32(t_exec_limit - 1)
            exit_price_raw = float(exec_close_1m[exit_exec_idx])
        else:
            continue
        (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
        ) = _apply_no_risk_trade_to_state(
            entry_idx,
            np.int8(dir_arr[trade_index]),
            exit_exec_idx,
            exit_price_raw,
            exec_open_1m,
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            sizing_mode_code,
            sizing_value,
            sizing_bound,
            fee_rate,
            slippage_rate,
            safe_profit_percent,
            profit_lock_enabled,
        )

    total_return_pct = ((equity / init_cash_quote) - 1.0) * 100.0
    if gross_loss_quote > 0.0:
        profit_factor = gross_profit_quote / gross_loss_quote
    elif gross_profit_quote > 0.0:
        profit_factor = np.inf
    else:
        profit_factor = 0.0

    if max_drawdown_pct > 0.0:
        return_over_max_drawdown = total_return_pct / max_drawdown_pct
    elif total_return_pct > 0.0:
        return_over_max_drawdown = np.inf
    else:
        return_over_max_drawdown = 0.0

    if closed_trade_count > 0:
        win_rate_pct = (float(win_count) / float(closed_trade_count)) * 100.0
        avg_trade_ret_pct = total_trade_return_pct / float(closed_trade_count)
        avg_trade_exec_bars = total_trade_exec_bars / float(closed_trade_count)
    else:
        win_rate_pct = 0.0
        avg_trade_ret_pct = 0.0
        avg_trade_exec_bars = 0.0

    exposure_pct = (exposure_bars / float(t_exec_limit)) * 100.0 if t_exec_limit > 0 else 0.0
    sharpe_trades = _trade_sharpe_kernel(
        closed_trade_count,
        sum_trade_return,
        sum_trade_return_squared,
        bars_per_year_exec,
        t_exec_limit,
    )
    return (
        total_return_pct,
        max_drawdown_pct,
        return_over_max_drawdown,
        profit_factor,
        closed_trade_count,
        sharpe_trades,
        win_rate_pct,
        avg_trade_ret_pct,
        avg_trade_exec_bars,
        exposure_pct,
    )


@nb.njit(parallel=True, cache=True, fastmath=True)
def evaluate_no_risk_streaming_two(
    combo_left_idx: np.ndarray,
    combo_right_idx: np.ndarray,
    left_trade_t: np.ndarray,
    right_trade_t: np.ndarray,
    sig_entry_exec_idx: np.ndarray,
    exec_open_1m: np.ndarray,
    exec_close_1m: np.ndarray,
    t_exec_limit: np.int32,
    init_cash_quote: float,
    sizing_mode_code: np.int8,
    sizing_value: float,
    sizing_bound: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    profit_lock_enabled: np.int8,
    bars_per_year_exec: float,
    close_on_end: np.int8,
    direction_mode: np.int8,
    out_total_return_pct: np.ndarray,
    out_max_drawdown_pct: np.ndarray,
    out_return_over_max_drawdown: np.ndarray,
    out_profit_factor: np.ndarray,
    out_trade_count: np.ndarray,
    out_sharpe_trades: np.ndarray,
    out_win_rate_pct: np.ndarray,
    out_avg_trade_ret_pct: np.ndarray,
    out_avg_trade_exec_bars: np.ndarray,
    out_exposure_pct: np.ndarray,
) -> None:
    combo_count = int(combo_left_idx.shape[0])
    signal_count = int(left_trade_t.shape[1])
    for combo_pos in nb.prange(combo_count):
        left_row = combo_left_idx[combo_pos]
        right_row = combo_right_idx[combo_pos]
        available_quote = init_cash_quote
        safe_quote = 0.0
        equity = init_cash_quote
        peak_equity = equity
        max_drawdown_pct = 0.0
        gross_profit_quote = 0.0
        gross_loss_quote = 0.0
        closed_trade_count = np.int32(0)
        win_count = np.int32(0)
        sum_trade_return = 0.0
        sum_trade_return_squared = 0.0
        total_trade_return_pct = 0.0
        total_trade_exec_bars = 0.0
        exposure_bars = 0.0
        current_dir = np.int8(0)
        current_entry = np.int32(0)

        for signal_idx in range(signal_count):
            raw_dir = _consensus_dir2(
                left_trade_t[left_row, signal_idx],
                right_trade_t[right_row, signal_idx],
            )
            dirn = _apply_direction_mode(raw_dir, direction_mode)
            if dirn == 0 and not (direction_mode == _DIRECTION_LONG_ONLY and current_dir != 0):
                continue
            entry_exec = sig_entry_exec_idx[signal_idx]
            if entry_exec >= t_exec_limit:
                break
            if dirn == 0:
                (
                    available_quote,
                    safe_quote,
                    equity,
                    peak_equity,
                    max_drawdown_pct,
                    gross_profit_quote,
                    gross_loss_quote,
                    closed_trade_count,
                    win_count,
                    sum_trade_return,
                    sum_trade_return_squared,
                    total_trade_return_pct,
                    total_trade_exec_bars,
                    exposure_bars,
                ) = _apply_no_risk_trade_to_state(
                    current_entry,
                    current_dir,
                    np.int32(entry_exec),
                    float(exec_open_1m[entry_exec]),
                    exec_open_1m,
                    available_quote,
                    safe_quote,
                    equity,
                    peak_equity,
                    max_drawdown_pct,
                    gross_profit_quote,
                    gross_loss_quote,
                    closed_trade_count,
                    win_count,
                    sum_trade_return,
                    sum_trade_return_squared,
                    total_trade_return_pct,
                    total_trade_exec_bars,
                    exposure_bars,
                    sizing_mode_code,
                    sizing_value,
                    sizing_bound,
                    fee_rate,
                    slippage_rate,
                    safe_profit_percent,
                    profit_lock_enabled,
                )
                current_dir = np.int8(0)
                current_entry = np.int32(0)
            elif current_dir == 0:
                current_dir = dirn
                current_entry = np.int32(entry_exec)
            elif dirn != current_dir:
                (
                    available_quote,
                    safe_quote,
                    equity,
                    peak_equity,
                    max_drawdown_pct,
                    gross_profit_quote,
                    gross_loss_quote,
                    closed_trade_count,
                    win_count,
                    sum_trade_return,
                    sum_trade_return_squared,
                    total_trade_return_pct,
                    total_trade_exec_bars,
                    exposure_bars,
                ) = _apply_no_risk_trade_to_state(
                    current_entry,
                    current_dir,
                    np.int32(entry_exec),
                    float(exec_open_1m[entry_exec]),
                    exec_open_1m,
                    available_quote,
                    safe_quote,
                    equity,
                    peak_equity,
                    max_drawdown_pct,
                    gross_profit_quote,
                    gross_loss_quote,
                    closed_trade_count,
                    win_count,
                    sum_trade_return,
                    sum_trade_return_squared,
                    total_trade_return_pct,
                    total_trade_exec_bars,
                    exposure_bars,
                    sizing_mode_code,
                    sizing_value,
                    sizing_bound,
                    fee_rate,
                    slippage_rate,
                    safe_profit_percent,
                    profit_lock_enabled,
                )
                current_dir = dirn
                current_entry = np.int32(entry_exec)

        if current_dir != 0 and close_on_end == 1 and t_exec_limit > 0:
            exit_exec_idx = np.int32(t_exec_limit - 1)
            (
                available_quote,
                safe_quote,
                equity,
                peak_equity,
                max_drawdown_pct,
                gross_profit_quote,
                gross_loss_quote,
                closed_trade_count,
                win_count,
                sum_trade_return,
                sum_trade_return_squared,
                total_trade_return_pct,
                total_trade_exec_bars,
                exposure_bars,
            ) = _apply_no_risk_trade_to_state(
                current_entry,
                current_dir,
                exit_exec_idx,
                float(exec_close_1m[exit_exec_idx]),
                exec_open_1m,
                available_quote,
                safe_quote,
                equity,
                peak_equity,
                max_drawdown_pct,
                gross_profit_quote,
                gross_loss_quote,
                closed_trade_count,
                win_count,
                sum_trade_return,
                sum_trade_return_squared,
                total_trade_return_pct,
                total_trade_exec_bars,
                exposure_bars,
                sizing_mode_code,
                sizing_value,
                sizing_bound,
                fee_rate,
                slippage_rate,
                safe_profit_percent,
                profit_lock_enabled,
            )
        _write_metric_outputs(
            combo_pos,
            equity,
            init_cash_quote,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            bars_per_year_exec,
            t_exec_limit,
            out_total_return_pct,
            out_max_drawdown_pct,
            out_return_over_max_drawdown,
            out_profit_factor,
            out_trade_count,
            out_sharpe_trades,
            out_win_rate_pct,
            out_avg_trade_ret_pct,
            out_avg_trade_exec_bars,
            out_exposure_pct,
        )


@nb.njit(parallel=True, cache=True, fastmath=True)
def evaluate_no_risk_event_segments_two(
    combo_left_idx: np.ndarray,
    combo_right_idx: np.ndarray,
    left_segment_starts: np.ndarray,
    left_segment_ends: np.ndarray,
    left_segment_values: np.ndarray,
    left_segment_counts: np.ndarray,
    right_segment_starts: np.ndarray,
    right_segment_ends: np.ndarray,
    right_segment_values: np.ndarray,
    right_segment_counts: np.ndarray,
    sig_entry_exec_idx: np.ndarray,
    exec_open_1m: np.ndarray,
    exec_close_1m: np.ndarray,
    t_exec_limit: np.int32,
    init_cash_quote: float,
    sizing_mode_code: np.int8,
    sizing_value: float,
    sizing_bound: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    profit_lock_enabled: np.int8,
    bars_per_year_exec: float,
    close_on_end: np.int8,
    direction_mode: np.int8,
    out_total_return_pct: np.ndarray,
    out_max_drawdown_pct: np.ndarray,
    out_return_over_max_drawdown: np.ndarray,
    out_profit_factor: np.ndarray,
    out_trade_count: np.ndarray,
    out_sharpe_trades: np.ndarray,
    out_win_rate_pct: np.ndarray,
    out_avg_trade_ret_pct: np.ndarray,
    out_avg_trade_exec_bars: np.ndarray,
    out_exposure_pct: np.ndarray,
) -> None:
    combo_count = int(combo_left_idx.shape[0])
    for combo_pos in nb.prange(combo_count):
        left_row = combo_left_idx[combo_pos]
        right_row = combo_right_idx[combo_pos]
        available_quote = init_cash_quote
        safe_quote = 0.0
        equity = init_cash_quote
        peak_equity = equity
        max_drawdown_pct = 0.0
        gross_profit_quote = 0.0
        gross_loss_quote = 0.0
        closed_trade_count = np.int32(0)
        win_count = np.int32(0)
        sum_trade_return = 0.0
        sum_trade_return_squared = 0.0
        total_trade_return_pct = 0.0
        total_trade_exec_bars = 0.0
        exposure_bars = 0.0
        current_dir = np.int8(0)
        current_entry = np.int32(0)
        left_segment_idx = 0
        right_segment_idx = 0

        while (
            left_segment_idx < left_segment_counts[left_row]
            and right_segment_idx < right_segment_counts[right_row]
        ):
            left_start = left_segment_starts[left_row, left_segment_idx]
            left_end = left_segment_ends[left_row, left_segment_idx]
            right_start = right_segment_starts[right_row, right_segment_idx]
            right_end = right_segment_ends[right_row, right_segment_idx]
            segment_start = left_start if left_start >= right_start else right_start
            segment_end = left_end if left_end <= right_end else right_end

            if segment_start < segment_end:
                raw_dir = _consensus_dir2(
                    left_segment_values[left_row, left_segment_idx],
                    right_segment_values[right_row, right_segment_idx],
                )
                dirn = _apply_direction_mode(raw_dir, direction_mode)
                if dirn != 0 or (direction_mode == _DIRECTION_LONG_ONLY and current_dir != 0):
                    entry_exec = sig_entry_exec_idx[segment_start]
                    if entry_exec >= t_exec_limit:
                        break
                    if dirn == 0:
                        (
                            available_quote,
                            safe_quote,
                            equity,
                            peak_equity,
                            max_drawdown_pct,
                            gross_profit_quote,
                            gross_loss_quote,
                            closed_trade_count,
                            win_count,
                            sum_trade_return,
                            sum_trade_return_squared,
                            total_trade_return_pct,
                            total_trade_exec_bars,
                            exposure_bars,
                        ) = _apply_no_risk_trade_to_state(
                            current_entry,
                            current_dir,
                            np.int32(entry_exec),
                            float(exec_open_1m[entry_exec]),
                            exec_open_1m,
                            available_quote,
                            safe_quote,
                            equity,
                            peak_equity,
                            max_drawdown_pct,
                            gross_profit_quote,
                            gross_loss_quote,
                            closed_trade_count,
                            win_count,
                            sum_trade_return,
                            sum_trade_return_squared,
                            total_trade_return_pct,
                            total_trade_exec_bars,
                            exposure_bars,
                            sizing_mode_code,
                            sizing_value,
                            sizing_bound,
                            fee_rate,
                            slippage_rate,
                            safe_profit_percent,
                            profit_lock_enabled,
                        )
                        current_dir = np.int8(0)
                        current_entry = np.int32(0)
                    elif current_dir == 0:
                        current_dir = dirn
                        current_entry = np.int32(entry_exec)
                    elif dirn != current_dir:
                        (
                            available_quote,
                            safe_quote,
                            equity,
                            peak_equity,
                            max_drawdown_pct,
                            gross_profit_quote,
                            gross_loss_quote,
                            closed_trade_count,
                            win_count,
                            sum_trade_return,
                            sum_trade_return_squared,
                            total_trade_return_pct,
                            total_trade_exec_bars,
                            exposure_bars,
                        ) = _apply_no_risk_trade_to_state(
                            current_entry,
                            current_dir,
                            np.int32(entry_exec),
                            float(exec_open_1m[entry_exec]),
                            exec_open_1m,
                            available_quote,
                            safe_quote,
                            equity,
                            peak_equity,
                            max_drawdown_pct,
                            gross_profit_quote,
                            gross_loss_quote,
                            closed_trade_count,
                            win_count,
                            sum_trade_return,
                            sum_trade_return_squared,
                            total_trade_return_pct,
                            total_trade_exec_bars,
                            exposure_bars,
                            sizing_mode_code,
                            sizing_value,
                            sizing_bound,
                            fee_rate,
                            slippage_rate,
                            safe_profit_percent,
                            profit_lock_enabled,
                        )
                        current_dir = dirn
                        current_entry = np.int32(entry_exec)

            if left_end == segment_end:
                left_segment_idx += 1
            if right_end == segment_end:
                right_segment_idx += 1

        if current_dir != 0 and close_on_end == 1 and t_exec_limit > 0:
            exit_exec_idx = np.int32(t_exec_limit - 1)
            (
                available_quote,
                safe_quote,
                equity,
                peak_equity,
                max_drawdown_pct,
                gross_profit_quote,
                gross_loss_quote,
                closed_trade_count,
                win_count,
                sum_trade_return,
                sum_trade_return_squared,
                total_trade_return_pct,
                total_trade_exec_bars,
                exposure_bars,
            ) = _apply_no_risk_trade_to_state(
                current_entry,
                current_dir,
                exit_exec_idx,
                float(exec_close_1m[exit_exec_idx]),
                exec_open_1m,
                available_quote,
                safe_quote,
                equity,
                peak_equity,
                max_drawdown_pct,
                gross_profit_quote,
                gross_loss_quote,
                closed_trade_count,
                win_count,
                sum_trade_return,
                sum_trade_return_squared,
                total_trade_return_pct,
                total_trade_exec_bars,
                exposure_bars,
                sizing_mode_code,
                sizing_value,
                sizing_bound,
                fee_rate,
                slippage_rate,
                safe_profit_percent,
                profit_lock_enabled,
            )
        _write_metric_outputs(
            combo_pos,
            equity,
            init_cash_quote,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            bars_per_year_exec,
            t_exec_limit,
            out_total_return_pct,
            out_max_drawdown_pct,
            out_return_over_max_drawdown,
            out_profit_factor,
            out_trade_count,
            out_sharpe_trades,
            out_win_rate_pct,
            out_avg_trade_ret_pct,
            out_avg_trade_exec_bars,
            out_exposure_pct,
        )


@nb.njit(parallel=True, cache=True, fastmath=True)
def evaluate_no_risk_event_segments_n(
    combo_idx_by_indicator: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    segment_values: np.ndarray,
    segment_counts: np.ndarray,
    segment_pos_workspace: np.ndarray,
    sig_entry_exec_idx: np.ndarray,
    exec_open_1m: np.ndarray,
    exec_close_1m: np.ndarray,
    t_exec_limit: np.int32,
    init_cash_quote: float,
    sizing_mode_code: np.int8,
    sizing_value: float,
    sizing_bound: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    profit_lock_enabled: np.int8,
    bars_per_year_exec: float,
    close_on_end: np.int8,
    direction_mode: np.int8,
    out_total_return_pct: np.ndarray,
    out_max_drawdown_pct: np.ndarray,
    out_return_over_max_drawdown: np.ndarray,
    out_profit_factor: np.ndarray,
    out_trade_count: np.ndarray,
    out_sharpe_trades: np.ndarray,
    out_win_rate_pct: np.ndarray,
    out_avg_trade_ret_pct: np.ndarray,
    out_avg_trade_exec_bars: np.ndarray,
    out_exposure_pct: np.ndarray,
) -> None:
    arity = int(combo_idx_by_indicator.shape[0])
    combo_count = int(combo_idx_by_indicator.shape[1])
    for combo_pos in nb.prange(combo_count):
        for indicator_pos in range(arity):
            segment_pos_workspace[combo_pos, indicator_pos] = np.int32(0)

        available_quote = init_cash_quote
        safe_quote = 0.0
        equity = init_cash_quote
        peak_equity = equity
        max_drawdown_pct = 0.0
        gross_profit_quote = 0.0
        gross_loss_quote = 0.0
        closed_trade_count = np.int32(0)
        win_count = np.int32(0)
        sum_trade_return = 0.0
        sum_trade_return_squared = 0.0
        total_trade_return_pct = 0.0
        total_trade_exec_bars = 0.0
        exposure_bars = 0.0
        current_dir = np.int8(0)
        current_entry = np.int32(0)

        while True:
            active = True
            segment_start = np.int32(0)
            segment_end = np.int32(2147483647)
            for indicator_pos in range(arity):
                row_idx = combo_idx_by_indicator[indicator_pos, combo_pos]
                segment_idx = segment_pos_workspace[combo_pos, indicator_pos]
                if segment_idx >= segment_counts[indicator_pos, row_idx]:
                    active = False
                    break
                start_value = segment_starts[indicator_pos, row_idx, segment_idx]
                end_value = segment_ends[indicator_pos, row_idx, segment_idx]
                if start_value > segment_start:
                    segment_start = start_value
                if end_value < segment_end:
                    segment_end = end_value
            if not active:
                break

            if segment_start < segment_end:
                first_row_idx = combo_idx_by_indicator[0, combo_pos]
                first_segment_idx = segment_pos_workspace[combo_pos, 0]
                raw_dir = np.int8(segment_values[0, first_row_idx, first_segment_idx])
                if raw_dir != 0:
                    for indicator_pos in range(1, arity):
                        row_idx = combo_idx_by_indicator[indicator_pos, combo_pos]
                        segment_idx = segment_pos_workspace[combo_pos, indicator_pos]
                        if segment_values[indicator_pos, row_idx, segment_idx] != raw_dir:
                            raw_dir = np.int8(0)
                            break
                dirn = _apply_direction_mode(raw_dir, direction_mode)
                if dirn != 0 or (direction_mode == _DIRECTION_LONG_ONLY and current_dir != 0):
                    entry_exec = sig_entry_exec_idx[segment_start]
                    if entry_exec >= t_exec_limit:
                        break
                    if dirn == 0:
                        (
                            available_quote,
                            safe_quote,
                            equity,
                            peak_equity,
                            max_drawdown_pct,
                            gross_profit_quote,
                            gross_loss_quote,
                            closed_trade_count,
                            win_count,
                            sum_trade_return,
                            sum_trade_return_squared,
                            total_trade_return_pct,
                            total_trade_exec_bars,
                            exposure_bars,
                        ) = _apply_no_risk_trade_to_state(
                            current_entry,
                            current_dir,
                            np.int32(entry_exec),
                            float(exec_open_1m[entry_exec]),
                            exec_open_1m,
                            available_quote,
                            safe_quote,
                            equity,
                            peak_equity,
                            max_drawdown_pct,
                            gross_profit_quote,
                            gross_loss_quote,
                            closed_trade_count,
                            win_count,
                            sum_trade_return,
                            sum_trade_return_squared,
                            total_trade_return_pct,
                            total_trade_exec_bars,
                            exposure_bars,
                            sizing_mode_code,
                            sizing_value,
                            sizing_bound,
                            fee_rate,
                            slippage_rate,
                            safe_profit_percent,
                            profit_lock_enabled,
                        )
                        current_dir = np.int8(0)
                        current_entry = np.int32(0)
                    elif current_dir == 0:
                        current_dir = dirn
                        current_entry = np.int32(entry_exec)
                    elif dirn != current_dir:
                        (
                            available_quote,
                            safe_quote,
                            equity,
                            peak_equity,
                            max_drawdown_pct,
                            gross_profit_quote,
                            gross_loss_quote,
                            closed_trade_count,
                            win_count,
                            sum_trade_return,
                            sum_trade_return_squared,
                            total_trade_return_pct,
                            total_trade_exec_bars,
                            exposure_bars,
                        ) = _apply_no_risk_trade_to_state(
                            current_entry,
                            current_dir,
                            np.int32(entry_exec),
                            float(exec_open_1m[entry_exec]),
                            exec_open_1m,
                            available_quote,
                            safe_quote,
                            equity,
                            peak_equity,
                            max_drawdown_pct,
                            gross_profit_quote,
                            gross_loss_quote,
                            closed_trade_count,
                            win_count,
                            sum_trade_return,
                            sum_trade_return_squared,
                            total_trade_return_pct,
                            total_trade_exec_bars,
                            exposure_bars,
                            sizing_mode_code,
                            sizing_value,
                            sizing_bound,
                            fee_rate,
                            slippage_rate,
                            safe_profit_percent,
                            profit_lock_enabled,
                        )
                        current_dir = dirn
                        current_entry = np.int32(entry_exec)

            for indicator_pos in range(arity):
                row_idx = combo_idx_by_indicator[indicator_pos, combo_pos]
                segment_idx = segment_pos_workspace[combo_pos, indicator_pos]
                if segment_ends[indicator_pos, row_idx, segment_idx] == segment_end:
                    segment_pos_workspace[combo_pos, indicator_pos] = np.int32(
                        segment_idx + 1
                    )

        if current_dir != 0 and close_on_end == 1 and t_exec_limit > 0:
            exit_exec_idx = np.int32(t_exec_limit - 1)
            (
                available_quote,
                safe_quote,
                equity,
                peak_equity,
                max_drawdown_pct,
                gross_profit_quote,
                gross_loss_quote,
                closed_trade_count,
                win_count,
                sum_trade_return,
                sum_trade_return_squared,
                total_trade_return_pct,
                total_trade_exec_bars,
                exposure_bars,
            ) = _apply_no_risk_trade_to_state(
                current_entry,
                current_dir,
                exit_exec_idx,
                float(exec_close_1m[exit_exec_idx]),
                exec_open_1m,
                available_quote,
                safe_quote,
                equity,
                peak_equity,
                max_drawdown_pct,
                gross_profit_quote,
                gross_loss_quote,
                closed_trade_count,
                win_count,
                sum_trade_return,
                sum_trade_return_squared,
                total_trade_return_pct,
                total_trade_exec_bars,
                exposure_bars,
                sizing_mode_code,
                sizing_value,
                sizing_bound,
                fee_rate,
                slippage_rate,
                safe_profit_percent,
                profit_lock_enabled,
            )
        _write_metric_outputs(
            combo_pos,
            equity,
            init_cash_quote,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            bars_per_year_exec,
            t_exec_limit,
            out_total_return_pct,
            out_max_drawdown_pct,
            out_return_over_max_drawdown,
            out_profit_factor,
            out_trade_count,
            out_sharpe_trades,
            out_win_rate_pct,
            out_avg_trade_ret_pct,
            out_avg_trade_exec_bars,
            out_exposure_pct,
        )


def _zero_stage_timings() -> dict[str, float]:
    return {
        SERVICE_WARMUP_STAGE_NAME: 0.0,
        SELF_CHECK_STAGE_NAME: 0.0,
        EXACT_SCORING_STAGE_NAME: 0.0,
        HEAP_UPDATE_STAGE_NAME: 0.0,
        TOP_RESULT_PROXY_FILL_STAGE_NAME: 0.0,
    }


def _reject_unless_no_risk(
    *,
    normalized_request: Mapping[str, Any],
    combo_planning_result: BacktestComboPlanningResult,
) -> None:
    risk = normalized_request.get("risk")
    if not isinstance(risk, Mapping) or risk.get("mode") != "none":
        raise BacktestNoRiskExactRejected("no-risk exact scoring requires risk.mode='none'")
    if combo_planning_result.backend.risk_mode != "none":
        raise BacktestNoRiskExactRejected("combo planning backend must use risk_mode='none'")


def _execution_settings_from_normalized(
    normalized_request: Mapping[str, Any],
    *,
    bars_per_year_exec: float,
) -> _ExecutionSettings:
    execution = normalized_request.get("execution")
    if not isinstance(execution, Mapping):
        raise BacktestNoRiskExactRejected("normalized_request.execution must be a mapping")
    direction_mode = str(execution.get("direction_mode", "long_short_reversal"))
    if direction_mode == "long_only":
        direction_mode_code = _DIRECTION_LONG_ONLY
    elif direction_mode == "long_short_reversal":
        direction_mode_code = _DIRECTION_LONG_SHORT_REVERSAL
    else:
        raise BacktestNoRiskExactRejected(f"Unsupported direction_mode={direction_mode!r}")

    sizing = execution.get("sizing", {"mode": "all_in"})
    if not isinstance(sizing, Mapping):
        raise BacktestNoRiskExactRejected("execution.sizing must be a mapping")
    sizing_mode = str(sizing.get("mode", "all_in"))
    sizing_mode_code, sizing_value, sizing_bound = _sizing_kernel_values(sizing_mode, sizing)

    profit_lock = execution.get("profit_lock", {"enabled": False})
    if not isinstance(profit_lock, Mapping):
        raise BacktestNoRiskExactRejected("execution.profit_lock must be a mapping")
    profit_lock_enabled = np.int8(1 if bool(profit_lock.get("enabled", False)) else 0)
    safe_profit_percent = float(profit_lock.get("safe_profit_percent", 0.0))
    return _ExecutionSettings(
        direction_mode=direction_mode,
        direction_mode_code=direction_mode_code,
        fee_rate=float(execution.get("fee_rate", 0.0)),
        slippage_rate=float(execution.get("slippage_rate", 0.0)),
        initial_cash_quote=float(execution.get("initial_cash_quote", 10000.0)),
        bars_per_year_exec=bars_per_year_exec,
        sizing_mode_code=sizing_mode_code,
        sizing_value=sizing_value,
        sizing_bound=sizing_bound,
        profit_lock_enabled=profit_lock_enabled,
        safe_profit_percent=safe_profit_percent,
        close_on_end=np.int8(1 if bool(execution.get("close_on_end", True)) else 0),
    )


def _sizing_kernel_values(
    sizing_mode: str,
    sizing: Mapping[str, Any],
) -> tuple[np.int8, float, float]:
    if sizing_mode == "all_in":
        return _SIZING_ALL_IN, 0.0, 0.0
    if sizing_mode == "fixed_quote":
        return (
            _SIZING_FIXED_QUOTE,
            float(sizing.get("quote_amount", sizing.get("fixed_quote", 0.0))),
            0.0,
        )
    if sizing_mode == "fixed_equity_pct":
        return _SIZING_FIXED_EQUITY_PCT, float(sizing.get("equity_pct", 100.0)), 0.0
    if sizing_mode == "fixed_equity_pct_min_quote":
        return (
            _SIZING_FIXED_EQUITY_PCT_MIN_QUOTE,
            float(sizing.get("equity_pct", 100.0)),
            float(sizing.get("min_quote", 0.0)),
        )
    if sizing_mode == "fixed_equity_pct_max_quote":
        return (
            _SIZING_FIXED_EQUITY_PCT_MAX_QUOTE,
            float(sizing.get("equity_pct", 100.0)),
            float(sizing.get("max_quote", 0.0)),
        )
    raise BacktestNoRiskExactRejected(f"Unsupported sizing mode={sizing_mode!r}")


def _ranking_from_normalized(normalized_request: Mapping[str, Any]) -> tuple[str, str]:
    ranking = normalized_request.get("ranking")
    if isinstance(ranking, Mapping):
        metric = str(ranking.get("primary_metric", "total_return_pct"))
        direction = str(ranking.get("direction", "desc"))
    else:
        metric = str(normalized_request.get("sort_metric", "total_return_pct"))
        direction = "desc"
    if direction not in {"asc", "desc"}:
        raise BacktestNoRiskExactRejected("ranking.direction must be 'asc' or 'desc'")
    _validate_ranking_metric(metric)
    return metric, direction


def _validate_ranking_metric(metric: str) -> None:
    allowed = {
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "trade_count",
        "sharpe_trades",
        "win_rate_pct",
        "avg_trade_ret_pct",
        "avg_trade_exec_bars",
        "exposure_pct",
    }
    if metric not in allowed:
        raise BacktestNoRiskExactRejected(f"Unsupported ranking metric={metric!r}")


def _score_value(*, scores: _NoRiskScoreArrays, metric: str, index: int) -> float:
    if metric == "total_return_pct":
        return float(scores.total_return_pct[index])
    if metric == "max_drawdown_pct":
        return float(scores.max_drawdown_pct[index])
    if metric == "return_over_max_drawdown":
        return float(scores.return_over_max_drawdown[index])
    if metric == "profit_factor":
        return float(scores.profit_factor[index])
    if metric == "trade_count":
        return float(scores.trade_count[index])
    if metric == "sharpe_trades":
        return float(scores.sharpe_trades[index])
    if metric == "win_rate_pct":
        return float(scores.win_rate_pct[index])
    if metric == "avg_trade_ret_pct":
        return float(scores.avg_trade_ret_pct[index])
    if metric == "avg_trade_exec_bars":
        return float(scores.avg_trade_exec_bars[index])
    if metric == "exposure_pct":
        return float(scores.exposure_pct[index])
    raise BacktestNoRiskExactRejected(f"Unsupported ranking metric={metric!r}")


def _ordered_pools(
    prepared_result: BacktestPreparePoolsResult,
) -> tuple[PreparedIndicatorPool, ...]:
    pool_by_id = {pool.indicator_id: pool for pool in prepared_result.indicator_pools}
    try:
        return tuple(pool_by_id[indicator_id] for indicator_id in prepared_result.indicator_ids)
    except KeyError as error:
        raise BacktestNoRiskExactRejected("prepared pools missing indicator ids") from error


def _filter_result_as_combo_chunk(filter_result: BacktestProxyFilterResult) -> BacktestComboChunk:
    return BacktestComboChunk(
        indicator_ids=filter_result.indicator_ids,
        rows_by_indicator=filter_result.selected_rows_by_indicator,
    )


def _backend_display_name(combo_planning_result: BacktestComboPlanningResult) -> str:
    backend = combo_planning_result.backend
    if backend.backend_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND:
        return f"event_segments_{backend.arity}_no_risk"
    return backend.backend_id


__all__ = [
    "BENCHMARK_TOP_K_DEFAULT",
    "EXACT_SCORING_STAGE_NAME",
    "HEAP_UPDATE_STAGE_NAME",
    "NO_RISK_IMPLEMENTATION_ID",
    "SAMPLE_WARMUP_TOP_K",
    "SELF_CHECK_STAGE_NAME",
    "SERVICE_WARMUP_STAGE_NAME",
    "TOP_RESULT_PROXY_FILL_STAGE_NAME",
    "BacktestNoRiskExactRejected",
    "BacktestNoRiskExactScoringService",
    "apply_direction_mode_py",
    "build_trade_list_for_indicator_rows_slow",
    "evaluate_no_risk_event_segments_n",
    "evaluate_no_risk_event_segments_two",
    "evaluate_no_risk_exact_chunk",
    "evaluate_no_risk_reference_rows_slow",
    "evaluate_no_risk_streaming_two",
    "heap_update",
    "proxy_for_indicator_rows",
    "run_fast_vs_reference_self_check_two",
    "score_trade_list_no_risk",
    "top_result_proxy_fill",
]
