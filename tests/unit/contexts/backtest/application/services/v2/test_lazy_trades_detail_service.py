from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast
from uuid import uuid4

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
    FilesystemBacktestArtifactContextResolver,
)
from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestNoRiskTopResult,
    BacktestPreparePoolsConfig,
    BacktestTpSlHitTimesSubset,
    BacktestTpSlTopResult,
)
from trading.contexts.backtest.application.ports import (
    BacktestLazyTradesCacheKey,
    BacktestLazyTradesCacheReadResult,
)
from trading.contexts.backtest.application.services.v2 import (
    BACKTEST_ERROR_VARIANT_CONFLICT,
    DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS,
    LAZY_TRADES_CACHE_HIT_STAGE_NAME,
    LAZY_TRADES_COMPUTE_STAGE_NAME,
    BacktestLazyTradesDetailService,
    BacktestPreparePoolsService,
    BacktestTopResultAssemblyService,
    BacktestTpSlHitTimesService,
)
from trading.contexts.backtest.application.services.v2 import lazy_trades_detail as detail_module
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobTopVariant,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import UserId


def test_lazy_trades_cache_miss_recomputes_and_writes_cache(tmp_path: Path) -> None:
    service, cache, job, row = _service_fixture(tmp_path=tmp_path, risk_mode="none")

    result = service.execute(
        job=job,
        row=row,
        public_variant_key=row.payload_json["public_variant_key"],
    )

    assert result.cache["status"] == "miss"
    assert LAZY_TRADES_COMPUTE_STAGE_NAME in result.timing
    assert len(cache.writes) == 1
    assert result.trades
    assert result.trades[0]["exit_reason"] == "close_on_end"
    assert result.summary_metrics["trade_count"] == pytest.approx(float(len(result.trades)))
    assert result.canonical_variant_params["execution"]["sizing"]["mode"] == "fixed_equity_pct"
    assert result.canonical_variant_params["execution"]["profit_lock"]["enabled"] is False


def test_lazy_trades_cache_hit_returns_cached_payload_without_recompute(tmp_path: Path) -> None:
    service, cache, job, row = _service_fixture(tmp_path=tmp_path, risk_mode="none")
    cached_payload = _cached_payload(job=job, row=row)
    cache.read_result = BacktestLazyTradesCacheReadResult(status="hit", payload=cached_payload)
    service = BacktestLazyTradesDetailService(
        prepare_pools=_FailingPreparePools(),  # type: ignore[arg-type]
        tp_sl_hit_times=_FailingHitTimes(),  # type: ignore[arg-type]
        cache=cache,
    )

    result = service.execute(
        job=job,
        row=row,
        public_variant_key=row.payload_json["public_variant_key"],
    )

    assert result.cache["status"] == "hit"
    assert result.trades[0]["exit_reason"] == "signal"
    assert LAZY_TRADES_CACHE_HIT_STAGE_NAME in result.timing
    assert cache.writes == ()


def test_lazy_trades_cache_read_failure_recomputes_successfully(tmp_path: Path) -> None:
    service, cache, job, row = _service_fixture(tmp_path=tmp_path, risk_mode="none")
    cache.read_result = BacktestLazyTradesCacheReadResult(
        status="read_failed",
        warning="bad json",
    )

    result = service.execute(
        job=job,
        row=row,
        public_variant_key=row.payload_json["public_variant_key"],
    )

    assert result.cache["status"] == "read_failed"
    assert result.cache["warning"] == "bad json"
    assert result.trades
    assert len(cache.writes) == 1


def test_lazy_trades_cache_key_uses_funding_manifest_hash(tmp_path: Path) -> None:
    service, _cache, job, row = _service_fixture(tmp_path=tmp_path, risk_mode="none")
    job = _job_with_funding_manifest_hash(job=job, funding_manifest_hash="f" * 64)

    probe = service.read_cached(
        job=job,
        row=row,
        public_variant_key=row.payload_json["public_variant_key"],
    )

    assert probe.cache_key.funding_manifest_hash == "f" * 64
    assert probe.cache_key.as_mapping()["funding_manifest_hash"] == "f" * 64


def test_lazy_trades_cache_write_failure_returns_recomputed_payload(tmp_path: Path) -> None:
    service, cache, job, row = _service_fixture(tmp_path=tmp_path, risk_mode="none")
    cache.raise_on_write = RuntimeError("disk full")

    result = service.execute(
        job=job,
        row=row,
        public_variant_key=row.payload_json["public_variant_key"],
    )

    assert result.cache["status"] == "write_failed"
    assert result.cache["warning"] == "disk full"
    assert result.trades


def test_lazy_trades_recompute_retains_top_variant_row_filtered_by_prefilter(
    tmp_path: Path,
) -> None:
    service, _cache, job, row = _service_fixture(
        tmp_path=tmp_path,
        risk_mode="none",
        top_row_id=0,
        row_prefilter_top_fraction=0.5,
    )

    result = service.execute(
        job=job,
        row=row,
        public_variant_key=row.payload_json["public_variant_key"],
    )

    indicator_params = result.canonical_variant_params["indicators"][0]
    assert indicator_params["row_id"] == 0
    assert result.trades


def test_lazy_trades_detects_variant_key_hash_mismatch(tmp_path: Path) -> None:
    service, _cache, job, row = _service_fixture(tmp_path=tmp_path, risk_mode="none")
    bad_row = BacktestJobTopVariant(
        job_id=row.job_id,
        rank=row.rank,
        variant_key="f" * 64,
        indicator_variant_key=row.indicator_variant_key,
        variant_index=row.variant_index,
        total_return_pct=row.total_return_pct,
        payload_json=dict(row.payload_json),
        summary_metrics_json=dict(row.summary_metrics_json),
        best_tp_pct=row.best_tp_pct,
        best_sl_pct=row.best_sl_pct,
        report_table_md=None,
        trades_json=None,
        updated_at=row.updated_at,
    )

    with pytest.raises(RoehubError) as exc_info:
        service.execute(
            job=job,
            row=bad_row,
            public_variant_key=str(row.payload_json["public_variant_key"]),
        )

    assert exc_info.value.code == BACKTEST_ERROR_VARIANT_CONFLICT


def test_lazy_trades_tp_sl_payload_preserves_selected_cell(tmp_path: Path) -> None:
    service, _cache, job, row = _service_fixture(tmp_path=tmp_path, risk_mode="tp_sl_grid")

    result = service.execute(
        job=job,
        row=row,
        public_variant_key=row.payload_json["public_variant_key"],
    )

    assert result.readable_params["risk_mode"] == "tp_sl_grid"
    assert result.summary_metrics["best_tp_pct"] == pytest.approx(2.0)
    assert result.summary_metrics["best_sl_pct"] == pytest.approx(1.0)
    assert result.trades
    assert result.trades[0]["exit_reason"] in {
        "signal",
        "take_profit",
        "stop_loss",
        "close_on_end",
    }


def test_lazy_trades_tp_only_payload_uses_disabled_sl_sentinel(tmp_path: Path) -> None:
    service, _cache, job, row = _service_fixture(
        tmp_path=tmp_path,
        risk_mode="tp_sl_grid_tp_only",
    )

    assert row.best_sl_pct is None

    result = service.execute(
        job=job,
        row=row,
        public_variant_key=row.payload_json["public_variant_key"],
    )

    assert result.readable_params["best_sl_pct"] is None
    assert result.summary_metrics["best_tp_pct"] == pytest.approx(2.0)
    assert result.summary_metrics["best_sl_pct"] is None
    assert result.cache["ttl_seconds"] == DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS
    assert result.trades


def test_tp_sl_detail_row_can_report_take_profit_exit_reason() -> None:
    hit_times = BacktestTpSlHitTimesSubset(
        tp_values=np.asarray([0.02], dtype=np.float32),
        sl_values=np.asarray([0.01], dtype=np.float32),
        long_tp=np.asarray([[4, 4, 2, 4]], dtype=np.uint32),
        long_sl=np.asarray([[4, 4, 4, 4]], dtype=np.uint32),
        short_tp=np.asarray([[4, 4, 4, 4]], dtype=np.uint32),
        short_sl=np.asarray([[4, 4, 4, 4]], dtype=np.uint32),
        sentinel_index=4,
    )
    runtime = _Runtime()

    rows = detail_module._tp_sl_trade_rows(
        entry_abs=np.asarray([1], dtype=np.int32),
        dir_arr=np.asarray([1], dtype=np.int8),
        sig_exit_abs=np.asarray([3], dtype=np.int32),
        best_tp_idx=0,
        best_sl_idx=0,
        hit_times=hit_times,
        runtime=runtime,
        execution_settings=cast(Any, runtime.execution_settings),
        open_time_15m=np.asarray([0, 900000, 1800000, 2700000], dtype=np.int64),
        close_time_15m=np.asarray([899999, 1799999, 2699999, 3599999], dtype=np.int64),
    )

    assert rows[0]["exit_reason"] == "take_profit"
    assert rows[0]["exit_bar_index"] == 2


def test_tp_sl_detail_row_uses_same_bar_stop_loss_precedence() -> None:
    hit_times = BacktestTpSlHitTimesSubset(
        tp_values=np.asarray([0.02], dtype=np.float32),
        sl_values=np.asarray([0.01], dtype=np.float32),
        long_tp=np.asarray([[4, 4, 2, 4]], dtype=np.uint32),
        long_sl=np.asarray([[4, 4, 2, 4]], dtype=np.uint32),
        short_tp=np.asarray([[4, 4, 4, 4]], dtype=np.uint32),
        short_sl=np.asarray([[4, 4, 4, 4]], dtype=np.uint32),
        sentinel_index=4,
    )
    runtime = _Runtime()

    rows = detail_module._tp_sl_trade_rows(
        entry_abs=np.asarray([1], dtype=np.int32),
        dir_arr=np.asarray([1], dtype=np.int8),
        sig_exit_abs=np.asarray([3], dtype=np.int32),
        best_tp_idx=0,
        best_sl_idx=0,
        hit_times=hit_times,
        runtime=runtime,
        execution_settings=cast(Any, runtime.execution_settings),
        open_time_15m=np.asarray([0, 900000, 1800000, 2700000], dtype=np.int64),
        close_time_15m=np.asarray([899999, 1799999, 2699999, 3599999], dtype=np.int64),
    )

    assert rows[0]["exit_reason"] == "stop_loss"
    assert rows[0]["exit_bar_index"] == 2


def test_funding_events_overlay_uses_entry_exclusive_exit_inclusive() -> None:
    funding_arrays = SimpleNamespace(
        funding_time=np.asarray([1_000, 2_000, 3_000, 4_000], dtype=np.int64),
        funding_rate=np.asarray([0.1, 0.2, -0.1, 0.3], dtype=np.float64),
        mark_price=np.asarray([100.0, 101.0, 102.0, 103.0], dtype=np.float64),
        funding_interval_minutes=np.asarray([480, 480, 480, 480], dtype=np.uint16),
        data_quality=np.asarray([1, 1, 1, 1], dtype=np.uint8),
    )

    events = detail_module._funding_events_overlay(
        trades=[
            {
                "trade_index": 7,
                "entry_timestamp": "1970-01-01T00:00:01Z",
                "exit_timestamp": "1970-01-01T00:00:03Z",
                "side": "long",
                "quantity": 2.0,
            }
        ],
        funding_arrays=funding_arrays,
    )

    assert [event["funding_rate"] for event in events] == [0.2, -0.1]
    assert events[0]["kind"] == "funding_event"
    assert events[0]["trade_index"] == 7
    assert events[0]["estimated_pnl_quote"] == pytest.approx(-40.4)
    assert events[1]["timestamp"] == "1970-01-01T00:00:03Z"


@dataclass
class _MemoryCache:
    read_result: BacktestLazyTradesCacheReadResult = BacktestLazyTradesCacheReadResult(
        status="miss"
    )
    writes: tuple[tuple[BacktestLazyTradesCacheKey, Mapping[str, Any]], ...] = ()
    raise_on_write: Exception | None = None

    def read(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        now: datetime,
        ttl_seconds: int,
    ) -> BacktestLazyTradesCacheReadResult:
        _ = cache_key, now, ttl_seconds
        return self.read_result

    def write(
        self,
        *,
        cache_key: BacktestLazyTradesCacheKey,
        payload: Mapping[str, Any],
        now: datetime,
        ttl_seconds: int,
    ) -> None:
        _ = now, ttl_seconds
        if self.raise_on_write is not None:
            raise self.raise_on_write
        self.writes = (*self.writes, (cache_key, payload))

    def read_page(self, **kwargs: Any) -> BacktestLazyTradesCacheReadResult:
        _ = kwargs
        return self.read_result

    def read_series(self, **kwargs: Any) -> BacktestLazyTradesCacheReadResult:
        _ = kwargs
        return self.read_result

    def read_monthly_stats(self, **kwargs: Any) -> BacktestLazyTradesCacheReadResult:
        _ = kwargs
        return self.read_result

    def read_symbol_stats(self, **kwargs: Any) -> BacktestLazyTradesCacheReadResult:
        _ = kwargs
        return self.read_result

    def read_csv(self, **kwargs: Any) -> BacktestLazyTradesCacheReadResult:
        _ = kwargs
        return self.read_result


class _FailingPreparePools:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"prepare_pools should not be called on cache hit: {name}")


class _FailingHitTimes:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"hit_times should not be called on cache hit: {name}")


class _Runtime:
    def __init__(self) -> None:
        self.price_open_15m = np.asarray([100.0, 100.0, 102.0, 102.0], dtype=np.float32)
        self.last_close_15m = 102.0
        self.log_fac_tp_long = np.asarray([math.log(1.02 * (1.0 - 0.001) ** 2)])
        self.log_fac_sl_long = np.asarray([math.log(0.99 * (1.0 - 0.001) ** 2)])
        self.log_fac_tp_short = self.log_fac_tp_long
        self.log_fac_sl_short = self.log_fac_sl_long
        self.log_fee_two_sides = math.log((1.0 - 0.001) ** 2)
        self.close_on_end = np.int8(1)
        self.t_exec_abs_15m = np.int32(4)
        self.initial_cash_quote = 10000.0
        self.sizing_mode_code = np.int8(1)
        self.quote_amount = 100.0
        self.equity_pct = 100.0
        self.min_quote = 100.0
        self.max_quote = 100.0
        self.safe_profit_percent = 30.0
        self.use_profit_lock = np.int8(0)
        self.execution_settings = _ExecutionSettings()


class _ExecutionSettings:
    direction_mode = "long_short_reversal"
    sizing_mode_code = np.int8(1)
    quote_amount = 100.0
    equity_pct = 100.0
    min_quote = 100.0
    max_quote = 100.0
    fee_rate = 0.001
    safe_profit_percent = 30.0
    use_profit_lock = np.int8(0)


def _service_fixture(
    *,
    tmp_path: Path,
    risk_mode: str,
    top_row_id: int = 1,
    row_prefilter_top_fraction: float | None = None,
) -> tuple[BacktestLazyTradesDetailService, _MemoryCache, BacktestJob, BacktestJobTopVariant]:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    prepare_pools = BacktestPreparePoolsService(
        artifact_array_loader=loader,
        defaults_provider=YamlBacktestGridDefaultsProvider.from_yaml(
            config_path="configs/prod/indicators.yaml",
        ),
        config=BacktestPreparePoolsConfig(row_prefilter_top_fraction=row_prefilter_top_fraction)
        if row_prefilter_top_fraction is not None
        else BacktestPreparePoolsConfig(),
    )
    cache = _MemoryCache()
    service = BacktestLazyTradesDetailService(
        prepare_pools=prepare_pools,
        tp_sl_hit_times=BacktestTpSlHitTimesService(artifact_array_loader=loader),
        cache=cache,
    )
    metadata = _artifact_metadata(store=store)
    request = _normalized_request(risk_mode=risk_mode)
    request["artifact_metadata"] = metadata.as_mapping()
    job_id = uuid4()
    job = BacktestJob.create_queued(
        job_id=job_id,
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000401"),
        mode="template",
        created_at=datetime.now(UTC),
        request_json=request,
        request_hash="d" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="e" * 64,
        backtest_runtime_config_hash="e" * 64,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot="slot_a",
            artifact_slot_generation=metadata.artifact_slot_generation,
            artifact_manifest_hash=metadata.artifact_manifest_hash,
            artifact_asof_date=metadata.artifact_asof_date,
        ),
        execution_mode="sync_inline",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="15m",
        requested_top_n=100,
        ranking_primary_metric="total_return_pct",
    )
    row = (
        BacktestTopResultAssemblyService()
        .assemble(
            job_id=job_id,
            normalized_request=request,
            top_results=(_top_result(risk_mode=risk_mode, row_id=top_row_id),),
            updated_at=datetime.now(UTC),
        )
        .top_variants[0]
    )
    return service, cache, job, row


def _top_result(
    *,
    risk_mode: str,
    row_id: int = 1,
) -> BacktestNoRiskTopResult | BacktestTpSlTopResult:
    metadata = {"ma.ema.source": "close", "ma.ema.window": 5 + row_id}
    if risk_mode in {"tp_sl_grid", "tp_sl_grid_tp_only"}:
        return BacktestTpSlTopResult(
            rank=1,
            score=1.0,
            indicator_rows={"ma.ema": row_id},
            best_tp_idx=0,
            best_sl_idx=0,
            metrics={
                "total_return_pct": 1.0,
                "trade_count": 1.0,
                "best_tp_pct": 2.0,
                "best_sl_pct": 1.0,
            },
            metadata=metadata,
        )
    return BacktestNoRiskTopResult(
        rank=1,
        score=1.0,
        indicator_rows={"ma.ema": row_id},
        metrics={"total_return_pct": 1.0, "trade_count": 1.0},
        metadata=metadata,
    )


def _artifact_metadata(*, store: Any) -> BacktestArtifactMetadata:
    resolver = FilesystemBacktestArtifactContextResolver(artifact_loader=store.loader)
    return resolver.resolve_context(
        coordinates=BacktestCoordinates(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        )
    )


def _normalized_request(*, risk_mode: str) -> dict[str, Any]:
    risk = {"mode": "none"}
    if risk_mode == "tp_sl_grid":
        risk = {
            "mode": "tp_sl_grid",
            "tp": {"start_pct": 2.0, "stop_pct": 2.0, "step_pct": 1.0},
            "sl": {"start_pct": 1.0, "stop_pct": 1.0, "step_pct": 1.0},
        }
    elif risk_mode == "tp_sl_grid_tp_only":
        risk = {
            "mode": "tp_sl_grid",
            "tp": {"start_pct": 2.0, "stop_pct": 2.0, "step_pct": 1.0},
            "sl": {"enabled": False},
        }
    return {
        "coordinates": {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "BTCUSDT",
        },
        "timeframe": "15m",
        "time_range": {
            "start": "1970-01-01T00:00:01Z",
            "end": "1970-01-01T00:00:04Z",
        },
        "indicators": [
            {
                "indicator_id": "ma.ema",
                "sources": ["close"],
                "window": {"start": 5, "stop": 6, "step": 1},
            }
        ],
        "risk": risk,
        "execution": {
            "direction_mode": "long_short_reversal",
            "fee_rate": 0.00075,
            "slippage_rate": 0.0001,
            "initial_cash_quote": 10000.0,
            "sizing": {"mode": "fixed_equity_pct", "equity_pct": 10.0},
            "profit_lock": {"enabled": False},
            "close_on_end": True,
        },
        "ranking": {"primary_metric": "total_return_pct", "direction": "desc"},
        "top_n": 100,
    }


def _cached_payload(*, job: BacktestJob, row: BacktestJobTopVariant) -> dict[str, Any]:
    return {
        "job_id": str(job.job_id),
        "variant_key": row.payload_json["public_variant_key"],
        "variant_hash": row.variant_key,
        "request_hash": job.request_hash,
        "engine_params_hash": job.engine_params_hash,
        "artifact_manifest_hash": job.artifact_pin.artifact_manifest_hash
        if job.artifact_pin is not None
        else "a" * 64,
        "summary_metrics": dict(row.summary_metrics_json),
        "canonical_variant_params": dict(row.payload_json["canonical_variant_params"]),
        "readable_params": dict(row.payload_json["readable_params"]),
        "trades": [
            {
                "trade_index": 0,
                "entry_timestamp": "1970-01-01T00:00:00Z",
                "exit_timestamp": "1970-01-01T00:15:00Z",
                "entry_bar_index": 1,
                "exit_bar_index": 2,
                "side": "long",
                "direction": "long",
                "entry_price": 100.0,
                "exit_price": 101.0,
                "quantity": 1.0,
                "notional_quote": 100.0,
                "return_pct": 1.0,
                "net_pnl_quote": 1.0,
                "fee_quote": 0.0,
                "slippage_quote": 0.0,
                "exit_reason": "signal",
            }
        ],
        "chart_overlay": {"schema": "backtest_chart_overlay_v1", "markers": [], "segments": []},
        "cache": {"status": "miss"},
        "timing": {},
    }


def _job_with_funding_manifest_hash(
    *,
    job: BacktestJob,
    funding_manifest_hash: str,
) -> BacktestJob:
    request = dict(job.request_json)
    artifact_metadata = dict(request["artifact_metadata"])
    artifact_metadata["funding_manifest_hash"] = funding_manifest_hash
    request["artifact_metadata"] = artifact_metadata
    return replace(job, request_json=request)
