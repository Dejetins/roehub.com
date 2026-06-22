from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

import numpy as np

import trading.contexts.backtest.application.services.v2.no_risk_exact as no_risk_module
import trading.contexts.backtest.application.services.v2.tp_sl_exact as tp_sl_module
from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestLazyTradesDetailReadModel,
    BacktestNoRiskExactConfig,
    BacktestPreparePoolsResult,
    BacktestTpSlExactConfig,
    build_backtest_funding_read_model,
)
from trading.contexts.backtest.application.ports.lazy_trades_cache import (
    BacktestLazyTradesCache,
    BacktestLazyTradesCacheKey,
    build_lazy_trades_cache_key,
    normalize_json_payload,
)
from trading.contexts.backtest.application.services.v2.execution_sizing import (
    ExecutionSettings,
    execution_quote_amount_py,
    execution_settings_from_normalized,
)
from trading.contexts.backtest.application.services.v2.prepare_pools import (
    BacktestPreparePoolsService,
)
from trading.contexts.backtest.application.services.v2.tp_sl_funding import (
    resolve_tp_sl_selected_exit,
)
from trading.contexts.backtest.application.services.v2.tp_sl_hit_times import (
    BacktestTpSlHitTimesService,
)
from trading.contexts.backtest.domain.entities import BacktestJob, BacktestJobTopVariant
from trading.platform.errors import RoehubError

LAZY_TRADES_COMPUTE_STAGE_NAME = "lazy_trades_compute"
LAZY_TRADES_CACHE_HIT_STAGE_NAME = "lazy_trades_cache_hit"
DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS = 14 * 24 * 60 * 60
BACKTEST_ERROR_VARIANT_CONFLICT = "backtest.variant_conflict"
MAX_FUNDING_OVERLAY_EVENTS = 1_000
_FUNDING_SUMMARY_KEYS = frozenset(
    {
        "total_return_pct_net_of_funding",
        "funding_return_pct",
        "funding_pnl_quote",
        "funding_events_count",
    }
)


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesDetailConfig:
    cache_ttl_seconds: int = DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS

    def __post_init__(self) -> None:
        if self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be > 0")


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesCacheProbeResult:
    detail: BacktestLazyTradesDetailReadModel | None
    cache_key: BacktestLazyTradesCacheKey
    cache_status: str
    cache_warning: str | None
    ttl_seconds: int
    cache_lookup_s: float

    @property
    def is_hit(self) -> bool:
        return self.detail is not None


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesDetailService:
    prepare_pools: BacktestPreparePoolsService
    tp_sl_hit_times: BacktestTpSlHitTimesService
    cache: BacktestLazyTradesCache
    config: BacktestLazyTradesDetailConfig = BacktestLazyTradesDetailConfig()

    def read_cached(
        self,
        *,
        job: BacktestJob,
        row: BacktestJobTopVariant,
        public_variant_key: str,
        now: datetime | None = None,
    ) -> BacktestLazyTradesCacheProbeResult:
        checked = _checked_variant_identity(row=row, public_variant_key=public_variant_key)
        reference_now = (now or datetime.now(UTC)).astimezone(UTC)
        artifact_metadata = _artifact_metadata_from_job(job=job)
        cache_key = build_lazy_trades_cache_key(
            job_id=str(job.job_id),
            variant_key=checked.public_variant_key,
            variant_hash=checked.variant_hash,
            request_hash=job.request_hash,
            engine_params_hash=_engine_params_hash(job=job),
            artifact_manifest_hash=artifact_metadata.artifact_manifest_hash,
            funding_manifest_hash=artifact_metadata.funding_manifest_hash,
        )

        lookup_start = time.perf_counter()
        cache_read = self.cache.read(
            cache_key=cache_key,
            now=reference_now,
            ttl_seconds=self.config.cache_ttl_seconds,
        )
        cache_lookup_s = time.perf_counter() - lookup_start
        if cache_read.is_hit and cache_read.payload is not None:
            payload = dict(cache_read.payload)
            timing = dict(_mapping(payload.get("timing")))
            timing["cache_lookup_s"] = cache_lookup_s
            timing[LAZY_TRADES_CACHE_HIT_STAGE_NAME] = cache_lookup_s
            payload["timing"] = timing
            payload["cache"] = _cache_payload(
                status="hit",
                cache_key=cache_key,
                ttl_seconds=self.config.cache_ttl_seconds,
                warning=cache_read.warning,
            )
            return BacktestLazyTradesCacheProbeResult(
                detail=_read_model_from_payload(payload=payload),
                cache_key=cache_key,
                cache_status=cache_read.status,
                cache_warning=cache_read.warning,
                ttl_seconds=self.config.cache_ttl_seconds,
                cache_lookup_s=cache_lookup_s,
            )

        return BacktestLazyTradesCacheProbeResult(
            detail=None,
            cache_key=cache_key,
            cache_status=cache_read.status,
            cache_warning=cache_read.warning,
            ttl_seconds=self.config.cache_ttl_seconds,
            cache_lookup_s=cache_lookup_s,
        )

    def execute(
        self,
        *,
        job: BacktestJob,
        row: BacktestJobTopVariant,
        public_variant_key: str,
        now: datetime | None = None,
    ) -> BacktestLazyTradesDetailReadModel:
        checked = _checked_variant_identity(row=row, public_variant_key=public_variant_key)
        reference_now = (now or datetime.now(UTC)).astimezone(UTC)
        artifact_metadata = _artifact_metadata_from_job(job=job)
        cache_key = build_lazy_trades_cache_key(
            job_id=str(job.job_id),
            variant_key=checked.public_variant_key,
            variant_hash=checked.variant_hash,
            request_hash=job.request_hash,
            engine_params_hash=_engine_params_hash(job=job),
            artifact_manifest_hash=artifact_metadata.artifact_manifest_hash,
            funding_manifest_hash=artifact_metadata.funding_manifest_hash,
        )

        lookup_start = time.perf_counter()
        cache_read = self.cache.read(
            cache_key=cache_key,
            now=reference_now,
            ttl_seconds=self.config.cache_ttl_seconds,
        )
        cache_lookup_s = time.perf_counter() - lookup_start
        if cache_read.is_hit and cache_read.payload is not None:
            payload = dict(cache_read.payload)
            timing = dict(_mapping(payload.get("timing")))
            timing["cache_lookup_s"] = cache_lookup_s
            timing[LAZY_TRADES_CACHE_HIT_STAGE_NAME] = cache_lookup_s
            payload["timing"] = timing
            payload["cache"] = _cache_payload(
                status="hit",
                cache_key=cache_key,
                ttl_seconds=self.config.cache_ttl_seconds,
                warning=cache_read.warning,
            )
            return _read_model_from_payload(payload=payload)

        compute_start = time.perf_counter()
        payload = self._recompute_payload(
            job=job,
            row=row,
            checked=checked,
            artifact_metadata=artifact_metadata,
            cache_key=cache_key,
        )
        compute_s = time.perf_counter() - compute_start
        timing = dict(_mapping(payload.get("timing")))
        timing["cache_lookup_s"] = cache_lookup_s
        timing[LAZY_TRADES_COMPUTE_STAGE_NAME] = compute_s
        payload["timing"] = timing
        cache_status = cache_read.status
        cache_warning = cache_read.warning
        try:
            self.cache.write(
                cache_key=cache_key,
                payload=payload,
                now=reference_now,
                ttl_seconds=self.config.cache_ttl_seconds,
            )
        except Exception as error:  # noqa: BLE001
            cache_status = "write_failed"
            cache_warning = str(error)
        payload["cache"] = _cache_payload(
            status=cache_status,
            cache_key=cache_key,
            ttl_seconds=self.config.cache_ttl_seconds,
            warning=cache_warning,
        )
        return _read_model_from_payload(payload=payload)

    def _recompute_payload(
        self,
        *,
        job: BacktestJob,
        row: BacktestJobTopVariant,
        checked: "_CheckedVariantIdentity",
        artifact_metadata: BacktestArtifactMetadata,
        cache_key: BacktestLazyTradesCacheKey,
    ) -> dict[str, Any]:
        normalized_request = dict(job.request_json)
        coordinates = _coordinates_from_request(normalized_request)
        required_row_ids_by_indicator = _row_ids_by_indicator_from_top_variant(row=row)
        try:
            context = self.prepare_pools.resolve_artifact_context(
                coordinates=coordinates,
                artifact_metadata=artifact_metadata,
            )
            runtime_arrays = self.prepare_pools.open_artifact_arrays(
                normalized_request=normalized_request,
                context=context,
            )
            request_slice = self.prepare_pools.prepare_request_slice(
                normalized_request=normalized_request,
                runtime_arrays=runtime_arrays,
            )
            prepared = self.prepare_pools.prepare_pools_core(
                normalized_request=normalized_request,
                runtime_arrays=runtime_arrays,
                request_slice=request_slice,
                required_row_ids_by_indicator=required_row_ids_by_indicator,
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise RoehubError(
                code="backtest.artifacts_unavailable",
                message="Backtest artifacts are unavailable for lazy trades recompute",
                details={"job_id": str(job.job_id), "reason": str(error), "retryable": True},
            ) from error

        local_indices = _local_indices_from_row(row=row, prepared=prepared)
        risk_mode = _risk_mode(normalized_request)
        if risk_mode == "tp_sl_grid":
            summary_metrics, trades, detail_metadata = self._tp_sl_detail(
                normalized_request=normalized_request,
                prepared=prepared,
                local_indices=local_indices,
                runtime_arrays=runtime_arrays,
                row=row,
                context=context,
            )
        elif risk_mode == "none":
            summary_metrics, trades, detail_metadata = self._no_risk_detail(
                normalized_request=normalized_request,
                prepared=prepared,
                local_indices=local_indices,
                runtime_arrays=runtime_arrays,
            )
        else:
            raise RoehubError(
                code="backtest.invalid_request",
                message="Unsupported risk mode for lazy trades detail",
                details={"risk_mode": risk_mode},
            )
        summary_metrics = _merge_top_funding_summary(
            recomputed_summary=summary_metrics,
            row_summary=row.summary_metrics_json,
        )
        chart_overlay = _chart_overlay(trades=trades)
        funding_events, overlay_status, overlay_warning_code = self._funding_overlay(
            normalized_request=normalized_request,
            context=context,
            artifact_metadata=artifact_metadata,
            trades=trades,
        )
        if funding_events:
            chart_overlay["funding_events"] = funding_events
            chart_overlay["funding_events_count"] = len(funding_events)
            chart_overlay["funding_events_truncated"] = (
                len(funding_events) >= MAX_FUNDING_OVERLAY_EVENTS
            )
        if artifact_metadata.funding_manifest_hash is not None:
            chart_overlay["funding_manifest_hash"] = artifact_metadata.funding_manifest_hash
        funding = build_backtest_funding_read_model(
            summary_metrics=summary_metrics,
            payload=row.payload_json,
            artifact_metadata=artifact_metadata.as_mapping(),
            funding_events_count=len(funding_events) if funding_events else None,
            overlay_status=overlay_status,
            overlay_warning_code=overlay_warning_code,
        )
        payload = {
            "job_id": str(job.job_id),
            "variant_key": checked.public_variant_key,
            "variant_hash": checked.variant_hash,
            "request_hash": job.request_hash,
            "engine_params_hash": _engine_params_hash(job=job),
            "artifact_manifest_hash": artifact_metadata.artifact_manifest_hash,
            "funding_manifest_hash": artifact_metadata.funding_manifest_hash,
            "summary_metrics": summary_metrics,
            "canonical_variant_params": checked.canonical_variant_params,
            "readable_params": checked.readable_params,
            "trades": trades,
            "chart_overlay": chart_overlay,
            "funding": funding,
            "cache": _cache_payload(
                status="miss",
                cache_key=cache_key,
                ttl_seconds=self.config.cache_ttl_seconds,
            ),
            "timing": {},
            "detail_metadata": detail_metadata,
        }
        return normalize_json_payload(payload)

    def _funding_overlay(
        self,
        *,
        normalized_request: Mapping[str, Any],
        context: Any,
        artifact_metadata: BacktestArtifactMetadata,
        trades: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], str, str | None]:
        if not _should_load_funding_overlay(
            normalized_request=normalized_request,
            artifact_metadata=artifact_metadata,
        ):
            return [], "not_applicable", None
        try:
            funding_arrays = self.prepare_pools.artifact_array_loader.load_funding_arrays(
                context=context,
            )
        except Exception:  # noqa: BLE001
            return [], "unavailable", "funding_events_unavailable"
        events = _funding_events_overlay(
            trades=trades,
            funding_arrays=funding_arrays,
        )
        return events, "ready", None

    def _no_risk_detail(
        self,
        *,
        normalized_request: Mapping[str, Any],
        prepared: BacktestPreparePoolsResult,
        local_indices: tuple[int, ...],
        runtime_arrays: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        execution_settings = execution_settings_from_normalized(
            normalized_request,
            expected_direction_mode=_direction_mode(normalized_request),
            config=BacktestNoRiskExactConfig(),
        )
        open_1m = np.asarray(prepared.execution_open_1m, dtype=np.float32)
        close_1m = np.asarray(prepared.execution_close_1m, dtype=np.float32)
        summary = no_risk_module.evaluate_no_risk_reference_rows_slow(
            prepared_result=prepared,
            local_indices=local_indices,
            execution_settings=execution_settings,
            execution_open_1m=open_1m,
            execution_close_1m=close_1m,
        )
        entry_arr, dir_arr, exit_arr = no_risk_module.build_trade_list_for_indicator_rows_slow(
            prepared_result=prepared,
            local_indices=local_indices,
            direction_mode=execution_settings.direction_mode,
        )
        trades = _no_risk_trade_rows(
            entry_arr=entry_arr,
            dir_arr=dir_arr,
            sig_exit_arr=exit_arr,
            execution_open_1m=open_1m,
            execution_close_1m=close_1m,
            open_time_1m=np.asarray(runtime_arrays.price_arrays_1m.open_time),
            close_time_1m=np.asarray(runtime_arrays.price_arrays_1m.close_time),
            execution_settings=execution_settings,
            t_exec=int(prepared.execution_mapping.t_exec_limit_1m),
        )
        return (
            dict(summary),
            trades,
            {
                "risk_mode": "none",
                "timeframe": prepared.timeframe,
                "execution_timeframe": "1m",
            },
        )

    def _tp_sl_detail(
        self,
        *,
        normalized_request: Mapping[str, Any],
        prepared: BacktestPreparePoolsResult,
        local_indices: tuple[int, ...],
        runtime_arrays: Any,
        row: BacktestJobTopVariant,
        context: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        hit_times_result = self.tp_sl_hit_times.execute(
            normalized_request=normalized_request,
            context=context,
        )
        execution_settings = execution_settings_from_normalized(
            normalized_request,
            expected_direction_mode=_direction_mode(normalized_request),
            config=BacktestTpSlExactConfig(),
        )
        runtime = tp_sl_module._tp_sl_runtime_context_from_prepared(
            prepared_result=prepared,
            hit_times=hit_times_result.hit_times,
            execution_settings=execution_settings,
        )
        best_tp_idx = _level_index_for_requested_axis(
            values=hit_times_result.hit_times.tp_values,
            pct_value=row.best_tp_pct,
            axis="best_tp_pct",
            enabled=_risk_axis_enabled(normalized_request, "tp"),
        )
        best_sl_idx = _level_index_for_requested_axis(
            values=hit_times_result.hit_times.sl_values,
            pct_value=row.best_sl_pct,
            axis="best_sl_pct",
            enabled=_risk_axis_enabled(normalized_request, "sl"),
        )
        entry_abs, dir_arr, sig_exit_abs = (
            tp_sl_module.build_trade_list_15m_for_indicator_rows_slow(
                prepared_result=prepared,
                local_indices=local_indices,
                direction_mode=execution_settings.direction_mode,
            )
        )
        trade_returns, bars_held = tp_sl_module._selected_cell_trade_returns(
            entry_abs=entry_abs,
            dir_arr=dir_arr,
            sig_exit_abs=sig_exit_abs,
            best_tp_idx=best_tp_idx,
            best_sl_idx=best_sl_idx,
            hit_times=hit_times_result.hit_times,
            runtime=runtime,
        )
        summary: dict[str, Any] = dict(
            tp_sl_module._summary_metrics_from_trade_returns(
                trade_returns=trade_returns,
                bars_held=bars_held,
                t_exec_abs=int(runtime.t_exec_abs_15m),
                runtime=runtime,
            )
        )
        summary["best_tp_pct"] = float(row.best_tp_pct) if row.best_tp_pct is not None else None
        summary["best_sl_pct"] = float(row.best_sl_pct) if row.best_sl_pct is not None else None
        trades = _tp_sl_trade_rows(
            entry_abs=entry_abs,
            dir_arr=dir_arr,
            sig_exit_abs=sig_exit_abs,
            best_tp_idx=best_tp_idx,
            best_sl_idx=best_sl_idx,
            hit_times=hit_times_result.hit_times,
            runtime=runtime,
            execution_settings=execution_settings,
            open_time_15m=np.asarray(runtime_arrays.price_arrays_15m.open_time),
            close_time_15m=np.asarray(runtime_arrays.price_arrays_15m.close_time),
        )
        return (
            summary,
            trades,
            {
                "risk_mode": "tp_sl_grid",
                "timeframe": prepared.timeframe,
                "hit_times_manifest_hash": hit_times_result.hit_times_manifest_hash,
                "hit_times_path": "hit_times/15m",
                "best_tp_idx": best_tp_idx,
                "best_sl_idx": best_sl_idx,
            },
        )


@dataclass(frozen=True, slots=True)
class _CheckedVariantIdentity:
    public_variant_key: str
    variant_hash: str
    canonical_variant_params: dict[str, Any]
    readable_params: dict[str, Any]


def _checked_variant_identity(
    *,
    row: BacktestJobTopVariant,
    public_variant_key: str,
) -> _CheckedVariantIdentity:
    payload = dict(row.payload_json)
    payload_public_key = payload.get("public_variant_key")
    payload_variant_hash = payload.get("variant_hash")
    if not isinstance(payload_public_key, str) or payload_public_key != public_variant_key:
        raise _variant_conflict(
            message="Top variant public key contradicts requested route key",
            details={
                "route_variant_key": public_variant_key,
                "payload_public_variant_key": payload_public_key,
            },
        )
    if not isinstance(payload_variant_hash, str) or payload_variant_hash != row.variant_key:
        raise _variant_conflict(
            message="Top variant storage key contradicts payload variant hash",
            details={
                "storage_variant_key": row.variant_key,
                "payload_variant_hash": payload_variant_hash,
            },
        )
    canonical = payload.get("canonical_variant_params")
    readable = payload.get("readable_params")
    if not isinstance(canonical, Mapping) or not isinstance(readable, Mapping):
        raise _variant_conflict(
            message="Top variant payload is missing canonical/readable params",
            details={"variant_key": public_variant_key, "variant_hash": payload_variant_hash},
        )
    return _CheckedVariantIdentity(
        public_variant_key=payload_public_key,
        variant_hash=payload_variant_hash,
        canonical_variant_params=dict(canonical),
        readable_params=dict(readable),
    )


def _variant_conflict(*, message: str, details: Mapping[str, Any]) -> RoehubError:
    return RoehubError(
        code=BACKTEST_ERROR_VARIANT_CONFLICT,
        message=message,
        details=dict(details),
    )


def _artifact_metadata_from_job(*, job: BacktestJob) -> BacktestArtifactMetadata:
    request = dict(job.request_json)
    raw_artifact = request.get("artifact_metadata")
    artifact = dict(raw_artifact) if isinstance(raw_artifact, Mapping) else {}
    if job.artifact_pin is not None:
        artifact.setdefault("artifact_slot", job.artifact_pin.artifact_slot)
        artifact.setdefault("artifact_slot_generation", job.artifact_pin.artifact_slot_generation)
        artifact.setdefault("artifact_manifest_hash", job.artifact_pin.artifact_manifest_hash)
        artifact.setdefault("artifact_asof_date", job.artifact_pin.artifact_asof_date)
    required = (
        "artifact_slot",
        "artifact_slot_generation",
        "artifact_manifest_hash",
        "artifact_asof_date",
    )
    missing = [key for key in required if key not in artifact]
    if missing:
        raise RoehubError(
            code="backtest.invalid_request",
            message="Backtest job is missing artifact metadata for lazy trades recompute",
            details={"job_id": str(job.job_id), "missing": missing},
        )
    return BacktestArtifactMetadata(
        artifact_slot=str(artifact["artifact_slot"]),
        artifact_slot_generation=int(artifact["artifact_slot_generation"]),
        artifact_manifest_hash=str(artifact["artifact_manifest_hash"]),
        artifact_asof_date=str(artifact["artifact_asof_date"]),
        hit_times_manifest_hash=None
        if artifact.get("hit_times_manifest_hash") is None
        else str(artifact["hit_times_manifest_hash"]),
        published_at_utc=str(artifact.get("published_at_utc", "")),
        funding_manifest_hash=None
        if artifact.get("funding_manifest_hash") is None
        else str(artifact["funding_manifest_hash"]),
        funding_coverage_status=None
        if artifact.get("funding_coverage_status") is None
        else str(artifact["funding_coverage_status"]),
        funding_coverage_policy=None
        if artifact.get("funding_coverage_policy") is None
        else str(artifact["funding_coverage_policy"]),
        funding_rows_count=_optional_int(artifact.get("funding_rows_count")),
        funding_expected_event_count=_optional_int(artifact.get("funding_expected_event_count")),
        funding_missing_event_count=_optional_int(artifact.get("funding_missing_event_count")),
        funding_reason_codes=_string_tuple(artifact.get("funding_reason_codes")),
    )


def _coordinates_from_request(request: Mapping[str, Any]) -> BacktestCoordinates:
    coordinates = _mapping(request.get("coordinates"))
    return BacktestCoordinates(
        exchange=str(coordinates.get("exchange", "")),
        market_type=str(coordinates.get("market_type", "")),
        symbol=str(coordinates.get("symbol", "")),
    )


def _engine_params_hash(*, job: BacktestJob) -> str:
    return job.engine_params_hash or job.backtest_runtime_config_hash


def _local_indices_from_row(
    *,
    row: BacktestJobTopVariant,
    prepared: BacktestPreparePoolsResult,
) -> tuple[int, ...]:
    row_id_by_indicator = _row_ids_by_indicator_from_top_variant(row=row)
    local_indices: list[int] = []
    pools_by_id = {pool.indicator_id: pool for pool in prepared.indicator_pools}
    for indicator_id in prepared.indicator_ids:
        if indicator_id not in row_id_by_indicator:
            raise _variant_conflict(
                message="Top variant canonical params do not cover prepared indicator",
                details={"indicator_id": indicator_id, "variant_hash": row.variant_key},
            )
        row_ids = row_id_by_indicator[indicator_id]
        if len(row_ids) != 1:
            raise _variant_conflict(
                message="Top variant canonical params have invalid row id payload",
                details={"indicator_id": indicator_id, "variant_hash": row.variant_key},
            )
        row_id = row_ids[0]
        pool = pools_by_id[indicator_id]
        matches = np.flatnonzero(np.asarray(pool.row_ids, dtype=np.int64) == row_id)
        if int(matches.size) != 1:
            raise _variant_conflict(
                message="Top variant row id is not present in prepared pool",
                details={
                    "indicator_id": indicator_id,
                    "row_id": row_id,
                    "variant_hash": row.variant_key,
                },
            )
        local_indices.append(int(matches[0]))
    return tuple(local_indices)


def _row_ids_by_indicator_from_top_variant(
    *,
    row: BacktestJobTopVariant,
) -> dict[str, tuple[int, ...]]:
    canonical = _mapping(dict(row.payload_json).get("canonical_variant_params"))
    indicators = canonical.get("indicators")
    if not isinstance(indicators, Sequence) or isinstance(indicators, (str, bytes)):
        raise _variant_conflict(
            message="Top variant canonical params have invalid indicators payload",
            details={"variant_hash": row.variant_key},
        )
    row_ids_by_indicator: dict[str, tuple[int, ...]] = {}
    for item in indicators:
        if not isinstance(item, Mapping):
            continue
        indicator_id = str(item.get("indicator_id"))
        row_ids_by_indicator[indicator_id] = (int(item.get("row_id", -1)),)
    return row_ids_by_indicator


def _no_risk_trade_rows(
    *,
    entry_arr: np.ndarray,
    dir_arr: np.ndarray,
    sig_exit_arr: np.ndarray,
    execution_open_1m: np.ndarray,
    execution_close_1m: np.ndarray,
    open_time_1m: np.ndarray,
    close_time_1m: np.ndarray,
    execution_settings: ExecutionSettings,
    t_exec: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available_quote = execution_settings.initial_cash_quote
    safe_quote = 0.0
    equity = execution_settings.initial_cash_quote
    for trade_index in range(int(entry_arr.size)):
        entry_idx = int(entry_arr[trade_index])
        if entry_idx >= t_exec:
            continue
        signal_exit_idx = int(sig_exit_arr[trade_index])
        if signal_exit_idx < t_exec:
            exit_idx = signal_exit_idx
            exit_price_raw = float(execution_open_1m[exit_idx])
            exit_reason = "signal"
        elif execution_settings.close_on_end == 1 and t_exec > 0:
            exit_idx = t_exec - 1
            exit_price_raw = float(execution_close_1m[exit_idx])
            exit_reason = "close_on_end"
        else:
            continue
        quote_amount = execution_quote_amount_py(
            available_quote=available_quote,
            equity=equity,
            sizing_mode_code=execution_settings.sizing_mode_code,
            quote_amount=execution_settings.quote_amount,
            equity_pct=execution_settings.equity_pct,
            min_quote=execution_settings.min_quote,
            max_quote=execution_settings.max_quote,
        )
        if quote_amount <= 0.0:
            continue
        direction = int(dir_arr[trade_index])
        entry_price_raw = float(execution_open_1m[entry_idx])
        if direction == 1:
            entry_price = entry_price_raw * (1.0 + execution_settings.slippage_rate)
            exit_price = exit_price_raw * (1.0 - execution_settings.slippage_rate)
        else:
            entry_price = entry_price_raw * (1.0 - execution_settings.slippage_rate)
            exit_price = exit_price_raw * (1.0 + execution_settings.slippage_rate)
        qty_base = quote_amount / entry_price
        entry_fee = quote_amount * execution_settings.fee_rate
        available_quote -= quote_amount + entry_fee
        exit_quote_amount = qty_base * exit_price
        exit_fee = exit_quote_amount * execution_settings.fee_rate
        gross_pnl = (
            exit_quote_amount - quote_amount
            if direction == 1
            else quote_amount - exit_quote_amount
        )
        available_quote += quote_amount + gross_pnl - exit_fee
        net_pnl = gross_pnl - entry_fee - exit_fee
        if execution_settings.use_profit_lock == 1 and net_pnl > 0.0:
            locked = net_pnl * (execution_settings.safe_profit_percent / 100.0)
            available_quote -= locked
            safe_quote += locked
        equity = available_quote + safe_quote
        rows.append(
            _trade_row(
                trade_index=len(rows),
                entry_bar_index=entry_idx,
                exit_bar_index=exit_idx,
                entry_timestamp=_timestamp_for_index(open_time_1m, entry_idx),
                exit_timestamp=_timestamp_for_index(
                    close_time_1m if exit_reason == "close_on_end" else open_time_1m,
                    exit_idx,
                ),
                side=_side(direction),
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=qty_base,
                notional_quote=quote_amount,
                gross_pnl_quote=gross_pnl,
                net_pnl_quote=net_pnl,
                return_pct=(net_pnl / quote_amount) * 100.0,
                fee_quote=entry_fee + exit_fee,
                slippage_quote=abs(entry_price - entry_price_raw) * qty_base
                + abs(exit_price - exit_price_raw) * qty_base,
                exit_reason=exit_reason,
                equity_after=equity,
                safe_quote_after=safe_quote,
                timeframe="1m",
            )
        )
    return rows


def _tp_sl_trade_rows(
    *,
    entry_abs: np.ndarray,
    dir_arr: np.ndarray,
    sig_exit_abs: np.ndarray,
    best_tp_idx: int,
    best_sl_idx: int,
    hit_times: Any,
    runtime: Any,
    execution_settings: ExecutionSettings,
    open_time_15m: np.ndarray,
    close_time_15m: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available_quote = float(runtime.initial_cash_quote)
    safe_quote = 0.0
    equity = float(runtime.initial_cash_quote)
    for trade_idx in range(int(entry_abs.shape[0])):
        direction = int(dir_arr[trade_idx])
        entry_idx = int(entry_abs[trade_idx])
        signal_exit_idx = int(sig_exit_abs[trade_idx])
        log_value, closed = tp_sl_module._tp_sl_trade_log_contrib_and_closed(
            np.int8(direction),
            np.int32(entry_idx),
            np.int32(signal_exit_idx),
            np.int32(best_tp_idx),
            np.int32(best_sl_idx),
            runtime.price_open_15m,
            runtime.last_close_15m,
            hit_times.long_tp,
            hit_times.long_sl,
            hit_times.short_tp,
            hit_times.short_sl,
            runtime.log_fac_tp_long,
            runtime.log_fac_sl_long,
            runtime.log_fac_tp_short,
            runtime.log_fac_sl_short,
            runtime.log_fee_two_sides,
            runtime.close_on_end,
            runtime.t_exec_abs_15m,
        )
        if int(closed) == 0:
            continue
        selected_exit = resolve_tp_sl_selected_exit(
            direction=direction,
            entry_abs=entry_idx,
            signal_exit_abs=signal_exit_idx,
            best_tp_idx=best_tp_idx,
            best_sl_idx=best_sl_idx,
            hit_times=hit_times,
            runtime=runtime,
        )
        exit_idx = selected_exit.exit_abs
        exit_reason = selected_exit.reason
        quote_amount = execution_quote_amount_py(
            available_quote=available_quote,
            equity=equity,
            sizing_mode_code=execution_settings.sizing_mode_code,
            quote_amount=execution_settings.quote_amount,
            equity_pct=execution_settings.equity_pct,
            min_quote=execution_settings.min_quote,
            max_quote=execution_settings.max_quote,
        )
        if quote_amount <= 0.0:
            continue
        trade_return = -1.0 if float(log_value) <= -1.0e200 else math.exp(float(log_value)) - 1.0
        pnl = quote_amount * trade_return
        available_quote += pnl
        if runtime.use_profit_lock == 1 and pnl > 0.0:
            locked = pnl * (runtime.safe_profit_percent / 100.0)
            available_quote -= locked
            safe_quote += locked
        equity = available_quote + safe_quote
        entry_price = float(runtime.price_open_15m[entry_idx])
        exit_price = _tp_sl_exit_price(
            direction=direction,
            reason=exit_reason,
            entry_price=entry_price,
            exit_idx=exit_idx,
            best_tp_pct=float(hit_times.tp_values[best_tp_idx] * np.float32(100.0)),
            best_sl_pct=float(hit_times.sl_values[best_sl_idx] * np.float32(100.0)),
            runtime=runtime,
        )
        rows.append(
            _trade_row(
                trade_index=len(rows),
                entry_bar_index=entry_idx,
                exit_bar_index=exit_idx,
                entry_timestamp=_timestamp_for_index(open_time_15m, entry_idx),
                exit_timestamp=_timestamp_for_index(
                    close_time_15m if exit_reason == "close_on_end" else open_time_15m,
                    exit_idx,
                ),
                side=_side(direction),
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quote_amount / entry_price if entry_price > 0.0 else None,
                notional_quote=quote_amount,
                gross_pnl_quote=pnl,
                net_pnl_quote=pnl,
                return_pct=trade_return * 100.0,
                fee_quote=quote_amount
                * (1.0 - ((1.0 - execution_settings.fee_rate) ** 2)),
                slippage_quote=0.0,
                exit_reason=exit_reason,
                equity_after=equity,
                safe_quote_after=safe_quote,
                timeframe="15m",
            )
        )
    return rows


def _tp_sl_exit_price(
    *,
    direction: int,
    reason: str,
    entry_price: float,
    exit_idx: int,
    best_tp_pct: float,
    best_sl_pct: float,
    runtime: Any,
) -> float:
    if reason == "take_profit":
        factor = 1.0 + (best_tp_pct / 100.0) if direction == 1 else 1.0 - (best_tp_pct / 100.0)
        return entry_price * factor
    if reason == "stop_loss":
        factor = 1.0 - (best_sl_pct / 100.0) if direction == 1 else 1.0 + (best_sl_pct / 100.0)
        return entry_price * factor
    if reason == "close_on_end":
        return float(runtime.last_close_15m)
    return float(runtime.price_open_15m[exit_idx])


def _trade_row(
    *,
    trade_index: int,
    entry_bar_index: int,
    exit_bar_index: int,
    entry_timestamp: str | None,
    exit_timestamp: str | None,
    side: str,
    entry_price: float,
    exit_price: float,
    quantity: float | None,
    notional_quote: float,
    gross_pnl_quote: float,
    net_pnl_quote: float,
    return_pct: float,
    fee_quote: float,
    slippage_quote: float,
    exit_reason: str,
    equity_after: float,
    safe_quote_after: float,
    timeframe: str,
) -> dict[str, Any]:
    return {
        "trade_index": trade_index,
        "entry_bar_index": entry_bar_index,
        "exit_bar_index": exit_bar_index,
        "entry_timestamp": entry_timestamp,
        "exit_timestamp": exit_timestamp,
        "side": side,
        "direction": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "notional_quote": notional_quote,
        "gross_pnl_quote": gross_pnl_quote,
        "net_pnl_quote": net_pnl_quote,
        "return_pct": return_pct,
        "fee_quote": fee_quote,
        "slippage_quote": slippage_quote,
        "exit_reason": exit_reason,
        "equity_after": equity_after,
        "safe_quote_after": safe_quote_after,
        "timeframe": timeframe,
    }


def _chart_overlay(*, trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    markers: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for trade in trades:
        trade_index = int(trade["trade_index"])
        side = str(trade["side"])
        markers.append(
            {
                "id": f"trade_{trade_index}_entry",
                "trade_index": trade_index,
                "kind": "entry",
                "timestamp": trade.get("entry_timestamp"),
                "bar_index": trade.get("entry_bar_index"),
                "price": trade.get("entry_price"),
                "side": side,
            }
        )
        markers.append(
            {
                "id": f"trade_{trade_index}_exit",
                "trade_index": trade_index,
                "kind": "exit",
                "timestamp": trade.get("exit_timestamp"),
                "bar_index": trade.get("exit_bar_index"),
                "price": trade.get("exit_price"),
                "side": side,
                "exit_reason": trade.get("exit_reason"),
            }
        )
        segments.append(
            {
                "id": f"trade_{trade_index}_segment",
                "trade_index": trade_index,
                "side": side,
                "entry": {
                    "timestamp": trade.get("entry_timestamp"),
                    "bar_index": trade.get("entry_bar_index"),
                    "price": trade.get("entry_price"),
                },
                "exit": {
                    "timestamp": trade.get("exit_timestamp"),
                    "bar_index": trade.get("exit_bar_index"),
                    "price": trade.get("exit_price"),
                    "reason": trade.get("exit_reason"),
                },
                "return_pct": trade.get("return_pct"),
            }
        )
    return {
        "schema": "backtest_chart_overlay_v1",
        "markers": markers,
        "segments": segments,
    }


def _merge_top_funding_summary(
    *,
    recomputed_summary: Mapping[str, Any],
    row_summary: Mapping[str, Any],
) -> dict[str, Any]:
    summary = dict(recomputed_summary)
    for key in _FUNDING_SUMMARY_KEYS:
        if key in row_summary:
            summary[key] = row_summary[key]
    return summary


def _should_load_funding_overlay(
    *,
    normalized_request: Mapping[str, Any],
    artifact_metadata: BacktestArtifactMetadata,
) -> bool:
    if artifact_metadata.funding_manifest_hash is None:
        return False
    coordinates = _mapping(normalized_request.get("coordinates"))
    if str(coordinates.get("market_type", "")).lower() != "futures":
        return False
    execution = _mapping(normalized_request.get("execution"))
    funding = _mapping(execution.get("funding"))
    return str(funding.get("mode", "off")) != "off"


def _funding_events_overlay(
    *,
    trades: Sequence[Mapping[str, Any]],
    funding_arrays: Any,
) -> list[dict[str, Any]]:
    funding_time = np.asarray(funding_arrays.funding_time, dtype=np.int64)
    funding_rate = np.asarray(funding_arrays.funding_rate, dtype=np.float64)
    mark_price = np.asarray(funding_arrays.mark_price, dtype=np.float64)
    funding_interval = np.asarray(funding_arrays.funding_interval_minutes, dtype=np.uint16)
    data_quality = np.asarray(funding_arrays.data_quality, dtype=np.uint8)
    events: list[dict[str, Any]] = []
    for trade in trades:
        entry_ms = _timestamp_ms(trade.get("entry_timestamp"))
        exit_ms = _timestamp_ms(trade.get("exit_timestamp"))
        if entry_ms is None or exit_ms is None or exit_ms < entry_ms:
            continue
        event_start = int(np.searchsorted(funding_time, entry_ms, side="right"))
        event_stop = int(np.searchsorted(funding_time, exit_ms, side="right"))
        for event_idx in range(event_start, event_stop):
            event = _funding_overlay_event(
                trade=trade,
                event_idx=event_idx,
                funding_time=funding_time,
                funding_rate=funding_rate,
                mark_price=mark_price,
                funding_interval=funding_interval,
                data_quality=data_quality,
            )
            events.append(event)
            if len(events) >= MAX_FUNDING_OVERLAY_EVENTS:
                return events
    return events


def _funding_overlay_event(
    *,
    trade: Mapping[str, Any],
    event_idx: int,
    funding_time: np.ndarray,
    funding_rate: np.ndarray,
    mark_price: np.ndarray,
    funding_interval: np.ndarray,
    data_quality: np.ndarray,
) -> dict[str, Any]:
    trade_index = int(trade.get("trade_index", 0))
    side = str(trade.get("side") or trade.get("direction") or "")
    rate = float(funding_rate[event_idx])
    price = _optional_float(mark_price[event_idx])
    quantity = _optional_float(trade.get("quantity"))
    direction = 1 if side == "long" else -1 if side == "short" else 0
    pnl_quote = (
        -float(direction) * quantity * price * rate
        if direction and quantity is not None and price is not None
        else None
    )
    return {
        "id": f"funding_{event_idx}_trade_{trade_index}",
        "kind": "funding_event",
        "trade_index": trade_index,
        "timestamp": _format_timestamp_ms(int(funding_time[event_idx])),
        "funding_time": _format_timestamp_ms(int(funding_time[event_idx])),
        "funding_rate": rate,
        "mark_price": price,
        "funding_interval_minutes": int(funding_interval[event_idx]),
        "data_quality": int(data_quality[event_idx]),
        "side": side or None,
        "estimated_pnl_quote": pnl_quote,
    }


def _cache_payload(
    *,
    status: str,
    cache_key: BacktestLazyTradesCacheKey,
    ttl_seconds: int,
    warning: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "cache_key": cache_key.digest,
        "cache_key_fields": cache_key.as_mapping(),
        "ttl_seconds": ttl_seconds,
        "ttl_hours": ttl_seconds / 3600.0,
    }
    if warning:
        payload["warning"] = warning
    return payload


def _read_model_from_payload(*, payload: Mapping[str, Any]) -> BacktestLazyTradesDetailReadModel:
    summary_metrics = _mapping(payload.get("summary_metrics"))
    return BacktestLazyTradesDetailReadModel(
        job_id=str(payload["job_id"]),
        variant_key=str(payload["variant_key"]),
        variant_hash=str(payload["variant_hash"]),
        request_hash=str(payload["request_hash"]),
        engine_params_hash=str(payload["engine_params_hash"]),
        artifact_manifest_hash=str(payload["artifact_manifest_hash"]),
        summary_metrics=summary_metrics,
        canonical_variant_params=_mapping(payload.get("canonical_variant_params")),
        readable_params=_mapping(payload.get("readable_params")),
        trades=tuple(
            dict(item) for item in payload.get("trades", ()) if isinstance(item, Mapping)
        ),
        chart_overlay=_mapping(payload.get("chart_overlay")),
        cache=_mapping(payload.get("cache")),
        timing=_mapping(payload.get("timing")),
        funding_manifest_hash=None
        if payload.get("funding_manifest_hash") is None
        else str(payload["funding_manifest_hash"]),
        funding=_mapping(payload.get("funding"))
        or build_backtest_funding_read_model(summary_metrics=summary_metrics),
    )


def _level_index(*, values: np.ndarray, pct_value: float | None, axis: str) -> int:
    if pct_value is None:
        raise RoehubError(
            code="backtest.invalid_request",
            message=f"{axis} is required for TP/SL lazy trades",
            details={"axis": axis},
        )
    target = np.float32(float(pct_value) / 100.0)
    matches = np.flatnonzero(np.isclose(values, target, rtol=0.0, atol=1e-7))
    if int(matches.size) != 1:
        raise RoehubError(
            code="backtest.tp_sl_grid_not_covered",
            message="Persisted best TP/SL cell is not covered by hit_times/15m subset",
            details={"axis": axis, "pct_value": pct_value, "matches": int(matches.size)},
        )
    return int(matches[0])


def _level_index_for_requested_axis(
    *,
    values: np.ndarray,
    pct_value: float | None,
    axis: str,
    enabled: bool,
) -> int:
    if enabled:
        return _level_index(values=values, pct_value=pct_value, axis=axis)
    if int(values.shape[0]) < 1:
        raise RoehubError(
            code="backtest.tp_sl_grid_not_covered",
            message="Disabled TP/SL axis has no sentinel level",
            details={"axis": axis},
        )
    return 0


def _risk_axis_enabled(request: Mapping[str, Any], axis: str) -> bool:
    risk = _mapping(request.get("risk"))
    axis_config = _mapping(risk.get(axis))
    return axis_config.get("enabled") is not False


def _risk_mode(request: Mapping[str, Any]) -> str:
    return str(_mapping(request.get("risk")).get("mode", "none"))


def _direction_mode(request: Mapping[str, Any]) -> str:
    return str(_mapping(request.get("execution")).get("direction_mode", "long_short_reversal"))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _timestamp_ms(value: Any) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None
    return int(timestamp.timestamp() * 1000)


def _format_timestamp_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000.0, tz=UTC).isoformat().replace("+00:00", "Z")


def _optional_float(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved):
        return None
    return resolved


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _side(direction: int) -> str:
    return "long" if direction == 1 else "short"


def _timestamp_for_index(values: np.ndarray, index: int) -> str | None:
    if index < 0 or index >= int(values.shape[0]):
        return None
    raw = int(values[index])
    return datetime.fromtimestamp(raw / 1000.0, tz=UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "BACKTEST_ERROR_VARIANT_CONFLICT",
    "DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS",
    "LAZY_TRADES_CACHE_HIT_STAGE_NAME",
    "LAZY_TRADES_COMPUTE_STAGE_NAME",
    "BacktestLazyTradesDetailConfig",
    "BacktestLazyTradesDetailService",
    "BacktestLazyTradesCacheProbeResult",
]
