from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import ceil
from typing import Any, Literal, Mapping

from fastapi import APIRouter

from apps.api.dto.ui_backtests import (
    BacktestConfigDraftResponse,
    BacktestFooterStatusResponse,
    BacktestIndicatorCatalogResponse,
    BacktestIndicatorCatalogRowResponse,
    BacktestInstrumentUniverseResponse,
    BacktestJobTableFiltersResponse,
    BacktestJobTableResponse,
    BacktestJobTableRowResponse,
    BacktestOptimizationOverviewResponse,
    BacktestOptionResponse,
    BacktestRecentEventResponse,
    BacktestRecentEventsResponse,
    BacktestRefreshControlResponse,
    BacktestWorkstationResponse,
    BacktestWorkstationSourceResponse,
)
from apps.api.routes.ui_backtests import (
    build_ui_backtests_router as build_ui_backtests_api_router,
)
from apps.api.wiring.modules.backtest import (
    _build_jobs_use_case,
    _with_local_dev_default,
)
from trading.contexts.backtest.adapters.outbound import (
    BacktestArtifactPathBuilderV2,
    FilesystemBacktestArtifactContextResolver,
    YamlBacktestGridDefaultsProvider,
    build_backtest_artifacts_runtime_config_hash,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestPreflightService,
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
)
from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (
    YamlBacktestArtifactLoaderV2,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal

_DEFAULT_REFRESH_INTERVAL_SECONDS = 15
_MINIMUM_MANUAL_REFRESH_SECONDS = 10
_JOB_TABLE_LIMIT = 50
_INSTRUMENT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_MARKETS = (("binance", "Binance"),)
_MARKET_TYPES = (("spot", "Spot"),)
_EVENT_SOURCE = "backtest_job_events"
_JOB_SOURCE = "backtest_jobs"
_DEFAULT_STRATEGY = "mean_reversion.py"


@dataclass(frozen=True, slots=True)
class _RefreshDecision:
    status: Literal["fresh", "rate_limited"]
    next_allowed_refresh_at: datetime | None
    retry_after_seconds: int | None


class BacktestWorkstationManualRefreshLimiter:
    def __init__(self, *, interval_seconds: int = _MINIMUM_MANUAL_REFRESH_SECONDS) -> None:
        if interval_seconds < 1:
            raise ValueError(
                "BacktestWorkstationManualRefreshLimiter requires positive interval_seconds"
            )
        self._interval = timedelta(seconds=interval_seconds)
        self._next_allowed_by_user: dict[str, datetime] = {}

    def resolve(
        self,
        *,
        user_id: str,
        requested_at: datetime,
        refresh: Literal["initial", "auto", "manual"],
    ) -> _RefreshDecision:
        if refresh != "manual":
            return _RefreshDecision(
                status="fresh",
                next_allowed_refresh_at=self._next_allowed_by_user.get(user_id),
                retry_after_seconds=None,
            )
        next_allowed = self._next_allowed_by_user.get(user_id)
        if next_allowed is not None and requested_at < next_allowed:
            return _RefreshDecision(
                status="rate_limited",
                next_allowed_refresh_at=next_allowed,
                retry_after_seconds=max(1, ceil((next_allowed - requested_at).total_seconds())),
            )
        next_allowed = requested_at + self._interval
        self._next_allowed_by_user[user_id] = next_allowed
        return _RefreshDecision(
            status="fresh",
            next_allowed_refresh_at=next_allowed,
            retry_after_seconds=None,
        )


class BacktestWorkstationQueryService:
    def __init__(
        self,
        *,
        runtime_defaults_service: BacktestRuntimeDefaultsService,
        jobs_use_case: BacktestJobsUseCase | None,
        instrument_symbols: tuple[str, ...] = _INSTRUMENT_SYMBOLS,
        refresh_limiter: BacktestWorkstationManualRefreshLimiter | None = None,
    ) -> None:
        self._runtime_defaults_service = runtime_defaults_service
        self._jobs_use_case = jobs_use_case
        self._instrument_symbols = instrument_symbols or _INSTRUMENT_SYMBOLS
        self._refresh_limiter = refresh_limiter or BacktestWorkstationManualRefreshLimiter()

    def get_workstation(
        self,
        *,
        principal: CurrentUserPrincipal,
        cursor: str | None,
        state: str | None,
        query: str,
        exchange: str | None,
        market_type: str | None,
        symbol: str | None,
        launched_from: str | None,
        launched_to: str | None,
        refresh: Literal["initial", "auto", "manual"],
    ) -> BacktestWorkstationResponse:
        generated_at = datetime.now(UTC)
        refresh_decision = self._refresh_limiter.resolve(
            user_id=str(principal.user_id),
            requested_at=generated_at,
            refresh=refresh,
        )
        runtime_defaults = self._runtime_defaults_service.execute().as_mapping()
        job_table = self._build_job_table(
            principal=principal,
            state=state,
            cursor=cursor,
            query=query,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            launched_from=launched_from,
            launched_to=launched_to,
        )
        optimization = _build_optimization_overview(job_table=job_table)
        sources = [
            _source("runtime_defaults", "available", generated_at),
            _source(
                _JOB_SOURCE,
                "available" if self._jobs_use_case is not None else "unavailable",
                generated_at,
                detail=None
                if self._jobs_use_case is not None
                else "STRATEGY_PG_DSN is not configured for job history",
            ),
            _source(
                _EVENT_SOURCE,
                "available",
                generated_at,
                detail="recent events are derived from bounded job history in Stage 8",
            ),
        ]
        refresh_status: Literal["fresh", "degraded", "rate_limited"] = (
            "rate_limited"
            if refresh_decision.status == "rate_limited"
            else "degraded"
            if any(source.status != "available" for source in sources)
            else "fresh"
        )

        return BacktestWorkstationResponse(
            generated_at=_iso(generated_at),
            refresh_status=refresh_status,
            next_allowed_refresh_at=_iso_or_none(refresh_decision.next_allowed_refresh_at),
            retry_after_seconds=refresh_decision.retry_after_seconds,
            sources=sources,
            runtime_defaults=runtime_defaults,
            config_draft=_build_config_draft(runtime_defaults=runtime_defaults),
            ai_configurator_state={
                "state": "placeholder",
                "enabled": False,
                "stage": "Stage 10",
                "suggested_strategy": _DEFAULT_STRATEGY,
            },
            instrument_universe=_build_instrument_universe(
                runtime_defaults=runtime_defaults,
                instrument_symbols=self._instrument_symbols,
            ),
            indicator_catalog=_build_indicator_catalog(runtime_defaults=runtime_defaults),
            optimization_overview=optimization,
            recent_events=_build_recent_events(job_table=job_table, generated_at=generated_at),
            job_table=job_table,
            footer_status=BacktestFooterStatusResponse(
                api="available",
                worker="configured" if self._jobs_use_case is not None else "unavailable",
                queue=optimization.state,
                generated_at=_iso(generated_at),
                data="bounded workstation read-model",
            ),
            refresh_control=BacktestRefreshControlResponse(
                manual=True,
                autorefresh_presets=["off", "10s", "15s", "30s", "1m", "5m"],
                default_preset=f"{_DEFAULT_REFRESH_INTERVAL_SECONDS}s",
                generated_at=_iso(generated_at),
                refresh_status=refresh_status,
                next_allowed_refresh_at=_iso_or_none(refresh_decision.next_allowed_refresh_at),
                retry_after_seconds=refresh_decision.retry_after_seconds,
            ),
        )

    def _build_job_table(
        self,
        *,
        principal: CurrentUserPrincipal,
        state: str | None,
        cursor: str | None,
        query: str,
        exchange: str | None,
        market_type: str | None,
        symbol: str | None,
        launched_from: str | None,
        launched_to: str | None,
    ) -> BacktestJobTableResponse:
        filters = BacktestJobTableFiltersResponse(
            state=state,
            cursor=cursor,
            query=query,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            launched_from=launched_from,
            launched_to=launched_to,
            limit=_JOB_TABLE_LIMIT,
            sort="created_desc",
        )
        if self._jobs_use_case is None:
            return BacktestJobTableResponse(
                source=_JOB_SOURCE,
                state="unavailable",
                filters=filters,
                items=[],
                next_cursor=None,
                degradation_reason="backtest jobs repository is not configured",
            )
        result = self._jobs_use_case.list(
            user_id=principal.user_id,
            state=state,
            risk_mode=None,
            limit=_JOB_TABLE_LIMIT,
            cursor=cursor,
        )
        rows = [_build_job_row(item.as_mapping()) for item in result.items]
        if query:
            normalized_query = query.casefold()
            rows = [
                row
                for row in rows
                if normalized_query in row.job_id.casefold()
                or normalized_query in row.strategy.casefold()
                or normalized_query in row.indicator_summary.casefold()
            ]
        if exchange:
            rows = [row for row in rows if row.exchange.casefold() == exchange.casefold()]
        if market_type:
            rows = [
                row for row in rows if row.market_type.casefold() == market_type.casefold()
            ]
        if symbol:
            rows = [row for row in rows if row.symbol.casefold() == symbol.casefold()]
        if launched_from:
            rows = [row for row in rows if row.created_at[:10] >= launched_from]
        if launched_to:
            rows = [row for row in rows if row.created_at[:10] <= launched_to]
        return BacktestJobTableResponse(
            source=_JOB_SOURCE,
            state="ready" if rows else "empty",
            filters=filters,
            items=rows,
            next_cursor=result.next_cursor,
        )


def build_ui_backtests_router(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_backtests_router requires current_user_dependency")
    effective_environ = _with_local_dev_default(environ=environ)
    artifact_config_path = resolve_backtest_artifacts_config_path(environ=effective_environ)
    artifact_config = load_backtest_artifacts_runtime_config(artifact_config_path)
    defaults_provider = YamlBacktestGridDefaultsProvider.from_environ(
        environ=effective_environ,
        artifact_config_path=artifact_config_path,
    )
    artifact_path_builder = BacktestArtifactPathBuilderV2(
        root=artifact_config.artifact_root_path()
    )
    artifact_loader = YamlBacktestArtifactLoaderV2(path_resolver=artifact_path_builder)
    artifact_array_loader = FilesystemBacktestArtifactArrayLoader(
        artifact_loader=artifact_loader
    )
    runtime_config = BacktestRuntimeConfig(
        hit_times_tp_levels_pct=artifact_config.hit_times_grid.tp_levels_pct,
        hit_times_sl_levels_pct=artifact_config.hit_times_grid.sl_levels_pct,
        artifact_config_hash=build_backtest_artifacts_runtime_config_hash(
            config=artifact_config
        ),
    )
    runtime_defaults_service = BacktestRuntimeDefaultsService(
        defaults_provider=defaults_provider,
        runtime_config=runtime_config,
    )
    artifact_context_resolver = FilesystemBacktestArtifactContextResolver(
        artifact_loader=artifact_loader
    )
    preflight_service = BacktestPreflightService(
        defaults_provider=defaults_provider,
        artifact_context_resolver=artifact_context_resolver,
        runtime_config=runtime_config,
    )
    jobs_use_case = _build_jobs_use_case(
        environ=effective_environ,
        defaults_provider=defaults_provider,
        artifact_array_loader=artifact_array_loader,
        preflight_service=preflight_service,
        runtime_config=runtime_config,
    )
    return build_ui_backtests_api_router(
        workstation_service=BacktestWorkstationQueryService(
            runtime_defaults_service=runtime_defaults_service,
            jobs_use_case=jobs_use_case,
            instrument_symbols=_discover_artifact_symbols(artifact_config=artifact_config),
        ),
        current_user_dependency=current_user_dependency,
    )


def _source(
    name: str,
    status: Literal["available", "degraded", "unavailable"],
    generated_at: datetime,
    *,
    detail: str | None = None,
) -> BacktestWorkstationSourceResponse:
    return BacktestWorkstationSourceResponse(
        name=name,
        status=status,
        generated_at=_iso(generated_at),
        detail=detail,
    )


def _build_config_draft(*, runtime_defaults: Mapping[str, Any]) -> BacktestConfigDraftResponse:
    indicator_ids = list(runtime_defaults.get("supported_indicator_ids") or [])
    indicator_sources = dict(runtime_defaults.get("indicator_sources") or {})
    indicator_param_specs = dict(runtime_defaults.get("indicator_param_specs") or {})
    ranking_default = dict(runtime_defaults.get("ranking_default") or {})
    execution_defaults = dict(runtime_defaults.get("execution_defaults") or {})
    default_end = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    return BacktestConfigDraftResponse(
        coordinates={"exchange": "binance", "market_type": "spot", "symbol": "BTCUSDT"},
        timeframe=_first(runtime_defaults.get("supported_timeframes"), default="15m"),
        time_range={"start": "2023-01-01T00:00:00Z", "end": f"{default_end}T00:00:00Z"},
        indicators=_default_indicator_grid(
            indicator_ids=indicator_ids,
            indicator_sources=indicator_sources,
            indicator_param_specs=indicator_param_specs,
        ),
        risk={"mode": _first(runtime_defaults.get("risk_modes"), default="none")},
        execution={
            "direction_mode": _first(
                runtime_defaults.get("direction_modes"),
                default="long_short_reversal",
            ),
            "fee_rate": execution_defaults.get("fee_rate", 0.00075),
            "slippage_rate": execution_defaults.get("slippage_rate", 0.0001),
            "initial_cash_quote": execution_defaults.get("initial_cash_quote", 10000.0),
            "sizing": execution_defaults.get(
                "sizing",
                {"mode": "fixed_equity_pct", "equity_pct": 10.0},
            ),
            "profit_lock": execution_defaults.get("profit_lock", {"enabled": False}),
            "close_on_end": execution_defaults.get("close_on_end", True),
        },
        ranking={
            "primary_metric": ranking_default.get("primary_metric", "total_return_pct"),
            "direction": ranking_default.get("direction", "desc"),
        },
        top_n=int(runtime_defaults.get("top_n_default") or 100),
    )


def _build_instrument_universe(
    *,
    runtime_defaults: Mapping[str, Any],
    instrument_symbols: tuple[str, ...],
) -> BacktestInstrumentUniverseResponse:
    selected_symbol = _first(instrument_symbols, default="BTCUSDT")
    return BacktestInstrumentUniverseResponse(
        source="artifact_manifests",
        state="ready",
        markets=[BacktestOptionResponse(value=value, label=label) for value, label in _MARKETS],
        market_types=[
            BacktestOptionResponse(value=value, label=label) for value, label in _MARKET_TYPES
        ],
        symbols=[
            BacktestOptionResponse(value=symbol, label=symbol)
            for symbol in instrument_symbols
        ],
        timeframes=[
            BacktestOptionResponse(value=value, label=value)
            for value in list(runtime_defaults.get("supported_timeframes") or ["15m"])
        ],
        selected_symbols=[selected_symbol],
    )


def _discover_artifact_symbols(*, artifact_config: Any) -> tuple[str, ...]:
    root = artifact_config.artifact_root_path()
    discovered: set[str] = set()
    for exchange, _label in _MARKETS:
        for market_type, _market_label in _MARKET_TYPES:
            symbol_root = root / exchange / market_type
            if not symbol_root.exists() or not symbol_root.is_dir():
                continue
            for child in symbol_root.iterdir():
                if child.is_dir() and (child / "current.yaml").exists():
                    discovered.add(child.name.upper())
    return tuple(sorted(discovered)) or _INSTRUMENT_SYMBOLS


def _build_indicator_catalog(
    *,
    runtime_defaults: Mapping[str, Any],
) -> BacktestIndicatorCatalogResponse:
    indicator_sources = dict(runtime_defaults.get("indicator_sources") or {})
    indicator_param_specs = dict(runtime_defaults.get("indicator_param_specs") or {})
    indicator_ids = list(runtime_defaults.get("supported_indicator_ids") or [])
    rows = [
        _build_indicator_catalog_row(
            indicator_id=indicator_id,
            sources=list(indicator_sources.get(indicator_id) or []),
            param_specs=dict(indicator_param_specs.get(indicator_id) or {}),
        )
        for indicator_id in indicator_ids
    ]
    return BacktestIndicatorCatalogResponse(
        source="runtime_defaults",
        state="ready" if rows else "empty",
        items=rows,
        total_combinations_estimate=max(1, len(rows)) * 3600,
    )


def _build_indicator_catalog_row(
    *,
    indicator_id: str,
    sources: list[str],
    param_specs: Mapping[str, Any],
) -> BacktestIndicatorCatalogRowResponse:
    params = dict(param_specs.get("params") or {})
    primary_spec = dict(params.get("window") or next(iter(params.values()), {}))
    return BacktestIndicatorCatalogRowResponse(
        indicator_id=indicator_id,
        label=_indicator_label(indicator_id),
        family=indicator_id.split(".", maxsplit=1)[0] if "." in indicator_id else "other",
        min_value=primary_spec.get("start"),
        max_value=primary_spec.get("stop_incl"),
        step=primary_spec.get("step"),
        sources=sources,
        param_specs=dict(param_specs),
    )


def _build_optimization_overview(
    *,
    job_table: BacktestJobTableResponse,
) -> BacktestOptimizationOverviewResponse:
    queued = sum(1 for row in job_table.items if row.state == "queued")
    running = sum(1 for row in job_table.items if row.state == "running")
    completed = sum(1 for row in job_table.items if row.state == "succeeded")
    active_rows = [row for row in job_table.items if row.state in {"queued", "running"}]
    selected = active_rows[0] if active_rows else (job_table.items[0] if job_table.items else None)
    return BacktestOptimizationOverviewResponse(
        source=job_table.source,
        state="ready" if selected is not None else job_table.state,
        active_job_id=selected.job_id if selected is not None else None,
        progress_percent=selected.progress_percent if selected is not None else 0,
        processed_units=0,
        total_units=0,
        completed_jobs=completed,
        running_jobs=running,
        queued_jobs=queued,
        estimated_remaining="01:18:42" if active_rows else None,
        degradation_reason=job_table.degradation_reason,
    )


def _build_recent_events(
    *,
    job_table: BacktestJobTableResponse,
    generated_at: datetime,
) -> BacktestRecentEventsResponse:
    if not job_table.items:
        return BacktestRecentEventsResponse(
            source=_EVENT_SOURCE,
            state="empty",
            items=[
                BacktestRecentEventResponse(
                    timestamp=_iso(generated_at),
                    level="info",
                    message="workstation initialized; no recent jobs",
                    job_id=None,
                )
            ],
        )
    return BacktestRecentEventsResponse(
        source=_EVENT_SOURCE,
        state="ready",
        items=[
            BacktestRecentEventResponse(
                timestamp=_iso(generated_at),
                level="info",
                message=f"job {row.state}: {row.strategy}",
                job_id=row.job_id,
            )
            for row in job_table.items[:5]
        ],
    )


def _build_job_row(item: Mapping[str, Any]) -> BacktestJobTableRowResponse:
    request = dict(item.get("request") or {})
    coordinates = dict(request.get("coordinates") or {})
    execution = dict(request.get("execution") or {})
    indicators = list(request.get("indicators") or [])
    progress = dict(item.get("progress") or {})
    terminal_summary = dict(item.get("terminal_summary") or {})
    metrics = dict(terminal_summary.get("metrics") or {})
    period = _format_period(request.get("time_range"))
    return BacktestJobTableRowResponse(
        job_id=str(item.get("job_id") or ""),
        state=str(item.get("state") or "unknown"),
        strategy=_DEFAULT_STRATEGY,
        exchange=str(coordinates.get("exchange") or "--"),
        market_type=str(coordinates.get("market_type") or "--"),
        symbol=str(coordinates.get("symbol") or "--"),
        created_at=str(item.get("created_at") or ""),
        indicator_summary=", ".join(
            _indicator_label(str(row.get("indicator_id") or "")) for row in indicators
        )
        or "Indicators",
        period=period,
        direction=str(execution.get("direction_mode") or "--"),
        combinations=int(terminal_summary.get("candidate_combinations") or 0) or None,
        best_return_pct=_metric_float(metrics, "total_return_pct"),
        best_sharpe=_metric_float(metrics, "sharpe"),
        avg_drawdown_pct=_metric_float(metrics, "max_drawdown_pct"),
        profit_factor=_metric_float(metrics, "profit_factor"),
        win_rate_pct=_metric_float(metrics, "win_rate_pct"),
        trades_count=_metric_int(metrics, "trade_count"),
        progress_percent=int(progress.get("percent") or 0),
        refresh_status=str(item.get("refresh_status") or "poll"),
        retry_after_seconds=int(item.get("retry_after_seconds") or 0),
        links=dict(item.get("links") or {}),
        actions={
            "can_cancel": str(item.get("state") or "") in {"queued", "running"},
            "can_delete": str(item.get("state") or "") in {"succeeded", "failed", "cancelled"},
            "can_open_top": str(item.get("state") or "") == "succeeded",
        },
    )


def _default_indicator_grid(
    *,
    indicator_ids: list[str],
    indicator_sources: Mapping[str, Any],
    indicator_param_specs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = ["ma.dema"] if "ma.dema" in indicator_ids else indicator_ids[:1] or ["ma.dema"]
    rows: list[dict[str, Any]] = []
    for indicator_id in selected:
        spec = dict(indicator_param_specs.get(indicator_id) or {})
        params = dict(spec.get("params") or {})
        row: dict[str, Any] = {
            "indicator_id": indicator_id,
            "params": {
                name: _default_param_value(param_spec)
                for name, param_spec in params.items()
                if isinstance(param_spec, Mapping)
            },
        }
        window = row["params"].get("window")
        if isinstance(window, Mapping):
            row["window"] = {
                "start": window.get("start"),
                "stop": window.get("stop"),
                "step": window.get("step"),
            }
        sources = list(indicator_sources.get(indicator_id) or [])
        if sources:
            row["sources"] = [sources[0]]
        rows.append(row)
    return rows


def _default_param_value(spec: Mapping[str, Any]) -> dict[str, Any] | int | float | str | None:
    mode = spec.get("mode")
    if mode == "range":
        return {
            "start": spec.get("start"),
            "stop": spec.get("stop_incl"),
            "step": spec.get("step"),
        }
    values = list(spec.get("values") or [])
    return values[0] if values else None


def _indicator_label(indicator_id: str) -> str:
    return indicator_id.rsplit(".", maxsplit=1)[-1].replace("_", " ").upper()


def _first(value: Any, *, default: str) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, tuple) and value:
        return str(value[0])
    return default


def _metric_float(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else None


def _metric_int(metrics: Mapping[str, Any], key: str) -> int | None:
    value = metrics.get(key)
    return int(value) if isinstance(value, int | float) else None


def _format_period(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "--"
    start = str(value.get("start") or "")[:10]
    end = str(value.get("end") or "")[:10]
    if start and end:
        return f"{start} -> {end}"
    return start or end or "--"


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _iso_or_none(value: datetime | None) -> str | None:
    return _iso(value) if value is not None else None


__all__ = [
    "BacktestWorkstationManualRefreshLimiter",
    "BacktestWorkstationQueryService",
    "build_ui_backtests_router",
]
