"""Exchange-centered catalog projection; public candles plus scoped collection intent."""

from datetime import UTC, datetime
from typing import Any, Callable, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from apps.api.routes.market_data_reference import _resolve_research_scope
from trading.contexts.backtest.application.ports import ResearchOrganizationScopeResolver
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.market_data.adapters.outbound.persistence.postgres.catalog_snapshot_repository import (  # noqa: E501
    PostgresCatalogSnapshotRepository,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_bounds_repository import (  # noqa: E501
    PostgresHistoryBoundsRepository,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.instrument_selection_repository import (  # noqa: E501
    PostgresInstrumentSelectionRepository,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.work_request_repository import (  # noqa: E501
    PostgresMarketDataWorkRequestRepository,
)
from trading.contexts.market_data.application.ports.stores.catalog_coverage_reader import (
    CatalogCoverageReader,
)
from trading.contexts.market_data.application.services.work_requests import WorkRequest
from trading.contexts.market_data.application.use_cases import ListEnabledMarketsUseCase


def build_catalog_workspace_router(
    *,
    current_user_dependency: Callable[[Request], CurrentUserPrincipal],
    scope_resolver: ResearchOrganizationScopeResolver | None,
    selections: PostgresInstrumentSelectionRepository,
    snapshots: PostgresCatalogSnapshotRepository,
    jobs: PostgresMarketDataWorkRequestRepository,
    markets: ListEnabledMarketsUseCase,
    coverage: CatalogCoverageReader,
    history_bounds: PostgresHistoryBoundsRepository | None = None,
) -> APIRouter:
    router = APIRouter(tags=["market-data"])

    @router.get("/market-data/workspace/instruments")
    def instruments(
        exchange: Literal["binance", "bybit"],
        start_at: datetime,
        end_at: datetime,
        market_id: int | None = Query(default=None, ge=1),
        q: str = Query(default="", max_length=64),
        history: Literal["all", "complete", "partial", "empty", "active", "failed"] = "all",
        stream: Literal["all", "on", "off", "pinned"] = "all",
        sort: Literal["symbol", "market", "coverage", "stream"] = "symbol",
        descending: bool = False,
        offset: int = Query(default=0, ge=0, le=80000),
        limit: int = Query(default=50, ge=1, le=100),
        snapshot: str = Query(default="", max_length=150),
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> dict[str, Any]:
        scope = _resolve_research_scope(resolver=scope_resolver, principal=principal)
        now = datetime.now(UTC)
        try:
            expected = WorkRequest("candle_ingestion", 1, ("BTCUSDT",), start_at, end_at).validate(
                now=now
            )
            if expected > 10080:
                raise ValueError("catalog coverage reads are limited to seven days")
            ids = tuple(UUID(s) for s in snapshot.split(",")) if snapshot else ()
        except ValueError as error:
            raise HTTPException(422, "Invalid catalog range or snapshot") from error
        sources = {
            m.market_id.value: m
            for m in markets.execute()
            if m.exchange_name == exchange and (market_id is None or m.market_id.value == market_id)
        }
        if not sources:
            raise HTTPException(404, "Exchange market is unavailable")
        catalogs = snapshots.current_catalogs(market_ids=list(sources), snapshot_ids=ids)
        if ids and {UUID(str(c["snapshot_id"])) for c in catalogs} != set(ids):
            raise HTTPException(409, "Catalog snapshot expired; refresh the list")
        selected = {
            (r.instrument_id.market_id.value, str(r.instrument_id.symbol))
            for r in selections.list_for_organization(organization_id=scope.organization_id)
        }
        pins: dict[str, list[dict[str, str]]] = {}
        for pin in selections.strategy_pins(organization_id=scope.organization_id):
            pins.setdefault(str(pin["instrument_key"]), []).append(
                {
                    "strategy_id": str(pin["strategy_id"]),
                    "name": str(pin["name"]),
                    "state": str(pin["state"]),
                }
            )
        counts = coverage.counts(market_ids=list(sources), start_at=start_at, end_at=end_at)
        requests = {
            (int(r["market_id"]), str(r["symbol"])): r
            for r in jobs.instrument_jobs(
                organization_id=scope.organization_id.value,
                market_ids=list(sources),
                start_at=start_at,
                end_at=end_at,
            )
        }
        rows: list[dict[str, Any]] = []
        for catalog in catalogs:
            market = sources[int(catalog["market_id"])]
            if len(catalog["items"]) > 20000:
                raise HTTPException(503, "Catalog exceeds projection limit")
            for item in catalog["items"]:
                symbol = str(item["symbol"])
                if q.strip().upper() not in symbol:
                    continue
                key = (market.market_id.value, symbol)
                strategies = pins.get(f"{market.market_code}:{symbol}", [])
                effective = key in selected or bool(strategies)
                if (
                    (stream == "on" and not effective)
                    or (stream == "off" and effective)
                    or (stream == "pinned" and not strategies)
                ):
                    continue
                actual = min(counts.get(key, 0), expected)
                state = "complete" if actual == expected else "partial" if actual else "empty"
                job = requests.get(key)
                job_minutes = (
                    int(
                        (job.get("end_at", end_at) - job.get("start_at", start_at)).total_seconds()
                        // 60
                    )
                    if job
                    else expected
                )
                active = job is not None and job["state"] in (
                    "queued",
                    "running",
                    "pause_requested",
                    "paused",
                    "retry_wait",
                    "cancel_requested",
                )
                if history in {"complete", "partial", "empty"} and state != history:
                    continue
                if history == "active" and not active:
                    continue
                if history == "failed" and (job is None or job["state"] != "failed"):
                    continue
                rows.append(
                    {
                        **{
                            k: item.get(k)
                            for k in (
                                "base_asset",
                                "quote_asset",
                                "price_step",
                                "qty_step",
                                "min_notional",
                            )
                        },
                        "market_id": key[0],
                        "symbol": symbol,
                        "market_type": market.market_type,
                        "exchange_name": exchange,
                        "selected": key in selected,
                        "effective": effective,
                        "strategy_pinned": bool(strategies),
                        "strategies": strategies,
                        "refreshed_at": catalog["refreshed_at"],
                        "coverage_state": state,
                        "coverage_percent": round(actual / expected * 100, 2),
                        "actual_candles": actual,
                        "expected_candles": expected,
                        "job": None
                        if job is None
                        else {
                            "job_id": str(job["job_id"]),
                            "state": job["state"],
                            "progress_percent": round(
                                min(
                                    100,
                                    max(
                                        0,
                                        (
                                            int(job["completed_units"])
                                            - (int(job["position"]) - 1) * job_minutes
                                        )
                                        / job_minutes
                                        * 100,
                                    ),
                                ),
                                2,
                            ),
                        },
                    }
                )
        field = {
            "symbol": "symbol",
            "market": "market_type",
            "coverage": "coverage_percent",
            "stream": "effective",
        }[sort]
        rows.sort(key=lambda r: (r[field], r["symbol"], r["market_id"]), reverse=descending)
        return {
            "exchange": exchange,
            "start_at": start_at,
            "end_at": end_at,
            "observed_at": now,
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "snapshot": ",".join(str(c["snapshot_id"]) for c in catalogs),
            "missing_market_ids": [
                m for m in sources if m not in {int(c["market_id"]) for c in catalogs}
            ],
            "items": rows[offset : offset + limit],
        }

    @router.get("/market-data/workspace/collection")
    def collection(
        market_id: int = Query(ge=1),
        symbol: str = Query(min_length=2, max_length=64),
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> dict[str, Any]:
        scope = _resolve_research_scope(resolver=scope_resolver, principal=principal)
        market = next((m for m in markets.execute() if m.market_id.value == market_id), None)
        if market is None:
            raise HTTPException(404, "Exchange market is unavailable")
        catalog = snapshots.latest(market_id=market_id)
        found = (
            snapshots.page(snapshot_id=catalog["snapshot_id"], prefix=symbol, after="", limit=1)
            if catalog
            else []
        )
        if not found or found[0]["item"]["symbol"] != symbol:
            raise HTTPException(404, "Instrument is unavailable")
        selected = any(
            r.instrument_id.market_id.value == market_id and str(r.instrument_id.symbol) == symbol
            for r in selections.list_for_organization(organization_id=scope.organization_id)
        )
        pins = [
            {"strategy_id": str(p["strategy_id"]), "name": str(p["name"]), "state": str(p["state"])}
            for p in selections.strategy_pins(organization_id=scope.organization_id)
            if p["instrument_key"] == f"{market.market_code}:{symbol}"
        ]
        now = datetime.now(UTC)
        latest = coverage.latest(
            market_id=market_id, symbol=symbol, before=now.replace(second=0, microsecond=0)
        )
        return {
            "market_id": market_id,
            "symbol": symbol,
            "selected": selected,
            "effective": selected or bool(pins),
            "strategy_pinned": bool(pins),
            "strategies": pins,
            "last_candle_at": latest,
            "observed_at": now,
        }

    @router.get("/market-data/workspace/history-bounds")
    def available_history(
        market_id: int = Query(ge=1),
        symbol: str = Query(min_length=2, max_length=64),
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> dict[str, Any]:
        _resolve_research_scope(resolver=scope_resolver, principal=principal)
        if history_bounds is None:
            raise HTTPException(503, "History discovery is unavailable")
        if not any(m.market_id.value == market_id for m in markets.execute()):
            raise HTTPException(404, "Exchange market is unavailable")
        catalog = snapshots.latest(market_id=market_id)
        found = (
            snapshots.page(snapshot_id=catalog["snapshot_id"], prefix=symbol, after="", limit=1)
            if catalog
            else []
        )
        if not found or found[0]["item"]["symbol"] != symbol:
            raise HTTPException(404, "Instrument is unavailable")
        now = datetime.now(UTC)
        row = history_bounds.read_or_request(market_id=market_id, symbol=symbol, now=now)
        if row is None:
            raise HTTPException(
                503, "History discovery queue is full", headers={"Retry-After": "30"}
            )
        return {
            "market_id": market_id,
            "symbol": symbol,
            "state": row["state"],
            "first_open_at": row["first_open_at"],
            "end_at": now.replace(second=0, microsecond=0),
            "observed_at": now,
        }

    return router
