from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.routes.market_data_catalog import build_catalog_workspace_router
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
from trading.contexts.market_data.application.use_cases import ListEnabledMarketsUseCase
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    OrganizationId,
    PaidLevel,
    Symbol,
    UserId,
)


@pytest.fixture
def catalog_surface():
    org = OrganizationId(uuid4())
    ids = [uuid4(), uuid4()]
    end = datetime(2024, 1, 2, tzinfo=UTC)
    seen = []

    def current(request: Request):
        if request.headers.get("x-user") != "allowed":
            raise HTTPException(401)
        return CurrentUserPrincipal(UserId(uuid4()), PaidLevel.free())

    def scoped(**kw):
        seen.append(kw)
        assert kw["organization_id"] in (org, org.value)

    def item(symbol):
        return dict(
            symbol=symbol,
            base_asset="BTC",
            quote_asset="USDT",
            price_step=0.01,
            qty_step=0.01,
            min_notional=5,
            internal_secret="must-not-leak",
        )

    rows = [
        dict(
            snapshot_id=ids[0],
            market_id=1,
            refreshed_at=end,
            items=[item("BTCUSDT"), item("ETHUSDT")],
        ),
        dict(snapshot_id=ids[1], market_id=2, refreshed_at=end, items=[item("BTCUSDT")]),
    ]
    pins = [
        dict(
            instrument_key="binance:spot:BTCUSDT",
            strategy_id=uuid4(),
            name="Owned strategy",
            state="running",
            internal_secret="must-not-leak",
        )
    ]
    selections = SimpleNamespace(
        list_for_organization=lambda **kw: (
            scoped(**kw)
            or [SimpleNamespace(instrument_id=InstrumentId(MarketId(1), Symbol("ETHUSDT")))]
        ),
        strategy_pins=lambda **kw: (scoped(**kw) or pins),
    )
    app = FastAPI()
    snapshots = SimpleNamespace(
        current_catalogs=lambda **kw: [r for r in rows if r["market_id"] in kw["market_ids"]],
        latest=lambda **kw: rows[0],
        page=lambda **kw: [dict(item=item(kw["prefix"]))],
    )
    probes = []
    history_bounds = SimpleNamespace(
        read_or_request=lambda **kw: probes.append(kw)
        or dict(state="ready", first_open_at=datetime(2017, 8, 17, 4, tzinfo=UTC))
    )
    app.state.history_bounds = history_bounds
    app.include_router(
        build_catalog_workspace_router(
            history_bounds=cast(PostgresHistoryBoundsRepository, history_bounds),
            current_user_dependency=current,
            scope_resolver=cast(ResearchOrganizationScopeResolver, SimpleNamespace(
                resolve=lambda **kw: SimpleNamespace(organization_id=org)
            )),
            selections=cast(PostgresInstrumentSelectionRepository, selections),
            snapshots=cast(PostgresCatalogSnapshotRepository, snapshots),
            jobs=cast(PostgresMarketDataWorkRequestRepository, SimpleNamespace(
                instrument_jobs=lambda **kw: (
                    scoped(**kw)
                    or [
                        dict(
                            market_id=1,
                            symbol="ETHUSDT",
                            job_id=uuid4(),
                            state="running",
                            completed_units=2160,
                            position=2,
                            total_units=2880,
                        )
                    ]
                )
            )),
            markets=cast(ListEnabledMarketsUseCase, SimpleNamespace(
                execute=lambda: [
                    SimpleNamespace(
                        market_id=MarketId(m),
                        exchange_name="binance",
                        market_type="spot" if m == 1 else "futures",
                        market_code="binance:spot" if m == 1 else "binance:futures",
                    )
                    for m in (1, 2)
                ]
            )),
            coverage=cast(CatalogCoverageReader, SimpleNamespace(
                counts=lambda **kw: {(1, "BTCUSDT"): 1440, (1, "ETHUSDT"): 720},
                latest=lambda **kw: end,
            )),
        )
    )
    app.state.probes = probes
    return (
        TestClient(app),
        seen,
        dict(
            exchange="binance",
            start_at=(end - timedelta(days=1)).isoformat(),
            end_at=end.isoformat(),
        ),
        ids,
    )


def test_auth_and_scoped_projection_with_multiple_markets(catalog_surface):
    client, seen, params, _ = catalog_surface
    assert client.get("/market-data/workspace/instruments", params=params).status_code == 401
    assert not seen
    r = client.get(
        "/market-data/workspace/instruments", params=params, headers={"x-user": "allowed"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    assert [(r["symbol"], r["market_id"]) for r in body["items"]] == [
        ("BTCUSDT", 1),
        ("BTCUSDT", 2),
        ("ETHUSDT", 1),
    ]
    assert body["items"][0]["strategy_pinned"] and body["items"][0]["effective"]
    assert body["items"][0]["strategies"][0]["name"] == "Owned strategy"
    assert body["items"][1]["coverage_percent"] == 0
    assert body["items"][2]["coverage_percent"] == 50
    assert body["items"][2]["job"]["progress_percent"] == 50  # Per instrument, not batch 75%.
    assert "must-not-leak" not in r.text


def test_filters_and_sort_are_applied_before_pagination(catalog_surface):
    client, _, params, _ = catalog_surface

    def get(**extra):
        return client.get(
            "/market-data/workspace/instruments",
            params={**params, **extra},
            headers={"x-user": "allowed"},
        ).json()

    assert get(history="empty")["items"][0]["market_id"] == 2
    assert get(stream="pinned")["total"] == 1
    assert get(stream="off")["total"] == 1
    assert get(history="active")["items"][0]["symbol"] == "ETHUSDT"
    assert get(market_id=2)["total"] == 1
    assert (
        get(sort="coverage", descending=True, offset=1, limit=1)["items"][0]["symbol"] == "ETHUSDT"
    )
    assert get(q="USDT")["total"] == 3
    assert get(q="NOTFOUND")["total"] == 0


def test_expired_snapshot_and_invalid_range_fail_closed(catalog_surface):
    client, _, params, _ = catalog_surface
    for extra, status in [
        ({"snapshot": str(uuid4())}, 409),
        ({"snapshot": "bad"}, 422),
        ({"start_at": "2020-01-01T00:00:00Z"}, 422),
        ({"market_id": 4}, 404),
    ]:
        assert (
            client.get(
                "/market-data/workspace/instruments",
                params={**params, **extra},
                headers={"x-user": "allowed"},
            ).status_code
            == status
        )


def test_collection_exposes_scoped_strategy_reasons_and_observed_candle(catalog_surface):
    client, seen, _, _ = catalog_surface
    r = client.get(
        "/market-data/workspace/collection",
        params={"market_id": 1, "symbol": "BTCUSDT"},
        headers={"x-user": "allowed"},
    )
    assert r.status_code == 200
    assert r.json()["effective"] and r.json()["strategy_pinned"]
    assert r.json()["last_candle_at"] == "2024-01-02T00:00:00Z"
    assert "must-not-leak" not in r.text


def test_history_defaults_are_instrument_scoped_and_admitted_only_after_auth(catalog_surface):
    client, _, _, _ = catalog_surface
    path = "/market-data/workspace/history-bounds"
    assert client.get(path, params=dict(market_id=1, symbol="BTCUSDT")).status_code == 401
    assert not client.app.state.probes
    assert (
        client.get(
            path, params=dict(market_id=77, symbol="BTCUSDT"), headers={"x-user": "allowed"}
        ).status_code
        == 404
    )
    assert not client.app.state.probes
    result = client.get(
        path, params=dict(market_id=1, symbol="BTCUSDT"), headers={"x-user": "allowed"}
    )
    assert result.status_code == 200
    assert result.json()["first_open_at"] == "2017-08-17T04:00:00Z"
    assert result.json()["state"] == "ready"
    assert client.app.state.probes[0]["symbol"] == "BTCUSDT"
    assert datetime.fromisoformat(result.json()["end_at"]).second == 0


def test_full_metadata_queue_is_an_explicit_failure_not_a_fake_queued_job(catalog_surface):
    client, _, _, _ = catalog_surface
    client.app.state.history_bounds.read_or_request = lambda **_: None
    response = client.get(
        "/market-data/workspace/history-bounds",
        params=dict(market_id=1, symbol="BTCUSDT"),
        headers={"x-user": "allowed"},
    )
    assert response.status_code == 503 and response.headers["Retry-After"] == "30"
