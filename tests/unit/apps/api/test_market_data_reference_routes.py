from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_market_data_reference_router
from trading.contexts.backtest.application.ports import (
    ResearchOrganizationScope,
    ResearchOrganizationScopeResolver,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.market_data.application.dto.reference_api import (
    BTCUSDTMarketReadinessReport,
    BTCUSDTMarketReadinessRow,
    EnabledMarketReference,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    OrganizationId,
    PaidLevel,
    Symbol,
    UserId,
)

# WEB-EPIC-07 mapping:
# - Scope 1: unit tests for auth-only access, deterministic payload mapping,
#   and q/limit semantics for market-data reference endpoints.


class _HeaderCurrentUserDependency:
    """
    Auth dependency stub resolving principal from deterministic request header.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - tests/unit/apps/api/test_market_data_reference_routes.py
      - apps/api/routes/market_data_reference.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
    """

    def __call__(self, request: Request) -> CurrentUserPrincipal:
        """
        Resolve principal from `X-User-Id` header or raise deterministic 401.

        Parameters:
        - request: incoming HTTP request.

        Returns:
        - `CurrentUserPrincipal` for authenticated route handler execution.

        Assumptions/Invariants:
        - Header contains valid UUID string when provided.

        Errors/Exceptions:
        - Raises `HTTPException` 401 for missing header.

        Side effects:
        - None.
        """
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "unauthorized",
                    "message": "Authentication required",
                },
            )
        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel.free(),
        )


class _ScopeResolver:
    def resolve(self, *, user_id: UserId) -> ResearchOrganizationScope:
        return ResearchOrganizationScope(
            organization_id=OrganizationId.from_string(
                "00000000-0000-0000-0000-000000000001"
            ),
            user_id=user_id,
        )


class _AmbiguousScopeResolver:
    def resolve(self, *, user_id: UserId) -> ResearchOrganizationScope:
        _ = user_id
        raise RoehubError(
            code="research.organization_scope_ambiguous",
            message="Research organization scope is ambiguous",
            details={"reason": "multiple_active_memberships"},
        )


@dataclass(frozen=True, slots=True)
class _FakeListEnabledMarketsUseCase:
    """
    Fake use-case returning deterministic enabled markets payload.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - tests/unit/apps/api/test_market_data_reference_routes.py
      - apps/api/routes/market_data_reference.py
      - src/trading/contexts/market_data/application/use_cases/list_enabled_markets.py
    """

    rows: tuple[EnabledMarketReference, ...]

    def execute(self) -> tuple[EnabledMarketReference, ...]:
        """
        Return preconfigured enabled markets rows.

        Parameters:
        - None.

        Returns:
        - Tuple of enabled market rows.

        Assumptions/Invariants:
        - Fixture rows are already validated.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        return self.rows


class _FakeSearchEnabledTradableInstrumentsUseCase:
    """
    Fake search use-case capturing endpoint query argument mappings.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - tests/unit/apps/api/test_market_data_reference_routes.py
      - apps/api/routes/market_data_reference.py
      - src/trading/contexts/market_data/application/use_cases/
        search_enabled_tradable_instruments.py
    """

    def __init__(
        self,
        *,
        rows_by_market: dict[int, tuple[InstrumentId, ...]],
    ) -> None:
        """
        Store deterministic response rows keyed by market id.

        Parameters:
        - rows_by_market: mapping `market_id -> instrument rows`.

        Returns:
        - None.

        Assumptions/Invariants:
        - Unknown market ids map to empty tuple.

        Errors/Exceptions:
        - None.

        Side effects:
        - Initializes call-capture list.
        """
        self._rows_by_market = rows_by_market
        self.calls: list[tuple[int, str | None, int | None]] = []

    def execute(
        self,
        *,
        market_id: MarketId,
        q: str | None = None,
        limit: int | None = None,
    ) -> tuple[InstrumentId, ...]:
        """
        Capture incoming arguments and return configured market rows.

        Parameters:
        - market_id: requested market id.
        - q: optional symbol prefix from query string.
        - limit: optional limit value from query string.

        Returns:
        - Tuple of configured instrument ids.

        Assumptions/Invariants:
        - Route validation already enforces valid integer limit range.

        Errors/Exceptions:
        - None.

        Side effects:
        - Appends call details into `calls`.
        """
        self.calls.append((market_id.value, q, limit))
        return self._rows_by_market.get(market_id.value, ())


class _FakeBTCUSDTMarketReadinessUseCase:
    def __init__(self, *, report: BTCUSDTMarketReadinessReport) -> None:
        self._report = report
        self.calls = 0

    def execute(self) -> BTCUSDTMarketReadinessReport:
        self.calls += 1
        return self._report


def test_get_market_data_markets_returns_enabled_items() -> None:
    """
    Verify `/market-data/markets` returns expected payload for authenticated user.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Route maps use-case rows directly into response DTO.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    client, _ = _build_client(
        list_use_case=_FakeListEnabledMarketsUseCase(
            rows=(
                _market(1, "binance", "spot", "binance:spot"),
                _market(3, "bybit", "spot", "bybit:spot"),
            )
        ),
        search_use_case=_FakeSearchEnabledTradableInstrumentsUseCase(rows_by_market={}),
    )

    response = client.get(
        "/market-data/markets",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000101"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "market_id": 1,
                "exchange_name": "binance",
                "market_type": "spot",
                "market_code": "binance:spot",
            },
            {
                "market_id": 3,
                "exchange_name": "bybit",
                "market_type": "spot",
                "market_code": "bybit:spot",
            },
        ]
    }


def test_get_market_data_instruments_uses_default_limit_and_max_limit() -> None:
    """
    Verify instruments endpoint applies default limit and accepts max limit value.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - FastAPI query defaults set `limit=50`.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    search_use_case = _FakeSearchEnabledTradableInstrumentsUseCase(
        rows_by_market={1: (_instrument(1, "BTCUSDT"),)}
    )
    client, _ = _build_client(
        list_use_case=_FakeListEnabledMarketsUseCase(rows=()),
        search_use_case=search_use_case,
    )
    headers = {"x-user-id": "00000000-0000-0000-0000-000000000102"}

    response_default = client.get(
        "/market-data/instruments",
        params={"market_id": 1},
        headers=headers,
    )
    response_max = client.get(
        "/market-data/instruments",
        params={"market_id": 1, "limit": 200},
        headers=headers,
    )

    assert response_default.status_code == 200
    assert response_max.status_code == 200
    assert search_use_case.calls[0] == (1, None, 50)
    assert search_use_case.calls[1] == (1, None, 200)


def test_get_market_data_instruments_returns_empty_for_unknown_market() -> None:
    """
    Verify unknown/disabled market id is represented as `200 {"items": []}`.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Use-case returns empty tuple for unknown market.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    search_use_case = _FakeSearchEnabledTradableInstrumentsUseCase(
        rows_by_market={1: (_instrument(1, "BTCUSDT"),)}
    )
    client, _ = _build_client(
        list_use_case=_FakeListEnabledMarketsUseCase(rows=()),
        search_use_case=search_use_case,
    )

    response = client.get(
        "/market-data/instruments",
        params={"market_id": 999, "q": "BTC"},
        headers={"x-user-id": "00000000-0000-0000-0000-000000000103"},
    )

    assert response.status_code == 200
    assert response.json() == {"items": []}
    assert search_use_case.calls == [(999, "BTC", 50)]


def test_get_market_data_instruments_maps_payload_and_forwards_q_limit() -> None:
    """
    Verify instruments endpoint maps payload deterministically and forwards q/limit.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Route forwards raw query values to use-case and maps `InstrumentId` rows into API DTO.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    search_use_case = _FakeSearchEnabledTradableInstrumentsUseCase(
        rows_by_market={
            1: (
                _instrument(1, "BTCUSDT"),
                _instrument(1, "ETHUSDT"),
            )
        }
    )
    client, _ = _build_client(
        list_use_case=_FakeListEnabledMarketsUseCase(rows=()),
        search_use_case=search_use_case,
    )

    response = client.get(
        "/market-data/instruments",
        params={"market_id": 1, "q": "ETH", "limit": 2},
        headers={"x-user-id": "00000000-0000-0000-0000-000000000105"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {"market_id": 1, "symbol": "BTCUSDT"},
            {"market_id": 1, "symbol": "ETHUSDT"},
        ]
    }
    assert search_use_case.calls == [(1, "ETH", 2)]


def test_get_market_data_instruments_rejects_limit_above_max_with_422() -> None:
    """
    Verify invalid `limit` query parameter produces deterministic validation payload.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - API error handler maps FastAPI validation errors to Roehub 422 shape.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    client, _ = _build_client(
        list_use_case=_FakeListEnabledMarketsUseCase(rows=()),
        search_use_case=_FakeSearchEnabledTradableInstrumentsUseCase(rows_by_market={}),
    )

    response = client.get(
        "/market-data/instruments",
        params={"market_id": 1, "limit": 201},
        headers={"x-user-id": "00000000-0000-0000-0000-000000000104"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Validation failed",
            "details": {
                "errors": [
                    {
                        "path": "query.limit",
                        "code": "less_than_equal",
                        "message": "Input should be less than or equal to 200",
                    }
                ]
            },
        }
    }


def test_get_btcusdt_market_readiness_returns_reference_and_stream_matrix() -> None:
    readiness_use_case = _FakeBTCUSDTMarketReadinessUseCase(
        report=_readiness_report(
            row=_readiness_row(
                market_id=1,
                exchange_name="binance",
                market_type="spot",
                market_code="binance:spot",
                readiness_state="ready",
                reason_codes=("btcusdt_market_ready",),
                reference_state="ready",
                reference_reason_codes=("reference_ready",),
                stream_state="ready",
                stream_reason_code="market_data_stream_ready",
            )
        )
    )
    client, _ = _build_client(
        list_use_case=_FakeListEnabledMarketsUseCase(rows=()),
        search_use_case=_FakeSearchEnabledTradableInstrumentsUseCase(rows_by_market={}),
        btcusdt_readiness_use_case=readiness_use_case,
    )

    response = client.get(
        "/market-data/btcusdt-readiness",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000106"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert readiness_use_case.calls == 1
    assert payload["symbol"] == "BTCUSDT"
    assert payload["freshness_threshold_seconds"] == 180
    assert payload["items"][0]["instrument_key"] == "binance:spot:BTCUSDT"
    assert payload["items"][0]["readiness_state"] == "ready"
    assert payload["items"][0]["reason_codes"] == ["btcusdt_market_ready"]
    assert payload["items"][0]["price_step"] == 0.01
    assert payload["items"][0]["qty_step"] == 0.00001
    assert payload["items"][0]["min_notional"] == 10.0
    assert payload["items"][0]["stream_name"] == "md.candles.1m.binance:spot:BTCUSDT"


def test_market_data_reference_routes_require_authentication() -> None:
    """
    Verify both reference endpoints are auth-protected and return deterministic 401.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Header-based auth stub is representative for route guard behavior.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    client, _ = _build_client(
        list_use_case=_FakeListEnabledMarketsUseCase(rows=()),
        search_use_case=_FakeSearchEnabledTradableInstrumentsUseCase(rows_by_market={}),
    )

    markets_response = client.get("/market-data/markets")
    instruments_response = client.get(
        "/market-data/instruments",
        params={"market_id": 1},
    )
    readiness_response = client.get("/market-data/btcusdt-readiness")

    assert markets_response.status_code == 401
    assert instruments_response.status_code == 401
    assert readiness_response.status_code == 401
    assert markets_response.json() == {
        "detail": {
            "error": "unauthorized",
            "message": "Authentication required",
        }
    }
    assert instruments_response.json() == {
        "detail": {
            "error": "unauthorized",
            "message": "Authentication required",
        }
    }
    assert readiness_response.json() == {
        "detail": {
            "error": "unauthorized",
            "message": "Authentication required",
        }
    }


def _build_client(
    *,
    list_use_case: _FakeListEnabledMarketsUseCase,
    search_use_case: _FakeSearchEnabledTradableInstrumentsUseCase,
    btcusdt_readiness_use_case: _FakeBTCUSDTMarketReadinessUseCase | None = None,
    organization_scope_resolver: ResearchOrganizationScopeResolver | None = _ScopeResolver(),
) -> tuple[TestClient, _FakeSearchEnabledTradableInstrumentsUseCase]:
    """
    Build FastAPI test client with market-data reference router and shared error handlers.

    Parameters:
    - list_use_case: fake enabled markets use-case.
    - search_use_case: fake search use-case.

    Returns:
    - Tuple `(client, search_use_case)` for request and call-capture assertions.

    Assumptions/Invariants:
    - Route builder accepts duck-typed use-case fakes with `execute` methods.

    Errors/Exceptions:
    - Propagates route build-time `ValueError` on invalid dependencies.

    Side effects:
    - Creates new in-memory FastAPI app instance.
    """
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_market_data_reference_router(
            list_enabled_markets_use_case=list_use_case,  # type: ignore[arg-type]
            search_enabled_tradable_instruments_use_case=search_use_case,  # type: ignore[arg-type]
            btcusdt_market_readiness_use_case=(
                btcusdt_readiness_use_case or _FakeBTCUSDTMarketReadinessUseCase(
                    report=_readiness_report(
                        row=_readiness_row(
                            market_id=None,
                            exchange_name="binance",
                            market_type="spot",
                            market_code="binance:spot",
                            readiness_state="blocked",
                            reason_codes=("reference_market_missing",),
                            reference_state="missing",
                            reference_reason_codes=("reference_market_missing",),
                            stream_state="pending",
                            stream_reason_code="market_data_readiness_reader_unavailable",
                        )
                    )
                )
            ),  # type: ignore[arg-type]
            current_user_dependency=_HeaderCurrentUserDependency(),  # type: ignore[arg-type]
            organization_scope_resolver=organization_scope_resolver,
        )
    )
    return TestClient(app), search_use_case


@pytest.mark.parametrize(
    ("scope_resolver", "expected_status", "expected_code"),
    [
        (None, 503, "research.organization_scope_unavailable"),
        (
            _AmbiguousScopeResolver(),
            409,
            "research.organization_scope_ambiguous",
        ),
    ],
)
def test_market_data_reference_fails_closed_before_clickhouse_read(
    scope_resolver: ResearchOrganizationScopeResolver | None,
    expected_status: int,
    expected_code: str,
) -> None:
    search_use_case = _FakeSearchEnabledTradableInstrumentsUseCase(rows_by_market={})
    client, _ = _build_client(
        list_use_case=_FakeListEnabledMarketsUseCase(rows=()),
        search_use_case=search_use_case,
        organization_scope_resolver=scope_resolver,
    )

    response = client.get(
        "/market-data/instruments?market_id=1",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000001"},
    )

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert search_use_case.calls == []


def _readiness_report(
    *,
    row: BTCUSDTMarketReadinessRow,
) -> BTCUSDTMarketReadinessReport:
    checked_at = datetime(2027, 1, 15, 8, 0, tzinfo=UTC)
    return BTCUSDTMarketReadinessReport(
        symbol="BTCUSDT",
        freshness_threshold_seconds=180,
        rows=(row,),
        checked_at=checked_at,
    )


def _readiness_row(
    *,
    market_id: int | None,
    exchange_name: str,
    market_type: str,
    market_code: str,
    readiness_state: str,
    reason_codes: tuple[str, ...],
    reference_state: str,
    reference_reason_codes: tuple[str, ...],
    stream_state: str,
    stream_reason_code: str,
) -> BTCUSDTMarketReadinessRow:
    checked_at = datetime(2027, 1, 15, 8, 0, tzinfo=UTC)
    return BTCUSDTMarketReadinessRow(
        market_id=MarketId(market_id) if market_id is not None else None,
        exchange_name=exchange_name,
        market_type=market_type,
        market_code=market_code,
        symbol="BTCUSDT",
        instrument_key=f"{exchange_name}:{market_type}:BTCUSDT",
        readiness_state=readiness_state,  # type: ignore[arg-type]
        reason_codes=reason_codes,
        reference_state=reference_state,  # type: ignore[arg-type]
        reference_reason_codes=reference_reason_codes,
        market_enabled=market_id is not None,
        status="ENABLED" if market_id is not None else None,
        is_tradable=1 if market_id is not None else None,
        base_asset="BTC" if market_id is not None else None,
        quote_asset="USDT" if market_id is not None else None,
        price_step=0.01 if market_id is not None else None,
        qty_step=0.00001 if market_id is not None else None,
        min_notional=10.0 if market_id is not None else None,
        stream_state=stream_state,  # type: ignore[arg-type]
        stream_reason_code=stream_reason_code,
        stream_name=f"md.candles.1m.{exchange_name}:{market_type}:BTCUSDT",
        stream_length=12 if stream_state == "ready" else None,
        stream_last_message_id="1800000000000-0" if stream_state == "ready" else None,
        stream_last_observed_at=checked_at if stream_state == "ready" else None,
        stream_age_seconds=30 if stream_state == "ready" else None,
        checked_at=checked_at,
    )


def _market(
    market_id: int,
    exchange_name: str,
    market_type: str,
    market_code: str,
) -> EnabledMarketReference:
    """
    Build enabled market fixture row.

    Parameters:
    - market_id: market identifier.
    - exchange_name: exchange literal.
    - market_type: market type literal.
    - market_code: market code literal.

    Returns:
    - Enabled market read-model fixture.

    Assumptions/Invariants:
    - Input values satisfy dataclass invariants.

    Errors/Exceptions:
    - Propagates constructor validation errors.

    Side effects:
    - None.
    """
    return EnabledMarketReference(
        market_id=MarketId(market_id),
        exchange_name=exchange_name,
        market_type=market_type,
        market_code=market_code,
    )


def _instrument(market_id: int, symbol: str) -> InstrumentId:
    """
    Build instrument id fixture row.

    Parameters:
    - market_id: market identifier.
    - symbol: instrument symbol string.

    Returns:
    - Instrument id fixture object.

    Assumptions/Invariants:
    - Symbol normalization is handled by shared-kernel primitive.

    Errors/Exceptions:
    - Propagates constructor validation errors.

    Side effects:
    - None.
    """
    return InstrumentId(market_id=MarketId(market_id), symbol=Symbol(symbol))
