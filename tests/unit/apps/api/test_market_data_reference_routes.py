from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_market_data_reference_router
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.market_data.application.dto.reference_api import EnabledMarketReference
from trading.shared_kernel.primitives import InstrumentId, MarketId, PaidLevel, Symbol, UserId


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
    client, _ = _build_client(
        list_use_case=_FakeListEnabledMarketsUseCase(rows=()),
        search_use_case=_FakeSearchEnabledTradableInstrumentsUseCase(
            rows_by_market={1: (_instrument(1, "BTCUSDT"),)}
        ),
    )

    response = client.get(
        "/market-data/instruments",
        params={"market_id": 999, "q": "BTC"},
        headers={"x-user-id": "00000000-0000-0000-0000-000000000103"},
    )

    assert response.status_code == 200
    assert response.json() == {"items": []}


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

    assert markets_response.status_code == 401
    assert instruments_response.status_code == 401
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


def _build_client(
    *,
    list_use_case: _FakeListEnabledMarketsUseCase,
    search_use_case: _FakeSearchEnabledTradableInstrumentsUseCase,
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
            current_user_dependency=_HeaderCurrentUserDependency(),  # type: ignore[arg-type]
        )
    )
    return TestClient(app), search_use_case


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
