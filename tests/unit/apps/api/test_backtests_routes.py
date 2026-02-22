from dataclasses import dataclass
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.dto import build_sha256_from_payload
from apps.api.routes import build_backtests_router
from trading.contexts.backtest.application.dto import (
    BacktestMetricRowV1,
    BacktestReportV1,
    BacktestVariantPayloadV1,
    BacktestVariantPreview,
    RunBacktestResponse,
)
from trading.contexts.backtest.application.ports import BacktestStrategySnapshot
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    PaidLevel,
    Symbol,
    Timeframe,
    UserId,
)


class _HeaderCurrentUserDependency:
    """
    Request dependency resolving authenticated principal from `X-User-Id` header.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - tests/unit/apps/api/test_backtests_routes.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
    """

    def __call__(self, request: Request):
        """
        Resolve principal or raise deterministic HTTP 401 payload.

        Args:
            request: HTTP request object.
        Returns:
            object: CurrentUserPrincipal-compatible object.
        Assumptions:
            Header contains UUID string when provided.
        Raises:
            HTTPException: If authentication header is missing.
        Side Effects:
            None.
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

        from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal

        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel.free(),
        )


@dataclass(frozen=True, slots=True)
class _StaticStrategyReader:
    """
    Minimal strategy reader fake returning one preconfigured snapshot.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - tests/unit/apps/api/test_backtests_routes.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
    """

    snapshot: BacktestStrategySnapshot | None = None

    def load_any(self, *, strategy_id: UUID) -> BacktestStrategySnapshot | None:
        """
        Return preconfigured snapshot independent from requested strategy id.

        Args:
            strategy_id: Requested strategy identifier.
        Returns:
            BacktestStrategySnapshot | None: Configured snapshot value.
        Assumptions:
            Tests verify route behavior, not repository lookup semantics.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = strategy_id
        return self.snapshot


class _FakeRunBacktestUseCase:
    """
    Minimal use-case fake returning preconfigured result or raising configured error.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - tests/unit/apps/api/test_backtests_routes.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    def __init__(self, *, result: Any = None, error: Exception | None = None) -> None:
        """
        Store deterministic fake behavior for endpoint tests.

        Args:
            result: Value returned by execute when no error is configured.
            error: Optional exception raised by execute.
        Returns:
            None.
        Assumptions:
            Endpoint tests inspect only routing/mapping/error behavior.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._result = result
        self._error = error

    def execute(self, *, request, current_user):
        """
        Return configured result or raise configured error.

        Args:
            request: Application request DTO.
            current_user: Current user port object.
        Returns:
            Any: Configured result payload.
        Assumptions:
            Request/current_user are ignored in route-contract unit tests.
        Raises:
            Exception: Configured exception.
        Side Effects:
            None.
        """
        _ = request, current_user
        if self._error is not None:
            raise self._error
        return self._result


def _build_client(
    *,
    use_case: _FakeRunBacktestUseCase,
    strategy_reader: _StaticStrategyReader | None = None,
) -> TestClient:
    """
    Build minimal FastAPI TestClient with backtests router and shared error handlers.

    Args:
        use_case: Fake use-case used by endpoint handler.
        strategy_reader: Optional strategy reader fake.
    Returns:
        TestClient: Configured client instance.
    Assumptions:
        Shared API error handlers provide deterministic Roehub/422 payloads.
    Raises:
        ValueError: If router dependencies are invalid.
    Side Effects:
        None.
    """
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_backtests_router(
            run_use_case=use_case,  # type: ignore[arg-type]
            strategy_reader=strategy_reader or _StaticStrategyReader(),
            current_user_dependency=_HeaderCurrentUserDependency(),
        )
    )
    return TestClient(app)


def test_post_backtests_forbids_extra_fields_with_deterministic_422_payload() -> None:
    """
    Verify strict request DTO rejects extra fields with deterministic validation payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Pydantic `extra=forbid` is enabled for `BacktestsPostRequest` models.
    Raises:
        AssertionError: If payload or status code deviates from deterministic contract.
    Side Effects:
        None.
    """
    client = _build_client(use_case=_FakeRunBacktestUseCase(result=_template_mode_response()))

    payload = {
        "time_range": {
            "start": "2026-02-16T00:00:00Z",
            "end": "2026-02-16T01:00:00Z",
        },
        "template": _template_payload(),
        "unexpected_field": "boom",
    }
    response = client.post(
        "/backtests",
        json=payload,
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Validation failed",
            "details": {
                "errors": [
                    {
                        "path": "body.unexpected_field",
                        "code": "extra_forbidden",
                        "message": "Extra inputs are not permitted",
                    }
                ]
            },
        }
    }


def test_post_backtests_rejects_mode_conflict_with_deterministic_validation_error() -> None:
    """
    Verify route returns deterministic `validation_error` for `strategy_id xor template` conflict.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Mode exclusivity is validated by API->application request mapper.
    Raises:
        AssertionError: If payload or status code deviates from deterministic contract.
    Side Effects:
        None.
    """
    client = _build_client(use_case=_FakeRunBacktestUseCase(result=_template_mode_response()))

    payload = {
        "time_range": {
            "start": "2026-02-16T00:00:00Z",
            "end": "2026-02-16T01:00:00Z",
        },
        "strategy_id": "00000000-0000-0000-0000-000000000123",
        "template": _template_payload(),
    }
    response = client.post(
        "/backtests",
        json=payload,
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "POST /backtests requires exactly one mode: strategy_id xor template",
            "details": {
                "errors": [
                    {
                        "path": "body.strategy_id",
                        "code": "mode_conflict",
                        "message": "Provide exactly one of strategy_id or template",
                    },
                    {
                        "path": "body.template",
                        "code": "mode_conflict",
                        "message": "Provide exactly one of strategy_id or template",
                    },
                ]
            },
        }
    }


def test_post_backtests_returns_401_when_unauthenticated() -> None:
    """
    Verify endpoint is protected by identity dependency and returns deterministic 401 payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Authentication dependency requires `X-User-Id` header in route tests.
    Raises:
        AssertionError: If status code or payload deviates from expected unauthorized contract.
    Side Effects:
        None.
    """
    client = _build_client(use_case=_FakeRunBacktestUseCase(result=_template_mode_response()))

    response = client.post(
        "/backtests",
        json={
            "time_range": {
                "start": "2026-02-16T00:00:00Z",
                "end": "2026-02-16T01:00:00Z",
            },
            "template": _template_payload(),
        },
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "error": "unauthorized",
            "message": "Authentication required",
        }
    }


def test_post_backtests_maps_saved_mode_forbidden_error() -> None:
    """
    Verify saved-mode ownership failure is mapped to deterministic `forbidden` payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Backtest use-case is source of truth for ownership policy.
    Raises:
        AssertionError: If status code or payload differs from Roehub contract.
    Side Effects:
        None.
    """
    client = _build_client(
        use_case=_FakeRunBacktestUseCase(
            error=RoehubError(
                code="forbidden",
                message="Backtest strategy does not belong to current user",
                details={"strategy_id": "00000000-0000-0000-0000-000000000123"},
            )
        )
    )

    response = client.post(
        "/backtests",
        json=_saved_mode_payload(),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "error": {
            "code": "forbidden",
            "message": "Backtest strategy does not belong to current user",
            "details": {"strategy_id": "00000000-0000-0000-0000-000000000123"},
        }
    }


def test_post_backtests_maps_saved_mode_not_found_error() -> None:
    """
    Verify saved-mode missing strategy is mapped to deterministic `not_found` payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Backtest use-case emits Roehub `not_found` for missing/deleted strategy.
    Raises:
        AssertionError: If status code or payload differs from Roehub contract.
    Side Effects:
        None.
    """
    client = _build_client(
        use_case=_FakeRunBacktestUseCase(
            error=RoehubError(
                code="not_found",
                message="Backtest strategy was not found",
                details={"strategy_id": "00000000-0000-0000-0000-000000000123"},
            )
        )
    )

    response = client.post(
        "/backtests",
        json=_saved_mode_payload(),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "not_found",
            "message": "Backtest strategy was not found",
            "details": {"strategy_id": "00000000-0000-0000-0000-000000000123"},
        }
    }


def test_post_backtests_saved_response_includes_hashes_and_explicit_payload() -> None:
    """
    Verify saved-mode response includes `spec_hash`, `engine_params_hash`, and payload blocks.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Route computes hashes from canonical JSON payloads.
    Raises:
        AssertionError: If payload misses required fields or hashes.
    Side Effects:
        None.
    """
    snapshot_payload = {
        "schema_version": 1,
        "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
        "timeframe": "1m",
        "indicators": [
            {"id": "ma.sma", "inputs": {"source": "close"}, "params": {"window": 20}}
        ],
    }
    strategy_snapshot = BacktestStrategySnapshot(
        strategy_id=UUID("00000000-0000-0000-0000-000000000123"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000777"),
        is_deleted=False,
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={"window": ExplicitValuesSpec(name="window", values=(20,))},
            ),
        ),
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ma.sma",
                inputs={"source": "close"},
                params={"window": 20},
            ),
        ),
        spec_payload=snapshot_payload,
    )

    client = _build_client(
        use_case=_FakeRunBacktestUseCase(result=_saved_mode_response()),
        strategy_reader=_StaticStrategyReader(snapshot=strategy_snapshot),
    )

    response = client.post(
        "/backtests",
        json=_saved_mode_payload(),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["spec_hash"] == build_sha256_from_payload(payload=snapshot_payload)
    assert body["grid_request_hash"] is None
    assert isinstance(body["engine_params_hash"], str)
    assert len(body["engine_params_hash"]) == 64
    assert body["variants"][0]["payload"] == {
        "indicator_selections": [
            {
                "indicator_id": "ma.sma",
                "inputs": {"source": "close"},
                "params": {"window": 20},
            }
        ],
        "signal_params": {"ma.sma": {"cross_up": 0.5}},
        "risk_params": {
            "sl_enabled": True,
            "sl_pct": 2.0,
            "tp_enabled": True,
            "tp_pct": 4.0,
        },
        "execution_params": {
            "fee_pct": 0.075,
            "fixed_quote": 100.0,
            "init_cash_quote": 10000.0,
            "safe_profit_percent": 30.0,
            "slippage_pct": 0.01,
        },
        "direction_mode": "long-short",
        "sizing_mode": "all_in",
    }


def test_post_backtests_sorts_variants_deterministically_for_equal_returns() -> None:
    """
    Verify route output preserves deterministic tie-break sorting by `variant_key` asc.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Route defensively sorts variants by `(total_return_pct desc, variant_key asc)`.
    Raises:
        AssertionError: If response order breaks deterministic tie-break contract.
    Side Effects:
        None.
    """
    client = _build_client(
        use_case=_FakeRunBacktestUseCase(result=_loose_unsorted_template_response())
    )

    response = client.post(
        "/backtests",
        json={
            "time_range": {
                "start": "2026-02-16T00:00:00Z",
                "end": "2026-02-16T01:00:00Z",
            },
            "template": _template_payload(),
        },
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 200
    variant_keys = [item["variant_key"] for item in response.json()["variants"]]
    assert variant_keys == ["a" * 64, "b" * 64]


def _template_payload() -> dict[str, Any]:
    """
    Build minimal valid ad-hoc template payload for API route tests.

    Args:
        None.
    Returns:
        dict[str, Any]: Template request JSON payload.
    Assumptions:
        One indicator grid is sufficient for endpoint contract tests.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "instrument_id": {
            "market_id": 1,
            "symbol": "BTCUSDT",
        },
        "timeframe": "1m",
        "indicator_grids": [
            {
                "indicator_id": "ma.sma",
                "params": {
                    "window": {"mode": "explicit", "values": [20]},
                },
            }
        ],
        "signal_grids": {
            "ma.sma": {
                "cross_up": {"mode": "explicit", "values": [0.5]},
            }
        },
        "risk_grid": {
            "sl_enabled": True,
            "tp_enabled": True,
            "sl": {"mode": "explicit", "values": [2.0]},
            "tp": {"mode": "explicit", "values": [4.0]},
        },
        "execution": {
            "init_cash_quote": 10000,
            "fee_pct": 0.075,
            "slippage_pct": 0.01,
            "fixed_quote": 100,
            "safe_profit_percent": 30,
        },
    }


def _saved_mode_payload() -> dict[str, Any]:
    """
    Build minimal valid saved-mode payload for route tests.

    Args:
        None.
    Returns:
        dict[str, Any]: Saved-mode request JSON payload.
    Assumptions:
        `strategy_id` mode is enough for mapping/error tests.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "time_range": {
            "start": "2026-02-16T00:00:00Z",
            "end": "2026-02-16T01:00:00Z",
        },
        "strategy_id": "00000000-0000-0000-0000-000000000123",
    }


def _variant(
    *,
    variant_index: int,
    variant_key: str,
    total_return_pct: float,
) -> BacktestVariantPreview:
    """
    Build deterministic variant preview fixture for route mapping tests.

    Args:
        variant_index: Deterministic variant index.
        variant_key: Deterministic variant key (64 hex characters).
        total_return_pct: Ranking metric value.
    Returns:
        BacktestVariantPreview: Variant fixture.
    Assumptions:
        Payload contains explicit saveable indicator/signal/risk/execution values.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return BacktestVariantPreview(
        variant_index=variant_index,
        variant_key=variant_key,
        indicator_variant_key="1" * 64,
        total_return_pct=total_return_pct,
        payload=BacktestVariantPayloadV1(
            indicator_selections=(
                IndicatorVariantSelection(
                    indicator_id="ma.sma",
                    inputs={"source": "close"},
                    params={"window": 20},
                ),
            ),
            signal_params={"ma.sma": {"cross_up": 0.5}},
            risk_params={
                "sl_enabled": True,
                "sl_pct": 2.0,
                "tp_enabled": True,
                "tp_pct": 4.0,
            },
            execution_params={
                "init_cash_quote": 10000.0,
                "fee_pct": 0.075,
                "slippage_pct": 0.01,
                "fixed_quote": 100.0,
                "safe_profit_percent": 30.0,
            },
            direction_mode="long-short",
            sizing_mode="all_in",
        ),
        report=BacktestReportV1(
            rows=(BacktestMetricRowV1(metric="Total Return [%]", value=f"{total_return_pct:.2f}"),),
            table_md="|Metric|Value|\n|---|---|\n|Total Return [%]|1.00|",
        ),
    )


def _saved_mode_response() -> RunBacktestResponse:
    """
    Build deterministic saved-mode response fixture.

    Args:
        None.
    Returns:
        RunBacktestResponse: Saved-mode use-case response fixture.
    Assumptions:
        Response is already sorted by ranking contract.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestResponse(
        mode="saved",
        strategy_id=UUID("00000000-0000-0000-0000-000000000123"),
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        warmup_bars=200,
        top_k=2,
        preselect=100,
        top_trades_n=1,
        variants=(
            _variant(variant_index=0, variant_key="a" * 64, total_return_pct=12.0),
            _variant(variant_index=1, variant_key="b" * 64, total_return_pct=10.0),
        ),
        total_indicator_compute_calls=1,
    )


def _template_mode_response() -> RunBacktestResponse:
    """
    Build deterministic template-mode response fixture.

    Args:
        None.
    Returns:
        RunBacktestResponse: Template-mode use-case response fixture.
    Assumptions:
        Response is already sorted by ranking contract.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestResponse(
        mode="template",
        strategy_id=None,
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        warmup_bars=200,
        top_k=2,
        preselect=100,
        top_trades_n=1,
        variants=(
            _variant(variant_index=0, variant_key="a" * 64, total_return_pct=12.0),
        ),
        total_indicator_compute_calls=1,
    )


class _LooseRunBacktestResponse:
    """
    Loose response object for testing route-level defensive variant sorting.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - tests/unit/apps/api/test_backtests_routes.py
      - apps/api/dto/backtests.py
      - apps/api/routes/backtests.py
    """

    def __init__(self) -> None:
        """
        Build unsorted tie-case response payload for route sorting tests.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Attributes match subset of `RunBacktestResponse` used by response mapper.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.mode = "template"
        self.strategy_id = None
        self.instrument_id = InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT"))
        self.timeframe = Timeframe("1m")
        self.warmup_bars = 200
        self.top_k = 2
        self.preselect = 100
        self.top_trades_n = 1
        self.variants = (
            _variant(variant_index=1, variant_key="b" * 64, total_return_pct=12.0),
            _variant(variant_index=0, variant_key="a" * 64, total_return_pct=12.0),
        )
        self.total_indicator_compute_calls = 1


def _loose_unsorted_template_response() -> _LooseRunBacktestResponse:
    """
    Build unsorted tie-case response fixture for deterministic output-order test.

    Args:
        None.
    Returns:
        _LooseRunBacktestResponse: Fixture object with unsorted variants.
    Assumptions:
        Route re-sorts variants by deterministic key contract.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _LooseRunBacktestResponse()
