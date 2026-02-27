from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes import build_indicators_router
from trading.contexts.indicators.adapters.outbound.compute_numba import NumbaIndicatorCompute
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    EstimateResult,
    IndicatorTensor,
)
from trading.contexts.indicators.application.dto.registry_view import MergedIndicatorView
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.ports.feeds import CandleFeed
from trading.contexts.indicators.application.ports.registry import IndicatorRegistry
from trading.contexts.indicators.domain.definitions import all_defs
from trading.contexts.indicators.domain.entities import IndicatorDef, IndicatorId
from trading.contexts.indicators.domain.errors import UnknownIndicatorError
from trading.contexts.indicators.domain.specifications import GridSpec
from trading.platform.config import IndicatorsComputeNumbaConfig
from trading.shared_kernel.primitives import (
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)


class _RegistryStub(IndicatorRegistry):
    """
    Deterministic indicator registry stub for API compute endpoint tests.
    """

    def __init__(self, *, defs: tuple[IndicatorDef, ...]) -> None:
        """
        Build immutable lookup map from provided hard definitions.

        Args:
            defs: Hard indicator definitions.
        Returns:
            None.
        Assumptions:
            Indicator ids are unique across provided definitions.
        Raises:
            ValueError: If duplicate indicator id is detected.
        Side Effects:
            None.
        """
        defs_by_id: dict[str, IndicatorDef] = {}
        for definition in defs:
            key = definition.indicator_id.value
            if key in defs_by_id:
                raise ValueError(f"duplicate indicator_id: {key}")
            defs_by_id[key] = definition
        self._defs = defs
        self._defs_by_id = defs_by_id

    def list_defs(self) -> tuple[IndicatorDef, ...]:
        """
        Return hard definitions in deterministic order.

        Args:
            None.
        Returns:
            tuple[IndicatorDef, ...]: Hard definitions snapshot.
        Assumptions:
            Snapshot is immutable for test runtime.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._defs

    def get_def(self, indicator_id: IndicatorId) -> IndicatorDef:
        """
        Resolve one definition by id.

        Args:
            indicator_id: Target indicator identifier.
        Returns:
            IndicatorDef: Matching hard definition.
        Assumptions:
            Indicator id normalization is enforced by value object.
        Raises:
            UnknownIndicatorError: If id is absent.
        Side Effects:
            None.
        """
        definition = self._defs_by_id.get(indicator_id.value)
        if definition is None:
            raise UnknownIndicatorError(f"unknown indicator_id: {indicator_id.value}")
        return definition

    def list_merged(self) -> tuple[MergedIndicatorView, ...]:
        """
        Return empty merged view for protocol completeness in these tests.

        Args:
            None.
        Returns:
            tuple[MergedIndicatorView, ...]: Empty placeholder tuple.
        Assumptions:
            Compute endpoint tests do not require merged registry payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        return ()

    def get_merged(self, indicator_id: IndicatorId) -> MergedIndicatorView:
        """
        Always raise because merged item resolution is not used in these tests.

        Args:
            indicator_id: Target indicator identifier.
        Returns:
            MergedIndicatorView: Never returns.
        Assumptions:
            Merged-view contract is irrelevant for compute endpoint tests.
        Raises:
            UnknownIndicatorError: Always.
        Side Effects:
            None.
        """
        raise UnknownIndicatorError(f"unknown indicator_id: {indicator_id.value}")


class _CandleFeedStub(CandleFeed):
    """
    Deterministic CandleFeed stub with call-tracking for API tests.
    """

    def __init__(self, *, candles: CandleArrays) -> None:
        """
        Store fixed dense candles payload for all requests.

        Args:
            candles: Dense candles payload returned on each load call.
        Returns:
            None.
        Assumptions:
            Payload satisfies CandleArrays invariants.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._candles = candles
        self.calls: list[tuple[MarketId, Symbol, TimeRange]] = []

    def load_1m_dense(
        self,
        market_id: MarketId,
        symbol: Symbol,
        time_range: TimeRange,
    ) -> CandleArrays:
        """
        Return fixed candles payload and record invocation arguments.

        Args:
            market_id: Requested market id.
            symbol: Requested symbol.
            time_range: Requested time interval.
        Returns:
            CandleArrays: Preconfigured dense payload.
        Assumptions:
            Stub does not mutate returned arrays.
        Raises:
            None.
        Side Effects:
            Appends one tuple to `calls` history.
        """
        self.calls.append((market_id, symbol, time_range))
        return self._candles


class _ComputeSpy(IndicatorCompute):
    """
    Compute port spy ensuring `POST /indicators/compute` avoids adapter `estimate`.
    """

    def __init__(self, *, delegate: IndicatorCompute) -> None:
        """
        Store delegated compute adapter and initialize call counters.

        Args:
            delegate: Real compute adapter used for `compute` and `warmup`.
        Returns:
            None.
        Assumptions:
            Delegate satisfies `IndicatorCompute` contract.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._delegate = delegate
        self.estimate_calls = 0
        self.compute_calls = 0

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Fail test if router calls compute adapter estimate preflight.

        Args:
            grid: Requested compute grid.
            max_variants_guard: Variants guard value.
        Returns:
            EstimateResult: Never returns.
        Assumptions:
            Single-preflight router flow should not call this method.
        Raises:
            AssertionError: Always, because estimate path must stay unused in route.
        Side Effects:
            Increments `estimate_calls` counter.
        """
        _ = (grid, max_variants_guard)
        self.estimate_calls += 1
        raise AssertionError("route must not call IndicatorCompute.estimate preflight")

    def compute(self, req: ComputeRequest) -> IndicatorTensor:
        """
        Delegate actual compute call and record invocation count.

        Args:
            req: Compute request DTO.
        Returns:
            IndicatorTensor: Delegate compute result.
        Assumptions:
            Delegate compute behavior is already covered by dedicated engine tests.
        Raises:
            Any exception raised by delegated compute adapter.
        Side Effects:
            Increments `compute_calls` counter.
        """
        self.compute_calls += 1
        return self._delegate.compute(req)

    def warmup(self) -> None:
        """
        Delegate warmup for protocol completeness.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup behavior is delegated unchanged.
        Raises:
            Any exception raised by delegated warmup implementation.
        Side Effects:
            Triggers delegate warmup when called.
        """
        self._delegate.warmup()


def _time_range() -> TimeRange:
    """
    Build deterministic UTC half-open time range for API tests.

    Args:
        None.
    Returns:
        TimeRange: Stable one-hour range.
    Assumptions:
        Range aligns with `1m` timeframe.
    Raises:
        None.
    Side Effects:
        None.
    """
    start = UtcTimestamp(datetime(2026, 2, 11, 10, 0, 0, tzinfo=timezone.utc))
    end = UtcTimestamp(datetime(2026, 2, 11, 11, 0, 0, tzinfo=timezone.utc))
    return TimeRange(start=start, end=end)


def _candles(*, t_size: int = 60) -> CandleArrays:
    """
    Build deterministic dense candles payload for API compute tests.

    Args:
        t_size: Number of one-minute rows.
    Returns:
        CandleArrays: Dense OHLCV payload.
    Assumptions:
        Payload matches request time range and symbol metadata.
    Raises:
        ValueError: If generated arrays violate DTO invariants.
    Side Effects:
        Allocates numpy arrays.
    """
    ts_open = np.arange(t_size, dtype=np.int64) * np.int64(60_000)
    base = np.linspace(100.0, 130.0, t_size, dtype=np.float32)
    base[[5, 13, 21]] = np.nan
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=_time_range(),
        timeframe=Timeframe("1m"),
        ts_open=ts_open,
        open=base,
        high=base + np.float32(1.0),
        low=base - np.float32(1.0),
        close=base + np.float32(0.5),
        volume=np.linspace(10.0, 20.0, t_size, dtype=np.float32),
    )


def _compute_adapter(*, cache_dir: Path) -> NumbaIndicatorCompute:
    """
    Build real Numba compute adapter for API endpoint tests.

    Args:
        cache_dir: Numba cache path.
    Returns:
        NumbaIndicatorCompute: Compute adapter instance.
    Assumptions:
        Hard indicator definitions are deterministic.
    Raises:
        ValueError: If config values are invalid.
    Side Effects:
        None.
    """
    config = IndicatorsComputeNumbaConfig(
        numba_num_threads=1,
        numba_cache_dir=cache_dir,
        max_compute_bytes_total=5 * 1024**3,
    )
    return NumbaIndicatorCompute(defs=all_defs(), config=config)


def _client(*, compute: IndicatorCompute, candle_feed: CandleFeed) -> TestClient:
    """
    Build TestClient with compute dependencies wired.

    Args:
        compute: Compute adapter implementation.
        candle_feed: CandleFeed stub.
    Returns:
        TestClient: Ready API client.
    Assumptions:
        Router guards use defaults suitable for unit tests.
    Raises:
        ValueError: If router configuration is invalid.
    Side Effects:
        None.
    """
    app = FastAPI()
    registry = _RegistryStub(defs=all_defs())
    app.include_router(
        build_indicators_router(
            registry=registry,
            compute=compute,
            candle_feed=candle_feed,
            max_variants_per_compute=600_000,
            max_compute_bytes_total=5 * 1024**3,
        )
    )
    return TestClient(app)


def _valid_compute_payload() -> dict[str, Any]:
    """
    Build deterministic valid payload for `POST /indicators/compute`.

    Args:
        None.
    Returns:
        dict[str, Any]: Valid compute payload with one MA indicator block.
    Assumptions:
        `ma.sma` and requested params/sources exist in hard definitions.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "market_id": 1,
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "time_range": {
            "start": "2026-02-11T10:00:00Z",
            "end": "2026-02-11T11:00:00Z",
        },
        "indicator": {
            "indicator_id": "ma.sma",
            "source": {"mode": "explicit", "values": ["close", "open"]},
            "params": {
                "window": {"mode": "explicit", "values": [10, 20]},
            },
        },
        "layout": "time_major",
    }


def test_post_indicators_compute_returns_compact_tensor_metadata(tmp_path: Path) -> None:
    """
    Verify compute endpoint executes and returns compact tensor metadata payload.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Compute request is valid and fits default guards.
    Raises:
        AssertionError: If status code or response shape differs from contract.
    Side Effects:
        None.
    """
    feed = _CandleFeedStub(candles=_candles())
    client = _client(compute=_compute_adapter(cache_dir=tmp_path / "numba-cache"), candle_feed=feed)

    response = client.post("/indicators/compute", json=_valid_compute_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["indicator_id"] == "ma.sma"
    assert payload["layout"] == "time_major"
    assert payload["shape"] == [60, 4]
    assert payload["dtype"] == "float32"
    assert payload["c_contiguous"] is True
    assert payload["meta"]["t"] == 60
    assert payload["meta"]["variants"] == 4
    assert payload["axes"][0] == {"name": "source", "values": ["close", "open"]}
    assert payload["axes"][1] == {"name": "window", "values": [10, 20]}
    assert len(feed.calls) == 1


def test_post_indicators_compute_returns_422_when_request_guard_exceeded(tmp_path: Path) -> None:
    """
    Verify compute endpoint returns deterministic variants-guard error payload.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Request guard below actual variants should fail with 422.
    Raises:
        AssertionError: If response payload shape differs from guard contract.
    Side Effects:
        None.
    """
    feed = _CandleFeedStub(candles=_candles())
    client = _client(compute=_compute_adapter(cache_dir=tmp_path / "numba-cache"), candle_feed=feed)

    payload = _valid_compute_payload()
    payload["max_variants_guard"] = 1

    response = client.post("/indicators/compute", json=payload)

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "error": "max_variants_per_compute_exceeded",
            "total_variants": 4,
            "max_variants_per_compute": 1,
        }
    }
    assert len(feed.calls) == 0


def test_post_indicators_compute_uses_single_preflight_path_without_compute_estimate(
    tmp_path: Path,
) -> None:
    """
    Verify compute route does not call `IndicatorCompute.estimate` preflight chain.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Router preflight is fully handled by `BatchEstimator.estimate_batch`.
    Raises:
        AssertionError: If route still calls compute adapter estimate path.
    Side Effects:
        Executes one in-memory API request.
    """
    feed = _CandleFeedStub(candles=_candles())
    spy = _ComputeSpy(delegate=_compute_adapter(cache_dir=tmp_path / "numba-cache"))
    client = _client(compute=spy, candle_feed=feed)

    response = client.post("/indicators/compute", json=_valid_compute_payload())

    assert response.status_code == 200
    assert spy.estimate_calls == 0
    assert spy.compute_calls == 1
