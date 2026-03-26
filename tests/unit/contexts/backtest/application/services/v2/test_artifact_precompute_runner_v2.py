from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    ArtifactPrecomputeFixtureV2,
    build_artifact_precompute_fixture_v2,
)
from trading.contexts.backtest.application.services import (
    ARTIFACT_PLACEHOLDER_SHA256_V2,
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCanonicalPriceExportResultV2,
    BacktestArtifactPrecomputeRunnerV2,
)
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)

_BASE_TIME_UTC = datetime(2026, 3, 26, 0, 0, tzinfo=timezone.utc)


class _FakeCanonicalCandleReader:
    """
    Deterministic in-memory canonical candle reader for R3-01 precompute tests.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    def __init__(self, *, rows: tuple[CandleWithMeta, ...]) -> None:
        """
        Store deterministic canonical candle rows and initialize recorded call history.

        Args:
            rows: Full in-memory canonical candle sequence available to the fake reader.
        Returns:
            None.
        Assumptions:
            Tests intentionally control both row ordering and returned time-range filtering.
        Raises:
            None.
        Side Effects:
            Stores read-call history in memory for assertions.
        """
        self._rows = rows
        self.calls: list[TimeRange] = []

    def read_1m(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        """
        Return rows whose `ts_open` falls into the requested `TimeRange [start, end)`.

        Args:
            instrument_id: Ignored shared-kernel identity passed by the production runner.
            time_range: Source reread window requested by the runner.
        Returns:
            Iterator[CandleWithMeta]: Filtered canonical candle iterator.
        Assumptions:
            Tests validate only time-range behavior, not instrument-id branching.
        Raises:
            None.
        Side Effects:
            Appends the requested time range to the in-memory call log.
        """
        del instrument_id
        self.calls.append(time_range)
        return iter(
            tuple(
                row
                for row in self._rows
                if time_range.start.value <= row.candle.ts_open.value < time_range.end.value
            )
        )


def test_backtest_artifact_precompute_runner_v2_builds_initial_canonical_1m_export(
    tmp_path: Path,
) -> None:
    """
    Verify R3-01 writes deterministic `prices/1m/*` files and strict root-manifest coverage.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Inactive slot starts without a pre-existing root manifest or price arrays.
    Raises:
        AssertionError: If written arrays, manifest metadata, or slot identity are incorrect.
    Side Effects:
        Creates strict artifact files under `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    fixture = build_artifact_precompute_fixture_v2(tmp_path=tmp_path, price_tail_bars_1m=2)
    reader = _FakeCanonicalCandleReader(rows=_build_canonical_rows_v2(bar_indexes=(0, 1, 2, 3, 4)))
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=reader,
    )

    result = runner.export_canonical_price_1m(_request_v2(fixture=fixture, end_minute=5))
    manifest = fixture.loader.load_slot_manifest(fixture.coordinates, fixture.inactive_slot)
    open_time = np.load(result.price_paths.open_time, allow_pickle=False)
    close_time = np.load(result.price_paths.close_time, allow_pickle=False)
    ohlcv = np.load(result.price_paths.ohlcv, allow_pickle=False)

    assert len(reader.calls) == 1
    assert reader.calls[0] == TimeRange(
        start=UtcTimestamp(_BASE_TIME_UTC),
        end=UtcTimestamp(_BASE_TIME_UTC + timedelta(minutes=5)),
    )
    assert result.slot == fixture.inactive_slot
    assert result.slot_generation == 5
    assert result.coverage.bar_count == 5
    assert result.reused_prefix_bars == 0
    assert result.rewritten_tail_bars == 5
    assert open_time.dtype == np.int64
    assert close_time.dtype == np.int64
    assert ohlcv.dtype == np.float32
    assert open_time.shape == (5,)
    assert close_time.shape == (5,)
    assert ohlcv.shape == (5, 5)
    assert np.all(close_time > open_time)
    assert tuple(item.timeframe for item in manifest.prices) == ("1m",)
    assert manifest.mappings == ()
    assert manifest.signals.supported_timeframes == ()
    assert manifest.signals.supported_indicator_ids == ()
    assert manifest.signals.manifests == ()
    assert manifest.hit_times.manifest_path == "hit_times/1m/manifest.yaml"
    assert manifest.hit_times.manifest_sha256 == ARTIFACT_PLACEHOLDER_SHA256_V2


def test_backtest_artifact_precompute_runner_v2_uses_deterministic_tail_update(
    tmp_path: Path,
) -> None:
    """
    Verify R3-01 reuses prefix bars and rereads only the configured tail overlap.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Inactive slot already contains a previous strict `prices/1m` export for the same symbol.
    Raises:
        AssertionError: If tail reread bounds or merged array contents are incorrect.
    Side Effects:
        Rewrites inactive-slot arrays and manifest under `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    fixture = build_artifact_precompute_fixture_v2(tmp_path=tmp_path, price_tail_bars_1m=2)
    initial_reader = _FakeCanonicalCandleReader(
        rows=_build_canonical_rows_v2(bar_indexes=(0, 1, 2, 3, 4, 5))
    )
    initial_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=initial_reader,
    )
    initial_runner.export_canonical_price_1m(_request_v2(fixture=fixture, end_minute=6))

    updated_reader = _FakeCanonicalCandleReader(
        rows=_build_canonical_rows_v2(
            bar_indexes=(4, 5, 6, 7),
            price_offset=1000.0,
            volume_offset=50.0,
        )
    )
    updated_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=updated_reader,
    )

    result = updated_runner.export_canonical_price_1m(_request_v2(fixture=fixture, end_minute=8))
    open_time = np.load(result.price_paths.open_time, allow_pickle=False)
    ohlcv = np.load(result.price_paths.ohlcv, allow_pickle=False)

    assert len(updated_reader.calls) == 1
    assert updated_reader.calls[0] == TimeRange(
        start=UtcTimestamp(_BASE_TIME_UTC + timedelta(minutes=4)),
        end=UtcTimestamp(_BASE_TIME_UTC + timedelta(minutes=8)),
    )
    assert result.reused_prefix_bars == 4
    assert result.rewritten_tail_bars == 4
    assert open_time.tolist() == [
        _epoch_ms_for_minute_v2(minute)
        for minute in (0, 1, 2, 3, 4, 5, 6, 7)
    ]
    np.testing.assert_allclose(
        ohlcv[:4],
        _expected_ohlcv_matrix_v2(bar_indexes=(0, 1, 2, 3)),
    )
    np.testing.assert_allclose(
        ohlcv[4:],
        _expected_ohlcv_matrix_v2(
            bar_indexes=(4, 5, 6, 7),
            price_offset=1000.0,
            volume_offset=50.0,
        ),
    )


def test_backtest_artifact_precompute_runner_v2_is_byte_stable_for_identical_inputs(
    tmp_path: Path,
) -> None:
    """
    Verify repeated R3-01 runs with identical source data keep byte-stable outputs.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Request identity and `generated_at_utc` stay fixed across both runs.
    Raises:
        AssertionError: If one written array or root manifest changes bytes unnecessarily.
    Side Effects:
        Rewrites inactive-slot files under `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    fixture = build_artifact_precompute_fixture_v2(tmp_path=tmp_path, price_tail_bars_1m=2)
    request = _request_v2(
        fixture=fixture,
        end_minute=5,
        asof_date="2026-03-26",
        generated_at_utc="2026-03-26T03:00:00Z",
    )
    rows = _build_canonical_rows_v2(bar_indexes=(0, 1, 2, 3, 4))

    first_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(rows=rows),
    )
    first_result = first_runner.export_canonical_price_1m(request)
    first_bytes = _read_export_bytes_v2(first_result)

    second_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(rows=rows),
    )
    second_result = second_runner.export_canonical_price_1m(request)
    second_bytes = _read_export_bytes_v2(second_result)

    assert first_bytes == second_bytes


def test_backtest_artifact_precompute_runner_v2_rejects_non_monotonic_source_timestamps(
    tmp_path: Path,
) -> None:
    """
    Verify R3-01 fails fast on duplicate/non-monotone canonical `ts_open` sequence.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Source candles are not silently normalized or deduplicated by the precompute runner.
    Raises:
        AssertionError: If invalid source rows do not raise the stable monotonicity error.
    Side Effects:
        Writes only the pointer/config fixture under `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    fixture = build_artifact_precompute_fixture_v2(tmp_path=tmp_path, price_tail_bars_1m=2)
    reader = _FakeCanonicalCandleReader(
        rows=_build_canonical_rows_v2(bar_indexes=(0, 1, 1, 2))
    )
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=reader,
    )

    with pytest.raises(ValueError, match="strictly increasing by open_time"):
        runner.export_canonical_price_1m(_request_v2(fixture=fixture, end_minute=3))


def _request_v2(
    *,
    fixture: ArtifactPrecomputeFixtureV2,
    end_minute: int,
    asof_date: str = "2026-03-26",
    generated_at_utc: str = "2026-03-26T03:00:00Z",
) -> ArtifactCanonicalPriceExportRequestV2:
    """
    Build one explicit R3-01 export request for fixture-backed runner tests.

    Args:
        fixture: Minimal strict precompute fixture.
        end_minute: Exclusive end minute offset relative to `_BASE_TIME_UTC`.
        asof_date: Strict as-of date literal for the export request.
        generated_at_utc: Strict deterministic UTC generation timestamp.
    Returns:
        ArtifactCanonicalPriceExportRequestV2: Explicit runner request DTO.
    Assumptions:
        All tests use the same base UTC minute grid for clarity.
    Raises:
        ValueError: If request identity violates strict export contracts.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return ArtifactCanonicalPriceExportRequestV2(
        coordinates=fixture.coordinates,
        time_range=TimeRange(
            start=UtcTimestamp(_BASE_TIME_UTC),
            end=UtcTimestamp(_BASE_TIME_UTC + timedelta(minutes=end_minute)),
        ),
        asof_date=asof_date,
        generated_at_utc=generated_at_utc,
    )


def _build_canonical_rows_v2(
    *,
    bar_indexes: tuple[int, ...],
    price_offset: float = 0.0,
    volume_offset: float = 0.0,
) -> tuple[CandleWithMeta, ...]:
    """
    Build deterministic canonical candle rows for runner tests on a `1m` UTC grid.

    Args:
        bar_indexes: Minute offsets relative to `_BASE_TIME_UTC`.
        price_offset: Optional constant added to OHLC values for update scenarios.
        volume_offset: Optional constant added to `volume_base`.
    Returns:
        tuple[CandleWithMeta, ...]: Deterministic canonical candle rows.
    Assumptions:
        Tests use `volume_base` as the fifth `ohlcv` field in exported arrays.
    Raises:
        ValueError: If one constructed candle violates shared-kernel invariants.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/contexts/market_data/application/dto/candle_with_meta.py
    """
    return tuple(
        _build_canonical_row_v2(
            bar_index=bar_index,
            price_offset=price_offset,
            volume_offset=volume_offset,
        )
        for bar_index in bar_indexes
    )


def _build_canonical_row_v2(
    *,
    bar_index: int,
    price_offset: float,
    volume_offset: float,
) -> CandleWithMeta:
    """
    Build one deterministic canonical candle row for the requested UTC minute offset.

    Args:
        bar_index: Minute offset relative to `_BASE_TIME_UTC`.
        price_offset: Optional constant added to OHLC values.
        volume_offset: Optional constant added to `volume_base`.
    Returns:
        CandleWithMeta: Deterministic canonical candle row.
    Assumptions:
        Candle meta is minimal and stable because runner tests exercise only price export logic.
    Raises:
        ValueError: If constructed candle fields violate shared-kernel invariants.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/shared_kernel/primitives/candle.py
    """
    ts_open = _BASE_TIME_UTC + timedelta(minutes=bar_index)
    ts_close = ts_open + timedelta(minutes=1)
    base_price = float(bar_index + 1) + price_offset
    return CandleWithMeta(
        candle=Candle(
            instrument_id=_instrument_id_v2(),
            ts_open=UtcTimestamp(ts_open),
            ts_close=UtcTimestamp(ts_close),
            open=base_price,
            high=base_price + 0.5,
            low=base_price - 0.25,
            close=base_price + 0.25,
            volume_base=10.0 + float(bar_index) + volume_offset,
            volume_quote=None,
        ),
        meta=CandleMeta(
            source="rest",
            ingested_at=UtcTimestamp(ts_close),
            ingest_id=None,
            instrument_key="binance:spot:BTCUSDT",
            trades_count=1,
            taker_buy_volume_base=None,
            taker_buy_volume_quote=None,
        ),
    )


def _instrument_id_v2() -> InstrumentId:
    """
    Build the canonical instrument identity shared by all deterministic test candles.

    Args:
        None.
    Returns:
        InstrumentId: Shared-kernel `InstrumentId` instance for deterministic tests.
    Assumptions:
        All runner tests target the same `binance/spot/BTCUSDT` symbol root.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    """
    return InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT"))


def _expected_ohlcv_matrix_v2(
    *,
    bar_indexes: tuple[int, ...],
    price_offset: float = 0.0,
    volume_offset: float = 0.0,
) -> np.ndarray:
    """
    Build the expected exported `ohlcv` matrix for deterministic runner assertions.

    Args:
        bar_indexes: Minute offsets relative to `_BASE_TIME_UTC`.
        price_offset: Optional constant added to OHLC values.
        volume_offset: Optional constant added to `volume_base`.
    Returns:
        np.ndarray: Expected float32 `ohlcv` matrix.
    Assumptions:
        Exported fifth column stores `volume_base`.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    rows: list[tuple[float, float, float, float, float]] = []
    for bar_index in bar_indexes:
        base_price = float(bar_index + 1) + price_offset
        rows.append(
            (
                base_price,
                base_price + 0.5,
                base_price - 0.25,
                base_price + 0.25,
                10.0 + float(bar_index) + volume_offset,
            )
        )
    return np.asarray(rows, dtype=np.float32)


def _epoch_ms_for_minute_v2(minute: int) -> int:
    """
    Convert deterministic test minute offset into epoch milliseconds.

    Args:
        minute: Minute offset relative to `_BASE_TIME_UTC`.
    Returns:
        int: Epoch milliseconds for the minute bucket open.
    Assumptions:
        Base test grid uses exact UTC minute boundaries.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/shared-kernel-primitives.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/shared_kernel/primitives/utc_timestamp.py
    """
    return int((_BASE_TIME_UTC + timedelta(minutes=minute)).timestamp() * 1000)


def _read_export_bytes_v2(
    result: ArtifactCanonicalPriceExportResultV2,
) -> tuple[bytes, bytes, bytes, bytes]:
    """
    Read root-manifest and `.npy` bytes for byte-stability assertions.

    Args:
        result: Structured export result returned by the runner.
    Returns:
        tuple[bytes, bytes, bytes, bytes]: Bytes for root manifest and three `prices/1m` files.
    Assumptions:
        Result exposes `manifest_path` and `price_paths` exactly as the production DTO does.
    Raises:
        OSError: If one file cannot be read.
    Side Effects:
        Reads artifact files from disk.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return (
        result.manifest_path.read_bytes(),
        result.price_paths.open_time.read_bytes(),
        result.price_paths.close_time.read_bytes(),
        result.price_paths.ohlcv.read_bytes(),
    )
