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
    ARTIFACT_MAPPING_TIMEFRAMES_V2,
    ARTIFACT_PLACEHOLDER_SHA256_V2,
    ARTIFACT_PRICE_TIMEFRAMES_V2,
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
_FULL_BUILD_DAYS_V2 = 3
_FULL_BUILD_MINUTES_V2 = _FULL_BUILD_DAYS_V2 * 24 * 60


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
    Verify R3-02 writes `prices/1m` plus every rolled request timeframe deterministically.

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
    reader = _FakeCanonicalCandleReader(
        rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
    )
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=reader,
    )

    result = runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)
    )
    manifest = fixture.loader.load_slot_manifest(fixture.coordinates, fixture.inactive_slot)
    open_time, close_time, ohlcv = _load_price_arrays_v2(fixture=fixture, timeframe="1m")
    fifteen_minute_open, fifteen_minute_close, fifteen_minute_ohlcv = _load_price_arrays_v2(
        fixture=fixture,
        timeframe="15m",
    )
    one_hour_open, one_hour_close, one_hour_ohlcv = _load_price_arrays_v2(
        fixture=fixture,
        timeframe="1h",
    )
    fifteen_minute_mapping_open, fifteen_minute_mapping_close = _load_mapping_arrays_v2(
        fixture=fixture,
        timeframe="15m",
    )
    one_hour_mapping_open, one_hour_mapping_close = _load_mapping_arrays_v2(
        fixture=fixture,
        timeframe="1h",
    )
    three_day_mapping_open, three_day_mapping_close = _load_mapping_arrays_v2(
        fixture=fixture,
        timeframe="3d",
    )
    three_day_open, three_day_close, three_day_ohlcv = _load_price_arrays_v2(
        fixture=fixture,
        timeframe="3d",
    )
    expected_bar_counts = {
        "1m": _FULL_BUILD_MINUTES_V2,
        "15m": 288,
        "30m": 144,
        "1h": 72,
        "2h": 36,
        "4h": 18,
        "6h": 12,
        "8h": 9,
        "1d": 3,
        "2d": 1,
        "3d": 1,
    }

    assert len(reader.calls) == 1
    assert reader.calls[0] == TimeRange(
        start=UtcTimestamp(_BASE_TIME_UTC),
        end=UtcTimestamp(_BASE_TIME_UTC + timedelta(minutes=_FULL_BUILD_MINUTES_V2)),
    )
    assert result.slot == fixture.inactive_slot
    assert result.slot_generation == 5
    assert result.coverage.bar_count == _FULL_BUILD_MINUTES_V2
    assert result.reused_prefix_bars == 0
    assert result.rewritten_tail_bars == _FULL_BUILD_MINUTES_V2
    assert open_time.dtype == np.int64
    assert close_time.dtype == np.int64
    assert ohlcv.dtype == np.float32
    assert open_time.shape == (_FULL_BUILD_MINUTES_V2,)
    assert close_time.shape == (_FULL_BUILD_MINUTES_V2,)
    assert ohlcv.shape == (_FULL_BUILD_MINUTES_V2, 5)
    assert np.all(close_time > open_time)
    assert tuple(item.timeframe for item in manifest.prices) == ARTIFACT_PRICE_TIMEFRAMES_V2
    assert tuple(item.timeframe for item in manifest.mappings) == ARTIFACT_MAPPING_TIMEFRAMES_V2
    assert (
        {item.timeframe: item.coverage.bar_count for item in manifest.prices}
        == expected_bar_counts
    )
    assert manifest.signals.supported_timeframes == ()
    assert manifest.signals.supported_indicator_ids == ()
    assert manifest.signals.manifests == ()
    assert manifest.hit_times.manifest_path == "hit_times/1m/manifest.yaml"
    assert manifest.hit_times.manifest_sha256 == ARTIFACT_PLACEHOLDER_SHA256_V2
    assert fifteen_minute_open.shape == (288,)
    assert fifteen_minute_close.shape == (288,)
    assert fifteen_minute_ohlcv.shape == (288, 5)
    assert fifteen_minute_open[0] == _epoch_ms_for_minute_v2(0)
    assert fifteen_minute_close[0] == _epoch_ms_for_minute_v2(15)
    np.testing.assert_allclose(
        fifteen_minute_ohlcv[0],
        _expected_bucket_ohlcv_v2(bar_indexes=tuple(range(15))),
    )
    assert one_hour_open.shape == (72,)
    assert one_hour_close.shape == (72,)
    assert one_hour_close[-1] == _epoch_ms_for_minute_v2(_FULL_BUILD_MINUTES_V2)
    np.testing.assert_allclose(
        one_hour_ohlcv[0],
        _expected_bucket_ohlcv_v2(bar_indexes=tuple(range(60))),
    )
    assert fifteen_minute_mapping_open.dtype == np.uint32
    assert fifteen_minute_mapping_close.dtype == np.uint32
    assert fifteen_minute_mapping_open.shape == (288,)
    assert fifteen_minute_mapping_close.shape == (288,)
    np.testing.assert_array_equal(
        fifteen_minute_mapping_open,
        np.arange(0, _FULL_BUILD_MINUTES_V2, 15, dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        fifteen_minute_mapping_close,
        np.arange(14, _FULL_BUILD_MINUTES_V2, 15, dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        one_hour_mapping_open,
        np.arange(0, _FULL_BUILD_MINUTES_V2, 60, dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        one_hour_mapping_close,
        np.arange(59, _FULL_BUILD_MINUTES_V2, 60, dtype=np.uint32),
    )
    assert three_day_open.tolist() == [_epoch_ms_for_minute_v2(0)]
    assert three_day_close.tolist() == [_epoch_ms_for_minute_v2(_FULL_BUILD_MINUTES_V2)]
    assert three_day_mapping_open.tolist() == [0]
    assert three_day_mapping_close.tolist() == [_FULL_BUILD_MINUTES_V2 - 1]
    np.testing.assert_allclose(
        three_day_ohlcv[0],
        _expected_bucket_ohlcv_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2))),
    )


def test_backtest_artifact_precompute_runner_v2_uses_deterministic_tail_update(
    tmp_path: Path,
) -> None:
    """
    Verify R3-02 reuses `1m` prefix bars and rebuilds only affected rolled buckets.

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
        rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
    )
    initial_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=initial_reader,
    )
    initial_runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)
    )
    initial_one_minute_open, _, initial_one_minute_ohlcv = _load_price_arrays_v2(
        fixture=fixture,
        timeframe="1m",
    )
    _, _, initial_fifteen_minute_ohlcv = _load_price_arrays_v2(fixture=fixture, timeframe="15m")
    _, _, initial_one_hour_ohlcv = _load_price_arrays_v2(fixture=fixture, timeframe="1h")
    _, _, initial_two_day_ohlcv = _load_price_arrays_v2(fixture=fixture, timeframe="2d")
    _, _, initial_three_day_ohlcv = _load_price_arrays_v2(fixture=fixture, timeframe="3d")

    updated_reader = _FakeCanonicalCandleReader(
        rows=_build_canonical_rows_v2(
            bar_indexes=(_FULL_BUILD_MINUTES_V2 - 2, _FULL_BUILD_MINUTES_V2 - 1),
            price_offset=1000.0,
            volume_offset=50.0,
        )
    )
    updated_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=updated_reader,
    )

    result = updated_runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)
    )
    open_time, _, ohlcv = _load_price_arrays_v2(fixture=fixture, timeframe="1m")
    _, _, fifteen_minute_ohlcv = _load_price_arrays_v2(fixture=fixture, timeframe="15m")
    _, _, one_hour_ohlcv = _load_price_arrays_v2(fixture=fixture, timeframe="1h")
    _, _, two_day_ohlcv = _load_price_arrays_v2(fixture=fixture, timeframe="2d")
    _, _, three_day_ohlcv = _load_price_arrays_v2(fixture=fixture, timeframe="3d")
    offset_overrides = {
        _FULL_BUILD_MINUTES_V2 - 2: (1000.0, 50.0),
        _FULL_BUILD_MINUTES_V2 - 1: (1000.0, 50.0),
    }

    assert len(updated_reader.calls) == 1
    assert updated_reader.calls[0] == TimeRange(
        start=UtcTimestamp(_BASE_TIME_UTC + timedelta(minutes=_FULL_BUILD_MINUTES_V2 - 2)),
        end=UtcTimestamp(_BASE_TIME_UTC + timedelta(minutes=_FULL_BUILD_MINUTES_V2)),
    )
    assert result.reused_prefix_bars == _FULL_BUILD_MINUTES_V2 - 2
    assert result.rewritten_tail_bars == 2
    assert open_time.tolist() == [
        _epoch_ms_for_minute_v2(minute) for minute in range(_FULL_BUILD_MINUTES_V2)
    ]
    np.testing.assert_allclose(ohlcv[:-2], initial_one_minute_ohlcv[:-2])
    np.testing.assert_allclose(
        ohlcv[-2:],
        _expected_ohlcv_matrix_v2(
            bar_indexes=(_FULL_BUILD_MINUTES_V2 - 2, _FULL_BUILD_MINUTES_V2 - 1),
            price_offset=1000.0,
            volume_offset=50.0,
        ),
    )
    np.testing.assert_allclose(fifteen_minute_ohlcv[:-1], initial_fifteen_minute_ohlcv[:-1])
    np.testing.assert_allclose(
        fifteen_minute_ohlcv[-1],
        _expected_bucket_ohlcv_v2(
            bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2 - 15, _FULL_BUILD_MINUTES_V2)),
            offset_overrides=offset_overrides,
        ),
    )
    np.testing.assert_allclose(one_hour_ohlcv[:-1], initial_one_hour_ohlcv[:-1])
    np.testing.assert_allclose(
        one_hour_ohlcv[-1],
        _expected_bucket_ohlcv_v2(
            bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2 - 60, _FULL_BUILD_MINUTES_V2)),
            offset_overrides=offset_overrides,
        ),
    )
    np.testing.assert_allclose(two_day_ohlcv, initial_two_day_ohlcv)
    np.testing.assert_allclose(
        three_day_ohlcv[0],
        _expected_bucket_ohlcv_v2(
            bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)),
            offset_overrides=offset_overrides,
        ),
    )
    assert initial_one_minute_open.tolist() == open_time.tolist()


def test_backtest_artifact_precompute_runner_v2_appends_mapping_tail_deterministically(
    tmp_path: Path,
) -> None:
    """
    Verify R3-03 reuses existing mapping prefix rows and appends only the rebuilt tail.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Inactive slot already contains a previous strict export and the next request extends the
        timeline by exactly one additional `1h` window.
    Raises:
        AssertionError: If mapping prefix rows change or appended rows use incorrect `1m` indexes.
    Side Effects:
        Rewrites inactive-slot price and mapping arrays under `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    initial_end_minute = _FULL_BUILD_MINUTES_V2
    updated_end_minute = _FULL_BUILD_MINUTES_V2 + 60
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=2,
        mapping_tail_bars_1m=10,
    )
    initial_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(initial_end_minute)))
        ),
    )
    initial_runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=initial_end_minute)
    )
    initial_fifteen_minute_open, initial_fifteen_minute_close = _load_mapping_arrays_v2(
        fixture=fixture,
        timeframe="15m",
    )
    initial_one_hour_open, initial_one_hour_close = _load_mapping_arrays_v2(
        fixture=fixture,
        timeframe="1h",
    )

    updated_reader = _FakeCanonicalCandleReader(
        rows=_build_canonical_rows_v2(
            bar_indexes=tuple(range(initial_end_minute - 2, updated_end_minute))
        )
    )
    updated_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=updated_reader,
    )
    updated_runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=updated_end_minute)
    )
    updated_fifteen_minute_open, updated_fifteen_minute_close = _load_mapping_arrays_v2(
        fixture=fixture,
        timeframe="15m",
    )
    updated_one_hour_open, updated_one_hour_close = _load_mapping_arrays_v2(
        fixture=fixture,
        timeframe="1h",
    )

    assert updated_reader.calls[0] == TimeRange(
        start=UtcTimestamp(_BASE_TIME_UTC + timedelta(minutes=initial_end_minute - 2)),
        end=UtcTimestamp(_BASE_TIME_UTC + timedelta(minutes=updated_end_minute)),
    )
    np.testing.assert_array_equal(
        updated_fifteen_minute_open[: initial_fifteen_minute_open.shape[0]],
        initial_fifteen_minute_open,
    )
    np.testing.assert_array_equal(
        updated_fifteen_minute_close[: initial_fifteen_minute_close.shape[0]],
        initial_fifteen_minute_close,
    )
    np.testing.assert_array_equal(
        updated_one_hour_open[: initial_one_hour_open.shape[0]],
        initial_one_hour_open,
    )
    np.testing.assert_array_equal(
        updated_one_hour_close[: initial_one_hour_close.shape[0]],
        initial_one_hour_close,
    )
    np.testing.assert_array_equal(
        updated_fifteen_minute_open[initial_fifteen_minute_open.shape[0] :],
        np.asarray([4320, 4335, 4350, 4365], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        updated_fifteen_minute_close[initial_fifteen_minute_close.shape[0] :],
        np.asarray([4334, 4349, 4364, 4379], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        updated_one_hour_open[initial_one_hour_open.shape[0] :],
        np.asarray([4320], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        updated_one_hour_close[initial_one_hour_close.shape[0] :],
        np.asarray([4379], dtype=np.uint32),
    )


def test_backtest_artifact_precompute_runner_v2_is_byte_stable_for_identical_inputs(
    tmp_path: Path,
) -> None:
    """
    Verify repeated R3-02 runs with identical source data keep byte-stable outputs.

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
        end_minute=_FULL_BUILD_MINUTES_V2,
        asof_date="2026-03-26",
        generated_at_utc="2026-03-26T03:00:00Z",
    )
    rows = _build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))

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
    Verify R3-02 fails fast on duplicate/non-monotone canonical `ts_open` sequence.

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


def test_backtest_artifact_precompute_runner_v2_rejects_non_aligned_rollup_source(
    tmp_path: Path,
) -> None:
    """
    Verify R3-02 fails fast when canonical `1m` source bars are not minute-aligned.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Rollup source must come from exact epoch-aligned `prices/1m` bars with no timezone drift.
    Raises:
        AssertionError: If misaligned source timestamps do not raise the stable boundary error.
    Side Effects:
        Writes only the pointer/config fixture under `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/shared_kernel/primitives/timeframe.py
    """
    fixture = build_artifact_precompute_fixture_v2(tmp_path=tmp_path, price_tail_bars_1m=2)
    reader = _FakeCanonicalCandleReader(
        rows=(
            _build_canonical_row_v2(
                bar_index=0,
                price_offset=0.0,
                volume_offset=0.0,
                open_offset_seconds=30,
            ),
            _build_canonical_row_v2(
                bar_index=1,
                price_offset=0.0,
                volume_offset=0.0,
                open_offset_seconds=30,
            ),
        )
    )
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=reader,
    )

    with pytest.raises(ValueError, match="epoch-aligned to 1m bucket boundaries"):
        runner.export_canonical_price_1m(_request_v2(fixture=fixture, end_minute=2))


def _request_v2(
    *,
    fixture: ArtifactPrecomputeFixtureV2,
    end_minute: int,
    asof_date: str = "2026-03-26",
    generated_at_utc: str = "2026-03-26T03:00:00Z",
) -> ArtifactCanonicalPriceExportRequestV2:
    """
    Build one explicit R3-02 export request for fixture-backed runner tests.

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
    open_offset_seconds: int = 0,
) -> CandleWithMeta:
    """
    Build one deterministic canonical candle row for the requested UTC minute offset.

    Args:
        bar_index: Minute offset relative to `_BASE_TIME_UTC`.
        price_offset: Optional constant added to OHLC values.
        volume_offset: Optional constant added to `volume_base`.
        open_offset_seconds: Optional positive second offset used for boundary-failure tests.
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
    ts_open = _BASE_TIME_UTC + timedelta(minutes=bar_index, seconds=open_offset_seconds)
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
    return np.asarray(
        [
            _bar_ohlcv_tuple_v2(
                bar_index=bar_index,
                price_offset=price_offset,
                volume_offset=volume_offset,
            )
            for bar_index in bar_indexes
        ],
        dtype=np.float32,
    )


def _expected_bucket_ohlcv_v2(
    *,
    bar_indexes: tuple[int, ...],
    offset_overrides: dict[int, tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Aggregate deterministic `1m` test rows into one expected rolled OHLCV bucket.

    Args:
        bar_indexes: Minute offsets that belong to the target rolled bucket.
        offset_overrides: Optional per-minute `(price_offset, volume_offset)` overrides.
    Returns:
        np.ndarray: Expected rolled OHLCV row with shape `(5,)`.
    Assumptions:
        Test candles increase monotonically by base price, so `high` comes from the last minute and
        `low` from the first unless an override changes the ordering.
    Raises:
        ValueError: If `bar_indexes` is empty.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if len(bar_indexes) == 0:
        raise ValueError("bar_indexes must contain at least one minute for bucket expectations")
    rows = np.asarray(
        [
            _bar_ohlcv_tuple_v2(
                bar_index=bar_index,
                price_offset=_offset_override_value_v2(
                    offset_overrides=offset_overrides,
                    bar_index=bar_index,
                    position=0,
                ),
                volume_offset=_offset_override_value_v2(
                    offset_overrides=offset_overrides,
                    bar_index=bar_index,
                    position=1,
                ),
            )
            for bar_index in bar_indexes
        ],
        dtype=np.float32,
    )
    return np.asarray(
        (
            float(rows[0, 0]),
            float(np.max(rows[:, 1])),
            float(np.min(rows[:, 2])),
            float(rows[-1, 3]),
            float(np.sum(rows[:, 4], dtype=np.float64)),
        ),
        dtype=np.float32,
    )


def _bar_ohlcv_tuple_v2(
    *,
    bar_index: int,
    price_offset: float,
    volume_offset: float,
) -> tuple[float, float, float, float, float]:
    """
    Build one deterministic OHLCV tuple for a single canonical `1m` test bar.

    Args:
        bar_index: Minute offset relative to `_BASE_TIME_UTC`.
        price_offset: Optional constant added to OHLC values.
        volume_offset: Optional constant added to `volume_base`.
    Returns:
        tuple[float, float, float, float, float]: Deterministic OHLCV tuple.
    Assumptions:
        Test prices follow the same formula as `_build_canonical_row_v2`.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """
    base_price = float(bar_index + 1) + price_offset
    return (
        base_price,
        base_price + 0.5,
        base_price - 0.25,
        base_price + 0.25,
        10.0 + float(bar_index) + volume_offset,
    )


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


def _load_price_arrays_v2(
    *,
    fixture: ArtifactPrecomputeFixtureV2,
    timeframe: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one materialized `prices/<tf>` family from the inactive slot for assertions.

    Args:
        fixture: Strict precompute fixture with builder and loader.
        timeframe: Price timeframe literal to read.
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: `open_time`, `close_time`, and `ohlcv` arrays.
    Assumptions:
        Runner tests inspect only the inactive slot written by `export_canonical_price_1m(...)`.
    Raises:
        FileNotFoundError: If one expected `.npy` file is missing.
    Side Effects:
        Reads three `.npy` files from disk.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    paths = fixture.loader.resolve_price_paths(
        fixture.coordinates,
        fixture.inactive_slot,
        timeframe,
    )
    return (
        np.load(paths.open_time, allow_pickle=False),
        np.load(paths.close_time, allow_pickle=False),
        np.load(paths.ohlcv, allow_pickle=False),
    )


def _load_mapping_arrays_v2(
    *,
    fixture: ArtifactPrecomputeFixtureV2,
    timeframe: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one materialized `mappings/<tf>` family from the inactive slot for assertions.

    Args:
        fixture: Strict precompute fixture with builder and loader.
        timeframe: Mapping timeframe literal to read.
    Returns:
        tuple[np.ndarray, np.ndarray]: `bar_open_1m_idx` and `bar_close_1m_idx` arrays.
    Assumptions:
        Runner tests inspect only the inactive slot written by `export_canonical_price_1m(...)`.
    Raises:
        FileNotFoundError: If one expected `.npy` file is missing.
    Side Effects:
        Reads two `.npy` files from disk.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    paths = fixture.loader.resolve_mapping_paths(
        fixture.coordinates,
        fixture.inactive_slot,
        timeframe,
    )
    return (
        np.load(paths.bar_open_1m_idx, allow_pickle=False),
        np.load(paths.bar_close_1m_idx, allow_pickle=False),
    )


def _offset_override_value_v2(
    *,
    offset_overrides: dict[int, tuple[float, float]] | None,
    bar_index: int,
    position: int,
) -> float:
    """
    Read one optional `(price_offset, volume_offset)` override component for a test bar.

    Args:
        offset_overrides: Optional mapping keyed by minute offset.
        bar_index: Minute offset looked up in `offset_overrides`.
        position: Tuple position to return (`0` for price, `1` for volume).
    Returns:
        float: Override value when present, otherwise `0.0`.
    Assumptions:
        Offset overrides are sparse and default to zero in deterministic runner tests.
    Raises:
        IndexError: If `position` is outside the override tuple shape.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """
    if offset_overrides is None:
        return 0.0
    return float(offset_overrides.get(bar_index, (0.0, 0.0))[position])


def _read_export_bytes_v2(
    result: ArtifactCanonicalPriceExportResultV2,
) -> tuple[bytes, ...]:
    """
    Read root-manifest plus all materialized `prices/<tf>` and `mappings/<tf>` bytes for
    byte-stability assertions.

    Args:
        result: Structured export result returned by the runner.
    Returns:
        tuple[bytes, ...]: Bytes for root manifest and every emitted price/mapping artifact file.
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
    slot_root = result.manifest_path.parent
    payloads: list[bytes] = [result.manifest_path.read_bytes()]
    for timeframe in ARTIFACT_PRICE_TIMEFRAMES_V2:
        payloads.extend(
            (
                (slot_root / "prices" / timeframe / "open_time.i64.npy").read_bytes(),
                (slot_root / "prices" / timeframe / "close_time.i64.npy").read_bytes(),
                (slot_root / "prices" / timeframe / "ohlcv.f32.npy").read_bytes(),
            )
        )
    for timeframe in ARTIFACT_MAPPING_TIMEFRAMES_V2:
        payloads.extend(
            (
                (slot_root / "mappings" / timeframe / "bar_open_1m_idx.u32.npy").read_bytes(),
                (slot_root / "mappings" / timeframe / "bar_close_1m_idx.u32.npy").read_bytes(),
            )
        )
    return tuple(payloads)
