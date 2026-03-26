from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Iterator, Mapping, cast

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    ArtifactPrecomputeFixtureV2,
    build_artifact_precompute_fixture_v2,
)
from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    AtomicArtifactCurrentPointerWriterV2,
)
from trading.contexts.backtest.application.ports import BacktestJobRepository
from trading.contexts.backtest.application.services import (
    ARTIFACT_MAPPING_TIMEFRAMES_V2,
    ARTIFACT_PLACEHOLDER_SHA256_V2,
    ARTIFACT_PRICE_TIMEFRAMES_V2,
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCanonicalPriceExportResultV2,
    ArtifactSignalValidationSpecV2,
    ArtifactSlotPublishErrorV2,
    BacktestArtifactManifestValidatorV2,
    BacktestArtifactPrecomputeRunnerV2,
    BacktestArtifactSlotPublisherV2,
    BacktestSignalRulesEngineV2,
)
from trading.contexts.indicators.adapters.outbound.registry import YamlIndicatorRegistry
from trading.contexts.indicators.application.dto import (
    ComputeRequest,
    EstimateResult,
    IndicatorTensor,
    TensorMeta,
)
from trading.contexts.indicators.application.services import GridBuilder
from trading.contexts.indicators.domain.definitions import all_defs
from trading.contexts.indicators.domain.entities import AxisDef, IndicatorId, Layout
from trading.contexts.indicators.domain.specifications import (
    ExplicitValuesSpec,
    GridParamSpec,
    GridSpec,
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


@dataclass(frozen=True, slots=True)
class _PrecomputeSignalDefaultsProvider:
    """
    Defaults-provider wrapper overriding only compute grids for small R4-02 test matrices.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/defaults/
        indicators_yaml_defaults_provider.py
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """

    delegate: YamlBacktestGridDefaultsProvider
    overrides: Mapping[str, GridSpec]

    def compute_defaults(self, *, indicator_id: str) -> GridSpec | None:
        """
        Resolve compute defaults with explicit small-grid overrides for selected indicators.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            GridSpec | None: Overridden small grid or the delegate-provided defaults.
        Assumptions:
            Only a tiny R4-02 subset is overridden; all other indicators keep canonical defaults.
        Raises:
            ValueError: Propagated from the delegate or `GridSpec` normalization.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/ports/staged_runner.py
        """
        normalized_indicator_id = indicator_id.strip().lower()
        override = self.overrides.get(normalized_indicator_id)
        if override is not None:
            return override
        return self.delegate.compute_defaults(indicator_id=normalized_indicator_id)

    def signal_param_defaults(self, *, indicator_id: str) -> Mapping[str, GridParamSpec]:
        """
        Delegate signal-parameter defaults to the canonical YAML provider.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            Mapping[str, object]: Canonical default-only signal params mapping.
        Assumptions:
            R4-02 tests intentionally preserve the authoritative signal default semantics.
        Raises:
            ValueError: Propagated from the delegate provider.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/defaults/
            indicators_yaml_defaults_provider.py
        """
        return self.delegate.signal_param_defaults(indicator_id=indicator_id)

    def supported_indicator_ids(self) -> tuple[str, ...]:
        """
        Return the canonical full supported-indicator catalog for engine startup validation.

        Args:
            None.
        Returns:
            tuple[str, ...]: Canonical full supported indicator catalog.
        Assumptions:
            Startup fail-fast validation must still see the real production/test catalog.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
        """
        return self.delegate.supported_indicator_ids()

    def allowed_source_values(self, *, indicator_id: str) -> tuple[str, ...]:
        """
        Delegate allowed-source catalog lookups to the canonical YAML provider.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            tuple[str, ...]: Canonical allowed `inputs.source` literals.
        Assumptions:
            Source validation should remain aligned with runtime defaults.
        Raises:
            ValueError: Propagated from the delegate provider.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
        """
        return self.delegate.allowed_source_values(indicator_id=indicator_id)


@dataclass(frozen=True, slots=True)
class _DeterministicSignalCompute:
    """
    Small deterministic compute adapter producing rolling-mean tensors for signal tests.

    Docs:
      - docs/architecture/indicators/indicators-overview.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """

    grid_builder: GridBuilder
    time_lengths: list[int] = field(default_factory=list, compare=False)

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Estimate variant count using the shared grid-builder materialization order.

        Args:
            grid: Indicator grid to estimate.
            max_variants_guard: Upper bound for allowed variants.
        Returns:
            EstimateResult: Deterministic estimate snapshot.
        Assumptions:
            Tests use the same grid materialization path for estimate and compute.
        Raises:
            ValueError: If the estimate would exceed the explicit variants guard.
        Side Effects:
            None.
        Docs:
          - docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
        Related:
          - src/trading/contexts/indicators/application/dto/estimate_result.py
        """
        materialized = self.grid_builder.materialize_indicator(grid=grid)
        if materialized.variants > max_variants_guard:
            raise ValueError(
                "variants exceed guard: "
                f"variants={materialized.variants}, max_variants_guard={max_variants_guard}"
            )
        return EstimateResult(
            indicator_id=grid.indicator_id,
            axes=tuple(_axis_def_from_materialized_axis_v2(axis) for axis in materialized.axes),
            variants=materialized.variants,
            max_variants_guard=max_variants_guard,
        )

    def compute(self, req: ComputeRequest) -> IndicatorTensor:
        """
        Materialize a deterministic variant-major tensor using source-specific rolling means.

        Args:
            req: Compute request with candles and the explicit grid.
        Returns:
            IndicatorTensor: Variant-major float32 tensor with deterministic rolling-mean outputs.
        Assumptions:
            Test grids use only `source` and `window` axes, so finite-window outputs are enough
            for bounded tail-rebuild assertions.
        Raises:
            ValueError: If one source series cannot be resolved or the guard is exceeded.
        Side Effects:
            Allocates one small in-memory tensor and records requested timeline lengths.
        Docs:
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
          - docs/architecture/indicators/indicators-overview.md
        Related:
          - src/trading/contexts/indicators/application/dto/indicator_tensor.py
        """
        materialized = self.grid_builder.materialize_indicator(grid=req.grid)
        if materialized.variants > req.max_variants_guard:
            raise ValueError(
                "variants exceed guard: "
                f"variants={materialized.variants}, max_variants_guard={req.max_variants_guard}"
        )
        bar_count = int(req.candles.close.shape[0])
        self.time_lengths.append(bar_count)
        values = np.empty((materialized.variants, bar_count), dtype=np.float32)
        axes = tuple(_axis_def_from_materialized_axis_v2(axis) for axis in materialized.axes)
        axis_values = tuple(axis.values for axis in materialized.axes)
        ordered_rows = product(*axis_values) if len(axis_values) > 0 else ((),)
        for row_index, value_row in enumerate(ordered_rows):
            source_name = "close"
            window = 1
            for axis, value in zip(materialized.axes, value_row):
                if axis.name == "source":
                    source_name = str(value)
                if axis.name == "window":
                    window = int(value)
            base_series = _source_series_for_compute_v2(
                candles=req.candles,
                source_name=source_name,
            )
            values[row_index, :] = _rolling_mean_series_for_compute_v2(
                source=base_series,
                window=window,
            )
        return IndicatorTensor(
            indicator_id=req.grid.indicator_id,
            layout=Layout.VARIANT_MAJOR,
            axes=axes,
            values=values,
            meta=TensorMeta(t=bar_count, variants=materialized.variants),
        )

    def warmup(self) -> None:
        """
        No-op warmup required by the compute protocol in these unit tests.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Synthetic compute has no caches or JIT state to initialize.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/indicators/indicators-overview.md
        Related:
          - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
        """
        return None


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


class _ZeroBlockingRepositoryV2:
    """
    Fake job repository returning zero blocking pins for the R3-04 publish integration tests.
    """

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        """
        Return zero blocking jobs for the explicit inactive-slot pin query.

        Args:
            market_id: Canonical market id for the symbol under publish.
            symbol: Instrument symbol under publish.
            artifact_slot: Candidate inactive slot literal.
            artifact_manifest_hash: SHA-256 hash of the inactive slot root manifest.
        Returns:
            int: Always `0`.
        Assumptions:
            Integration tests here focus on successful publish and later-stage validation
            failures, not on pin-guard rejection.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        del market_id, symbol, artifact_slot, artifact_manifest_hash
        return 0


def _build_signal_test_defaults_provider_v2() -> _PrecomputeSignalDefaultsProvider:
    """
    Build a small-grid defaults provider wrapper for deterministic R4-02 runner tests.

    Args:
        None.
    Returns:
        _PrecomputeSignalDefaultsProvider: Wrapper with small MA override grids.
    Assumptions:
        Prod defaults carry the full source catalog required by the v2 signal-rules engine.
    Raises:
        FileNotFoundError: If `configs/prod/indicators.yaml` is unavailable.
    Side Effects:
        Reads the repository-local prod defaults YAML.
    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/adapters/outbound/defaults/
        indicators_yaml_defaults_provider.py
    """
    delegate = YamlBacktestGridDefaultsProvider.from_yaml(
        config_path=Path("configs/prod/indicators.yaml")
    )
    small_axes = {
        "source": ExplicitValuesSpec(name="source", values=("close", "open")),
        "window": ExplicitValuesSpec(name="window", values=(5, 10, 15)),
    }
    overrides = {
        indicator_id: GridSpec(
            indicator_id=IndicatorId(indicator_id),
            params={"window": small_axes["window"]},
            source=small_axes["source"],
        )
        for indicator_id in ("ma.ema", "ma.sma")
    }
    return _PrecomputeSignalDefaultsProvider(delegate=delegate, overrides=overrides)


def _signal_grid_builder_v2() -> GridBuilder:
    """
    Build the shared grid builder used by small deterministic signal export tests.

    Args:
        None.
    Returns:
        GridBuilder: Grid builder backed by the repository-local test indicator registry.
    Assumptions:
        Hard indicator defs plus `configs/test/indicators.yaml` validate successfully.
    Raises:
        FileNotFoundError: If the test indicator config is missing.
        ValueError: If the registry cannot be built deterministically.
    Side Effects:
        Reads the repository-local indicator defaults YAML.
    Docs:
      - docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/indicators/adapters/outbound/registry/yaml_indicator_registry.py
      - src/trading/contexts/indicators/application/services/grid_builder.py
    """
    registry = YamlIndicatorRegistry.from_yaml(
        defs=all_defs(),
        config_path=Path("configs/test/indicators.yaml"),
    )
    return GridBuilder(registry=registry)


def _axis_def_from_materialized_axis_v2(axis: object) -> AxisDef:
    """
    Convert one materialized grid axis into the explicit `AxisDef` tensor metadata contract.

    Args:
        axis: Materialized axis returned by `GridBuilder.materialize_indicator`.
    Returns:
        AxisDef: Explicit tensor metadata axis.
    Assumptions:
        Materialized axis values are homogeneous by type for these test grids.
    Raises:
        ValueError: If the axis values contain unsupported scalar types.
    Side Effects:
        None.
    Docs:
      - docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related:
      - src/trading/contexts/indicators/domain/entities/axis_def.py
      - src/trading/contexts/indicators/application/services/grid_builder.py
    """
    axis_name = cast(str, getattr(axis, "name"))
    values = tuple(cast(tuple[object, ...], getattr(axis, "values")))
    if len(values) == 0:
        raise ValueError("materialized axis requires non-empty values")
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return AxisDef(name=axis_name, values_int=tuple(cast(int, value) for value in values))
    if all(isinstance(value, float) for value in values):
        return AxisDef(name=axis_name, values_float=tuple(cast(float, value) for value in values))
    if all(isinstance(value, str) for value in values):
        return AxisDef(name=axis_name, values_enum=tuple(cast(str, value) for value in values))
    raise ValueError(f"unsupported materialized axis values for {axis_name!r}: {values!r}")


def _source_series_for_compute_v2(*, candles: object, source_name: str) -> np.ndarray:
    """
    Resolve one deterministic source series for the synthetic compute adapter.

    Args:
        candles: Candle-arrays payload passed into the synthetic compute adapter.
        source_name: Requested source literal.
    Returns:
        np.ndarray: Float32 bar-aligned source series.
    Assumptions:
        Tests use only standard price-derived source literals supported by MA indicators.
    Raises:
        ValueError: If the requested source literal is unsupported.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/indicators/indicators-overview.md
    Related:
      - src/trading/contexts/indicators/application/dto/candle_arrays.py
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """
    normalized_source = source_name.strip().lower()
    candle_arrays = cast("object", candles)
    if normalized_source == "close":
        return np.ascontiguousarray(cast(np.ndarray, getattr(candle_arrays, "close")))
    if normalized_source == "open":
        return np.ascontiguousarray(cast(np.ndarray, getattr(candle_arrays, "open")))
    if normalized_source == "high":
        return np.ascontiguousarray(cast(np.ndarray, getattr(candle_arrays, "high")))
    if normalized_source == "low":
        return np.ascontiguousarray(cast(np.ndarray, getattr(candle_arrays, "low")))
    if normalized_source == "hlc3":
        return np.ascontiguousarray(
            (
                cast(np.ndarray, getattr(candle_arrays, "high"))
                + cast(np.ndarray, getattr(candle_arrays, "low"))
                + cast(np.ndarray, getattr(candle_arrays, "close"))
            )
            / np.float32(3.0),
            dtype=np.float32,
        )
    if normalized_source == "ohlc4":
        return np.ascontiguousarray(
            (
                cast(np.ndarray, getattr(candle_arrays, "open"))
                + cast(np.ndarray, getattr(candle_arrays, "high"))
                + cast(np.ndarray, getattr(candle_arrays, "low"))
                + cast(np.ndarray, getattr(candle_arrays, "close"))
            )
            / np.float32(4.0),
            dtype=np.float32,
    )
    raise ValueError(f"unsupported synthetic source literal: {normalized_source!r}")


def _rolling_mean_series_for_compute_v2(*, source: np.ndarray, window: int) -> np.ndarray:
    """
    Build one deterministic float32 rolling-mean series with warmup `NaN`s.

    Args:
        source: One-dimensional float32 source series.
        window: Positive rolling window size.
    Returns:
        np.ndarray: Float32 rolling-mean vector with `NaN` before full-window coverage.
    Assumptions:
        Test compute mirrors finite-window warmup semantics closely enough for tail-rebuild
        assertions without invoking the production indicator engine.
    Raises:
        ValueError: If `window` is non-positive or `source` is not one-dimensional.
    Side Effects:
        Allocates one output vector.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window!r}")
    if source.ndim != 1:
        raise ValueError(f"source must be 1D, got {source.shape!r}")
    output = np.full(source.shape[0], np.float32(np.nan), dtype=np.float32)
    if source.shape[0] < window:
        return output
    cumulative = np.cumsum(np.asarray(source, dtype=np.float64))
    for index in range(window - 1, source.shape[0]):
        start = index + 1 - window
        window_sum = cumulative[index] - (0.0 if start == 0 else cumulative[start - 1])
        output[index] = np.float32(window_sum / float(window))
    return output


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
    assert {
        item.timeframe: item.coverage.bar_count for item in manifest.prices
    } == expected_bar_counts
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


def test_backtest_artifact_precompute_runner_v2_materializes_signal_artifacts_and_root_catalog(
    tmp_path: Path,
) -> None:
    """
    Verify R4-02 writes strict per-target signal artifacts and populates the root catalog.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Signal targets are explicitly enabled both for precompute and for full validation.
    Raises:
        AssertionError: If signal matrices, manifests, root catalog, or validator output drift.
    Side Effects:
        Writes signal artifacts under the inactive slot in `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    signal_targets = (("15m", "ma.ema"), ("15m", "ma.sma"), ("1h", "ma.sma"))
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=2,
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=False,
    )
    defaults_provider = _build_signal_test_defaults_provider_v2()
    grid_builder = _signal_grid_builder_v2()
    signal_rules_engine = BacktestSignalRulesEngineV2(defaults_provider=defaults_provider)
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=_DeterministicSignalCompute(grid_builder=grid_builder),
        indicator_grid_builder=grid_builder,
    )

    result = runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)
    )
    root_manifest = fixture.loader.load_slot_manifest(fixture.coordinates, fixture.inactive_slot)
    ema_manifest = fixture.loader.load_signal_manifest(
        fixture.coordinates,
        fixture.inactive_slot,
        "15m",
        "ma.ema",
    )
    fifteen_minute_sma_manifest = fixture.loader.load_signal_manifest(
        fixture.coordinates,
        fixture.inactive_slot,
        "15m",
        "ma.sma",
    )
    one_hour_sma_manifest = fixture.loader.load_signal_manifest(
        fixture.coordinates,
        fixture.inactive_slot,
        "1h",
        "ma.sma",
    )
    ema_matrix = np.load(
        fixture.loader.resolve_signal_paths(
            fixture.coordinates,
            fixture.inactive_slot,
            "15m",
            "ma.ema",
        ).signals,
        allow_pickle=False,
    )
    fifteen_minute_sma_matrix = np.load(
        fixture.loader.resolve_signal_paths(
            fixture.coordinates,
            fixture.inactive_slot,
            "15m",
            "ma.sma",
        ).signals,
        allow_pickle=False,
    )
    one_hour_sma_matrix = np.load(
        fixture.loader.resolve_signal_paths(
            fixture.coordinates,
            fixture.inactive_slot,
            "1h",
            "ma.sma",
        ).signals,
        allow_pickle=False,
    )
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=fixture.loader)
    validation_result = validator.validate_slot(
        coordinates=fixture.coordinates,
        slot=fixture.inactive_slot,
        validation_spec=fixture.runtime_config.to_validation_spec(),
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert result.slot == fixture.inactive_slot
    assert result.slot_generation == 5
    assert fixture.runtime_settings.signal_artifacts == (
        ArtifactSignalValidationSpecV2(timeframe="15m", indicator_id="ma.ema"),
        ArtifactSignalValidationSpecV2(timeframe="15m", indicator_id="ma.sma"),
        ArtifactSignalValidationSpecV2(timeframe="1h", indicator_id="ma.sma"),
    )
    assert root_manifest.signals.supported_timeframes == ("15m", "1h")
    assert root_manifest.signals.supported_indicator_ids == ("ma.ema", "ma.sma")
    assert tuple(
        (entry.timeframe, entry.indicator_id, entry.manifest_path)
        for entry in root_manifest.signals.manifests
    ) == (
        ("15m", "ma.ema", "signals/15m/ma.ema/manifest.yaml"),
        ("15m", "ma.sma", "signals/15m/ma.sma/manifest.yaml"),
        ("1h", "ma.sma", "signals/1h/ma.sma/manifest.yaml"),
    )
    assert ema_manifest.rows_count == 6
    assert ema_manifest.timeline.bar_count == 288
    assert ema_manifest.signals.dtype == "int8"
    assert ema_manifest.signals.shape == (6, 288)
    assert ema_manifest.signals.axis_order == ("variant", "time")
    assert dict(ema_manifest.grid.signals_v1_params_defaults) == {}
    assert fifteen_minute_sma_manifest.rows_count == 6
    assert fifteen_minute_sma_manifest.timeline.bar_count == 288
    assert fifteen_minute_sma_manifest.signals.dtype == "int8"
    assert fifteen_minute_sma_manifest.signals.shape == (6, 288)
    assert fifteen_minute_sma_manifest.signals.axis_order == ("variant", "time")
    assert dict(fifteen_minute_sma_manifest.grid.signals_v1_params_defaults) == {}
    assert one_hour_sma_manifest.rows_count == 6
    assert one_hour_sma_manifest.timeline.bar_count == 72
    assert one_hour_sma_manifest.signals.dtype == "int8"
    assert one_hour_sma_manifest.signals.shape == (6, 72)
    assert one_hour_sma_manifest.signals.axis_order == ("variant", "time")
    assert dict(one_hour_sma_manifest.grid.signals_v1_params_defaults) == {}
    assert ema_matrix.dtype == np.int8
    assert fifteen_minute_sma_matrix.dtype == np.int8
    assert one_hour_sma_matrix.dtype == np.int8
    assert ema_matrix.shape == (6, 288)
    assert fifteen_minute_sma_matrix.shape == (6, 288)
    assert one_hour_sma_matrix.shape == (6, 72)
    assert set(np.unique(ema_matrix).tolist()) <= {-1, 0, 1}
    assert set(np.unique(fifteen_minute_sma_matrix).tolist()) <= {-1, 0, 1}
    assert set(np.unique(one_hour_sma_matrix).tolist()) <= {-1, 0, 1}
    assert len(validation_result.signal_manifests) == 3
    assert validation_result.hit_times_manifest is None
    assert validation_result.diagnostics == ()


def test_backtest_artifact_precompute_runner_v2_reuses_signal_prefix_and_rebuilds_tail(
    tmp_path: Path,
) -> None:
    """
    Verify R4-03 keeps the unchanged signal prefix and rebuilds only the bounded tail window.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Existing inactive-slot signal artifacts are valid and the second run updates only the last
        30 canonical `1m` bars.
    Raises:
        AssertionError: If compute stays full-history, prefix columns drift, or validation fails.
    Side Effects:
        Rewrites signal artifacts under the inactive slot in `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    signal_targets = (("15m", "ma.ema"),)
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=30,
        mapping_tail_bars_1m=30,
        signal_tail_bars_1m=30,
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=False,
    )
    defaults_provider = _build_signal_test_defaults_provider_v2()
    grid_builder = _signal_grid_builder_v2()
    signal_rules_engine = BacktestSignalRulesEngineV2(defaults_provider=defaults_provider)
    request = _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)

    initial_compute = _DeterministicSignalCompute(grid_builder=grid_builder)
    initial_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=initial_compute,
        indicator_grid_builder=grid_builder,
    )
    initial_runner.export_canonical_price_1m(request)
    initial_matrix = _load_signal_matrix_v2(
        fixture=fixture,
        timeframe="15m",
        indicator_id="ma.ema",
    )

    updated_compute = _DeterministicSignalCompute(grid_builder=grid_builder)
    updated_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(
                bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2 - 30, _FULL_BUILD_MINUTES_V2)),
                price_offset=-1000.0,
            )
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=updated_compute,
        indicator_grid_builder=grid_builder,
    )
    updated_runner.export_canonical_price_1m(request)
    updated_matrix = _load_signal_matrix_v2(
        fixture=fixture,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    validation_result = BacktestArtifactManifestValidatorV2(
        artifact_loader=fixture.loader
    ).validate_slot(
        coordinates=fixture.coordinates,
        slot=fixture.inactive_slot,
        validation_spec=fixture.runtime_config.to_validation_spec(),
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert initial_compute.time_lengths == [288]
    assert len(updated_compute.time_lengths) == 1
    assert 2 <= updated_compute.time_lengths[0] < initial_compute.time_lengths[0]
    np.testing.assert_array_equal(updated_matrix[:, :-2], initial_matrix[:, :-2])
    assert not np.array_equal(updated_matrix[:, -2:], initial_matrix[:, -2:])
    assert validation_result.diagnostics == ()


def test_backtest_artifact_precompute_runner_v2_signal_tail_rebuild_is_byte_stable_for_identical_inputs(  # noqa: E501
    tmp_path: Path,
) -> None:
    """
    Verify identical R4-03 signal inputs produce byte-stable manifests and `signals.i8.npy`.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Finite-window test compute plus identical source data should keep the rebuilt tail exactly
        equal to the original full-build result.
    Raises:
        AssertionError: If repeated signal export changes bytes or unexpectedly recomputes the
            full signal timeline.
    Side Effects:
        Rewrites signal artifacts under the inactive slot in `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    signal_targets = (("15m", "ma.ema"),)
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=30,
        mapping_tail_bars_1m=30,
        signal_tail_bars_1m=30,
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=False,
    )
    defaults_provider = _build_signal_test_defaults_provider_v2()
    grid_builder = _signal_grid_builder_v2()
    signal_rules_engine = BacktestSignalRulesEngineV2(defaults_provider=defaults_provider)
    request = _request_v2(
        fixture=fixture,
        end_minute=_FULL_BUILD_MINUTES_V2,
        asof_date="2026-03-26",
        generated_at_utc="2026-03-26T03:00:00Z",
    )

    first_compute = _DeterministicSignalCompute(grid_builder=grid_builder)
    first_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=first_compute,
        indicator_grid_builder=grid_builder,
    )
    first_runner.export_canonical_price_1m(request)
    first_bytes = _read_signal_export_bytes_v2(
        fixture=fixture,
        signal_targets=signal_targets,
    )

    second_compute = _DeterministicSignalCompute(grid_builder=grid_builder)
    second_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(
                bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2 - 30, _FULL_BUILD_MINUTES_V2))
            )
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=second_compute,
        indicator_grid_builder=grid_builder,
    )
    second_runner.export_canonical_price_1m(request)
    second_bytes = _read_signal_export_bytes_v2(
        fixture=fixture,
        signal_targets=signal_targets,
    )

    assert first_compute.time_lengths == [288]
    assert len(second_compute.time_lengths) == 1
    assert 2 <= second_compute.time_lengths[0] < first_compute.time_lengths[0]
    assert first_bytes == second_bytes


def test_backtest_artifact_precompute_runner_v2_rejects_drifted_existing_signal_artifact(
    tmp_path: Path,
) -> None:
    """
    Verify R4-03 fails fast when an existing reusable signal file drifts from its manifest hash.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Missing files may trigger a full rebuild, but existing-file hash drift must stop the run.
    Raises:
        AssertionError: If drift is silently ignored or converted into a best-effort rebuild.
    Side Effects:
        Mutates one existing signal file in the inactive slot under `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    signal_targets = (("15m", "ma.ema"),)
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=30,
        mapping_tail_bars_1m=30,
        signal_tail_bars_1m=30,
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=False,
    )
    defaults_provider = _build_signal_test_defaults_provider_v2()
    grid_builder = _signal_grid_builder_v2()
    signal_rules_engine = BacktestSignalRulesEngineV2(defaults_provider=defaults_provider)
    request = _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)
    initial_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=_DeterministicSignalCompute(grid_builder=grid_builder),
        indicator_grid_builder=grid_builder,
    )
    initial_runner.export_canonical_price_1m(request)

    signal_paths = fixture.loader.resolve_signal_paths(
        fixture.coordinates,
        fixture.inactive_slot,
        "15m",
        "ma.ema",
    )
    corrupted_matrix = np.load(signal_paths.signals, allow_pickle=False)
    corrupted_matrix[0, -1] = np.int8(-1)
    with signal_paths.signals.open("wb") as handle:
        np.save(handle, corrupted_matrix, allow_pickle=False)

    updated_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(
                bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2 - 30, _FULL_BUILD_MINUTES_V2))
            )
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=_DeterministicSignalCompute(grid_builder=grid_builder),
        indicator_grid_builder=grid_builder,
    )

    with pytest.raises(ValueError, match="manifest sha256 must match actual file"):
        updated_runner.export_canonical_price_1m(request)


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
    reader = _FakeCanonicalCandleReader(rows=_build_canonical_rows_v2(bar_indexes=(0, 1, 1, 2)))
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


def test_backtest_artifact_precompute_runner_v2_build_publish_prices_mappings_flow_switches_pointer_in_stable_order(  # noqa: E501
    tmp_path: Path,
) -> None:
    """
    Verify R3-04 runs `precheck -> build inactive slot -> validate -> atomically switch
    current.yaml` with deterministic pointer payload ordering.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Source-of-truth runtime config still contains later-stage validation targets, while the
        R3-04 flow derives an explicit prices+mappings validation spec from that config.
    Raises:
        AssertionError: If the published pointer payload or build/publish result identity drifts.
    Side Effects:
        Builds and publishes one inactive slot under `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=2,
        validation_signal_artifacts=(("15m", "ma.ema"),),
        require_hit_times_manifest=True,
    )
    reader = _FakeCanonicalCandleReader(
        rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
    )
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=reader,
    )
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=fixture.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=fixture.builder),
        job_repository=cast(BacktestJobRepository, _ZeroBlockingRepositoryV2()),
        now_provider=lambda: datetime(2026, 3, 26, 3, 4, 5, tzinfo=timezone.utc),
    )

    result = publisher.build_publish_prices_mappings_slot(
        request=_request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2),
        precompute_runner=runner,
        validation_spec=fixture.runtime_config.to_prices_mappings_publish_validation_spec(),
    )

    assert len(reader.calls) == 1
    assert result.precheck.ready is True
    assert result.build_result.slot == fixture.inactive_slot
    assert result.build_result.slot_generation == 5
    assert result.publish_result.published_pointer.active_slot == fixture.inactive_slot
    assert result.publish_result.published_pointer.slot_generation == 5
    assert result.publish_result.validation.signal_manifests == ()
    assert result.publish_result.validation.hit_times_manifest is None
    assert result.validation_spec.signal_artifacts == ()
    assert result.validation_spec.require_hit_times_manifest is False
    assert (
        result.publish_result.published_pointer.manifest_sha256
        == result.build_result.manifest_sha256
    )
    assert _pointer_lines_v2(fixture.builder.current_pointer_path(fixture.coordinates)) == (
        "schema_version: 1",
        f"active_slot: {fixture.inactive_slot}",
        "slot_generation: 5",
        'asof_date: "2026-03-26"',
        f'manifest_sha256: "{result.build_result.manifest_sha256}"',
        'published_at_utc: "2026-03-26T03:04:05Z"',
    )


def test_backtest_artifact_precompute_runner_v2_full_validation_spec_still_rejects_missing_signals_and_hit_times(  # noqa: E501
    tmp_path: Path,
) -> None:
    """
    Verify later-stage full validation still fails deterministically for a runner-built R3-04 slot.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        R3-04 writes only `prices + mappings`, while the full validation plan still expects real
        `signals` and `hit_times` families.
    Raises:
        AssertionError: If publish unexpectedly switches `current.yaml` or diagnostics drift.
    Side Effects:
        Builds one inactive slot under `tmp_path` without publishing it.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=2,
        validation_signal_artifacts=(("15m", "ma.ema"),),
        require_hit_times_manifest=True,
    )
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
        ),
    )
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=fixture.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=fixture.builder),
        job_repository=cast(BacktestJobRepository, _ZeroBlockingRepositoryV2()),
    )

    runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)
    )
    precheck = publisher.precheck_publish(fixture.coordinates)

    with pytest.raises(ArtifactSlotPublishErrorV2) as error_info:
        publisher.publish(
            precheck=precheck,
            validation_spec=fixture.runtime_config.to_validation_spec(),
            asof_date="2026-03-26",
        )

    current_pointer = fixture.loader.load_current_pointer(fixture.coordinates)

    assert error_info.value.code == "slot_validation_failed"
    assert error_info.value.diagnostics[0].code == "root_manifest_signal_targets_mismatch"
    assert current_pointer.active_slot == fixture.active_slot
    assert current_pointer.slot_generation == 4


def _load_signal_matrix_v2(
    *,
    fixture: ArtifactPrecomputeFixtureV2,
    timeframe: str,
    indicator_id: str,
) -> np.ndarray:
    """
    Load one strict `signals/<tf>/<indicator_id>/signals.i8.npy` matrix for assertions.

    Args:
        fixture: Minimal strict precompute fixture.
        timeframe: Signal timeframe literal.
        indicator_id: Signal indicator identifier.
    Returns:
        np.ndarray: Loaded compact `int8` signal matrix with shape `[V, T_tf]`.
    Assumptions:
        The caller already materialized the target signal artifact into the inactive slot.
    Raises:
        FileNotFoundError: If the deterministic signal path is missing.
        ValueError: If numpy cannot load the stored `.npy` payload.
    Side Effects:
        Reads one signal matrix from disk.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    return np.load(
        fixture.loader.resolve_signal_paths(
            fixture.coordinates,
            fixture.inactive_slot,
            timeframe,
            indicator_id,
        ).signals,
        allow_pickle=False,
    )


def _read_signal_export_bytes_v2(
    *,
    fixture: ArtifactPrecomputeFixtureV2,
    signal_targets: tuple[tuple[str, str], ...],
) -> tuple[bytes, ...]:
    """
    Read deterministic per-target signal manifest bytes and signal matrix bytes.

    Args:
        fixture: Minimal strict precompute fixture.
        signal_targets: Explicit ordered `(timeframe, indicator_id)` signal targets.
    Returns:
        tuple[bytes, ...]: Stable byte snapshots for per-target signal files.
    Assumptions:
        Identical rebuild inputs with fixed `generated_at_utc` should keep emitted file bytes
        unchanged across repeated runs.
    Raises:
        FileNotFoundError: If one emitted file is missing.
        OSError: If one file cannot be read.
    Side Effects:
        Reads manifest and `.npy` files from disk.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    snapshots: list[bytes] = []
    for timeframe, indicator_id in signal_targets:
        signal_paths = fixture.loader.resolve_signal_paths(
            fixture.coordinates,
            fixture.inactive_slot,
            timeframe,
            indicator_id,
        )
        snapshots.append(signal_paths.manifest.read_bytes())
        snapshots.append(signal_paths.signals.read_bytes())
    return tuple(snapshots)


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


def _pointer_lines_v2(path: Path) -> tuple[str, ...]:
    """
    Read the strict `current.yaml` payload as raw lines for ordering assertions.

    Args:
        path: Absolute pointer path written by the atomic current-pointer writer.
    Returns:
        tuple[str, ...]: Non-empty YAML lines in on-disk order without trailing newlines.
    Assumptions:
        R3-04 keeps serialized field order stable as
        `schema_version -> active_slot -> slot_generation -> asof_date -> manifest_sha256 ->
        published_at_utc`.
    Raises:
        OSError: If `current.yaml` cannot be read.
    Side Effects:
        Reads one UTF-8 file from disk.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/current_pointer_writer.py
    """
    return tuple(line for line in path.read_text(encoding="utf-8").splitlines() if line != "")


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
