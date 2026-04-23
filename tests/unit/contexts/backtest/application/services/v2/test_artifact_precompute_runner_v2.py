from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, cast

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
from trading.contexts.backtest.application.services.signals_from_indicators_v1 import (
    indicator_primary_output_series_from_tensor_v1,
)
from trading.contexts.backtest_artifacts.application.services.v2 import (
    artifact_precompute_runner as artifact_precompute_runner_module,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_validator import (  # noqa: E501
    BacktestArtifactManifestValidatorV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_precompute_coordinator import (  # noqa: E501
    ArtifactPrecomputeCoordinatorV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_precompute_runner import (
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCanonicalPriceExportResultV2,
    BacktestArtifactPrecomputeRunnerV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_publisher import (
    ArtifactSlotPublishErrorV2,
    BacktestArtifactSlotPublisherV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ARTIFACT_MAPPING_TIMEFRAMES_V2,
    ARTIFACT_PRICE_TIMEFRAMES_V2,
    ARTIFACT_SIGNAL_TIMEFRAMES_V2,
    HIT_TIMES_TIMEFRAME_LITERAL_V2,
    SIGNAL_FEATURE_NAMES_V2,
    ArtifactCoordinatesV2,
    ArtifactPrecomputeExecutionPolicyV2,
    ArtifactSignalChunkPlanningRequestV2,
    ArtifactSignalValidationSpecV2,
    ArtifactStageRebuildStatsV2,
    ArtifactTailRebuildBarsV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.signal_chunk_planner_v2 import (
    DeterministicSignalChunkPlannerV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.signal_rules_engine_v2 import (
    BacktestSignalRulesEngineV2,
    SignalRuleEvaluationRequestV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.signal_rules_engine_v2 import (
    supported_indicator_ids_for_signal_rules_v2 as supported_indicator_ids_for_signals_v1,
)
from trading.contexts.indicators.adapters.outbound.registry import YamlIndicatorRegistry
from trading.contexts.indicators.application.dto import (
    CandleArrays,
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
from trading.contexts.market_data.application.dto import (
    CandleWithMeta,
    CanonicalCandleBatch1m,
)
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)

_BASE_TIME_UTC = datetime(2026, 3, 26, 0, 0, tzinfo=timezone.utc)
_FULL_BUILD_DAYS_V2 = 3
_FULL_BUILD_MINUTES_V2 = _FULL_BUILD_DAYS_V2 * 24 * 60
_HIT_TIMES_MINUTES_PER_BAR_V2 = int(
    Timeframe(HIT_TIMES_TIMEFRAME_LITERAL_V2).duration().total_seconds() // 60
)
_CANONICAL_WIDENED_TP_LEVELS_PCT_V2 = tuple(value / 2.0 for value in range(1, 101))
_CANONICAL_WIDENED_SL_LEVELS_PCT_V2 = tuple(value / 2.0 for value in range(1, 51))
_TARGET_VARIANT_CEILINGS_V2 = {
    "momentum.trix": 1000,
    "momentum.stoch": 150,
    "trend.adx": 300,
    "volatility.hv": 350,
    "momentum.roc": 250,
    "momentum.rsi": 250,
    "volatility.stddev": 250,
    "volatility.variance": 250,
    "trend.psar": 50,
    "structure.pivots": 30,
    "trend.linreg_slope": 350,
    "structure.distance_to_ma_norm": 350,
    "structure.percent_rank": 350,
    "structure.zscore": 350,
}


def _hit_times_bars_from_one_minute_bars_v2(one_minute_bars: int) -> int:
    return one_minute_bars // _HIT_TIMES_MINUTES_PER_BAR_V2


def _hit_times_tail_bars_from_one_minute_tail_v2(one_minute_tail_bars: int) -> int:
    return (
        one_minute_tail_bars + _HIT_TIMES_MINUTES_PER_BAR_V2 - 1
    ) // _HIT_TIMES_MINUTES_PER_BAR_V2
_EXPECTED_VARIANT_COUNTS_BY_ENV_V2 = {
    "dev": {
        "momentum.roc": 60,
        "momentum.rsi": 60,
        "momentum.stoch": 30,
        "momentum.trix": 192,
        "structure.distance_to_ma_norm": 48,
        "structure.percent_rank": 48,
        "structure.pivots": 25,
        "structure.zscore": 48,
        "trend.adx": 24,
        "trend.linreg_slope": 48,
        "trend.psar": 16,
        "volatility.hv": 96,
        "volatility.stddev": 48,
        "volatility.variance": 48,
    },
    "prod": {
        "momentum.roc": 60,
        "momentum.rsi": 60,
        "momentum.stoch": 30,
        "momentum.trix": 192,
        "structure.distance_to_ma_norm": 48,
        "structure.percent_rank": 48,
        "structure.pivots": 25,
        "structure.zscore": 48,
        "trend.adx": 24,
        "trend.linreg_slope": 48,
        "trend.psar": 16,
        "volatility.hv": 96,
        "volatility.stddev": 48,
        "volatility.variance": 48,
    },
    "test": {
        "momentum.roc": 6,
        "momentum.rsi": 6,
        "momentum.stoch": 1,
        "momentum.trix": 6,
        "structure.distance_to_ma_norm": 6,
        "structure.percent_rank": 6,
        "structure.pivots": 1,
        "structure.zscore": 6,
        "trend.adx": 1,
        "trend.linreg_slope": 6,
        "trend.psar": 1,
        "volatility.hv": 6,
        "volatility.stddev": 6,
        "volatility.variance": 6,
    },
}
_ZERO_AXIS_SIGNAL_TARGETS_V2 = (
    ("15m", "structure.candle_stats"),
    ("15m", "volatility.tr"),
    ("15m", "volume.ad_line"),
    ("15m", "volume.obv"),
)


@dataclass(frozen=True, slots=True)
class _PrecomputeSignalDefaultsProvider:
    """
    Defaults-provider wrapper overriding only compute grids for small R4-02 test matrices.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
        """
        return self.delegate.allowed_source_values(indicator_id=indicator_id)


@dataclass(frozen=True, slots=True)
class _MissingComputeDefaultsProviderV2:
    """
    Provider wrapper hiding selected compute defaults while preserving all other contracts.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/defaults/
        indicators_yaml_defaults_provider.py
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """

    delegate: YamlBacktestGridDefaultsProvider
    hidden_indicator_ids: tuple[str, ...]

    def compute_defaults(self, *, indicator_id: str) -> GridSpec | None:
        """
        Hide compute defaults for the selected indicator ids.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            GridSpec | None: `None` for hidden ids, otherwise the delegate-provided defaults.
        Assumptions:
            Hidden ids are normalized lower-case literals selected by the test case.
        Raises:
            ValueError: Propagated from the delegate provider.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/defaults/
            indicators_yaml_defaults_provider.py
        """
        normalized_indicator_id = indicator_id.strip().lower()
        if normalized_indicator_id in self.hidden_indicator_ids:
            return None
        return self.delegate.compute_defaults(indicator_id=normalized_indicator_id)

    def signal_param_defaults(self, *, indicator_id: str) -> Mapping[str, GridParamSpec]:
        """
        Delegate signal default resolution unchanged.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            Mapping[str, GridParamSpec]: Canonical signal defaults mapping.
        Assumptions:
            Tests modify only compute-default availability.
        Raises:
            ValueError: Propagated from the delegate provider.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/defaults/
            indicators_yaml_defaults_provider.py
        """
        return self.delegate.signal_param_defaults(indicator_id=indicator_id)

    def supported_indicator_ids(self) -> tuple[str, ...]:
        """
        Return the unchanged supported indicator catalog.

        Args:
            None.
        Returns:
            tuple[str, ...]: Canonical supported indicator ids.
        Assumptions:
            Signal registry membership must remain unchanged while compute defaults are hidden.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
        """
        return self.delegate.supported_indicator_ids()

    def allowed_source_values(self, *, indicator_id: str) -> tuple[str, ...]:
        """
        Delegate allowed-source lookup unchanged.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            tuple[str, ...]: Canonical allowed source literals.
        Assumptions:
            Hiding compute defaults must not alter source catalogs.
        Raises:
            ValueError: Propagated from the delegate provider.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/defaults/
            indicators_yaml_defaults_provider.py
        """
        return self.delegate.allowed_source_values(indicator_id=indicator_id)


@dataclass(frozen=True, slots=True)
class _DeterministicSignalCompute:
    """
    Small deterministic compute adapter producing rolling-mean tensors for signal tests.

    Docs:
      - docs/architecture/indicators/indicators-overview.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """

    grid_builder: GridBuilder
    time_lengths: list[int] = field(default_factory=list, compare=False)
    variant_counts: list[int] = field(default_factory=list, compare=False)

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
            Allocates one small in-memory tensor and records requested timeline/variant sizes.
        Docs:
          - docs/architecture/backtest/README.md
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
        self.variant_counts.append(materialized.variants)
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


class _SnapshotRehydratingSignalComputeFactoryV2:
    """
    Rehydrate deterministic test compute from a same-process worker snapshot.

    Docs:
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    @classmethod
    def from_signal_chunk_worker_snapshot_v2(
        cls,
        *,
        snapshot: Mapping[str, object],
    ) -> _DeterministicSignalCompute:
        """
        Rebuild deterministic test compute from the captured grid-builder snapshot.

        Args:
            snapshot: Same-process worker snapshot containing the shared `GridBuilder`.
        Returns:
            _DeterministicSignalCompute: Deterministic compute adapter for fake worker execution.
        Assumptions:
            The fake process-pool test executes in one process and may reuse Python objects
            directly.
        Raises:
            KeyError: If the snapshot omits `grid_builder`.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_artifact_precompute_runner_v2.py
        """
        del cls
        return _DeterministicSignalCompute(
            grid_builder=cast(GridBuilder, snapshot["grid_builder"])
        )


@dataclass(frozen=True, slots=True, eq=False)
class _ImmediateFutureV2:
    """
    Resolved future stub used by fake spawned-worker tests.

    Docs:
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """

    value: object

    def result(self) -> object:
        """
        Return the already computed synchronous result.

        Args:
            None.
        Returns:
            object: Precomputed future payload.
        Assumptions:
            Fake executor tests run chunk work eagerly in the caller process.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_artifact_precompute_runner_v2.py
        """
        return self.value


class _FakeProcessPoolExecutorV2:
    """
    Synchronous `ProcessPoolExecutor` stub capturing worker bootstrap and submit kwargs.

    Docs:
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    submitted_kwargs: list[Mapping[str, object]] = []
    initializer_bootstraps: list[object] = []

    def __init__(
        self,
        *,
        max_workers: int,
        mp_context: object,
        initializer: Callable[..., object] | None = None,
        initargs: tuple[object, ...] = (),
    ) -> None:
        """
        Store fake process-pool bootstrap inputs for later synchronous execution.

        Args:
            max_workers: Requested worker count, unused beyond interface compatibility.
            mp_context: Requested multiprocessing context, unused in the fake executor.
            initializer: Optional worker initializer callable.
            initargs: Optional initializer arguments.
        Returns:
            None.
        Assumptions:
            The fake executor only needs interface compatibility and captured bootstrap payloads.
        Raises:
            None.
        Side Effects:
            Stores the initializer and its arguments in memory.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_artifact_precompute_runner_v2.py
        """
        del max_workers, mp_context
        self._initializer = initializer
        self._initargs = initargs

    def __enter__(self) -> "_FakeProcessPoolExecutorV2":
        """
        Run the captured worker initializer once and return the fake executor.

        Args:
            None.
        Returns:
            _FakeProcessPoolExecutorV2: This fake executor instance.
        Assumptions:
            The regression test only needs one shared bootstrap event for the in-process stub.
        Raises:
            Exception: Propagates any initializer failure unchanged.
        Side Effects:
            Captures the bootstrap payload and initializes worker-local state in-process.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_artifact_precompute_runner_v2.py
        """
        if self._initializer is not None:
            _FakeProcessPoolExecutorV2.initializer_bootstraps.append(self._initargs[0])
            self._initializer(*self._initargs)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: object | None,
    ) -> bool:
        """
        Propagate exceptions from the fake executor context body unchanged.

        Args:
            exc_type: Exception type raised inside the context body, if any.
            exc: Exception instance raised inside the context body, if any.
            exc_tb: Traceback raised inside the context body, if any.
        Returns:
            bool: Always `False` so the caller still sees any failure.
        Assumptions:
            The fake executor should mirror the real context-manager propagation behavior.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_artifact_precompute_runner_v2.py
        """
        del exc_type, exc, exc_tb
        return False

    def submit(
        self,
        fn: object,
        /,
        *args: object,
        **kwargs: object,
    ) -> _ImmediateFutureV2:
        """
        Execute one submitted chunk job synchronously and capture the submitted kwargs.

        Args:
            fn: Submitted callable.
            *args: Positional arguments for the callable.
            **kwargs: Keyword arguments for the callable.
        Returns:
            _ImmediateFutureV2: Already resolved synchronous future stub.
        Assumptions:
            Regression coverage needs the exact submitted kwargs more than true multiprocessing.
        Raises:
            Exception: Propagates any submitted callable failure unchanged.
        Side Effects:
            Appends the submitted kwargs to the shared in-memory capture list.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_artifact_precompute_runner_v2.py
        """
        _FakeProcessPoolExecutorV2.submitted_kwargs.append(dict(kwargs))
        return _ImmediateFutureV2(value=cast(Any, fn)(*args, **kwargs))


class _FakeCanonicalCandleReader:
    """
    Deterministic in-memory canonical candle reader for R3-01 precompute tests.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
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

    def read_1m_arrays(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> CanonicalCandleBatch1m:
        """
        Return one strict columnar batch for the requested `TimeRange [start, end)`.

        Args:
            instrument_id: Ignored shared-kernel identity passed by the production runner.
            time_range: Source reread window requested by the runner.
        Returns:
            CanonicalCandleBatch1m: Filtered canonical candle batch aligned to the requested
                half-open interval.
        Assumptions:
            Tests validate time-range behavior and precompute ordering, not transport shape.
        Raises:
            None.
        Side Effects:
            Appends the requested time range to the same in-memory call log as `read_1m(...)`.
        """
        del instrument_id
        self.calls.append(time_range)
        rows = tuple(
            row
            for row in self._rows
            if time_range.start.value <= row.candle.ts_open.value < time_range.end.value
        )
        row_count = len(rows)
        if row_count == 0:
            return CanonicalCandleBatch1m(
                open_time_ms=np.empty(0, dtype=np.int64),
                close_time_ms=np.empty(0, dtype=np.int64),
                ohlcv_f32=np.empty((0, 5), dtype=np.float32),
            )
        return CanonicalCandleBatch1m(
            open_time_ms=np.ascontiguousarray(
                [int(row.candle.ts_open.value.timestamp() * 1000) for row in rows],
                dtype=np.int64,
            ),
            close_time_ms=np.ascontiguousarray(
                [int(row.candle.ts_close.value.timestamp() * 1000) for row in rows],
                dtype=np.int64,
            ),
            ohlcv_f32=np.ascontiguousarray(
                [
                    [
                        float(row.candle.open),
                        float(row.candle.high),
                        float(row.candle.low),
                        float(row.candle.close),
                        float(row.candle.volume_base),
                    ]
                    for row in rows
                ],
                dtype=np.float32,
            ),
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
          - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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


def _build_long_window_signal_test_defaults_provider_v2() -> _PrecomputeSignalDefaultsProvider:
    """
    Build a deterministic defaults provider with warmup-heavy signal windows for tail proofs.

    Args:
        None.
    Returns:
        _PrecomputeSignalDefaultsProvider: Wrapper forcing one long-window `ma.sma` grid.
    Assumptions:
        The long-window scenario must materially exceed the naive `signal_tail_bars_1m` budget.
    Raises:
        FileNotFoundError: If `configs/prod/indicators.yaml` is unavailable.
    Side Effects:
        Reads the repository-local prod defaults YAML.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """
    delegate = YamlBacktestGridDefaultsProvider.from_yaml(
        config_path=Path("configs/prod/indicators.yaml")
    )
    overrides = {
        "ma.sma": GridSpec(
            indicator_id=IndicatorId("ma.sma"),
            params={"window": ExplicitValuesSpec(name="window", values=(40,))},
            source=ExplicitValuesSpec(name="source", values=("close",)),
        )
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
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/indicators/adapters/outbound/registry/yaml_indicator_registry.py
      - src/trading/contexts/indicators/application/services/grid_builder.py
    """
    registry = YamlIndicatorRegistry.from_yaml(
        defs=all_defs(),
        config_path=Path("configs/test/indicators.yaml"),
    )
    return GridBuilder(registry=registry)


def _variant_counts_for_env_v2(*, env_name: str) -> dict[str, int]:
    """
    Materialize real compute-grid variant counts for the narrowed indicator families.

    Args:
        env_name: Environment name under `configs/<env>/indicators.yaml`.
    Returns:
        dict[str, int]: Deterministic indicator-id -> materialized variant count mapping.
    Assumptions:
        The checked-in indicators YAML is valid for both defaults-provider and registry loading.
    Raises:
        FileNotFoundError: If the env-specific indicators YAML is missing.
        ValueError: If one grid cannot be materialized deterministically.
    Side Effects:
        Reads the repository-local config file and materializes indicator grids in memory.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - configs/dev/indicators.yaml
      - configs/prod/indicators.yaml
      - configs/test/indicators.yaml
      - src/trading/contexts/backtest/adapters/outbound/defaults/
        indicators_yaml_defaults_provider.py
    """
    config_path = Path(f"configs/{env_name}/indicators.yaml")
    defaults_provider = YamlBacktestGridDefaultsProvider.from_yaml(config_path=config_path)
    grid_builder = GridBuilder(
        registry=YamlIndicatorRegistry.from_yaml(defs=all_defs(), config_path=config_path)
    )
    counts: dict[str, int] = {}
    for indicator_id in _TARGET_VARIANT_CEILINGS_V2:
        compute_grid = defaults_provider.compute_defaults(indicator_id=indicator_id)
        assert compute_grid is not None
        materialized_grid = grid_builder.materialize_indicator(grid=compute_grid)
        counts[indicator_id] = materialized_grid.variants
    return counts


def _candle_arrays_from_loaded_prices_v2(
    *,
    timeframe: str,
    open_time: np.ndarray,
    close_time: np.ndarray,
    ohlcv: np.ndarray,
) -> CandleArrays:
    """
    Convert loaded strict price arrays into `CandleArrays` for deterministic test compute.

    Args:
        timeframe: Price timeframe literal represented by the arrays.
        open_time: Epoch-millis open timestamps.
        close_time: Epoch-millis close timestamps.
        ohlcv: Strict `[T, 5]` float32 OHLCV matrix.
    Returns:
        CandleArrays: Dense candle arrays aligned to the provided timeline.
    Assumptions:
        Test fixtures use the same single instrument identity for all artifact exports.
    Raises:
        ValueError: If one provided array violates `CandleArrays` constructor invariants.
    Side Effects:
        Allocates contiguous arrays for test compute.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(
            start=UtcTimestamp(datetime.fromtimestamp(int(open_time[0]) / 1000.0, timezone.utc)),
            end=UtcTimestamp(datetime.fromtimestamp(int(close_time[-1]) / 1000.0, timezone.utc)),
        ),
        timeframe=Timeframe(timeframe),
        ts_open=np.ascontiguousarray(open_time, dtype=np.int64),
        open=np.ascontiguousarray(ohlcv[:, 0], dtype=np.float32),
        high=np.ascontiguousarray(ohlcv[:, 1], dtype=np.float32),
        low=np.ascontiguousarray(ohlcv[:, 2], dtype=np.float32),
        close=np.ascontiguousarray(ohlcv[:, 3], dtype=np.float32),
        volume=np.ascontiguousarray(ohlcv[:, 4], dtype=np.float32),
    )


def _time_axis_prefix_sha256_v2(*, array: np.ndarray, prefix_bars: int) -> str:
    """
    Hash the unchanged time-axis prefix of one artifact array for byte-stability proofs.

    Args:
        array: Artifact array whose time axis is the last dimension or the only dimension.
        prefix_bars: Number of leading time bars to include in the digest.
    Returns:
        str: Lowercase SHA-256 digest of the selected prefix bytes.
    Assumptions:
        Artifact arrays in these proofs are either `[T]`, `[T, 5]`, or `[level, T]`/`[V, T]`.
    Raises:
        ValueError: If `prefix_bars` is outside the array time-axis bounds.
    Side Effects:
        Allocates one contiguous prefix slice before hashing.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """
    if prefix_bars < 0:
        raise ValueError(f"prefix_bars must be >= 0; got {prefix_bars!r}")
    if array.ndim == 1:
        time_axis_size = int(array.shape[0])
    elif array.ndim == 2 and array.shape[1] == 5:
        time_axis_size = int(array.shape[0])
    else:
        time_axis_size = int(array.shape[-1])
    if prefix_bars > time_axis_size:
        raise ValueError(
            f"prefix_bars must be <= time axis size {time_axis_size}; got {prefix_bars!r}"
        )
    if array.ndim == 1:
        prefix = np.ascontiguousarray(array[:prefix_bars])
    elif array.ndim == 2 and array.shape[1] == 5:
        prefix = np.ascontiguousarray(array[:prefix_bars, :])
    else:
        prefix = np.ascontiguousarray(array[..., :prefix_bars])
    return sha256(prefix.tobytes()).hexdigest()


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
      - docs/architecture/backtest/README.md
      - docs/architecture/indicators/indicators-overview.md
    Related:
      - src/trading/contexts/indicators/application/dto/candle_arrays.py
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py
    """
    normalized_source = source_name.strip().lower()
    candle_arrays = candles
    if normalized_source == "close":
        source = cast(np.ndarray, getattr(candle_arrays, "close"))
    elif normalized_source == "open":
        source = cast(np.ndarray, getattr(candle_arrays, "open"))
    elif normalized_source == "high":
        source = cast(np.ndarray, getattr(candle_arrays, "high"))
    elif normalized_source == "low":
        source = cast(np.ndarray, getattr(candle_arrays, "low"))
    elif normalized_source == "hlc3":
        source = (
            cast(np.ndarray, getattr(candle_arrays, "high"))
            + cast(np.ndarray, getattr(candle_arrays, "low"))
            + cast(np.ndarray, getattr(candle_arrays, "close"))
        ) / np.float32(3.0)
    elif normalized_source == "ohlc4":
        source = (
            cast(np.ndarray, getattr(candle_arrays, "open"))
            + cast(np.ndarray, getattr(candle_arrays, "high"))
            + cast(np.ndarray, getattr(candle_arrays, "low"))
            + cast(np.ndarray, getattr(candle_arrays, "close"))
        ) / np.float32(4.0)
    else:
        raise ValueError(f"unsupported synthetic source literal: {normalized_source!r}")
    return np.ascontiguousarray(source, dtype=np.float32)


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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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


def test_deterministic_signal_chunk_planner_v2_uses_expected_chunk_count_and_ranges() -> None:
    """
    Verify ChunkPlanner emits stable contiguous row ranges for a bounded worker budget.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        A `64`-row worker budget for `1200` variants should produce `ceil(1200 / 64) = 19`
        deterministic chunk jobs.
    Raises:
        AssertionError: If chunk count, row coverage, or stable ordering drifts.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_chunk_planner_v2.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    planner = DeterministicSignalChunkPlannerV2()

    jobs = planner.plan(
        request=ArtifactSignalChunkPlanningRequestV2(
            indicator_id="ma.ema",
            timeframe="15m",
            timeline_bar_count=288,
            variant_count=1200,
            estimated_bytes_per_row=1024,
            worker_memory_budget_bytes=64 * 1024,
            signal_chunk_rows_min=32,
            signal_chunk_rows_max=256,
        )
    )

    assert len(jobs) == 19
    assert jobs[0].chunk_index == 0
    assert jobs[0].chunk_count == 19
    assert jobs[0].row_start_inclusive == 0
    assert jobs[0].row_end_exclusive == 64
    assert jobs[0].chunk_rows == 64
    assert jobs[1].row_start_inclusive == 64
    assert jobs[1].row_end_exclusive == 128
    assert jobs[-1].chunk_index == 18
    assert jobs[-1].chunk_count == 19
    assert jobs[-1].row_start_inclusive == 1152
    assert jobs[-1].row_end_exclusive == 1200
    assert jobs[-1].chunk_rows == 48
    assert tuple(
        (job.row_start_inclusive, job.row_end_exclusive) for job in jobs
    ) == tuple(
        (index * 64, min(1200, (index + 1) * 64))
        for index in range(19)
    )
    assert sum(job.chunk_rows for job in jobs) == 1200


def test_build_artifact_precompute_fixture_v2_expands_all_supported_v1_signal_targets(
    tmp_path: Path,
) -> None:
    """
    Verify synthetic fixtures preserve full-registry `all_supported_v1` target expansion.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Fixture helpers should reuse the real config loader's deterministic expansion order.
    Raises:
        AssertionError: If validation-plan or runtime-settings signal targets drift.
    Side Effects:
        Writes one strict config under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        validation_signal_artifacts="all_supported_v1",
        precompute_signal_artifacts="all_supported_v1",
    )
    expected_targets = tuple(
        ArtifactSignalValidationSpecV2(timeframe=timeframe, indicator_id=indicator_id)
        for timeframe in ARTIFACT_SIGNAL_TIMEFRAMES_V2
        for indicator_id in supported_indicator_ids_for_signals_v1()
    )

    assert (
        tuple(
            item.to_validation_spec()
            for item in fixture.runtime_config.validation_plan.signal_artifacts
        )
        == expected_targets
    )
    assert fixture.runtime_settings.signal_artifacts == expected_targets


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
      - docs/architecture/backtest/README.md
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
    hit_times_manifest = fixture.loader.load_hit_times_manifest(
        fixture.coordinates,
        fixture.inactive_slot,
    )
    tp_values, sl_values, long_tp, long_sl, short_tp, short_sl = _load_hit_times_arrays_v2(
        fixture=fixture,
    )
    expected_tp_level_count = len(fixture.runtime_settings.hit_times_tp_levels_pct)
    expected_sl_level_count = len(fixture.runtime_settings.hit_times_sl_levels_pct)
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
    assert (
        manifest.hit_times.manifest_path
        == f"hit_times/{HIT_TIMES_TIMEFRAME_LITERAL_V2}/manifest.yaml"
    )
    assert manifest.hit_times.manifest_sha256 != "0" * 64
    expected_hit_times_bar_count = _hit_times_bars_from_one_minute_bars_v2(_FULL_BUILD_MINUTES_V2)
    assert hit_times_manifest.timeline_bar_count == expected_hit_times_bar_count
    assert hit_times_manifest.sentinel_index == expected_hit_times_bar_count
    assert tp_values.dtype == np.float32
    assert sl_values.dtype == np.float32
    assert long_tp.dtype == np.uint32
    assert long_sl.dtype == np.uint32
    assert short_tp.dtype == np.uint32
    assert short_sl.dtype == np.uint32
    assert hit_times_manifest.tp_values.shape == (expected_tp_level_count,)
    assert hit_times_manifest.sl_values.shape == (expected_sl_level_count,)
    assert hit_times_manifest.long_tp.array.shape == (
        expected_tp_level_count,
        expected_hit_times_bar_count,
    )
    assert hit_times_manifest.long_sl.array.shape == (
        expected_sl_level_count,
        expected_hit_times_bar_count,
    )
    assert hit_times_manifest.short_tp.array.shape == (
        expected_tp_level_count,
        expected_hit_times_bar_count,
    )
    assert hit_times_manifest.short_sl.array.shape == (
        expected_sl_level_count,
        expected_hit_times_bar_count,
    )
    assert tp_values.shape == (expected_tp_level_count,)
    assert sl_values.shape == (expected_sl_level_count,)
    assert long_tp.shape == (expected_tp_level_count, expected_hit_times_bar_count)
    assert long_sl.shape == (expected_sl_level_count, expected_hit_times_bar_count)
    assert short_tp.shape == (expected_tp_level_count, expected_hit_times_bar_count)
    assert short_sl.shape == (expected_sl_level_count, expected_hit_times_bar_count)
    assert np.all(np.diff(tp_values) > 0)
    assert np.all(np.diff(sl_values) > 0)
    assert np.all(long_tp <= expected_hit_times_bar_count)
    assert np.all(long_sl <= expected_hit_times_bar_count)
    assert np.all(short_tp <= expected_hit_times_bar_count)
    assert np.all(short_sl <= expected_hit_times_bar_count)
    assert np.all(long_tp[1:, :] >= long_tp[:-1, :])
    assert np.all(long_sl[1:, :] >= long_sl[:-1, :])
    assert np.all(short_tp[1:, :] >= short_tp[:-1, :])
    assert np.all(short_sl[1:, :] >= short_sl[:-1, :])
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


def test_backtest_artifact_precompute_runner_v2_builds_widened_hit_times_manifest_shapes(
    tmp_path: Path,
) -> None:
    """
    Verify the runner materializes the exact widened canonical `hit_times/15m` shape contract.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Milestone C widens the canonical artifact grid without changing runtime consumers in this
        prompt, so the precompute runner must emit the full widened manifest/table shapes itself.
    Raises:
        AssertionError: If the widened grid is not written into `tp_values`, `sl_values`, or the
            four strict hit-times tables exactly.
    Side Effects:
        Builds one inactive-slot export under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - configs/prod/backtest_artifacts.yaml
    """
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        hit_times_tp_levels_pct=_CANONICAL_WIDENED_TP_LEVELS_PCT_V2,
        hit_times_sl_levels_pct=_CANONICAL_WIDENED_SL_LEVELS_PCT_V2,
        max_hit_times_cells=1_500_000,
        max_hit_times_cells_full_rebuild=1_500_000,
    )
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
        ),
    )

    runner.export_canonical_price_1m(
        _request_v2(
            fixture=fixture,
            end_minute=_FULL_BUILD_MINUTES_V2,
        )
    )

    hit_times_manifest = fixture.loader.load_hit_times_manifest(
        fixture.coordinates,
        fixture.inactive_slot,
    )
    tp_values, sl_values, long_tp, long_sl, short_tp, short_sl = _load_hit_times_arrays_v2(
        fixture=fixture,
    )

    np.testing.assert_allclose(
        tp_values,
        np.asarray(_CANONICAL_WIDENED_TP_LEVELS_PCT_V2, dtype=np.float32) / np.float32(100.0),
    )
    np.testing.assert_allclose(
        sl_values,
        np.asarray(_CANONICAL_WIDENED_SL_LEVELS_PCT_V2, dtype=np.float32) / np.float32(100.0),
    )
    assert hit_times_manifest.tp_values.shape == (100,)
    assert hit_times_manifest.sl_values.shape == (50,)
    expected_hit_times_bar_count = _hit_times_bars_from_one_minute_bars_v2(_FULL_BUILD_MINUTES_V2)
    assert hit_times_manifest.long_tp.array.shape == (100, expected_hit_times_bar_count)
    assert hit_times_manifest.long_sl.array.shape == (50, expected_hit_times_bar_count)
    assert hit_times_manifest.short_tp.array.shape == (100, expected_hit_times_bar_count)
    assert hit_times_manifest.short_sl.array.shape == (50, expected_hit_times_bar_count)
    assert long_tp.shape == (100, expected_hit_times_bar_count)
    assert long_sl.shape == (50, expected_hit_times_bar_count)
    assert short_tp.shape == (100, expected_hit_times_bar_count)
    assert short_sl.shape == (50, expected_hit_times_bar_count)


def test_artifact_precompute_coordinator_v2_rejects_nested_timeframe_sessions_when_limited_to_one(
) -> None:
    """
    Verify `max_open_timeframe_sessions=1` prevents accidental nested timeframe sessions.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R12 keeps session ownership explicit even before chunked signal workers are introduced.
    Raises:
        AssertionError: If a nested second session does not raise ValueError.
    Side Effects:
        Emits in-memory coordinator logs during the context-manager lifecycle.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_coordinator.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    coordinator = ArtifactPrecomputeCoordinatorV2(
        coordinates=ArtifactCoordinatesV2(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        ),
        slot="slot_a",
        slot_generation=1,
        force_full_rebuild=False,
        execution_policy=ArtifactPrecomputeExecutionPolicyV2(
            max_open_timeframe_sessions=1,
            signal_worker_processes=4,
            signal_worker_memory_budget_bytes=2_147_483_648,
            signal_chunk_rows_min=32,
            signal_chunk_rows_max=256,
        ),
    )

    with coordinator.open_timeframe_session(timeframe="15m"):
        with pytest.raises(ValueError, match="max_open_timeframe_sessions"):
            with coordinator.open_timeframe_session(timeframe="1h"):
                pass


def test_backtest_artifact_precompute_runner_v2_emits_deterministic_stage_results_and_progress_events(  # noqa: E501
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Verify the R12 runner emits ordered stage results and structured progress events.

    Args:
        tmp_path: pytest temporary path fixture.
        caplog: pytest log capture fixture.
    Returns:
        None.
    Assumptions:
        Progress events are emitted through structured logs while `stage_results` stay attached
        to the internal export DTO.
    Raises:
        AssertionError: If stage order, timeframe order, or required progress events drift.
    Side Effects:
        Creates strict artifact files under `tmp_path` and captures runner logs.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_coordinator.py
    """
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=2,
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

    with caplog.at_level(logging.INFO):
        result = runner.export_canonical_price_1m(
            _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)
        )

    stage_ids = tuple(stage_result.stage_output.stage for stage_result in result.stage_results)

    assert len(result.stage_results) == len(ARTIFACT_MAPPING_TIMEFRAMES_V2) + 3
    assert stage_ids[0] == "canonical_prices"
    assert stage_ids[1] == "hit_times"
    assert stage_ids[2 : 2 + len(ARTIFACT_MAPPING_TIMEFRAMES_V2)] == (
        "timeframe_session",
    ) * len(ARTIFACT_MAPPING_TIMEFRAMES_V2)
    assert stage_ids[-1] == "root_manifest"
    assert result.stage_results[2].stage_output.current_timeframe == "15m"
    assert result.stage_results[-2].stage_output.current_timeframe == "3d"
    assert result.stage_results[-1].stage_output.details["manifest_path"] == str(
        result.manifest_path
    )

    progress_messages = [record.getMessage() for record in caplog.records if "event=" in record.msg]

    assert any(
        "event=artifact_precompute_stage_started" in message
        and "stage=canonical_prices" in message
        for message in progress_messages
    )
    assert any(
        "event=artifact_precompute_stage_finished" in message
        and "stage=root_manifest" in message
        for message in progress_messages
    )
    timeframe_started_messages = [
        message for message in progress_messages if "event=timeframe_started" in message
    ]
    timeframe_finished_messages = [
        message for message in progress_messages if "event=timeframe_finished" in message
    ]
    assert len(timeframe_started_messages) == len(ARTIFACT_MAPPING_TIMEFRAMES_V2)
    assert len(timeframe_finished_messages) == len(ARTIFACT_MAPPING_TIMEFRAMES_V2)
    assert "current_timeframe=15m" in timeframe_started_messages[0]
    assert "current_timeframe=3d" in timeframe_finished_messages[-1]
    assert all('"open_timeframe_sessions":1' in message for message in timeframe_started_messages)
    assert all('"open_timeframe_sessions":0' in message for message in timeframe_finished_messages)
    assert all(
        '"max_open_timeframe_sessions":1' in message for message in timeframe_started_messages
    )
    assert all(
        '"max_open_timeframe_sessions":1' in message for message in timeframe_finished_messages
    )
    hit_times_finished_index = next(
        index
        for index, message in enumerate(progress_messages)
        if "event=artifact_precompute_stage_finished" in message and "stage=hit_times" in message
    )
    first_timeframe_started_index = next(
        index
        for index, message in enumerate(progress_messages)
        if "event=timeframe_started" in message
    )
    assert hit_times_finished_index < first_timeframe_started_index
    assert any(
        "event=artifact_precompute_finished" in message and "\"stage_results\"" in message
        for message in progress_messages
    )


def test_backtest_artifact_precompute_runner_v2_materializes_hit_times_and_full_validation_passes(
    tmp_path: Path,
) -> None:
    """
    Verify runner-built slots can pass full validation when only `hit_times/15m` is required.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        R5-01 should materialize strict hit-times artifacts even when no signal targets are
        configured.
    Raises:
        AssertionError: If built hit-times artifacts drift or strict validation rejects the slot.
    Side Effects:
        Writes hit-times artifacts under the inactive slot in `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=2,
        validation_signal_artifacts=(),
        require_hit_times_manifest=True,
    )
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
        ),
    )
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=fixture.loader)

    result = runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)
    )
    validation_result = validator.validate_slot(
        coordinates=fixture.coordinates,
        slot=fixture.inactive_slot,
        validation_spec=fixture.runtime_config.to_validation_spec(),
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert result.slot == fixture.inactive_slot
    assert validation_result.hit_times_manifest is not None
    assert validation_result.hit_times_manifest.timeline_bar_count == (
        _hit_times_bars_from_one_minute_bars_v2(_FULL_BUILD_MINUTES_V2)
    )
    assert validation_result.signal_manifests == ()
    assert validation_result.diagnostics == ()


def test_backtest_artifact_precompute_runner_v2_uses_full_rebuild_hit_times_budget_for_bootstrap(
    tmp_path: Path,
) -> None:
    """
    Verify bootstrap/full-rebuild hit-times use the dedicated full-rebuild cell budget.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        First publish for one symbol root may need a larger hit-times budget than incremental
        steady-state tail refreshes.
    Raises:
        AssertionError: If bootstrap still uses the tighter incremental hit-times budget.
    Side Effects:
        Writes a small strict hit-times family under the inactive slot in `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
    """
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=2,
        validation_signal_artifacts=(),
        require_hit_times_manifest=True,
        max_hit_times_cells=10_000,
        max_hit_times_cells_full_rebuild=20_000,
    )
    timeline_bars = 3 * 24 * 60
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(timeline_bars)))
        ),
    )

    result = runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=timeline_bars)
    )
    hit_times_manifest = fixture.loader.load_hit_times_manifest(
        fixture.coordinates,
        fixture.inactive_slot,
    )

    expected_hit_times_bar_count = _hit_times_bars_from_one_minute_bars_v2(timeline_bars)
    assert hit_times_manifest.timeline_bar_count == expected_hit_times_bar_count
    assert result.stage_rebuild_stats.hit_times == ArtifactStageRebuildStatsV2(
        reused_prefix_bars=0,
        rewritten_tail_bars=expected_hit_times_bar_count,
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
    ema_features_manifest = fixture.loader.load_signal_features_manifest(
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
    fifteen_minute_sma_features_manifest = fixture.loader.load_signal_features_manifest(
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
    one_hour_sma_features_manifest = fixture.loader.load_signal_features_manifest(
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
    ema_feature_matrix = _load_signal_features_matrix_v2(
        fixture=fixture,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    fifteen_minute_sma_feature_matrix = _load_signal_features_matrix_v2(
        fixture=fixture,
        timeframe="15m",
        indicator_id="ma.sma",
    )
    one_hour_sma_feature_matrix = _load_signal_features_matrix_v2(
        fixture=fixture,
        timeframe="1h",
        indicator_id="ma.sma",
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
    assert ema_manifest.signal_features is not None
    assert ema_features_manifest.feature_names == SIGNAL_FEATURE_NAMES_V2
    assert ema_features_manifest.features.dtype == "float32"
    assert ema_features_manifest.features.shape == (6, len(SIGNAL_FEATURE_NAMES_V2))
    assert ema_features_manifest.features.axis_order == ("variant", "feature")
    assert dict(ema_manifest.grid.signals_v1_params_defaults) == {}
    assert fifteen_minute_sma_manifest.rows_count == 6
    assert fifteen_minute_sma_manifest.timeline.bar_count == 288
    assert fifteen_minute_sma_manifest.signals.dtype == "int8"
    assert fifteen_minute_sma_manifest.signals.shape == (6, 288)
    assert fifteen_minute_sma_manifest.signals.axis_order == ("variant", "time")
    assert fifteen_minute_sma_manifest.signal_features is not None
    assert fifteen_minute_sma_features_manifest.feature_names == SIGNAL_FEATURE_NAMES_V2
    assert fifteen_minute_sma_features_manifest.features.dtype == "float32"
    assert fifteen_minute_sma_features_manifest.features.shape == (
        6,
        len(SIGNAL_FEATURE_NAMES_V2),
    )
    assert fifteen_minute_sma_features_manifest.features.axis_order == (
        "variant",
        "feature",
    )
    assert dict(fifteen_minute_sma_manifest.grid.signals_v1_params_defaults) == {}
    assert one_hour_sma_manifest.rows_count == 6
    assert one_hour_sma_manifest.timeline.bar_count == 72
    assert one_hour_sma_manifest.signals.dtype == "int8"
    assert one_hour_sma_manifest.signals.shape == (6, 72)
    assert one_hour_sma_manifest.signals.axis_order == ("variant", "time")
    assert one_hour_sma_manifest.signal_features is not None
    assert one_hour_sma_features_manifest.feature_names == SIGNAL_FEATURE_NAMES_V2
    assert one_hour_sma_features_manifest.features.dtype == "float32"
    assert one_hour_sma_features_manifest.features.shape == (
        6,
        len(SIGNAL_FEATURE_NAMES_V2),
    )
    assert one_hour_sma_features_manifest.features.axis_order == (
        "variant",
        "feature",
    )
    assert dict(one_hour_sma_manifest.grid.signals_v1_params_defaults) == {}
    assert ema_matrix.dtype == np.int8
    assert fifteen_minute_sma_matrix.dtype == np.int8
    assert one_hour_sma_matrix.dtype == np.int8
    assert ema_feature_matrix.dtype == np.float32
    assert fifteen_minute_sma_feature_matrix.dtype == np.float32
    assert one_hour_sma_feature_matrix.dtype == np.float32
    assert ema_matrix.shape == (6, 288)
    assert fifteen_minute_sma_matrix.shape == (6, 288)
    assert one_hour_sma_matrix.shape == (6, 72)
    assert ema_feature_matrix.shape == (6, len(SIGNAL_FEATURE_NAMES_V2))
    assert fifteen_minute_sma_feature_matrix.shape == (6, len(SIGNAL_FEATURE_NAMES_V2))
    assert one_hour_sma_feature_matrix.shape == (6, len(SIGNAL_FEATURE_NAMES_V2))
    assert set(np.unique(ema_matrix).tolist()) <= {-1, 0, 1}
    assert set(np.unique(fifteen_minute_sma_matrix).tolist()) <= {-1, 0, 1}
    assert set(np.unique(one_hour_sma_matrix).tolist()) <= {-1, 0, 1}
    np.testing.assert_allclose(
        ema_feature_matrix,
        _expected_signal_features_matrix_v2(signal_matrix=ema_matrix),
    )
    np.testing.assert_allclose(
        fifteen_minute_sma_feature_matrix,
        _expected_signal_features_matrix_v2(signal_matrix=fifteen_minute_sma_matrix),
    )
    np.testing.assert_allclose(
        one_hour_sma_feature_matrix,
        _expected_signal_features_matrix_v2(signal_matrix=one_hour_sma_matrix),
    )
    assert len(validation_result.signal_manifests) == 3
    assert validation_result.hit_times_manifest is None
    assert validation_result.diagnostics == ()


def test_backtest_artifact_precompute_runner_v2_chunked_signal_output_matches_single_chunk_reference(  # noqa: E501
    tmp_path: Path,
) -> None:
    """
    Verify chunked signal materialization matches the single-chunk reference semantics.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        A `2`-row chunk size splits the small MA grid into bounded chunk-local compute calls.
    Raises:
        AssertionError: If chunked output, manifest hashes, or per-call variant counts drift.
    Side Effects:
        Materializes one chunked slot and one reference slot under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/signal_chunk_planner_v2.py
    """
    signal_targets = (("15m", "ma.ema"),)
    defaults_provider = _build_signal_test_defaults_provider_v2()
    grid_builder = _signal_grid_builder_v2()
    signal_rules_engine = BacktestSignalRulesEngineV2(defaults_provider=defaults_provider)
    chunked_fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path / "chunked",
        price_tail_bars_1m=2,
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=False,
        signal_worker_processes=2,
        signal_chunk_rows_min=2,
        signal_chunk_rows_max=2,
    )
    reference_fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path / "reference",
        price_tail_bars_1m=2,
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=False,
    )
    chunked_compute = _DeterministicSignalCompute(grid_builder=grid_builder)
    reference_compute = _DeterministicSignalCompute(grid_builder=grid_builder)
    rows = _build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))

    BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=chunked_fixture.runtime_settings,
        artifact_loader=chunked_fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(rows=rows),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=chunked_compute,
        indicator_grid_builder=grid_builder,
    ).export_canonical_price_1m(
        _request_v2(fixture=chunked_fixture, end_minute=_FULL_BUILD_MINUTES_V2)
    )
    BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=reference_fixture.runtime_settings,
        artifact_loader=reference_fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(rows=rows),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=reference_compute,
        indicator_grid_builder=grid_builder,
    ).export_canonical_price_1m(
        _request_v2(fixture=reference_fixture, end_minute=_FULL_BUILD_MINUTES_V2)
    )

    chunked_matrix = _load_signal_matrix_v2(
        fixture=chunked_fixture,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    reference_matrix = _load_signal_matrix_v2(
        fixture=reference_fixture,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    chunked_manifest = chunked_fixture.loader.load_signal_manifest(
        chunked_fixture.coordinates,
        chunked_fixture.inactive_slot,
        "15m",
        "ma.ema",
    )
    reference_manifest = reference_fixture.loader.load_signal_manifest(
        reference_fixture.coordinates,
        reference_fixture.inactive_slot,
        "15m",
        "ma.ema",
    )

    assert chunked_compute.variant_counts == [1, 1, 1, 1, 1, 1]
    assert chunked_compute.time_lengths == [288, 288, 288, 288, 288, 288]
    assert reference_compute.variant_counts == [3, 3]
    np.testing.assert_array_equal(chunked_matrix, reference_matrix)
    assert chunked_manifest.grid.variant_keys_sha256 == reference_manifest.grid.variant_keys_sha256
    assert chunked_manifest.signals.sha256 == reference_manifest.signals.sha256
    assert chunked_manifest.signals.axis_order == ("variant", "time")


def test_backtest_artifact_precompute_runner_v2_reports_chunk_progress_and_session_totals(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Verify chunked signal execution exposes structured per-chunk progress and session totals.

    Args:
        tmp_path: pytest temporary path fixture.
        caplog: pytest log capture fixture.
    Returns:
        None.
    Assumptions:
        One `15m` MA target with `2`-row chunks should emit exactly three chunk start/finish
        pairs.
    Raises:
        AssertionError: If structured logs or stage-result counters drift.
    Side Effects:
        Materializes one chunked signal artifact under `tmp_path` and captures INFO logs.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_coordinator.py
    """
    signal_targets = (("15m", "ma.ema"),)
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=2,
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=False,
        signal_worker_processes=2,
        signal_chunk_rows_min=2,
        signal_chunk_rows_max=2,
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

    with caplog.at_level(logging.INFO):
        result = runner.export_canonical_price_1m(
            _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)
        )

    fifteen_minute_stage = next(
        stage_result
        for stage_result in result.stage_results
        if stage_result.stage_output.current_timeframe == "15m"
    )
    progress_messages = [record.getMessage() for record in caplog.records if "event=" in record.msg]
    chunk_started = [
        message
        for message in progress_messages
        if "event=artifact_precompute_chunk_started" in message
    ]
    chunk_finished = [
        message
        for message in progress_messages
        if "event=artifact_precompute_chunk_finished" in message
    ]

    assert fifteen_minute_stage.stage_output.details["completed_chunks_total"] == 3
    assert fifteen_minute_stage.stage_output.details["completed_indicators_total"] == 1
    assert len(chunk_started) == 3
    assert len(chunk_finished) == 3
    assert any('"current_indicator_id":"ma.ema"' in message for message in chunk_started)
    assert any(
        '"chunk_index":0' in message and '"chunk_count":3' in message
        for message in chunk_started
    )
    assert any(
        '"chunk_index":1' in message and '"chunk_count":3' in message
        for message in chunk_started
    )
    assert any(
        '"chunk_index":2' in message and '"chunk_count":3' in message
        for message in chunk_started
    )
    assert any('"completed_chunks_total":1' in message for message in chunk_finished)
    assert any('"completed_chunks_total":2' in message for message in chunk_finished)
    assert any('"completed_chunks_total":3' in message for message in chunk_finished)


def test_backtest_artifact_precompute_runner_v2_reuses_one_timeframe_price_load_across_indicators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify one timeframe session loads `prices/<tf>` once for mappings and once for shared reuse.

    Args:
        tmp_path: pytest temporary path fixture.
        monkeypatch: pytest fixture used to wrap the internal price-array loader.
    Returns:
        None.
    Assumptions:
        Two indicators on the same timeframe should share one session-owned price payload instead
        of reloading `prices/<tf>` per indicator target.
    Raises:
        AssertionError: If the same timeframe is reloaded once per indicator.
    Side Effects:
        Materializes one narrowed timeframe session under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    signal_targets = (("15m", "ma.ema"), ("15m", "ma.sma"))
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=2,
        validation_price_timeframes=("1m", "15m"),
        validation_mapping_timeframes=("15m",),
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
    recorded_loads: list[tuple[str, str]] = []
    original_loader = artifact_precompute_runner_module._load_materialized_price_arrays_v2

    def _record_loads(
        *,
        artifact_loader: Any,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
        manifest_section: Any,
        location_prefix: str,
    ) -> object:
        """
        Record one price-array load call before delegating to the real helper.

        Args:
            artifact_loader: Original helper artifact loader dependency.
            coordinates: Artifact coordinates for the load request.
            slot: Inactive slot literal for the load request.
            timeframe: Price timeframe being loaded.
            manifest_section: Strict price-manifest section for the load request.
            location_prefix: Human-readable loader location label.
        Returns:
            object: Real helper result.
        Assumptions:
            The wrapper preserves the helper behavior and only records call metadata.
        Raises:
            Exception: Propagates any real helper failure unchanged.
        Side Effects:
            Appends `(timeframe, location_prefix)` to the in-memory call log.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        recorded_loads.append(
            (
                timeframe,
                location_prefix,
            )
        )
        return original_loader(
            artifact_loader=artifact_loader,
            coordinates=coordinates,
            slot=slot,
            timeframe=timeframe,
            manifest_section=manifest_section,
            location_prefix=location_prefix,
        )

    monkeypatch.setattr(
        artifact_precompute_runner_module,
        "_load_materialized_price_arrays_v2",
        _record_loads,
    )

    result = runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)
    )
    fifteen_minute_loads = [
        location_prefix
        for timeframe, location_prefix in recorded_loads
        if timeframe == "15m"
    ]
    fifteen_minute_stage = next(
        stage_result
        for stage_result in result.stage_results
        if stage_result.stage_output.current_timeframe == "15m"
    )

    assert fifteen_minute_loads == [
        "materialized prices[15m] mapping target",
        "materialized prices[15m] timeframe session",
    ]
    assert fifteen_minute_stage.stage_output.details["completed_indicators_total"] == 2


def test_write_signal_matrix_in_chunks_v2_process_workers_do_not_receive_candles_per_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify spawned chunk jobs reuse worker-bootstrapped candles instead of per-submit payloads.

    Args:
        tmp_path: pytest temporary path fixture.
        monkeypatch: pytest fixture used to replace process-pool internals with a synchronous stub.
    Returns:
        None.
    Assumptions:
        R12 process workers should receive the bounded candle payload once via pool bootstrap and
        never in each `submit(...)` call.
    Raises:
        AssertionError: If per-chunk submit kwargs still contain full `CandleArrays`.
    Side Effects:
        Materializes one compact signal matrix under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    _FakeProcessPoolExecutorV2.submitted_kwargs = []
    _FakeProcessPoolExecutorV2.initializer_bootstraps = []
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        validation_price_timeframes=("1m", "15m"),
        validation_mapping_timeframes=("15m",),
        signal_worker_processes=2,
        signal_chunk_rows_min=2,
        signal_chunk_rows_max=2,
    )
    defaults_provider = _build_signal_test_defaults_provider_v2()
    grid_builder = _signal_grid_builder_v2()
    signal_rules_engine = BacktestSignalRulesEngineV2(defaults_provider=defaults_provider)
    signal_target = ArtifactSignalValidationSpecV2(timeframe="15m", indicator_id="ma.ema")
    defaults_grid = defaults_provider.compute_defaults(indicator_id=signal_target.indicator_id)
    assert defaults_grid is not None
    compute_grid = artifact_precompute_runner_module._grid_with_layout_v2(
        grid=defaults_grid,
        indicator_id=signal_target.indicator_id,
        layout=Layout.VARIANT_MAJOR,
    )
    materialized_grid = grid_builder.materialize_indicator(grid=compute_grid)
    signal_rows = artifact_precompute_runner_module._build_signal_variant_rows_v2(
        coordinates=fixture.coordinates,
        timeframe=signal_target.timeframe,
        materialized_grid=materialized_grid,
    )
    timeframe_millis = 15 * 60 * 1000
    open_time = (
        np.arange(20, dtype=np.int64) * timeframe_millis
        + int(_BASE_TIME_UTC.timestamp() * 1000)
    )
    close_time = open_time + timeframe_millis
    ohlcv = np.column_stack(
        (
            np.linspace(100.0, 119.0, num=20, dtype=np.float32),
            np.linspace(101.0, 120.0, num=20, dtype=np.float32),
            np.linspace(99.0, 118.0, num=20, dtype=np.float32),
            np.linspace(100.5, 119.5, num=20, dtype=np.float32),
            np.linspace(1000.0, 1019.0, num=20, dtype=np.float32),
        )
    )
    candles = _candle_arrays_from_loaded_prices_v2(
        timeframe="15m",
        open_time=open_time,
        close_time=close_time,
        ohlcv=ohlcv,
    )
    signal_shape = (len(signal_rows), int(candles.close.shape[0]))
    rule_spec = signal_rules_engine.rule_spec(indicator_id=signal_target.indicator_id)
    chunk_jobs = artifact_precompute_runner_module._plan_signal_chunk_jobs_v2(
        runtime_settings=fixture.runtime_settings,
        signal_target=signal_target,
        timeline_bar_count=signal_shape[1],
        compute_bar_count=int(candles.close.shape[0]),
        variant_count=signal_shape[0],
        dependency_count=len(rule_spec.required_dependency_ids),
    )
    chunk_blocks = tuple(
        artifact_precompute_runner_module._build_signal_chunk_blocks_v2(
            materialized_grid=materialized_grid,
            chunk_job=chunk_job,
        )
        for chunk_job in chunk_jobs
    )
    default_inputs_source, signal_params_defaults = signal_rules_engine.resolved_defaults(
        indicator_id=signal_target.indicator_id
    )
    signal_paths = fixture.loader.resolve_signal_paths(
        fixture.coordinates,
        fixture.inactive_slot,
        signal_target.timeframe,
        signal_target.indicator_id,
    )
    monkeypatch.setattr(
        artifact_precompute_runner_module,
        "_resolve_indicator_compute_worker_factory_v2",
        lambda *, indicator_compute: (
            _SnapshotRehydratingSignalComputeFactoryV2,
            {"grid_builder": grid_builder},
        ),
    )
    monkeypatch.setattr(
        artifact_precompute_runner_module,
        "ProcessPoolExecutor",
        _FakeProcessPoolExecutorV2,
    )
    monkeypatch.setattr(
        artifact_precompute_runner_module,
        "as_completed",
        lambda futures: tuple(futures),
    )
    monkeypatch.setattr(
        artifact_precompute_runner_module,
        "_SIGNAL_CHUNK_WORKER_STATE_V2",
        None,
    )

    artifact_precompute_runner_module._write_signal_matrix_in_chunks_v2(
        coordinates=fixture.coordinates,
        slot=fixture.inactive_slot,
        slot_generation=5,
        force_full_rebuild=False,
        signal_target=signal_target,
        signal_paths=signal_paths,
        signal_shape=signal_shape,
        candles=candles,
        signal_worker_processes=fixture.runtime_settings.execution_policy.signal_worker_processes,
        chunk_jobs=chunk_jobs,
        chunk_blocks=chunk_blocks,
        signal_rows=signal_rows,
        signal_tail_plan=artifact_precompute_runner_module._SignalArtifactTailPlanV2(
            reused_prefix_bars=0,
            compute_start_idx=0,
            trim_prefix_bars=0,
            effective_tail_bars=signal_shape[1],
        ),
        existing_signal_artifact=None,
        indicator_compute=_DeterministicSignalCompute(grid_builder=grid_builder),
        rule_spec=rule_spec,
        default_inputs_source=default_inputs_source,
        signal_params_defaults=signal_params_defaults,
        max_signal_rows_per_artifact=fixture.runtime_settings.max_signal_rows_per_artifact,
    )

    assert len(_FakeProcessPoolExecutorV2.initializer_bootstraps) == 1
    assert len(_FakeProcessPoolExecutorV2.submitted_kwargs) == len(chunk_jobs)
    assert all(
        submission["candles"] is None
        for submission in _FakeProcessPoolExecutorV2.submitted_kwargs
    )
    assert np.load(signal_paths.signals, allow_pickle=False).shape == signal_shape


def test_backtest_artifact_precompute_runner_v2_uses_timeframe_local_non_signal_and_signal_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify R12-03 no longer executes old broad stage batches across all target timeframes.

    Args:
        tmp_path: pytest temporary path fixture.
        monkeypatch: pytest fixture used to wrap internal stage helpers.
    Returns:
        None.
    Assumptions:
        The delivered pipeline keeps `hit_times/15m` in canonical scope, then interleaves
        `rolled_prices -> mappings -> signals` inside each timeframe session before moving on.
    Raises:
        AssertionError: If helper execution drifts back to `all rolled prices -> all mappings ->
            all signals`.
    Side Effects:
        Materializes strict artifacts under `tmp_path` while recording internal helper order.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    signal_targets = (("15m", "ma.ema"), ("30m", "ma.sma"))
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=2,
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=True,
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
    helper_order: list[tuple[str, ...]] = []

    original_hit_times = artifact_precompute_runner_module._materialize_hit_times_artifacts_v2
    original_rolled_prices = (
        artifact_precompute_runner_module._materialize_rolled_price_timeframe_v2
    )
    original_mappings = artifact_precompute_runner_module._materialize_mapping_timeframe_v2
    original_signals = artifact_precompute_runner_module._materialize_signal_artifact_v2

    def _record_hit_times(**kwargs):
        """
        Record the canonical-scope `hit_times/15m` execution before delegating to the real helper.

        Args:
            **kwargs: Original helper keyword arguments.
        Returns:
            _HitTimesArtifactBuildResultV2: Real helper result.
        Assumptions:
            The wrapper preserves the original helper behavior and only records call ordering.
        Raises:
            Exception: Propagates any real helper failure unchanged.
        Side Effects:
            Appends one `("hit_times", "1m")` marker to the in-memory order log.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        helper_order.append(("hit_times", "1m"))
        return original_hit_times(**kwargs)

    def _record_rolled_prices(timeframe: str, **kwargs):
        """
        Record one `rolled_prices/<tf>` materialization before calling the real helper.

        Args:
            timeframe: Target rolled timeframe literal.
            **kwargs: Original helper keyword arguments.
        Returns:
            ArtifactPriceTimeframeManifestV2: Real helper result.
        Assumptions:
            Recording does not change the deterministic per-timeframe build behavior.
        Raises:
            Exception: Propagates any real helper failure unchanged.
        Side Effects:
            Appends one `("rolled_prices", timeframe)` marker to the in-memory order log.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        helper_order.append(("rolled_prices", timeframe))
        return original_rolled_prices(timeframe, **kwargs)

    def _record_mappings(timeframe: str, **kwargs):
        """
        Record one `mappings/<tf>` materialization before calling the real helper.

        Args:
            timeframe: Target mapping timeframe literal.
            **kwargs: Original helper keyword arguments.
        Returns:
            _MappingArtifactMaterializationResultV2: Real helper result.
        Assumptions:
            Recording does not change the deterministic per-timeframe build behavior.
        Raises:
            Exception: Propagates any real helper failure unchanged.
        Side Effects:
            Appends one `("mappings", timeframe)` marker to the in-memory order log.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        helper_order.append(("mappings", timeframe))
        return original_mappings(timeframe, **kwargs)

    def _record_signals(*, signal_target: ArtifactSignalValidationSpecV2, **kwargs):
        """
        Record one `signals/<tf>/<indicator_id>` materialization before delegating.

        Args:
            signal_target: Explicit signal target being materialized.
            **kwargs: Original helper keyword arguments.
        Returns:
            _SignalArtifactMaterializationResultV2: Real helper result.
        Assumptions:
            Recording does not change the deterministic per-target signal build behavior.
        Raises:
            Exception: Propagates any real helper failure unchanged.
        Side Effects:
            Appends one `("signals", timeframe, indicator_id)` marker to the in-memory order log.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        helper_order.append(("signals", signal_target.timeframe, signal_target.indicator_id))
        return original_signals(signal_target=signal_target, **kwargs)

    monkeypatch.setattr(
        artifact_precompute_runner_module,
        "_materialize_hit_times_artifacts_v2",
        _record_hit_times,
    )
    monkeypatch.setattr(
        artifact_precompute_runner_module,
        "_materialize_rolled_price_timeframe_v2",
        _record_rolled_prices,
    )
    monkeypatch.setattr(
        artifact_precompute_runner_module,
        "_materialize_mapping_timeframe_v2",
        _record_mappings,
    )
    monkeypatch.setattr(
        artifact_precompute_runner_module,
        "_materialize_signal_artifact_v2",
        _record_signals,
    )

    runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)
    )

    assert helper_order[0] == ("hit_times", "1m")
    assert helper_order.index(("hit_times", "1m")) < helper_order.index(("rolled_prices", "15m"))
    assert helper_order.index(("rolled_prices", "15m")) < helper_order.index(("mappings", "15m"))
    assert helper_order.index(("mappings", "15m")) < helper_order.index(
        ("signals", "15m", "ma.ema")
    )
    assert helper_order.index(("signals", "15m", "ma.ema")) < helper_order.index(
        ("rolled_prices", "30m")
    )
    assert helper_order.index(("rolled_prices", "30m")) < helper_order.index(("mappings", "30m"))
    assert helper_order.index(("mappings", "30m")) < helper_order.index(
        ("signals", "30m", "ma.sma")
    )
    assert helper_order.index(("signals", "30m", "ma.sma")) < helper_order.index(
        ("rolled_prices", "1h")
    )


def test_backtest_artifact_precompute_runner_v2_uses_runtime_configured_timeframe_set(
    tmp_path: Path,
) -> None:
    """
    Verify timeframe sessions follow runtime-config timeframes instead of the hardcoded full set.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        A narrowed validation plan should materialize only its configured `prices/<tf>`,
        `mappings/<tf>`, and signal targets.
    Raises:
        AssertionError: If the runner silently falls back to the full constant timeframe loop.
    Side Effects:
        Materializes one narrowed inactive slot under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    signal_targets = (("15m", "ma.ema"), ("1h", "ma.sma"))
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        validation_price_timeframes=("1m", "15m", "1h"),
        validation_mapping_timeframes=("15m", "1h"),
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=False,
    )
    defaults_provider = _build_signal_test_defaults_provider_v2()
    grid_builder = _signal_grid_builder_v2()
    signal_rules_engine = BacktestSignalRulesEngineV2(defaults_provider=defaults_provider)
    result = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=_DeterministicSignalCompute(grid_builder=grid_builder),
        indicator_grid_builder=grid_builder,
    ).export_canonical_price_1m(_request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2))
    slot_manifest = fixture.loader.load_slot_manifest(
        fixture.coordinates,
        fixture.inactive_slot,
    )
    stage_timeframes = tuple(
        stage_result.stage_output.current_timeframe
        for stage_result in result.stage_results
        if stage_result.stage_output.stage == "timeframe_session"
    )

    assert stage_timeframes == ("15m", "1h")
    assert tuple(item.timeframe for item in slot_manifest.prices) == ("1m", "15m", "1h")
    assert tuple(item.timeframe for item in slot_manifest.mappings) == ("15m", "1h")
    assert tuple(
        (item.timeframe, item.indicator_id) for item in slot_manifest.signals.manifests
    ) == signal_targets


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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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

    assert initial_compute.time_lengths == [288, 288]
    assert initial_compute.variant_counts == [3, 3]
    assert len(updated_compute.time_lengths) == 2
    assert updated_compute.variant_counts == [3, 3]
    assert len(set(updated_compute.time_lengths)) == 1
    assert 2 <= updated_compute.time_lengths[0] < initial_compute.time_lengths[0]
    assert updated_compute.time_lengths[0] == 17
    np.testing.assert_array_equal(updated_matrix[:, :-2], initial_matrix[:, :-2])
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
      - docs/architecture/backtest/README.md
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

    assert first_compute.time_lengths == [288, 288]
    assert first_compute.variant_counts == [3, 3]
    assert len(second_compute.time_lengths) == 2
    assert second_compute.variant_counts == [3, 3]
    assert len(set(second_compute.time_lengths)) == 1
    assert 2 <= second_compute.time_lengths[0] < first_compute.time_lengths[0]
    assert first_bytes == second_bytes


def test_backtest_artifact_precompute_runner_v2_proves_repeated_daily_run_rewrites_only_bounded_tail(  # noqa: E501
    tmp_path: Path,
) -> None:
    """
    Prove a repeated daily run preserves unchanged prefixes and rewrites only the bounded tail.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        The second run extends source data by a bounded suffix and updates only the overlapping
        tail for `prices`, `mappings`, `signals`, and `hit_times`.
    Raises:
        AssertionError: If stage stats drift, unchanged prefixes lose byte-stability, or full-slot
            validation fails after the incremental run.
    Side Effects:
        Rewrites the inactive slot twice under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    initial_end_minute = _FULL_BUILD_MINUTES_V2
    updated_end_minute = _FULL_BUILD_MINUTES_V2 + 60
    signal_targets = (("15m", "ma.ema"),)
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=30,
        mapping_tail_bars_1m=30,
        signal_tail_bars_1m=30,
        hit_times_tail_bars_1m=30,
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=True,
    )
    defaults_provider = _build_signal_test_defaults_provider_v2()
    grid_builder = _signal_grid_builder_v2()
    signal_rules_engine = BacktestSignalRulesEngineV2(defaults_provider=defaults_provider)
    initial_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(initial_end_minute)))
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=_DeterministicSignalCompute(grid_builder=grid_builder),
        indicator_grid_builder=grid_builder,
    )
    initial_runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=initial_end_minute)
    )
    initial_open_time, _, initial_ohlcv = _load_price_arrays_v2(fixture=fixture, timeframe="1m")
    initial_signal_matrix = _load_signal_matrix_v2(
        fixture=fixture,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    initial_mapping_lengths = {
        timeframe: _load_mapping_arrays_v2(fixture=fixture, timeframe=timeframe)[0].shape[0]
        for timeframe in ARTIFACT_MAPPING_TIMEFRAMES_V2
    }
    initial_one_hour_mapping_open, initial_one_hour_mapping_close = _load_mapping_arrays_v2(
        fixture=fixture,
        timeframe="1h",
    )
    _, _, initial_long_tp, initial_long_sl, initial_short_tp, initial_short_sl = (
        _load_hit_times_arrays_v2(fixture=fixture)
    )

    updated_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(
                bar_indexes=tuple(range(initial_end_minute - 30, updated_end_minute)),
                price_offset=1000.0,
                volume_offset=50.0,
            )
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=_DeterministicSignalCompute(grid_builder=grid_builder),
        indicator_grid_builder=grid_builder,
    )
    result = updated_runner.export_canonical_price_1m(
        _request_v2(
            fixture=fixture,
            end_minute=updated_end_minute,
            asof_date="2026-03-29",
            generated_at_utc="2026-03-29T03:00:00Z",
        )
    )
    updated_open_time, _, updated_ohlcv = _load_price_arrays_v2(fixture=fixture, timeframe="1m")
    updated_signal_matrix = _load_signal_matrix_v2(
        fixture=fixture,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    updated_one_hour_mapping_open, updated_one_hour_mapping_close = _load_mapping_arrays_v2(
        fixture=fixture,
        timeframe="1h",
    )
    updated_mapping_lengths = {
        timeframe: _load_mapping_arrays_v2(fixture=fixture, timeframe=timeframe)[0].shape[0]
        for timeframe in ARTIFACT_MAPPING_TIMEFRAMES_V2
    }
    _, _, updated_long_tp, updated_long_sl, updated_short_tp, updated_short_sl = (
        _load_hit_times_arrays_v2(fixture=fixture)
    )
    validation_result = BacktestArtifactManifestValidatorV2(
        artifact_loader=fixture.loader
    ).validate_slot(
        coordinates=fixture.coordinates,
        slot=fixture.inactive_slot,
        validation_spec=fixture.runtime_config.to_validation_spec(),
        expected_asof_date="2026-03-29",
        expected_slot_generation=5,
    )
    expected_mapping_reused_prefix_bars = sum(initial_mapping_lengths.values())
    expected_mapping_rewritten_bars = sum(
        updated_mapping_lengths[timeframe] - initial_mapping_lengths[timeframe]
        for timeframe in ARTIFACT_MAPPING_TIMEFRAMES_V2
    )

    assert result.reused_prefix_bars == initial_end_minute - 30
    assert result.rewritten_tail_bars == 90
    assert result.stage_rebuild_stats.prices == ArtifactStageRebuildStatsV2(
        reused_prefix_bars=initial_end_minute - 30,
        rewritten_tail_bars=90,
    )
    assert result.stage_rebuild_stats.signals == ArtifactStageRebuildStatsV2(
        reused_prefix_bars=286,
        rewritten_tail_bars=6,
    )
    expected_hit_times_reused_prefix_bars = _hit_times_bars_from_one_minute_bars_v2(
        initial_end_minute - 30
    )
    expected_hit_times_rewritten_tail_bars = (
        _hit_times_bars_from_one_minute_bars_v2(updated_end_minute)
        - expected_hit_times_reused_prefix_bars
    )
    assert result.stage_rebuild_stats.hit_times == ArtifactStageRebuildStatsV2(
        reused_prefix_bars=expected_hit_times_reused_prefix_bars,
        rewritten_tail_bars=expected_hit_times_rewritten_tail_bars,
    )
    assert result.stage_rebuild_stats.mappings == ArtifactStageRebuildStatsV2(
        reused_prefix_bars=expected_mapping_reused_prefix_bars,
        rewritten_tail_bars=expected_mapping_rewritten_bars,
    )
    assert result.tail_rebuild_bars == ArtifactTailRebuildBarsV2(
        prices=90,
        mappings=expected_mapping_rewritten_bars,
        signals=6,
        hit_times=expected_hit_times_rewritten_tail_bars,
    )
    assert _time_axis_prefix_sha256_v2(
        array=updated_open_time,
        prefix_bars=result.stage_rebuild_stats.prices.reused_prefix_bars,
    ) == _time_axis_prefix_sha256_v2(
        array=initial_open_time,
        prefix_bars=result.stage_rebuild_stats.prices.reused_prefix_bars,
    )
    assert _time_axis_prefix_sha256_v2(
        array=updated_ohlcv,
        prefix_bars=result.stage_rebuild_stats.prices.reused_prefix_bars,
    ) == _time_axis_prefix_sha256_v2(
        array=initial_ohlcv,
        prefix_bars=result.stage_rebuild_stats.prices.reused_prefix_bars,
    )
    assert _time_axis_prefix_sha256_v2(
        array=updated_one_hour_mapping_open,
        prefix_bars=initial_one_hour_mapping_open.shape[0],
    ) == _time_axis_prefix_sha256_v2(
        array=initial_one_hour_mapping_open,
        prefix_bars=initial_one_hour_mapping_open.shape[0],
    )
    assert _time_axis_prefix_sha256_v2(
        array=updated_one_hour_mapping_close,
        prefix_bars=initial_one_hour_mapping_close.shape[0],
    ) == _time_axis_prefix_sha256_v2(
        array=initial_one_hour_mapping_close,
        prefix_bars=initial_one_hour_mapping_close.shape[0],
    )
    assert _time_axis_prefix_sha256_v2(
        array=updated_signal_matrix,
        prefix_bars=result.stage_rebuild_stats.signals.reused_prefix_bars,
    ) == _time_axis_prefix_sha256_v2(
        array=initial_signal_matrix,
        prefix_bars=result.stage_rebuild_stats.signals.reused_prefix_bars,
    )
    assert _time_axis_prefix_sha256_v2(
        array=updated_long_tp,
        prefix_bars=result.stage_rebuild_stats.hit_times.reused_prefix_bars,
    ) == _time_axis_prefix_sha256_v2(
        array=initial_long_tp,
        prefix_bars=result.stage_rebuild_stats.hit_times.reused_prefix_bars,
    )
    assert _time_axis_prefix_sha256_v2(
        array=updated_long_sl,
        prefix_bars=result.stage_rebuild_stats.hit_times.reused_prefix_bars,
    ) == _time_axis_prefix_sha256_v2(
        array=initial_long_sl,
        prefix_bars=result.stage_rebuild_stats.hit_times.reused_prefix_bars,
    )
    assert _time_axis_prefix_sha256_v2(
        array=updated_short_tp,
        prefix_bars=result.stage_rebuild_stats.hit_times.reused_prefix_bars,
    ) == _time_axis_prefix_sha256_v2(
        array=initial_short_tp,
        prefix_bars=result.stage_rebuild_stats.hit_times.reused_prefix_bars,
    )
    assert _time_axis_prefix_sha256_v2(
        array=updated_short_sl,
        prefix_bars=result.stage_rebuild_stats.hit_times.reused_prefix_bars,
    ) == _time_axis_prefix_sha256_v2(
        array=initial_short_sl,
        prefix_bars=result.stage_rebuild_stats.hit_times.reused_prefix_bars,
    )
    assert updated_open_time.shape[0] == initial_open_time.shape[0] + 60
    assert updated_one_hour_mapping_open.shape[0] == initial_one_hour_mapping_open.shape[0] + 1
    assert updated_signal_matrix.shape[1] == initial_signal_matrix.shape[1] + 4
    expected_hit_times_appended_bars = (
        _hit_times_bars_from_one_minute_bars_v2(updated_end_minute)
        - _hit_times_bars_from_one_minute_bars_v2(initial_end_minute)
    )
    assert updated_long_tp.shape[1] == initial_long_tp.shape[1] + expected_hit_times_appended_bars
    assert validation_result.diagnostics == ()


def test_backtest_artifact_precompute_runner_v2_falls_back_to_full_hit_times_rebuild_on_grid_drift(  # noqa: E501
    tmp_path: Path,
) -> None:
    """
    Verify `hit_times/15m` switches to deterministic full rebuild when grid reuse drifts.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Price reuse may still remain incremental while only the hit-times stage falls back.
    Raises:
        AssertionError: If hit-times drift is silently reused instead of forcing a full rebuild.
    Side Effects:
        Rewrites the inactive slot under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=30,
        mapping_tail_bars_1m=30,
        signal_tail_bars_1m=30,
        hit_times_tail_bars_1m=30,
        require_hit_times_manifest=True,
    )
    initial_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
        ),
    )
    initial_runner.export_canonical_price_1m(
        _request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2)
    )

    drifted_runtime_settings = replace(
        fixture.runtime_settings,
        hit_times_tp_levels_pct=(1.0, 2.0),
        config_sha256="b" * 64,
    )
    drifted_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=drifted_runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(
                bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2 - 30, _FULL_BUILD_MINUTES_V2))
            )
        ),
    )
    result = drifted_runner.export_canonical_price_1m(
        _request_v2(
            fixture=fixture,
            end_minute=_FULL_BUILD_MINUTES_V2,
            asof_date="2026-03-28",
            generated_at_utc="2026-03-28T03:00:00Z",
        )
    )
    tp_values, _, _, _, _, _ = _load_hit_times_arrays_v2(fixture=fixture)

    assert result.stage_rebuild_stats.prices == ArtifactStageRebuildStatsV2(
        reused_prefix_bars=_FULL_BUILD_MINUTES_V2 - 30,
        rewritten_tail_bars=30,
    )
    assert result.stage_rebuild_stats.hit_times == ArtifactStageRebuildStatsV2(
        reused_prefix_bars=0,
        rewritten_tail_bars=_hit_times_bars_from_one_minute_bars_v2(_FULL_BUILD_MINUTES_V2),
    )
    np.testing.assert_allclose(tp_values, np.asarray([0.01, 0.02], dtype=np.float32))


def test_backtest_artifact_precompute_runner_v2_keeps_long_window_signal_tail_correct_with_warmup(  # noqa: E501
    tmp_path: Path,
) -> None:
    """
    Prove warmup-heavy signal tails match a full rebuild and fail under a naive tail cut.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `signal_tail_bars_1m=30` yields only two `15m` target bars, so a `window=40` signal needs
        extra warmup context to stay correct.
    Raises:
        AssertionError: If the shipped warmup-aware rebuild diverges from a full rebuild or if the
            naive short-context tail cut accidentally stays correct.
    Side Effects:
        Materializes one incremental slot and one full-reference slot under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    signal_targets = (("15m", "ma.sma"),)
    initial_end_minute = _FULL_BUILD_MINUTES_V2
    updated_end_minute = _FULL_BUILD_MINUTES_V2 + 60
    defaults_provider = _build_long_window_signal_test_defaults_provider_v2()
    grid_builder = _signal_grid_builder_v2()
    signal_rules_engine = BacktestSignalRulesEngineV2(defaults_provider=defaults_provider)
    incremental_fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path / "incremental",
        price_tail_bars_1m=30,
        mapping_tail_bars_1m=30,
        signal_tail_bars_1m=30,
        hit_times_tail_bars_1m=30,
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=False,
    )
    initial_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=incremental_fixture.runtime_settings,
        artifact_loader=incremental_fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(initial_end_minute)))
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=_DeterministicSignalCompute(grid_builder=grid_builder),
        indicator_grid_builder=grid_builder,
    )
    initial_runner.export_canonical_price_1m(
        _request_v2(fixture=incremental_fixture, end_minute=initial_end_minute)
    )
    updated_compute = _DeterministicSignalCompute(grid_builder=grid_builder)
    updated_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=incremental_fixture.runtime_settings,
        artifact_loader=incremental_fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(
                bar_indexes=tuple(range(initial_end_minute - 30, updated_end_minute)),
                price_offset=1000.0,
            )
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=updated_compute,
        indicator_grid_builder=grid_builder,
    )
    incremental_result = updated_runner.export_canonical_price_1m(
        _request_v2(
            fixture=incremental_fixture,
            end_minute=updated_end_minute,
            asof_date="2026-03-29",
            generated_at_utc="2026-03-29T03:00:00Z",
        )
    )
    incremental_matrix = _load_signal_matrix_v2(
        fixture=incremental_fixture,
        timeframe="15m",
        indicator_id="ma.sma",
    )

    reference_fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path / "reference",
        price_tail_bars_1m=30,
        mapping_tail_bars_1m=30,
        signal_tail_bars_1m=30,
        hit_times_tail_bars_1m=30,
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=False,
    )
    reference_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=reference_fixture.runtime_settings,
        artifact_loader=reference_fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(
                bar_indexes=tuple(range(updated_end_minute)),
                price_offset=1000.0,
            )
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=signal_rules_engine,
        indicator_compute=_DeterministicSignalCompute(grid_builder=grid_builder),
        indicator_grid_builder=grid_builder,
    )
    reference_runner.export_canonical_price_1m(
        _request_v2(
            fixture=reference_fixture,
            end_minute=updated_end_minute,
            asof_date="2026-03-29",
            generated_at_utc="2026-03-29T03:00:00Z",
        )
    )
    reference_matrix = _load_signal_matrix_v2(
        fixture=reference_fixture,
        timeframe="15m",
        indicator_id="ma.sma",
    )
    reference_open_time, reference_close_time, reference_ohlcv = _load_price_arrays_v2(
        fixture=reference_fixture,
        timeframe="15m",
    )
    naive_tail_bars = incremental_result.stage_rebuild_stats.signals.rewritten_tail_bars
    tail_candles = _candle_arrays_from_loaded_prices_v2(
        timeframe="15m",
        open_time=reference_open_time[-naive_tail_bars:],
        close_time=reference_close_time[-naive_tail_bars:],
        ohlcv=reference_ohlcv[-naive_tail_bars:, :],
    )
    compute_grid = GridSpec(
        indicator_id=IndicatorId("ma.sma"),
        params={"window": ExplicitValuesSpec(name="window", values=(40,))},
        source=ExplicitValuesSpec(name="source", values=("close",)),
        layout_preference=Layout.VARIANT_MAJOR,
    )
    naive_tensor = _DeterministicSignalCompute(grid_builder=grid_builder).compute(
        ComputeRequest(
            candles=tail_candles,
            grid=compute_grid,
            max_variants_guard=incremental_fixture.runtime_settings.max_signal_rows_per_artifact,
        )
    )
    naive_primary_output = indicator_primary_output_series_from_tensor_v1(
        tensor=naive_tensor,
        variant_index=0,
    )
    naive_signal_codes = signal_rules_engine.evaluate(
        request=SignalRuleEvaluationRequestV2(
            indicator_id="ma.sma",
            candles=tail_candles,
            primary_output=naive_primary_output,
            inputs_source="close",
            signal_params={},
            dependency_outputs={},
        )
    ).signal_codes
    assert incremental_result.stage_rebuild_stats.signals == ArtifactStageRebuildStatsV2(
        reused_prefix_bars=286,
        rewritten_tail_bars=6,
    )
    assert updated_compute.time_lengths == [46]
    np.testing.assert_array_equal(incremental_matrix, reference_matrix)
    assert np.isnan(naive_primary_output).all()
    assert not np.array_equal(naive_signal_codes, reference_matrix[0, -naive_tail_bars:])


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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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

    with pytest.raises(ValueError, match=r"signals\.sha256 must match the actual file"):
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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


def test_backtest_artifact_precompute_runner_v2_materializes_zero_axis_signal_targets_without_yaml_compute_defaults(  # noqa: E501
    tmp_path: Path,
) -> None:
    """
    Verify the four approved zero-axis signal targets materialize via hard-definition fallback.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        These targets intentionally keep `compute_defaults(...) is None` in YAML-backed providers.
    Raises:
        AssertionError: If export raises, variant ordering drifts, or matrices stop being
            single-row.
    Side Effects:
        Writes a small inactive-slot fixture under the pytest temp directory.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - tests/unit/contexts/backtest/adapters/test_indicators_yaml_defaults_provider.py
    """
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=30,
        mapping_tail_bars_1m=30,
        signal_tail_bars_1m=30,
        hit_times_tail_bars_1m=30,
        validation_signal_artifacts=_ZERO_AXIS_SIGNAL_TARGETS_V2,
        precompute_signal_artifacts=_ZERO_AXIS_SIGNAL_TARGETS_V2,
        require_hit_times_manifest=False,
    )
    delegate = YamlBacktestGridDefaultsProvider.from_yaml(
        config_path=Path("configs/test/indicators.yaml")
    )
    defaults_provider = _MissingComputeDefaultsProviderV2(
        delegate=delegate,
        hidden_indicator_ids=tuple(
            indicator_id for _, indicator_id in _ZERO_AXIS_SIGNAL_TARGETS_V2
        ),
    )
    grid_builder = _signal_grid_builder_v2()

    BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=BacktestSignalRulesEngineV2(defaults_provider=defaults_provider),
        indicator_compute=_DeterministicSignalCompute(grid_builder=grid_builder),
        indicator_grid_builder=grid_builder,
    ).export_canonical_price_1m(_request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2))

    for timeframe, indicator_id in _ZERO_AXIS_SIGNAL_TARGETS_V2:
        manifest = fixture.loader.load_signal_manifest(
            fixture.coordinates,
            fixture.inactive_slot,
            timeframe,
            indicator_id,
        )
        matrix = _load_signal_matrix_v2(
            fixture=fixture,
            timeframe=timeframe,
            indicator_id=indicator_id,
        )
        assert manifest.rows_count == 1
        assert manifest.signals.axis_order == ("variant", "time")
        assert matrix.shape[0] == 1
        assert matrix.dtype == np.int8


def test_backtest_artifact_precompute_runner_v2_keeps_fail_fast_for_axis_bearing_signal_target_without_defaults(  # noqa: E501
    tmp_path: Path,
) -> None:
    """
    Verify missing compute defaults still fail fast for signal targets with real compute axes.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `trend.adx` remains axis-bearing and therefore must not use the zero-axis fallback.
    Raises:
        AssertionError: If the export unexpectedly succeeds or raises the wrong error.
    Side Effects:
        Allocates a temporary inactive-slot fixture.
    Docs:
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/indicators/domain/definitions/volatility.py
    """
    signal_targets = (("15m", "trend.adx"),)
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=30,
        mapping_tail_bars_1m=30,
        signal_tail_bars_1m=30,
        hit_times_tail_bars_1m=30,
        validation_signal_artifacts=signal_targets,
        precompute_signal_artifacts=signal_targets,
        require_hit_times_manifest=False,
    )
    delegate = YamlBacktestGridDefaultsProvider.from_yaml(
        config_path=Path("configs/test/indicators.yaml")
    )
    defaults_provider = _MissingComputeDefaultsProviderV2(
        delegate=delegate,
        hidden_indicator_ids=("trend.adx",),
    )
    grid_builder = _signal_grid_builder_v2()

    with pytest.raises(
        ValueError,
        match="signal target requires compute defaults for indicator_id 'trend\\.adx'",
    ):
        BacktestArtifactPrecomputeRunnerV2(
            runtime_settings=fixture.runtime_settings,
            artifact_loader=fixture.loader,
            canonical_candle_reader=_FakeCanonicalCandleReader(
                rows=_build_canonical_rows_v2(bar_indexes=tuple(range(_FULL_BUILD_MINUTES_V2)))
            ),
            defaults_provider=defaults_provider,
            signal_rules_engine=BacktestSignalRulesEngineV2(defaults_provider=defaults_provider),
            indicator_compute=_DeterministicSignalCompute(grid_builder=grid_builder),
            indicator_grid_builder=grid_builder,
        ).export_canonical_price_1m(_request_v2(fixture=fixture, end_minute=_FULL_BUILD_MINUTES_V2))


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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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


def _load_signal_features_matrix_v2(
    *,
    fixture: ArtifactPrecomputeFixtureV2,
    timeframe: str,
    indicator_id: str,
) -> np.ndarray:
    """
    Load one strict `signal_features/<tf>/<indicator_id>/features.f32.npy` matrix.

    Args:
        fixture: Minimal strict precompute fixture.
        timeframe: Signal timeframe literal.
        indicator_id: Signal indicator identifier.
    Returns:
        np.ndarray: Loaded compact `float32` feature matrix with shape `[V, feature]`.
    Assumptions:
        The caller already materialized the target additive feature artifact into the inactive
        slot.
    Raises:
        FileNotFoundError: If the deterministic feature path is missing.
        ValueError: If numpy cannot load the stored `.npy` payload.
    Side Effects:
        Reads one feature matrix from disk.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
    """
    return np.load(
        fixture.loader.resolve_signal_features_paths(
            fixture.coordinates,
            fixture.inactive_slot,
            timeframe,
            indicator_id,
        ).features,
        allow_pickle=False,
    )


def _expected_signal_features_matrix_v2(*, signal_matrix: np.ndarray) -> np.ndarray:
    """
    Derive the expected fixed row-local feature matrix from one emitted signal matrix.

    Args:
        signal_matrix: Strict emitted signal matrix with shape `[variant, time]`.
    Returns:
        np.ndarray: Deterministic `float32` feature matrix with shape `[variant, feature]`.
    Assumptions:
        Runner tests must assert the documented fixed feature ordering without depending on the
        production helper internals.
    Raises:
        ValueError: If the provided signal matrix is not two-dimensional or has empty axes.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - tests/unit/contexts/backtest/application/services/v2/test_signal_features_loader_v2.py
    """
    if signal_matrix.ndim != 2:
        raise ValueError(f"signal_matrix must be 2D; got ndim={signal_matrix.ndim!r}")
    variant_count = int(signal_matrix.shape[0])
    timeline_bar_count = int(signal_matrix.shape[1])
    if variant_count <= 0 or timeline_bar_count <= 0:
        raise ValueError("signal_matrix must have positive variant and timeline dimensions")
    nonzero_count = np.count_nonzero(signal_matrix != 0, axis=1).astype(np.float32, copy=False)
    long_count = np.count_nonzero(signal_matrix > 0, axis=1).astype(np.float32, copy=False)
    short_count = np.count_nonzero(signal_matrix < 0, axis=1).astype(np.float32, copy=False)
    activity_ratio = np.ascontiguousarray(
        nonzero_count / np.float32(timeline_bar_count),
        dtype=np.float32,
    )
    direction_balance = np.zeros(variant_count, dtype=np.float32)
    np.divide(
        long_count - short_count,
        nonzero_count,
        out=direction_balance,
        where=nonzero_count > 0.0,
    )
    if timeline_bar_count < 2:
        transition_count = np.zeros(variant_count, dtype=np.float32)
    else:
        transition_count = np.count_nonzero(
            signal_matrix[:, 1:] != signal_matrix[:, :-1],
            axis=1,
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(
        np.column_stack(
            (
                nonzero_count,
                long_count,
                short_count,
                activity_ratio,
                direction_balance,
                transition_count,
            )
        ),
        dtype=np.float32,
    )


def _read_signal_export_bytes_v2(
    *,
    fixture: ArtifactPrecomputeFixtureV2,
    signal_targets: tuple[tuple[str, str], ...],
) -> tuple[bytes, ...]:
    """
    Read deterministic per-target signal and signal-feature manifest bytes and array bytes.

    Args:
        fixture: Minimal strict precompute fixture.
        signal_targets: Explicit ordered `(timeframe, indicator_id)` signal targets.
    Returns:
        tuple[bytes, ...]: Stable byte snapshots for per-target signal and feature files.
    Assumptions:
        Identical rebuild inputs with fixed `generated_at_utc` should keep emitted file bytes
        unchanged across repeated runs.
    Raises:
        FileNotFoundError: If one emitted file is missing.
        OSError: If one file cannot be read.
    Side Effects:
        Reads manifest and `.npy` files from disk.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
        signal_features_paths = fixture.loader.resolve_signal_features_paths(
            fixture.coordinates,
            fixture.inactive_slot,
            timeframe,
            indicator_id,
        )
        snapshots.append(signal_paths.manifest.read_bytes())
        snapshots.append(signal_paths.signals.read_bytes())
        snapshots.append(signal_features_paths.manifest.read_bytes())
        snapshots.append(signal_features_paths.features.read_bytes())
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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


def _load_hit_times_arrays_v2(
    *,
    fixture: ArtifactPrecomputeFixtureV2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the materialized strict `hit_times/15m` family from the inactive slot for assertions.

    Args:
        fixture: Strict precompute fixture with builder and loader.
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            `tp_values`, `sl_values`, `long_tp`, `long_sl`, `short_tp`, and `short_sl`.
    Assumptions:
        Runner tests inspect only the inactive slot written by `export_canonical_price_1m(...)`.
    Raises:
        FileNotFoundError: If one expected hit-times artifact file is missing.
    Side Effects:
        Reads six `.npy` files from disk.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    paths = fixture.loader.resolve_hit_times_paths(
        fixture.coordinates,
        fixture.inactive_slot,
    )
    return (
        np.load(paths.tp_values, allow_pickle=False),
        np.load(paths.sl_values, allow_pickle=False),
        np.load(paths.long_tp, allow_pickle=False),
        np.load(paths.long_sl, allow_pickle=False),
        np.load(paths.short_tp, allow_pickle=False),
        np.load(paths.short_sl, allow_pickle=False),
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
    Read root-manifest plus all materialized price, mapping, and hit-times bytes.

    Args:
        result: Structured export result returned by the runner.
    Returns:
        tuple[bytes, ...]: Bytes for root manifest and every emitted artifact file family.
    Assumptions:
        Result exposes `manifest_path` and `price_paths` exactly as the production DTO does.
    Raises:
        OSError: If one file cannot be read.
    Side Effects:
        Reads artifact files from disk.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
    payloads.extend(
        (
            (
                slot_root / "hit_times" / HIT_TIMES_TIMEFRAME_LITERAL_V2 / "manifest.yaml"
            ).read_bytes(),
            (
                slot_root / "hit_times" / HIT_TIMES_TIMEFRAME_LITERAL_V2 / "tp_values.f32.npy"
            ).read_bytes(),
            (
                slot_root / "hit_times" / HIT_TIMES_TIMEFRAME_LITERAL_V2 / "sl_values.f32.npy"
            ).read_bytes(),
            (
                slot_root / "hit_times" / HIT_TIMES_TIMEFRAME_LITERAL_V2 / "long_tp.u32.npy"
            ).read_bytes(),
            (
                slot_root / "hit_times" / HIT_TIMES_TIMEFRAME_LITERAL_V2 / "long_sl.u32.npy"
            ).read_bytes(),
            (
                slot_root / "hit_times" / HIT_TIMES_TIMEFRAME_LITERAL_V2 / "short_tp.u32.npy"
            ).read_bytes(),
            (
                slot_root / "hit_times" / HIT_TIMES_TIMEFRAME_LITERAL_V2 / "short_sl.u32.npy"
            ).read_bytes(),
        )
    )
    return tuple(payloads)


def test_target_indicator_variant_counts_match_narrowed_catalog_per_env_v2() -> None:
    """
    Verify real defaults-provider plus grid-builder materialize the narrowed target counts.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `prod` and `dev` catalogs should stay aligned, while `test` keeps a documented compact
        subset with the same targeted families present.
    Raises:
        AssertionError: If one env drifts from the locked materialized counts.
    Side Effects:
        Reads checked-in config files and materializes indicator grids in memory.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - configs/dev/indicators.yaml
      - configs/prod/indicators.yaml
      - configs/test/indicators.yaml
      - src/trading/contexts/backtest/adapters/outbound/defaults/
        indicators_yaml_defaults_provider.py
    """
    for env_name, expected_counts in _EXPECTED_VARIANT_COUNTS_BY_ENV_V2.items():
        assert _variant_counts_for_env_v2(env_name=env_name) == expected_counts


def test_target_indicator_variant_counts_stay_within_operational_ceilings_v2() -> None:
    """
    Verify narrowed heavy families stay below the R13-01 operational ceilings.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Ceiling checks must use the real defaults-provider and shared grid-builder path.
    Raises:
        AssertionError: If one targeted indicator exceeds its accepted ceiling.
    Side Effects:
        Reads checked-in config files and materializes indicator grids in memory.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - configs/dev/indicators.yaml
      - configs/prod/indicators.yaml
      - configs/test/indicators.yaml
      - src/trading/contexts/indicators/application/services/grid_builder.py
    """
    for env_name in ("dev", "prod", "test"):
        counts = _variant_counts_for_env_v2(env_name=env_name)
        for indicator_id, ceiling in _TARGET_VARIANT_CEILINGS_V2.items():
            assert counts[indicator_id] <= ceiling
