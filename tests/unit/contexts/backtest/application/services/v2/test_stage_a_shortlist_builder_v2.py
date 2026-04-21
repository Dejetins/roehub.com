from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from types import SimpleNamespace
from typing import Any, Mapping, cast

import numpy as np
import pytest

import trading.contexts.backtest_artifacts.application.services.numba_runtime_v1 as numba_runtime_module
from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    SyntheticArtifactStoreV2,
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.application.dto import BacktestRankingConfig
from trading.contexts.backtest.application.services import (
    BacktestStageABaseVariant,
)
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1
from trading.contexts.backtest_artifacts.application.services.v2 import (
    artifact_runtime_core_v2 as artifact_runtime_core_module,
)
from trading.contexts.backtest_artifacts.application.services.v2 import (
    stage_a_shortlist_builder_v2 as stage_a_shortlist_builder_module,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_backed_stage_b_scorer_v2 import (  # noqa: E501
    BacktestArtifactBackedStageBScorerV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_runtime_plan_v2 import (
    BacktestSignalFeaturesAccessPlanV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_resolver import (
    ArtifactSlotResolverV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactPinnedIdentityV2,
    ArtifactSlotPinnedRuntimeContextV2,
    artifact_market_id_from_coordinates_v2,
)
from trading.contexts.backtest_artifacts.application.services.v2.generic_row_scorer_v2 import (
    GenericRowScorerV2,
    GenericRowScoringInputV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.price_arrays_loader import (
    MmapPriceArraysLoaderV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.signal_features_loader_v2 import (
    MmapSignalFeaturesLoaderV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.signal_matrix_loader import (
    MmapSignalMatrixLoaderV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.stage_a_shortlist_builder_v2 import (  # noqa: E501
    BacktestStageAShortlistBuilderV2,
    PreparedIndicatorChunkInputsV2,
    PreparedIndicatorRowPlanV2,
    compute_target_slice_by_close_time_v2,
)
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp


@dataclass(frozen=True, slots=True)
class _FakeAxis:
    """
    Minimal indicator axis fixture used by artifact Stage A shortlist builder tests.
    """

    name: str
    values: tuple[int | float | str, ...]


@dataclass(frozen=True, slots=True)
class _FakeIndicatorPlan:
    """
    Minimal indicator plan fixture matching the row-addressing data required by the builder.
    """

    indicator_id: str
    axes: tuple[_FakeAxis, ...]


class _FakeGridContext:
    """
    Minimal Stage A grid context fixture for artifact-backed shortlist builder tests.
    """

    def __init__(
        self,
        *,
        base_variants: tuple[BacktestStageABaseVariant, ...],
        indicator_plans: tuple[_FakeIndicatorPlan, ...],
        timeframe_code: str = "15m",
        direction_mode: str = "long-short",
        sizing_mode: str = "all_in",
        execution_params: Mapping[str, float | int | str | bool | None] | None = None,
        signal_features_access: tuple[BacktestSignalFeaturesAccessPlanV2, ...] = (),
    ) -> None:
        """
        Initialize the minimal deterministic grid context consumed by the shortlist builder.

        Args:
            base_variants: Stage A base variants in deterministic enumeration order.
            indicator_plans: Indicator plans whose axis order defines artifact row indexes.
            timeframe_code: Request timeframe literal used for artifact loading.
            direction_mode: Direction mode literal.
            sizing_mode: Sizing mode literal.
            execution_params: Optional execution overrides.
            signal_features_access:
                Optional additive warm-cache access metadata mirroring the runtime-plan surface.
        Returns:
            None.
        Assumptions:
            Stage A builder needs only Stage A enumeration and execution defaults here.
        Raises:
            ValueError: If no base variants are provided.
        Side Effects:
            None.
        """
        if len(base_variants) == 0:
            raise ValueError("_FakeGridContext requires at least one base variant")
        self._base_variants = base_variants
        self.indicator_plans = indicator_plans
        self.timeframe_code = timeframe_code
        self.direction_mode = direction_mode
        self.sizing_mode = sizing_mode
        self.execution_params = execution_params or {}
        self.signal_features_access = signal_features_access
        self.stage_a_variants_total = len(base_variants)

    def iter_stage_a_variants(self) -> tuple[BacktestStageABaseVariant, ...]:
        """
        Return deterministic Stage A base variants in the authored order.

        Args:
            None.
        Returns:
            tuple[BacktestStageABaseVariant, ...]: Stage A base variants fixture.
        Assumptions:
            Tests control the full deterministic order explicitly.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._base_variants

    def stage_a_variant_for_index(
        self,
        *,
        stage_a_index: int,
    ) -> BacktestStageABaseVariant:
        """
        Resolve one deterministic Stage A variant by its flat index.

        Args:
            stage_a_index: Zero-based Stage A index in fixture order.
        Returns:
            BacktestStageABaseVariant: Fixture variant for the requested Stage A index.
        Assumptions:
            Test fixtures keep `stage_a_index` aligned with tuple order so narrowed-frontier
            enumeration can rebuild exact variants without scanning the full raw grid.
        Raises:
            ValueError: If the requested Stage A index falls outside the fixture range.
        Side Effects:
            None.
        """
        if stage_a_index < 0 or stage_a_index >= len(self._base_variants):
            raise ValueError(
                f"_FakeGridContext.stage_a_variant_for_index out of range: {stage_a_index}"
            )
        return self._base_variants[stage_a_index]


class _RecordingSignalMatrixLoader:
    """
    Signal loader wrapper recording subset row selections used by the shortlist builder.
    """

    def __init__(self, *, wrapped: MmapSignalMatrixLoaderV2) -> None:
        """
        Initialize recording wrapper around the real mmap signal loader.

        Args:
            wrapped: Real signal loader used for underlying subset reads.
        Returns:
            None.
        Assumptions:
            Builder tests need real subset row values plus visibility into row selections.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory call log.
        """
        self._wrapped = wrapped
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def load_signal_matrix(self, **kwargs: Any) -> Any:
        """
        Delegate full matrix loads to the wrapped loader for protocol completeness.

        Args:
            **kwargs: Loader keyword arguments.
        Returns:
            Any: Wrapped full matrix payload.
        Assumptions:
            Stage A shortlist builder should prefer `load_signal_rows(...)`.
        Raises:
            None.
        Side Effects:
            Delegates to the wrapped loader.
        """
        return self._wrapped.load_signal_matrix(**kwargs)

    def load_signal_rows(self, **kwargs: Any) -> np.ndarray:
        """
        Record one subset row request and delegate to the wrapped loader.

        Args:
            **kwargs: Loader keyword arguments.
        Returns:
            np.ndarray: Wrapped subset row result.
        Assumptions:
            `row_selection` is a tuple of strictly increasing row indexes in these tests.
        Raises:
            None.
        Side Effects:
            Appends one `(indicator_id, row_selection)` tuple to the in-memory log.
        """
        indicator_id = str(kwargs["indicator_id"])
        row_selection = tuple(int(value) for value in kwargs["row_selection"])
        self.calls.append((indicator_id, row_selection))
        return self._wrapped.load_signal_rows(**kwargs)


class _RecordingSignalFeaturesLoader:
    """
    Signal-feature loader wrapper recording whether warm-cache rows are touched lazily.
    """

    def __init__(self, *, wrapped: MmapSignalFeaturesLoaderV2) -> None:
        """
        Initialize recording wrapper around the real mmap signal-feature loader.

        Args:
            wrapped: Real feature loader used for underlying strict row reads.
        Returns:
            None.
        Assumptions:
            Milestone C exact path should not access feature artifacts until explicitly requested.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory call logs.
        """
        self._wrapped = wrapped
        self.row_calls: list[tuple[str, tuple[int, ...] | slice]] = []

    def load_signal_features_matrix(self, **kwargs: Any) -> Any:
        """
        Delegate full matrix loads to the wrapped loader for protocol completeness.

        Args:
            **kwargs: Loader keyword arguments.
        Returns:
            Any: Wrapped strict full-matrix payload.
        Assumptions:
            This wrapper records lazy row access only; matrix loads are delegated transparently.
        Raises:
            None.
        Side Effects:
            Delegates to the wrapped loader.
        """
        return self._wrapped.load_signal_features_matrix(**kwargs)

    def try_load_signal_features_matrix(self, **kwargs: Any) -> Any:
        """
        Delegate optional full matrix loads to the wrapped loader.

        Args:
            **kwargs: Loader keyword arguments.
        Returns:
            Any: Wrapped optional full-matrix payload.
        Assumptions:
            Builder tests care only about row-access timing, not direct matrix calls.
        Raises:
            None.
        Side Effects:
            Delegates to the wrapped loader.
        """
        return self._wrapped.try_load_signal_features_matrix(**kwargs)

    def load_signal_feature_rows(self, **kwargs: Any) -> Any:
        """
        Record one strict feature-row request and delegate to the wrapped loader.

        Args:
            **kwargs: Loader keyword arguments.
        Returns:
            Any: Wrapped selected feature-row payload.
        Assumptions:
            `row_selection` mirrors the deterministic signal-row ordering for the same chunk.
        Raises:
            None.
        Side Effects:
            Appends one `(indicator_id, row_selection)` tuple to the in-memory log.
        """
        indicator_id = str(kwargs["indicator_id"])
        row_selection = cast(tuple[int, ...] | slice, kwargs["row_selection"])
        self.row_calls.append((indicator_id, row_selection))
        return self._wrapped.load_signal_feature_rows(**kwargs)

    def try_load_signal_feature_rows(self, **kwargs: Any) -> Any:
        """
        Record one optional feature-row request and delegate to the wrapped loader.

        Args:
            **kwargs: Loader keyword arguments.
        Returns:
            Any: Wrapped selected feature-row payload or `None`.
        Assumptions:
            Optional access is the Milestone C exact-path baseline for legacy-compatible slots.
        Raises:
            None.
        Side Effects:
            Appends one `(indicator_id, row_selection)` tuple to the in-memory log.
        """
        indicator_id = str(kwargs["indicator_id"])
        row_selection = cast(tuple[int, ...] | slice, kwargs["row_selection"])
        self.row_calls.append((indicator_id, row_selection))
        return self._wrapped.try_load_signal_feature_rows(**kwargs)


class _FailingSignalMatrixLoader:
    """
    Signal loader stub that fails fast when Stage B unexpectedly rebuilds retained candidates.
    """

    def load_signal_matrix(self, **kwargs: Any) -> Any:
        """
        Fail fast on full-matrix reads because retained candidates must not rebuild here.

        Args:
            **kwargs: Loader keyword arguments.
        Returns:
            Any: This method never returns.
        Assumptions:
            The retained exact payload should already satisfy the Stage B scorer for this test.
        Raises:
            AssertionError: Always, because Stage B must not request full signal matrices here.
        Side Effects:
            None.
        """
        raise AssertionError(
            "Stage B should not load full signal matrices when retained exact payload is present"
        )

    def load_signal_rows(self, **kwargs: Any) -> np.ndarray:
        """
        Fail fast on subset-row reads because retained candidates must not rebuild here.

        Args:
            **kwargs: Loader keyword arguments.
        Returns:
            np.ndarray: This method never returns.
        Assumptions:
            The retained exact payload should already satisfy the Stage B scorer for this test.
        Raises:
            AssertionError: Always, because Stage B must not reload retained signal rows here.
        Side Effects:
            None.
        """
        raise AssertionError(
            "Stage B should consume retained exact payload without reloading signal rows"
        )


class _FlatPriceLoader:
    """
    Minimal price loader returning deterministic flat price arrays for tie-break tests.
    """

    def load_price_arrays(
        self,
        *,
        context: Any,
        timeframe: str,
    ) -> Any:
        """
        Return deterministic price arrays for either request timeframe or `1m` execution timeline.

        Args:
            context: Ignored artifact context.
            timeframe: Requested timeframe literal.
        Returns:
            Any: Simple namespace carrying `close_time` and `ohlcv`.
        Assumptions:
            Tie-break tests need equal metrics and therefore use flat prices only.
        Raises:
            ValueError: If an unsupported timeframe is requested.
        Side Effects:
            None.
        """
        _ = context
        if timeframe == "15m":
            return SimpleNamespace(
                close_time=np.array([2_599, 4_599], dtype=np.int64),
                ohlcv=np.array(
                    [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
                    dtype=np.float32,
                ),
            )
        if timeframe == "1m":
            return SimpleNamespace(
                close_time=np.array([1_599, 2_599, 3_599, 4_599], dtype=np.int64),
                ohlcv=np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
        raise ValueError(f"unsupported timeframe: {timeframe}")

    def load_mapping_arrays(
        self,
        *,
        context: Any,
        timeframe: str,
    ) -> Any:
        """
        Return deterministic `bar_close_1m_idx` mappings for tie-break tests.

        Args:
            context: Ignored artifact context.
            timeframe: Requested timeframe literal.
        Returns:
            Any: Simple namespace carrying `bar_close_1m_idx`.
        Assumptions:
            Tie-break tests use only request-timeframe to `1m` close mappings.
        Raises:
            ValueError: If an unsupported timeframe is requested.
        Side Effects:
            None.
        """
        _ = context
        if timeframe != "15m":
            raise ValueError(f"unsupported timeframe: {timeframe}")
        return SimpleNamespace(bar_close_1m_idx=np.array([1, 3], dtype=np.uint32))

    def load_hit_times_arrays(self, *, context: Any) -> Any:
        """
        Reject unsupported hit-times loading in Stage A shortlist builder tests.

        Args:
            context: Ignored artifact context.
        Returns:
            Any: Never returns.
        Assumptions:
            Stage A no-risk shortlist flow must not touch hit-times artifacts.
        Raises:
            AssertionError: Always.
        Side Effects:
            None.
        """
        _ = context
        raise AssertionError("Stage A shortlist builder must not load hit_times/1m artifacts")


class _ZeroSignalLoader:
    """
    Minimal signal loader returning deterministic zero rows for tie-break tests.
    """

    def load_signal_matrix(self, **kwargs: Any) -> Any:
        """
        Reject unexpected full-matrix loads in tie-break tests.

        Args:
            **kwargs: Ignored loader keyword arguments.
        Returns:
            Any: Never returns.
        Assumptions:
            Builder should use subset row reads only.
        Raises:
            AssertionError: Always.
        Side Effects:
            None.
        """
        _ = kwargs
        raise AssertionError("Stage A shortlist builder must not call load_signal_matrix")

    def load_signal_rows(self, **kwargs: Any) -> np.ndarray:
        """
        Return all-neutral subset rows for deterministic shortlist tie-break assertions.

        Args:
            **kwargs: Loader keyword arguments carrying `row_selection`.
        Returns:
            np.ndarray: All-neutral `int8` subset rows.
        Assumptions:
            Every requested row receives the same zero signal timeline.
        Raises:
            None.
        Side Effects:
            None.
        """
        row_selection = tuple(int(value) for value in kwargs["row_selection"])
        return np.zeros((len(row_selection), 2), dtype=np.int8)


class _InMemorySignalRowsLoader:
    """
    Minimal in-memory signal-row loader for deterministic combo proxy prefilter tests.
    """

    def __init__(self, *, matrices_by_indicator: Mapping[str, np.ndarray]) -> None:
        """
        Initialize one deterministic in-memory signal catalog keyed by indicator id.

        Args:
            matrices_by_indicator: Full signal matrices keyed by indicator id.
        Returns:
            None.
        Assumptions:
            Combo proxy prefilter tests need explicit subset row reads without artifact IO.
        Raises:
            None.
        Side Effects:
            Stores normalized matrices and initializes an in-memory call log.
        """
        self._matrices_by_indicator = {
            indicator_id: np.asarray(matrix, dtype=np.int8)
            for indicator_id, matrix in matrices_by_indicator.items()
        }
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def load_signal_matrix(self, **kwargs: Any) -> Any:
        """
        Reject unexpected full-matrix loads in combo proxy prefilter tests.

        Args:
            **kwargs: Ignored loader keyword arguments.
        Returns:
            Any: Never returns.
        Assumptions:
            Stage A shortlist builder should use subset row loading only on this path.
        Raises:
            AssertionError: Always.
        Side Effects:
            None.
        """
        _ = kwargs
        raise AssertionError("Stage A shortlist builder must not call load_signal_matrix")

    def load_signal_rows(self, **kwargs: Any) -> np.ndarray:
        """
        Return deterministic selected signal rows and record the explicit row selection.

        Args:
            **kwargs: Loader keyword arguments carrying `indicator_id` and `row_selection`.
        Returns:
            np.ndarray: Selected `np.int8` signal rows in the requested order.
        Assumptions:
            Combo proxy prefilter tests keep signal matrices fully in memory.
        Raises:
            KeyError: If the indicator id is unknown in the test fixture.
        Side Effects:
            Appends one `(indicator_id, row_selection)` tuple to the in-memory log.
        """
        indicator_id = str(kwargs["indicator_id"])
        row_selection = tuple(int(value) for value in kwargs["row_selection"])
        self.calls.append((indicator_id, row_selection))
        matrix = self._matrices_by_indicator[indicator_id]
        return np.asarray(matrix[row_selection, :], dtype=np.int8)


class _ComboProxyPriceLoader:
    """
    Minimal price loader returning deterministic rising timelines for combo proxy tests.
    """

    def load_price_arrays(
        self,
        *,
        context: Any,
        timeframe: str,
    ) -> Any:
        """
        Return deterministic request-timeframe and `1m` execution prices for combo tests.

        Args:
            context: Ignored artifact context fixture.
            timeframe: Requested timeframe literal.
        Returns:
            Any: Namespace carrying `close_time` and `ohlcv`.
        Assumptions:
            Rising prices make stronger confirmation-heavy combos outrank weaker combos.
        Raises:
            ValueError: If an unsupported timeframe is requested.
        Side Effects:
            None.
        """
        _ = context
        if timeframe == "15m":
            return SimpleNamespace(
                close_time=np.array([2_599, 4_599, 6_599], dtype=np.int64),
                ohlcv=np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                        [4.0, 4.0, 4.0, 4.0, 4.0],
                    ],
                    dtype=np.float32,
                ),
            )
        if timeframe == "1m":
            return SimpleNamespace(
                close_time=np.array(
                    [1_599, 2_599, 3_599, 4_599, 5_599, 6_599],
                    dtype=np.int64,
                ),
                ohlcv=np.array(
                    [
                        [1.00, 1.00, 1.00, 1.05, 1.00],
                        [1.05, 1.05, 1.05, 1.10, 1.00],
                        [1.10, 1.10, 1.10, 1.20, 1.00],
                        [1.20, 1.20, 1.20, 1.30, 1.00],
                        [1.30, 1.30, 1.30, 1.40, 1.00],
                        [1.40, 1.40, 1.40, 1.50, 1.00],
                    ],
                    dtype=np.float32,
                ),
            )
        raise ValueError(f"unsupported timeframe: {timeframe}")

    def load_mapping_arrays(
        self,
        *,
        context: Any,
        timeframe: str,
    ) -> Any:
        """
        Return deterministic request-timeframe to `1m` close mappings for combo tests.

        Args:
            context: Ignored artifact context fixture.
            timeframe: Requested timeframe literal.
        Returns:
            Any: Namespace carrying `bar_close_1m_idx`.
        Assumptions:
            Combo proxy prefilter integration tests use three request-timeframe bars.
        Raises:
            ValueError: If an unsupported timeframe is requested.
        Side Effects:
            None.
        """
        _ = context
        if timeframe != "15m":
            raise ValueError(f"unsupported timeframe: {timeframe}")
        return SimpleNamespace(bar_close_1m_idx=np.array([1, 3, 5], dtype=np.uint32))

    def load_hit_times_arrays(self, *, context: Any) -> Any:
        """
        Reject unsupported hit-times loading in combo proxy prefilter tests.

        Args:
            context: Ignored artifact context fixture.
        Returns:
            Any: Never returns.
        Assumptions:
            Stage A shortlist flow must not touch hit-times artifacts here.
        Raises:
            AssertionError: Always.
        Side Effects:
            None.
        """
        _ = context
        raise AssertionError("Stage A shortlist builder must not load hit_times/1m artifacts")


@pytest.fixture()
def synthetic_artifact_store_v2(tmp_path: Any) -> SyntheticArtifactStoreV2:
    """
    Build a strict synthetic artifact store used by artifact-backed shortlist builder tests.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        SyntheticArtifactStoreV2: Deterministic strict artifact store fixture.
    Assumptions:
        Builder tests need strict price, mapping, and signal artifacts under one slot root.
    Raises:
        OSError: If the synthetic artifact tree cannot be created.
    Side Effects:
        Creates a temporary artifact tree under `tmp_path`.
    """
    return build_synthetic_artifact_store_v2(tmp_path=tmp_path)


def test_stage_a_shortlist_builder_v2_chunked_processing_matches_single_batch_reference(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify chunked variant processing yields the same deterministic shortlist as one large batch.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Synthetic store publishes two Stage A rows where the long row should outrank the short row.
    Raises:
        AssertionError: If chunk equivalence or deterministic ordering drifts.
    Side Effects:
        Memory-maps strict artifact arrays from the synthetic store.
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    grid_context = _grid_context_for_windows(windows=(10, 20))
    target_time_range = _synthetic_target_time_range()
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=MmapPriceArraysLoaderV2(artifact_loader=store.loader),
        signal_matrix_loader=MmapSignalMatrixLoaderV2(artifact_loader=store.loader),
    )

    single_batch = builder.build_shortlist(
        grid_context=cast(Any, grid_context),
        artifact_context=context,
        target_time_range=target_time_range,
        shortlist_limit=2,
        batch_size=8,
    )
    chunked = builder.build_shortlist(
        grid_context=cast(Any, grid_context),
        artifact_context=context,
        target_time_range=target_time_range,
        shortlist_limit=2,
        batch_size=1,
    )

    assert tuple(row.base_variant.base_variant_key for row in single_batch) == tuple(
        row.base_variant.base_variant_key for row in chunked
    )
    assert tuple(round(row.total_return_pct, 6) for row in single_batch) == tuple(
        round(row.total_return_pct, 6) for row in chunked
    )
    assert single_batch[0].base_variant.stage_a_index == 1
    assert single_batch[0].total_return_pct > single_batch[1].total_return_pct


def test_stage_a_shortlist_builder_v2_uses_subset_row_loading_for_selected_variants_only(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify artifact Stage A loads only selected signal rows instead of materializing full matrices.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        One selected variant should request only its own signal row from the mmap loader, and
        streaming exact scoring should reuse that retained chunk instead of reloading a deferred
        replay batch.
    Raises:
        AssertionError: If builder loads unrelated rows or uses full-matrix reads.
    Side Effects:
        Memory-maps strict artifact arrays from the synthetic store.
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    recording_loader = _RecordingSignalMatrixLoader(
        wrapped=MmapSignalMatrixLoaderV2(artifact_loader=store.loader)
    )
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=MmapPriceArraysLoaderV2(artifact_loader=store.loader),
        signal_matrix_loader=recording_loader,
    )

    shortlist = builder.build_shortlist(
        grid_context=cast(Any, _grid_context_for_windows(windows=(20,))),
        artifact_context=context,
        target_time_range=_synthetic_target_time_range(),
        shortlist_limit=1,
        batch_size=1,
    )

    assert len(shortlist) == 1
    assert recording_loader.calls == [
        ("ma.ema", (0,)),
        ("ma.ema", (0,)),
    ]


def test_stage_a_shortlist_builder_v2_prefilters_rows_before_exact_evaluation(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify Stage A narrows the retained frontier before exact evaluation and keeps it deterministic.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        The synthetic store's rising second bar should make the long row survive the price-aware
        row-local prefilter while the short row is removed before exact work, and streaming exact
        scoring should avoid a later retained-batch reload.
    Raises:
        AssertionError: If the retained frontier is not narrowed deterministically.
    Side Effects:
        Memory-maps strict artifact arrays from the synthetic store.
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    recording_loader = _RecordingSignalMatrixLoader(
        wrapped=MmapSignalMatrixLoaderV2(artifact_loader=store.loader)
    )
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=MmapPriceArraysLoaderV2(artifact_loader=store.loader),
        signal_matrix_loader=recording_loader,
    )

    shortlist = builder.build_shortlist(
        grid_context=cast(Any, _grid_context_for_windows(windows=(10, 20))),
        artifact_context=context,
        target_time_range=_synthetic_target_time_range(),
        shortlist_limit=1,
        batch_size=1,
    )

    assert len(shortlist) == 1
    assert shortlist[0].base_variant.stage_a_index == 1
    assert recording_loader.calls == [
        ("ma.ema", (0, 1)),
        ("ma.ema", (1,)),
    ]


def test_stage_a_shortlist_builder_v2_hands_retained_exact_payload_into_stage_b(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify retained candidates keep fast Stage B breadth scoring while exact
    finalist replay stays available.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        The in-memory Stage A shortlist is the new internal hand-off surface for retained
        candidates, while Stage B breadth ranking should keep the fast Stage B path enabled for
        `total_return_pct` and exact replay must remain authoritative for finalist details.
    Raises:
        AssertionError: If the retained payload is dropped, breadth scoring falls back to signal
            reloads, or exact finalist replay stops working.
    Side Effects:
        Memory-maps strict artifact arrays from the synthetic store.
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    target_time_range = _synthetic_target_time_range()
    grid_context = _grid_context_for_windows(windows=(20,))
    price_loader = MmapPriceArraysLoaderV2(artifact_loader=store.loader)
    shortlist = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=MmapSignalMatrixLoaderV2(artifact_loader=store.loader),
    ).build_shortlist(
        grid_context=cast(Any, grid_context),
        artifact_context=context,
        target_time_range=target_time_range,
        shortlist_limit=1,
        batch_size=1,
    )

    assert len(shortlist) == 1
    retained_exact_payload = shortlist[0].retained_exact_payload
    assert shortlist[0].no_risk_metrics is not None
    assert retained_exact_payload is not None
    assert retained_exact_payload.memory_shape_bucket == "compact_trade_arrays"
    assert retained_exact_payload.trade_count > 0
    assert not hasattr(retained_exact_payload, "final_signal_row")

    signal_prices = price_loader.load_price_arrays(
        context=context,
        timeframe=grid_context.timeframe_code,
    )
    scorer = BacktestArtifactBackedStageBScorerV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=cast(Any, _FailingSignalMatrixLoader()),
        artifact_context=context,
        target_time_range=target_time_range,
        report_target_slice=compute_target_slice_by_close_time_v2(
            close_time=signal_prices.close_time,
            target_time_range=target_time_range,
        ),
        direction_mode=grid_context.direction_mode,
        sizing_mode=grid_context.sizing_mode,
        execution_params=grid_context.execution_params,
        market_id=artifact_market_id_from_coordinates_v2(store.coordinates),
        signal_timeframe=grid_context.timeframe_code,
        indicator_grids=(),
    )
    tp_pct = float(scorer._local_hit_times.tp_values[0] * 100.0)
    sl_pct = float(scorer._local_hit_times.sl_values[0] * 100.0)

    stage_b_tasks = artifact_runtime_core_module.iter_stage_b_tasks_v2(
        template=cast(
            Any,
            SimpleNamespace(
                direction_mode=grid_context.direction_mode,
                sizing_mode=grid_context.sizing_mode,
                execution_params=grid_context.execution_params,
            ),
        ),
        runtime_plan=cast(
            Any,
            SimpleNamespace(
                risk_variants=(
                    SimpleNamespace(
                        risk_index=0,
                        risk_params={
                            "tp_enabled": True,
                            "tp_pct": tp_pct,
                            "sl_enabled": True,
                            "sl_pct": sl_pct,
                        },
                    ),
                )
            ),
        ),
        shortlist=shortlist,
    )
    assert len(stage_b_tasks) == 1
    task = stage_b_tasks[0]
    assert task.retained_exact_payload is retained_exact_payload
    assert task.retained_exact_payload is not None
    assert task.retained_exact_payload.memory_shape_bucket == "compact_trade_arrays"

    artifact_runtime_core_module.prime_retained_exact_payload_if_supported_v2(
        scorer=scorer,
        task=task,
    )
    scorer.configure_stage_ranking_context(
        stage="stage_b",
        primary_metric="total_return_pct",
    )
    scored_row, metrics = artifact_runtime_core_module.score_stage_b_task_with_metrics_v2(
        task=task,
        candles=cast(Any, SimpleNamespace()),
        score_variant_metric=scorer.score_variant_metric,
    )

    assert scored_row.variant_key == task.variant_key
    assert scored_row.best_tp_pct == tp_pct
    assert scored_row.best_sl_pct == sl_pct
    assert metrics["total_return_pct"] == scored_row.total_return_pct
    assert "trade_count" not in metrics
    assert scored_row.summary_metrics_json == {"total_return_pct": scored_row.total_return_pct}

    details = scorer.score_variant_with_details(
        stage="stage_b",
        candles=cast(Any, SimpleNamespace()),
        indicator_selections=task.indicator_selections,
        signal_params=task.signal_params,
        risk_params=task.risk_params,
        indicator_variant_key=task.indicator_variant_key,
        variant_key=task.variant_key,
    )

    assert details.metrics["total_return_pct"] == scored_row.total_return_pct
    assert "trade_count" in details.metrics


def test_stage_a_shortlist_builder_v2_keeps_retained_frontier_row_order_explicit(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify the retained frontier keeps deterministic ranked row ordering explicit and stable.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        The synthetic long row should outrank the short row in row-local prefilter order.
    Raises:
        AssertionError: If the retained frontier loses its explicit deterministic row ordering.
    Side Effects:
        Memory-maps strict artifact arrays from the synthetic store.
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    grid_context = _grid_context_for_windows(windows=(10, 20))
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=MmapPriceArraysLoaderV2(artifact_loader=store.loader),
        signal_matrix_loader=MmapSignalMatrixLoaderV2(artifact_loader=store.loader),
    )
    signal_prices = builder.price_arrays_loader.load_price_arrays(
        context=context,
        timeframe="15m",
    )
    signal_target_slice = compute_target_slice_by_close_time_v2(
        close_time=signal_prices.close_time,
        target_time_range=_synthetic_target_time_range(),
    )
    row_plans = tuple(
        PreparedIndicatorRowPlanV2.from_indicator_plan(plan=plan)
        for plan in cast(Any, grid_context).indicator_plans
    )

    retained_frontier = builder._build_row_prefilter_frontier(
        row_plans=row_plans,
        grid_context=cast(Any, grid_context),
        artifact_context=context,
        signal_target_slice=signal_target_slice,
        local_signal_close=np.asarray(
            signal_prices.ohlcv[signal_target_slice, 3],
            dtype=np.float64,
        ),
        execution_params=builder._resolve_execution_params(
            grid_context=cast(Any, grid_context),
            market_id=artifact_market_id_from_coordinates_v2(context.coordinates),
        ),
        shortlist_limit=2,
        cancel_checker=None,
    )

    assert retained_frontier["ma.ema"].retained_row_indexes == (1, 0)


def test_stage_a_shortlist_builder_v2_batch_prefilter_proxy_scores_match_scalar_reference() -> None:
    """
    Verify batched Stage A row-prefilter proxy scores match the scalar reference helper.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The new batch-friendly kernel must preserve the exact same fee-adjusted proxy score as
        the scalar reference path for every retained row.
    Raises:
        AssertionError: If the batched proxy scores drift from scalar reference results.
    Side Effects:
        May trigger Numba compilation on first use.
    """
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=cast(Any, object()),
        signal_matrix_loader=cast(Any, object()),
    )
    signal_rows = np.asarray(
        (
            (1, 1, 0, -1, 0),
            (0, 1, 1, 0, 0),
            (-1, 0, 1, 1, 0),
        ),
        dtype=np.int8,
    )
    local_signal_close = np.asarray(
        (100.0, 101.0, 102.0, 99.0, 103.0),
        dtype=np.float64,
    )
    execution_params = ExecutionParamsV1(
        direction_mode="long-short",
        sizing_mode="all_in",
        init_cash_quote=10000.0,
        fixed_quote=100.0,
        safe_profit_percent=30.0,
        fee_pct=0.075,
        slippage_pct=0.01,
    )

    batch_scores = builder._prefilter_proxy_scores_for_rows(
        signal_rows=signal_rows,
        local_signal_close=local_signal_close,
        execution_params=execution_params,
    )
    scalar_scores = np.asarray(
        [
            builder._prefilter_proxy_score_for_row(
                row_index=row_index,
                signal_rows=signal_rows,
                local_signal_close=local_signal_close,
                execution_params=execution_params,
            )
            for row_index in range(int(signal_rows.shape[0]))
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(batch_scores, scalar_scores)


def test_stage_a_shortlist_builder_v2_numeric_prefilter_ranking_matches_generic_reference() -> None:
    """
    Verify numeric Stage A row ranking preserves the former generic tie-break semantics.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The canonical parity path should avoid `GenericRowScorerV2` row objects while keeping the
        same proxy-first ordering and lexicographic `stable_identity` tie-breaks as the replaced
        generic reference path.
    Raises:
        AssertionError: If the numeric ranking drifts from the former generic reference order.
    Side Effects:
        May trigger Numba compilation on first use.
    """
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=cast(Any, object()),
        signal_matrix_loader=cast(Any, object()),
    )
    signal_rows = np.zeros((12, 5), dtype=np.int8)
    local_signal_close = np.asarray(
        (100.0, 100.0, 100.0, 100.0, 100.0),
        dtype=np.float64,
    )
    execution_params = ExecutionParamsV1(
        direction_mode="long-short",
        sizing_mode="all_in",
        init_cash_quote=10000.0,
        fixed_quote=100.0,
        safe_profit_percent=30.0,
        fee_pct=0.075,
        slippage_pct=0.01,
    )

    ranked_row_indexes = builder._rank_prefilter_row_indexes_numeric(
        indicator_id="ma.ema",
        signal_rows=signal_rows,
        local_signal_close=local_signal_close,
        execution_params=execution_params,
        scoring=builder.row_scorer.scoring,
    )
    generic_reference = GenericRowScorerV2()
    scored_rows = generic_reference.score_rows(
        rows=tuple(
            GenericRowScoringInputV2(
                indicator_id="ma.ema",
                row_index=row_index,
                stable_identity=f"ma.ema:{row_index}",
                signal_row=np.asarray(signal_rows[row_index, :], dtype=np.int8),
            )
            for row_index in range(int(signal_rows.shape[0]))
        )
    )
    proxy_scores = builder._prefilter_proxy_scores_for_rows(
        signal_rows=signal_rows,
        local_signal_close=local_signal_close,
        execution_params=execution_params,
    )
    scorer_sorted_row_indexes = np.asarray(
        [payload.row_index for payload in scored_rows],
        dtype=np.int64,
    )
    generic_ranked_row_indexes = scorer_sorted_row_indexes[
        np.argsort(-proxy_scores[scorer_sorted_row_indexes], kind="mergesort")
    ]

    assert tuple(int(value) for value in ranked_row_indexes.tolist()) == tuple(
        int(value) for value in generic_ranked_row_indexes.tolist()
    )
    assert tuple(int(value) for value in ranked_row_indexes[:6].tolist()) == (
        0,
        1,
        10,
        11,
        2,
        3,
    )


def test_stage_a_shortlist_builder_v2_row_prefilter_uses_batch_proxy_scores(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify Stage A row prefilter avoids both scalar proxy helpers and GenericRowScorerV2 objects.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
        monkeypatch: Pytest fixture used to fail fast if the scalar helper is called.
    Returns:
        None.
    Assumptions:
        Stage A row prefilter should rank rows through the batch-friendly proxy-score path while
        preserving the same deterministic retained order without calling `GenericRowScorerV2`.
    Raises:
        AssertionError: If the row-prefilter path falls back to scalar proxy scoring or the
            universal generic row-scorer object path.
    Side Effects:
        Memory-maps strict artifact arrays from the synthetic store.
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    grid_context = _grid_context_for_windows(windows=(10, 20))
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=MmapPriceArraysLoaderV2(artifact_loader=store.loader),
        signal_matrix_loader=MmapSignalMatrixLoaderV2(artifact_loader=store.loader),
    )
    signal_prices = builder.price_arrays_loader.load_price_arrays(
        context=context,
        timeframe="15m",
    )
    signal_target_slice = compute_target_slice_by_close_time_v2(
        close_time=signal_prices.close_time,
        target_time_range=_synthetic_target_time_range(),
    )
    row_plans = tuple(
        PreparedIndicatorRowPlanV2.from_indicator_plan(plan=plan)
        for plan in cast(Any, grid_context).indicator_plans
    )

    def _raise_scalar_proxy_call(*args: Any, **kwargs: Any) -> float:
        """
        Fail fast if the deprecated scalar row-prefilter proxy helper is used.

        Args:
            *args: Ignored positional arguments from the patched method call.
            **kwargs: Ignored keyword arguments from the patched method call.
        Returns:
            float: This helper never returns successfully.
        Assumptions:
            The batch-friendly proxy-score path is the only acceptable hot-path implementation.
        Raises:
            AssertionError: Always, because the scalar path should stay unused here.
        Side Effects:
            None.
        """
        raise AssertionError("Stage A row prefilter should use batch proxy scores")

    def _raise_generic_row_scorer_call(*args: Any, **kwargs: Any) -> Any:
        """
        Fail fast if Stage A re-enters the deprecated GenericRowScorerV2 hot path.

        Args:
            *args: Ignored positional arguments from the patched method call.
            **kwargs: Ignored keyword arguments from the patched method call.
        Returns:
            Any: This helper never returns successfully.
        Assumptions:
            The canonical parity row prefilter should stay numeric and matrix-first.
        Raises:
            AssertionError: Always, because GenericRowScorerV2 must be bypassed here.
        Side Effects:
            None.
        """
        raise AssertionError("Stage A row prefilter should bypass GenericRowScorerV2")

    monkeypatch.setattr(
        stage_a_shortlist_builder_module.BacktestStageAShortlistBuilderV2,
        "_prefilter_proxy_score_for_row",
        _raise_scalar_proxy_call,
    )
    monkeypatch.setattr(
        GenericRowScorerV2,
        "score_rows",
        _raise_generic_row_scorer_call,
    )

    retained_frontier = builder._build_row_prefilter_frontier(
        row_plans=row_plans,
        grid_context=cast(Any, grid_context),
        artifact_context=context,
        signal_target_slice=signal_target_slice,
        local_signal_close=np.asarray(
            signal_prices.ohlcv[signal_target_slice, 3],
            dtype=np.float64,
        ),
        execution_params=builder._resolve_execution_params(
            grid_context=cast(Any, grid_context),
            market_id=artifact_market_id_from_coordinates_v2(context.coordinates),
        ),
        shortlist_limit=2,
        cancel_checker=None,
    )

    assert retained_frontier["ma.ema"].retained_row_indexes == (1, 0)


def test_stage_a_shortlist_builder_v2_row_prefilter_keeps_signal_features_out_of_hot_path(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify canonical Stage A row prefilter does not require optional cached `signal_features`.

    Args:
        synthetic_artifact_store_v2: Fixture with strict signal and signal-feature artifacts.
    Returns:
        None.
    Assumptions:
        The numeric parity hot path should rank retained rows from signal matrices alone while
        leaving warm-cache `signal_features` access lazy and opt-in.
    Raises:
        AssertionError: If the row prefilter eagerly reads cached feature rows.
    Side Effects:
        Memory-maps strict signal rows from the synthetic store.
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    grid_context = _grid_context_for_windows(windows=(10, 20))
    _attach_signal_features_access(grid_context=grid_context)
    price_loader = MmapPriceArraysLoaderV2(artifact_loader=store.loader)
    signal_prices = price_loader.load_price_arrays(context=context, timeframe="15m")
    signal_target_slice = compute_target_slice_by_close_time_v2(
        close_time=signal_prices.close_time,
        target_time_range=_synthetic_target_time_range(),
    )
    recording_feature_loader = _RecordingSignalFeaturesLoader(
        wrapped=MmapSignalFeaturesLoaderV2(artifact_loader=store.loader)
    )
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=MmapSignalMatrixLoaderV2(artifact_loader=store.loader),
        signal_features_loader=recording_feature_loader,
    )
    row_plans = tuple(
        PreparedIndicatorRowPlanV2.from_indicator_plan(plan=plan)
        for plan in cast(Any, grid_context).indicator_plans
    )

    retained_frontier = builder._build_row_prefilter_frontier(
        row_plans=row_plans,
        grid_context=cast(Any, grid_context),
        artifact_context=context,
        signal_target_slice=signal_target_slice,
        local_signal_close=np.asarray(
            signal_prices.ohlcv[signal_target_slice, 3],
            dtype=np.float64,
        ),
        execution_params=builder._resolve_execution_params(
            grid_context=cast(Any, grid_context),
            market_id=artifact_market_id_from_coordinates_v2(context.coordinates),
        ),
        shortlist_limit=2,
        cancel_checker=None,
    )

    assert retained_frontier["ma.ema"].retained_row_indexes == (1, 0)
    assert recording_feature_loader.row_calls == []


def test_stage_a_shortlist_builder_v2_row_prefilter_does_not_read_generic_row_scorer_config(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify the numeric parity prefilter does not depend on GenericRowScorerV2 config fallback.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        `C3` requires the canonical row-prefilter hot path to bypass `GenericRowScorerV2`
        entirely, including fallback shortlist-weight access when no execution profile is present.
    Raises:
        AssertionError: If the row-prefilter path still reads `builder.row_scorer.scoring`.
    Side Effects:
        Memory-maps strict signal rows from the synthetic store.
    """

    class _ExplodingRowScorer:
        """
        Minimal sentinel scorer that fails if Stage A reads generic scorer config.
        """

        @property
        def scoring(self) -> Any:
            """
            Fail fast if the numeric parity path still reads GenericRowScorerV2 weights.

            Args:
                None.
            Returns:
                Any: This property never returns successfully.
            Assumptions:
                The parity hot path should source default numeric weights without consulting the
                generic scorer object.
            Raises:
                AssertionError: Always, because the generic scorer config must stay unused here.
            Side Effects:
                None.
            """
            raise AssertionError("Stage A row prefilter should not read GenericRowScorerV2 config")

    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    grid_context = _grid_context_for_windows(windows=(10, 20))
    price_loader = MmapPriceArraysLoaderV2(artifact_loader=store.loader)
    signal_prices = price_loader.load_price_arrays(context=context, timeframe="15m")
    signal_target_slice = compute_target_slice_by_close_time_v2(
        close_time=signal_prices.close_time,
        target_time_range=_synthetic_target_time_range(),
    )
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=MmapSignalMatrixLoaderV2(artifact_loader=store.loader),
    )
    object.__setattr__(builder, "row_scorer", cast(Any, _ExplodingRowScorer()))
    row_plans = tuple(
        PreparedIndicatorRowPlanV2.from_indicator_plan(plan=plan)
        for plan in cast(Any, grid_context).indicator_plans
    )

    retained_frontier = builder._build_row_prefilter_frontier(
        row_plans=row_plans,
        grid_context=cast(Any, grid_context),
        artifact_context=context,
        signal_target_slice=signal_target_slice,
        local_signal_close=np.asarray(
            signal_prices.ohlcv[signal_target_slice, 3],
            dtype=np.float64,
        ),
        execution_params=builder._resolve_execution_params(
            grid_context=cast(Any, grid_context),
            market_id=artifact_market_id_from_coordinates_v2(context.coordinates),
        ),
        shortlist_limit=2,
        cancel_checker=None,
    )

    assert retained_frontier["ma.ema"].retained_row_indexes == (1, 0)


def test_stage_a_shortlist_builder_v2_exposes_optional_signal_features_warm_cache(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify Stage A can expose cached feature rows as an additive warm-cache runtime surface.

    Args:
        synthetic_artifact_store_v2: Fixture with strict signal and signal-feature artifacts.
    Returns:
        None.
    Assumptions:
        Warm-cache access must reuse the same deterministic row ordering as signal-row loading.
    Raises:
        AssertionError: If feature rows are missing or lose deterministic alignment.
    Side Effects:
        Memory-maps strict signal and `signal_features` artifacts from the synthetic store.
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    grid_context = _grid_context_for_windows(windows=(10,))
    _attach_signal_features_access(grid_context=grid_context)
    price_loader = MmapPriceArraysLoaderV2(artifact_loader=store.loader)
    signal_prices = price_loader.load_price_arrays(context=context, timeframe="15m")
    signal_target_slice = compute_target_slice_by_close_time_v2(
        close_time=signal_prices.close_time,
        target_time_range=_synthetic_target_time_range(),
    )
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=MmapSignalMatrixLoaderV2(artifact_loader=store.loader),
        signal_features_loader=MmapSignalFeaturesLoaderV2(artifact_loader=store.loader),
    )
    row_plans = tuple(
        PreparedIndicatorRowPlanV2.from_indicator_plan(plan=plan)
        for plan in cast(Any, grid_context).indicator_plans
    )

    chunk_inputs = builder.load_chunk_runtime_inputs(
        row_plans=row_plans,
        chunk_variants=cast(Any, grid_context).iter_stage_a_variants(),
        grid_context=cast(Any, grid_context),
        artifact_context=context,
        signal_target_slice=signal_target_slice,
    )

    assert len(chunk_inputs) == 1
    assert isinstance(chunk_inputs[0], PreparedIndicatorChunkInputsV2)
    assert chunk_inputs[0].signal_row_selection == (0,)
    feature_rows = chunk_inputs[0].load_signal_feature_rows()
    assert feature_rows is not None
    assert feature_rows.feature_names == (
        "nonzero_count",
        "long_count",
        "short_count",
        "activity_ratio",
        "direction_balance",
        "transition_count",
    )
    np.testing.assert_allclose(
        feature_rows.rows,
        np.array(((1.0, 0.0, 1.0, 0.5, -1.0, 1.0),), dtype=np.float32),
    )


def test_prepared_indicator_chunk_inputs_v2_keeps_signal_row_selection_additive() -> None:
    """
    Verify retained row-address metadata stays additive on prepared chunk inputs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Older callers may construct prepared chunk inputs without explicit row addressing, and the
        retained-frontier cutover should not force them to pass new constructor arguments.
    Raises:
        AssertionError: If omitted row-address metadata no longer defaults deterministically.
    Side Effects:
        None.
    """
    chunk_inputs = PreparedIndicatorChunkInputsV2(
        indicator_id="ma.ema",
        signal_rows=np.zeros((2, 3), dtype=np.int8),
    )

    assert chunk_inputs.signal_row_selection == (0, 1)


def test_stage_a_shortlist_builder_v2_keeps_signal_features_lazy_until_explicit_access(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify load_chunk_runtime_inputs does not eagerly touch warm-cache feature artifacts.

    Args:
        synthetic_artifact_store_v2: Fixture with strict signal and signal-feature artifacts.
    Returns:
        None.
    Assumptions:
        Milestone C should carry feature-access metadata through exact path without preloading the
        additive feature family.
    Raises:
        AssertionError: If Stage A eagerly accesses feature rows before explicit demand.
    Side Effects:
        Reads strict signal rows and then, only after explicit access, memory-maps feature rows.
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    grid_context = _grid_context_for_windows(windows=(10,))
    _attach_signal_features_access(grid_context=grid_context)
    price_loader = MmapPriceArraysLoaderV2(artifact_loader=store.loader)
    signal_prices = price_loader.load_price_arrays(context=context, timeframe="15m")
    signal_target_slice = compute_target_slice_by_close_time_v2(
        close_time=signal_prices.close_time,
        target_time_range=_synthetic_target_time_range(),
    )
    recording_feature_loader = _RecordingSignalFeaturesLoader(
        wrapped=MmapSignalFeaturesLoaderV2(artifact_loader=store.loader)
    )
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=MmapSignalMatrixLoaderV2(artifact_loader=store.loader),
        signal_features_loader=recording_feature_loader,
    )
    row_plans = tuple(
        PreparedIndicatorRowPlanV2.from_indicator_plan(plan=plan)
        for plan in cast(Any, grid_context).indicator_plans
    )

    chunk_inputs = builder.load_chunk_runtime_inputs(
        row_plans=row_plans,
        chunk_variants=cast(Any, grid_context).iter_stage_a_variants(),
        grid_context=cast(Any, grid_context),
        artifact_context=context,
        signal_target_slice=signal_target_slice,
    )

    assert recording_feature_loader.row_calls == []
    feature_rows = chunk_inputs[0].load_signal_feature_rows()
    assert feature_rows is not None
    assert recording_feature_loader.row_calls == [("ma.ema", (0,))]


def test_stage_a_shortlist_builder_v2_keeps_exact_shortlist_unchanged_without_signal_features(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
    tmp_path: Any,
) -> None:
    """
    Verify exact Stage A shortlist semantics stay unchanged when warm-cache artifacts are absent.

    Args:
        synthetic_artifact_store_v2: Fixture with additive signal-feature artifacts.
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Feature access in Milestone C is optional and must not change winners or scores.
    Raises:
        AssertionError: If exact shortlist ordering or scores drift between slot variants.
    Side Effects:
        Builds one extra synthetic legacy-style store without `signal_features`.
    """
    feature_store = synthetic_artifact_store_v2
    legacy_store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path / "legacy_store",
        inactive_include_signal_features=False,
    )
    target_time_range = _synthetic_target_time_range()
    grid_context = _grid_context_for_windows(windows=(10, 20))
    _attach_signal_features_access(grid_context=grid_context)

    feature_shortlist = _build_shortlist_with_optional_features(
        store=feature_store,
        grid_context=grid_context,
        target_time_range=target_time_range,
    )
    legacy_shortlist = _build_shortlist_with_optional_features(
        store=legacy_store,
        grid_context=grid_context,
        target_time_range=target_time_range,
    )

    assert tuple(row.base_variant.base_variant_key for row in feature_shortlist) == tuple(
        row.base_variant.base_variant_key for row in legacy_shortlist
    )
    assert tuple(round(row.total_return_pct, 6) for row in feature_shortlist) == tuple(
        round(row.total_return_pct, 6) for row in legacy_shortlist
    )


def test_stage_a_shortlist_builder_v2_breaks_metric_ties_by_base_variant_key() -> None:
    """
    Verify shortlist ordering remains deterministic when no-risk metrics tie exactly.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        All-neutral signals and flat prices produce identical zero metrics for every variant.
    Raises:
        AssertionError: If base-variant key tie-break ordering drifts.
    Side Effects:
        None.
    """
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=_FlatPriceLoader(),
        signal_matrix_loader=_ZeroSignalLoader(),
    )
    grid_context = _FakeGridContext(
        base_variants=(
            _base_variant(stage_a_index=0, window=10, base_variant_key="b" * 64),
            _base_variant(stage_a_index=1, window=20, base_variant_key="a" * 64),
        ),
        indicator_plans=(
            _FakeIndicatorPlan(
                indicator_id="ma.ema",
                axes=(_FakeAxis(name="window", values=(10, 20)),),
            ),
        ),
    )

    shortlist = builder.build_shortlist(
        grid_context=cast(Any, grid_context),
        artifact_context=cast(
            ArtifactSlotPinnedRuntimeContextV2,
            SimpleNamespace(
                coordinates=SimpleNamespace(
                    exchange="binance", market_type="spot", symbol="BTCUSDT"
                )
            ),
        ),
        target_time_range=_synthetic_target_time_range(),
        shortlist_limit=2,
        ranking=BacktestRankingConfig(primary_metric="total_return_pct"),
        batch_size=1,
    )

    assert tuple(row.base_variant.base_variant_key for row in shortlist) == ("a" * 64, "b" * 64)


def test_stage_a_shortlist_builder_v2_streaming_exact_scoring_runs_per_retained_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify Stage A exact scoring streams per retained chunk instead of deferred replay.

    Args:
        monkeypatch: pytest fixture used to record streaming exact chunk merges.
    Returns:
        None.
    Assumptions:
        The three-indicator fixture keeps all rows through row prefilter, then combo proxy
        prefilter should retain chunk-local exact work immediately in deterministic Stage A order.
    Raises:
        AssertionError: If Stage A falls back to one deferred replay batch or chunk order drifts.
    Side Effects:
        Monkeypatches one builder method for streaming exact-merge inspection.
    """
    streaming_exact_chunks = _record_streaming_exact_chunks(monkeypatch)
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=_ComboProxyPriceLoader(),
        signal_matrix_loader=_combo_proxy_signal_loader(),
    )

    shortlist = builder.build_shortlist(
        grid_context=cast(Any, _combo_proxy_grid_context()),
        artifact_context=cast(Any, _combo_proxy_artifact_context()),
        target_time_range=_combo_proxy_target_time_range(),
        shortlist_limit=2,
        batch_size=6,
    )

    assert streaming_exact_chunks == [(0, 1, 2, 3), (6, 7)]
    assert tuple(row.base_variant.stage_a_index for row in shortlist) == (0, 1)


def test_stage_a_shortlist_builder_v2_checkpoints_report_narrowed_frontier_cardinality(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify Stage A progress checkpoints expose retained-frontier breadth instead of raw-grid total.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        The synthetic row prefilter keeps only one of two raw Stage A rows, so retained-frontier
        enumeration should report checkpoint totals of `1` instead of the raw
        `stage_a_variants_total=2`.
    Raises:
        AssertionError: If Stage A silently falls back to raw-grid checkpoint totals.
    Side Effects:
        Memory-maps strict artifact arrays from the synthetic store.
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    grid_context = _grid_context_for_windows(windows=(10, 20))
    checkpoints: list[tuple[int, int]] = []

    shortlist = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=MmapPriceArraysLoaderV2(artifact_loader=store.loader),
        signal_matrix_loader=MmapSignalMatrixLoaderV2(artifact_loader=store.loader),
    ).build_shortlist(
        grid_context=cast(Any, grid_context),
        artifact_context=context,
        target_time_range=_synthetic_target_time_range(),
        shortlist_limit=1,
        batch_size=1,
        on_checkpoint=lambda processed, total: checkpoints.append((processed, total)),
    )

    assert checkpoints == [(1, 1)]
    assert checkpoints[-1][1] < grid_context.stage_a_variants_total
    assert tuple(row.base_variant.stage_a_index for row in shortlist) == (1,)


def test_stage_a_shortlist_builder_v2_uses_explicit_retained_variants_when_present() -> None:
    """
    Verify narrowed runtime plans stream their explicit retained Stage A variants directly.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Hierarchical shortlist plans already preserve sparse exact `stage_a_index` ordering, so
        Stage A must not rebuild a wider cartesian frontier from per-indicator row pools.
    Raises:
        AssertionError: If explicit retained variants are ignored or batch bucketing drifts.
    Side Effects:
        None.
    """
    retained_variants = (
        _combo_proxy_base_variant(
            stage_a_index=3,
            alpha_window=10,
            beta_window=10,
            gamma_window=10,
        ),
        _combo_proxy_base_variant(
            stage_a_index=4,
            alpha_window=10,
            beta_window=10,
            gamma_window=20,
        ),
        _combo_proxy_base_variant(
            stage_a_index=7,
            alpha_window=20,
            beta_window=20,
            gamma_window=20,
        ),
    )
    grid_context = SimpleNamespace(retained_stage_a_variants=retained_variants)
    builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=_ComboProxyPriceLoader(),
        signal_matrix_loader=_combo_proxy_signal_loader(),
    )

    chunks = tuple(
        tuple(variant.stage_a_index for variant in chunk)
        for chunk in builder._iter_retained_stage_a_variant_chunks(
            row_plans=(),
            grid_context=cast(Any, grid_context),
            row_prefilter_frontier={},
            batch_size=3,
        )
    )

    assert chunks == ((3, 4), (7,))
    assert (
        builder._retained_stage_a_variants_total(
            row_plans=(),
            grid_context=cast(Any, grid_context),
            row_prefilter_frontier={},
        )
        == 3
    )


def test_stage_a_shortlist_builder_v2_no_risk_parity_bypasses_hybrid_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify no-risk parity bypasses reduced-plan survivors and exact-scores narrowed combos direct.

    Args:
        monkeypatch: Pytest fixture used to fail fast if combo proxy prefilter is called.
    Returns:
        None.
    Assumptions:
        D3 requires canonical parity Stage A to ignore `retained_stage_a_variants` inherited from
        hybrid reduced-plan semantics and enumerate notebook-shaped narrowed combos directly from
        retained row pools in deterministic order.
    Raises:
        AssertionError: If parity Stage A still depends on reduced-plan survivors or combo proxy
            chunk prefiltering.
    Side Effects:
        Monkeypatches one combo-proxy helper during one in-memory Stage A run.
    """

    def _raise_combo_proxy_call(*args: Any, **kwargs: Any) -> Any:
        """
        Fail fast if no-risk parity Stage A still calls combo proxy prefiltering.

        Args:
            *args: Ignored positional arguments from the patched method call.
            **kwargs: Ignored keyword arguments from the patched method call.
        Returns:
            Any: This helper never returns successfully.
        Assumptions:
            D3 parity path must exact-score every narrowed combo directly.
        Raises:
            AssertionError: Always, because combo proxy prefilter is forbidden for parity runs.
        Side Effects:
            None.
        """
        raise AssertionError("Stage A exact_no_risk_parity path must bypass combo proxy prefilter")

    monkeypatch.setattr(
        BacktestStageAShortlistBuilderV2,
        "_select_combo_proxy_retained_chunk_row_indexes",
        _raise_combo_proxy_call,
    )

    grid_context = _combo_proxy_grid_context()
    setattr(grid_context, "execution_profile", SimpleNamespace(mode="exact_no_risk_parity"))
    setattr(
        grid_context,
        "retained_stage_a_variants",
        (
            _combo_proxy_base_variant(
                stage_a_index=6,
                alpha_window=20,
                beta_window=15,
                gamma_window=1,
            ),
            _combo_proxy_base_variant(
                stage_a_index=7,
                alpha_window=20,
                beta_window=15,
                gamma_window=2,
            ),
        ),
    )

    shortlist = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=_ComboProxyPriceLoader(),
        signal_matrix_loader=_combo_proxy_signal_loader(),
    ).build_shortlist(
        grid_context=cast(Any, grid_context),
        artifact_context=cast(Any, _combo_proxy_artifact_context()),
        target_time_range=_combo_proxy_target_time_range(),
        shortlist_limit=2,
        batch_size=8,
        parallelism=numba_runtime_module.BacktestStageAParallelismConfigV1(
            stage_a_workers=1,
            numba_threads=1,
        ),
    )

    assert tuple(row.base_variant.stage_a_index for row in shortlist) == (0, 1)


def test_stage_a_shortlist_builder_v2_no_risk_parity_rejects_narrowed_combo_counter_drift() -> None:
    """
    Verify no-risk parity fails fast when runtime counters drift from live narrowed combos.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        D3 parity runs should expose narrowed combo cardinality as additive runtime evidence, and
        Stage A should reject stale/misaligned parity counters before scoring.
    Raises:
        AssertionError: If parity counter drift is not surfaced as deterministic validation.
    Side Effects:
        None.
    """
    grid_context = _combo_proxy_grid_context()
    setattr(grid_context, "execution_profile", SimpleNamespace(mode="exact_no_risk_parity"))
    setattr(grid_context, "no_risk_parity_counters", SimpleNamespace(narrowed_combo_total=7))

    with pytest.raises(
        ValueError,
        match="narrowed_combo_total counter drifted",
    ):
        BacktestStageAShortlistBuilderV2(
            price_arrays_loader=_ComboProxyPriceLoader(),
            signal_matrix_loader=_combo_proxy_signal_loader(),
        ).build_shortlist(
            grid_context=cast(Any, grid_context),
            artifact_context=cast(Any, _combo_proxy_artifact_context()),
            target_time_range=_combo_proxy_target_time_range(),
            shortlist_limit=2,
            batch_size=8,
            parallelism=numba_runtime_module.BacktestStageAParallelismConfigV1(
                stage_a_workers=1,
                numba_threads=1,
            ),
        )


def test_stage_a_shortlist_builder_v2_no_risk_parity_uses_bounded_pair_first_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify D4 parity scoring bypasses generic dense kernels and uses bounded pair blocks.

    Args:
        monkeypatch: pytest fixture used to record pair-block sizes and fail fast on generic paths.
    Returns:
        None.
    Assumptions:
        The canonical two-indicator `exact_no_risk_parity` path should exact-score through the new
        pair-first kernel route only, while broad dense aggregation and generic retained batching
        remain available for non-parity paths.
    Raises:
        AssertionError: If parity re-enters the generic dense kernels or stops using bounded pair
            blocks.
    Side Effects:
        Monkeypatches the pair-first block size and pair-batch builder during one in-memory Stage
        A run.
    """
    recorded_pair_block_sizes: list[int] = []
    recorded_runtime_input_block_sizes: list[int] = []
    original_pair_batch_builder = (
        stage_a_shortlist_builder_module._build_compact_trade_batch_for_normalized_signal_pairs_v2
    )
    original_load_chunk_runtime_inputs = (
        stage_a_shortlist_builder_module.BacktestStageAShortlistBuilderV2.load_chunk_runtime_inputs
    )

    def _raise_generic_dense_kernel(*args: Any, **kwargs: Any) -> Any:
        """
        Fail fast if the two-indicator no-risk parity path re-enters generic dense kernels.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.
        Returns:
            Any: This helper never returns.
        Assumptions:
            D4 should keep generic dense aggregation and compaction unavailable on this parity
            route while leaving them untouched for other paths.
        Raises:
            AssertionError: Always.
        Side Effects:
            None.
        """
        raise AssertionError("two-indicator exact_no_risk_parity must bypass generic dense kernels")

    def _record_pair_batch_builder(**kwargs: Any) -> Any:
        """
        Record one parity pair-block cardinality before delegating to the live pair builder.

        Args:
            **kwargs: Pair-first batch-builder keyword arguments including `left_signal_rows`.
        Returns:
            Any: Live pair-first compact-trade batch.
        Assumptions:
            Each call corresponds to one bounded pair block emitted by the D4 parity path.
        Raises:
            None.
        Side Effects:
            Appends one observed pair-block size to the in-memory log.
        """
        recorded_pair_block_sizes.append(int(kwargs["left_signal_rows_i8"].shape[0]))
        return original_pair_batch_builder(**kwargs)

    def _record_load_chunk_runtime_inputs(self: Any, **kwargs: Any) -> Any:
        """
        Record one exact-stage runtime-input block size before delegating to the live loader.

        Args:
            self: Stage A shortlist builder under test.
            **kwargs: Runtime-input loader keyword arguments including `chunk_variants`.
        Returns:
            Any: Live prepared chunk inputs for the requested exact-stage block.
        Assumptions:
            D4 should reload signal rows for parity exact work per bounded pair block instead of
            materializing runtime inputs for the whole retained raw chunk first.
        Raises:
            None.
        Side Effects:
            Appends one observed exact-stage chunk size to the in-memory log.
        """
        recorded_runtime_input_block_sizes.append(len(kwargs["chunk_variants"]))
        return original_load_chunk_runtime_inputs(self, **kwargs)

    monkeypatch.setattr(
        stage_a_shortlist_builder_module,
        "_PAIR_FIRST_NO_RISK_EXACT_BLOCK_SIZE_V2",
        2,
    )
    monkeypatch.setattr(
        stage_a_shortlist_builder_module,
        "aggregate_ordered_final_signal_rows_v2",
        _raise_generic_dense_kernel,
    )
    monkeypatch.setattr(
        stage_a_shortlist_builder_module,
        "build_compact_trade_batch_v2",
        _raise_generic_dense_kernel,
    )
    monkeypatch.setattr(
        stage_a_shortlist_builder_module,
        "_build_compact_trade_batch_for_normalized_signal_pairs_v2",
        _record_pair_batch_builder,
    )
    monkeypatch.setattr(
        stage_a_shortlist_builder_module.BacktestStageAShortlistBuilderV2,
        "load_chunk_runtime_inputs",
        _record_load_chunk_runtime_inputs,
    )

    grid_context = _pair_first_no_risk_grid_context()
    setattr(grid_context, "execution_profile", SimpleNamespace(mode="exact_no_risk_parity"))

    signal_loader = _pair_first_no_risk_signal_loader()

    shortlist = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=_ComboProxyPriceLoader(),
        signal_matrix_loader=signal_loader,
    ).build_shortlist(
        grid_context=cast(Any, grid_context),
        artifact_context=cast(Any, _combo_proxy_artifact_context()),
        target_time_range=_combo_proxy_target_time_range(),
        shortlist_limit=2,
        batch_size=8,
        parallelism=numba_runtime_module.BacktestStageAParallelismConfigV1(
            stage_a_workers=1,
            numba_threads=1,
        ),
    )

    assert recorded_pair_block_sizes == [2, 2]
    assert recorded_runtime_input_block_sizes == [2, 2]
    assert len(shortlist) == 2
    assert all(row.retained_exact_payload is not None for row in shortlist)
    assert all(
        row.retained_exact_payload is not None
        and row.retained_exact_payload.memory_shape_bucket == "compact_trade_arrays"
        for row in shortlist
    )


def test_stage_a_shortlist_builder_v2_streaming_exact_shortlist_is_batch_invariant() -> None:
    """
    Verify streaming exact scoring keeps the deterministic Stage A shortlist stable across batches.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The active Stage A path now exact-scores retained chunks immediately, but the final
        shortlist identity should remain deterministic across scan chunk sizes.
    Raises:
        AssertionError: If streaming exact scoring changes the final shortlist across batches.
    Side Effects:
        None.
    """
    small_batch_shortlist = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=_ComboProxyPriceLoader(),
        signal_matrix_loader=_combo_proxy_signal_loader(),
    ).build_shortlist(
        grid_context=cast(Any, _combo_proxy_grid_context()),
        artifact_context=cast(Any, _combo_proxy_artifact_context()),
        target_time_range=_combo_proxy_target_time_range(),
        shortlist_limit=2,
        batch_size=1,
    )
    large_batch_shortlist = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=_ComboProxyPriceLoader(),
        signal_matrix_loader=_combo_proxy_signal_loader(),
    ).build_shortlist(
        grid_context=cast(Any, _combo_proxy_grid_context()),
        artifact_context=cast(Any, _combo_proxy_artifact_context()),
        target_time_range=_combo_proxy_target_time_range(),
        shortlist_limit=2,
        batch_size=8,
    )

    assert tuple(row.base_variant.base_variant_key for row in small_batch_shortlist) == tuple(
        row.base_variant.base_variant_key for row in large_batch_shortlist
    )


def test_stage_a_shortlist_builder_v2_applies_stage_a_workers_to_live_kernel_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify live Stage A aggregation observes the configured in-process Numba thread budget.

    Args:
        monkeypatch: pytest fixture used to record live aggregation-thread observations.
    Returns:
        None.
    Assumptions:
        Parallel Stage A should stay single-process and drive kernel-backed aggregation under the
        configured `stage_a_workers` budget instead of silently falling back to serial Python.
    Raises:
        AssertionError: If the live aggregation path does not observe the expected thread count.
    Side Effects:
        Monkeypatches the ordered aggregation helper during one in-memory Stage A run.
    """
    observed_numba_threads: list[int] = []
    original_aggregate = stage_a_shortlist_builder_module.aggregate_ordered_final_signal_rows_v2

    def _recording_aggregate(**kwargs: Any) -> np.ndarray:
        """
        Record the live Numba thread budget before delegating to the real aggregation helper.

        Args:
            **kwargs: Ordered aggregation keyword arguments forwarded to the live helper.
        Returns:
            np.ndarray: Aggregated Stage A `final_signal` matrix from the live helper.
        Assumptions:
            The builder invokes ordered final-signal aggregation inside the Stage A Numba thread
            scope, so the observed thread count reflects `stage_a_workers` on the hot path.
        Raises:
            None.
        Side Effects:
            Appends one observed Numba thread count to the in-memory log.
        """
        observed_numba_threads.append(numba_runtime_module.current_backtest_numba_threads_v1())
        return original_aggregate(**kwargs)

    monkeypatch.setattr(
        stage_a_shortlist_builder_module,
        "aggregate_ordered_final_signal_rows_v2",
        _recording_aggregate,
    )

    shortlist = stage_a_shortlist_builder_module.BacktestStageAShortlistBuilderV2(
        price_arrays_loader=_ComboProxyPriceLoader(),
        signal_matrix_loader=_combo_proxy_signal_loader(),
    ).build_shortlist(
        grid_context=cast(Any, _combo_proxy_grid_context()),
        artifact_context=cast(Any, _combo_proxy_artifact_context()),
        target_time_range=_combo_proxy_target_time_range(),
        shortlist_limit=2,
        batch_size=6,
        parallelism=numba_runtime_module.BacktestStageAParallelismConfigV1(
            stage_a_workers=2,
            numba_threads=2,
        ),
    )

    assert tuple(row.base_variant.stage_a_index for row in shortlist) == (0, 1)
    assert observed_numba_threads
    assert max(observed_numba_threads) == 2


def test_stage_a_shortlist_builder_v2_streaming_exact_scoring_removes_deferred_replay_helpers() -> (
    None
):
    """
    Verify Stage A no longer exposes deferred retained replay helpers on the active builder.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Streaming exact scoring should replace the helper pair that rebuilt retained exact batches
        after scanning the full Stage A space.
    Raises:
        AssertionError: If the deferred replay helpers remain reachable on the builder.
    Side Effects:
        None.
    """
    assert not hasattr(
        BacktestStageAShortlistBuilderV2,
        "_load_retained_exact_final_signal_batch",
    )
    assert not hasattr(
        BacktestStageAShortlistBuilderV2,
        "_score_retained_exact_candidates_into_heap",
    )


def test_stage_a_shortlist_builder_v2_materializes_exact_payloads_only_for_shortlisted_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify bounded retained exact chunk work only materializes payload objects for shortlisted rows.

    Args:
        monkeypatch: pytest fixture used to record retained exact batch and payload calls.
    Returns:
        None.
    Assumptions:
        The combo proxy fixture retains four exact candidates inside one retained chunk out of
        eight Stage A variants, and only the final deterministic shortlist should receive
        internal exact payload objects.
    Raises:
        AssertionError: If retained exact batching sees the full breadth or materializes payloads
            for non-shortlisted rows.
    Side Effects:
        Monkeypatches the internal retained exact batch builder used by Stage A.
    """
    grid_context = _combo_proxy_grid_context()
    retained_batch_row_counts: list[int] = []
    retained_batch_signal_counts: list[int] = []
    retained_batch_max_trade_counts: list[int] = []
    materialized_payload_row_indexes: list[int] = []
    original_builder = stage_a_shortlist_builder_module.build_compact_trade_batch_v2

    def _recording_builder(**kwargs: Any) -> Any:
        """
        Record retained exact batch size, bounded width, and selective payload materialization.

        Args:
            **kwargs: Dense retained-batch builder keyword arguments including `final_signal`.
        Returns:
            Any: Original dense batch object wrapped with payload-materialization logging.
        Assumptions:
            The retained exact batch builder receives only one retained chunk after combo pruning.
        Raises:
            None.
        Side Effects:
            Appends retained batch shape data and later payload row indexes to in-memory logs.
        """
        retained_final_signal = np.asarray(kwargs["final_signal"])
        retained_batch_row_counts.append(int(retained_final_signal.shape[0]))
        retained_batch_signal_counts.append(int(retained_final_signal.shape[1]))
        wrapped_batch = original_builder(**kwargs)
        retained_batch_max_trade_counts.append(int(wrapped_batch.max_trade_count))

        class _RecordingBatch:
            """
            Minimal proxy recording which retained rows materialize internal exact payloads.
            """

            def __init__(self, *, wrapped: Any) -> None:
                """
                Initialize proxy wrapper around one dense retained exact batch object.

                Args:
                    wrapped: Dense retained exact batch object returned by the real builder.
                Returns:
                    None.
                Assumptions:
                    Stage A accesses batch arrays directly and calls `exact_payload_at(...)` only
                    for shortlisted rows.
                Raises:
                    None.
                Side Effects:
                    Stores one wrapped batch reference.
                """
                self._wrapped = wrapped

            def __getattr__(self, name: str) -> Any:
                """
                Delegate all non-overridden attribute access to the wrapped batch object.

                Args:
                    name: Attribute name requested by Stage A.
                Returns:
                    Any: Attribute value from the wrapped batch object.
                Assumptions:
                    Dense batch arrays and helper methods live on the wrapped object.
                Raises:
                    AttributeError: Propagated when the wrapped object lacks the attribute.
                Side Effects:
                    None.
                """
                return getattr(self._wrapped, name)

            def exact_payload_at(self, *, row_index: int) -> Any:
                """
                Record one shortlisted payload materialization before delegating.

                Args:
                    row_index: Batch-local retained candidate index being materialized.
                Returns:
                    Any: Wrapped exact payload for the same retained row.
                Assumptions:
                    Materialization order should follow the deterministic shortlist decisions.
                Raises:
                    None.
                Side Effects:
                    Appends the row index to the in-memory materialization log.
                """
                materialized_payload_row_indexes.append(row_index)
                return self._wrapped.exact_payload_at(row_index=row_index)

        return _RecordingBatch(wrapped=wrapped_batch)

    monkeypatch.setattr(
        stage_a_shortlist_builder_module,
        "build_compact_trade_batch_v2",
        _recording_builder,
    )

    shortlist = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=_ComboProxyPriceLoader(),
        signal_matrix_loader=_combo_proxy_signal_loader(),
    ).build_shortlist(
        grid_context=cast(Any, grid_context),
        artifact_context=cast(Any, _combo_proxy_artifact_context()),
        target_time_range=_combo_proxy_target_time_range(),
        shortlist_limit=2,
        batch_size=8,
    )

    assert retained_batch_row_counts == [4]
    assert sum(retained_batch_row_counts) == 4
    assert grid_context.stage_a_variants_total == 8
    assert sum(retained_batch_row_counts) < grid_context.stage_a_variants_total
    assert retained_batch_max_trade_counts == [1]
    assert retained_batch_max_trade_counts[0] < retained_batch_signal_counts[0]
    assert materialized_payload_row_indexes == [0, 1]
    assert tuple(row.base_variant.stage_a_index for row in shortlist) == (0, 1)


def _inactive_context(store: SyntheticArtifactStoreV2) -> ArtifactSlotPinnedRuntimeContextV2:
    """
    Resolve one deterministic pinned context for the synthetic inactive slot.

    Args:
        store: Synthetic artifact store fixture.
    Returns:
        ArtifactSlotPinnedRuntimeContextV2: Pinned inactive-slot runtime context.
    Assumptions:
        Builder tests exercise background-style explicit slot loading against the inactive slot.
    Raises:
        ValueError: If the synthetic slot metadata is inconsistent.
    Side Effects:
        Reads strict slot metadata from the synthetic store.
    """
    resolver = ArtifactSlotResolverV2(artifact_loader=store.loader)
    return resolver.resolve_pinned_context(
        store.coordinates,
        ArtifactPinnedIdentityV2(
            artifact_slot=store.inactive_slot,
            slot_generation=5,
            artifact_asof_date="2026-03-26",
            artifact_manifest_hash="b" * 64,
        ),
    )


def _record_streaming_exact_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, ...]]:
    """
    Record Stage A streaming exact chunk ordering emitted by combo proxy prefilter.

    Args:
        monkeypatch: pytest fixture used to wrap the builder method.
    Returns:
        list[tuple[int, ...]]: Stage A indexes exact-scored in each retained chunk merge.
    Assumptions:
        Streaming exact scoring should merge each retained chunk separately instead of stacking
        one deferred replay batch.
    Raises:
        None.
    Side Effects:
        Monkeypatches
        `BacktestStageAShortlistBuilderV2._merge_retained_exact_payload_chunk_into_heap`.
    """
    streaming_exact_chunks: list[tuple[int, ...]] = []
    original_method = BacktestStageAShortlistBuilderV2._merge_retained_exact_payload_chunk_into_heap

    def _recording_method(self: Any, **kwargs: Any) -> None:
        """
        Record one retained chunk before delegating to the exact scorer.

        Args:
            self: Builder instance under test.
            **kwargs: Exact-scoring keyword arguments including `chunk_variants`.
        Returns:
            None.
        Assumptions:
            Retained chunk order should stay deterministic and should surface multiple calls when
            Stage A avoids deferred replay.
        Raises:
            None.
        Side Effects:
            Appends exact-scored Stage A indexes to the in-memory log.
        """
        streaming_exact_chunks.append(
            tuple(variant.stage_a_index for variant in kwargs["chunk_variants"])
        )
        original_method(self, **kwargs)

    monkeypatch.setattr(
        BacktestStageAShortlistBuilderV2,
        "_merge_retained_exact_payload_chunk_into_heap",
        _recording_method,
    )
    return streaming_exact_chunks


def _attach_signal_features_access(*, grid_context: _FakeGridContext) -> None:
    """
    Attach additive warm-cache access metadata to the lightweight fake runtime-plan fixture.

    Args:
        grid_context: Minimal Stage A grid context fixture used by these tests.
    Returns:
        None.
    Assumptions:
        Tests use one `15m/ma.ema` signal target and keep feature access optional.
    Raises:
        None.
    Side Effects:
        Adds `signal_features_access` attribute to the mutable fake grid-context instance.
    """
    grid_context.signal_features_access = (
        BacktestSignalFeaturesAccessPlanV2(
            indicator_id="ma.ema",
            timeframe="15m",
            optional=True,
        ),
    )


def _build_shortlist_with_optional_features(
    *,
    store: SyntheticArtifactStoreV2,
    grid_context: _FakeGridContext,
    target_time_range: TimeRange,
) -> tuple[Any, ...]:
    """
    Build one deterministic exact shortlist with optional feature warm-cache access enabled.

    Args:
        store: Synthetic artifact store fixture.
        grid_context: Minimal Stage A grid context carrying additive feature-access metadata.
        target_time_range: Requested trading window.
    Returns:
        tuple[Any, ...]: Exact Stage A shortlist rows.
    Assumptions:
        Builder must keep exact semantics identical whether feature artifacts exist or not.
    Raises:
        ValueError: Propagated from runtime builder if one artifact-local contract drifts.
    Side Effects:
        Memory-maps strict synthetic artifact families via the builder loaders.
    """
    return BacktestStageAShortlistBuilderV2(
        price_arrays_loader=MmapPriceArraysLoaderV2(artifact_loader=store.loader),
        signal_matrix_loader=MmapSignalMatrixLoaderV2(artifact_loader=store.loader),
        signal_features_loader=MmapSignalFeaturesLoaderV2(artifact_loader=store.loader),
    ).build_shortlist(
        grid_context=cast(Any, grid_context),
        artifact_context=_inactive_context(store),
        target_time_range=target_time_range,
        shortlist_limit=2,
        batch_size=1,
    )


def _grid_context_for_windows(*, windows: tuple[int, ...]) -> _FakeGridContext:
    """
    Build a minimal deterministic grid context whose row order matches the synthetic signal store.

    Args:
        windows: Explicit window values defining Stage A compute row order.
    Returns:
        _FakeGridContext: Minimal grid context fixture for shortlist builder tests.
    Assumptions:
        Synthetic signal matrix row ordering follows the explicit `window` axis ordering.
    Raises:
        None.
    Side Effects:
        None.
    """
    base_variants = tuple(
        _base_variant(
            stage_a_index=index,
            window=window,
            base_variant_key=f"{index + 1}" * 64,
        )
        for index, window in enumerate(windows)
    )
    return _FakeGridContext(
        base_variants=base_variants,
        indicator_plans=(
            _FakeIndicatorPlan(
                indicator_id="ma.ema",
                axes=(_FakeAxis(name="window", values=windows),),
            ),
        ),
    )


def _combo_proxy_grid_context() -> _FakeGridContext:
    """
    Build a three-indicator Stage A grid context for combo proxy prefilter tests.

    Args:
        None.
    Returns:
        _FakeGridContext: Deterministic three-indicator cartesian Stage A fixture.
    Assumptions:
        Every indicator keeps both rows through row prefilter so combo proxy prefilter becomes the
        first narrowing layer that reduces the retained exact-candidate frontier.
    Raises:
        None.
    Side Effects:
        None.
    """
    indicator_rows = (
        ("alpha", (10, 20)),
        ("beta", (5, 15)),
        ("gamma", (1, 2)),
    )
    base_variants = tuple(
        _combo_proxy_base_variant(
            stage_a_index=stage_a_index,
            alpha_window=alpha_window,
            beta_window=beta_window,
            gamma_window=gamma_window,
        )
        for stage_a_index, (alpha_window, beta_window, gamma_window) in enumerate(
            product(*(rows for _, rows in indicator_rows))
        )
    )
    return _FakeGridContext(
        base_variants=base_variants,
        indicator_plans=tuple(
            _FakeIndicatorPlan(
                indicator_id=indicator_id,
                axes=(_FakeAxis(name="window", values=windows),),
            )
            for indicator_id, windows in indicator_rows
        ),
    )


def _combo_proxy_base_variant(
    *,
    stage_a_index: int,
    alpha_window: int,
    beta_window: int,
    gamma_window: int,
) -> BacktestStageABaseVariant:
    """
    Build one deterministic three-indicator Stage A base variant for combo proxy tests.

    Args:
        stage_a_index: Stage A flat index.
        alpha_window: `alpha.window` parameter value.
        beta_window: `beta.window` parameter value.
        gamma_window: `gamma.window` parameter value.
    Returns:
        BacktestStageABaseVariant: Minimal three-indicator Stage A base variant fixture.
    Assumptions:
        Indicator ordering must stay explicit so retained-frontier ordering remains reviewable.
    Raises:
        None.
    Side Effects:
        None.
    """
    return BacktestStageABaseVariant(
        stage_a_index=stage_a_index,
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="alpha",
                inputs={"source": "close"},
                params={"window": alpha_window},
            ),
            IndicatorVariantSelection(
                indicator_id="beta",
                inputs={"source": "close"},
                params={"window": beta_window},
            ),
            IndicatorVariantSelection(
                indicator_id="gamma",
                inputs={"source": "close"},
                params={"window": gamma_window},
            ),
        ),
        signal_params={},
        indicator_variant_key=f"{stage_a_index:x}" * 64,
        base_variant_key=f"{stage_a_index + 8:x}" * 64,
    )


def _combo_proxy_signal_loader() -> _InMemorySignalRowsLoader:
    """
    Build one deterministic in-memory signal loader for combo proxy prefilter tests.

    Args:
        None.
    Returns:
        _InMemorySignalRowsLoader: Loader carrying aligned three-indicator test matrices.
    Assumptions:
        Each indicator's second row weakens consensus after the first bar while still surviving
        row prefilter, making combo proxy prefilter the first narrowing step.
    Raises:
        None.
    Side Effects:
        None.
    """
    strong_then_hold = np.array((1, 1, 1), dtype=np.int8)
    weak_open_only = np.array((1, 0, 0), dtype=np.int8)
    return _InMemorySignalRowsLoader(
        matrices_by_indicator={
            "alpha": np.vstack((strong_then_hold, weak_open_only)),
            "beta": np.vstack((strong_then_hold, weak_open_only)),
            "gamma": np.vstack((strong_then_hold, weak_open_only)),
        }
    )


def _pair_first_no_risk_grid_context() -> _FakeGridContext:
    """
    Build a two-indicator Stage A grid context for the bounded pair-first parity path.

    Args:
        None.
    Returns:
        _FakeGridContext: Deterministic two-indicator cartesian Stage A fixture.
    Assumptions:
        D4 targets the canonical two-indicator no-risk parity class, so tests model that exact
        shape and let row prefilter retain the full 2x2 narrowed frontier.
    Raises:
        None.
    Side Effects:
        None.
    """
    indicator_rows = (
        ("ma.dema", (12, 24)),
        ("ma.hma", (16, 32)),
    )
    base_variants = tuple(
        _pair_first_no_risk_base_variant(
            stage_a_index=stage_a_index,
            dema_window=dema_window,
            hma_window=hma_window,
        )
        for stage_a_index, (dema_window, hma_window) in enumerate(
            product(*(rows for _, rows in indicator_rows))
        )
    )
    return _FakeGridContext(
        base_variants=base_variants,
        indicator_plans=tuple(
            _FakeIndicatorPlan(
                indicator_id=indicator_id,
                axes=(_FakeAxis(name="window", values=windows),),
            )
            for indicator_id, windows in indicator_rows
        ),
    )


def _pair_first_no_risk_base_variant(
    *,
    stage_a_index: int,
    dema_window: int,
    hma_window: int,
) -> BacktestStageABaseVariant:
    """
    Build one deterministic two-indicator Stage A base variant for pair-first parity tests.

    Args:
        stage_a_index: Stage A flat index.
        dema_window: `ma.dema.window` parameter value.
        hma_window: `ma.hma.window` parameter value.
    Returns:
        BacktestStageABaseVariant: Minimal two-indicator Stage A base variant fixture.
    Assumptions:
        Indicator ordering must stay explicit so the pair-first path preserves deterministic row
        pairing and shortlist tie semantics.
    Raises:
        None.
    Side Effects:
        None.
    """
    return BacktestStageABaseVariant(
        stage_a_index=stage_a_index,
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ma.dema",
                inputs={"source": "close"},
                params={"window": dema_window},
            ),
            IndicatorVariantSelection(
                indicator_id="ma.hma",
                inputs={"source": "close"},
                params={"window": hma_window},
            ),
        ),
        signal_params={},
        indicator_variant_key=f"{stage_a_index + 1:x}" * 64,
        base_variant_key=f"{stage_a_index + 9:x}" * 64,
    )


def _pair_first_no_risk_signal_loader() -> _InMemorySignalRowsLoader:
    """
    Build deterministic two-indicator signal rows for the bounded pair-first parity tests.

    Args:
        None.
    Returns:
        _InMemorySignalRowsLoader: In-memory signal loader aligned to the two-indicator fixture.
    Assumptions:
        Rising execution prices should make the fully confirming first pair outperform the other
        narrowed combos while still leaving four variants so D4 block splitting is observable.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _InMemorySignalRowsLoader(
        matrices_by_indicator={
            "ma.dema": np.array([[1, 1, -1], [1, 0, -1]], dtype=np.int8),
            "ma.hma": np.array([[1, 1, -1], [1, -1, -1]], dtype=np.int8),
        }
    )


def _combo_proxy_artifact_context() -> Any:
    """
    Build one minimal artifact context fixture for combo proxy prefilter tests.

    Args:
        None.
    Returns:
        Any: Namespace exposing only the coordinates fields required by the builder.
    Assumptions:
        Combo proxy tests use in-memory loaders and therefore need only market-id resolution.
    Raises:
        None.
    Side Effects:
        None.
    """
    return SimpleNamespace(
        coordinates=SimpleNamespace(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        )
    )


def _base_variant(
    *,
    stage_a_index: int,
    window: int,
    base_variant_key: str,
) -> BacktestStageABaseVariant:
    """
    Build one deterministic Stage A base variant fixture for artifact-backed shortlist tests.

    Args:
        stage_a_index: Stage A flat index.
        window: Indicator `window` parameter used for artifact row addressing.
        base_variant_key: Deterministic base variant key literal.
    Returns:
        BacktestStageABaseVariant: Minimal Stage A base variant fixture.
    Assumptions:
        Builder tests use one indicator and empty default-only signal params.
    Raises:
        None.
    Side Effects:
        None.
    """
    return BacktestStageABaseVariant(
        stage_a_index=stage_a_index,
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ma.ema",
                inputs={"source": "close"},
                params={"window": window},
            ),
        ),
        signal_params={},
        indicator_variant_key=f"{stage_a_index + 3}" * 64,
        base_variant_key=base_variant_key,
    )


def _synthetic_target_time_range() -> TimeRange:
    """
    Build the deterministic target range selecting both synthetic `15m` bars and last three `1m`.

    Args:
        None.
    Returns:
        TimeRange: Requested `[Start, End)` window compatible with synthetic artifact timestamps.
    Assumptions:
        Synthetic artifacts use millisecond close times `[2599, 4599]` for `15m` bars.
    Raises:
        None.
    Side Effects:
        None.
    """
    return TimeRange(
        start=UtcTimestamp(datetime(1970, 1, 1, 0, 0, 2, tzinfo=timezone.utc)),
        end=UtcTimestamp(datetime(1970, 1, 1, 0, 0, 5, tzinfo=timezone.utc)),
    )


def _combo_proxy_target_time_range() -> TimeRange:
    """
    Build the deterministic target range selecting all combo proxy test bars.

    Args:
        None.
    Returns:
        TimeRange: Requested `[Start, End)` window covering all synthetic combo-proxy bars.
    Assumptions:
        Combo proxy tests use request-timeframe close times `[2599, 4599, 6599]`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return TimeRange(
        start=UtcTimestamp(datetime(1970, 1, 1, 0, 0, 2, tzinfo=timezone.utc)),
        end=UtcTimestamp(datetime(1970, 1, 1, 0, 0, 7, tzinfo=timezone.utc)),
    )
