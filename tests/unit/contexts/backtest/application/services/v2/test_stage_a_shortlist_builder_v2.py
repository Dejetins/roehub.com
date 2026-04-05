from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping, cast

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    SyntheticArtifactStoreV2,
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.application.dto import BacktestRankingConfig
from trading.contexts.backtest.application.services import (
    ArtifactPinnedIdentityV2,
    ArtifactSlotPinnedRuntimeContextV2,
    ArtifactSlotResolverV2,
    BacktestSignalFeaturesAccessPlanV2,
    BacktestStageABaseVariant,
    BacktestStageAShortlistBuilderV2,
    MmapPriceArraysLoaderV2,
    MmapSignalFeaturesLoaderV2,
    MmapSignalMatrixLoaderV2,
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
        One selected variant should request only its own signal row from the mmap loader.
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
    assert recording_loader.calls == [("ma.ema", (0,))]


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
        artifact_context=cast(ArtifactSlotPinnedRuntimeContextV2, SimpleNamespace(
            coordinates=SimpleNamespace(exchange="binance", market_type="spot", symbol="BTCUSDT")
        )),
        target_time_range=_synthetic_target_time_range(),
        shortlist_limit=2,
        ranking=BacktestRankingConfig(primary_metric="total_return_pct"),
        batch_size=1,
    )

    assert tuple(row.base_variant.base_variant_key for row in shortlist) == ("a" * 64, "b" * 64)


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
