from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests.unit.contexts.backtest.application.services.v2 import (
    test_derived_artifact_materialization as derived,
)
from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
    FilesystemBacktestArtifactContextResolver,
)
from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestPreparePoolsConfig,
)
from trading.contexts.backtest.application.dto.input_recipe import BacktestInputRecipe
from trading.contexts.backtest.application.services.v2 import (
    ARTIFACT_ARRAY_MMAP_LOAD_SEGMENT,
    ARTIFACT_ARRAY_OPEN_SEGMENT,
    ARTIFACT_CONTEXT_RESOLVE_SEGMENT,
    ARTIFACT_MANIFEST_LOAD_SEGMENT,
    PREPARE_POOLS_CORE_STAGE_NAME,
    PREPARE_POOLS_TOTAL_STAGE_NAME,
    REQUEST_SLICE_PREPARE_SEGMENT,
    ROW_PREFILTER_SEGMENT,
    SEGMENT_BUILD_SEGMENT,
    SIGNAL_ROW_SELECTION_SEGMENT,
    TIME_RANGE_SLICE_SEGMENT,
    BacktestPreparePoolsRejected,
    BacktestPreparePoolsService,
    build_signal_segments,
    notebook_compatible_prepare_pools_core_s,
    time_range_slice,
)
from trading.contexts.backtest.application.services.v2.prepare_pools import (
    _timeframe_from_normalized,
)
from trading.contexts.backtest_artifacts.application.services.v2 import (
    artifact_manifest_validator as prepared_validator,
)


def test_prepare_pools_prepares_indicator_pool_from_normalized_request(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    service = _service(store=store, top_fraction=1.0)
    metadata = _artifact_metadata(store=store)

    first = service.execute(
        normalized_request=_normalized_request(),
        artifact_metadata=metadata,
    )
    second = service.execute(
        normalized_request=_normalized_request(),
        artifact_metadata=metadata,
    )
    pool = first.indicator_pools[0]

    assert first.timeframe == "15m"
    assert first.indicator_ids == ("ma.ema",)
    assert first.time_slice_start_15m == 0
    assert first.time_slice_stop_15m == 2
    assert first.trade_T_length == 2
    assert first.eval_T_length == 1
    assert first.signal_returns_15m.tolist() == pytest.approx([(1.2 / 1.1) - 1.0])
    assert first.execution_mapping.signal_entry_exec_idx_15m.tolist() == [2, 4]
    assert first.execution_mapping.t_exec_limit_1m == 4

    assert pool.row_ids.tolist() == [0, 1]
    assert pool.trade_T.flags.c_contiguous
    assert pool.trade_T.tolist() == [[-1, 0], [1, 0]]
    assert pool.eval_T.tolist() == [[-1], [1]]
    assert [item.as_mapping() for item in pool.metadata] == [
        {"indicator_id": "ma.ema", "row_id": 0, "source": "close", "window": 5},
        {"indicator_id": "ma.ema", "row_id": 1, "source": "close", "window": 6},
    ]
    assert pool.nonzero.tolist() == [1, 1]
    assert pool.change_count.tolist() == [1, 1]
    assert pool.segments.starts.tolist() == [[0, 1], [0, 1]]
    assert pool.segments.ends.tolist() == [[1, 2], [1, 2]]
    assert pool.segments.values.tolist() == [[-1, 0], [1, 0]]
    assert pool.segments.counts.tolist() == [2, 2]
    assert len(first.row_metadata_order_hash) == 64
    assert first.row_metadata_order_hash == second.row_metadata_order_hash
    assert first.timing.stage_name == PREPARE_POOLS_TOTAL_STAGE_NAME
    assert first.timing.prepare_pools_core_s == first.timing.subsegments[
        PREPARE_POOLS_CORE_STAGE_NAME
    ]
    assert first.timing.prepare_pools_total_s == first.timing.wall_time_s
    assert notebook_compatible_prepare_pools_core_s(first.timing) == first.timing.subsegments[
        PREPARE_POOLS_CORE_STAGE_NAME
    ]
    assert set(first.timing.subsegments) == {
        ARTIFACT_CONTEXT_RESOLVE_SEGMENT,
        ARTIFACT_ARRAY_OPEN_SEGMENT,
        REQUEST_SLICE_PREPARE_SEGMENT,
        ARTIFACT_MANIFEST_LOAD_SEGMENT,
        ARTIFACT_ARRAY_MMAP_LOAD_SEGMENT,
        TIME_RANGE_SLICE_SEGMENT,
        SIGNAL_ROW_SELECTION_SEGMENT,
        ROW_PREFILTER_SEGMENT,
        SEGMENT_BUILD_SEGMENT,
        PREPARE_POOLS_CORE_STAGE_NAME,
        PREPARE_POOLS_TOTAL_STAGE_NAME,
    }


@pytest.mark.parametrize(
    "timeframe",
    ("15m", "30m", "1h", "2h", "4h", "6h", "8h", "1d", "2d", "3d"),
)
def test_timeframe_normalization_accepts_supported_artifact_timeframes(timeframe: str) -> None:
    request = _normalized_request()
    request["timeframe"] = timeframe

    assert _timeframe_from_normalized(request) == timeframe


def test_prepare_pools_core_excludes_context_open_and_slice_overhead(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    spy_loader = _CountingArtifactArrayLoader(
        FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    )
    service = _service(store=store, top_fraction=1.0, loader=spy_loader)
    request = _normalized_request()
    coordinates = BacktestCoordinates(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )

    context = service.resolve_artifact_context(
        coordinates=coordinates,
        artifact_metadata=_artifact_metadata(store=store),
    )
    runtime_arrays = service.open_artifact_arrays(
        normalized_request=request,
        context=context,
    )
    request_slice = service.prepare_request_slice(
        normalized_request=request,
        runtime_arrays=runtime_arrays,
    )
    counts_before_core = dict(spy_loader.counts)

    result = service.prepare_pools_core(
        normalized_request=request,
        runtime_arrays=runtime_arrays,
        request_slice=request_slice,
    )

    assert spy_loader.counts == counts_before_core
    assert result.timing.stage_name == PREPARE_POOLS_CORE_STAGE_NAME
    assert result.timing.prepare_pools_core_s == result.timing.wall_time_s
    assert ARTIFACT_CONTEXT_RESOLVE_SEGMENT not in result.timing.subsegments
    assert ARTIFACT_ARRAY_OPEN_SEGMENT not in result.timing.subsegments
    assert REQUEST_SLICE_PREPARE_SEGMENT not in result.timing.subsegments
    assert ARTIFACT_MANIFEST_LOAD_SEGMENT not in result.timing.subsegments
    assert ARTIFACT_ARRAY_MMAP_LOAD_SEGMENT not in result.timing.subsegments
    assert TIME_RANGE_SLICE_SEGMENT not in result.timing.subsegments
    assert set(result.timing.subsegments) == {
        SIGNAL_ROW_SELECTION_SEGMENT,
        ROW_PREFILTER_SEGMENT,
        SEGMENT_BUILD_SEGMENT,
        PREPARE_POOLS_CORE_STAGE_NAME,
    }
    assert result.indicator_pools[0].trade_T.tolist() == [[-1, 0], [1, 0]]


def test_prepare_pools_row_prefilter_keeps_top_adjusted_row(tmp_path: Path) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    service = _service(store=store, top_fraction=0.5)

    result = service.execute(
        normalized_request=_normalized_request(fee_rate=0.0),
        artifact_metadata=_artifact_metadata(store=store),
    )
    pool = result.indicator_pools[0]

    assert pool.row_ids.tolist() == [1]
    assert pool.proxy.tolist() == pytest.approx([(1.2 / 1.1) - 1.0])
    assert pool.row_score.tolist() == pytest.approx([(1.2 / 1.1) - 1.0])
    assert [item.row_id for item in pool.metadata] == [1]


def test_prepare_pools_core_retains_required_variant_rows_for_lazy_detail(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    service = _service(store=store, top_fraction=0.5)
    request = _normalized_request(fee_rate=0.0)
    context = service.resolve_artifact_context(
        coordinates=BacktestCoordinates(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        ),
        artifact_metadata=_artifact_metadata(store=store),
    )
    runtime_arrays = service.open_artifact_arrays(
        normalized_request=request,
        context=context,
    )
    request_slice = service.prepare_request_slice(
        normalized_request=request,
        runtime_arrays=runtime_arrays,
    )

    result = service.prepare_pools_core(
        normalized_request=request,
        runtime_arrays=runtime_arrays,
        request_slice=request_slice,
        required_row_ids_by_indicator={"ma.ema": (0,)},
    )

    pool = result.indicator_pools[0]
    assert pool.row_ids.tolist() == [0, 1]
    assert [item.row_id for item in pool.metadata] == [0, 1]


def test_prepare_pools_rejects_time_range_outside_artifact_coverage(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    request = _normalized_request()
    request["time_range"] = {
        "start": "1970-01-01T00:00:10Z",
        "end": "1970-01-01T00:00:11Z",
    }

    with pytest.raises(BacktestPreparePoolsRejected):
        _service(store=store, top_fraction=1.0).execute(
            normalized_request=request,
            artifact_metadata=_artifact_metadata(store=store),
        )


def test_prepare_pools_rejects_mapping_close_index_outside_1m_coverage(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    mapping_paths = store.builder.mapping_paths(store.coordinates, store.active_slot, "15m")
    with mapping_paths.bar_close_1m_idx.open("wb") as file_handle:
        np.save(file_handle, np.asarray([1, 99], dtype=np.uint32), allow_pickle=False)

    with pytest.raises(BacktestPreparePoolsRejected):
        _service(store=store, top_fraction=1.0).execute(
            normalized_request=_normalized_request(),
            artifact_metadata=_artifact_metadata(store=store),
        )


def test_time_range_slice_uses_half_open_15m_open_time() -> None:
    result = time_range_slice(
        open_time_15m=np.asarray([1000, 3000, 5000], dtype=np.int64),
        close_time_15m=np.asarray([2999, 4999, 6999], dtype=np.int64),
        time_range={
            "start": "1970-01-01T00:00:01Z",
            "end": "1970-01-01T00:00:05Z",
        },
    )

    assert result == slice(0, 2)


def test_build_signal_segments_compresses_change_points() -> None:
    segments = build_signal_segments(np.asarray([[1, 1, 0, -1]], dtype=np.int8))

    assert segments.starts.tolist() == [[0, 2, 3]]
    assert segments.ends.tolist() == [[2, 3, 4]]
    assert segments.values.tolist() == [[1, 0, -1]]
    assert segments.counts.tolist() == [3]
    assert segments.change_count.tolist() == [2]


def _service(
    *,
    store: Any,
    top_fraction: float,
    loader: Any | None = None,
) -> BacktestPreparePoolsService:
    return BacktestPreparePoolsService(
        artifact_array_loader=loader
        or FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader),
        defaults_provider=YamlBacktestGridDefaultsProvider.from_yaml(
            config_path="configs/prod/indicators.yaml",
        ),
        config=BacktestPreparePoolsConfig(row_prefilter_top_fraction=top_fraction),
    )


def _artifact_metadata(*, store: Any) -> BacktestArtifactMetadata:
    resolver = FilesystemBacktestArtifactContextResolver(artifact_loader=store.loader)
    return resolver.resolve_context(
        coordinates=BacktestCoordinates(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        )
    )


def _normalized_request(*, fee_rate: float = 0.00075) -> dict[str, Any]:
    return {
        "coordinates": {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "BTCUSDT",
        },
        "timeframe": "15m",
        "time_range": {
            "start": "1970-01-01T00:00:01Z",
            "end": "1970-01-01T00:00:04Z",
        },
        "indicators": [
            {
                "indicator_id": "ma.ema",
                "sources": ["close"],
                "window": {"start": 5, "stop": 6, "step": 1},
            }
        ],
        "risk": {"mode": "none"},
        "execution": {
            "direction_mode": "long_short_reversal",
            "fee_rate": fee_rate,
            "slippage_rate": 0.0001,
            "initial_cash_quote": 10000.0,
            "sizing": {"mode": "fixed_equity_pct", "equity_pct": 10.0},
            "profit_lock": {"enabled": False},
            "close_on_end": True,
        },
        "ranking": {
            "primary_metric": "total_return_pct",
            "direction": "desc",
        },
        "top_n": 100,
    }


class _CountingArtifactArrayLoader:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.counts = {
            "resolve_context": 0,
            "load_price_arrays": 0,
            "load_mapping_arrays": 0,
            "load_signal_matrix": 0,
            "load_signal_rows": 0,
        }

    def resolve_context(self, **kwargs: Any) -> Any:
        self.counts["resolve_context"] += 1
        return self._inner.resolve_context(**kwargs)

    def load_price_arrays(self, **kwargs: Any) -> Any:
        self.counts["load_price_arrays"] += 1
        return self._inner.load_price_arrays(**kwargs)

    def load_mapping_arrays(self, **kwargs: Any) -> Any:
        self.counts["load_mapping_arrays"] += 1
        return self._inner.load_mapping_arrays(**kwargs)

    def load_signal_matrix(self, **kwargs: Any) -> Any:
        self.counts["load_signal_matrix"] += 1
        return self._inner.load_signal_matrix(**kwargs)

    def load_signal_rows(self, **kwargs: Any) -> Any:
        self.counts["load_signal_rows"] += 1
        return self._inner.load_signal_rows(**kwargs)


# Reuse production Numba materialization, never modifying its module-scoped source.
built = derived.built


@pytest.fixture
def prepared_source(built, tmp_path):
    fixture, runner, snapshot, export_request = built
    root = tmp_path / "published"
    shutil.copytree(fixture.builder.root, root)
    path_builder = replace(fixture.builder, root=root)
    loader = replace(fixture.loader, path_resolver=path_builder)
    fixture = replace(fixture, builder=path_builder, loader=loader)
    runner = replace(runner, artifact_loader=loader, canonical_candle_reader=None)
    return fixture, runner, snapshot, export_request


def _input_context(fixture, slot):
    path = fixture.loader.resolve_slot_manifest_path(fixture.coordinates, slot)
    manifest = fixture.loader.load_slot_manifest(fixture.coordinates, slot)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=fixture.loader)
    metadata = BacktestArtifactMetadata(
        artifact_slot=slot, artifact_slot_generation=manifest.slot_generation,
        artifact_manifest_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        artifact_asof_date=manifest.asof_date, hit_times_manifest_hash=None,
        published_at_utc="2026-03-25T02:00:00Z",
    )
    context = loader.resolve_context(
        coordinates=BacktestCoordinates(**asdict(fixture.coordinates)),
        artifact_metadata=metadata,
    )
    return loader, context


def _input_recipe(prepared_source, context, *, risk=True):
    fixture, runner, original, _ = prepared_source
    # S02's snapshot helper only records prices. Use the loader's exact source domains,
    # including mappings, for S03 attestation and subsequent prepared-context binding.
    snapshot = derived.snapshot_for(fixture, original.slot)
    refs = tuple(ref for ref in context.references
                 if ref.domain.role.startswith(("prices.", "mappings.")))
    snapshot = replace(snapshot, source_file_identities=refs,
                       consumed_domains=tuple(ref.domain for ref in refs))
    request = derived.request_for(
        (fixture, runner, snapshot, None), risk=risk, selected=(1, 3, 5, 7),
    )
    rows = tuple(sorted(request.rows, key=lambda row: (row.indicator_id, row.row_id)))
    domain = next(ref.domain for ref in refs if ref.domain.role == "prices.15m.ohlcv")
    return BacktestInputRecipe(
        snapshot=snapshot, requested_start_utc=domain.origin_utc,
        requested_end_utc=domain.end_utc, rows=rows, rule_version=request.rule_version,
        defaults_sha256=runner.derivative_defaults_sha256(rows),
        compute_version=request.compute_version, precision_version=request.precision_version,
        funding_policy="not_applicable", funding_fingerprint=None,
        tp_levels_pct=request.tp_levels_pct, sl_levels_pct=request.sl_levels_pct,
    )


def _prepare_inputs(prepared_source, context, recipe, output, *, runner=None):
    fixture, real_runner, _, _ = prepared_source
    service = BacktestPreparePoolsService(
        artifact_array_loader=FilesystemBacktestArtifactArrayLoader(artifact_loader=fixture.loader),
        defaults_provider=real_runner.defaults_provider,
        config=BacktestPreparePoolsConfig(row_prefilter_top_fraction=1.0),
    )
    result = service.prepare_artifact_inputs(
        recipe=recipe, context=context, builder=runner or real_runner,
        validator=prepared_validator.BacktestArtifactManifestValidatorV2(artifact_loader=fixture.loader),
        output_directory=output, organization_id="00000000-0000-0000-0000-000000000001",
        job_id="00000000-0000-0000-0000-000000000002",
        owner_token="s03-owner", attempt=1, output_root_id="s03-attempt",
        max_generated_bytes=10_000_000, max_compute_bytes=10_000_000,
    )
    assert (output / "prepared-inputs.yaml").is_file()
    loader = service.artifact_array_loader
    bound = loader.with_prepared_inputs(
        context=context, prepared=result, output_directory=output,
        acknowledged_prepared_sha256=result.content_sha256,
    )
    return service, bound, result


def _prepared_core(service, context, recipe):
    request = _normalized_request()
    request["coordinates"] = asdict(recipe.snapshot.coordinates)
    request["time_range"] = {
        "start": recipe.requested_start_utc, "end": recipe.requested_end_utc,
    }
    request["indicators"] = [
        {"indicator_id": name, "sources": ["close", "open"],
         "window": {"start": 21, "stop": 100, "step": 79}}
        for name in ("ma.ema", "ma.sma", "ma.wma")
    ]
    arrays = service.open_artifact_arrays(normalized_request=request, context=context)
    sliced = service.prepare_request_slice(normalized_request=request, runtime_arrays=arrays)
    return service.prepare_pools_core(
        normalized_request=request, runtime_arrays=arrays, request_slice=sliced,
    )


def _assert_numerical_result_equal(actual, expected):
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected):
        for field in fields(expected):
            if field.name != "timing":
                _assert_numerical_result_equal(getattr(actual, field.name),
                                               getattr(expected, field.name))
    elif isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_numerical_result_equal(left, right)
    else:
        assert actual == expected


class _RecordingDerivedBuilder:
    def __init__(self, inner):
        self.inner = inner
        self.requests = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def materialize_derived(self, request, *, output_directory):
        self.requests.append(request)
        return self.inner.materialize_derived(request, output_directory=output_directory)


def _rewrite_published_inventory(prepared_source, mode):
    """Edit only the test-owned slot, retaining real production numerical rows."""
    fixture, runner, snapshot, _ = prepared_source
    path = fixture.loader.resolve_slot_manifest_path(fixture.coordinates, snapshot.slot)
    root = path.parent
    payload = yaml.safe_load(path.read_text())
    payload["schema_version"] = 2
    catalog = payload["signals"]
    if mode == "generated":
        payload.pop("hit_times")
        payload["signals"] = {
            "supported_timeframes": [], "supported_indicator_ids": [], "manifests": [],
        }
        for directory in ("signals", "signal_features", "hit_times"):
            shutil.rmtree(root / directory)
    else:
        catalog["manifests"] = [entry for entry in catalog["manifests"]
                                if entry["indicator_id"] != "ma.wma"]
        catalog["supported_indicator_ids"].remove("ma.wma")
        for directory in ("signals", "signal_features"):
            shutil.rmtree(root / directory / "15m" / "ma.wma")
        entry = next(item for item in catalog["manifests"] if item["indicator_id"] == "ma.sma")
        signal_path = root / entry["manifest_path"]
        signal = yaml.safe_load(signal_path.read_text())
        _compact_published_array(root, signal["signals"], [0, 1])
        signal["rows_count"] = 2
        grid = runner.indicator_grid_builder.materialize_indicator(
            grid=runner.defaults_provider.compute_defaults(indicator_id="ma.sma"),
        )
        canonical = derived.runner_module._build_signal_variant_rows_v2(
            coordinates=fixture.coordinates, timeframe="15m", materialized_grid=grid,
            row_ids=(0, 1),
        )
        signal["grid"]["variant_keys_sha256"] = (
            derived.runner_module._variant_keys_sha256_v2(signal_rows=canonical)
        )
        feature_ref = signal["signal_features"]
        feature_path = root / feature_ref["manifest_path"]
        feature = yaml.safe_load(feature_path.read_text())
        _compact_published_array(root, feature["features"], [0, 1])
        feature["rows_count"] = 2
        feature_path.write_text(yaml.safe_dump(feature))
        feature_ref["manifest_sha256"] = hashlib.sha256(feature_path.read_bytes()).hexdigest()
        signal_path.write_text(yaml.safe_dump(signal))
        entry["manifest_sha256"] = hashlib.sha256(signal_path.read_bytes()).hexdigest()
        hit_path = root / payload["hit_times"]["manifest_path"]
        hit = yaml.safe_load(hit_path.read_text())
        for family, indices in (("tp", [0]), ("sl", [1])):
            _compact_published_array(root, hit[f"{family}_values"], indices)
            for direction in ("long", "short"):
                _compact_published_array(root, hit["tables"][f"{direction}_{family}"], indices)
        hit_path.write_text(yaml.safe_dump(hit))
        payload["hit_times"]["manifest_sha256"] = hashlib.sha256(hit_path.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(payload))


def _compact_published_array(root, metadata, indices):
    path = root / metadata["path"]
    values = np.load(path, allow_pickle=False)[indices].copy()
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    metadata["shape"] = list(values.shape)
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mode", ["native", "generated", "mixed"])
def test_prepared_inputs_real_ngm_sparse_core_and_risk_parity(
    prepared_source, tmp_path, mode,
):
    fixture, runner, snapshot, _ = prepared_source
    loader, native = _input_context(fixture, snapshot.slot)
    recipe = _input_recipe(prepared_source, native)
    service = BacktestPreparePoolsService(
        artifact_array_loader=loader, defaults_provider=runner.defaults_provider,
        config=BacktestPreparePoolsConfig(row_prefilter_top_fraction=1.0),
    )
    expected = _prepared_core(service, native, recipe)
    # Independent IDs from source-major [close, open] x windows [20,21,50,100].
    for pool in expected.indicator_pools:
        assert pool.row_ids.tolist() == [1, 3, 5, 7]
        assert [(item.source, item.window) for item in pool.metadata] == [
            ("close", 21), ("close", 100), ("open", 21), ("open", 100),
        ]
    native_grid = loader.load_hit_times_grid_arrays(context=native)
    native_tables = loader.load_hit_times_table_arrays(context=native)
    expected_tables = {name: getattr(native_tables, name).copy()
                       for name in ("long_tp", "long_sl", "short_tp", "short_sl")}
    expected_levels = {side: getattr(native_grid, f"{side}_values").copy()
                       for side in ("tp", "sl")}
    root = native.source.slot_root_path
    candle_bytes = {path.relative_to(root): path.read_bytes()
                    for path in (root / "prices").rglob("*.npy")}
    if mode != "native":
        _rewrite_published_inventory(prepared_source, mode)
    loader, context = _input_context(fixture, snapshot.slot)
    recipe = _input_recipe(prepared_source, context)
    before = {path.relative_to(root): path.read_bytes()
              for path in root.rglob("*") if path.is_file()}
    recording = _RecordingDerivedBuilder(runner)
    service, bound, prepared = _prepare_inputs(
        prepared_source, context, recipe, tmp_path / "result", runner=recording,
    )
    actual = _prepared_core(service, bound, recipe)
    _assert_numerical_result_equal(actual, expected)
    assert actual.row_metadata_order_hash == expected.row_metadata_order_hash
    if mode == "native":
        assert recording.requests == []
        assert set(prepared.provenance) == {"reused"}
    else:
        assert len(recording.requests) == 1
        missing = recording.requests[0]
        expected_rows = [(name, row) for name in ("ma.ema", "ma.sma", "ma.wma")
                         for row in (1, 3, 5, 7)
                         if mode != "mixed" or name == "ma.wma"
                         or (name == "ma.sma" and row != 1)]
        assert [(row.indicator_id, row.row_id) for row in missing.rows] == expected_rows
        assert missing.tp_levels_pct == ((2.0,) if mode == "mixed" else (0.5, 2.0))
        assert missing.sl_levels_pct == (() if mode == "mixed" else (1.0,))
        assert set(prepared.provenance) == (
            {"generated", "reused"} if mode == "mixed" else {"generated"}
        )
    actual_grid = loader.load_hit_times_grid_arrays(context=bound)
    actual_tables = loader.load_hit_times_table_arrays(context=bound)
    for family in ("tp", "sl"):
        original_levels = expected_levels[family]
        levels = getattr(actual_grid, f"{family}_values")
        for direction in ("long", "short"):
            name = f"{direction}_{family}"
            indexes = [int(np.flatnonzero(np.isclose(original_levels, level))[0])
                       for level in levels]
            np.testing.assert_array_equal(getattr(actual_tables, name),
                                          expected_tables[name][indexes])
    assert actual_grid.sentinel_index == native_grid.sentinel_index
    assert before == {path: (root / path).read_bytes() for path in before}
    assert candle_bytes == {path: (root / path).read_bytes() for path in candle_bytes}


@pytest.mark.parametrize("schema", [1, 2])
def test_prepared_inputs_native_schema1_and_candles_only_schema2_no_risk(
    prepared_source, tmp_path, monkeypatch, schema,
):
    fixture, runner, snapshot, _ = prepared_source
    path = fixture.loader.resolve_slot_manifest_path(fixture.coordinates, snapshot.slot)
    if schema == 2:
        payload = yaml.safe_load(path.read_text())
        payload["schema_version"] = 2
        payload.pop("hit_times")
        payload["signals"] = {
            "supported_timeframes": [], "supported_indicator_ids": [], "manifests": [],
        }
        path.write_text(yaml.safe_dump(payload))
        for directory in ("signals", "signal_features", "hit_times"):
            shutil.rmtree(path.parent / directory)
    loader, context = _input_context(fixture, snapshot.slot)
    recipe = _input_recipe(prepared_source, context, risk=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("no-risk preparation must not compute hit-times")

    from trading.contexts.backtest_artifacts.application.services.v2 import (
        artifact_precompute_runner as runner_module,
    )
    monkeypatch.setattr(runner_module, "materialize_hit_times_from_ohlcv_v2", forbidden)
    recording = _RecordingDerivedBuilder(runner)
    service, bound, prepared = _prepare_inputs(
        prepared_source, context, recipe, tmp_path / "result", runner=recording,
    )
    assert prepared.snapshot.source_schema == schema
    assert not any(ref.domain.role.startswith("hit_times.") for ref in prepared.derivatives)
    assert len(recording.requests) == (0 if schema == 1 else 1)
    result = _prepared_core(service, bound, recipe)
    assert all(pool.row_ids.tolist() == [1, 3, 5, 7] for pool in result.indicator_pools)
    assert not (tmp_path / "result" / "hit_times").exists()


@pytest.mark.parametrize(
    "damage", ["checksum", "dtype", "shape", "missing", "source_missing", "symlink",
               "source_pin", "version"],
)
def test_prepared_inputs_declared_corruption_never_falls_back(
    prepared_source, tmp_path, damage,
):
    fixture, runner, snapshot, _ = prepared_source
    _, context = _input_context(fixture, snapshot.slot)
    recipe = _input_recipe(prepared_source, context)
    ref = next(ref for ref in context.references if ref.domain.role == "signals.ma.ema")
    path = context.trusted_roots[ref.root_id] / ref.relative_path
    if damage == "checksum":
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\x7f")
    elif damage in ("dtype", "shape"):
        values = np.load(path, allow_pickle=False)
        values = values.astype(np.int16) if damage == "dtype" else values[:, :-1]
        with path.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        # Keep the declared dtype/shape, but refresh every checksum: this must fail
        # actual array validation, rather than merely the outer file hash check.
        root = context.source.slot_root_path
        manifest_path = root / "manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        entry = next(item for item in manifest["signals"]["manifests"]
                     if item["indicator_id"] == "ma.ema")
        signal_path = root / entry["manifest_path"]
        signal = yaml.safe_load(signal_path.read_text())
        signal["signals"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        signal_path.write_text(yaml.safe_dump(signal))
        entry["manifest_sha256"] = hashlib.sha256(signal_path.read_bytes()).hexdigest()
        manifest_path.write_text(yaml.safe_dump(manifest))
        _, context = _input_context(fixture, snapshot.slot)
        recipe = _input_recipe(prepared_source, context)
    elif damage == "source_missing":
        source_ref = next(ref for ref in context.references
                          if ref.domain.role == "prices.1m.ohlcv")
        (context.source.slot_root_path / source_ref.relative_path).unlink()
    elif damage == "missing":
        path.unlink()
    elif damage == "symlink":
        outside = tmp_path / "outside.npy"
        shutil.copyfile(path, outside)
        path.unlink()
        path.symlink_to(outside)
    elif damage == "source_pin":
        recipe = replace(recipe, snapshot=replace(recipe.snapshot, manifest_sha256="0" * 64))
    else:
        recipe = replace(recipe, compute_version="unknown/v9")
    recording = _RecordingDerivedBuilder(runner)
    output = tmp_path / "result"
    with pytest.raises((ValueError, FileNotFoundError, BacktestPreparePoolsRejected)):
        _prepare_inputs(prepared_source, context, recipe, output, runner=recording)
    assert recording.requests == []
    assert not (output / "prepared-inputs.yaml").exists()


@pytest.mark.parametrize("family", ["tp", "sl"])
def test_prepared_inputs_generates_only_missing_risk_family(
    prepared_source, tmp_path, family,
):
    fixture, runner, snapshot, _ = prepared_source
    loader, native = _input_context(fixture, snapshot.slot)
    original = loader.load_hit_times_table_arrays(context=native)
    expected = {name: getattr(original, name).copy()
                for name in ("long_tp", "long_sl", "short_tp", "short_sl")}
    root = native.source.slot_root_path
    root_path = root / "manifest.yaml"
    payload = yaml.safe_load(root_path.read_text())
    hit_path = root / payload["hit_times"]["manifest_path"]
    hit = yaml.safe_load(hit_path.read_text())
    # TP retains .5%, omits requested 2%; SL retains .5%, omits requested 1%.
    _compact_published_array(root, hit[f"{family}_values"], [0])
    for direction in ("long", "short"):
        _compact_published_array(root, hit["tables"][f"{direction}_{family}"], [0])
    hit_path.write_text(yaml.safe_dump(hit))
    payload["hit_times"]["manifest_sha256"] = hashlib.sha256(hit_path.read_bytes()).hexdigest()
    root_path.write_text(yaml.safe_dump(payload))
    loader, context = _input_context(fixture, snapshot.slot)
    recipe = _input_recipe(prepared_source, context)
    recording = _RecordingDerivedBuilder(runner)
    _, bound, _ = _prepare_inputs(
        prepared_source, context, recipe, tmp_path / "result", runner=recording,
    )
    assert len(recording.requests) == 1
    missing = recording.requests[0]
    assert missing.rows == ()
    assert missing.tp_levels_pct == ((2.0,) if family == "tp" else ())
    assert missing.sl_levels_pct == ((1.0,) if family == "sl" else ())
    tables = loader.load_hit_times_table_arrays(context=bound)
    grid = loader.load_hit_times_grid_arrays(context=bound)
    for side, source_levels in (("tp", [0.005, 0.01, 0.02]), ("sl", [0.005, 0.01])):
        indices = [int(np.flatnonzero(np.isclose(source_levels, level))[0])
                   for level in getattr(grid, f"{side}_values")]
        for direction in ("long", "short"):
            name = f"{direction}_{side}"
            np.testing.assert_array_equal(getattr(tables, name), expected[name][indices])


def test_prepared_inputs_binding_rejects_unacknowledged_manifest(prepared_source, tmp_path):
    fixture, _, snapshot, _ = prepared_source
    loader, context = _input_context(fixture, snapshot.slot)
    recipe = _input_recipe(prepared_source, context, risk=False)
    _, _, prepared = _prepare_inputs(prepared_source, context, recipe, tmp_path / "result")
    with pytest.raises(ValueError, match="acknowledged"):
        loader.with_prepared_inputs(
            context=context, prepared=prepared, output_directory=tmp_path / "result",
            acknowledged_prepared_sha256="0" * 64,
        )


def test_prepare_pools_rejects_out_of_range_mapping_with_valid_checksums(tmp_path: Path):
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    metadata = _artifact_metadata(store=store)
    root_manifest = store.loader.resolve_slot_manifest_path(store.coordinates, store.active_slot)
    path = store.builder.mapping_paths(store.coordinates, store.active_slot, "15m").bar_close_1m_idx
    with path.open("wb") as stream:
        np.save(stream, np.asarray([1, 99], dtype=np.uint32), allow_pickle=False)
    payload = yaml.safe_load(root_manifest.read_text())
    mapping = next(item for item in payload["mappings"] if item["timeframe"] == "15m")
    mapping["bar_close_1m_idx"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    root_manifest.write_text(yaml.safe_dump(payload))
    metadata = replace(
        metadata, artifact_manifest_hash=hashlib.sha256(root_manifest.read_bytes()).hexdigest(),
    )
    with pytest.raises(BacktestPreparePoolsRejected, match="close index exceeds"):
        _service(store=store, top_fraction=1.0).execute(
            normalized_request=_normalized_request(), artifact_metadata=metadata,
        )


@pytest.mark.parametrize("mutation", ["organization", "job", "owner", "attempt", "files", "proof"])
def test_prepared_inputs_acknowledgement_binds_complete_manifest(
    prepared_source, tmp_path, mutation,
):
    fixture, _, snapshot, _ = prepared_source
    loader, context = _input_context(fixture, snapshot.slot)
    recipe = _input_recipe(prepared_source, context, risk=False)
    output = tmp_path / "result"
    _, _, prepared = _prepare_inputs(prepared_source, context, recipe, output)
    if mutation in ("organization", "job"):
        changed = replace(prepared, **{
            f"{mutation}_id": "00000000-0000-0000-0000-000000000099",
        })
    elif mutation == "owner":
        changed = replace(prepared, owner_token="different-owner")
    elif mutation == "attempt":
        changed = replace(prepared, attempt=prepared.attempt + 1)
    elif mutation == "files":
        first = replace(prepared.derivatives[0], sha256="0" * 64)
        changed = replace(prepared, derivatives=(first, *prepared.derivatives[1:]))
    else:
        proof = prepared.snapshot.prefix_proof
        digests = ("0" * 64, *proof.role_digests[1:])
        changed_proof = replace(
            proof, role_digests=digests,
            source_domain_sha256=proof.combined_digest(proof.domains, digests),
        )
        changed = replace(prepared, snapshot=replace(prepared.snapshot, prefix_proof=changed_proof))
    assert changed.recipe_sha256 == prepared.recipe_sha256
    assert changed.content_sha256 != prepared.content_sha256
    # Even replacing the on-disk record cannot reuse the original acknowledgement.
    (output / "prepared-inputs.yaml").write_text(yaml.safe_dump(changed.as_mapping()))
    with pytest.raises(ValueError, match="acknowledged"):
        loader.with_prepared_inputs(
            context=context, prepared=changed, output_directory=output,
            acknowledged_prepared_sha256=prepared.content_sha256,
        )


@pytest.mark.parametrize("policy", ["strict", "degraded_with_warning"])
def test_prepared_inputs_reject_required_funding_absent_from_snapshot(
    prepared_source, tmp_path, policy,
):
    fixture, runner, snapshot, _ = prepared_source
    _, context = _input_context(fixture, snapshot.slot)
    recipe = replace(_input_recipe(prepared_source, context, risk=False),
                     funding_policy=policy, funding_fingerprint="a" * 64)
    recording = _RecordingDerivedBuilder(runner)
    output = tmp_path / "funding-required"
    with pytest.raises(BacktestPreparePoolsRejected, match="effective funding"):
        _prepare_inputs(prepared_source, context, recipe, output, runner=recording)
    assert recording.requests == []
    assert not output.exists()


def test_native_preparation_rejects_memory_before_payload_scan(
    prepared_source, tmp_path, monkeypatch,
):
    from trading.contexts.backtest.application.services.v2.prepare_pools import (
        BacktestPreparePoolsRejected,
    )

    fixture, runner, snapshot, _ = prepared_source
    loader, context = _input_context(fixture, snapshot.slot)
    recipe = _input_recipe(prepared_source, context, risk=False)
    validator = prepared_validator.BacktestArtifactManifestValidatorV2(
        artifact_loader=fixture.loader,
    )
    service = BacktestPreparePoolsService(
        artifact_array_loader=loader, defaults_provider=runner.defaults_provider,
        config=BacktestPreparePoolsConfig(row_prefilter_top_fraction=1.0),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("payload scan preceded memory admission")

    monkeypatch.setattr(type(validator), "attest_snapshot", forbidden)
    with pytest.raises(BacktestPreparePoolsRejected, match="max_compute_bytes"):
        service.prepare_artifact_inputs(
            recipe=recipe, context=context, builder=runner, validator=validator,
            output_directory=tmp_path / "rejected", organization_id="org", job_id="job",
            owner_token="test", attempt=1, output_root_id="attempt",
            max_generated_bytes=10_000_000, max_compute_bytes=1,
        )
    assert not (tmp_path / "rejected").exists()


def test_oversized_prepared_marker_never_commits(prepared_source, tmp_path, monkeypatch):
    fixture, _, snapshot, _ = prepared_source
    loader, context = _input_context(fixture, snapshot.slot)
    recipe = _input_recipe(prepared_source, context, risk=False)
    _, _, prepared = _prepare_inputs(prepared_source, context, recipe, tmp_path / "normal")
    assert set(prepared.provenance) == {"reused"}
    monkeypatch.setattr(type(prepared), "as_mapping", lambda _: {"oversized": "x" * (8 * 1024**2)})
    with pytest.raises(ValueError, match="metadata disk budget"):
        loader.write_prepared_manifest(prepared=prepared, output_directory=tmp_path / "oversized")
    assert not list((tmp_path / "oversized").iterdir())
