"""Real NPY and production indicator parity at the selected builder boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.unit.contexts.backtest.application.services.v2 import (
    test_artifact_precompute_runner_v2 as native,
)
from trading.contexts.backtest_artifacts.application.services.v2 import (
    artifact_precompute_runner as runner_module,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCandleSnapshot,
    ArtifactDerivativeBuildRequest,
    ArtifactDerivativeBuildResult,
    ArtifactInputFileReference,
    ArtifactPrefixDomain,
    ArtifactRequestedRow,
)
from trading.contexts.indicators.adapters.outbound import NumbaIndicatorCompute
from trading.platform.config.indicators_compute_numba import IndicatorsComputeNumbaConfig


def snapshot_for(fixture, slot):
    manifest = fixture.loader.load_slot_manifest(fixture.coordinates, slot)
    manifest_path = fixture.loader.resolve_slot_manifest_path(fixture.coordinates, slot)
    root = manifest_path.parent
    refs = []
    for price in manifest.prices:
        for name in ("open_time", "close_time", "ohlcv"):
            metadata = getattr(price, name)
            array = np.load(root / metadata.path, mmap_mode="r", allow_pickle=False)
            coverage = price.coverage
            refs.append(
                ArtifactInputFileReference(
                    "published-source",
                    metadata.path,
                    metadata.sha256,
                    ArtifactPrefixDomain(
                        f"prices.{price.timeframe}.{name}",
                        price.timeframe,
                        array.dtype.str,
                        metadata.axis_order,
                        datetime.fromtimestamp(
                            coverage.open_time_start / 1000, timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        datetime.fromtimestamp(
                            coverage.close_time_end / 1000, timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        array.shape[0],
                        tuple(array.shape),
                    ),
                )
            )
    return ArtifactCandleSnapshot(
        fixture.coordinates,
        manifest.schema_version,
        slot,
        manifest.slot_generation,
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "15m",
        "1m",
        tuple(refs),
        tuple(ref.domain for ref in refs),
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    directory = tmp_path_factory.mktemp("selected-native")
    targets = tuple(("15m", name) for name in ("ma.sma", "ma.ema", "ma.wma"))
    fixture = native.build_artifact_precompute_fixture_v2(
        tmp_path=directory,
        validation_signal_artifacts=targets,
        precompute_signal_artifacts=targets,
        signal_worker_processes=1,
        hit_times_tp_levels_pct=(0.5, 1.0, 2.0),
        hit_times_sl_levels_pct=(0.5, 1.0),
    )
    defaults = native._PrecomputeSignalDefaultsProvider(
        delegate=native.YamlBacktestGridDefaultsProvider.from_yaml(
            config_path=Path("configs/prod/indicators.yaml")
        ),
        overrides={
            name: native.GridSpec(
                indicator_id=native.IndicatorId(name),
                params={
                    "window": native.ExplicitValuesSpec(name="window", values=(20, 21, 50, 100))
                },
                source=native.ExplicitValuesSpec(name="source", values=("close", "open")),
            )
            for _, name in targets
        },
    )
    candles = []
    for index in range(7200):
        if index == 1007:  # Missing source minute, retained by the actual production rollup.
            continue
        row = native._build_canonical_row_v2(bar_index=index, price_offset=0, volume_offset=0)
        price = 100 + 4 * np.sin(index / 53) + np.cos(index / 137)
        candles.append(
            replace(
                row,
                candle=replace(
                    row.candle,
                    open=float(price),
                    high=float(price + 0.8),
                    low=float(price - 0.8),
                    close=float(price + 0.1),
                    volume_base=10.0,
                ),
            )
        )
    runner = native.BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=replace(
            fixture.runtime_settings, price_timeframes=("1m", "15m"), mapping_timeframes=("15m",)
        ),
        artifact_loader=fixture.loader,
        canonical_candle_reader=native._FakeCanonicalCandleReader(rows=tuple(candles)),
        defaults_provider=defaults,
        signal_rules_engine=native.BacktestSignalRulesEngineV2(defaults_provider=defaults),
        indicator_compute=NumbaIndicatorCompute(
            defs=native.all_defs(),
            config=IndicatorsComputeNumbaConfig(
                numba_num_threads=1, numba_cache_dir=directory / "jit"
            ),
        ),
        indicator_grid_builder=native._signal_grid_builder_v2(),
    )
    export_request = native._request_v2(fixture=fixture, end_minute=7200)
    result = runner.export_canonical_price_1m(export_request)
    snapshot = snapshot_for(fixture, result.slot)
    return fixture, runner, snapshot, export_request


def request_for(built, *, risk=False, selected=(1, 3, 6)):
    _, runner, snapshot, _ = built
    rows = []
    for indicator in ("ma.sma", "ma.ema", "ma.wma"):
        grid = runner.indicator_grid_builder.materialize_indicator(
            grid=runner.defaults_provider.compute_defaults(indicator_id=indicator)
        )
        values = list(product(*(axis.values for axis in grid.axes)))
        for index in selected:
            parameters = dict(zip((axis.name for axis in grid.axes), values[index], strict=True))
            source = parameters.pop("source")
            rows.append(
                ArtifactRequestedRow(indicator, source, index, tuple(sorted(parameters.items())))
            )
    requested = tuple(rows)
    return ArtifactDerivativeBuildRequest(
        snapshot,
        requested,
        "signals/v1",
        runner.derivative_defaults_sha256(requested),
        "numba/v1",
        "f32-f64/v1",
        (0.5, 2.0) if risk else (),
        (1.0,) if risk else (),
        "attempt-source",
        "test-owner",
        1,
        10_000_000,
        10_000_000,
    )


def test_selected_real_compute_sparse_rows_features_and_risk(built, tmp_path):
    fixture, runner, snapshot, _ = built
    request = request_for(built, risk=True)
    output = tmp_path / "inputs"
    pointer = fixture.loader.resolve_current_pointer_path(fixture.coordinates)
    pointer_bytes = pointer.read_bytes()
    result = replace(runner, canonical_candle_reader=None).materialize_derived(
        request, output_directory=output
    )
    raw = yaml.safe_load((output / "manifest.yaml").read_text())
    assert raw["manifest_kind"] == "derived_artifacts"
    assert ArtifactDerivativeBuildResult.from_mapping(raw["result"]) == result
    source_root = fixture.loader.resolve_slot_manifest_path(
        fixture.coordinates, snapshot.slot
    ).parent
    for ref in result.files:
        actual = np.load(output / ref.relative_path, allow_pickle=False)
        assert hashlib.sha256((output / ref.relative_path).read_bytes()).hexdigest() == ref.sha256
        if ref.domain.role.startswith("signals."):
            indicator = ref.domain.role.removeprefix("signals.")
            path = fixture.loader.resolve_signal_paths(
                fixture.coordinates, snapshot.slot, "15m", indicator
            ).signals
            expected = np.load(path)[list(ref.row_ids)]
        elif ref.domain.role.startswith("signal_features."):
            indicator = ref.domain.role.removeprefix("signal_features.")
            path = fixture.loader.resolve_signal_features_paths(
                fixture.coordinates, snapshot.slot, "15m", indicator
            ).features
            expected = np.load(path)[list(ref.row_ids)]
        else:
            name = ref.domain.role.removeprefix("hit_times.")
            path = getattr(
                fixture.loader.resolve_hit_times_paths(fixture.coordinates, snapshot.slot), name
            )
            indexes = [0, 2] if "tp" in name else [1]
            expected = np.load(path)[indexes]
        np.testing.assert_array_equal(actual, expected)
    assert pointer.read_bytes() == pointer_bytes
    for ref in snapshot.source_file_identities:
        assert (
            hashlib.sha256((source_root / ref.relative_path).read_bytes()).hexdigest() == ref.sha256
        )


def test_no_risk_no_canonical_reads_no_nested_pools(built, tmp_path, monkeypatch):
    _, runner, _, _ = built

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden dispatch")

    monkeypatch.setattr(runner_module, "materialize_hit_times_from_ohlcv_v2", forbidden)
    monkeypatch.setattr(runner_module, "ThreadPoolExecutor", forbidden)
    monkeypatch.setattr(runner_module, "ProcessPoolExecutor", forbidden)
    output = tmp_path / "inputs"
    result = replace(runner, canonical_candle_reader=None).materialize_derived(
        request_for(built), output_directory=output
    )
    assert len(result.files) == 6
    assert not (output / "hit_times").exists()


@pytest.mark.parametrize("budget", ["max_generated_bytes", "max_compute_bytes"])
def test_budget_rejected_before_output_allocation(built, tmp_path, budget):
    _, runner, _, _ = built
    output = tmp_path / "inputs"
    with pytest.raises(ValueError, match=budget):
        runner.materialize_derived(
            replace(request_for(built), **{budget: 1}), output_directory=output
        )
    assert not output.exists()


@pytest.mark.parametrize("point", ["_execute_signal_chunk_job_v2", "_write_npy_atomically_v2"])
def test_injected_failure_never_commits_manifest(built, tmp_path, monkeypatch, point):
    _, runner, _, _ = built

    def fail(*args, **kwargs):
        raise OSError("injected disk/chunk failure")

    monkeypatch.setattr(runner_module, point, fail)
    output = tmp_path / "inputs"
    with pytest.raises(OSError, match="injected"):
        runner.materialize_derived(request_for(built), output_directory=output)
    assert not (output / "manifest.yaml").exists()
    assert not list(output.rglob("*.tmp"))


def test_row_identity_and_versions_fail_closed(built, tmp_path):
    _, runner, _, _ = built
    request = request_for(built)
    wrong = replace(request.rows[0], parameters=(("window", 99),))
    with pytest.raises(ValueError, match="canonical ID"):
        runner.materialize_derived(
            replace(request, rows=(wrong,) + request.rows[1:]), output_directory=tmp_path / "wrong"
        )
    with pytest.raises(ValueError, match="version"):
        runner.materialize_derived(
            replace(request, compute_version="unknown/v9"), output_directory=tmp_path / "version"
        )


def test_private_publication_candidate_preserves_every_source_byte(built, tmp_path):
    fixture, runner, snapshot, request = built
    source = fixture.loader.resolve_slot_manifest_path(fixture.coordinates, snapshot.slot).parent
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()
    }
    output = tmp_path / "candidate"
    result = runner.export_canonical_price_1m(request, output_directory=output)
    assert result.manifest_path == output / "manifest.yaml"
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    for indicator in ("ma.sma", "ma.ema", "ma.wma"):
        original = fixture.loader.resolve_signal_paths(
            fixture.coordinates, snapshot.slot, "15m", indicator
        ).signals
        np.testing.assert_array_equal(
            np.load(original), np.load(output / original.relative_to(source))
        )


def test_prefix_hash_chunking_preserves_c_order_endian_and_nan_bits():
    raw = np.arange(600_000, dtype=np.uint32).reshape(1000, 600)
    raw[3, 7] = 0x7FC01234
    original = raw.view(np.float32)
    for array in (original[:, :300], original.byteswap().view(">f4")[:, :300]):
        domain = ArtifactPrefixDomain(
            "prices.15m.ohlcv",
            "15m",
            "<f4",
            ("time", "field"),
            "2025-01-01T00:00:00Z",
            "2025-02-01T00:00:00Z",
            array.shape[0],
            array.shape,
        )
        descriptor = json.dumps(
            domain.as_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        expected = hashlib.sha256(
            len(descriptor).to_bytes(8, "little")
            + descriptor
            + original[:, :300].tobytes(order="C")
        ).hexdigest()
        before = array.tobytes()
        assert domain.payload_digest(array) == expected
        assert array.tobytes() == before


@pytest.mark.parametrize("family", ["tp", "sl"])
def test_only_missing_risk_family_without_signal_dependencies(built, tmp_path, family):
    fixture, runner, snapshot, _ = built
    request = replace(
        request_for(built, risk=True),
        rows=(),
        defaults_sha256=runner.derivative_defaults_sha256(()),
        tp_levels_pct=(2.0,) if family == "tp" else (),
        sl_levels_pct=(1.0,) if family == "sl" else (),
    )
    runner = replace(
        runner,
        canonical_candle_reader=None,
        defaults_provider=None,
        indicator_compute=None,
        indicator_grid_builder=None,
        signal_rules_engine=None,
        runtime_settings=replace(runner.runtime_settings, signal_artifacts=()),
    )
    output = tmp_path / "inputs"
    result = runner.materialize_derived(request, output_directory=output)
    assert len(result.files) == 3
    assert not (output / "signals").exists()
    for ref in result.files:
        role = ref.domain.role.removeprefix("hit_times.")
        original = getattr(
            fixture.loader.resolve_hit_times_paths(fixture.coordinates, snapshot.slot), role
        )
        index = 2 if family == "tp" else 1
        np.testing.assert_array_equal(
            np.load(output / ref.relative_path), np.load(original)[[index]]
        )


def test_closed_file_corruption_cannot_commit_manifest(built, tmp_path, monkeypatch):
    _, runner, _, _ = built
    original = runner_module._write_npy_atomically_v2

    def corrupt(*, path, array):
        original(path=path, array=np.zeros((1,), dtype=np.int16))

    monkeypatch.setattr(runner_module, "_write_npy_atomically_v2", corrupt)
    output = tmp_path / "inputs"
    with pytest.raises(ValueError, match="closed derivative"):
        runner.materialize_derived(request_for(built), output_directory=output)
    assert not (output / "manifest.yaml").exists()


def test_computes_only_missing_rows_with_complete_seed_history(built, tmp_path, monkeypatch):
    _, runner, snapshot, _ = built
    request = request_for(built, selected=(3,))
    original = runner_module._execute_signal_chunk_job_v2
    seen = []

    def record(**kwargs):
        seen.append(
            (
                kwargs["rule_spec"].indicator_id,
                len(kwargs["signal_rows"]),
                kwargs["candles"].close.shape[0],
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(runner_module, "_execute_signal_chunk_job_v2", record)
    result = runner.materialize_derived(request, output_directory=tmp_path / "inputs")
    bars = next(d.shape[0] for d in snapshot.consumed_domains if d.role == "prices.15m.open_time")
    assert seen == [(indicator, 1, bars) for indicator in ("ma.ema", "ma.sma", "ma.wma")]
    assert all(ref.row_ids == (3,) for ref in result.files)


def test_derivative_constructor_requires_reader_only_at_export(built):
    _, runner, _, export_request = built
    runner = replace(runner, canonical_candle_reader=None)
    with pytest.raises(ValueError, match="canonical_candle_reader"):
        runner.export_canonical_price_1m(export_request)


def test_snapshot_identity_must_match_published_manifest(built, tmp_path):
    _, runner, snapshot, _ = built
    request = request_for(built)
    ref = snapshot.source_file_identities[0]
    forged = replace(
        snapshot,
        source_file_identities=(replace(ref, sha256="a" * 64),)
        + snapshot.source_file_identities[1:],
    )
    with pytest.raises(ValueError, match="published manifest"):
        runner.materialize_derived(
            replace(request, snapshot=forged), output_directory=tmp_path / "inputs"
        )


def test_false_prefix_end_rejected_against_actual_timestamps(built, tmp_path):
    _, runner, snapshot, _ = built
    consumed = tuple(replace(d, end_utc="2026-03-30T00:00:00Z") for d in snapshot.consumed_domains)
    with pytest.raises(ValueError, match="source timestamps"):
        runner.materialize_derived(
            replace(request_for(built), snapshot=replace(snapshot, consumed_domains=consumed)),
            output_directory=tmp_path / "inputs",
        )


@pytest.mark.parametrize("corruption", ["grid", "monotonicity"])
def test_risk_contents_validated_after_writer_returns(built, tmp_path, monkeypatch, corruption):
    _, runner, _, _ = built
    original = runner_module._write_npy_atomically_v2

    def corrupt(*, path, array):
        if corruption == "grid" and path.name == "tp_values.f32.npy":
            array = array.copy()
            array[0] = np.float32(0.02)
        if corruption == "monotonicity" and path.name == "long_tp.u32.npy":
            array = array.copy()
            array[0, :] = array.shape[1]
            array[1, :] = 0
        original(path=path, array=array)

    monkeypatch.setattr(runner_module, "_write_npy_atomically_v2", corrupt)
    output = tmp_path / "inputs"
    with pytest.raises(ValueError, match=f"risk {corruption}"):
        runner.materialize_derived(request_for(built, risk=True), output_directory=output)
    assert not (output / "manifest.yaml").exists()


def test_hit_time_admission_accounts_for_numba_thread_scratch(built, tmp_path, monkeypatch):
    _, runner, snapshot, _ = built
    bars = next(d.shape[0] for d in snapshot.consumed_domains if d.role == "prices.15m.open_time")
    request = replace(
        request_for(built),
        rows=(),
        defaults_sha256=runner.derivative_defaults_sha256(()),
        tp_levels_pct=tuple((index + 1) / 2 for index in range(32)),
        max_compute_bytes=1_048_576 + bars * (128 + 64 * 6) + 1,
    )
    monkeypatch.setattr(runner_module, "get_num_threads", lambda: 32)
    output = tmp_path / "inputs"
    with pytest.raises(ValueError, match="hit-times exceed max_compute_bytes"):
        runner.materialize_derived(request, output_directory=output)
    assert not output.exists()


def test_on_demand_publication_skips_derivatives_and_retains_source(built, tmp_path, monkeypatch):
    module = runner_module

    fixture, runner, _, request = built
    builder = replace(fixture.builder, root=tmp_path / "on-demand")
    loader = replace(fixture.loader, path_resolver=builder)
    runner = replace(runner, artifact_loader=loader, runtime_settings=replace(
        runner.runtime_settings, signal_artifacts=(), precompute_hit_times=False,
    ))

    def forbidden(*args, **kwargs):
        raise AssertionError("proactive derivative generation under on_demand")

    monkeypatch.setattr(module, "_materialize_hit_times_artifacts_v2", forbidden)
    monkeypatch.setattr(module, "_materialize_signal_artifact_v2", forbidden)
    result = runner.export_canonical_price_1m(replace(
        request, target_slot="slot_a", target_slot_generation=1, force_full_rebuild=True,
    ))
    manifest = loader.load_slot_manifest(fixture.coordinates, result.slot)
    assert manifest.schema_version == 2
    assert manifest.hit_times is None
    assert manifest.signals.manifests == ()
    assert len(manifest.prices) == 2
    assert len(manifest.mappings) == 1
    assert not list(builder.root.rglob("long_tp.npy"))
    assert result.stage_rebuild_stats.hit_times.rewritten_tail_bars == 0


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("risk", [False, True])
def test_real_preflight_freezes_metadata_without_payload_scan(
    built, tmp_path, monkeypatch, partial, risk,
):
    import shutil
    from dataclasses import asdict

    from tests.unit.contexts.backtest.application.services.v2 import (
        test_backtest_preflight_service as preflight_tests,
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs.artifact_array_loader import (
        FilesystemBacktestArtifactArrayLoader,
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs.artifact_context_resolver import (
        FilesystemBacktestArtifactContextResolver,
    )
    from trading.contexts.backtest.application.services.v2.preflight import BacktestPreflightService

    fixture, runner, snapshot, _ = built
    root = tmp_path / "source"
    shutil.copytree(fixture.builder.root, root)
    loader = replace(fixture.loader, path_resolver=replace(fixture.builder, root=root))
    path = loader.resolve_slot_manifest_path(fixture.coordinates, snapshot.slot)
    manifest = yaml.safe_load(path.read_text())
    if partial:
        manifest["schema_version"] = 2
        manifest.pop("hit_times")
        manifest["signals"]["manifests"] = []
        path.write_text(yaml.safe_dump(manifest))
    pointer = loader.resolve_current_pointer_path(fixture.coordinates)
    pointer.write_text(yaml.safe_dump({
        "schema_version": 1, "active_slot": snapshot.slot,
        "slot_generation": snapshot.generation, "asof_date": manifest["asof_date"],
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "published_at_utc": "2026-03-25T02:00:00Z",
    }))
    request = preflight_tests._valid_request()
    request["coordinates"] = asdict(fixture.coordinates)
    request["time_range"] = {"start": snapshot.consumed_domains[0].origin_utc,
                             "end": manifest["asof_date"] + "T12:00:00Z"}
    request["indicators"] = [{"indicator_id": "ma.ema", "sources": ["close"],
                              "window": {"start": 21, "stop": 21, "step": 1}}]
    if risk:
        request["risk"] = {
            "mode": "tp_sl_grid",
            "tp": {"start_pct": 0.5, "stop_pct": 2.0, "step_pct": 1.5},
            "sl": {"start_pct": 1.0, "stop_pct": 1.0, "step_pct": 1.0},
        }
    service = BacktestPreflightService(
        defaults_provider=runner.defaults_provider,
        artifact_context_resolver=FilesystemBacktestArtifactContextResolver(artifact_loader=loader),
        runtime_config=replace(
            preflight_tests._runtime_config(),
            artifact_config_hash=runner.runtime_settings.config_sha256,
            hit_times_tp_levels_pct=runner.runtime_settings.hit_times_tp_levels_pct,
            hit_times_sl_levels_pct=runner.runtime_settings.hit_times_sl_levels_pct,
        ),
        artifact_array_loader=FilesystemBacktestArtifactArrayLoader(artifact_loader=loader),
        indicator_grid_builder=runner.indicator_grid_builder,
    )
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*")
              if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight may not read array payload or compute derivatives")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(type(runner), "materialize_derived", forbidden)
    result = service.execute(request)
    assert result.input_recipe is not None
    assert result.input_recipe.snapshot.prefix_proof.state == "pending"
    assert result.input_readiness is not None
    assert result.input_readiness["status"] == ("requires_materialization" if partial else "ready")
    changed_storage = replace(service, runtime_config=replace(
        service.runtime_config, artifact_config_hash="f" * 64,
    )).execute(request)
    assert changed_storage.request_hash == result.request_hash
    assert changed_storage.result_config_hash == result.result_config_hash
    assert changed_storage.input_recipe is not None
    assert changed_storage.input_recipe.semantic_sha256 == result.input_recipe.semantic_sha256
    public = json.dumps(result.as_mapping())
    assert str(root) not in public
    assert "source_file_identities" not in public
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*")
                      if p.is_file()}

    # A corrupted length must fail before NumPy attempts to read its declared header body.
    source_path = path.parent / manifest["prices"][0]["ohlcv"]["path"]
    original = source_path.read_bytes()
    source_path.write_bytes(b"\x93NUMPY\x02\x00" + (100_000_000).to_bytes(4, "little"))
    with monkeypatch.context() as bounded:
        bounded.setattr(np.lib.format, "read_array_header_2_0", forbidden)
        with pytest.raises(preflight_tests.BacktestPreflightRejected) as rejected:
            service.execute(request)
        assert "metadata budget" in str(rejected.value.__cause__)
    source_path.write_bytes(original)

    # Matching declared dtype/header still must satisfy the role's financial storage contract.
    array_meta = manifest["prices"][0]["ohlcv"]
    np.save(source_path, np.zeros(array_meta["shape"], dtype=np.float64), allow_pickle=False)
    array_meta["dtype"] = "float64"
    array_meta["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(manifest))
    pointer_payload = yaml.safe_load(pointer.read_text())
    pointer_payload["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    pointer.write_text(yaml.safe_dump(pointer_payload))
    with pytest.raises(preflight_tests.BacktestPreflightRejected) as rejected:
        service.execute(request)
    assert "semantic dtype/shape/axes" in str(rejected.value.__cause__)


@pytest.mark.parametrize("direction", ["long_only", "short", "long_short_reversal"])
@pytest.mark.parametrize("risk_mode", ["none", "tp_only", "sl_only", "both"])
@pytest.mark.parametrize("touch", ["tp", "sl", "tie", "none", "last", "reversal"])
def test_materialized_risk_has_independent_financial_exits(
    built, tmp_path, direction, risk_mode, touch,
):
    """Real 15m NPY generation/loading with prescribed signals and independent trade exits."""
    from types import SimpleNamespace
    from typing import Any, cast

    from tests.unit.contexts.backtest.application.services.v2 import (
        test_prepare_pools_service as source,
    )
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_tp_sl_exact_scoring_service as scoring,
    )
    from tests.unit.contexts.backtest.application.services.v2.test_lazy_trades_detail_service import (  # noqa: E501
        _MemoryCache,
    )
    from trading.contexts.backtest.application.services.v2 import (
        BacktestComboPlanningService,
        BacktestLazyTradesDetailService,
        BacktestNoRiskExactScoringService,
        BacktestTpSlExactScoringService,
        BacktestTpSlHitTimesService,
    )

    fixture, runner, _, _ = built
    loader = replace(fixture.loader, path_resolver=replace(fixture.builder, root=tmp_path / "root"))
    fixture = replace(fixture, loader=loader, builder=loader.path_resolver)
    side = -1 if direction == "short" else 1
    candles = []
    for minute in range(60):
        bar = minute // 15
        hit = bar == (3 if touch == "last" else 2)
        high = 101.0 if hit and touch in ("tp", "tie", "last") else 100.0
        low = 99.0 if hit and touch in ("sl", "tie") else 100.0
        if side == -1:
            high, low = 200.0 - low, 200.0 - high
        row = native._build_canonical_row_v2(bar_index=minute, price_offset=0, volume_offset=0)
        candles.append(replace(row, candle=replace(
            row.candle, open=100.0, high=high, low=low, close=100.0, volume_base=10.0,
        )))
    runner = replace(runner, artifact_loader=loader,
                     canonical_candle_reader=native._FakeCanonicalCandleReader(rows=tuple(candles)),
                     runtime_settings=replace(runner.runtime_settings, signal_artifacts=(),
                                              precompute_hit_times=False))
    export = replace(native._request_v2(fixture=fixture, end_minute=60),
                     target_slot="slot_a", target_slot_generation=1, force_full_rebuild=True)
    published = runner.export_canonical_price_1m(export)
    snapshot = snapshot_for(fixture, published.slot)
    arrays, context = source._input_context(fixture, published.slot)
    owned = fixture, runner, snapshot, export
    recipe = source._input_recipe(owned, context, risk=risk_mode != "none")
    if risk_mode != "none":
        recipe = replace(recipe, tp_levels_pct=(1.0,) if risk_mode != "sl_only" else (),
                         sl_levels_pct=(1.0,) if risk_mode != "tp_only" else ())
    _, bound, prepared_inputs = source._prepare_inputs(owned, context, recipe, tmp_path / "inputs")
    request = scoring._normalized_request(
        direction_mode=direction, market_type="futures", funding_mode="include_when_futures",
    )
    funding = scoring._funding_arrays(
        funding_time=(900_000, 1_800_000, 2_700_000, 3_600_000),
        funding_rate=(0.001,) * 4, mark_price=(100.0,) * 4,
    )
    request["risk"] = {"mode": "none"} if risk_mode == "none" else {
        "mode": "tp_sl_grid",
        "tp": {"enabled": risk_mode != "sl_only", "start_pct": 1.0,
               "stop_pct": 1.0, "step_pct": 1.0},
        "sl": {"enabled": risk_mode != "tp_only", "start_pct": 1.0,
               "stop_pct": 1.0, "step_pct": 1.0},
    }
    prepared = scoring._prepared_result(
        indicator_ids=("alpha",), trade_rows_by_id={"alpha": (
            [[1, -1, -1, -1]] if touch == "reversal" and direction == "long_short_reversal"
            else [[side, side, side, side]]
        )},
        open_1m=[100.0] * 4, close_1m=[100.0] * 4,
    )
    planning = BacktestComboPlanningService().execute(prepared_result=prepared,
                                                      normalized_request=request)
    hit_service = BacktestTpSlHitTimesService(artifact_array_loader=arrays)
    detail = BacktestLazyTradesDetailService(prepare_pools=cast(Any, object()),
                                           tp_sl_hit_times=hit_service, cache=_MemoryCache())
    times = SimpleNamespace(open_time=np.arange(4) * 900_000,
                            close_time=np.arange(4) * 900_000 + 899_999)
    runtime_arrays = SimpleNamespace(price_arrays_1m=times, price_arrays_15m=times)
    if risk_mode == "none":
        result = BacktestNoRiskExactScoringService().execute(
            prepared_result=prepared, combo_planning_result=planning, normalized_request=request,
            funding_arrays=funding,
        )
        _, trades, _ = detail._no_risk_detail(normalized_request=request, prepared=prepared,
                                             local_indices=(0,), runtime_arrays=runtime_arrays)
        assert not any(ref.domain.role.startswith("hit_times.")
                       for ref in prepared_inputs.derivatives)
    else:
        hits = hit_service.execute(normalized_request=request, context=bound)
        assert hits is not None and hits.hit_times.sentinel_index == 4
        result = BacktestTpSlExactScoringService().execute(
            prepared_result=prepared, combo_planning_result=planning,
            normalized_request=request, hit_times_result=hits, funding_arrays=funding,
        )
        _, trades, _ = detail._tp_sl_detail(
            normalized_request=request, prepared=prepared, local_indices=(0,),
            runtime_arrays=runtime_arrays, context=bound,
            row=cast(Any, SimpleNamespace(best_tp_pct=1.0 if risk_mode != "sl_only" else None,
                                          best_sl_pct=1.0 if risk_mode != "tp_only" else None)),
        )
    if touch == "reversal" and direction == "long_short_reversal":
        assert [(trade["side"], trade["entry_bar_index"], trade["exit_bar_index"],
                 trade["exit_reason"]) for trade in trades] == [
            ("long", 1, 2, "signal"), ("short", 2, 3, "close_on_end"),
        ]
        assert result.top_results[0].metrics["trade_count"] == 2
        assert result.top_results[0].metrics["total_return_pct"] == pytest.approx(0, abs=1e-5)
        assert result.top_results[0].metrics["funding_events_count"] == 2
        assert result.top_results[0].metrics["funding_pnl_quote"] == pytest.approx(0, abs=1e-5)
        return
    sl = risk_mode in ("sl_only", "both") and touch in ("sl", "tie")
    tp = risk_mode in ("tp_only", "both") and touch in ("tp", "tie", "last")
    expected_return = -1.0 if sl else 1.0 if tp else 0.0
    expected_exit = 2 if sl or (tp and touch != "last") else 3
    reason = "stop_loss" if sl else "take_profit" if tp else "close_on_end"
    assert len(trades) == 1
    trade = trades[0]
    assert trade["entry_bar_index"] == 1
    assert trade["exit_bar_index"] == expected_exit
    assert trade["exit_reason"] == reason
    expected_time = expected_exit * 900_000 + (899_999 if reason == "close_on_end" else 0)
    assert datetime.fromisoformat(trade["exit_timestamp"].replace("Z", "+00:00")).timestamp() * (
        1000
    ) == expected_time
    assert trade["exit_price"] == pytest.approx(100 + side * expected_return, abs=1e-5)
    assert trade["net_pnl_quote"] == pytest.approx(expected_return * 100, abs=1e-3)
    assert result.top_results[0].metrics["total_return_pct"] == pytest.approx(
        expected_return, abs=1e-5,
    )
    # Entry-time event is excluded; an exit-time event is included; later events are excluded.
    expected_events = 1 if expected_exit == 2 else 2
    metrics = result.top_results[0].metrics
    assert metrics["funding_events_count"] == expected_events
    assert metrics["funding_pnl_quote"] == pytest.approx(-side * 10 * expected_events, abs=1e-5)


def test_generated_manifest_slot_is_reserved_before_allocation(built, tmp_path):
    _, runner, _, _ = built
    output = tmp_path / "metadata-bound"
    with pytest.raises(ValueError, match="max_generated_bytes"):
        runner.materialize_derived(
            replace(request_for(built), max_generated_bytes=8 * 1024**2),
            output_directory=output,
        )
    assert not output.exists()
