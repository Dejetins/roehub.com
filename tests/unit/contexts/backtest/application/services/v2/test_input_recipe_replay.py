"""Original-domain replay using production NPY preparation and detail consumers."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import UUID

import numba as nb
import numpy as np
import pytest
import yaml

from tests.unit.contexts.backtest.application.services.v2 import (
    test_lazy_trades_detail_service as details,
)
from tests.unit.contexts.backtest.application.services.v2 import (
    test_prepare_pools_service as source,
)
from trading.contexts.backtest.adapters.outbound.cache_fs.lazy_trades_cache import (
    LocalFileBacktestLazyTradesCache,
)
from trading.contexts.backtest.application.dto.artifact_inputs import BacktestAttemptInputs
from trading.contexts.backtest.application.ports.lazy_trades_cache import (
    build_lazy_trades_cache_key,
)
from trading.contexts.backtest.application.services.v2.lazy_trades_detail import (
    BacktestLazyTradesDetailService,
)
from trading.contexts.backtest.application.services.v2.prepare_pools import (
    BacktestPreparePoolsService,
)
from trading.contexts.backtest.application.services.v2.tp_sl_hit_times import (
    BacktestTpSlHitTimesService,
)
from trading.contexts.backtest.domain.entities import BacktestJobArtifactPin
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_validator import (  # noqa: E501
    BacktestArtifactManifestValidatorV2,
)
from trading.platform.errors import RoehubError

built = source.built
prepared_source = source.prepared_source


def replay_fixture(prepared_source, tmp_path, *, risk=True, direction="long_only"):
    fixture, runner, snapshot, _ = prepared_source
    loader, context = source._input_context(fixture, snapshot.slot)
    recipe = source._input_recipe(prepared_source, context, risk=risk)
    funding = context.source.slot_manifest.funding
    if funding is not None and funding.coverage_status == "ready":
        refs = tuple(
            ref
            for ref in context.references
            if ref.domain.role.startswith(("prices.", "mappings.", "funding."))
        )
        recipe = replace(
            recipe,
            funding_policy="degraded_with_warning",
            funding_fingerprint=funding.funding_manifest_hash,
            snapshot=replace(
                recipe.snapshot,
                source_file_identities=refs,
                consumed_domains=tuple(ref.domain for ref in refs),
            ),
        )
    # All rows of one genuine indicator; selected detail will build only row 1.
    rows = tuple(row for row in recipe.rows if row.indicator_id == "ma.ema")
    recipe = replace(recipe, rows=rows, defaults_sha256=runner.derivative_defaults_sha256(rows))
    _, _, job, _ = details._service_fixture(tmp_path=tmp_path / "identity", risk_mode="none")
    request = dict(job.request_json)
    request["coordinates"] = asdict(recipe.snapshot.coordinates)
    request["time_range"] = {"start": recipe.requested_start_utc, "end": recipe.requested_end_utc}
    request["indicators"] = [
        {
            "indicator_id": "ma.ema",
            "sources": ["close", "open"],
            "window": {"start": 21, "stop": 100, "step": 79},
        }
    ]
    request["execution"] = {
        **request["execution"],
        "direction_mode": direction,
        "funding": {"mode": "include_when_futures", "coverage_policy": "degraded_with_warning"},
    }
    request["top_n"] = 10
    request["risk"] = (
        {
            "mode": "tp_sl_grid",
            "tp": {"start_pct": 0.5, "stop_pct": 2.0, "step_pct": 1.5},
            "sl": {"start_pct": 1.0, "stop_pct": 1.0, "step_pct": 1.0},
        }
        if risk
        else {"mode": "none"}
    )
    metadata = replace(
        details.detail_module._artifact_metadata_from_job(job=job),
        artifact_slot=recipe.snapshot.slot,
        artifact_slot_generation=recipe.snapshot.generation,
        artifact_manifest_hash=recipe.snapshot.manifest_sha256,
        artifact_asof_date=context.source.artifact_asof_date,
    )
    if funding is not None:
        metadata = replace(
            metadata,
            funding_manifest_hash=funding.funding_manifest_hash,
            funding_coverage_status=funding.coverage_status,
            funding_coverage_policy=funding.coverage_policy,
            funding_rows_count=funding.rows_count,
            funding_expected_event_count=funding.expected_event_count,
            funding_missing_event_count=funding.missing_event_count,
            funding_reason_codes=funding.reason_codes,
        )
    request["artifact_metadata"] = metadata.as_mapping()
    job = replace(
        job,
        job_id=UUID("00000000-0000-0000-0000-000000000002"),
        request_json=request,
        attempt=1,
        artifact_pin=BacktestJobArtifactPin(
            cast(Literal["slot_a", "slot_b"], recipe.snapshot.slot),
            recipe.snapshot.generation,
            recipe.snapshot.manifest_sha256,
            metadata.artifact_asof_date,
        ),
    )
    top = details._top_result(risk_mode="tp_sl_grid" if risk else "none", row_id=1)
    top = replace(top, metadata={"ma.ema.source": "close", "ma.ema.window": 21})
    row = (
        details.BacktestTopResultAssemblyService()
        .assemble(
            job_id=job.job_id,
            normalized_request=request,
            top_results=(top,),
            updated_at=datetime.now(UTC),
        )
        .top_variants[0]
    )
    service = BacktestLazyTradesDetailService(
        prepare_pools=BacktestPreparePoolsService(
            artifact_array_loader=loader, defaults_provider=runner.defaults_provider
        ),
        tp_sl_hit_times=BacktestTpSlHitTimesService(artifact_array_loader=loader),
        cache=details._MemoryCache(),
        derivative_builder=runner,
        input_validator=BacktestArtifactManifestValidatorV2(artifact_loader=fixture.loader),
    )
    baseline = service.execute(
        job=job, row=row, public_variant_key=row.payload_json["public_variant_key"]
    )
    _, bound, prepared = source._prepare_inputs(
        prepared_source, context, recipe, tmp_path / "original-attempt"
    )
    bound.close_mmaps()
    shutil.rmtree(tmp_path / "original-attempt")
    job = replace(
        job,
        input_recipe_json=recipe.as_mapping(),
        preparation_provenance_json=prepared.as_mapping(),
    )
    return service, job, row, recipe, baseline


def native_top_ten_fixture(prepared_source, tmp_path, *, risk):
    """Compute real top10 with the production full-job runtime before source cleanup."""
    from trading.contexts.backtest.application.dto import (
        BacktestComboPlanningConfig,
        BacktestCostEstimate,
        BacktestPreflightResult,
        BacktestPreparePoolsConfig,
    )
    from trading.contexts.backtest.application.services.v2.combo_planning import (
        BacktestComboPlanningService,
    )
    from trading.contexts.backtest.application.services.v2.compute_policy import (
        BacktestComputePolicy,
    )
    from trading.contexts.backtest.application.services.v2.job_orchestration import (
        BacktestRuntimeJobOrchestrationService,
    )
    from trading.contexts.backtest.application.services.v2.job_scheduling import (
        BacktestNumbaThreadDecision,
    )
    from trading.contexts.backtest.application.services.v2.no_risk_exact import (
        BacktestNoRiskExactScoringService,
    )
    from trading.contexts.backtest.application.services.v2.tp_sl_exact import (
        BacktestTpSlExactScoringService,
    )

    service, job, _, recipe, _ = replay_fixture(prepared_source, tmp_path, risk=risk)
    fixture, runner, snapshot, _ = prepared_source
    loader, context = source._input_context(fixture, snapshot.slot)
    rows = source._input_recipe(prepared_source, context, risk=risk).rows
    recipe = replace(recipe, rows=rows, defaults_sha256=runner.derivative_defaults_sha256(rows))
    request = dict(job.request_json)
    request["indicators"] = [
        {**request["indicators"][0], "indicator_id": name}
        for name in ("ma.ema", "ma.sma", "ma.wma")
    ]
    request["quality_constraints"] = {"min_closed_trades": 1}
    job = replace(
        job, request_json=request, input_recipe_json=None, preparation_provenance_json=None
    )
    preflight = BacktestPreflightResult(
        normalized_request=request,
        request_hash=job.request_hash,
        result_config_hash=job.backtest_runtime_config_hash,
        artifact_metadata=details.detail_module._artifact_metadata_from_job(job=job),
        cost_estimate=BacktestCostEstimate(
            indicator_rows=12, candidate_combinations=64, tp_sl_cells=2 if risk else 0,
            cost_class="light",
        ),
    )
    runtime = BacktestRuntimeJobOrchestrationService(
        compute_policy=BacktestComputePolicy(
            threads=BacktestNumbaThreadDecision(
                num_threads=nb.get_num_threads(), source="fixture_numba_budget"
            )
        ),
        prepare_pools=replace(
            service.prepare_pools,
            config=BacktestPreparePoolsConfig(row_prefilter_top_fraction=1.0),
        ),
        combo_planning=BacktestComboPlanningService(
            config=BacktestComboPlanningConfig(combo_top_frac=1.0, combo_min_confirm=1)
        ),
        no_risk_exact=BacktestNoRiskExactScoringService(),
        tp_sl_hit_times=service.tp_sl_hit_times,
        tp_sl_exact=BacktestTpSlExactScoringService(),
        artifact_array_loader=loader,
    )
    result = runtime.execute(job_id=job.job_id, preflight=preflight, updated_at=datetime.now(UTC))
    assert len(result.top_variants) == 10
    row = result.top_variants[0]
    service = replace(service, cache=details._MemoryCache())
    baseline = service.execute(
        job=job, row=row, public_variant_key=row.payload_json["public_variant_key"]
    )
    _, bound, prepared = source._prepare_inputs(
        prepared_source, context, recipe, tmp_path / "top10-attempt"
    )
    bound.close_mmaps()
    shutil.rmtree(tmp_path / "top10-attempt")
    job = replace(
        job, input_recipe_json=recipe.as_mapping(),
        preparation_provenance_json=prepared.as_mapping(),
    )
    return service, job, result.top_variants, recipe, baseline


def append_source(prepared_source):
    """Append one 15m bar and 15 execution bars; original payload bytes remain exact."""
    fixture, _, original, _ = prepared_source
    path = fixture.loader.resolve_slot_manifest_path(fixture.coordinates, original.slot)
    root = path.parent
    payload = yaml.safe_load(path.read_text())
    for price in payload["prices"]:
        n = 15 if price["timeframe"] == "1m" else 1
        step = 60_000 if n == 15 else 900_000
        for name in ("open_time", "close_time", "ohlcv"):
            item = price[name]
            array = np.load(root / item["path"], allow_pickle=False)
            tail = (
                array[-1] + step * np.arange(1, n + 1, dtype=array.dtype)
                if name != "ohlcv"
                else np.repeat(array[-1:], n, axis=0)
            )
            write_array(root, item, np.concatenate((array, tail)))
        price["coverage"]["bar_count"] += n
        price["coverage"]["close_time_end"] += 900_000
    for mapping in payload["mappings"]:
        for name in ("bar_open_1m_idx", "bar_close_1m_idx"):
            item = mapping[name]
            array = np.load(root / item["path"], allow_pickle=False)
            write_array(root, item, np.concatenate((array, array[-1:] + np.uint32(15))))
    if payload.get("funding", {}).get("coverage_status") == "ready":
        from tests.unit.contexts.backtest.application.services.v2 import artifact_testkit_v2 as kit

        funding = payload["funding"]
        for name in (
            "funding_time",
            "funding_rate",
            "mark_price",
            "funding_interval_minutes",
            "data_quality",
        ):
            item = funding[name]
            array = np.load(root / item["path"])
            tail = array[-1:].copy()
            if name == "funding_time":
                tail += 900_000
            write_array(root, item, np.concatenate((array, tail)))
        funding["rows_count"] += 1
        funding["expected_event_count"] += 1
        funding["funding_manifest_hash"] = kit._funding_manifest_hash(funding)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def write_array(root, metadata, array):
    path = root / metadata["path"]
    np.save(path, array, allow_pickle=False)
    metadata["shape"] = list(array.shape)
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def replay_service(service, job, prepared_source, tmp_path):
    fixture, _, snapshot, _ = prepared_source
    _, candidate = source._input_context(fixture, snapshot.slot)
    metadata = replace(
        details.detail_module._artifact_metadata_from_job(job=job),
        artifact_manifest_hash=candidate.source.artifact_manifest_hash,
        artifact_slot_generation=candidate.source.slot_generation,
        artifact_asof_date=candidate.source.artifact_asof_date,
    )
    recording = source._RecordingDerivedBuilder(service.derivative_builder)
    result = replace(
        service,
        replay_metadata=metadata,
        derivative_builder=recording,
        cache=LocalFileBacktestLazyTradesCache(root=tmp_path / "cache"),
        attempt_inputs=BacktestAttemptInputs(
            tmp_path / "inputs",
            str(job.organization_id),
            str(job.job_id),
            "replay-owner",
            2,
            10_000_000,
            10_000_000,
            lambda prepared: prepared.content_sha256,
        ),
    )
    return result, recording


@pytest.mark.parametrize("risk", [False, True])
@pytest.mark.parametrize("append", [False, True])
def test_cleanup_restart_selected_details_series_csv_and_original_sentinels(
    prepared_source,
    tmp_path,
    risk,
    append,
):
    service, job, row, recipe, baseline = replay_fixture(prepared_source, tmp_path, risk=risk)
    native_cache = LocalFileBacktestLazyTradesCache(root=tmp_path / "native-cache")
    native_key = service.read_cached(
        job=job, row=row, public_variant_key=row.payload_json["public_variant_key"]
    ).cache_key
    assert isinstance(service.cache, details._MemoryCache)
    native_cache.write(
        cache_key=native_key,
        payload=service.cache.writes[0][1],
        now=datetime.now(UTC),
        ttl_seconds=100,
    )
    native_candles = replace(service, cache=native_cache).price_candles(job=job, max_bars=100)
    source._rewrite_published_inventory(prepared_source, "generated")
    if append:
        append_source(prepared_source)
    service, recording = replay_service(service, job, prepared_source, tmp_path / "restart")
    actual = service.execute(
        job=job, row=row, public_variant_key=row.payload_json["public_variant_key"]
    )
    assert actual.trades == baseline.trades
    assert actual.summary_metrics == baseline.summary_metrics
    assert actual.chart_overlay == baseline.chart_overlay
    assert actual.funding == baseline.funding
    assert actual.artifact_manifest_hash == baseline.artifact_manifest_hash
    assert len(recording.requests) == 1
    request = recording.requests[0]
    assert [item.row_id for item in request.rows] == [1]
    assert request.tp_levels_pct == ((2.0,) if risk else ())
    assert request.sl_levels_pct == ((1.0,) if risk else ())
    assert request.snapshot.consumed_domains == recipe.snapshot.consumed_domains
    if risk:
        tables = np.load(tmp_path / "restart/inputs/hit_times/15m/long_tp.u32.npy")
        bars = next(
            d.shape[0] for d in recipe.snapshot.consumed_domains if d.role == "prices.15m.open_time"
        )
        assert tables.shape == (1, bars)
        assert np.max(tables) <= bars
        assert tables[0, -1] == bars
    now = datetime.now(UTC)
    key = service.read_cached(
        job=job, row=row, public_variant_key=row.payload_json["public_variant_key"]
    ).cache_key
    for method, kwargs in (
        ("read_csv", {"max_rows": None}),
        ("read_series", {"kind": "equity", "points": 100}),
    ):
        expected = getattr(native_cache, method)(
            cache_key=native_key, now=now, ttl_seconds=100, **kwargs
        )
        actual = getattr(service.cache, method)(cache_key=key, now=now, ttl_seconds=100, **kwargs)
        assert actual.is_hit and expected.is_hit
        keys = (
            ("content", "row_count", "total_rows")
            if method == "read_csv"
            else ("points", "source_points", "returned_points", "downsampled")
        )
        for name in keys:
            assert actual.payload[name] == expected.payload[name]
    assert service.price_candles(job=job, max_bars=100) == native_candles


@pytest.mark.parametrize("changed", ["candle", "mapping", "math", "attestation"])
def test_replay_rejects_changed_consumed_history_and_versions(prepared_source, tmp_path, changed):
    service, job, row, recipe, _ = replay_fixture(prepared_source, tmp_path)
    source._rewrite_published_inventory(prepared_source, "generated")
    if changed in ("candle", "mapping"):
        fixture, _, snapshot, _ = prepared_source
        path = fixture.loader.resolve_slot_manifest_path(fixture.coordinates, snapshot.slot)
        payload = yaml.safe_load(path.read_text())
        item = (
            payload["prices"][0]["ohlcv"]
            if changed == "candle"
            else payload["mappings"][0]["bar_open_1m_idx"]
        )
        array = np.load(path.parent / item["path"])
        array.flat[0] += 1
        write_array(path.parent, item, array)
        path.write_text(yaml.safe_dump(payload))
    elif changed == "math":
        recipe = replace(recipe, compute_version="unavailable/v99")
        assert job.preparation_provenance_json is not None
        proof = dict(job.preparation_provenance_json)
        proof["recipe_sha256"] = recipe.semantic_sha256
        job = replace(job, input_recipe_json=recipe.as_mapping(), preparation_provenance_json=proof)
    else:
        job = replace(job, preparation_provenance_json=None)
    service, recording = replay_service(service, job, prepared_source, tmp_path / "restart")
    with pytest.raises((RoehubError, ValueError)):
        service.execute(job=job, row=row, public_variant_key=row.payload_json["public_variant_key"])
    assert not recording.requests
    assert not (tmp_path / "restart/inputs/prepared-inputs.yaml").exists()


def test_recipe_cache_namespace_excludes_storage_but_preserves_org_source_variant():
    args = dict(
        organization_id="org1",
        job_id="job1",
        variant_key="v1",
        variant_hash="h1",
        request_hash="r1",
        engine_params_hash="e1",
        artifact_manifest_hash="a1",
        recipe_sha256="s1",
    )
    original = build_lazy_trades_cache_key(**args)
    assert (
        original.digest
        == build_lazy_trades_cache_key(
            **{
                **args,
                "artifact_manifest_hash": "different-storage",
                "engine_params_hash": "retention",
            }
        ).digest
    )
    for name in ("organization_id", "recipe_sha256", "variant_hash"):
        assert original.digest != build_lazy_trades_cache_key(**{**args, name: "changed"}).digest
    assert original.digest != build_lazy_trades_cache_key(**{**args, "recipe_sha256": None}).digest


def with_futures_funding(prepared_source):
    """Use genuine nonempty event NPY arrays on the native deterministic price fixture."""
    from tests.unit.contexts.backtest.application.services.v2 import artifact_testkit_v2 as kit
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_derived_artifact_materialization as derived,
    )

    fixture, runner, snapshot, request = prepared_source
    old_path = fixture.loader.resolve_slot_manifest_path(fixture.coordinates, snapshot.slot)
    coordinates = replace(fixture.coordinates, market_type="futures")
    new_path = fixture.loader.resolve_slot_manifest_path(coordinates, snapshot.slot)
    new_path.parent.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(old_path.parent, new_path.parent)
    payload = yaml.safe_load(new_path.read_text())
    payload["identity"]["market_type"] = "futures"
    for entry in payload["signals"]["manifests"]:
        signal_path = new_path.parent / entry["manifest_path"]
        signal = yaml.safe_load(signal_path.read_text())
        grid = runner.indicator_grid_builder.materialize_indicator(
            grid=runner.defaults_provider.compute_defaults(indicator_id=entry["indicator_id"]),
        )
        rows = derived.runner_module._build_signal_variant_rows_v2(
            coordinates=coordinates,
            timeframe="15m",
            materialized_grid=grid,
            row_ids=tuple(range(signal["rows_count"])),
        )
        signal["grid"]["variant_keys_sha256"] = derived.runner_module._variant_keys_sha256_v2(
            signal_rows=rows
        )
        signal_path.write_text(yaml.safe_dump(signal))
        entry["manifest_sha256"] = hashlib.sha256(signal_path.read_bytes()).hexdigest()
    funding_paths = fixture.builder.funding_paths(coordinates, snapshot.slot)
    times = np.load(new_path.parent / payload["prices"][1]["open_time"]["path"])
    values = {
        "funding_time": times,
        "funding_rate": np.full(times.shape, 0.001, dtype=np.float64),
        "mark_price": np.full(times.shape, 100.0, dtype=np.float64),
        "funding_interval_minutes": np.full(times.shape, 15, dtype=np.uint16),
        "data_quality": np.ones(times.shape, dtype=np.uint8),
    }
    for name, array in values.items():
        path = getattr(funding_paths, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
    funding = {
        "coverage_status": "ready",
        "coverage_policy": "ready",
        "rows_count": len(times),
        "expected_event_count": len(times),
        "missing_event_count": 0,
        "reason_codes": [],
    }
    for name, array in values.items():
        path = getattr(funding_paths, name)
        funding[name] = {
            "path": str(path.relative_to(new_path.parent)),
            "dtype": array.dtype.name,
            "shape": list(array.shape),
            "axis_order": ["funding_event"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    funding["funding_manifest_hash"] = kit._funding_manifest_hash(funding)
    payload["funding"] = funding
    new_path.write_text(yaml.safe_dump(payload))
    fixture = replace(fixture, coordinates=coordinates)
    snapshot = derived.snapshot_for(fixture, snapshot.slot)
    return fixture, runner, snapshot, request


@pytest.mark.parametrize("risk", [False, True])
@pytest.mark.parametrize("direction", ["long_only", "short", "long_short_reversal"])
def test_futures_funding_replay_boundaries_and_consumed_event_mutation(
    prepared_source,
    tmp_path,
    risk,
    direction,
):
    prepared_source = with_futures_funding(prepared_source)
    service, job, row, recipe, baseline = replay_fixture(
        prepared_source,
        tmp_path,
        risk=risk,
        direction=direction,
    )
    events = baseline.chart_overlay.get("funding_events", [])
    assert events, "nonempty effective funding is required for this proof"
    trades = {trade["trade_index"]: trade for trade in baseline.trades}
    for event in events:
        trade = trades[event["trade_index"]]
        assert trade["entry_timestamp"] < event["timestamp"] <= trade["exit_timestamp"]
        expected = (-1 if trade["side"] == "long" else 1) * trade["quantity"] * 100.0 * 0.001
        assert event["estimated_pnl_quote"] == pytest.approx(expected)
    source._rewrite_published_inventory(prepared_source, "generated")
    append_source(prepared_source)
    service, recording = replay_service(service, job, prepared_source, tmp_path / "replay")
    actual = service.execute(
        job=job, row=row, public_variant_key=row.payload_json["public_variant_key"]
    )
    assert actual.chart_overlay == baseline.chart_overlay
    assert actual.funding == baseline.funding
    assert actual.trades == baseline.trades
    assert len(recording.requests) == 1
    fixture, _, snapshot, _ = prepared_source
    path = fixture.loader.resolve_slot_manifest_path(fixture.coordinates, snapshot.slot)
    payload = yaml.safe_load(path.read_text())
    item = payload["funding"]["funding_rate"]
    array = np.load(path.parent / item["path"])
    array[1] *= -1
    write_array(path.parent, item, array)
    from tests.unit.contexts.backtest.application.services.v2 import artifact_testkit_v2 as kit

    payload["funding"]["funding_manifest_hash"] = kit._funding_manifest_hash(payload["funding"])
    path.write_text(yaml.safe_dump(payload))
    service, recording = replay_service(service, job, prepared_source, tmp_path / "mutated")
    with pytest.raises(RoehubError) as failure:
        service.execute(job=job, row=row, public_variant_key=row.payload_json["public_variant_key"])
    assert failure.value.code == "backtest.artifacts_unavailable"
    assert not recording.requests


def test_selected_float32_level_uses_canonical_recipe_value():
    from trading.contexts.backtest.application.services.v2.lazy_trades_detail import (
        _canonical_selected_level,
    )

    for level in (7.5, 13.5, 15.0):
        stored = float(np.float32(level / 100) * np.float32(100))
        assert _canonical_selected_level(stored, (level,)) == (level,)


def test_frozen_queued_preflight_does_not_resolve_current_pointer(prepared_source, tmp_path):
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_backtest_preflight_service as preflight_tests,
    )
    from trading.contexts.backtest.application.services.v2.preflight import BacktestPreflightService

    _, job, _, recipe, _ = replay_fixture(prepared_source, tmp_path, risk=False)
    _, runner, _, _ = prepared_source
    resolver = preflight_tests._FakeArtifactResolver()
    preflight = BacktestPreflightService(
        defaults_provider=runner.defaults_provider,
        artifact_context_resolver=resolver,
        runtime_config=preflight_tests._runtime_config(),
    )
    metadata = details.detail_module._artifact_metadata_from_job(job=job)
    metadata = replace(metadata, artifact_asof_date=recipe.requested_end_utc[:10])
    first = preflight.execute(job.request_json, stored_artifact_metadata=metadata)
    job = replace(
        job,
        request_json={**first.normalized_request, "artifact_metadata": metadata.as_mapping()},
        request_hash=first.request_hash,
    )
    changed = replace(
        preflight, runtime_config=replace(preflight.runtime_config, artifact_config_hash="b" * 64)
    )
    replay = changed.validate_stored_recipe(job=job)
    assert replay.input_recipe == recipe
    assert replay.normalized_request == first.normalized_request
    assert replay.request_hash == first.request_hash
    assert replay.result_config_hash == job.backtest_runtime_config_hash
    assert replay.artifact_metadata == metadata
    assert resolver.coordinates == ()


def test_append_funding_event_inside_original_domain_is_rejected(prepared_source, tmp_path):
    from tests.unit.contexts.backtest.application.services.v2 import artifact_testkit_v2 as kit

    prepared_source = with_futures_funding(prepared_source)
    service, job, row, recipe, _ = replay_fixture(prepared_source, tmp_path, risk=False)
    source._rewrite_published_inventory(prepared_source, "generated")
    append_source(prepared_source)
    fixture, _, snapshot, _ = prepared_source
    path = fixture.loader.resolve_slot_manifest_path(fixture.coordinates, snapshot.slot)
    payload = yaml.safe_load(path.read_text())
    item = payload["funding"]["funding_time"]
    times = np.load(path.parent / item["path"])
    domain = next(d for d in recipe.snapshot.consumed_domains if d.role == "funding.funding_time")
    times[-1] = (
        int(datetime.fromisoformat(domain.end_utc.replace("Z", "+00:00")).timestamp() * 1000) - 1
    )
    write_array(path.parent, item, times)
    payload["funding"]["funding_manifest_hash"] = kit._funding_manifest_hash(payload["funding"])
    path.write_text(yaml.safe_dump(payload))
    service, recording = replay_service(service, job, prepared_source, tmp_path / "replay")
    with pytest.raises(RoehubError) as failure:
        service.execute(job=job, row=row, public_variant_key=row.payload_json["public_variant_key"])
    assert failure.value.details is not None
    assert "funding original event domain changed" in failure.value.details["reason"]
    assert not recording.requests
