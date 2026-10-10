"""Frozen managed selections and real native fixed-fixture publication parity."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2 import (
    test_artifact_precompute_runner_v2 as n,
)
from tests.unit.contexts.backtest.application.services.v2.test_derived_artifact_materialization import (  # noqa: E501
    snapshot_for,
)
from tests.unit.contexts.backtest_artifacts.test_automatic_base import START, Source, frozen, setup
from tests.unit.contexts.backtest_artifacts.test_management_profiles import (
    catalog as catalog_fixture,
)
from trading.contexts.backtest_artifacts.adapters.outbound.defaults import (
    YamlBacktestGridDefaultsProvider as ArtifactGridDefaults,
)
from trading.contexts.backtest_artifacts.application.services.derivative_profiles import (
    DerivativeDemand,
    ManagedDerivativePublisher,
    plan_derivatives,
)
from trading.contexts.backtest_artifacts.application.services.management import (
    Coordinate,
    Grid,
    History,
    Profile,
    SourceTarget,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactDerivativeBuildRequest,
    ArtifactRequestedRow,
)
from trading.contexts.indicators.adapters.outbound import NumbaIndicatorCompute
from trading.platform.config.indicators_compute_numba import IndicatorsComputeNumbaConfig

catalog = catalog_fixture


def profile(catalog, names=("ma.sma",), *, tp=(0.5, 2.0), sl=()):
    p = catalog.default_profile()
    entries = []
    for item in p.indicators:
        if item.indicator_id in names:
            params = {
                k: Grid(mode="explicit", values=(v.expand()[1], v.expand()[-1]))
                for k, v in item.params.items()
            }
            entries.append(
                item.model_copy(
                    update={
                        "params": params,
                        "sources": ("close",) if "close" in item.sources else item.sources,
                    }
                )
            )
    return catalog.validate(
        p.model_copy(
            update={
                "mode": "precompute",
                "indicators": tuple(entries),
                "derivative_timeframes": ("15m",),
                "tp_levels_pct": tp,
                "sl_levels_pct": sl,
            }
        )
    )


def demand(p, sha="a" * 64, *, minutes=7200, request_id=None):
    return DerivativeDemand.freeze(
        request_id=request_id or str(uuid4()),
        coordinate=Coordinate(exchange="binance", market_type="spot", symbol="BTCUSDT"),
        source=SourceTarget(
            history=History(
                kind="range", start_utc=START, end_utc=START + timedelta(minutes=minutes)
            ),
            revision="fixture-v1",
            watermark=START + timedelta(minutes=minutes),
            compute_identity="signals/v1|numba/v1|f32-f64/v1",
            funding_identity="not_applicable",
        ),
        source_sha256=sha,
        root_id="local-test",
        root_revision=1,
        profile=p,
    )


def test_all_default_exclusion_empty_and_budget(catalog):
    p = catalog.default_profile().model_copy(update={"mode": "precompute"})
    all_plan = plan_derivatives((demand(p),), grid_builder=n._signal_grid_builder_v2())
    assert len(all_plan.selected_rows) == 400
    assert len({i for _, i, _ in all_plan.selected_rows}) == 40
    excluded = p.indicators[0].indicator_id
    one = p.model_copy(update={"indicators": p.indicators[1:]})
    plan = plan_derivatives((demand(one),), grid_builder=n._signal_grid_builder_v2())
    assert len(plan.selected_rows) == 390
    assert all(i != excluded for _, i, _ in plan.selected_rows)
    empty = profile(catalog, names=())
    plan = plan_derivatives((demand(empty),), grid_builder=n._signal_grid_builder_v2())
    assert not plan.selected_rows and plan.tp and not plan.sl
    with pytest.raises(ValueError, match="resource_conflict"):
        plan_derivatives((demand(p),), grid_builder=n._signal_grid_builder_v2(), max_rows=1)


def test_frozen_union_versions_and_history(catalog):
    p = profile(catalog)
    first = demand(p)
    original = first.profile_json
    p.indicators[0].params.clear()  # nested model mutation cannot change the admitted bytes
    assert first.profile_json == original
    second = demand(profile(catalog, names=("ma.ema",)), request_id="second")
    plan = plan_derivatives((first, second), grid_builder=n._signal_grid_builder_v2())
    assert {i for _, i, _ in plan.selected_rows} == {"ma.sma", "ma.ema"}
    for changed in (replace(second, source_sha256="b" * 64), replace(second, root_revision=2)):
        with pytest.raises(ValueError, match="incompatible_demand"):
            plan_derivatives((first, changed), grid_builder=n._signal_grid_builder_v2())
    invalid = profile(catalog).model_copy(
        update={
            "history": History(
                kind="range", start_utc=START - timedelta(days=1), end_utc=START + timedelta(days=1)
            )
        }
    )
    with pytest.raises(ValueError, match="profile_history_outside_source"):
        plan_derivatives((demand(invalid),), grid_builder=n._signal_grid_builder_v2())


def test_mode_off_does_not_publish(catalog):
    p = catalog.default_profile()
    plan = plan_derivatives((demand(p),), grid_builder=n._signal_grid_builder_v2())
    assert not plan.active
    assert (
        ManagedDerivativePublisher(cast(Any, object()), "local-test", 1, Path("/unused")).publish(
            plan=plan
        )
        is None
    )


def build(tmp_path, catalog, *, tp=(0.5, 2.0), sl=(), names=("ma.sma", "ma.ema"), repo=None):
    f, base = setup(tmp_path, repo=repo)
    src = Source(7200, missing=(1007,))
    ix = np.arange(len(src.values))
    # Independent deterministic non-monotonic OHLCV fixture.
    price = 100 + 4 * np.sin(ix / 53) + np.cos(ix / 137)
    src.values[:, 0] = price
    src.values[:, 1] = price + 0.8
    src.values[:, 2] = price - 0.8
    src.values[:, 3] = price + 0.1
    snapshot = frozen(f, src, 7200)
    p = profile(catalog, names=names, tp=tp, sl=sl)
    plan = plan_derivatives((demand(p, snapshot.sha256),), grid_builder=n._signal_grid_builder_v2())
    runner = replace(
        base.runner,
        indicator_grid_builder=n._signal_grid_builder_v2(),
        defaults_provider=n.YamlBacktestGridDefaultsProvider.from_yaml(
            config_path=Path("configs/dev/indicators.yaml")
        ),
        signal_rules_engine=n.BacktestSignalRulesEngineV2(
            defaults_provider=n.YamlBacktestGridDefaultsProvider.from_yaml(
                config_path=Path("configs/dev/indicators.yaml")
            )
        ),
        indicator_compute=NumbaIndicatorCompute(
            defs=n.all_defs(),
            config=IndicatorsComputeNumbaConfig(
                numba_num_threads=1, numba_cache_dir=tmp_path / "jit"
            ),
        ),
    )
    base = replace(base, runner=runner)
    result = ManagedDerivativePublisher(base, "local-test", 1, f.builder.root).publish(
        plan=plan,
        coordinates=f.coordinates,
        source=snapshot,
        generated_at=START + timedelta(days=20),
    )
    return f, base, plan, result, snapshot


@pytest.mark.parametrize(
    "tp,sl,names",
    [
        ((0.5, 2.0), (), ("ma.sma", "ma.ema")),
        ((), (0.5,), ("ma.sma", "ma.ema")),
        ((0.5,), (1.0,), ("ma.sma", "ma.ema")),
        ((0.5,), (), ()),
        ((), (), ("volume.obv",)),
    ],
)
def test_native_publication_selected_rows_independent_risk_and_parity(
    tmp_path, catalog, tp, sl, names
):
    f, base, plan, result, source = build(tmp_path, catalog, tp=tp, sl=sl, names=names)
    assert result is not None and result.publication is not None
    slot = result.publication.published_pointer.active_slot
    manifest = f.loader.load_slot_manifest(f.coordinates, slot)
    assert manifest.schema_version == 2
    assert {(r.timeframe, r.indicator_id) for r in manifest.signals.manifests} == {
        (tf, ind) for tf, ind, _ in plan.selected_rows
    }
    configured, spec = plan.configure(base.runner, tuple(m.timeframe for m in manifest.mappings))
    assert not result.publication.validation.diagnostics
    snapshot = snapshot_for(f, slot)
    rows = []
    for tf, ind, ids in plan.selected_rows:
        sm = f.loader.load_signal_manifest(f.coordinates, slot, tf, ind)
        assert sm.grid.row_ids == ids
        assert configured.indicator_grid_builder is not None
        assert configured.defaults_provider is not None
        grid_spec = configured.defaults_provider.compute_defaults(indicator_id=ind)
        assert grid_spec is not None
        grid = configured.indicator_grid_builder.materialize_indicator(grid=grid_spec)
        for row in ids:
            remainder = row
            values = {}
            for axis in reversed(grid.axes):
                remainder, idx = divmod(remainder, len(axis.values))
                values[axis.name] = axis.values[idx]
            rows.append(
                ArtifactRequestedRow(
                    ind, values.pop("source", None), row, tuple(sorted(values.items()))
                )
            )
    rows = tuple(rows)
    request = ArtifactDerivativeBuildRequest(
        snapshot,
        rows,
        "signals/v1",
        configured.derivative_defaults_sha256(rows),
        "numba/v1",
        "f32-f64/v1",
        tp,
        sl,
        "test-output",
        "owner",
        1,
        100_000_000,
        100_000_000,
    )
    # The actual prepared-input resolver reuses sparse canonical IDs and independent risk.
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_prepare_pools_service as prep,
    )
    from trading.contexts.backtest.application.dto.input_recipe import BacktestInputRecipe

    if rows:
        _, context = prep._input_context(f, slot)
        refs = tuple(
            ref
            for ref in context.references
            if ref.domain.role.startswith(("prices.", "mappings."))
            and ref.domain.timeframe in ("1m", "15m")
        )
        full_snapshot = replace(
            snapshot,
            source_file_identities=refs,
            consumed_domains=tuple(ref.domain for ref in refs),
        )
        domain = next(ref.domain for ref in refs if ref.domain.role == "prices.15m.ohlcv")
        recipe = BacktestInputRecipe(
            snapshot=full_snapshot,
            requested_start_utc=domain.origin_utc,
            requested_end_utc=domain.end_utc,
            rows=tuple(sorted(rows, key=lambda r: (r.indicator_id, r.row_id))),
            rule_version="signals/v1",
            defaults_sha256=configured.derivative_defaults_sha256(rows),
            compute_version="numba/v1",
            precision_version="f32-f64/v1",
            funding_policy="not_applicable",
            funding_fingerprint=None,
            tp_levels_pct=tp,
            sl_levels_pct=sl,
        )
        _, _, prepared = prep._prepare_inputs(
            (f, configured, snapshot, None), context, recipe, tmp_path / "prepared"
        )
        assert set(prepared.provenance) <= {"reused"}
        assert recipe.semantic_sha256
    generated = replace(configured, canonical_candle_reader=None).materialize_derived(
        request, output_directory=tmp_path / "ondemand"
    )
    for ref in generated.files:
        actual = np.load(tmp_path / "ondemand" / ref.relative_path)
        if ref.domain.role.startswith("signals."):
            path = f.loader.resolve_signal_paths(
                f.coordinates, slot, "15m", ref.domain.role.split(".", 1)[1]
            ).signals
        elif ref.domain.role.startswith("signal_features."):
            path = f.loader.resolve_signal_features_paths(
                f.coordinates, slot, "15m", ref.domain.role.split(".", 1)[1]
            ).features
        else:
            path = getattr(
                f.loader.resolve_hit_times_paths(f.coordinates, slot),
                ref.domain.role.split(".", 1)[1],
            )
        np.testing.assert_array_equal(actual, np.load(path))
    # Stale source identity fails before any publication mutation.
    pointer = f.loader.resolve_current_pointer_path(f.coordinates).read_bytes()
    with pytest.raises(ValueError, match="derivative_source_changed"):
        ManagedDerivativePublisher(base, "local-test", 1, f.builder.root).publish(
            plan=plan,
            coordinates=f.coordinates,
            source=replace(source, sha256="b" * 64),
            generated_at=START + timedelta(days=20),
        )
    assert f.loader.resolve_current_pointer_path(f.coordinates).read_bytes() == pointer


def test_independent_sma_oracle_and_corrupt_declared_file(tmp_path, catalog):
    from trading.contexts.backtest.adapters.outbound.artifacts_fs.artifact_array_loader import (
        FilesystemBacktestArtifactArrayLoader,
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs.artifact_context_resolver import (
        FilesystemBacktestArtifactContextResolver,
    )
    from trading.contexts.backtest.application.dto import BacktestCoordinates
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_validator import (  # noqa: E501
        BacktestArtifactManifestValidatorV2,
    )

    f, base, plan, result, _ = build(tmp_path, catalog, names=("ma.sma",), tp=(), sl=())
    assert result is not None and result.publication is not None
    slot = result.publication.published_pointer.active_slot
    manifest = f.loader.load_slot_manifest(f.coordinates, slot)
    assert manifest.hit_times is None
    root = manifest.path.parent
    section = next(p for p in manifest.prices if p.timeframe == "15m")
    values = np.load(root / section.ohlcv.path)
    configured, spec = plan.configure(base.runner, tuple(m.timeframe for m in manifest.mappings))
    selected = Profile.model_validate_json(plan.demands[0].profile_json).indicators[0]
    # Independent scalar moving mean, no native indicator engine or wrapper used as oracle.
    columns = {"open": 0, "high": 1, "low": 2, "close": 3}
    series = values[:, columns[selected.sources[0]]]
    expected = []
    for window in selected.params["window"].expand():
        assert isinstance(window, (int, float))
        signal = np.zeros(len(series), dtype=np.int8)
        for j in range(int(window) - 1, len(series)):
            average = np.float32(
                sum(float(x) for x in series[j - int(window) + 1 : j + 1]) / window
            )
            signal[j] = 1 if series[j] > average else (-1 if series[j] < average else 0)
        expected.append(signal)
    path = f.loader.resolve_signal_paths(f.coordinates, slot, "15m", "ma.sma").signals
    np.testing.assert_array_equal(np.load(path), expected)
    coordinates = BacktestCoordinates(exchange="binance", market_type="spot", symbol="BTCUSDT")
    metadata = FilesystemBacktestArtifactContextResolver(f.loader).resolve_context(
        coordinates=coordinates
    )
    loader = FilesystemBacktestArtifactArrayLoader(f.loader)
    context = loader.resolve_context(coordinates=coordinates, artifact_metadata=metadata)
    matrix = loader.load_signal_matrix(context=context, timeframe="15m", indicator_id="ma.sma")
    assert matrix.row_ids == plan.selected_rows[0][2]
    # Physical metadata still validates every advertised row, even unused ones.
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    validation = BacktestArtifactManifestValidatorV2(f.loader).validate_slot(
        coordinates=f.coordinates, slot=slot, validation_spec=spec
    )
    assert validation.diagnostics
    with pytest.raises(ValueError, match="SHA-256"):
        loader.load_signal_matrix(context=context, timeframe="15m", indicator_id="ma.sma")


def test_unavailable_version_and_no_compute_params(catalog):
    from trading.contexts.backtest_artifacts.adapters.outbound.config.backtest_artifacts_runtime_config import (  # noqa: E501
        load_backtest_artifacts_runtime_config,
    )
    from trading.contexts.backtest_artifacts.adapters.outbound.defaults.management_catalog import (
        ManagementCatalog,
    )
    from trading.contexts.indicators.adapters.outbound.registry import YamlIndicatorRegistry

    unavailable = catalog.default_profile().indicators[0].indicator_id
    changed = ManagementCatalog(
        defaults=ArtifactGridDefaults.from_yaml(config_path=Path("configs/dev/indicators.yaml")),
        registry=YamlIndicatorRegistry.from_yaml(
            defs=n.all_defs(), config_path="configs/dev/indicators.yaml"
        ),
        config=load_backtest_artifacts_runtime_config("configs/dev/backtest_artifacts.yaml"),
        unavailable=frozenset((unavailable,)),
    )
    assert len(changed.default_profile().indicators) == 39
    assert len(changed.default_profile().catalog_indicators) == 40
    with pytest.raises(ValueError, match="catalog_changed"):
        changed.validate(catalog.default_profile())
    old = demand(profile(catalog))
    assert plan_derivatives((old,), grid_builder=n._signal_grid_builder_v2()).active
    no_params = tuple(i.indicator_id for i in catalog.default_profile().indicators if not i.params)
    assert len(no_params) == 4
    selected = profile(catalog, names=no_params, tp=(), sl=())
    plan = plan_derivatives((demand(selected),), grid_builder=n._signal_grid_builder_v2())
    assert {i for _, i, _ in plan.selected_rows} == set(no_params)


def test_frozen_funding_risk_parity_and_root_identity(tmp_path, catalog):
    import hashlib
    import json
    from dataclasses import asdict

    from trading.contexts.backtest_artifacts.application.services.automatic_base import (
        FrozenFunding,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
        ArtifactCoordinatesV2,
    )
    from trading.contexts.market_data.application.ports.stores.funding_rate_coverage_reader import (
        FundingRateArtifactRecord,
        FundingRateCoverageSnapshot,
    )
    from trading.shared_kernel.primitives import UtcTimestamp

    _, spot, _, _, _ = build(tmp_path / "spot", catalog, names=(), tp=(0.5,), sl=())
    coordinates = ArtifactCoordinatesV2("binance", "futures", "BTCUSDT")
    f, base = setup(tmp_path / "futures", coordinates=coordinates)
    source = frozen(f, Source(180), 180)
    funding = FundingRateCoverageSnapshot(
        "ready",
        "degraded_with_warning",
        source.time_range,
        UtcTimestamp(START),
        UtcTimestamp(START),
        1,
        1,
        0,
        (),
        (
            FundingRateArtifactRecord(
                source.instrument,
                "binance:futures:BTCUSDT",
                UtcTimestamp(START),
                0.0001,
                100.0,
                480,
                0,
            ),
        ),
    )
    source_digest = hashlib.sha256(
        (
            source.sha256
            + json.dumps(asdict(funding), sort_keys=True, default=str, allow_nan=False)
        ).encode()
    ).hexdigest()
    base = replace(
        base,
        runner=replace(
            spot.runner,
            artifact_loader=f.loader,
            funding_rate_coverage_reader=FrozenFunding(source.instrument, funding),
        ),
    )
    d = demand(profile(catalog, names=(), tp=(0.5,), sl=()), source_digest, minutes=180)
    target = SourceTarget.model_validate_json(d.source_json).model_copy(
        update={"funding_identity": "canonical_funding/v1"}
    )
    d = replace(
        d,
        coordinate_json=Coordinate(
            exchange="binance", market_type="futures", symbol="BTCUSDT"
        ).model_dump_json(),
        source_json=target.model_dump_json(),
    )
    plan = plan_derivatives((d,), grid_builder=n._signal_grid_builder_v2())
    pub = ManagedDerivativePublisher(base, "local-test", 1, f.builder.root)
    for wrong, reason in (
        (replace(pub, root_revision=2), "derivative_root_changed"),
        (replace(pub, root_path=tmp_path / "elsewhere"), "derivative_root_changed"),
    ):
        with pytest.raises(ValueError, match=reason):
            wrong.publish(
                plan=plan,
                coordinates=coordinates,
                source=source,
                generated_at=START + timedelta(days=20),
            )
    result = pub.publish(
        plan=plan, coordinates=coordinates, source=source, generated_at=START + timedelta(days=20)
    )
    assert result is not None
    assert result.source_sha256 == source_digest
    assert result is not None and result.publication is not None
    slot = result.publication.published_pointer.active_slot
    manifest = f.loader.load_slot_manifest(coordinates, slot)
    assert manifest.funding is not None
    assert manifest.funding.rows_count == 1
    root = manifest.path.parent
    assert manifest.funding.funding_rate is not None
    np.testing.assert_allclose(np.load(root / manifest.funding.funding_rate.path), [0.0001])
    configured, _ = plan.configure(base.runner, tuple(m.timeframe for m in manifest.mappings))
    request = ArtifactDerivativeBuildRequest(
        snapshot_for(f, slot),
        (),
        "signals/v1",
        configured.derivative_defaults_sha256(()),
        "numba/v1",
        "f32-f64/v1",
        (0.5,),
        (),
        "output",
        "owner",
        1,
        10_000_000,
        10_000_000,
    )
    output = tmp_path / "on-demand-futures"
    generated = configured.materialize_derived(request, output_directory=output)
    for ref in generated.files:
        path = getattr(
            f.loader.resolve_hit_times_paths(coordinates, slot), ref.domain.role.split(".", 1)[1]
        )
        np.testing.assert_array_equal(np.load(output / ref.relative_path), np.load(path))


def test_explicit_edit_rebase_preserves_exclusions_and_adds_new_ids(catalog):
    current = catalog.default_profile()
    new_id = current.catalog_indicators[-1].indicator_id
    excluded = current.indicators[0].indicator_id
    old = current.model_copy(
        update={
            "catalog_revision": "0" * 64,
            "catalog_indicators": current.catalog_indicators[:-1],
            "indicators": tuple(
                i for i in current.indicators if i.indicator_id not in (new_id, excluded)
            ),
        }
    )
    edited = catalog.rebase_profile(old)
    selected = {i.indicator_id for i in edited.indicators}
    assert new_id in selected and excluded not in selected
    assert edited.revision == old.revision + 1
    assert old.catalog_revision == "0" * 64
    assert len(catalog.items()) == 40


def test_union_does_not_invent_cross_products(catalog):
    full = next(i for i in catalog.default_profile().indicators if i.indicator_id == "ma.sma")
    windows = full.params["window"].expand()
    profiles = []
    for source, window in zip(("close", "open"), (windows[0], windows[-1])):
        item = full.model_copy(
            update={
                "sources": (source,),
                "params": {"window": Grid(mode="explicit", values=(window,))},
            }
        )
        profiles.append(
            catalog.validate(profile(catalog).model_copy(update={"indicators": (item,)}))
        )
    plan = plan_derivatives(
        tuple(demand(p) for p in profiles), grid_builder=n._signal_grid_builder_v2()
    )
    assert len(plan.selected_rows) == 1 and len(plan.selected_rows[0][2]) == 2


def test_bad_engine_and_funding_identity_prevent_publication(tmp_path, catalog):
    f, base, plan, _, source = build(tmp_path, catalog, names=(), tp=(0.5,))
    pointer = f.loader.resolve_current_pointer_path(f.coordinates).read_bytes()
    original = plan.demands[0]
    for field, reason in (
        ("compute_identity", "unsupported_compute_identity"),
        ("funding_identity", "unsupported_funding_identity"),
    ):
        target = SourceTarget.model_validate_json(original.source_json).model_copy(
            update={field: "changed"}
        )
        bad = plan_derivatives(
            (replace(original, source_json=target.model_dump_json()),),
            grid_builder=n._signal_grid_builder_v2(),
        )
        with pytest.raises(ValueError, match=reason):
            ManagedDerivativePublisher(base, "local-test", 1, f.builder.root).publish(
                plan=bad,
                coordinates=f.coordinates,
                source=source,
                generated_at=START + timedelta(days=20),
            )
    assert f.loader.resolve_current_pointer_path(f.coordinates).read_bytes() == pointer


def test_partial_derivatives_require_schema2(tmp_path, catalog):
    import yaml

    from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_validator import (  # noqa: E501
        BacktestArtifactManifestValidatorV2,
    )

    f, base, plan, result, _ = build(tmp_path, catalog, names=("ma.sma",))
    assert result is not None and result.publication is not None
    slot = result.publication.published_pointer.active_slot
    path = f.loader.resolve_slot_manifest_path(f.coordinates, slot)
    payload = yaml.safe_load(path.read_text())
    payload["schema_version"] = 1
    path.write_text(yaml.safe_dump(payload))
    manifest = f.loader.load_slot_manifest(f.coordinates, slot)
    _, spec = plan.configure(base.runner, tuple(m.timeframe for m in manifest.mappings))
    result = BacktestArtifactManifestValidatorV2(f.loader).validate_slot(
        coordinates=f.coordinates, slot=slot, validation_spec=spec
    )
    assert any(d.code == "partial_derivatives_require_schema2" for d in result.diagnostics)


def test_frozen_subset_catalog_preserves_legacy_demand_without_mutable_defaults(catalog):
    from trading.contexts.backtest_artifacts.application.services.derivative_profiles import (
        FrozenProfileDefaults,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.signal_rules_engine_v2 import (
        BacktestSignalRulesEngineV2,
    )
    selected = profile(catalog, names=("ma.ema",))
    subset = selected.model_copy(update={
        "catalog_indicators": tuple(x for x in selected.catalog_indicators
                                    if x.indicator_id == "ma.ema"),
    })
    plan = plan_derivatives((demand(subset),), grid_builder=n._signal_grid_builder_v2())
    assert {name for _, name, _ in plan.selected_rows} == {"ma.ema"}
    defaults = FrozenProfileDefaults(subset.model_dump_json())
    with pytest.raises(ValueError, match="catalog drift"):
        BacktestSignalRulesEngineV2(defaults_provider=defaults)
    engine = BacktestSignalRulesEngineV2(defaults_provider=defaults, indicator_scope=("ma.ema",))
    assert engine.supported_indicator_ids() == ("ma.ema",)
    assert engine.rule_spec(indicator_id=" MA.EMA ").indicator_id == "ma.ema"
    with pytest.raises(ValueError, match="outside frozen"):
        engine.resolved_defaults(indicator_id="ma.sma")
    with pytest.raises(ValueError, match="invalid frozen"):
        BacktestSignalRulesEngineV2(defaults_provider=defaults, indicator_scope=("unknown.rule",))
