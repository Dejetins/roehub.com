# ruff: noqa: F811
"""Final disposable integration and cross-stage regressions; no installed runtime."""

from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

import pytest

from apps.api.dto.backtest_artifacts import Command, CommandItem, PolicyPatch
from apps.api.wiring.artifact_service import ArtifactHttpService
from tests.real_infra.backtest.test_artifact_dispatch import (
    DSN,
    NOW,
    catalog,  # noqa: F401
    dispatch,  # noqa: F401
    item,
    managed,  # noqa: F401
    submit,
    worker,
)
from tests.real_infra.backtest.test_artifact_input_lifecycle import (  # noqa: F401
    lifecycle_database,
    live_jobs,
)
from tests.unit.contexts.backtest_artifacts.test_derivative_profiles import profile

pytestmark = pytest.mark.skipif(not DSN, reason="explicit disposable PG required")


def test_profile_edit_waits_for_future_occurrence(dispatch):
    m = dispatch
    r = m["repo"]
    p = item(m, profile=profile(m["catalog"], tp=(), sl=()))
    r.observe_source(
        installation_id=m["installation"], coordinate=p.coordinate, source=p.source, observed_at=NOW
    )
    service = ArtifactHttpService(r, m["installation"])

    def edit(version, selected, key):
        return service.submit(
            m["org"],
            m["actor"],
            key,
            Command(
                items=(
                    CommandItem(
                        coordinate=p.coordinate,
                        expected_version=version,
                        patch=PolicyPatch(profile=selected),
                    ),
                )
            ),
            "policy",
        )

    first = edit(0, p.profile, "enable-precompute")
    assert first.items[0].request_id is not None
    assert r.tick(installation=m["installation"], now=NOW) == 1
    count = len(r.rows("SELECT * FROM backtest_artifact_requests"))
    changed = p.profile.model_copy(update={"revision": 2, "indicators": ()})
    second = edit(1, changed, "edit-profile")
    assert second.items[0].request_id is None
    assert r.tick(installation=m["installation"], now=NOW + timedelta(minutes=1)) == 0
    assert len(r.rows("SELECT * FROM backtest_artifact_requests")) == count
    assert r.tick(installation=m["installation"], now=NOW + timedelta(days=1)) == 1
    assert (
        r.rows(
            "SELECT snapshot FROM backtest_artifact_requests "
            "WHERE trigger='schedule' ORDER BY created_at DESC"
        )[0]["snapshot"]["profile"]["revision"]
        == 2
    )


def test_live_budget_reduction_counts_existing_files(dispatch, tmp_path):
    m = dispatch
    f, w = worker(m, tmp_path)
    # A sparse disposable file represents retained artifact bytes without allocating payload.
    ballast = f.builder.root / "retained.npy"
    with ballast.open("wb") as stream:
        stream.truncate(100_000_000)
    m["db"].execute(
        query="UPDATE backtest_artifact_installation_settings SET budget_bytes=1150000000",
        parameters={},
    )
    submit(m, (item(m),))
    assert w.run_once(m["installation"]) is None
    assert m["repo"].rows("SELECT state,reason FROM backtest_artifact_requests")[0] == {
        "state": "error",
        "reason": "resource_conflict",
    }
    assert not m["repo"].rows("SELECT * FROM backtest_artifact_builds")


@pytest.mark.parametrize("revoked", ["membership", "coordinate"])
def test_retry_rechecks_revoked_membership(dispatch, tmp_path, revoked):
    m = dispatch
    p = item(m)
    submit(m, (p,))
    r = m["repo"]
    request = r.next_request(m["installation"])
    build = r.claim_group(
        installation=m["installation"],
        request=request,
        owner=uuid4(),
        incarnation=uuid4(),
        reservation=4096,
    )
    r.finish(build=build, outcome="retry_wait", reason="transient_io")
    if revoked == "membership":
        m["db"].execute(
            query=(
                "UPDATE identity_memberships SET status='revoked' "
                "WHERE organization_id=%(o)s AND user_id=%(u)s"
            ),
            parameters={"o": m["org"], "u": m["actor"]},
        )
    else:
        from trading.contexts.backtest_artifacts.application.services.management import (
            ManagementError,
        )

        class Denied:
            def require(self, **kwargs):
                raise ManagementError("coordinate_forbidden")

        r.access = Denied()
    assert (
        r.retry(
            build=build,
            owner_token=uuid4(),
            parent_incarnation=uuid4(),
            now=NOW + timedelta(days=365),
        )
        is None
    )
    assert r.rows("SELECT state FROM backtest_artifact_builds")[0]["state"] == "error"
    assert r.rows("SELECT state FROM backtest_artifact_requests")[0]["state"] == "error"


def test_raw_ingestion_scheduler_manual_and_actual_backtest_inputs(dispatch, live_jobs, tmp_path):
    from pathlib import Path

    import numpy as np

    from apps.scheduler.backtest_artifact_publisher.wiring.managed_app import compose_managed_app
    from apps.scheduler.backtest_artifact_publisher.wiring.modules.backtest_artifact_publisher import (  # noqa: E501
        FileBacktestArtifactPublisherHostLock,
    )
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_prepare_pools_service as prep,
    )
    from tests.unit.contexts.backtest_artifacts.test_automatic_base import DURATIONS
    from tests.unit.contexts.backtest_artifacts.test_derivative_profiles import (
        IndicatorsComputeNumbaConfig,
        NumbaIndicatorCompute,
        n,
        snapshot_for,
    )
    from trading.contexts.backtest.application.dto.input_recipe import BacktestInputRecipe
    from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.stored_coordinates import (  # noqa: E501
        PostgresStoredArtifactCoordinates,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
        ArtifactRequestedRow,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_index_reader import (  # noqa: E501
        PostgresCanonicalCandleIndexReader,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_reader import (  # noqa: E501
        PostgresCanonicalCandleReader,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (  # noqa: E501
        PostgresRawKlineWriter,
    )
    from trading.contexts.market_data.application.services.stored_candle_signal import (
        StoredCandleSignalWriter,
    )

    m = dispatch
    gateway = m["source_gateway"]
    f, w = worker(m, tmp_path)
    defaults = n.YamlBacktestGridDefaultsProvider.from_yaml(
        config_path=Path("configs/dev/indicators.yaml")
    )
    runner = replace(
        w.publisher.runner,
        canonical_candle_reader=PostgresCanonicalCandleReader(gateway),
        indicator_grid_builder=n._signal_grid_builder_v2(),
        defaults_provider=defaults,
        signal_rules_engine=n.BacktestSignalRulesEngineV2(defaults_provider=defaults),
        indicator_compute=NumbaIndicatorCompute(
            defs=n.all_defs(),
            config=IndicatorsComputeNumbaConfig(
                numba_num_threads=1, numba_cache_dir=tmp_path / "jit"
            ),
        ),
    )
    clock = [NOW]
    app = compose_managed_app(
        installation_id=m["installation"],
        gateway=m["db"],
        catalog=m["catalog"],
        access=m["repo"].access,
        publisher=replace(w.publisher, runner=runner),
        stored_coordinates=PostgresStoredArtifactCoordinates(gateway),
        candle_index=PostgresCanonicalCandleIndexReader(gateway),
        root=f.builder.root,
        root_id="backtest-primary",
        root_revision=1,
        budget_bytes=10737418240,
        host_lock=FileBacktestArtifactPublisherHostLock(tmp_path / "host.lock"),
        now=lambda: clock[0],
    )

    def committed(instrument):
        app.reconciler.observe(instrument, now=clock[0])

    writer = StoredCandleSignalWriter(PostgresRawKlineWriter(gateway), committed)
    rows = n._build_canonical_rows_v2(bar_indexes=tuple(i for i in range(10080) if i != 1007))
    for offset in range(0, len(rows), 1000):
        PostgresRawKlineWriter(gateway).write_1m(rows[offset : offset + 1000])
    # Coalesced post-commit hint observes the completed bounded bulk import.
    committed(rows[0].candle.instrument_id)
    assert (
        gateway.run(
            lambda c: c.execute(
                "SELECT count(*) FROM market_data.canonical_candles_1m"
            ).fetchone()
        )[0]
        == 10079
    )
    assert app.cycle() == "observed"
    r = app.worker.repository
    assert {x["state"] for x in r.rows("SELECT state FROM backtest_artifact_requests")} == {
        "ready"
    }
    pointer = f.loader.load_current_pointer(f.coordinates)
    manifest = f.loader.load_slot_manifest(f.coordinates, pointer.active_slot)
    assert {p.timeframe for p in manifest.prices} == set(DURATIONS)
    assert not manifest.signals.manifests
    # Each materialized timeframe is checked against an independent scalar rollup.
    one = next(p for p in manifest.prices if p.timeframe == "1m")
    opens = np.load(manifest.path.parent / one.open_time.path)
    values = np.load(manifest.path.parent / one.ohlcv.path)
    for section in manifest.prices:
        duration = DURATIONS[section.timeframe] * 60000
        expected = []
        for bucket in sorted(set(opens // duration)):
            ix = np.flatnonzero(opens // duration == bucket)
            if len(ix) == DURATIONS[section.timeframe] and opens[ix[0]] == bucket * duration:
                v = values[ix]
                expected.append(
                    [
                        v[0, 0],
                        max(v[:, 1]),
                        min(v[:, 2]),
                        v[-1, 3],
                        sum(float(x) for x in v[:, 4]),
                    ]
                )
        np.testing.assert_allclose(
            np.load(manifest.path.parent / section.ohlcv.path), expected, rtol=1e-6
        )
    service = ArtifactHttpService(r, m["installation"])
    c = item(m).coordinate
    selected = profile(m["catalog"], names=("ma.sma",), tp=(), sl=())
    result = service.submit(
        m["org"],
        m["actor"],
        "integrated-enable",
        Command(
            items=(
                CommandItem(
                    coordinate=c, expected_version=0, patch=PolicyPatch(profile=selected)
                ),
            )
        ),
        "policy",
    )
    assert result.items[0].request_id is not None
    assert app.cycle() == "observed"
    pointer = f.loader.load_current_pointer(f.coordinates)
    manifest = f.loader.load_slot_manifest(f.coordinates, pointer.active_slot)
    assert {x.indicator_id for x in manifest.signals.manifests} == {"ma.sma"}
    original_hash = pointer.manifest_sha256
    original_next = r.rows("SELECT next_due FROM backtest_artifact_policies")[0]["next_due"]
    service.submit(
        m["org"],
        m["actor"],
        "integrated-manual",
        Command(items=(CommandItem(coordinate=c, expected_version=1),)),
        "update_now",
    )
    assert app.cycle() == "observed"
    assert (
        r.rows("SELECT next_due FROM backtest_artifact_policies")[0]["next_due"]
        == original_next
    )
    # New persisted tail is deferred until the next occurrence and survives reconstruction.
    writer.write_1m(n._build_canonical_rows_v2(bar_indexes=tuple(range(10080, 10140))))
    count = len(r.rows("SELECT * FROM backtest_artifact_builds"))
    assert app.cycle() == "observed"
    assert len(r.rows("SELECT * FROM backtest_artifact_builds")) == count
    clock[0] += timedelta(days=1)
    assert app.cycle() == "observed"
    pointer = f.loader.load_current_pointer(f.coordinates)
    assert pointer.manifest_sha256 != original_hash
    snapshot = snapshot_for(f, pointer.active_slot)
    _, context = prep._input_context(f, pointer.active_slot)
    refs = tuple(
        ref
        for ref in context.references
        if ref.domain.role.startswith(("prices.", "mappings."))
        and ref.domain.timeframe in ("1m", "15m")
    )
    snapshot = replace(
        snapshot,
        source_file_identities=refs,
        consumed_domains=tuple(ref.domain for ref in refs),
    )
    domain = next(ref.domain for ref in refs if ref.domain.role == "prices.15m.ohlcv")
    for indicator, provenance in [("ma.sma", "reused"), ("ma.ema", "generated")]:
        assert runner.indicator_grid_builder is not None
        grid_spec = defaults.compute_defaults(indicator_id=indicator)
        assert grid_spec is not None
        grid = runner.indicator_grid_builder.materialize_indicator(grid=grid_spec)
        chosen = selected.indicators[0].params["window"].expand()[0]
        # Canonical native grid identity: source first axis, window second in this fixture.
        axes = {a.name: a.values for a in grid.axes}
        row_id = axes["source"].index("close") * len(axes["window"]) + axes["window"].index(
            chosen
        )
        requested = (ArtifactRequestedRow(indicator, "close", row_id, (("window", chosen),)),)
        recipe = BacktestInputRecipe(
            snapshot=snapshot,
            requested_start_utc=domain.origin_utc,
            requested_end_utc=domain.end_utc,
            rows=requested,
            rule_version="signals/v1",
            defaults_sha256=runner.derivative_defaults_sha256(requested),
            compute_version="numba/v1",
            precision_version="f32-f64/v1",
            funding_policy="not_applicable",
            funding_fingerprint=None,
            tp_levels_pct=(),
            sl_levels_pct=(),
        )
        _, _, prepared = prep._prepare_inputs(
            (f, runner, snapshot, None), context, recipe, tmp_path / indicator
        )
        assert provenance in prepared.provenance
        assert recipe.snapshot.manifest_sha256 == pointer.manifest_sha256
        assert f.loader.load_current_pointer(f.coordinates) == pointer
    complete_backtest(live_jobs, f, runner, snapshot, tmp_path / "full-job")
    builds = r.rows("SELECT * FROM backtest_artifact_builds")
    assert all(x["state"] == "ready" for x in builds)
    assert all(len(x["snapshot"]["frozen_source"]["revision"]) == 64 for x in builds)


def complete_backtest(live_jobs, fixture, runner, snapshot, directory):
    """Production HTTP admission -> PG recipe/pin -> actual child -> persisted completion."""
    import os
    from dataclasses import asdict
    from datetime import UTC, datetime
    from pathlib import Path
    from uuid import UUID

    import yaml

    from apps.worker.backtest_job_runner.wiring.modules.child_ipc import BacktestChildSuccessResult
    from apps.worker.backtest_job_runner.wiring.modules.child_process import (
        BacktestChildProcessExecutor,
    )
    from tests.unit.apps.api import test_backtests_routes as api
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_backtest_preflight_service as pf,
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
        FilesystemBacktestArtifactArrayLoader,
        FilesystemBacktestArtifactContextResolver,
    )
    from trading.contexts.backtest.application.services.v2.preflight import BacktestPreflightService
    from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase

    base, repo, leases, gateway = live_jobs
    directory.mkdir()
    runtime = replace(
        pf._runtime_config(), artifact_config_hash=runner.runtime_settings.config_sha256
    )
    preflight = BacktestPreflightService(
        defaults_provider=runner.defaults_provider,
        artifact_context_resolver=FilesystemBacktestArtifactContextResolver(
            artifact_loader=fixture.loader
        ),
        runtime_config=runtime,
        artifact_array_loader=FilesystemBacktestArtifactArrayLoader(artifact_loader=fixture.loader),
        indicator_grid_builder=runner.indicator_grid_builder,
    )
    use_case = BacktestJobsUseCase(
        job_repository=repo,
        preflight_service=preflight,
        runtime_config=runtime,
        organization_scope_resolver=api._MappingScopeResolver({base.user_id: base.organization_id}),
    )
    client = api._build_client(jobs_use_case=use_case)
    request = pf._valid_request()
    request["coordinates"] = asdict(fixture.coordinates)
    domain = next(d for d in snapshot.consumed_domains if d.role == "prices.15m.ohlcv")
    request["time_range"] = {"start": domain.origin_utc, "end": domain.end_utc}
    request["indicators"] = [
        {"indicator_id": name, "sources": ["close"], "window": {"start": 21, "stop": 21, "step": 1}}
        for name in ("ma.sma", "ma.ema", "ma.wma")
    ]
    request["quality_constraints"] = {"min_closed_trades": 1}
    headers = {"x-user-id": str(base.user_id.value), "Idempotency-Key": "s09-complete"}
    response = client.post("/backtests/preflight", headers=headers, json=request)
    assert response.status_code == 200, response.text
    assert response.json()["input_readiness"]["status"] == "requires_materialization"
    response = client.post("/backtests/jobs", headers=headers, json=request)
    assert response.status_code == 201, response.text
    job_id = UUID(response.json()["job_id"])
    claimed = leases.claim_next(now=datetime.now(UTC), locked_by="S09", lease_seconds=120)
    assert claimed is not None
    admitted = preflight.validate_stored_recipe(job=claimed)
    assert admitted.input_recipe is not None
    assert admitted.input_recipe.snapshot.manifest_sha256 == snapshot.manifest_sha256
    config = yaml.safe_load(fixture.config_path.read_text())
    config["backtest_artifacts"]["artifact_root"] = str(fixture.builder.root)
    config_path = directory / "artifacts.yaml"
    config_path.write_text(yaml.safe_dump(config))
    indicators = yaml.safe_load(Path("configs/dev/indicators.yaml").read_text())
    indicators["compute"]["numba"]["numba_num_threads"] = 1
    indicators_path = directory / "indicators.yaml"
    indicators_path.write_text(yaml.safe_dump(indicators))
    executor = BacktestChildProcessExecutor(
        environ={
            **os.environ,
            "ROEHUB_ENV": "test",
            "STRATEGY_PG_DSN": gateway._dsn,
            "ROEHUB_BACKTEST_ARTIFACTS_CONFIG": str(config_path),
            "ROEHUB_INDICATORS_CONFIG": str(indicators_path),
            "ROEHUB_BACKTEST_SCRATCH_ROOT": str(directory / "scratch"),
            "ROEHUB_BACKTEST_TRADES_CACHE_ROOT": str(directory / "cache"),
            "ROEHUB_BACKTEST_CPU_CAP": "1",
            "ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS": "1",
            "ROEHUB_BACKTEST_DISK_RESERVE_BYTES": "1",
            "NUMBA_NUM_THREADS": "1",
        },
        scheduling_class="heavy",
        light_max_actual_combinations=50000,
        timeout_seconds=60,
        job_repository=repo,
    )
    result = executor.execute(
        job_id=job_id,
        job=claimed,
        locked_by="S09",
        preflight=admitted,
        updated_at=datetime.now(UTC),
    )
    assert isinstance(result, BacktestChildSuccessResult)
    finished = repo.finish_with_top_variants(
        job_id=job_id,
        organization_id=claimed.organization_id,
        user_id=claimed.user_id,
        now=datetime.now(UTC),
        locked_by="S09",
        attempt=claimed.attempt,
        next_state="succeeded",
        top_variants=result.top_variants,
    )
    assert finished is not None and finished.state == "succeeded"
    assert finished.preparation_provenance_json is not None
    assert finished.input_recipe_json["snapshot"]["manifest_sha256"] == snapshot.manifest_sha256
    assert (
        repo.get_artifact_reader(
            organization_id=base.organization_id.value, owner_id=job_id, owner_kind="job"
        )
        is None
    )
    assert not list((directory / "scratch").glob("attempt-*"))
    assert client.get(f"/backtests/jobs/{job_id}", headers=headers).json()["state"] == "succeeded"


def test_budget_increase_is_used_without_worker_restart(dispatch, tmp_path):
    from tests.unit.contexts.backtest_artifacts.test_automatic_base import START
    from trading.contexts.backtest_artifacts.application.services.management import History

    m = dispatch
    _, w = worker(m, tmp_path)
    w.storage = replace(w.storage, budget_bytes=1)
    p = item(m)
    end = START + timedelta(minutes=180)
    p = p.model_copy(
        update={
            "source": p.source.model_copy(
                update={
                    "history": History(kind="range", start_utc=START, end_utc=end),
                    "watermark": end,
                }
            )
        }
    )
    submit(m, (p,))
    result = w.run_once(m["installation"])
    assert result is not None and result.publication is not None
    assert m["repo"].rows("SELECT state FROM backtest_artifact_builds")[0]["state"] == "ready"


@pytest.mark.parametrize(
    "code,reason", [("53300", "transient_database"), ("53200", "transient_database")]
)
def test_postgres_pressure_keeps_artifact_attempt_retryable(dispatch, tmp_path, code, reason):
    """Real PG lifecycle, adapter-classified failure before publication; no source absence."""
    from psycopg.errors import lookup

    DatabaseError = lookup(code)

    from tests.unit.contexts.backtest_artifacts.test_automatic_base import Source

    m = dispatch
    submit(m, (item(m),))
    _, w = worker(m, tmp_path)

    class Pressure(Source):
        def read_1m_arrays(self, instrument_id, time_range):
            raise DatabaseError("synthetic admission refusal")

    runner = replace(w.publisher.runner, canonical_candle_reader=Pressure(180))
    failing = replace(w, publisher=replace(w.publisher, runner=runner))
    try:
        failing.run_once(m["installation"])
    except DatabaseError:
        pass
    else:
        pytest.fail(str(m["repo"].rows("SELECT state,reason FROM backtest_artifact_requests")))
    assert list(m["repo"].rows("SELECT state FROM backtest_artifact_builds")) == [
        {"state": "retry_wait"}
    ]
    assert list(m["repo"].rows("SELECT state,reason FROM backtest_artifact_requests")) == [
        {"state": "retry_wait", "reason": reason}
    ]
