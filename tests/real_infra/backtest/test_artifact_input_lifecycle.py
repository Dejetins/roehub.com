"""Forced ownership interleavings against an explicitly disposable PostgreSQL target."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any, Literal, LiteralString, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from tests.unit.contexts.backtest.application.services.v2 import (
    test_prepare_pools_service as source,
)
from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
    PostgresBacktestJobRepository,
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest.application.ports.backtest_job_repositories import (
    ArtifactOwnershipConflict,
    ArtifactReaderReservation,
)

DSN = os.environ.get("ROEHUB_BACKTEST_TEST_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="explicit disposable PostgreSQL target required")


@pytest.fixture
def ownership():
    schema = "artifact_input_test_" + uuid4().hex
    with psycopg.connect(DSN) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        conn.execute("CREATE TABLE backtest_jobs(job_id UUID PRIMARY KEY)")
        conn.execute(
            sql.SQL(
                cast(
                    LiteralString,
                    Path("migrations/postgres/0028_backtest_input_recipe_v1.sql").read_text(),
                )
            )
        )
    dsn = make_conninfo(DSN, options=f"-c search_path={schema}")
    gateway = PsycopgBacktestPostgresGateway(dsn=dsn)
    try:
        yield PostgresBacktestJobRepository(gateway=gateway), gateway, dsn
    finally:
        with psycopg.connect(DSN) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def reader(kind="job", slot="slot_a"):
    return ArtifactReaderReservation(
        "binance",
        "spot",
        "BTCUSDT",
        slot,
        1,
        "a" * 64,
        kind,
        uuid4(),
        uuid4(),
        uuid4(),
        1,
        uuid4(),
    )


def writer(repo, slot="slot_a", generation: int | None = 1, digest: str | None = "a" * 64):
    return repo.reserve_artifact_writer(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
        slot=slot,
        expected_generation=generation,
        expected_manifest_sha256=digest,
        owner_token=uuid4(),
        attempt=1,
        parent_incarnation=uuid4(),
    )


@pytest.mark.parametrize("kind", ["job", "lazy"])
def test_reader_after_precheck_excludes_destructive_reservation(ownership, kind):
    repo, gateway, _ = ownership
    old_count = gateway.fetch_one(
        query="SELECT count(*) AS n FROM backtest_artifact_slot_readers", parameters={}
    )
    assert old_count is not None and old_count["n"] == 0
    admitted = repo.reserve_artifact_reader(reader=reader(kind))
    with ThreadPoolExecutor() as pool:
        future = pool.submit(writer, repo)
        with pytest.raises(ArtifactOwnershipConflict, match="pinned"):
            future.result(timeout=5)
    assert repo.release_artifact_reader(reader=admitted)
    reservation = writer(repo)
    repo.complete_artifact_writer(writer=reservation, generation=1, manifest_sha256="a" * 64)


def test_transaction_admission_rollback_and_lock_wait(ownership):
    repo, gateway, _ = ownership
    held, attempted = Event(), Event()

    def hold_reader():
        with pytest.raises(RuntimeError, match="rollback"):
            with repo.transaction():
                repo.reserve_artifact_reader(reader=reader())
                held.set()
                assert attempted.wait(5)
                raise RuntimeError("rollback")

    with ThreadPoolExecutor() as pool:
        first = pool.submit(hold_reader)
        assert held.wait(5)
        attempted.set()
        second = pool.submit(writer, repo)
        first.result(timeout=5)
        reservation = second.result(timeout=5)
    row = gateway.fetch_one(
        query="SELECT count(*) AS n FROM backtest_artifact_slot_readers", parameters={}
    )
    assert row is not None and row["n"] == 0
    repo.complete_artifact_writer(writer=reservation, generation=1, manifest_sha256="a" * 64)


def test_writer_serializes_both_slots_and_rejects_stale_generation(ownership):
    repo, _, _ = ownership
    first = writer(repo)
    with pytest.raises(ArtifactOwnershipConflict, match="coordinate_reserved"):
        writer(repo, slot="slot_b", generation=None, digest=None)
    repo.complete_artifact_writer(writer=first, generation=2, manifest_sha256="b" * 64)
    with pytest.raises(ArtifactOwnershipConflict, match="stale_preflight"):
        writer(repo)
    with pytest.raises(ArtifactOwnershipConflict, match="stale_preflight"):
        repo.reserve_artifact_reader(reader=reader())
    admitted = repo.reserve_artifact_reader(
        reader=replace(reader(), generation=2, manifest_sha256="b" * 64)
    )
    assert admitted.epoch == first.epoch
    assert not repo.release_artifact_reader(reader=replace(admitted, attempt=2))
    assert not repo.release_artifact_reader(reader=replace(admitted, parent_incarnation=uuid4()))
    assert repo.release_artifact_reader(reader=admitted)


def test_db_session_loss_does_not_release_durable_writer(ownership):
    repo, gateway, dsn = ownership
    reserved = writer(repo)
    with pytest.raises(psycopg.OperationalError):
        with gateway.transaction():
            row = gateway.fetch_one(query="SELECT pg_backend_pid() AS pid", parameters={})
            assert row is not None
            with psycopg.connect(dsn, autocommit=True) as other:
                other.execute("SELECT pg_terminate_backend(%s)", (row["pid"],))
            gateway.fetch_one(query="SELECT 1", parameters={})
    with pytest.raises(ArtifactOwnershipConflict, match="coordinate_reserved"):
        writer(repo)
    repo.quarantine_artifact_writer(writer=reserved)
    with pytest.raises(ArtifactOwnershipConflict, match="ownership_lost"):
        repo.complete_artifact_writer(writer=reserved, generation=1, manifest_sha256="a" * 64)
    with pytest.raises(ArtifactOwnershipConflict, match="coordinate_reserved"):
        writer(repo, slot="slot_b", generation=None, digest=None)


@pytest.fixture
def published_store(tmp_path, ownership):
    from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
        build_synthetic_artifact_store_v2,
    )
    from tests.unit.contexts.backtest.application.services.v2.test_artifact_slot_publisher_v2 import (  # noqa: E501
        _write_matching_artifact_runtime_config,
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
        AtomicArtifactCurrentPointerWriterV2,
    )
    from trading.contexts.backtest.adapters.outbound.config import (
        load_backtest_artifacts_runtime_config,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_publisher import (  # noqa: E501
        BacktestArtifactSlotPublisherV2,
    )

    repo, gateway, _ = ownership
    gateway.execute(
        query="""ALTER TABLE backtest_jobs ADD COLUMN market_id INTEGER,
        ADD COLUMN symbol TEXT, ADD COLUMN artifact_slot TEXT,
        ADD COLUMN artifact_manifest_hash TEXT, ADD COLUMN state TEXT,
        ADD COLUMN execution_mode TEXT, ADD COLUMN request_json JSONB,
        ADD COLUMN spec_payload_json JSONB""",
        parameters={},
    )
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=store.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=store.builder),
        job_repository=repo,
    )
    spec = load_backtest_artifacts_runtime_config(
        _write_matching_artifact_runtime_config(tmp_path)
    ).to_validation_spec()
    return store, publisher, spec


def source_reader(store, slot, kind):
    import hashlib

    path = store.builder.slot_manifest_path(store.coordinates, slot)
    manifest = store.loader.load_slot_manifest(store.coordinates, slot)
    return replace(
        reader(kind, slot),
        generation=manifest.slot_generation,
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@pytest.mark.parametrize("kind", ["job", "lazy"])
def test_publisher_rebuild_cannot_overwrite_new_reader_after_count(
    ownership, published_store, kind
):
    from typing import Any

    import numpy as np

    repo, _, _ = ownership
    store, publisher, spec = published_store
    precheck = publisher.precheck_publish(store.coordinates)
    assert precheck.ready
    pin = repo.reserve_artifact_reader(reader=source_reader(store, store.inactive_slot, kind))
    root = precheck.inactive_manifest_path.parent
    path = next(root.rglob("*.npy"))
    before = path.read_bytes()
    mapped = np.load(path, mmap_mode="r")
    try:
        # Production publisher reaches durable reservation after the old count.
        with pytest.raises(ArtifactOwnershipConflict, match="pinned"):
            publisher.build_and_publish(
                request=cast(Any, None),
                precheck=precheck,
                precompute_runner=cast(Any, None),
                validation_spec=spec,
            )
        assert path.read_bytes() == before and mapped.size > 0
    finally:
        mapped._mmap.close()
        repo.release_artifact_reader(reader=pin)
    assert not list(root.parent.glob(".publication-*"))


def publish_prebuilt(repo, store, publisher, spec, precheck):
    current_target = source_reader(store, precheck.inactive_slot, "job")
    reserved = writer(
        repo,
        slot=precheck.inactive_slot,
        generation=current_target.generation,
        digest=current_target.manifest_sha256,
    )
    with repo.transaction():
        result = publisher.publish(
            precheck=precheck, validation_spec=spec, asof_date="2026-03-26", writer=reserved
        )
        repo.complete_artifact_writer(
            writer=reserved,
            generation=result.published_pointer.slot_generation,
            manifest_sha256=result.published_pointer.manifest_sha256,
        )
    return result


@pytest.mark.parametrize("kind", ["job", "lazy"])
def test_publisher_cleanup_cannot_delete_reader_admitted_after_old_count(
    ownership, published_store, kind
):
    import numpy as np

    repo, _, _ = ownership
    store, publisher, spec = published_store
    precheck = publisher.precheck_publish(store.coordinates)
    old = source_reader(store, store.active_slot, kind)
    assert (
        repo.count_active_for_artifact_manifest(
            market_id=1,
            symbol="BTCUSDT",
            artifact_slot=old.slot,
            artifact_manifest_hash=old.manifest_sha256,
        )
        == 0
    )
    result = publish_prebuilt(repo, store, publisher, spec, precheck)
    with ThreadPoolExecutor() as pool:
        pin = pool.submit(repo.reserve_artifact_reader, reader=old).result(timeout=5)
    root = store.builder.slot_manifest_path(store.coordinates, old.slot).parent
    path = next(root.rglob("*.npy"))
    before = path.read_bytes()
    mapped = np.load(path, mmap_mode="r")
    try:
        publisher._cleanup_previous_slot_after_publish(
            precheck=precheck, published_pointer=result.published_pointer
        )
        assert path.read_bytes() == before and mapped.size > 0
    finally:
        mapped._mmap.close()
        repo.release_artifact_reader(reader=pin)
    publisher._cleanup_previous_slot_after_publish(
        precheck=precheck, published_pointer=result.published_pointer
    )
    assert not root.exists()
    assert store.builder.slot_manifest_path(store.coordinates, store.inactive_slot).exists()


def test_two_publishers_with_same_precheck_reject_stale_target(ownership, published_store):
    from typing import Any

    repo, _, _ = ownership
    store, publisher, spec = published_store
    first = publisher.precheck_publish(store.coordinates)
    stale = publisher.precheck_publish(store.coordinates)
    published = publish_prebuilt(repo, store, publisher, spec, first)
    before = {
        str(p): p.read_bytes()
        for p in first.inactive_manifest_path.parent.rglob("*")
        if p.is_file()
    }
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            publisher.build_and_publish,
            request=cast(Any, None),
            precheck=stale,
            precompute_runner=cast(Any, None),
            validation_spec=spec,
        )
        with pytest.raises(ArtifactOwnershipConflict, match="stale"):
            future.result(timeout=5)
    assert before == {
        str(p): p.read_bytes()
        for p in first.inactive_manifest_path.parent.rglob("*")
        if p.is_file()
    }
    assert store.loader.load_current_pointer(store.coordinates) == published.published_pointer


def test_real_private_builder_publishes_under_durable_coordinate_owner(
    ownership, tmp_path, monkeypatch
):
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_artifact_precompute_runner_v2 as native,
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
        AtomicArtifactCurrentPointerWriterV2,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_publisher import (  # noqa: E501
        BacktestArtifactSlotPublisherV2,
    )

    repo, gateway, _ = ownership
    fixture = native.build_artifact_precompute_fixture_v2(tmp_path=tmp_path, price_tail_bars_1m=2)
    canonical = native._FakeCanonicalCandleReader(
        rows=native._build_canonical_rows_v2(
            bar_indexes=tuple(range(native._FULL_BUILD_MINUTES_V2))
        )
    )
    runner = native.BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=canonical,
    )
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=fixture.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=fixture.builder),
        job_repository=repo,
    )
    result = publisher.build_publish_prices_mappings_slot(
        request=replace(
            native._request_v2(fixture=fixture, end_minute=native._FULL_BUILD_MINUTES_V2),
            force_full_rebuild=True,
        ),
        precompute_runner=runner,
        validation_spec=fixture.runtime_config.to_prices_mappings_publish_validation_spec(),
    )
    assert (
        fixture.loader.load_current_pointer(fixture.coordinates)
        == result.publish_result.published_pointer
    )
    assert result.build_result.manifest_path.is_file()
    assert not list(result.build_result.manifest_path.parent.parent.glob(".publication-*"))

    original_export = native.BacktestArtifactPrecomputeRunnerV2.export_canonical_price_1m
    source_pins = []

    def inspect_incremental_pin(self, request, **kwargs):
        assert request.reuse_source_slot == result.publish_result.published_pointer.active_slot
        rows = gateway.fetch_all(
            query="SELECT * FROM backtest_artifact_slot_readers", parameters={}
        )
        assert len(rows) == 1 and rows[0]["owner_kind"] == "publisher_source"
        assert rows[0]["slot"] == request.reuse_source_slot and rows[0]["state"] == "active"
        source_pins.append(rows[0]["owner_token"])
        return original_export(self, request, **kwargs)

    monkeypatch.setattr(
        native.BacktestArtifactPrecomputeRunnerV2,
        "export_canonical_price_1m",
        inspect_incremental_pin,
    )
    incremental = publisher.build_publish_prices_mappings_slot(
        request=native._request_v2(fixture=fixture, end_minute=native._FULL_BUILD_MINUTES_V2),
        precompute_runner=runner,
        validation_spec=fixture.runtime_config.to_prices_mappings_publish_validation_spec(),
    )
    assert len(source_pins) == 1
    assert (
        incremental.publish_result.published_pointer.slot_generation
        == result.publish_result.published_pointer.slot_generation + 1
    )
    assert (
        gateway.fetch_all(query="SELECT * FROM backtest_artifact_slot_readers", parameters={}) == ()
    )
    assert not list(incremental.build_result.manifest_path.parent.parent.glob(".publication-*"))


@pytest.fixture(scope="module")
def lifecycle_database():
    from apps.migrations.storage import apply_postgres_migrations

    database = "s04_lifecycle_" + uuid4().hex
    with psycopg.connect(DSN, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    dsn = make_conninfo(DSN, dbname=database)
    try:
        apply_postgres_migrations(
            dsn, repo_root=Path.cwd(), manifest_path=Path("migrations/postgres/manifest.json")
        )
        yield dsn
    finally:
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )


@pytest.fixture
def live_jobs(lifecycle_database):
    from tests.unit.contexts.backtest.application.use_cases.test_backtest_job_worker_use_case import (  # noqa: E501
        _queued_job,
    )
    from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
        PostgresBacktestJobLeaseRepository,
    )

    job = _queued_job()
    dsn = lifecycle_database
    installation = uuid4()
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "INSERT INTO identity_users(user_id,created_at) "
            "VALUES (%s,now()) ON CONFLICT DO NOTHING",
            (job.user_id.value,),
        )
        conn.execute(
            "INSERT INTO identity_installations(installation_id,display_name,created_at) "
            "VALUES (%s,'S04 disposable',now()) ON CONFLICT DO NOTHING",
            (installation,),
        )
        installation_row = conn.execute(
            "SELECT installation_id FROM identity_installations"
        ).fetchone()
        assert installation_row is not None
        installation = installation_row[0]
        conn.execute(
            "INSERT INTO identity_organizations"
            "(organization_id,installation_id,slug,display_name,created_at) "
            "VALUES (%s,%s,'s04-test','S04 disposable',now()) ON CONFLICT DO NOTHING",
            (job.organization_id.value, installation),
        )
        conn.execute(
            "INSERT INTO identity_memberships(organization_id,user_id,role,created_at,updated_at) "
            "VALUES (%s,%s,'owner',now(),now()) ON CONFLICT DO NOTHING",
            (job.organization_id.value, job.user_id.value),
        )
    gateway = PsycopgBacktestPostgresGateway(dsn=dsn)
    repo = PostgresBacktestJobRepository(gateway=gateway)
    leases = PostgresBacktestJobLeaseRepository(gateway=gateway)
    try:
        yield job, repo, leases, gateway
    finally:
        with psycopg.connect(dsn) as conn:
            conn.execute("DELETE FROM backtest_artifact_slot_readers")
            conn.execute("DELETE FROM backtest_artifact_slot_ownership")
            conn.execute("DELETE FROM backtest_jobs")


def test_expired_live_child_is_not_reclaimed_then_reaped_attempt_recovers(live_jobs, tmp_path):
    import subprocess
    import sys
    from datetime import UTC, datetime, timedelta

    from apps.worker.backtest_job_runner.wiring.modules.compute_resources import (
        BacktestAttemptDirectory,
        BacktestScratchLimits,
        recover_attempt_directories,
    )

    job, repo, leases, gateway = live_jobs
    repo.create(job=job)
    now = datetime.now(UTC)
    claimed = leases.claim_next(now=now, locked_by="original", lease_seconds=1)
    assert claimed is not None
    pin = repo.reserve_artifact_reader(
        reader=replace(
            reader(),
            generation=4,
            owner_id=job.job_id,
            organization_id=job.organization_id.value,
            attempt=claimed.attempt,
        )
    )
    attempt = BacktestAttemptDirectory.reserve(
        root=tmp_path,
        owner={k: None if v is None else str(v) for k, v in pin.parameters().items()},
        estimated_bytes=10,
        limits=BacktestScratchLimits(reserve_bytes=1),
    )
    (attempt.path / "partial.npy").write_bytes(b"partial")
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
        pass_fds=(attempt.lock_fd,),
    )
    os.close(attempt.lock_fd)
    try:
        gateway.execute(
            query="UPDATE backtest_jobs SET lease_expires_at=now()-interval '1 second'",
            parameters={},
        )
        assert (
            leases.claim_next(
                now=now + timedelta(seconds=5), locked_by="successor", lease_seconds=60
            )
            is None
        )
        assert (
            recover_attempt_directories(root=tmp_path, reconcile_owner=repo.recover_attempt_reader)
            == 0
        )
        assert attempt.path.is_dir()
        with pytest.raises(ArtifactOwnershipConflict, match="pinned"):
            writer(repo, generation=4)
        child.communicate(b"x", timeout=5)
        assert (
            recover_attempt_directories(root=tmp_path, reconcile_owner=repo.recover_attempt_reader)
            == 1
        )
        successor = leases.claim_next(
            now=datetime.now(UTC), locked_by="successor", lease_seconds=60
        )
        assert successor is not None and successor.attempt == claimed.attempt + 1
        assert not list(tmp_path.glob("attempt-*"))
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


# Actual native prepared proof, actual DB CAS, and the production socket receiver.

built = source.built
prepared_source = source.prepared_source


@pytest.mark.parametrize("fence", ["valid", "cancel", "expired", "stale_attempt"])
def test_prepared_proof_commits_before_child_ack(live_jobs, prepared_source, tmp_path, fence):
    import socket
    import threading
    import time
    from datetime import UTC, datetime

    from apps.worker.backtest_job_runner.wiring.modules.child_ipc import (
        acknowledge_prepared_over_socket,
    )
    from apps.worker.backtest_job_runner.wiring.modules.child_process import _PreparedInputReceiver
    from trading.contexts.backtest.domain.entities import BacktestJobArtifactPin

    job, repo, leases, gateway = live_jobs
    fixture, _, snapshot, _ = prepared_source
    _, context = source._input_context(fixture, snapshot.slot)
    recipe = source._input_recipe(prepared_source, context, risk=False)
    job = replace(
        job,
        input_recipe_json=recipe.as_mapping(),
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot=cast(Literal["slot_a", "slot_b"], recipe.snapshot.slot),
            artifact_slot_generation=recipe.snapshot.generation,
            artifact_manifest_hash=recipe.snapshot.manifest_sha256,
            artifact_asof_date=context.source.slot_manifest.asof_date,
        ),
    )
    queued = replace(
        reader(),
        slot=recipe.snapshot.slot,
        generation=recipe.snapshot.generation,
        manifest_sha256=recipe.snapshot.manifest_sha256,
        organization_id=job.organization_id.value,
        owner_id=job.job_id,
        owner_token=job.job_id,
        attempt=0,
        parent_incarnation=None,
    )
    with repo.transaction():
        queued = repo.reserve_artifact_reader(reader=queued)
        repo.create(job=job)
    claimed = leases.claim_next(now=datetime.now(UTC), locked_by="parent", lease_seconds=60)
    assert claimed is not None
    pin = repo.transfer_artifact_reader(
        previous=queued,
        attempt=claimed.attempt,
        parent_incarnation=uuid4(),
        owner_token=uuid4(),
        locked_by="parent",
    )
    _, _, prepared = source._prepare_inputs(prepared_source, context, recipe, tmp_path / "inputs")
    prepared = replace(
        prepared,
        organization_id=str(job.organization_id),
        job_id=str(job.job_id),
        owner_token=str(pin.owner_token),
        attempt=pin.attempt,
    )
    if fence == "cancel":
        gateway.execute(query="UPDATE backtest_jobs SET cancel_requested_at=now()", parameters={})
    elif fence == "expired":
        gateway.execute(
            query="UPDATE backtest_jobs SET lease_expires_at=now()-interval '1 second'",
            parameters={},
        )
    elif fence == "stale_attempt":
        gateway.execute(query="UPDATE backtest_jobs SET attempt=attempt+1", parameters={})
    parent, child = socket.socketpair()
    scored = []
    errors = []

    def score_after_ack():
        try:
            acknowledge_prepared_over_socket(
                fd=child.detach(), prepared=prepared, timeout_seconds=2
            )
            persisted = gateway.fetch_one(
                query="SELECT preparation_provenance_json FROM backtest_jobs WHERE job_id=%(id)s",
                parameters={"id": job.job_id},
            )
            assert (
                persisted is not None
                and persisted["preparation_provenance_json"] == prepared.as_mapping()
            )
            scored.append(True)
        except Exception as error:
            errors.append(error)

    def persist(event):
        assert not scored
        return repo.acknowledge_prepared_inputs(
            job=claimed, reader=pin, locked_by="parent", prepared=event
        )

    receiver = _PreparedInputReceiver(parent, persist)
    worker = threading.Thread(target=score_after_ack)
    worker.start()
    try:
        deadline = time.monotonic() + 3
        while worker.is_alive() and time.monotonic() < deadline:
            try:
                receiver.poll()
            except ArtifactOwnershipConflict:
                parent.close()
                break
            worker.join(0.01)
        worker.join(3)
        assert not worker.is_alive()
        assert scored == ([True] if fence == "valid" else [])
        assert bool(errors) == (fence != "valid")
    finally:
        parent.close()
        worker.join(3)
        repo.release_artifact_reader(reader=pin)


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "signal",
        "hit_times",
        "scoring",
        "timeout",
        "crash",
        "disk_full",
        "ready_manifest",
    ],
)
def test_native_child_failure_drills_release_owned_files_and_pins(
    live_jobs, prepared_source, tmp_path, mode
):
    import hashlib
    import threading
    from dataclasses import asdict
    from datetime import UTC, datetime

    import yaml

    from apps.worker.backtest_job_runner.wiring.modules.child_ipc import BacktestChildSuccessResult
    from apps.worker.backtest_job_runner.wiring.modules.child_process import (
        BacktestChildProcessError,
        BacktestChildProcessExecutor,
    )
    from tests.unit.apps.worker.backtest_job_runner.test_child_process_executor import _preflight
    from trading.contexts.backtest.application.use_cases.backtest_job_worker import (
        BacktestJobCancellationRequested,
    )
    from trading.contexts.backtest.domain.entities import BacktestJobArtifactPin

    job, repo, leases, gateway = live_jobs
    fixture, _, snapshot, _ = prepared_source
    source._rewrite_published_inventory(prepared_source, "generated")
    _, context = source._input_context(fixture, snapshot.slot)
    risk = mode in ("hit_times", "timeout")
    recipe = source._input_recipe(prepared_source, context, risk=risk)
    request = source._normalized_request()
    request["coordinates"] = asdict(recipe.snapshot.coordinates)
    request["time_range"] = {"start": recipe.requested_start_utc, "end": recipe.requested_end_utc}
    request["indicators"] = [
        {
            "indicator_id": name,
            "sources": ["close", "open"],
            "window": {"start": 21, "stop": 100, "step": 79},
        }
        for name in ("ma.ema", "ma.sma", "ma.wma")
    ]
    request["risk"] = (
        {
            "mode": "tp_sl_grid",
            "tp": {"start_pct": 0.5, "stop_pct": 2.0, "step_pct": 1.5},
            "sl": {"start_pct": 1.0, "stop_pct": 1.0, "step_pct": 1.0},
        }
        if risk
        else {"mode": "none"}
    )
    pin_value = BacktestJobArtifactPin(
        artifact_slot=cast(Literal["slot_a", "slot_b"], snapshot.slot),
        artifact_slot_generation=recipe.snapshot.generation,
        artifact_manifest_hash=recipe.snapshot.manifest_sha256,
        artifact_asof_date=context.source.slot_manifest.asof_date,
    )
    job = replace(
        job, input_recipe_json=recipe.as_mapping(), request_json=request, artifact_pin=pin_value
    )
    queued = replace(
        reader(),
        slot=pin_value.artifact_slot,
        generation=pin_value.artifact_slot_generation,
        manifest_sha256=pin_value.artifact_manifest_hash,
        owner_id=job.job_id,
        organization_id=job.organization_id.value,
        owner_token=job.job_id,
        attempt=0,
        parent_incarnation=None,
    )
    with repo.transaction():
        repo.reserve_artifact_reader(reader=queued)
        repo.create(job=job)
    claimed = leases.claim_next(now=datetime.now(UTC), locked_by="drill-parent", lease_seconds=120)
    assert claimed is not None
    config = yaml.safe_load(fixture.config_path.read_text())
    config["backtest_artifacts"]["artifact_root"] = str(fixture.builder.root)
    config_path = tmp_path / "artifacts.yaml"
    config_path.write_text(yaml.safe_dump(config))
    indicators = yaml.safe_load(Path("configs/prod/indicators.yaml").read_text())
    indicators["compute"]["numba"]["numba_num_threads"] = 1
    for name in ("ma.ema", "ma.sma", "ma.wma"):
        indicators["defaults"][name]["inputs"]["source"] = {
            "mode": "explicit",
            "values": ["close", "open"],
        }
        indicators["defaults"][name]["params"]["window"] = {
            "mode": "explicit",
            "values": [20, 21, 50, 100],
        }
    indicators_path = tmp_path / "indicators.yaml"
    indicators_path.write_text(yaml.safe_dump(indicators))
    scratch = tmp_path / "scratch"
    marker = tmp_path / "phase"
    executor = BacktestChildProcessExecutor(
        environ={
            **os.environ,
            "ROEHUB_BACKTEST_ARTIFACTS_CONFIG": str(config_path),
            "ROEHUB_INDICATORS_CONFIG": str(indicators_path),
            "ROEHUB_BACKTEST_SCRATCH_ROOT": str(scratch),
            "ROEHUB_BACKTEST_CPU_CAP": "1",
            "ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS": "1",
            "ROEHUB_BACKTEST_DISK_RESERVE_BYTES": "1",
            "S04_DRILL_MODE": mode,
            "S04_DRILL_MARKER": str(marker),
        },
        scheduling_class="heavy",
        light_max_actual_combinations=50000,
        timeout_seconds=8 if mode == "timeout" else 60,
        child_module="tests.unit.apps.worker.backtest_job_runner.materialization_drill_child",
        job_repository=repo,
    )
    preflight = replace(
        _preflight(),
        normalized_request=request,
        input_recipe=recipe,
        artifact_metadata=replace(
            _preflight().artifact_metadata,
            artifact_slot=pin_value.artifact_slot,
            artifact_slot_generation=pin_value.artifact_slot_generation,
            artifact_manifest_hash=pin_value.artifact_manifest_hash,
            artifact_asof_date=pin_value.artifact_asof_date,
        ),
    )
    original = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in fixture.builder.root.rglob("*")
        if p.is_file()
    }
    cancellation = threading.Event()
    stop = threading.Event()

    def cancel_at_native_boundary():
        while not stop.wait(0.01):
            if marker.exists():
                cancellation.set()
                return

    canceller = threading.Thread(target=cancel_at_native_boundary)
    if mode in ("signal", "hit_times", "scoring"):
        canceller.start()
    try:
        if mode == "success":
            result = executor.execute(
                job_id=job.job_id,
                job=claimed,
                locked_by="drill-parent",
                preflight=preflight,
                updated_at=datetime.now(UTC),
                cancel_event=cancellation,
            )
            assert isinstance(result, BacktestChildSuccessResult)
            assert result.stage_timings["parent_attempt_cleanup"] >= 0
        else:
            with pytest.raises((BacktestChildProcessError, BacktestJobCancellationRequested)):
                executor.execute(
                    job_id=job.job_id,
                    job=claimed,
                    locked_by="drill-parent",
                    preflight=preflight,
                    updated_at=datetime.now(UTC),
                    cancel_event=cancellation,
                )
            assert marker.exists(), f"native {mode} boundary was not reached"
        assert not list(scratch.glob("attempt-*"))
        assert (
            repo.get_artifact_reader(
                organization_id=job.organization_id.value, owner_id=job.job_id, owner_kind="job"
            )
            is None
        )
        assert original == {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in fixture.builder.root.rglob("*")
            if p.is_file()
        }
        persisted = gateway.fetch_one(
            query="SELECT preparation_provenance_json FROM backtest_jobs WHERE job_id=%(id)s",
            parameters={"id": job.job_id},
        )
        assert persisted is not None
        assert (persisted["preparation_provenance_json"] is not None) == (
            mode in ("success", "scoring")
        )
    finally:
        stop.set()
        if canceller.is_alive():
            canceller.join(2)


@pytest.mark.parametrize("after_switch", [False, True])
@pytest.mark.parametrize("fail_recovery_fsync", [False, True])
def test_bootstrap_db_session_loss_reconciles_real_installed_candidate(
    ownership, tmp_path, monkeypatch, after_switch, fail_recovery_fsync
):
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_artifact_precompute_runner_v2 as native,
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
        AtomicArtifactCurrentPointerWriterV2,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_publisher import (  # noqa: E501
        BacktestArtifactSlotPublisherV2,
    )

    repo, gateway, dsn = ownership
    fixture = native.build_artifact_precompute_fixture_v2(tmp_path=tmp_path, price_tail_bars_1m=2)
    fixture.builder.current_pointer_path(fixture.coordinates).unlink()
    runner = native.BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=native._FakeCanonicalCandleReader(
            rows=native._build_canonical_rows_v2(
                bar_indexes=tuple(range(native._FULL_BUILD_MINUTES_V2))
            )
        ),
    )
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=fixture.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=fixture.builder),
        job_repository=repo,
    )
    real_write = AtomicArtifactCurrentPointerWriterV2.write_current_pointer_atomically

    from trading.contexts.backtest_artifacts.application.services.v2 import (
        artifact_slot_publisher as publication,
    )

    disconnected = False
    real_sync = publication._fsync_directory

    def checked_sync(path):
        if disconnected and fail_recovery_fsync:
            # No filesystem synchronization failure may publish DB availability.
            current = gateway.fetch_all(
                query="SELECT state FROM backtest_artifact_slot_ownership", parameters={}
            )
            assert any(row["state"] in ("writing", "quarantined") for row in current)
            raise OSError("injected recovery durability failure")
        return real_sync(path)

    monkeypatch.setattr(publication, "_fsync_directory", checked_sync)

    def disconnect_at_switch(self, coordinates, pointer):
        nonlocal disconnected
        if after_switch:
            real_write(self, coordinates, pointer)
        backend = gateway.fetch_one(query="SELECT pg_backend_pid() AS pid", parameters={})
        assert backend is not None
        with psycopg.connect(dsn, autocommit=True) as other:
            other.execute("SELECT pg_terminate_backend(%s)", (backend["pid"],))
        disconnected = True
        gateway.fetch_one(query="SELECT 1", parameters={})
        raise AssertionError("disconnected session must not continue")

    monkeypatch.setattr(
        AtomicArtifactCurrentPointerWriterV2,
        "write_current_pointer_atomically",
        disconnect_at_switch,
    )
    spec = fixture.runtime_config.to_prices_mappings_publish_validation_spec()
    request = replace(
        native._request_v2(fixture=fixture, end_minute=native._FULL_BUILD_MINUTES_V2),
        force_full_rebuild=True,
    )
    with pytest.raises(OSError if fail_recovery_fsync else psycopg.OperationalError):
        publisher.build_publish_prices_mappings_slot(
            request=request, precompute_runner=runner, validation_spec=spec
        )
    if fail_recovery_fsync:
        state = gateway.fetch_all(
            query="SELECT state FROM backtest_artifact_slot_ownership", parameters={}
        )
        assert any(row["state"] == "quarantined" for row in state)
        monkeypatch.setattr(publication, "_fsync_directory", real_sync)
        assert (
            publisher.recover_publications(coordinates=fixture.coordinates, validation_spec=spec)
            == 1
        )
    assert not list(
        fixture.builder.current_pointer_path(fixture.coordinates).parent.glob(".publication-*")
    )
    precheck = publisher.precheck_publish(fixture.coordinates)
    assert precheck.ready
    assert (precheck.current_pointer is not None) == after_switch
    state = gateway.fetch_all(
        query="SELECT state FROM backtest_artifact_slot_ownership", parameters={}
    )
    assert all(row["state"] == "available" for row in state)
    if after_switch:
        assert precheck.current_pointer is not None
        actual = fixture.loader.load_slot_manifest(
            fixture.coordinates, precheck.current_pointer.active_slot
        )
        assert actual.slot_generation == precheck.current_pointer.slot_generation
    else:
        assert not precheck.inactive_manifest_path.exists()


def test_async_context_copy_never_shares_connection(ownership):
    import asyncio

    _, gateway, _ = ownership

    def backend():
        row = gateway.fetch_one(query="SELECT pg_backend_pid() AS pid", parameters={})
        assert row is not None
        return row["pid"]

    async def child_task():
        return backend()

    async def scenario():
        with gateway.transaction():
            parent = backend()
            assert backend() == parent
            assert await asyncio.to_thread(backend) != parent
            assert await asyncio.create_task(child_task()) != parent

    asyncio.run(scenario())


def test_recipe_admission_and_terminal_queued_pin_release_are_atomic(live_jobs):
    from datetime import UTC, datetime

    job, repo, leases, gateway = live_jobs
    pin = replace(
        reader(),
        owner_id=job.job_id,
        organization_id=job.organization_id.value,
        attempt=0,
        parent_incarnation=None,
    )
    with pytest.raises(RuntimeError):
        with repo.transaction():
            repo.reserve_artifact_reader(reader=pin)
            repo.create(job=job)
            raise RuntimeError("admission rollback")
    assert repo.get(job_id=job.job_id, organization_id=job.organization_id) is None
    assert (
        repo.get_artifact_reader(
            organization_id=job.organization_id.value, owner_id=job.job_id, owner_kind="job"
        )
        is None
    )
    with repo.transaction():
        repo.reserve_artifact_reader(reader=pin)
        repo.create(job=job)
    cancelled = repo.cancel(
        job_id=job.job_id,
        organization_id=job.organization_id,
        user_id=job.user_id,
        cancel_requested_at=datetime.now(UTC),
    )
    assert cancelled is not None and cancelled.state == "cancelled"
    assert (
        repo.get_artifact_reader(
            organization_id=job.organization_id.value, owner_id=job.job_id, owner_kind="job"
        )
        is None
    )


def test_lazy_expired_lease_with_live_child_is_not_reclaimed(live_jobs, tmp_path):
    import subprocess
    import sys
    from datetime import UTC, datetime

    from apps.worker.backtest_job_runner.wiring.modules.compute_resources import (
        BacktestAttemptDirectory,
        BacktestScratchLimits,
        recover_attempt_directories,
    )
    from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
        PostgresBacktestLazyTradesMaterializationRepository,
    )
    from trading.contexts.backtest.application.ports import BacktestLazyTradesMaterializationRequest

    job, repo, _, gateway = live_jobs
    repo.create(job=job)
    tasks = PostgresBacktestLazyTradesMaterializationRepository(gateway=gateway)
    tasks.request_materialization(
        request=BacktestLazyTradesMaterializationRequest(
            organization_id=job.organization_id,
            owner_user_id=job.user_id,
            job_id=job.job_id,
            public_variant_key="v1",
            variant_hash="b" * 64,
            request_hash="c" * 64,
            engine_params_hash="d" * 64,
            artifact_manifest_hash="a" * 64,
            cache_key="e" * 64,
            cache_status="miss",
            ttl_seconds=60,
            requested_at=datetime.now(UTC),
        )
    )
    task = tasks.claim_next(now=datetime.now(UTC), locked_by="lazy-parent", lease_seconds=60)
    assert task is not None
    pin = repo.reserve_artifact_reader(
        reader=replace(
            reader("lazy"),
            organization_id=job.organization_id.value,
            owner_id=task.task_id,
            attempt=task.attempt,
        )
    )
    attempt = BacktestAttemptDirectory.reserve(
        root=tmp_path,
        owner={
            key: None if value is None else str(value) for key, value in pin.parameters().items()
        },
        estimated_bytes=0,
        limits=BacktestScratchLimits(reserve_bytes=1),
    )
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
        pass_fds=(attempt.lock_fd,),
    )
    os.close(attempt.lock_fd)
    try:
        gateway.execute(
            query="UPDATE backtest_lazy_trades_materializations "
            "SET lease_expires_at=now()-interval '1 second'",
            parameters={},
        )
        assert (
            tasks.claim_next(now=datetime.now(UTC), locked_by="successor", lease_seconds=60) is None
        )
        with pytest.raises(ArtifactOwnershipConflict, match="lease_lost"):
            repo.validate_attempt_lease(
                owner_kind="lazy",
                organization_id=job.organization_id.value,
                owner_id=task.task_id,
                attempt=task.attempt,
                locked_by="lazy-parent",
            )
        assert (
            recover_attempt_directories(root=tmp_path, reconcile_owner=repo.recover_attempt_reader)
            == 0
        )
        child.communicate(b"x", timeout=5)
        assert (
            recover_attempt_directories(root=tmp_path, reconcile_owner=repo.recover_attempt_reader)
            == 1
        )
        successor = tasks.claim_next(now=datetime.now(UTC), locked_by="successor", lease_seconds=60)
        assert successor is not None and successor.attempt == task.attempt + 1
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_lease_expiring_while_admission_waits_never_launches_child(
    live_jobs, prepared_source, tmp_path, monkeypatch
):
    import fcntl
    import threading
    import time
    from datetime import UTC, datetime

    import apps.worker.backtest_job_runner.wiring.modules.child_process as parent_module
    from apps.worker.backtest_job_runner.wiring.modules.child_process import (
        BacktestChildProcessExecutor,
    )
    from apps.worker.backtest_job_runner.wiring.modules.compute_resources import (
        BacktestAttemptDirectory,
    )
    from tests.unit.apps.worker.backtest_job_runner.test_child_process_executor import _preflight
    from trading.contexts.backtest.domain.entities import BacktestJobArtifactPin

    job, repo, leases, gateway = live_jobs
    fixture, _, snapshot, _ = prepared_source
    _, context = source._input_context(fixture, snapshot.slot)
    recipe = source._input_recipe(prepared_source, context, risk=False)
    job = replace(
        job,
        input_recipe_json=recipe.as_mapping(),
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot=cast(Literal["slot_a", "slot_b"], snapshot.slot),
            artifact_slot_generation=recipe.snapshot.generation,
            artifact_manifest_hash=recipe.snapshot.manifest_sha256,
            artifact_asof_date=context.source.slot_manifest.asof_date,
        ),
    )
    pin = replace(
        reader(),
        slot=snapshot.slot,
        generation=recipe.snapshot.generation,
        manifest_sha256=recipe.snapshot.manifest_sha256,
        owner_id=job.job_id,
        organization_id=job.organization_id.value,
        attempt=0,
        parent_incarnation=None,
    )
    with repo.transaction():
        repo.reserve_artifact_reader(reader=pin)
        repo.create(job=job)
    claimed = leases.claim_next(now=datetime.now(UTC), locked_by="waiting-parent", lease_seconds=1)
    assert claimed is not None
    preflight = replace(
        _preflight(),
        input_recipe=recipe,
        artifact_metadata=replace(
            _preflight().artifact_metadata,
            artifact_slot=snapshot.slot,
            artifact_slot_generation=recipe.snapshot.generation,
            artifact_manifest_hash=recipe.snapshot.manifest_sha256,
            artifact_asof_date=context.source.slot_manifest.asof_date,
        ),
    )
    entered = threading.Event()
    real_reserve = BacktestAttemptDirectory.reserve

    def observed_reserve(**kwargs):
        entered.set()
        return real_reserve(**kwargs)

    monkeypatch.setattr(BacktestAttemptDirectory, "reserve", observed_reserve)
    monkeypatch.setattr(parent_module, "_estimated_attempt_bytes", lambda **_: 0)

    def forbidden_launch(*args, **kwargs):
        raise AssertionError("expired admission must not reach child launch")

    monkeypatch.setattr(BacktestChildProcessExecutor, "_execute_child", forbidden_launch)
    root = tmp_path / "scratch"
    root.mkdir()
    executor = BacktestChildProcessExecutor(
        environ={
            "ROEHUB_BACKTEST_SCRATCH_ROOT": str(root),
            "ROEHUB_BACKTEST_DISK_RESERVE_BYTES": "1",
        },
        scheduling_class="heavy",
        light_max_actual_combinations=10,
        timeout_seconds=5,
        job_repository=repo,
    )
    with (root / ".admission.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with ThreadPoolExecutor() as pool:
            future = pool.submit(
                executor.execute,
                job_id=job.job_id,
                job=claimed,
                locked_by="waiting-parent",
                preflight=preflight,
                updated_at=datetime.now(UTC),
            )
            assert entered.wait(5)
            assert claimed.lease_expires_at is not None
            while datetime.now(UTC) <= claimed.lease_expires_at:
                time.sleep(0.01)
            fcntl.flock(held, fcntl.LOCK_UN)
            with pytest.raises(ArtifactOwnershipConflict, match="lease_lost"):
                future.result(timeout=5)
    assert not list(root.glob("attempt-*"))
    remaining = repo.get_artifact_reader(
        organization_id=job.organization_id.value, owner_id=job.job_id, owner_kind="job"
    )
    assert remaining is not None and remaining.attempt == 0


@pytest.mark.parametrize("risk", [False, True])
@pytest.mark.parametrize("futures", [False, True])
def test_durable_replay_restart_real_lazy_child_and_api_candles(
    live_jobs,
    prepared_source,
    tmp_path,
    risk,
    futures,
    monkeypatch,
):
    """C09-C12: real repository/lease, process restart, append and consumer contents."""
    from datetime import UTC, datetime

    import yaml

    from apps.api.wiring.modules.backtest import _build_jobs_use_case
    from apps.worker.backtest_job_runner.wiring.modules.lazy_trades_child_process import (
        BacktestLazyTradesChildProcessExecutor,
    )
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_backtest_preflight_service as preflight_tests,
    )
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_input_recipe_replay as replay,
    )
    from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
        PostgresBacktestLazyTradesMaterializationRepository,
    )
    from trading.contexts.backtest.application.ports import (
        BacktestLazyTradesMaterializationRequest,
        ResearchOrganizationScopeResolver,
    )
    from trading.contexts.backtest.application.services.v2.preflight import BacktestPreflightService
    from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
        BacktestPreparedArtifactSet,
    )

    base, repo, leases, gateway = live_jobs
    if futures:
        prepared_source = replay.with_futures_funding(prepared_source)
    service, proven_job, top_ten, recipe, baseline = replay.native_top_ten_fixture(
        prepared_source,
        tmp_path,
        risk=risk,
    )
    row = top_ten[0]
    job = replace(proven_job, user_id=base.user_id, attempt=0, preparation_provenance_json=None)
    pin = replace(
        reader(),
        exchange=recipe.snapshot.coordinates.exchange,
        market_type=recipe.snapshot.coordinates.market_type,
        symbol=recipe.snapshot.coordinates.symbol,
        slot=recipe.snapshot.slot,
        generation=recipe.snapshot.generation,
        manifest_sha256=recipe.snapshot.manifest_sha256,
        organization_id=job.organization_id.value,
        owner_id=job.job_id,
        owner_token=job.job_id,
        attempt=0,
        parent_incarnation=None,
    )
    with repo.transaction():
        pin = repo.reserve_artifact_reader(reader=pin)
        repo.create(job=job)
    claimed = leases.claim_next(now=datetime.now(UTC), locked_by="S05-full", lease_seconds=120)
    assert claimed is not None
    pin = repo.transfer_artifact_reader(
        previous=pin,
        attempt=claimed.attempt,
        parent_incarnation=uuid4(),
        owner_token=uuid4(),
        locked_by="S05-full",
    )
    # A new recipe job may never succeed before durable attestation.
    assert (
        repo.finish_with_top_variants(
            attempt=claimed.attempt,
            job_id=job.job_id,
            organization_id=job.organization_id,
            user_id=job.user_id,
            now=datetime.now(UTC),
            locked_by="S05-full",
            next_state="succeeded",
            top_variants=top_ten,
        )
        is None
    )
    assert proven_job.preparation_provenance_json is not None
    prepared = BacktestPreparedArtifactSet.from_mapping(proven_job.preparation_provenance_json)
    prepared = replace(prepared, attempt=claimed.attempt, owner_token=str(pin.owner_token))
    repo.acknowledge_prepared_inputs(
        job=claimed, reader=pin, locked_by="S05-full", prepared=prepared
    )
    # Same owner string cannot commit results for a different attempt.
    assert (
        repo.finish_with_top_variants(
            attempt=claimed.attempt - 1,
            job_id=job.job_id,
            organization_id=job.organization_id,
            user_id=job.user_id,
            now=datetime.now(UTC),
            locked_by="S05-full",
            next_state="succeeded",
            top_variants=top_ten,
        )
        is None
    )
    finished = repo.finish_with_top_variants(
        attempt=claimed.attempt,
        job_id=job.job_id,
        organization_id=job.organization_id,
        user_id=job.user_id,
        now=datetime.now(UTC),
        locked_by="S05-full",
        next_state="succeeded",
        top_variants=top_ten,
    )
    assert finished is not None and finished.state == "succeeded"
    with pytest.raises(ArtifactOwnershipConflict):
        repo.acknowledge_prepared_inputs(
            job=claimed, reader=pin, locked_by="S05-full", prepared=prepared
        )
    assert repo.release_artifact_reader(reader=pin)
    # Legacy nullable rows remain readable beside new rows, without inferred provenance.
    old = replace(base, job_id=uuid4())
    repo.create(job=old)
    loaded_old = repo.get(
        job_id=old.job_id, organization_id=old.organization_id, user_id=old.user_id
    )
    assert loaded_old is not None and loaded_old.input_recipe_json is None
    assert loaded_old.preparation_provenance_json is None
    old_running = leases.claim_next(now=datetime.now(UTC), locked_by="S05-legacy", lease_seconds=60)
    assert old_running is not None and old_running.job_id == old.job_id
    old_completed = repo.finish_with_top_variants(
        job_id=old.job_id,
        organization_id=old.organization_id,
        user_id=old.user_id,
        now=datetime.now(UTC),
        locked_by="S05-legacy",
        attempt=old_running.attempt,
        next_state="succeeded",
        top_variants=(),
    )
    assert old_completed is not None and old_completed.input_recipe_json is None
    assert old_completed.preparation_provenance_json is None

    fixture, runner, snapshot, _ = prepared_source
    reservation = repo.reserve_artifact_writer(
        exchange=pin.exchange,
        market_type=pin.market_type,
        symbol=pin.symbol,
        slot=pin.slot,
        expected_generation=pin.generation,
        expected_manifest_sha256=pin.manifest_sha256,
        owner_token=uuid4(),
        attempt=1,
        parent_incarnation=uuid4(),
    )
    source._rewrite_published_inventory(prepared_source, "generated")
    replay.append_source(prepared_source)
    loader, candidate = source._input_context(fixture, snapshot.slot)
    pointer_path = fixture.builder.current_pointer_path(fixture.coordinates)
    pointer = {"schema_version": 1, "published_at_utc": "2026-10-08T00:00:00Z"}
    pointer.update(
        active_slot=candidate.source.artifact_slot,
        slot_generation=candidate.source.slot_generation,
        manifest_sha256=candidate.source.artifact_manifest_hash,
        asof_date=candidate.source.artifact_asof_date,
    )
    pointer_path.write_text(yaml.safe_dump(pointer))
    repo.complete_artifact_writer(
        writer=reservation,
        generation=candidate.source.slot_generation,
        manifest_sha256=candidate.source.artifact_manifest_hash,
    )
    original_proof = finished.preparation_provenance_json
    # New repository object represents worker restart; it reads only durable documents.
    repo = PostgresBacktestJobRepository(gateway=PsycopgBacktestPostgresGateway(dsn=gateway._dsn))
    reloaded = repo.get(job_id=job.job_id, organization_id=job.organization_id, user_id=job.user_id)
    assert reloaded is not None and reloaded.preparation_provenance_json == original_proof
    stored_top_ten = repo.list_top_variants(
        job_id=job.job_id, organization_id=job.organization_id
    )
    assert len(stored_top_ten) == 10
    import math

    for stored, original in zip(stored_top_ten, top_ten, strict=True):
        # The established SQL JSON contract omits non-finite summary metrics and
        # stamps the commit time; every financial finite value and identity survives.
        finite_metrics = {
            key: value for key, value in original.summary_metrics_json.items()
            if not isinstance(value, float) or math.isfinite(value)
        }
        assert replace(stored, updated_at=original.updated_at) == replace(
            original, summary_metrics_json=finite_metrics
        )
    row = stored_top_ten[0]

    config = yaml.safe_load(fixture.config_path.read_text())
    config["backtest_artifacts"]["artifact_root"] = str(fixture.builder.root)
    config_path = tmp_path / "replay-config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    indicators = yaml.safe_load(Path("configs/prod/indicators.yaml").read_text())
    indicators["compute"]["numba"]["numba_num_threads"] = 1
    for name in ("ma.ema", "ma.sma", "ma.wma"):
        indicators["defaults"][name]["inputs"]["source"] = {
            "mode": "explicit",
            "values": ["close", "open"],
        }
        indicators["defaults"][name]["params"]["window"] = {
            "mode": "explicit",
            "values": [20, 21, 50, 100],
        }
    indicators_path = tmp_path / "replay-indicators.yaml"
    indicators_path.write_text(yaml.safe_dump(indicators))
    environ = {
        **os.environ,
        "ROEHUB_ENV": "test",
        "STRATEGY_PG_DSN": gateway._dsn,
        "ROEHUB_BACKTEST_ARTIFACTS_CONFIG": str(config_path),
        "ROEHUB_INDICATORS_CONFIG": str(indicators_path),
        "ROEHUB_BACKTEST_SCRATCH_ROOT": str(tmp_path / "scratch"),
        "ROEHUB_BACKTEST_TRADES_CACHE_ROOT": str(tmp_path / "cache"),
        "ROEHUB_BACKTEST_DISK_RESERVE_BYTES": "1",
        "NUMBA_NUM_THREADS": "1",
    }
    key = service.read_cached(
        job=reloaded, row=row, public_variant_key=row.payload_json["public_variant_key"]
    ).cache_key
    tasks = PostgresBacktestLazyTradesMaterializationRepository(gateway=gateway)
    tasks.request_materialization(
        request=BacktestLazyTradesMaterializationRequest(
            organization_id=job.organization_id,
            owner_user_id=job.user_id,
            job_id=job.job_id,
            public_variant_key=row.payload_json["public_variant_key"],
            variant_hash=row.variant_key,
            request_hash=job.request_hash,
            engine_params_hash=job.engine_params_hash,
            artifact_manifest_hash=recipe.snapshot.manifest_sha256,
            cache_key=key.digest,
            cache_status="miss",
            ttl_seconds=3600,
            requested_at=datetime.now(UTC),
        )
    )
    task = tasks.claim_next(now=datetime.now(UTC), locked_by="S05-lazy", lease_seconds=120)
    assert task is not None
    result = BacktestLazyTradesChildProcessExecutor(
        environ=environ, timeout_seconds=60, job_repository=repo
    ).execute(task=task)
    assert result.cache_status == "miss"
    assert (
        tasks.finish_completed(
            task_id=task.task_id,
            owner_user_id=job.user_id,
            now=datetime.now(UTC),
            locked_by="S05-lazy",
            attempt=task.attempt - 1,
            cache_status="stale",
            cache_path=None,
        )
        is None
    )
    assert (
        tasks.finish_completed(
            task_id=task.task_id,
            owner_user_id=job.user_id,
            now=datetime.now(UTC),
            locked_by="S05-lazy",
            attempt=task.attempt,
            cache_status="miss",
            cache_path=result.cache_path,
        )
        is not None
    )
    assert not list((tmp_path / "scratch").glob("attempt-*"))
    assert (
        gateway.fetch_one(
            query="SELECT count(*) n FROM backtest_artifact_slot_readers", parameters={}
        )["n"]
        == 0
    )
    # API composition contains the validator/resolver and durable synchronous candle pin.
    preflight = BacktestPreflightService(
        defaults_provider=runner.defaults_provider,
        artifact_context_resolver=preflight_tests._FakeArtifactResolver(),
        runtime_config=preflight_tests._runtime_config(),
    )
    api = _build_jobs_use_case(
        environ=environ,
        defaults_provider=runner.defaults_provider,
        artifact_array_loader=loader,
        preflight_service=preflight,
        runtime_config=preflight.runtime_config,
        organization_scope_resolver=cast(ResearchOrganizationScopeResolver, object()),
        job_repository=repo,
    )
    assert api is not None and api.lazy_trades_service is not None
    validator_type = replay.BacktestArtifactManifestValidatorV2
    original_attest = validator_type.attest_snapshot
    observed_pins = []

    def observe_attestation(self, *, snapshot, trusted_roots):
        observed_pins.append(
            gateway.fetch_one(
                query="SELECT count(*) n FROM backtest_artifact_slot_readers",
                parameters={},
            )["n"]
        )
        with pytest.raises(ArtifactOwnershipConflict, match="pinned"):
            repo.reserve_artifact_writer(
                exchange=pin.exchange,
                market_type=pin.market_type,
                symbol=pin.symbol,
                slot=pin.slot,
                expected_generation=candidate.source.slot_generation,
                expected_manifest_sha256=candidate.source.artifact_manifest_hash,
                owner_token=uuid4(),
                attempt=1,
                parent_incarnation=uuid4(),
            )
        return original_attest(self, snapshot=snapshot, trusted_roots=trusted_roots)

    with monkeypatch.context() as patch:
        patch.setattr(validator_type, "attest_snapshot", observe_attestation)
        candles = api.lazy_trades_service.price_candles(job=reloaded, max_bars=100)
    assert observed_pins == [1]
    assert (
        gateway.fetch_one(
            query="SELECT count(*) n FROM backtest_artifact_slot_readers", parameters={}
        )["n"]
        == 0
    )
    offline, _ = replay.replay_service(service, reloaded, prepared_source, tmp_path / "offline")
    assert candles == offline.price_candles(job=reloaded, max_bars=100)
    cache = api.lazy_trades_service.cache
    trades = cache.read_page(
        cache_key=key, now=datetime.now(UTC), ttl_seconds=3600, page=1, page_size=500
    )
    assert trades.is_hit and trades.payload is not None
    assert trades.payload["items"] == tuple(baseline.trades)
    expected_cache = replay.LocalFileBacktestLazyTradesCache(root=tmp_path / "expected-cache")
    assert isinstance(service.cache, replay.details._MemoryCache)
    expected_cache.write(
        cache_key=key, payload=service.cache.writes[0][1], now=datetime.now(UTC), ttl_seconds=3600
    )
    for method, kwargs, fields in (
        ("read_csv", {"max_rows": None}, ("content", "row_count")),
        ("read_series", {"kind": "equity", "points": 100}, ("points", "source_points")),
    ):
        actual = getattr(cache, method)(
            cache_key=key, now=datetime.now(UTC), ttl_seconds=3600, **kwargs
        )
        expected = getattr(expected_cache, method)(
            cache_key=key, now=datetime.now(UTC), ttl_seconds=3600, **kwargs
        )
        for field in fields:
            assert actual.payload[field] == expected.payload[field]
    loaded = cache.read(cache_key=key, now=datetime.now(UTC), ttl_seconds=3600)
    assert loaded.payload is not None
    if futures:
        assert baseline.chart_overlay["funding_events"]
        assert loaded.payload["funding"] == baseline.funding
    final = repo.get(job_id=job.job_id, organization_id=job.organization_id, user_id=job.user_id)
    assert final is not None
    assert final.preparation_provenance_json == original_proof
    assert final.input_recipe_json == recipe.as_mapping()


def test_input_recipe_migration_preserves_existing_queued_and_completed_rows(live_jobs):
    """Apply the real additive SQL to a pre-0028 table holding both legacy states."""
    from datetime import UTC, datetime

    job, repo, _, gateway = live_jobs
    queued = repo.create(job=job)
    finished_at = datetime.now(UTC)
    completed = replace(
        job, job_id=uuid4(), state="succeeded", finished_at=finished_at, updated_at=finished_at
    )
    repo.create(job=completed)
    schema = "s05_legacy_upgrade_" + uuid4().hex
    new_columns = ("input_recipe_json", "preparation_provenance_json")
    with psycopg.connect(gateway._dsn) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        conn.execute("CREATE TABLE backtest_jobs (LIKE public.backtest_jobs INCLUDING ALL)")
        for column in new_columns:
            conn.execute(sql.SQL("ALTER TABLE backtest_jobs DROP COLUMN {}").format(
                sql.Identifier(column)
            ))
        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema=%s AND table_name='backtest_jobs' ORDER BY ordinal_position",
            (schema,),
        ).fetchall()
        names = sql.SQL(", ").join(sql.Identifier(item[0]) for item in columns)
        conn.execute(sql.SQL("INSERT INTO backtest_jobs ({}) SELECT {} FROM public.backtest_jobs")
                     .format(names, names))
        before = conn.execute(
            "SELECT job_id, to_jsonb(j) FROM backtest_jobs j ORDER BY job_id"
        ).fetchall()
        conn.execute(sql.SQL(cast(LiteralString, Path(
            "migrations/postgres/0028_backtest_input_recipe_v1.sql"
        ).read_text())))
        after = conn.execute(
            "SELECT job_id, to_jsonb(j) - 'input_recipe_json' - 'preparation_provenance_json' "
            "FROM backtest_jobs j ORDER BY job_id"
        ).fetchall()
        assert after == before
    upgraded = PostgresBacktestJobRepository(gateway=PsycopgBacktestPostgresGateway(
        dsn=make_conninfo(gateway._dsn, options=f"-c search_path={schema},public")
    ))
    try:
        for expected in (queued, completed):
            actual = upgraded.get(
                job_id=expected.job_id, organization_id=expected.organization_id,
                user_id=expected.user_id,
            )
            assert actual is not None and actual == expected
            assert actual.input_recipe_json is None and actual.preparation_provenance_json is None
    finally:
        with psycopg.connect(gateway._dsn) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_acknowledged_crash_recovery_requires_fresh_attempt_provenance(
    live_jobs,
    prepared_source,
    tmp_path,
):
    import subprocess
    import sys
    from datetime import UTC, datetime

    from apps.worker.backtest_job_runner.wiring.modules.compute_resources import (
        BacktestAttemptDirectory,
        BacktestScratchLimits,
        recover_attempt_directories,
    )
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_input_recipe_replay as replay,
    )

    base, repo, leases, gateway = live_jobs
    service, proven, row, recipe, _ = replay.replay_fixture(prepared_source, tmp_path, risk=True)
    job = replace(proven, user_id=base.user_id, attempt=0, preparation_provenance_json=None)
    repo.create(job=job)
    original = leases.claim_next(now=datetime.now(UTC), locked_by="same-owner", lease_seconds=60)
    assert original is not None
    pin = repo.reserve_artifact_reader(
        reader=replace(
            reader(),
            slot=recipe.snapshot.slot,
            generation=recipe.snapshot.generation,
            manifest_sha256=recipe.snapshot.manifest_sha256,
            organization_id=job.organization_id.value,
            owner_id=job.job_id,
            attempt=original.attempt,
        )
    )
    _, context = source._input_context(prepared_source[0], recipe.snapshot.slot)

    def prepare(current, reservation, directory):
        assert service.derivative_builder is not None and service.input_validator is not None
        return service.prepare_pools.prepare_artifact_inputs(
            recipe=recipe,
            context=context,
            builder=service.derivative_builder,
            validator=service.input_validator,
            output_directory=directory,
            organization_id=str(current.organization_id),
            job_id=str(current.job_id),
            owner_token=str(reservation.owner_token),
            attempt=current.attempt,
            output_root_id=f"attempt-{reservation.owner_token}",
            max_generated_bytes=10_000_000,
            max_compute_bytes=10_000_000,
        )

    abandoned = BacktestAttemptDirectory.reserve(
        root=tmp_path / "scratch",
        owner={k: None if v is None else str(v) for k, v in pin.parameters().items()},
        estimated_bytes=1_000_000,
        limits=BacktestScratchLimits(reserve_bytes=1),
    )
    prepared = prepare(original, pin, abandoned.path / "inputs")
    repo.acknowledge_prepared_inputs(
        job=original, reader=pin, locked_by="same-owner", prepared=prepared
    )
    child = subprocess.Popen(
        [sys.executable, "-c", "import os; os._exit(19)"], pass_fds=(abandoned.lock_fd,)
    )
    os.close(abandoned.lock_fd)
    assert child.wait(timeout=5) == 19
    gateway.execute(
        query="UPDATE backtest_jobs SET lease_expires_at=now()-interval '1 second'", parameters={}
    )
    assert (
        recover_attempt_directories(
            root=tmp_path / "scratch", reconcile_owner=repo.recover_attempt_reader
        )
        == 1
    )
    retried = leases.claim_next(now=datetime.now(UTC), locked_by="same-owner", lease_seconds=60)
    assert retried is not None and retried.attempt == original.attempt + 1
    assert retried.preparation_provenance_json is None
    assert retried.input_recipe_json == recipe.as_mapping()
    with pytest.raises(ArtifactOwnershipConflict):
        repo.acknowledge_prepared_inputs(
            job=original, reader=pin, locked_by="same-owner", prepared=prepared
        )
    finish: dict[str, Any] = dict(
        job_id=job.job_id,
        organization_id=job.organization_id,
        user_id=job.user_id,
        locked_by="same-owner",
        next_state="succeeded",
        top_variants=(row,),
    )
    assert (
        repo.finish_with_top_variants(**finish, now=datetime.now(UTC), attempt=retried.attempt)
        is None
    )
    new_pin = repo.reserve_artifact_reader(
        reader=replace(
            pin,
            owner_token=uuid4(),
            parent_incarnation=uuid4(),
            attempt=retried.attempt,
        )
    )
    ready = prepare(retried, new_pin, tmp_path / "retry-inputs")
    repo.acknowledge_prepared_inputs(
        job=retried, reader=new_pin, locked_by="same-owner", prepared=ready
    )
    assert (
        repo.finish_with_top_variants(**finish, now=datetime.now(UTC), attempt=original.attempt)
        is None
    )
    completed = repo.finish_with_top_variants(
        **finish, now=datetime.now(UTC), attempt=retried.attempt
    )
    assert completed is not None and completed.state == "succeeded"
    assert completed.preparation_provenance_json == ready.as_mapping()
    assert repo.release_artifact_reader(reader=new_pin)


@pytest.mark.parametrize("subprocess", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("risk", [False, True])
def test_s06_api_recipe_creation_idempotency_and_complete_top(
    live_jobs, built, tmp_path, partial, risk, subprocess,
):
    """Normal HTTP creation -> atomic recipe/pin -> native preparation/ACK -> durable top10."""
    import hashlib
    import shutil
    from dataclasses import asdict
    from datetime import UTC, datetime
    from uuid import UUID

    import numba as nb
    import yaml

    from tests.unit.apps.api import test_backtests_routes as api
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_backtest_preflight_service as preflight_tests,
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
        FilesystemBacktestArtifactArrayLoader,
        FilesystemBacktestArtifactContextResolver,
    )
    from trading.contexts.backtest.application.dto import (
        BacktestComboPlanningConfig,
        BacktestPreparePoolsConfig,
    )
    from trading.contexts.backtest.application.dto.artifact_inputs import BacktestAttemptInputs
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
    from trading.contexts.backtest.application.services.v2.preflight import BacktestPreflightService
    from trading.contexts.backtest.application.services.v2.prepare_pools import (
        BacktestPreparePoolsService,
    )
    from trading.contexts.backtest.application.services.v2.tp_sl_exact import (
        BacktestTpSlExactScoringService,
    )
    from trading.contexts.backtest.application.services.v2.tp_sl_hit_times import (
        BacktestTpSlHitTimesService,
    )
    from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_validator import (  # noqa: E501
        BacktestArtifactManifestValidatorV2,
    )

    base, repo, leases, gateway = live_jobs
    fixture, runner, snapshot, export = built
    loader = replace(fixture.loader, path_resolver=replace(
        fixture.builder, root=tmp_path / "store",
    ))
    runner = replace(runner, artifact_loader=loader, runtime_settings=replace(
        runner.runtime_settings,
        signal_artifacts=() if partial else runner.runtime_settings.signal_artifacts,
        precompute_hit_times=not partial,
    ))
    published = runner.export_canonical_price_1m(replace(
        export, target_slot="slot_a", target_slot_generation=1,
        asof_date=snapshot.consumed_domains[0].end_utc[:10], force_full_rebuild=True,
    ))
    pointer = loader.resolve_current_pointer_path(fixture.coordinates)
    pointer.write_text(yaml.safe_dump({
        "schema_version": 1, "active_slot": published.slot,
        "slot_generation": published.slot_generation, "asof_date": published.asof_date,
        "manifest_sha256": published.manifest_sha256,
        "published_at_utc": "2026-03-31T02:00:00Z",
    }))
    arrays = FilesystemBacktestArtifactArrayLoader(artifact_loader=loader)
    runtime_config = replace(
        preflight_tests._runtime_config(),
        artifact_config_hash=runner.runtime_settings.config_sha256,
        hit_times_tp_levels_pct=runner.runtime_settings.hit_times_tp_levels_pct,
        hit_times_sl_levels_pct=runner.runtime_settings.hit_times_sl_levels_pct,
    )
    preflight = BacktestPreflightService(
        defaults_provider=runner.defaults_provider,
        artifact_context_resolver=FilesystemBacktestArtifactContextResolver(artifact_loader=loader),
        runtime_config=runtime_config, artifact_array_loader=arrays,
        indicator_grid_builder=runner.indicator_grid_builder,
    )
    scope = api._MappingScopeResolver({base.user_id: base.organization_id})
    use_case = BacktestJobsUseCase(job_repository=repo, preflight_service=preflight,
                                  runtime_config=runtime_config, organization_scope_resolver=scope)
    client = api._build_client(jobs_use_case=use_case)
    request = preflight_tests._valid_request()
    request["coordinates"] = asdict(fixture.coordinates)
    request["time_range"] = {"start": snapshot.consumed_domains[0].origin_utc,
                             "end": snapshot.consumed_domains[0].end_utc}
    request["indicators"] = [{"indicator_id": name, "sources": ["close", "open"],
                              "window": {"start": 21, "stop": 100, "step": 79}}
                             for name in ("ma.ema", "ma.sma", "ma.wma")]
    request["quality_constraints"] = {"min_closed_trades": 1}
    if risk:
        request["risk"] = {
            "mode": "tp_sl_grid",
            "tp": {"start_pct": 0.5, "stop_pct": 2.0, "step_pct": 1.5},
            "sl": {"start_pct": 1.0, "stop_pct": 1.0, "step_pct": 1.0},
        }
    headers = {"x-user-id": str(base.user_id.value), "Idempotency-Key": "s06-create"}
    response = client.post("/backtests/preflight", headers=headers, json=request)
    assert response.status_code == 200, response.text
    assert response.json()["input_readiness"]["status"] == (
        "requires_materialization" if partial else "ready"
    )
    assert str(tmp_path) not in response.text
    first = client.post("/backtests/jobs", headers=headers, json=request)
    assert first.status_code == 201, first.text
    second = client.post("/backtests/jobs", headers=headers, json=request)
    assert second.status_code in (200, 201), second.text
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["idempotent_replay"]
    job_id = UUID(first.json()["job_id"])
    claimed = leases.claim_next(now=datetime.now(UTC), locked_by="S06", lease_seconds=120)
    assert claimed is not None and claimed.job_id == job_id
    assert claimed.input_recipe_json is not None
    frozen = preflight.validate_stored_recipe(job=claimed)
    assert frozen.input_recipe is not None
    pin = None
    attempts = None
    environ = None
    if subprocess:
        from apps.worker.backtest_job_runner.wiring.modules.child_ipc import (
            BacktestChildSuccessResult,
        )
        from apps.worker.backtest_job_runner.wiring.modules.child_process import (
            BacktestChildProcessExecutor,
        )

        config = yaml.safe_load(fixture.config_path.read_text())
        config["backtest_artifacts"]["artifact_root"] = str(loader.path_resolver.root)
        config_path = tmp_path / "child-artifacts.yaml"
        config_path.write_text(yaml.safe_dump(config))
        indicators = yaml.safe_load(Path("configs/prod/indicators.yaml").read_text())
        indicators["compute"]["numba"]["numba_num_threads"] = 1
        for name in ("ma.ema", "ma.sma", "ma.wma"):
            indicators["defaults"][name]["inputs"]["source"] = {
                "mode": "explicit", "values": ["close", "open"],
            }
            indicators["defaults"][name]["params"]["window"] = {
                "mode": "explicit", "values": [20, 21, 50, 100],
            }
        indicators_path = tmp_path / "child-indicators.yaml"
        indicators_path.write_text(yaml.safe_dump(indicators))
        environ = {
            **os.environ,
            "ROEHUB_ENV": "test",
            "STRATEGY_PG_DSN": gateway._dsn,
            "ROEHUB_BACKTEST_ARTIFACTS_CONFIG": str(config_path),
            "ROEHUB_INDICATORS_CONFIG": str(indicators_path),
            "ROEHUB_BACKTEST_SCRATCH_ROOT": str(tmp_path / "scratch"),
            "ROEHUB_BACKTEST_TRADES_CACHE_ROOT": str(tmp_path / "cache"),
            "ROEHUB_BACKTEST_CPU_CAP": "1",
            "ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS": "1",
            "ROEHUB_BACKTEST_DISK_RESERVE_BYTES": "1",
            "NUMBA_NUM_THREADS": "1",
        }
        result = BacktestChildProcessExecutor(
            environ=environ, scheduling_class="heavy", light_max_actual_combinations=50000,
            timeout_seconds=60, job_repository=repo,
        ).execute(job_id=job_id, job=claimed, locked_by="S06", preflight=frozen,
                  updated_at=datetime.now(UTC))
        assert isinstance(result, BacktestChildSuccessResult)
        assert not list((tmp_path / "scratch").glob("attempt-*"))
        assert repo.get_artifact_reader(
            organization_id=base.organization_id.value, owner_id=job_id, owner_kind="job",
        ) is None
    else:
        pin = replace(reader(), exchange=fixture.coordinates.exchange,
                      market_type=fixture.coordinates.market_type,
                      symbol=fixture.coordinates.symbol,
                      slot=published.slot, generation=published.slot_generation,
                      manifest_sha256=published.manifest_sha256,
                      organization_id=base.organization_id.value,
                      owner_id=job_id, owner_token=job_id, attempt=0, parent_incarnation=None)
        pin = repo.transfer_artifact_reader(previous=pin, attempt=claimed.attempt,
                                           parent_incarnation=uuid4(), owner_token=uuid4(),
                                           locked_by="S06")
        attempts = tmp_path / "attempt"

        def acknowledge(prepared):
            return repo.acknowledge_prepared_inputs(job=claimed, reader=pin, locked_by="S06",
                                                   prepared=prepared)

        runtime = BacktestRuntimeJobOrchestrationService(
            prepare_pools=BacktestPreparePoolsService(
                artifact_array_loader=arrays, defaults_provider=runner.defaults_provider,
                config=BacktestPreparePoolsConfig(row_prefilter_top_fraction=1.0),
            ),
            compute_policy=BacktestComputePolicy(threads=BacktestNumbaThreadDecision(
                num_threads=nb.get_num_threads(), source="s06_fixture_budget",
            )),
            combo_planning=BacktestComboPlanningService(config=BacktestComboPlanningConfig(
                combo_top_frac=1.0, combo_min_confirm=1,
            )),
            no_risk_exact=BacktestNoRiskExactScoringService(), artifact_array_loader=arrays,
            tp_sl_hit_times=BacktestTpSlHitTimesService(artifact_array_loader=arrays),
            tp_sl_exact=BacktestTpSlExactScoringService(),
            derivative_builder=runner,
            input_validator=BacktestArtifactManifestValidatorV2(artifact_loader=loader),
            attempt_inputs=BacktestAttemptInputs(
                attempts, str(base.organization_id.value), str(job_id), str(pin.owner_token),
                claimed.attempt, 2**30, 2**30, acknowledge,
            ),
        )
        result = runtime.execute(job_id=job_id, preflight=frozen, updated_at=datetime.now(UTC))
    assert len(result.top_variants) == 10
    finished = repo.finish_with_top_variants(
        job_id=job_id, organization_id=claimed.organization_id, user_id=claimed.user_id,
        now=datetime.now(UTC), locked_by="S06", attempt=claimed.attempt,
        next_state="succeeded", top_variants=result.top_variants,
    )
    assert finished is not None and finished.preparation_provenance_json is not None
    if not subprocess:
        assert pin is not None and attempts is not None
        assert repo.release_artifact_reader(reader=pin)
        assert sum(p.stat().st_size for p in attempts.rglob("*.npy")) <= (
            response.json()["input_readiness"]["estimated_generated_bytes_upper_bound"]
        )
        shutil.rmtree(attempts)
        assert not attempts.exists()
    top = client.get(f"/backtests/jobs/{job_id}/top", headers=headers)
    assert top.status_code == 200, top.text
    assert len(top.json()["items"]) == 10
    assert hashlib.sha256(published.manifest_path.read_bytes()).hexdigest() == (
        published.manifest_sha256
    )
    status = client.get(f"/backtests/jobs/{job_id}", headers=headers)
    assert status.status_code == 200, status.text
    assert status.json()["state"] == "succeeded"
    for item in top.json()["items"]:
        detail = client.get(
            f"/backtests/jobs/{job_id}/variants/{item['variant_key']}", headers=headers,
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["variant_hash"] == item["variant_hash"]
        assert str(tmp_path) not in detail.text
    if subprocess:
        assert environ is not None
        from apps.api.wiring.modules.backtest import _build_jobs_use_case
        from apps.worker.backtest_job_runner.wiring.modules.lazy_trades_child_process import (
            BacktestLazyTradesChildProcessExecutor,
        )
        from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
            PostgresBacktestLazyTradesMaterializationRepository,
        )
        from trading.shared_kernel.primitives import OrganizationId, UserId

        other_user = UserId(uuid4())
        other_org = OrganizationId(uuid4())
        with psycopg.connect(gateway._dsn) as connection:
            connection.execute("INSERT INTO identity_users(user_id,created_at) VALUES (%s,now())",
                               (other_user.value,))
            connection.execute(
                "INSERT INTO identity_organizations "
                "(organization_id,installation_id,slug,display_name,created_at) "
                "SELECT %s,installation_id,%s,'S07 other organization',now() "
                "FROM identity_organizations WHERE organization_id=%s",
                (other_org.value, "s07-" + other_org.value.hex[:16], base.organization_id.value),
            )
            connection.execute(
                "INSERT INTO identity_memberships "
                "(organization_id,user_id,role,created_at,updated_at) "
                "VALUES (%s,%s,'owner',now(),now())", (other_org.value, other_user.value),
            )
        scope = api._MappingScopeResolver({base.user_id: base.organization_id,
                                          other_user: other_org})
        wired = _build_jobs_use_case(
            environ=environ, defaults_provider=runner.defaults_provider,
            artifact_array_loader=arrays, preflight_service=preflight,
            runtime_config=runtime_config, organization_scope_resolver=scope,
            job_repository=repo,
        )
        assert wired is not None
        client = api._build_client(jobs_use_case=wired)
        other_headers = {"x-user-id": str(other_user), "Idempotency-Key": "s06-create"}
        other_created = client.post("/backtests/jobs", headers=other_headers, json=request)
        assert other_created.status_code == 201, other_created.text
        other_id = other_created.json()["job_id"]
        assert other_id != str(job_id)
        assert other_created.json()["request_hash"] == first.json()["request_hash"]
        assert client.get(f"/backtests/jobs/{other_id}", headers=other_headers).status_code == 200
        assert client.get(f"/backtests/jobs/{other_id}", headers=headers).status_code == 404
        other_replay = client.post("/backtests/jobs", headers=other_headers, json=request)
        assert other_replay.json()["job_id"] == other_id
        assert other_replay.json()["idempotent_replay"]
        # Cancel the queued second-org attempt through its normal route, releasing its pin.
        canceled = client.post(f"/backtests/jobs/{other_id}/cancel", headers=other_headers)
        assert canceled.status_code == 200, canceled.text
        variant = top.json()["items"][0]["variant_key"]
        url = f"/backtests/jobs/{job_id}/variants/{variant}"
        queued = client.post(url + "/trades", headers=headers)
        assert queued.status_code == 202, queued.text
        tasks = PostgresBacktestLazyTradesMaterializationRepository(gateway=gateway)
        task = tasks.claim_next(now=datetime.now(UTC), locked_by="S07-lazy", lease_seconds=120)
        assert task is not None
        lazy_result = BacktestLazyTradesChildProcessExecutor(
            environ=environ, timeout_seconds=60, job_repository=repo,
        ).execute(task=task)
        assert tasks.finish_completed(
            task_id=task.task_id, owner_user_id=task.owner_user_id, now=datetime.now(UTC),
            locked_by="S07-lazy", attempt=task.attempt, cache_status="miss",
            cache_path=lazy_result.cache_path,
        ) is not None
        for suffix in ("/trades", "/equity", "/drawdown", "/candles", "/trades.csv"):
            response = client.get(url + suffix, headers=headers)
            assert response.status_code == 200, response.text
            assert str(tmp_path) not in response.text
            assert response.content
        details = client.post(url + "/trades", headers=headers)
        assert details.status_code == 200, details.text
        # POST returns cache metadata; the paginated GET owns actual trade rows.
        assert details.json()["cache"]["status"] == "hit"
        page = client.get(url + "/trades", headers=headers)
        assert page.json()["items"]
        tasks_before_denied_reads = gateway.fetch_one(
            query="SELECT count(*) n FROM backtest_lazy_trades_materializations", parameters={},
        )
        for method, endpoint in (
            ("get", f"/backtests/jobs/{job_id}"),
            ("get", f"/backtests/jobs/{job_id}/top"),
            ("get", url), ("post", url + "/trades"),
            ("get", url + "/equity"), ("get", url + "/candles"),
            ("get", url + "/trades.csv"), ("get", url + "/trades"),
        ):
            denied = getattr(client, method)(endpoint, headers={"x-user-id": str(other_user)})
            assert denied.status_code == 404, denied.text
            assert denied.json()["error"]["code"] == "backtest.not_found"
            assert str(tmp_path) not in denied.text
        assert gateway.fetch_one(
            query="SELECT count(*) n FROM backtest_lazy_trades_materializations", parameters={},
        ) == tasks_before_denied_reads
        failed_variant = top.json()["items"][1]["variant_key"]
        failed_url = f"/backtests/jobs/{job_id}/variants/{failed_variant}"
        assert client.post(failed_url + "/trades", headers=headers).status_code == 202
        failed_task = tasks.claim_next(
            now=datetime.now(UTC), locked_by="S07-failed", lease_seconds=120,
        )
        assert failed_task is not None
        diagnostic = f"child stderr: {tmp_path}/private/attempt.npy traceback"
        diagnostic_json = {"stderr": diagnostic, "details": {"retryable": False}}
        stored_failure = tasks.finish_failed(
            task_id=failed_task.task_id, owner_user_id=failed_task.owner_user_id,
            now=datetime.now(UTC), locked_by="S07-failed", attempt=failed_task.attempt,
            last_error=diagnostic, last_error_json=diagnostic_json,
        )
        assert stored_failure is not None
        assert stored_failure.last_error == diagnostic
        assert stored_failure.last_error_json == diagnostic_json
        for method, suffix in (
            ("get", "/trades"), ("post", "/trades"), ("get", "/equity"),
            ("get", "/drawdown"), ("get", "/trades.csv"),
        ):
            failed_response = getattr(client, method)(failed_url + suffix, headers=headers)
            assert failed_response.status_code == 202, failed_response.text
            assert str(tmp_path) not in failed_response.text
            assert "traceback" not in failed_response.text
            metadata = failed_response.json()["materialization"]
            assert metadata["status"] == "failed"
            assert metadata["retryable"] is False
            assert metadata["last_error"] == "Result preparation failed."
            assert metadata["last_error_json"] == {"code": "backtest.materialization_failed"}
        # Candle reads are independent of lazy trade materialization.
        failed_candles = client.get(failed_url + "/candles", headers=headers)
        assert failed_candles.status_code == 200
        assert str(tmp_path) not in failed_candles.text
        persisted_failure = gateway.fetch_one(
            query="SELECT last_error,last_error_json FROM backtest_lazy_trades_materializations "
                  "WHERE task_id=%(task_id)s",
            parameters={"task_id": failed_task.task_id},
        )
        assert persisted_failure["last_error"] == diagnostic
        assert persisted_failure["last_error_json"] == diagnostic_json
        assert not list((tmp_path / "scratch").glob("attempt-*"))
        assert gateway.fetch_one(
            query="SELECT count(*) n FROM backtest_artifact_slot_readers", parameters={},
        )["n"] == 0


@pytest.mark.parametrize("mutation", ["pointer", "target"])
def test_publication_final_revalidation_rejects_post_build_identity_change(
    live_jobs, tmp_path, monkeypatch, mutation,
):
    """A valid private build cannot install over an identity changed after reservation."""
    import yaml

    from tests.unit.contexts.backtest.application.services.v2 import (
        test_artifact_precompute_runner_v2 as native,
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
        AtomicArtifactCurrentPointerWriterV2,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_publisher import (  # noqa: E501
        BacktestArtifactSlotPublisherV2,
    )

    _, repo, _, gateway = live_jobs
    fixture = native.build_artifact_precompute_fixture_v2(tmp_path=tmp_path, price_tail_bars_1m=2)
    runner = native.BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings, artifact_loader=fixture.loader,
        canonical_candle_reader=native._FakeCanonicalCandleReader(
            rows=native._build_canonical_rows_v2(
                bar_indexes=tuple(range(native._FULL_BUILD_MINUTES_V2)),
            ),
        ),
    )
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=fixture.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=fixture.builder),
        job_repository=repo,
    )
    runner.export_canonical_price_1m(replace(
        native._request_v2(fixture=fixture, end_minute=native._FULL_BUILD_MINUTES_V2),
        target_slot="slot_b", target_slot_generation=5, force_full_rebuild=True,
    ))
    precheck = publisher.precheck_publish(fixture.coordinates)
    target = precheck.inactive_manifest_path
    before = {p: p.read_bytes() for p in target.parent.rglob("*") if p.is_file()}
    pointer = fixture.builder.current_pointer_path(fixture.coordinates)
    real_export = native.BacktestArtifactPrecomputeRunnerV2.export_canonical_price_1m
    changed = []

    def mutate_after_complete_build(self, request, **kwargs):
        result = real_export(self, request, **kwargs)
        assert result.manifest_path != target
        if mutation == "pointer":
            document = yaml.safe_load(pointer.read_text())
            document["published_at_utc"] = "2026-10-08T12:00:00Z"
            pointer.write_text(yaml.safe_dump(document))
            changed.append(pointer.read_bytes())
        else:
            target.write_bytes(target.read_bytes() + b"\n# changed after private build\n")
            changed.append(target.read_bytes())
        return result

    monkeypatch.setattr(native.BacktestArtifactPrecomputeRunnerV2,
                        "export_canonical_price_1m", mutate_after_complete_build)
    with pytest.raises(ArtifactOwnershipConflict, match="stale"):
        publisher.build_and_publish(
            request=replace(
                native._request_v2(fixture=fixture, end_minute=native._FULL_BUILD_MINUTES_V2),
                force_full_rebuild=True, target_slot=precheck.inactive_slot,
                target_slot_generation=precheck.target_slot_generation,
            ),
            precheck=precheck, precompute_runner=runner,
            validation_spec=fixture.runtime_config.to_prices_mappings_publish_validation_spec(),
        )
    assert len(changed) == 1
    assert (pointer if mutation == "pointer" else target).read_bytes() == changed[0]
    for path, content in before.items():
        if mutation != "target" or path != target:
            assert path.read_bytes() == content
    assert not list(target.parent.parent.glob(".publication-*"))
    assert gateway.fetch_all(
        query="SELECT * FROM backtest_artifact_slot_readers", parameters={},
    ) == ()
    assert all(row["state"] == "available" for row in gateway.fetch_all(
        query="SELECT state FROM backtest_artifact_slot_ownership", parameters={},
    ))
