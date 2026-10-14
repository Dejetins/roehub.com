"""Actual PostgreSQL transactions and isolation; explicit disposable target required."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import LiteralString, cast
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from tests.unit.contexts.backtest_artifacts.test_management_profiles import catalog  # noqa: F401
from trading.contexts.backtest_artifacts.adapters.outbound.config.backtest_artifacts_runtime_config import (  # noqa: E501
    load_backtest_artifacts_runtime_config,
)
from trading.contexts.backtest_artifacts.adapters.outbound.config.legacy_management_import import (
    legacy_demand,
)
from trading.contexts.backtest_artifacts.adapters.outbound.defaults import (
    YamlBacktestGridDefaultsProvider,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.gateway import (
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.management_repository import (  # noqa: E501
    PostgresArtifactManagementRepository,
)
from trading.contexts.backtest_artifacts.application.services.management import (
    Coordinate,
    History,
    ManagementError,
    PolicyItem,
    Progress,
    Schedule,
    SourceTarget,
)

DSN = os.environ.get("ROEHUB_BACKTEST_TEST_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="explicit disposable PostgreSQL target required")
NOW = datetime(2026, 10, 8, tzinfo=UTC)


class Access:
    def require(self, *, organization_id, actor_id, coordinate):
        if coordinate.symbol not in ("BTCUSDT", "ETHUSDT") or coordinate.exchange != "binance":
            raise ManagementError("coordinate_forbidden")


@pytest.fixture
def managed(catalog):  # noqa: F811
    from tests.real_infra.market_data.pg_source_fixture import market_source

    with market_source(DSN) as source_gateway:
        schema = "artifact_management_test_" + uuid4().hex
        installation, org, other, actor, viewer, admin = [uuid4() for _ in range(6)]
        with psycopg.connect(source_gateway._dsn) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
            conn.execute("""CREATE TABLE identity_installations(installation_id UUID PRIMARY KEY);
                CREATE TABLE identity_users(user_id UUID PRIMARY KEY);
                CREATE TABLE identity_organizations(organization_id UUID PRIMARY KEY,
                  installation_id UUID NOT NULL REFERENCES identity_installations,status TEXT,
                  UNIQUE(installation_id,organization_id));
                CREATE TABLE identity_memberships(organization_id UUID,user_id UUID,
                    role TEXT,status TEXT);
                CREATE TABLE identity_installation_owners(installation_id UUID,user_id UUID);
                CREATE TABLE backtest_jobs(job_id UUID PRIMARY KEY);""")
            conn.execute("INSERT INTO identity_installations VALUES (%s)", (installation,))
            for user in (actor, viewer, admin):
                conn.execute("INSERT INTO identity_users VALUES (%s)", (user,))
            for tenant in (org, other):
                conn.execute(
                    "INSERT INTO identity_organizations VALUES (%s,%s,'active')",
                    (tenant, installation),
                )
                for user, role in ((actor, "owner"), (viewer, "viewer"), (admin, "admin")):
                    conn.execute(
                        "INSERT INTO identity_memberships VALUES (%s,%s,%s,'active')",
                        (tenant, user, role),
                    )
            conn.execute(
                "INSERT INTO identity_installation_owners VALUES (%s,%s)", (installation, actor)
            )
            for name in (
                "0028_backtest_input_recipe_v1.sql",
                "0029_backtest_artifact_management_v1.sql",
                "0030_backtest_artifact_dispatch_v1.sql",
            ):
                conn.execute(
                    sql.SQL(cast(LiteralString, (Path("migrations/postgres") / name).read_text()))
                )
        dsn = make_conninfo(source_gateway._dsn, options=f"-c search_path={schema}")
        db = PsycopgBacktestPostgresGateway(dsn=dsn)
        repo = PostgresArtifactManagementRepository(gateway=db, catalog=catalog, access=Access())
        schedule = Schedule(
            interval_days=1, timezone="Europe/Moscow", local_time=time(2), anchor=date(2026, 10, 8)
        )
        repo.initialize_settings(
            organization_id=org, actor_id=actor, root_id="backtest-primary", schedule=schedule
        )
        # Test-only activation: does not invoke any scheduler or change installation config.
        db.execute(
            query="""UPDATE backtest_artifact_installation_settings SET
            capability_enabled=TRUE,paused=FALSE,legacy_handoff='complete',budget_bytes=10737418240""",
            parameters={},
        )
        try:
            yield dict(
                source_gateway=source_gateway,
                repo=repo,
                db=db,
                dsn=dsn,
                installation=installation,
                org=org,
                other=other,
                actor=actor,
                viewer=viewer,
                admin=admin,
                schedule=schedule,
                catalog=catalog,
            )
        finally:
            with psycopg.connect(source_gateway._dsn) as conn:
                conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def item(m, symbol="BTCUSDT", *, version=0, profile=None, source=True):
    from tests.real_infra.market_data.test_postgres_readers_frontier import candle
    from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.publication_fence import (  # noqa: E501
        PostgresPublicationFence,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (  # noqa: E501
        PostgresRawKlineWriter,
    )

    coordinate = Coordinate(exchange="binance", market_type="spot", symbol=symbol)
    if source and not m["source_gateway"].run(
        lambda c: c.execute(
            "SELECT EXISTS(SELECT 1 FROM market_data.canonical_candles_1m "
            "WHERE market_id=1 AND symbol=%s)",
            (symbol,),
        ).fetchone()[0]
    ):
        PostgresRawKlineWriter(m["source_gateway"]).write_1m(
            [candle(symbol, i, base=datetime(2026, 10, 1, tzinfo=UTC)) for i in range(180)]
        )
    return PolicyItem(
        coordinate=Coordinate(exchange="binance", market_type="spot", symbol=symbol),
        expected_version=version,
        profile=profile or m["catalog"].default_profile(),
        schedule=m["schedule"],
        source=SourceTarget(
            history=History(kind="range", start_utc=datetime(2026, 10, 1, tzinfo=UTC), end_utc=NOW),
            revision="fixture-revision",
            **PostgresPublicationFence(m["db"]).capture(coordinate),
            watermark=NOW,
            compute_identity="v2",
            funding_identity="spot",
        )
        if source
        else None,
    )


def submit(m, items, key="command-0001", **overrides):
    args = dict(organization_id=m["org"], actor_id=m["actor"], items=items, key=key)
    args.update(overrides)
    return m["repo"].submit(**args)


def count(m, table):
    return m["db"].fetch_one(query=f"SELECT count(*) AS n FROM {table}", parameters={})["n"]


def claim(m, request):
    return m["repo"].claim(
        installation_id=m["installation"],
        request_id=UUID(request),
        owner_token=uuid4(),
        parent_incarnation=uuid4(),
        reservation_bytes=1024,
    )


def test_bulk_replay_concurrent_restart_and_body_conflict(managed):
    m = managed
    items = (item(m), item(m, "ETHUSDT"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: submit(m, items), range(4)))
    assert results == [results[0]] * 4
    assert count(m, "backtest_artifact_commands") == 1
    assert count(m, "backtest_artifact_requests") == 2
    repo = PostgresArtifactManagementRepository(
        gateway=PsycopgBacktestPostgresGateway(dsn=m["dsn"]), catalog=m["catalog"], access=Access()
    )
    restored = repo.command(organization_id=m["org"], actor_id=m["actor"], key="command-0001")
    assert restored is not None and restored["response"] == results[0]
    with pytest.raises(ManagementError, match="idempotency_conflict"):
        submit(m, items[:1])


def test_bulk_stale_and_invalid_rollback_no_partial_writes(managed):
    m = managed
    original = item(m)
    submit(m, (original,))
    with pytest.raises(ManagementError, match="stale_version"):
        submit(m, (item(m, "ETHUSDT"), original), key="command-0002")
    assert count(m, "backtest_artifact_policies") == 1
    assert count(m, "backtest_artifact_commands") == 1
    with pytest.raises(ManagementError, match="coordinate_forbidden"):
        submit(m, (item(m, "ETHUSDT"), item(m, "SECRET")), key="command-0003")
    assert count(m, "backtest_artifact_profiles") == 1


def test_roles_and_tenant_non_disclosing_reads(managed):
    m = managed
    with pytest.raises(ManagementError, match="forbidden"):
        submit(m, (item(m),), actor_id=m["viewer"])
    request = UUID(submit(m, (item(m),))["items"][0]["request_id"])
    assert (
        m["repo"].request(organization_id=m["other"], actor_id=m["actor"], request_id=request)
        is None
    )
    assert (
        m["repo"].events(organization_id=m["other"], actor_id=m["actor"], request_id=request) == ()
    )
    assert (
        m["repo"].command(organization_id=m["other"], actor_id=m["actor"], key="command-0001")
        is None
    )
    assert count(m, "backtest_artifact_commands") == 1
    m["db"].execute(
        query="UPDATE identity_memberships SET status='suspended' WHERE user_id=%(a)s",
        parameters={"a": m["actor"]},
    )
    with pytest.raises(ManagementError, match="forbidden"):
        submit(m, (item(m),))


def test_installation_defaults_cas_and_no_reseed(managed):
    m = managed
    with pytest.raises(ManagementError, match="installation_forbidden"):
        m["repo"].update_defaults(
            organization_id=m["org"],
            actor_id=m["admin"],
            expected_version=1,
            schedule=m["schedule"],
            budget_bytes=1234,
        )
    new = Schedule(interval_days=7, timezone="UTC", local_time=time(4), anchor=date(2026, 10, 1))
    result = m["repo"].update_defaults(
        organization_id=m["org"],
        actor_id=m["actor"],
        expected_version=1,
        schedule=new,
        budget_bytes=10737418240,
    )
    assert result["control_version"] == 2
    again = m["repo"].initialize_settings(
        organization_id=m["org"], actor_id=m["actor"], root_id="other", schedule=m["schedule"]
    )
    assert again["schedule"] == new.model_dump(mode="json")
    assert again["root_id"] == "backtest-primary"
    with pytest.raises(ManagementError, match="stale_version"):
        m["repo"].update_defaults(
            organization_id=m["org"],
            actor_id=m["actor"],
            expected_version=1,
            schedule=new,
            budget_bytes=10737418240,
        )


def test_concurrent_policy_cas_and_immutable_revision(managed):
    m = managed
    first = item(m)
    submit(m, (first,))
    changed = first.model_copy(update={"expected_version": 1})

    def mutate(n):
        try:
            return submit(m, (changed,), key=f"policy-000{n}", kind="policy")
        except ManagementError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(mutate, range(2)))
    assert sum(r == "stale_version" for r in results) == 1
    bad = first.profile.model_copy(update={"mode": "precompute"})
    with pytest.raises(ManagementError, match="immutable_profile_conflict"):
        submit(m, (item(m, version=2, profile=bad),), key="policy-0003", kind="policy")


def test_claim_cross_tenant_and_restart_no_steal(managed):
    m = managed
    a = submit(m, (item(m),))["items"][0]["request_id"]
    b = submit(m, (item(m),), organization_id=m["other"])["items"][0]["request_id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda r: claim(m, r), (a, b)))
    assert sum(x is not None for x in claims) == 1
    build = next(x for x in claims if x is not None)
    restarted = PostgresArtifactManagementRepository(
        gateway=PsycopgBacktestPostgresGateway(dsn=m["dsn"]), catalog=m["catalog"], access=Access()
    )
    m["repo"] = restarted
    assert claim(m, a) is None and claim(m, b) is None
    restarted.finish(build=build, outcome="quarantined", reason="worker_liveness_unknown")
    assert claim(m, a) is None and claim(m, b) is None
    with pytest.raises(ManagementError, match="ownership_lost"):
        restarted.finish(build=build, outcome="error", reason="failed")


def test_progress_retry_epoch_and_stale_owner(managed):
    m = managed
    request = submit(m, (item(m),))["items"][0]["request_id"]
    build = claim(m, request)
    progress = Progress(
        phase="base",
        timeframe="1m",
        completed_units=3,
        total_units=None,
        unit="bars",
        observed_at=NOW,
        last_checkpoint={"row": 3},
    )
    m["repo"].observe(build=build, progress=progress)
    with pytest.raises(ManagementError, match="stale_progress"):
        m["repo"].observe(build=build, progress=progress)
    with pytest.raises(ManagementError, match="ownership_lost"):
        m["repo"].finish(build=dict(build, owner_token=uuid4()), outcome="error")
    m["repo"].finish(build=build, outcome="retry_wait", reason="transient_io")
    retry = m["repo"].retry(
        build=build,
        owner_token=uuid4(),
        parent_incarnation=uuid4(),
        now=datetime.now(UTC) + timedelta(days=1),
    )
    assert retry["attempt"] == 2 and retry["epoch"] == 2 and retry["progress"] == {}
    with pytest.raises(ManagementError, match="ownership_lost"):
        m["repo"].observe(
            build=build,
            progress=progress.model_copy(update={"observed_at": NOW + timedelta(seconds=1)}),
        )
    assert count(m, "backtest_artifact_attempts") == 2
    assert (
        len(
            m["repo"].events(
                organization_id=m["org"], actor_id=m["actor"], request_id=UUID(request)
            )
        )
        == 2
    )


def test_system_dedupe_obligation_survives_capacity(managed):
    m = managed
    spec = item(m)

    def enqueue():
        return m["repo"].enqueue_system(
            installation_id=m["installation"], coordinate=spec.coordinate, source=spec.source
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        ids = list(pool.map(lambda _: enqueue(), range(3)))
    assert len(set(ids)) == 1
    m["db"].execute(
        query="UPDATE backtest_artifact_installation_settings SET global_queue_limit=1",
        parameters={},
    )
    other = item(m, "ETHUSDT")
    with pytest.raises(ManagementError, match="quota_exceeded"):
        m["repo"].enqueue_system(
            installation_id=m["installation"], coordinate=other.coordinate, source=other.source
        )
    assert count(m, "backtest_artifact_source_state") == 2
    assert count(m, "backtest_artifact_requests") == 1


def test_import_explicit_idempotent_conflict_and_preservation(managed):
    m = managed
    config = load_backtest_artifacts_runtime_config("configs/dev/backtest_artifacts.yaml")
    demand = legacy_demand(
        config,
        item(m).coordinate,
        defaults=YamlBacktestGridDefaultsProvider.from_yaml(
            config_path="configs/dev/indicators.yaml"
        ),
    )
    args = dict(organization_id=m["org"], actor_id=m["actor"], demands=(demand,))
    assert m["repo"].import_legacy(**args) == dict(imported=1, skipped=0, conflicted=0)
    assert m["repo"].import_legacy(**args) == dict(imported=0, skipped=1, conflicted=0)
    args["demands"] = ((demand[0], "a" * 64, demand[2]),)
    assert m["repo"].import_legacy(**args)["conflicted"] == 1
    assert count(m, "backtest_artifact_profiles") == 0
    with psycopg.connect(m["dsn"]) as conn:
        job = uuid4()
        conn.execute(
            """INSERT INTO backtest_jobs(job_id,input_recipe_json)
                        VALUES (%s,'{"preserved":true}')""",
            (job,),
        )
        conn.execute(
            sql.SQL(
                cast(
                    LiteralString,
                    Path(
                        "migrations/postgres/0029_backtest_artifact_management_v1.sql"
                    ).read_text(),
                )
            )
        )
        job_row = conn.execute(
            "SELECT input_recipe_json FROM backtest_jobs WHERE job_id=%s", (job,)
        ).fetchone()
        assert job_row is not None and job_row[0] == {"preserved": True}
        legacy_row = conn.execute(
            "SELECT snapshot FROM backtest_artifact_legacy_demands"
        ).fetchone()
        assert legacy_row is not None and legacy_row[0]["signals"] == "precompute"


def test_child_command_organization_constraint(managed):
    m = managed
    result = submit(m, (item(m),))
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        m["db"].execute(
            query="""UPDATE backtest_artifact_requests SET organization_id=%(org)s
            WHERE command_id=%(c)s""",
            parameters={"org": m["other"], "c": UUID(result["command_id"])},
        )


def test_conflicting_profile_inside_same_bulk_rolls_back(managed):
    m = managed
    first = item(m)
    second = item(m, "ETHUSDT", profile=first.profile.model_copy(update={"mode": "precompute"}))
    with pytest.raises(ManagementError, match="immutable_profile_conflict"):
        submit(m, (first, second))
    assert count(m, "backtest_artifact_commands") == 0
    assert count(m, "backtest_artifact_profiles") == 0
    assert count(m, "backtest_artifact_requests") == 0


def test_retry_budget_shrink_blocks_before_new_attempt(managed):
    m = managed
    request = submit(m, (item(m),))["items"][0]["request_id"]
    build = claim(m, request)
    m["repo"].finish(build=build, outcome="retry_wait", reason="transient_database")
    m["repo"].update_defaults(
        organization_id=m["org"],
        actor_id=m["actor"],
        expected_version=1,
        schedule=m["schedule"],
        budget_bytes=1073741824,
    )
    with pytest.raises(ManagementError, match="resource_conflict"):
        m["repo"].retry(
            build=build,
            owner_token=uuid4(),
            parent_incarnation=uuid4(),
            now=datetime.now(UTC) + timedelta(days=1),
        )
    assert count(m, "backtest_artifact_attempts") == 1


def test_command_recovery_checks_current_coordinate_access(managed):
    m = managed
    submit(m, (item(m),))

    class Revoked:
        def require(self, **kwargs):
            raise ManagementError("coordinate_forbidden")

    m["repo"].access = Revoked()
    with pytest.raises(ManagementError, match="coordinate_forbidden"):
        m["repo"].command(organization_id=m["org"], actor_id=m["actor"], key="command-0001")


def test_concurrent_retry_has_one_new_attempt_and_fences_loser(managed):
    m = managed
    request = submit(m, (item(m),))["items"][0]["request_id"]
    build = claim(m, request)
    m["repo"].finish(build=build, outcome="retry_wait", reason="transient_io")

    def retry():
        try:
            return m["repo"].retry(
                build=build,
                owner_token=uuid4(),
                parent_incarnation=uuid4(),
                now=datetime.now(UTC) + timedelta(days=1),
            )
        except ManagementError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: retry(), range(2)))
    assert sum(r == "ownership_lost" for r in outcomes) == 1
    assert count(m, "backtest_artifact_attempts") == 2
    assert (
        m["db"].fetch_one(
            query="SELECT count(*) n FROM backtest_artifact_builds " "WHERE state='running'",
            parameters={},
        )["n"]
        == 1
    )


def test_retry_racing_new_claim_preserves_global_budget(managed):
    m = managed
    requests = submit(m, (item(m), item(m, "ETHUSDT")))["items"]
    build = claim(m, requests[0]["request_id"])
    m["repo"].finish(build=build, outcome="retry_wait", reason="transient_io")
    with ThreadPoolExecutor(max_workers=2) as pool:
        retry = pool.submit(
            m["repo"].retry,
            build=build,
            owner_token=uuid4(),
            parent_incarnation=uuid4(),
            now=datetime.now(UTC) + timedelta(days=1),
        )
        fresh = pool.submit(claim, m, requests[1]["request_id"])
        assert sum(x is not None for x in (retry.result(), fresh.result())) == 1
    assert (
        m["db"].fetch_one(
            query="SELECT count(*) n FROM backtest_artifact_builds " "WHERE state='running'",
            parameters={},
        )["n"]
        == 1
    )


def test_database_failure_after_first_child_rolls_back_complete_bulk(managed):
    m = managed
    with psycopg.connect(m["dsn"]) as conn:
        conn.execute("""CREATE FUNCTION reject_second() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN IF NEW.symbol='ETHUSDT' THEN RAISE EXCEPTION 'disposable_failure'; END IF;
            RETURN NEW; END $$;
            CREATE TRIGGER reject_second BEFORE INSERT ON backtest_artifact_requests
            FOR EACH ROW EXECUTE FUNCTION reject_second();""")
    with pytest.raises(psycopg.errors.RaiseException, match="disposable_failure"):
        submit(m, (item(m), item(m, "ETHUSDT")))
    for table in (
        "backtest_artifact_commands",
        "backtest_artifact_profiles",
        "backtest_artifact_policies",
        "backtest_artifact_requests",
    ):
        assert count(m, table) == 0


def test_system_replay_does_not_regress_newer_desired_projection(managed):
    m = managed
    spec = item(m)
    args = dict(installation_id=m["installation"], coordinate=spec.coordinate, source=spec.source)
    original = m["repo"].enqueue_system(**args)
    m["repo"].enqueue_system(**args, occurrence="2026-10-09")
    assert m["repo"].enqueue_system(**args) == original
    projection = m["db"].fetch_one(
        query="SELECT desired_target FROM backtest_artifact_source_state", parameters={}
    )
    assert projection["desired_target"]["occurrence"] == "2026-10-09"
