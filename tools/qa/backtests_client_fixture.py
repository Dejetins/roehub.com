"""Disposable local Backtests proof stack, never imported by a production entrypoint.

Uses production auth, PostgreSQL, ClickHouse readers, filesystem artifacts and idle
job runner. Synthetic input data is explicit; no job is submitted by this tool.
Run from the repository root: .venv/bin/python -m tools.qa.backtests_client_fixture
Ctrl-C removes this run's containers, processes and private temporary state.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import httpx
import psycopg
from argon2 import PasswordHasher
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".local_artifacts/backtests-client"
PG_IMAGE = "postgres@sha256:cf78e76683b9ca8c5733cbbdce6c9262b45b6767934dd0a95e671f9a0fc20685"
CH_IMAGE = (
    "clickhouse/clickhouse-server@sha256:"
    "87e0a5b72f5465b18eacca7c76850e7ff551c9795c50e451f5646299e5e24146"
)
PREVIEW_SESSION_TTL_SECONDS = 365 * 24 * 60 * 60


def preview_identity_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Keep interactive local preview logins for a year; explicit test TTLs win."""
    if environ.get("ROEHUB_ENV", "dev") not in {"dev", "test"}:
        raise ValueError("The disposable preview session policy is local-only")
    result = dict(environ)
    result.setdefault("IDENTITY_SESSION_IDLE_TTL_SECONDS", str(PREVIEW_SESSION_TTL_SECONDS))
    result.setdefault("IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS", str(PREVIEW_SESSION_TTL_SECONDS))
    return result


def create_api() -> FastAPI:
    """Use the real composition roots; the optional fault is explicit test injection."""
    from apps.api.common import register_api_error_handlers
    from apps.api.wiring.modules.backtest import build_backtests_router
    from apps.api.wiring.modules.identity import build_identity_api_module
    from apps.api.wiring.modules.strategy import build_strategy_router
    from apps.api.wiring.modules.ui_backtests import build_ui_backtests_router
    from apps.api.wiring.modules.ui_strategies_dashboard import (
        build_ui_strategies_dashboard_router,
    )

    identity = build_identity_api_module(environ=preview_identity_environment(os.environ))
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(identity.router)
    app.include_router(
        build_ui_strategies_dashboard_router(
            environ=os.environ, current_user_dependency=identity.current_user_dependency
        )
    )
    app.include_router(
        build_strategy_router(
            environ=os.environ, current_user_dependency=identity.current_user_dependency
        )
    )
    app.include_router(
        build_backtests_router(
            environ=os.environ,
            current_user_dependency=identity.current_user_dependency,
        )
    )
    app.include_router(
        build_ui_backtests_router(
            environ=os.environ,
            current_user_dependency=identity.current_user_dependency,
        )
    )

    @app.middleware("http")
    async def identity_fault(request: Request, call_next):
        if request.url.path == "/auth/current-user" and (STATE / "identity-unavailable").exists():
            return JSONResponse({"detail": "Fixture identity outage"}, status_code=503)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"healthy": True}

    return app


def _docker(*args: str, input: str | None = None, environ=None) -> str:
    result = subprocess.run(
        ["docker", *args], input=input, text=True, capture_output=True, check=False, env=environ
    )
    if result.returncode:
        # Docker diagnostics contain no application payloads. Never dump the child environment.
        raise RuntimeError(f"Docker {args[0]} failed: {result.stderr[-1200:]}")
    return result.stdout.strip()


def _wait(check, *, seconds=90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if check():
                return
        except (OSError, httpx.HTTPError, psycopg.Error, RuntimeError):
            pass
        time.sleep(0.3)
    raise RuntimeError("Local proof prerequisite did not become ready")


def _seed_postgres(dsn: str, password: str):
    user, installation, organization = uuid4(), uuid4(), uuid4()
    with psycopg.connect(dsn) as connection:
        connection.execute(
            """INSERT INTO identity_users
            (user_id, paid_level, created_at, last_login_at) VALUES (%s, 'free', now(), now())""",
            (user,),
        )
        connection.execute(
            """INSERT INTO identity_local_accounts
            (user_id, username, display_name, password_hash, created_at, updated_at)
            VALUES (%s, 'proof-owner', 'Disposable proof user', %s, now(), now())""",
            (user, PasswordHasher().hash(password)),
        )
        connection.execute(
            """INSERT INTO identity_installations
            (installation_id, display_name, created_at) VALUES (%s, 'Disposable proof', now())""",
            (installation,),
        )
        connection.execute(
            """INSERT INTO identity_installation_owners
            VALUES (%s, %s, %s, now())""",
            (installation, user, user),
        )
        connection.execute(
            """INSERT INTO identity_organizations
            (organization_id, installation_id, slug, display_name, created_at)
            VALUES (%s, %s, 'proof-org', 'Disposable proof organization', now())""",
            (organization, installation),
        )
        connection.execute(
            """INSERT INTO identity_memberships
            (organization_id, user_id, role, created_at, updated_at)
            VALUES (%s, %s, 'owner', now(), now())""",
            (organization, user),
        )
    return str(user)


def _artifacts(env: dict[str, str], dsn: str) -> Path:
    """Build real artifact files from deterministic, disposable canonical input candles."""
    from datetime import UTC, datetime

    from apps.api.wiring.modules.indicators import (
        build_indicators_compute,
        build_indicators_registry,
    )
    from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
        build_artifact_precompute_fixture_v2,
    )
    from tests.unit.contexts.backtest.application.services.v2.test_artifact_precompute_runner_v2 import (  # noqa: E501
        _build_canonical_rows_v2,
        _FakeCanonicalCandleReader,
        _request_v2,
    )
    from trading.contexts.backtest.adapters.outbound import (
        PostgresBacktestJobRepository,
        PsycopgBacktestPostgresGateway,
        YamlBacktestGridDefaultsProvider,
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
        AtomicArtifactCurrentPointerWriterV2,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_precompute_runner import (  # noqa: E501
        BacktestArtifactPrecomputeRunnerV2,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_publisher import (  # noqa: E501
        BacktestArtifactSlotPublisherV2,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.signal_rules_engine_v2 import (
        BacktestSignalRulesEngineV2,
    )
    from trading.contexts.indicators.application.services import GridBuilder

    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=STATE,
        hit_times_tp_levels_pct=(1.0, 2.0),
        hit_times_sl_levels_pct=(1.0, 2.0),
        validation_price_timeframes=("1m", "15m"),
        validation_mapping_timeframes=("15m",),
        validation_signal_artifacts=(("15m", "ma.ema"),),
        precompute_signal_artifacts=(("15m", "ma.ema"),),
        require_hit_times_manifest=True,
        signal_worker_processes=1,
    )
    defaults = YamlBacktestGridDefaultsProvider.from_environ(environ=env)
    # Only the input candle reader is synthetic. Computation, manifests, validation,
    # publication and the job-blocking check use production implementations.
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=_build_canonical_rows_v2(bar_indexes=tuple(range(4320)))
        ),
        defaults_provider=defaults,
        signal_rules_engine=BacktestSignalRulesEngineV2(defaults_provider=defaults),
        indicator_compute=build_indicators_compute(environ=env),
        indicator_grid_builder=GridBuilder(registry=build_indicators_registry(environ=env)),
    )
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=fixture.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=fixture.builder),
        job_repository=PostgresBacktestJobRepository(
            gateway=PsycopgBacktestPostgresGateway(dsn=dsn)
        ),
        now_provider=lambda: datetime(2026, 3, 29, 3, 0, tzinfo=UTC),
    )
    precheck = publisher.precheck_publish(fixture.coordinates)
    runner.export_canonical_price_1m(
        _request_v2(
            fixture=fixture,
            end_minute=4320,
            asof_date="2026-03-29",
            generated_at_utc="2026-03-29T03:00:00Z",
        )
    )
    publisher.publish(
        precheck=precheck,
        validation_spec=fixture.runtime_config.to_validation_spec(),
        asof_date="2026-03-29",
    )
    print(
        "Synthetic 4320 candles -> production precompute/validate/local publish: passed", flush=True
    )
    return fixture.config_path


def run_stack():
    """Start only new, named loopback containers; refuse to overwrite another run."""
    STATE.mkdir(parents=True, exist_ok=False)
    STATE.chmod(0o700)
    suffix = secrets.token_hex(4)
    pg, ch = f"roehub-client-pg-{suffix}", f"roehub-client-ch-{suffix}"
    containers: list[str] = []
    processes: list[subprocess.Popen] = []
    logs = []
    # No ambient service credentials, provider endpoints or deployment environment.
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "TMPDIR", "SYSTEMROOT")
        if key in os.environ
    }
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
            "ROEHUB_ENV": "test",
            "NUMBA_NUM_THREADS": "1",
            "ROEHUB_NUMBA_NUM_THREADS": "1",
        }
    )
    password = secrets.token_urlsafe(30)

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        _docker(
            "run",
            "-d",
            "--rm",
            "--name",
            pg,
            "-p",
            "127.0.0.1::5432",
            "-e",
            "POSTGRES_USER=proof",
            "-e",
            "POSTGRES_DB=proof",
            "-e",
            "POSTGRES_PASSWORD",
            PG_IMAGE,
            environ={**env, "POSTGRES_PASSWORD": password},
        )
        containers.append(pg)
        port = _docker("port", pg, "5432/tcp").rsplit(":", 1)[1]
        dsn = f"postgresql://proof:{password}@127.0.0.1:{port}/proof"
        _wait(lambda: _docker("exec", pg, "pg_isready", "-U", "proof", "-d", "proof"))
        env.update({"POSTGRES_DSN": dsn, "STRATEGY_PG_DSN": dsn, "IDENTITY_PG_DSN": dsn})
        from apps.migrations.bootstrap import run_dev_db_bootstrap

        run_dev_db_bootstrap(
            identity_dsn=dsn, postgres_dsn=dsn, migrations_dir=ROOT / "migrations/postgres"
        )
        # Remaining canonical greenfield SQL follows the migration manifest order.
        manifest = json.loads((ROOT / "migrations/postgres/manifest.json").read_text())
        with psycopg.connect(dsn, autocommit=True) as connection:
            for phase in manifest["phases"]:
                for entry in phase["files"]:
                    if entry["path"][:4] >= "0014":
                        connection.execute(
                            (ROOT / "migrations/postgres" / entry["path"]).read_text()
                        )
        print("PostgreSQL migrations: passed", flush=True)
        subject = _seed_postgres(dsn, password)
        _docker(
            "run",
            "-d",
            "--rm",
            "--name",
            ch,
            "-p",
            "127.0.0.1::8123",
            "-e",
            "CLICKHOUSE_USER=proof",
            "-e",
            "CLICKHOUSE_PASSWORD",
            "-e",
            "CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1",
            CH_IMAGE,
            environ={**env, "CLICKHOUSE_PASSWORD": password},
        )
        containers.append(ch)
        ch_port = _docker("port", ch, "8123/tcp").rsplit(":", 1)[1]
        env.update(
            {
                "CH_HOST": "127.0.0.1",
                "CH_PORT": ch_port,
                "CH_USER": "proof",
                "CH_PASSWORD": password,
                "CH_DATABASE": "market_data",
            }
        )
        ch_url = f"http://127.0.0.1:{ch_port}"
        _wait(lambda: httpx.get(ch_url + "/ping").status_code == 200)
        _docker(
            "exec",
            "-i",
            ch,
            "clickhouse-client",
            "--multiquery",
            input=(ROOT / "migrations/clickhouse/market_data_ddl.sql").read_text(),
        )
        with httpx.Client(auth=("proof", password)) as client:
            for sql in (
                "INSERT INTO market_data.ref_market (market_id, exchange_name, market_type, "
                "market_code, is_enabled, count_symbols) "
                "VALUES (1,'binance','spot','binance:spot',1,1)",
                "INSERT INTO market_data.ref_instruments (market_id, symbol, status, is_tradable) "
                "VALUES (1,'BTCUSDT','ENABLED',1)",
            ):
                client.post(ch_url, content=sql).raise_for_status()
        print("ClickHouse schema and synthetic catalog: passed", flush=True)
        env["ROEHUB_INDICATORS_CONFIG"] = str(ROOT / "configs/test/indicators.yaml")
        config = _artifacts(env, dsn)
        env.update(
            {
                "ROEHUB_BACKTEST_ARTIFACTS_CONFIG": str(config),
                "ROEHUB_INDICATORS_CONFIG": str(ROOT / "configs/test/indicators.yaml"),
                "ROEHUB_BACKTEST_TRADES_CACHE_ROOT": str(STATE / "trades-cache"),
                "IDENTITY_LOCAL_RP_ID": "localhost",
                "IDENTITY_LOCAL_ORIGIN": "http://localhost:18480",
                "WEB_API_BASE_URL": "http://127.0.0.1:18480",
                "WEB_API_UPSTREAM_URL": "http://127.0.0.1:18481",
            }
        )
        private = STATE / "credentials.json"
        with private.open("x", encoding="utf-8") as handle:
            private.chmod(0o600)
            json.dump(
                {"username": "proof-owner", "password": password, "subject": subject, "dsn": dsn},
                handle,
            )

        def launch(name, args, extra=None):
            log = (STATE / f"{name}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [sys.executable, *args],
                    cwd=ROOT,
                    env={**env, **(extra or {})},
                    stdout=log,
                    stderr=log,
                )
            )
            if name == "runner":
                (STATE / "runner.pid").write_text(str(processes[-1].pid))

        for name, module, port, extra in (
            ("api", "tools.qa.backtests_client_fixture:create_api", "18481", {}),
            (
                "web",
                "apps.web.main.app:create_app",
                "18480",
                {"WEB_BACKTESTS_CLIENT_ENABLED": "true"},
            ),
            (
                "ssr",
                "apps.web.main.app:create_app",
                "18482",
                {"WEB_BACKTESTS_CLIENT_ENABLED": "false"},
            ),
        ):
            launch(
                name,
                [
                    "-m",
                    "uvicorn",
                    module,
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    port,
                    "--no-access-log",
                ],
                extra,
            )
        launch(
            "runner", ["-m", "apps.worker.backtest_job_runner.main.main", "--metrics-port", "18483"]
        )
        for port, path in (
            (18481, "/health"),
            (18480, "/health/live"),
            (18482, "/health/live"),
            (18483, "/metrics"),
        ):
            _wait(lambda: httpx.get(f"http://127.0.0.1:{port}{path}").status_code == 200)
        print(
            "Local proof ready: Web :18480, API :18481, SSR :18482, idle runner :18483", flush=True
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise RuntimeError("A local proof process exited unexpectedly")
    except KeyboardInterrupt:
        pass
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                # A proof may pause the idle scheduler to inspect real queued cancellation.
                process.send_signal(signal.SIGCONT)
            process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
        for container in reversed(containers):
            _docker("rm", "-f", container)
        shutil.rmtree(STATE)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in {"pause-runner", "resume-runner"}:
        # Only the scheduler launched by this disposable stack is targeted.
        pid = int((STATE / "runner.pid").read_text())
        os.kill(pid, signal.SIGSTOP if sys.argv[1] == "pause-runner" else signal.SIGCONT)
    elif len(sys.argv) == 2 and sys.argv[1] == "expire":
        private = json.loads((STATE / "credentials.json").read_text())
        with psycopg.connect(private["dsn"]) as connection:
            connection.execute("""UPDATE identity_sessions SET
                created_at=created_at - interval '2 days',
                last_seen_at=last_seen_at - interval '2 days',
                idle_expires_at=idle_expires_at - interval '2 days',
                absolute_expires_at=absolute_expires_at - interval '2 days'""")
    else:
        run_stack()
