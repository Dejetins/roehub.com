from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import quantiles
from typing import Any, Mapping, cast
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

_SMOKE_KEYCLOAK_SUBJECT = "codex:backtest-job-runner-prod-smoke"
_DEFAULT_API_BASE = "http://127.0.0.1:8000"
_DEFAULT_COOKIE_NAME = "roehub_session_id"
_DEFAULT_CACHE_ROOT = Path("/opt/roehub/state/backtest/trades_cache")
_RUNNER_METRICS_URL = "http://127.0.0.1:9204/metrics"
_PROMETHEUS_QUERY_URL = "http://127.0.0.1:9090/api/v1/query"
_RUNNER_LOG_FILES = (
    Path("/Users/daniildegtyarev/Library/Logs/roehub/backtest-job-runner.out.log"),
    Path("/Users/daniildegtyarev/Library/Logs/roehub/backtest-job-runner.err.log"),
)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be > 0")
    if args.poll_interval_seconds <= 0:
        raise SystemExit("--poll-interval-seconds must be > 0")

    started_at = datetime.now(UTC)
    dsn = _postgres_dsn(environ=os.environ)
    session_id: str | None = None
    evidence: dict[str, Any] = {
        "started_at": _format_datetime(started_at),
        "api_base": args.api_base,
        "instrument": "BTCUSDT",
        "timeframe": "15m",
    }
    try:
        user_id, session_id = _create_smoke_session(
            dsn=dsn,
            cookie_name=args.cookie_name,
            session_ttl_seconds=args.session_ttl_seconds,
        )
        client = _ApiClient(
            api_base=args.api_base,
            cookie_name=args.cookie_name,
            session_id=session_id,
        )
        evidence["smoke_user_id"] = user_id
        evidence["backlog_before"] = _backlog_snapshot(dsn=dsn)
        created = _create_controlled_job(client=client)
        job_id = str(created["job_id"])
        evidence["job_id"] = job_id
        evidence["job_create"] = _job_create_evidence(created=created)

        job_result = _wait_for_job_success(
            dsn=dsn,
            client=client,
            job_id=job_id,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        evidence["job_result"] = job_result

        top = client.request_json("GET", f"/backtests/jobs/{job_id}/top")
        top_items = _require_list(top.get("items"), "top.items")
        if not top_items:
            raise RuntimeError("controlled smoke job succeeded without top variants")
        top_variant = cast(dict[str, Any], top_items[0])
        variant_key = str(top_variant["variant_key"])
        evidence["top_variant"] = {
            "variant_key": variant_key,
            "variant_hash": str(top_variant["variant_hash"]),
            "rank": top_variant.get("rank"),
            "summary_metric_keys": sorted(
                str(key) for key in dict(top_variant.get("summary_metrics", {})).keys()
            ),
        }

        cache_key = _compute_lazy_cache_key(
            dsn=dsn,
            job_id=job_id,
            public_variant_key=variant_key,
            variant_hash=str(top_variant["variant_hash"]),
        )
        _clear_lazy_cache_and_task(dsn=dsn, cache_key=cache_key)
        evidence["lazy_detail"] = _run_lazy_detail_smoke(
            dsn=dsn,
            client=client,
            job_id=job_id,
            variant_key=variant_key,
            cache_key=cache_key,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        evidence["api_responsiveness"] = _run_api_responsiveness_smoke(
            client=client,
            job_id=job_id,
            requests=args.status_burst_requests,
        )
        evidence["metrics"] = _verify_runner_metrics()
        evidence["prometheus"] = _verify_prometheus_target()
        evidence["log_scan"] = _scan_runner_logs()
        evidence["backlog_after"] = _backlog_snapshot(dsn=dsn)
        evidence["finished_at"] = _format_datetime(datetime.now(UTC))
    finally:
        if session_id is not None:
            _revoke_smoke_session(dsn=dsn, session_id=session_id)

    rendered = json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the production backtest-job-runner R5 smoke on Mac Studio."
    )
    parser.add_argument("--api-base", default=_DEFAULT_API_BASE)
    parser.add_argument("--cookie-name", default=_DEFAULT_COOKIE_NAME)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--session-ttl-seconds", type=int, default=3600)
    parser.add_argument("--status-burst-requests", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".codex/tmp/backtest_job_runner_prod_smoke.json"),
    )
    return parser


def _postgres_dsn(*, environ: Mapping[str, str]) -> str:
    for key in ("STRATEGY_PG_DSN", "POSTGRES_DSN", "IDENTITY_PG_DSN"):
        value = environ.get(key, "").strip()
        if value:
            return value
    required = ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")
    missing = [key for key in required if not environ.get(key, "").strip()]
    if missing:
        raise RuntimeError(
            "Postgres DSN is required via STRATEGY_PG_DSN/POSTGRES_DSN or "
            f"{', '.join(missing)}"
        )
    return (
        "host=127.0.0.1 port=5432 "
        f"dbname={environ['POSTGRES_DB']} user={environ['POSTGRES_USER']} "
        f"password={environ['POSTGRES_PASSWORD']}"
    )


def _create_smoke_session(
    *,
    dsn: str,
    cookie_name: str,
    session_ttl_seconds: int,
) -> tuple[str, str]:
    if session_ttl_seconds <= 0:
        raise RuntimeError("session TTL must be positive")
    now = datetime.now(UTC)
    user_id = str(uuid4())
    session_id = str(uuid4())
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO identity_users
                    (user_id, keycloak_subject, paid_level, created_at, last_login_at, is_deleted)
                VALUES
                    (%(user_id)s, %(subject)s, 'ultra', %(now)s, %(now)s, FALSE)
                ON CONFLICT (keycloak_subject)
                    WHERE keycloak_subject IS NOT NULL
                DO UPDATE SET
                    paid_level = 'ultra',
                    last_login_at = EXCLUDED.last_login_at,
                    is_deleted = FALSE
                RETURNING user_id
                """,
                {"user_id": user_id, "subject": _SMOKE_KEYCLOAK_SUBJECT, "now": now},
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("smoke identity user upsert returned no row")
            persisted_user_id = str(cast(Mapping[str, Any], row)["user_id"])
            cursor.execute(
                """
                INSERT INTO identity_sessions
                    (
                        session_id,
                        user_id,
                        created_at,
                        last_seen_at,
                        idle_expires_at,
                        absolute_expires_at,
                        revoked_at
                    )
                VALUES
                    (
                        %(session_id)s,
                        %(user_id)s,
                        %(now)s,
                        %(now)s,
                        %(idle_expires_at)s,
                        %(absolute_expires_at)s,
                        NULL
                    )
                """,
                {
                    "session_id": session_id,
                    "user_id": persisted_user_id,
                    "now": now,
                    "idle_expires_at": now + timedelta(seconds=session_ttl_seconds),
                    "absolute_expires_at": now + timedelta(seconds=session_ttl_seconds),
                },
            )
    _ = cookie_name
    return persisted_user_id, session_id


def _revoke_smoke_session(*, dsn: str, session_id: str) -> None:
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE identity_sessions SET revoked_at = %(now)s WHERE session_id = %(id)s",
                {"now": datetime.now(UTC), "id": session_id},
            )


@dataclass(frozen=True, slots=True)
class _ApiClient:
    api_base: str
    cookie_name: str
    session_id: str

    def request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Cookie": f"{self.cookie_name}={self.session_id}",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        if extra_headers is not None:
            headers.update(extra_headers)
        request = urllib.request.Request(
            url=f"{self.api_base.rstrip('/')}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
                if not raw:
                    return {"_status": response.status}
                parsed = json.loads(raw)
                if not isinstance(parsed, Mapping):
                    raise RuntimeError(f"{path} did not return a JSON object")
                return {"_status": response.status, **dict(parsed)}
        except urllib.error.HTTPError as error:
            raw_error = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{method} {path} returned HTTP {error.code}: {raw_error[:1000]}"
            ) from error


def _create_controlled_job(*, client: _ApiClient) -> dict[str, Any]:
    created = client.request_json(
        "POST",
        "/backtests/jobs",
        payload=_controlled_request(),
        extra_headers={"Idempotency-Key": f"r5-runner-smoke-{uuid4()}"},
    )
    if created.get("_status") != 201:
        raise RuntimeError(f"expected job create HTTP 201, got {created.get('_status')}")
    if created.get("state") != "queued":
        raise RuntimeError(f"expected created job state queued, got {created.get('state')!r}")
    return created


def _controlled_request() -> dict[str, Any]:
    return {
        "coordinates": {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "BTCUSDT",
        },
        "timeframe": "15m",
        "time_range": {
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-02-01T00:00:00Z",
        },
        "indicators": [
            {
                "indicator_id": "ma.dema",
                "sources": ["close"],
                "window": {"start": 5, "stop": 5, "step": 1},
            }
        ],
        "risk": {"mode": "none"},
        "execution": {
            "direction_mode": "long_short_reversal",
            "fee_rate": 0.00075,
            "slippage_rate": 0.0001,
            "initial_cash_quote": 10000.0,
            "sizing": {"mode": "fixed_equity_pct", "equity_pct": 10.0},
            "profit_lock": {"enabled": False},
            "close_on_end": True,
        },
        "ranking": {"primary_metric": "total_return_pct", "direction": "desc"},
        "top_n": 1,
    }


def _job_create_evidence(*, created: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state": created.get("state"),
        "pipeline_stage": dict(created.get("progress", {})).get("pipeline_stage"),
        "requested_top_n": created.get("requested_top_n"),
        "request_hash": created.get("request_hash"),
    }


def _wait_for_job_success(
    *,
    dsn: str,
    client: _ApiClient,
    job_id: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    observed_states: list[str] = ["queued"]
    running_samples: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        api_payload = client.request_json("GET", f"/backtests/jobs/{job_id}")
        db_row = _job_db_row(dsn=dsn, job_id=job_id)
        state = str(db_row["state"])
        if state not in observed_states:
            observed_states.append(state)
        if state == "running":
            running_samples.append(_running_sample(row=db_row))
        if state in {"succeeded", "failed", "cancelled"}:
            terminal = {
                "state": state,
                "api_state": api_payload.get("state"),
                "started_at": _format_datetime(db_row.get("started_at")),
                "finished_at": _format_datetime(db_row.get("finished_at")),
                "attempt": db_row.get("attempt"),
                "last_error": db_row.get("last_error"),
                "top_variants_count": _top_variants_count(dsn=dsn, job_id=job_id),
            }
            break
        time.sleep(poll_interval_seconds)
    if terminal is None:
        raise RuntimeError(f"controlled smoke job did not finish before timeout: {job_id}")
    if terminal["state"] != "succeeded":
        raise RuntimeError(f"controlled smoke job terminal state is not succeeded: {terminal}")
    if terminal["top_variants_count"] <= 0:
        raise RuntimeError("controlled smoke job has no top variants")
    if "running" not in observed_states:
        raise RuntimeError(f"did not observe queued -> running -> succeeded: {observed_states}")
    if not running_samples:
        raise RuntimeError("did not capture running lease sample")
    first_running = running_samples[0]
    for field in ("locked_by", "started_at", "heartbeat_at", "lease_expires_at"):
        if not first_running.get(field):
            raise RuntimeError(f"running sample is missing {field}: {first_running}")
    return {
        "state_path": " -> ".join(observed_states),
        "required_path": "queued -> running -> succeeded",
        "running_sample": first_running,
        "terminal": terminal,
    }


def _job_db_row(*, dsn: str, job_id: str) -> dict[str, Any]:
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    job_id,
                    state,
                    started_at,
                    finished_at,
                    locked_by,
                    locked_at,
                    lease_expires_at,
                    heartbeat_at,
                    attempt,
                    last_error,
                    request_hash,
                    engine_params_hash,
                    backtest_runtime_config_hash,
                    artifact_manifest_hash
                FROM backtest_jobs
                WHERE job_id = %(job_id)s
                """,
                {"job_id": job_id},
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"backtest job not found in DB: {job_id}")
    return dict(row)


def _running_sample(*, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state": row.get("state"),
        "locked_by": row.get("locked_by"),
        "started_at": _format_datetime(row.get("started_at")),
        "heartbeat_at": _format_datetime(row.get("heartbeat_at")),
        "lease_expires_at": _format_datetime(row.get("lease_expires_at")),
        "attempt": row.get("attempt"),
    }


def _top_variants_count(*, dsn: str, job_id: str) -> int:
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) AS count FROM backtest_job_top_variants WHERE job_id = %(id)s",
                {"id": job_id},
            )
            row = cursor.fetchone()
    return int(cast(Mapping[str, Any], row)["count"])


def _compute_lazy_cache_key(
    *,
    dsn: str,
    job_id: str,
    public_variant_key: str,
    variant_hash: str,
) -> str:
    job = _job_db_row(dsn=dsn, job_id=job_id)
    engine_params_hash = str(
        job.get("engine_params_hash") or job.get("backtest_runtime_config_hash")
    )
    artifact_manifest_hash = str(job["artifact_manifest_hash"])
    payload = {
        "artifact_manifest_hash": artifact_manifest_hash,
        "engine_params_hash": engine_params_hash,
        "job_id": job_id,
        "request_hash": str(job["request_hash"]),
        "variant_hash": variant_hash,
        "variant_key": public_variant_key,
    }
    return _canonical_json_sha256(payload)


def _canonical_json_sha256(payload: Mapping[str, str]) -> str:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _clear_lazy_cache_and_task(*, dsn: str, cache_key: str) -> None:
    raw_cache_root = os.environ.get("ROEHUB_BACKTEST_TRADES_CACHE_ROOT", "").strip()
    cache_root = Path(raw_cache_root).expanduser() if raw_cache_root else _DEFAULT_CACHE_ROOT
    cache_path = cache_root / cache_key[:2] / f"{cache_key}.json"
    cache_path.unlink(missing_ok=True)
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM backtest_lazy_trades_materializations WHERE cache_key = %(key)s",
                {"key": cache_key},
            )


def _run_lazy_detail_smoke(
    *,
    dsn: str,
    client: _ApiClient,
    job_id: str,
    variant_key: str,
    cache_key: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    first = client.request_json("POST", f"/backtests/jobs/{job_id}/variants/{variant_key}/trades")
    if first.get("_status") != 202:
        raise RuntimeError(
            "expected lazy detail cache miss to return HTTP 202 after cache purge, "
            f"got {first.get('_status')} with cache={first.get('cache')}"
        )
    first_cache = dict(first.get("cache", {}))
    if first_cache.get("status") != "miss":
        raise RuntimeError(f"expected first lazy detail read cache miss, got {first_cache}")
    task_id = str(dict(first["materialization"])["task_id"])
    statuses = [str(first.get("status"))]
    deadline = time.monotonic() + timeout_seconds
    completed: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        row = _lazy_task_row(dsn=dsn, task_id=task_id)
        status = str(row["status"])
        if status not in statuses:
            statuses.append(status)
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"lazy materialization ended unsuccessfully: {dict(row)}")
        detail = client.request_json(
            "GET",
            f"/backtests/jobs/{job_id}/variants/{variant_key}/trades?page=1&page_size=5",
        )
        if detail.get("_status") == 200:
            completed = {
                "task_id": task_id,
                "status": status,
                "status_path": " -> ".join(statuses),
                "cache_key": cache_key,
                "initial_detail_read": {
                    "status": first.get("_status"),
                    "cache_status": first_cache.get("status"),
                    "materialization_status": first.get("status"),
                    "task_id": task_id,
                },
                "items_returned": len(_require_list(detail.get("items"), "trades.items")),
                "cache_status": dict(detail.get("cache", {})).get("status"),
            }
            break
        time.sleep(poll_interval_seconds)
    if completed is None:
        raise RuntimeError(f"lazy materialization did not complete before timeout: {task_id}")
    second = client.request_json("POST", f"/backtests/jobs/{job_id}/variants/{variant_key}/trades")
    if second.get("_status") != 200:
        raise RuntimeError(
            f"expected second lazy detail read HTTP 200, got {second.get('_status')}"
        )
    second_cache = dict(second.get("cache", {}))
    if second_cache.get("status") != "hit":
        raise RuntimeError(f"expected second lazy detail read cache hit, got {second_cache}")
    completed["second_detail_read"] = {
        "status": second.get("_status"),
        "cache_status": second_cache.get("status"),
        "trades_returned": len(_require_list(second.get("trades"), "trades")),
    }
    return completed


def _lazy_task_row(*, dsn: str, task_id: str) -> dict[str, Any]:
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    task_id,
                    status,
                    started_at,
                    finished_at,
                    locked_by,
                    heartbeat_at,
                    lease_expires_at,
                    attempt,
                    last_error,
                    cache_status,
                    cache_path
                FROM backtest_lazy_trades_materializations
                WHERE task_id = %(task_id)s
                """,
                {"task_id": task_id},
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"lazy materialization task not found: {task_id}")
    return dict(row)


def _run_api_responsiveness_smoke(
    *,
    client: _ApiClient,
    job_id: str,
    requests: int,
) -> dict[str, Any]:
    if requests <= 0:
        return {"requests": 0, "skipped": True}
    latencies_ms: list[float] = []
    for _index in range(requests):
        started = time.perf_counter()
        payload = client.request_json("GET", f"/backtests/jobs/{job_id}")
        if payload.get("_status") != 200 or payload.get("state") != "succeeded":
            raise RuntimeError(f"status responsiveness smoke got unexpected payload: {payload}")
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    sorted_latencies = sorted(latencies_ms)
    p95 = (
        quantiles(sorted_latencies, n=100)[94]
        if len(sorted_latencies) >= 100
        else max(sorted_latencies)
    )
    auth_started = time.perf_counter()
    auth_payload = client.request_json("GET", "/auth/current-user")
    auth_latency_ms = (time.perf_counter() - auth_started) * 1000.0
    if auth_payload.get("_status") != 200:
        raise RuntimeError(f"auth responsiveness smoke failed: {auth_payload}")
    return {
        "requests": requests,
        "status_latency_ms": {
            "min": min(sorted_latencies),
            "p50": sorted_latencies[len(sorted_latencies) // 2],
            "p95": p95,
            "max": max(sorted_latencies),
        },
        "auth_current_user_latency_ms": auth_latency_ms,
    }


def _verify_runner_metrics() -> dict[str, Any]:
    raw = _http_text(_RUNNER_METRICS_URL)
    required = (
        "backtest_runner_tasks_claimed_total",
        "backtest_runner_last_success_unixtime",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise RuntimeError(f"runner metrics endpoint is missing: {missing}")
    return {
        "url": _RUNNER_METRICS_URL,
        "required_metrics": list(required),
        "bytes": len(raw.encode("utf-8")),
    }


def _verify_prometheus_target() -> dict[str, Any]:
    query = urllib.parse.urlencode({"query": 'up{job="backtest-job-runner"}'})
    payload = json.loads(_http_text(f"{_PROMETHEUS_QUERY_URL}?{query}"))
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload}")
    result = payload.get("data", {}).get("result", [])
    if not isinstance(result, list) or not result:
        raise RuntimeError("Prometheus has no backtest-job-runner up{} result")
    values = [item.get("value", [None, "0"])[1] for item in result if isinstance(item, Mapping)]
    if "1" not in values:
        raise RuntimeError(f"Prometheus backtest-job-runner target is not up: {result}")
    return {"query": 'up{job="backtest-job-runner"}', "values": values}


def _http_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _scan_runner_logs() -> dict[str, Any]:
    forbidden = (
        "POSTGRES_PASSWORD",
        "KEYCLOAK_CLIENT_SECRET",
        "BEGIN PRIVATE KEY",
        '"trades":[',
        '"trades": [',
    )
    inspected: dict[str, int] = {}
    findings: list[str] = []
    for path in _RUNNER_LOG_FILES:
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]
        inspected[str(path)] = len(lines)
        joined = "\n".join(lines)
        findings.extend(pattern for pattern in forbidden if pattern in joined)
    if findings:
        raise RuntimeError(f"runner logs contain forbidden patterns: {sorted(set(findings))}")
    return {"files": inspected, "forbidden_patterns_found": []}


def _backlog_snapshot(*, dsn: str) -> dict[str, int]:
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    coalesce(sum(CASE WHEN state = 'queued' THEN 1 ELSE 0 END), 0) AS queued,
                    coalesce(sum(CASE WHEN state = 'running' THEN 1 ELSE 0 END), 0) AS running
                FROM backtest_jobs
                """
            )
            jobs = dict(cast(Mapping[str, Any], cursor.fetchone()))
            cursor.execute(
                """
                SELECT
                    coalesce(sum(CASE WHEN status = 'queued' THEN 1 ELSE 0 END), 0) AS queued,
                    coalesce(sum(CASE WHEN status = 'running' THEN 1 ELSE 0 END), 0) AS running
                FROM backtest_lazy_trades_materializations
                """
            )
            lazy = dict(cast(Mapping[str, Any], cursor.fetchone()))
    return {
        "full_jobs_queued": int(jobs["queued"]),
        "full_jobs_running": int(jobs["running"]),
        "lazy_detail_queued": int(lazy["queued"]),
        "lazy_detail_running": int(lazy["running"]),
    }


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{name} must be a list")
    return value


def _format_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return str(value)


if __name__ == "__main__":
    main()
