from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, cast
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_backtests_router
from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCostEstimate,
    BacktestPreflightResult,
    BacktestRuntimeGuardrails,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobListQuery,
)
from trading.contexts.backtest.application.services.v2 import BacktestRuntimeConfig
from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobErrorPayload,
    BacktestJobStageAShortlist,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import PaidLevel, UserId


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_stage_8_5_create_path_load_smoke",
        description="Local Stage 8.5 load smoke for bounded backtest job create path.",
    )
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.requests <= 0:
        raise ValueError("--requests must be > 0")

    trigger = _Trigger()
    repository = _Repository()
    app = FastAPI()
    register_api_error_handlers(app=app)
    use_case = BacktestJobsUseCase(
        job_repository=repository,
        preflight_service=cast(Any, _PreflightService()),
        runtime_config=BacktestRuntimeConfig(
            hit_times_tp_levels_pct=(2.0,),
            hit_times_sl_levels_pct=(1.0,),
            artifact_config_hash="e" * 64,
            guardrails=BacktestRuntimeGuardrails(
                max_active_jobs_per_user=args.requests + 1,
                max_queued_jobs_per_user=args.requests + 1,
                max_active_jobs_global=args.requests + 1,
            ),
        ),
        execution_trigger=trigger,
    )
    app.include_router(
        build_backtests_router(
            runtime_defaults_service=cast(Any, _RuntimeDefaultsService()),
            preflight_service=cast(Any, _PreflightService()),
            current_user_dependency=_current_user,
            jobs_use_case=use_case,
        )
    )
    client = TestClient(app)

    latencies_ms: list[float] = []
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for index in range(args.requests):
        start = time.perf_counter()
        response = client.post(
            "/backtests/jobs",
            headers={
                "x-user-id": "00000000-0000-0000-0000-000000008501",
                "Idempotency-Key": f"stage-8-5-load-{index}",
            },
            json=_request(),
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        if response.status_code != 201:
            raise AssertionError(
                f"unexpected response at request {index}: "
                f"{response.status_code} {response.text}"
            )
        payload = response.json()
        if payload["state"] != "queued":
            raise AssertionError(f"expected queued job, got {payload['state']!r}")
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start

    jobs = repository.jobs
    assert jobs is not None
    summary = {
        "requests": args.requests,
        "jobs_created": len(jobs),
        "trigger_calls": len(trigger.job_ids),
        "executor_configured": False,
        "states": sorted({job.state for job in jobs.values()}),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_to_wall_ratio": cpu_seconds / wall_seconds if wall_seconds > 0 else None,
        "latency_ms": {
            "min": min(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p95": _percentile(values=latencies_ms, percentile=0.95),
            "max": max(latencies_ms),
        },
    }
    if summary["jobs_created"] != args.requests:
        raise AssertionError("jobs_created mismatch")
    if summary["trigger_calls"] != args.requests:
        raise AssertionError("trigger_calls mismatch")
    if summary["latency_ms"]["p95"] > 25.0:
        raise AssertionError(f"p95 create latency too high: {summary['latency_ms']['p95']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def _percentile(*, values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _current_user(request: Request) -> CurrentUserPrincipal:
    raw_user_id = request.headers.get("x-user-id")
    if raw_user_id is None:
        raise HTTPException(status_code=401, detail={"error": "unauthorized"})
    return CurrentUserPrincipal(
        user_id=UserId.from_string(raw_user_id),
        paid_level=PaidLevel.free(),
    )


@dataclass
class _RuntimeDefaultsService:
    def execute(self) -> Any:
        raise AssertionError("runtime defaults are not part of create load smoke")


@dataclass
class _PreflightService:
    counter: int = 0

    def execute(self, payload: Mapping[str, Any]) -> BacktestPreflightResult:
        self.counter += 1
        request_hash = f"{self.counter:064x}"
        return BacktestPreflightResult(
            normalized_request=dict(payload),
            request_hash=request_hash,
            result_config_hash="e" * 64,
            artifact_metadata=_artifact_metadata(),
            cost_estimate=BacktestCostEstimate(
                indicator_rows=1,
                candidate_combinations=1,
                tp_sl_cells=0,
                cost_class="small",
            ),
        )


@dataclass
class _Trigger:
    job_ids: tuple[UUID, ...] = ()

    def enqueue(self, *, job: BacktestJob) -> None:
        self.job_ids = (*self.job_ids, job.job_id)


@dataclass
class _Repository:
    jobs: dict[UUID, BacktestJob] | None = None

    def __post_init__(self) -> None:
        if self.jobs is None:
            self.jobs = {}

    def create(self, *, job: BacktestJob) -> BacktestJob:
        assert self.jobs is not None
        self.jobs[job.job_id] = job
        return job

    def create_with_top_variants(
        self,
        *,
        job: BacktestJob,
        top_variants: tuple[BacktestJobTopVariant, ...],
        stage_a_shortlist: BacktestJobStageAShortlist | None = None,
    ) -> BacktestJob:
        _ = top_variants, stage_a_shortlist
        return self.create(job=job)

    def find_by_idempotency_key(
        self,
        *,
        user_id: UserId,
        idempotency_key_hash: str,
        created_after: datetime,
    ) -> BacktestJob | None:
        assert self.jobs is not None
        for job in self.jobs.values():
            idempotency = dict(job.request_json).get("idempotency")
            if (
                job.user_id == user_id
                and job.created_at >= created_after
                and isinstance(idempotency, dict)
                and idempotency.get("key_hash") == idempotency_key_hash
            ):
                return job
        return None

    def get(self, *, job_id: UUID, user_id: UserId | None = None) -> BacktestJob | None:
        assert self.jobs is not None
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if user_id is not None and job.user_id != user_id:
            return None
        return job

    def list_for_user(self, *, query: BacktestJobListQuery) -> BacktestJobListPage:
        assert self.jobs is not None
        items = tuple(job for job in self.jobs.values() if job.user_id == query.user_id)
        return BacktestJobListPage(items=items[: query.limit], next_cursor=None)

    def list_top_variants(self, *, job_id: UUID) -> tuple[BacktestJobTopVariant, ...]:
        _ = job_id
        return ()

    def get_top_variant_by_public_key(
        self,
        *,
        job_id: UUID,
        public_variant_key: str,
    ) -> BacktestJobTopVariant | None:
        _ = job_id, public_variant_key
        return None

    def cancel(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        cancel_requested_at: datetime,
    ) -> BacktestJob | None:
        job = self.get(job_id=job_id, user_id=user_id)
        if job is None:
            return None
        cancelled = job.request_cancel(changed_at=cancel_requested_at)
        assert self.jobs is not None
        self.jobs[job_id] = cancelled
        return cancelled

    def claim_for_inline_execution(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        now: datetime,
        locked_by: str,
        lease_expires_at: datetime,
    ) -> BacktestJob | None:
        raise AssertionError("sync_inline claim must not be used by create load smoke")

    def finish_with_top_variants(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        now: datetime,
        locked_by: str,
        next_state: BacktestJobState,
        top_variants: tuple[BacktestJobTopVariant, ...],
        last_error: str | None = None,
        last_error_json: BacktestJobErrorPayload | None = None,
    ) -> BacktestJob | None:
        raise AssertionError("worker finish must not be used by create load smoke")

    def count_active_for_user(self, *, user_id: UserId) -> int:
        assert self.jobs is not None
        return sum(1 for job in self.jobs.values() if job.user_id == user_id)

    def count_active_global(self) -> int:
        assert self.jobs is not None
        return len(self.jobs)

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        _ = market_id, symbol, artifact_slot, artifact_manifest_hash
        return 0


def _request() -> dict[str, Any]:
    return {
        "coordinates": {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "BTCUSDT",
        },
        "timeframe": "15m",
        "time_range": {
            "start": "2020-01-11T20:08:00Z",
            "end": "2026-04-11T20:08:00Z",
        },
        "indicators": [{"indicator_id": "ma.dema", "sources": ["close"]}],
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
        "top_n": 100,
    }


def _artifact_metadata() -> BacktestArtifactMetadata:
    return BacktestArtifactMetadata(
        artifact_slot="slot_a",
        artifact_slot_generation=4,
        artifact_manifest_hash="a" * 64,
        artifact_asof_date="2026-03-25",
        hit_times_manifest_hash="b" * 64,
        published_at_utc="2026-03-25T02:00:00Z",
    )


if __name__ == "__main__":
    raise SystemExit(main())
