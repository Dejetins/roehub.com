from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import quantiles
from typing import Any, Mapping, cast
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_backtests_router
from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCostEstimate,
    BacktestPreflightResult,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobListQuery,
    BacktestJobRepository,
    ResearchOrganizationScope,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestPreflightService,
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
)
from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobErrorPayload,
    BacktestJobStageAShortlist,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import OrganizationId, PaidLevel, UserId

_ORGANIZATION_ID = OrganizationId.from_string("00000000-0000-0000-0000-000000000001")


class _ScopeResolver:
    def resolve(self, *, user_id: UserId) -> ResearchOrganizationScope:
        return ResearchOrganizationScope(
            organization_id=_ORGANIZATION_ID,
            user_id=user_id,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".codex/tmp/stage_8_5_create_path_load_smoke.json"),
    )
    args = parser.parse_args()
    if args.requests <= 0:
        raise SystemExit("--requests must be > 0")

    repository = _Repository()
    trigger = _Trigger()
    client = _build_client(repository=repository, trigger=trigger)

    latencies_ms: list[float] = []
    states: list[str] = []
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for index in range(args.requests):
        started = time.perf_counter()
        response = client.post(
            "/backtests/jobs",
            headers={"x-user-id": "00000000-0000-0000-0000-000000000501"},
            json=_request(index=index),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)
        if response.status_code != 201:
            raise RuntimeError(f"unexpected status {response.status_code}: {response.text[:500]}")
        states.append(str(response.json()["state"]))

    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    sorted_latencies = sorted(latencies_ms)
    p95 = (
        quantiles(sorted_latencies, n=100)[94]
        if len(sorted_latencies) >= 100
        else max(sorted_latencies)
    )
    payload = {
        "requests": args.requests,
        "jobs_created": len(repository.jobs),
        "trigger_calls": len(trigger.calls),
        "states": sorted(set(states)),
        "executor_configured": False,
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_to_wall_ratio": cpu_seconds / wall_seconds if wall_seconds > 0 else None,
        "latency_ms": {
            "min": min(sorted_latencies),
            "p50": sorted_latencies[len(sorted_latencies) // 2],
            "p95": p95,
            "max": max(sorted_latencies),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


def _build_client(*, repository: "_Repository", trigger: "_Trigger") -> TestClient:
    defaults_provider = YamlBacktestGridDefaultsProvider.from_yaml(
        config_path="configs/prod/indicators.yaml"
    )
    runtime_config = BacktestRuntimeConfig(
        hit_times_tp_levels_pct=tuple(i / 2 for i in range(1, 101)),
        hit_times_sl_levels_pct=tuple(i / 2 for i in range(1, 51)),
        artifact_config_hash="a" * 64,
    )
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_backtests_router(
            runtime_defaults_service=BacktestRuntimeDefaultsService(
                defaults_provider=defaults_provider,
                runtime_config=runtime_config,
            ),
            preflight_service=cast(BacktestPreflightService, _PreflightService()),
            current_user_dependency=_current_user,
            organization_scope_resolver=_ScopeResolver(),
            jobs_use_case=BacktestJobsUseCase(
                job_repository=cast(BacktestJobRepository, repository),
                preflight_service=cast(BacktestPreflightService, _PreflightService()),
                runtime_config=runtime_config,
                organization_scope_resolver=_ScopeResolver(),
                execution_trigger=trigger,
            ),
        )
    )
    return TestClient(app)


def _current_user(request: Request) -> CurrentUserPrincipal:
    raw_user_id = request.headers.get("x-user-id")
    if raw_user_id is None:
        raise HTTPException(status_code=401, detail={"error": "unauthorized"})
    return CurrentUserPrincipal(
        user_id=UserId.from_string(raw_user_id),
        paid_level=PaidLevel.free(),
    )


class _PreflightService:
    def execute(self, payload: Mapping[str, Any]) -> BacktestPreflightResult:
        request = dict(payload)
        return BacktestPreflightResult(
            normalized_request=request,
            request_hash=_hash_for_index(value=request.get("nonce")),
            result_config_hash="e" * 64,
            artifact_metadata=BacktestArtifactMetadata(
                artifact_slot="slot_a",
                artifact_slot_generation=4,
                artifact_manifest_hash="a" * 64,
                artifact_asof_date="2026-03-25",
                hit_times_manifest_hash="b" * 64,
                published_at_utc="2026-03-25T02:00:00Z",
            ),
            cost_estimate=BacktestCostEstimate(
                indicator_rows=1,
                candidate_combinations=1,
                tp_sl_cells=0,
                cost_class="small",
            ),
        )


@dataclass
class _Trigger:
    calls: list[UUID] = field(default_factory=list)

    def enqueue(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        user_id: UserId,
        request_hash: str,
    ) -> None:
        _ = organization_id, user_id, request_hash
        self.calls.append(job_id)


@dataclass
class _Repository:
    jobs: dict[UUID, BacktestJob] = field(default_factory=dict)

    def create(self, *, job: BacktestJob) -> BacktestJob:
        self.jobs[job.job_id] = job
        return job

    def find_by_idempotency_key(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        idempotency_key_hash: str,
        created_after: datetime,
    ) -> BacktestJob | None:
        _ = organization_id, user_id, idempotency_key_hash, created_after
        return None

    def get(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        user_id: UserId | None = None,
    ) -> BacktestJob | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.organization_id != organization_id:
            return None
        if user_id is not None and job.user_id != user_id:
            return None
        return job

    def list_for_user(self, *, query: BacktestJobListQuery) -> BacktestJobListPage:
        items = tuple(
            job
            for job in self.jobs.values()
            if job.organization_id == query.organization_id and job.user_id == query.user_id
        )
        return BacktestJobListPage(items=items[: query.limit], next_cursor=None)

    def list_top_variants(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        limit: int | None = None,
    ) -> tuple[BacktestJobTopVariant, ...]:
        _ = job_id, organization_id, limit
        return ()

    def get_top_variant_by_public_key(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        public_variant_key: str,
    ) -> BacktestJobTopVariant | None:
        _ = job_id, organization_id, public_variant_key
        return None

    def cancel(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        user_id: UserId,
        cancel_requested_at: datetime,
    ) -> BacktestJob | None:
        job = self.get(
            job_id=job_id,
            organization_id=organization_id,
            user_id=user_id,
        )
        if job is None:
            return None
        cancelled = job.request_cancel(changed_at=cancel_requested_at)
        self.jobs[job_id] = cancelled
        return cancelled

    def create_with_top_variants(
        self,
        *,
        job: BacktestJob,
        top_variants: tuple[BacktestJobTopVariant, ...],
        stage_a_shortlist: BacktestJobStageAShortlist | None = None,
    ) -> BacktestJob:
        _ = top_variants, stage_a_shortlist
        return self.create(job=job)

    def finish_with_top_variants(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        user_id: UserId,
        now: datetime,
        locked_by: str,
        next_state: BacktestJobState,
        top_variants: tuple[BacktestJobTopVariant, ...],
        last_error: str | None = None,
        last_error_json: BacktestJobErrorPayload | None = None,
    ) -> BacktestJob | None:
        _ = locked_by, top_variants
        job = self.get(
            job_id=job_id,
            organization_id=organization_id,
            user_id=user_id,
        )
        if job is None:
            return None
        finished = job.finish(
            next_state=next_state,
            changed_at=now,
            last_error=last_error,
            last_error_json=last_error_json,
        )
        self.jobs[job_id] = finished
        return finished

    def count_active_for_user(self, *, organization_id: OrganizationId, user_id: UserId) -> int:
        _ = organization_id, user_id
        return 0

    def count_active_global(self) -> int:
        return 0

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


def _request(*, index: int) -> dict[str, Any]:
    return {
        "nonce": index,
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
        "indicators": [
            {
                "indicator_id": "ma.dema",
                "sources": ["close"],
                "window": {"start": 5, "stop": 10, "step": 1},
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
        "top_n": 100,
    }


def _hash_for_index(*, value: Any) -> str:
    index = int(value) if isinstance(value, int) else 0
    return f"{index:064x}"[-64:]


if __name__ == "__main__":
    main()
