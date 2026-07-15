from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from apps.worker.backtest_job_runner.wiring.modules.lazy_trades_compute import (
    build_lazy_trades_compute_service,
)
from trading.contexts.backtest.adapters.outbound import (
    PostgresBacktestJobRepository,
    PsycopgBacktestPostgresGateway,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import OrganizationId, UserId

log = logging.getLogger(__name__)
_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backtest-lazy-trades-child")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--organization-id", required=True)
    parser.add_argument("--owner-user-id", required=True)
    parser.add_argument("--variant-key", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)
    task_id = UUID(args.task_id)
    job_id = UUID(args.job_id)
    organization_id = OrganizationId.from_string(args.organization_id)
    owner_user_id = UserId.from_string(args.owner_user_id)
    output_path = Path(args.output_json)
    started_at = datetime.now(UTC)
    log.info(
        "lazy trades child process started: task_id=%s job_id=%s",
        task_id,
        job_id,
    )
    try:
        postgres_dsn = os.environ.get(_STRATEGY_PG_DSN_KEY, "").strip()
        if not postgres_dsn:
            raise ValueError(f"{_STRATEGY_PG_DSN_KEY} is required for lazy trades child")
        job_repository = PostgresBacktestJobRepository(
            gateway=PsycopgBacktestPostgresGateway(dsn=postgres_dsn)
        )
        job = job_repository.get(
            job_id=job_id,
            organization_id=organization_id,
            user_id=owner_user_id,
        )
        if job is None:
            raise RoehubError(
                code="backtest.job_not_found",
                message="Backtest job was not found for lazy trades materialization",
                details={
                    "job_id": str(job_id),
                    "task_id": str(task_id),
                    "retryable": False,
                },
            )
        row = job_repository.get_top_variant_by_public_key(
            job_id=job_id,
            organization_id=organization_id,
            public_variant_key=args.variant_key,
        )
        if row is None:
            raise RoehubError(
                code="backtest.variant_not_found",
                message="Backtest variant was not found for lazy trades materialization",
                details={
                    "job_id": str(job_id),
                    "variant_key": args.variant_key,
                    "task_id": str(task_id),
                    "retryable": False,
                },
            )
        detail = build_lazy_trades_compute_service(environ=os.environ).execute(
            job=job,
            row=row,
            public_variant_key=args.variant_key,
        )
        payload = {
            "cache_status": str(detail.cache.get("status", "unknown")),
            "cache_path": detail.cache.get("cache_path"),
        }
    except Exception:  # noqa: BLE001
        log.exception("lazy trades child process failed: task_id=%s job_id=%s", task_id, job_id)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    elapsed = (datetime.now(UTC) - started_at).total_seconds()
    log.info(
        "lazy trades child process finished: task_id=%s cache_status=%s " "elapsed_seconds=%.3f",
        task_id,
        payload["cache_status"],
        elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
