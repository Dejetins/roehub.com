from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from apps.worker.backtest_job_runner.wiring.modules.child_ipc import (
    child_promotion_to_mapping,
    child_success_to_mapping,
    preflight_from_mapping,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    BacktestJobSchedulingPromotionRequired,
    BacktestSchedulingClass,
    backtest_numba_environ,
)

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backtest-full-job-child")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--preflight-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--scheduling-class",
        choices=("light_candidate", "light", "heavy"),
        required=True,
    )
    parser.add_argument("--light-max-actual-combinations", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)
    job_id = UUID(args.job_id)
    preflight_path = Path(args.preflight_json)
    output_path = Path(args.output_json)
    _ = args.scheduling_class
    scheduling_class: BacktestSchedulingClass = "heavy"
    os.environ.update(
        backtest_numba_environ(environ=os.environ, scheduling_class=scheduling_class)
    )
    started_at = datetime.now(UTC)
    log.info(
        "backtest child process started: job_id=%s scheduling_class=%s "
        "numba_threads=%s numba_thread_source=%s",
        job_id,
        scheduling_class,
        os.environ.get("ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS"),
        os.environ.get("ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE"),
    )
    try:
        with preflight_path.open("r", encoding="utf-8") as handle:
            preflight = preflight_from_mapping(payload=json.load(handle))
        from apps.worker.backtest_job_runner.wiring.modules.full_job_compute import (
            build_full_job_compute_executor,
        )

        executor = build_full_job_compute_executor(environ=os.environ)
        result = executor.execute(
            job_id=job_id,
            preflight=preflight,
            updated_at=datetime.now(UTC),
            scheduling_class=scheduling_class,
            light_max_actual_combinations=args.light_max_actual_combinations,
        )
        payload = child_success_to_mapping(result=result)
    except BacktestJobSchedulingPromotionRequired as error:
        payload = child_promotion_to_mapping(promotion=error.promotion)
    except Exception:  # noqa: BLE001
        log.exception("backtest child process failed: job_id=%s", job_id)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    elapsed = (datetime.now(UTC) - started_at).total_seconds()
    log.info(
        "backtest child process finished: job_id=%s status=%s elapsed_seconds=%.3f",
        job_id,
        payload["status"],
        elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
