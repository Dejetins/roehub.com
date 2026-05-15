from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--preflight-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--scheduling-class", required=True)
    parser.add_argument("--light-max-actual-combinations", required=True)
    args = parser.parse_args()
    _ = args.preflight_json, args.scheduling_class, args.light_max_actual_combinations
    mode = os.environ.get("FAKE_CHILD_MODE", "success")
    if mode == "failure":
        return 7
    if mode == "timeout":
        time.sleep(10)
        return 0
    if mode == "cancel_wait":
        time.sleep(10)
        return 0
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "promote":
        payload = {
            "status": "promote_to_heavy",
            "estimated_combinations_upper_bound": 10,
            "actual_combinations": 100000,
            "reason": "light_candidate_exceeded_actual_threshold",
        }
    else:
        payload = {
            "status": "succeeded",
            "top_variants": [
                {
                    "job_id": args.job_id,
                    "rank": 1,
                    "variant_key": "a" * 64,
                    "indicator_variant_key": "b" * 64,
                    "variant_index": 0,
                    "total_return_pct": 12.5,
                    "payload_json": {"public_variant_key": "job_1"},
                    "summary_metrics_json": {"total_return_pct": 12.5},
                    "best_tp_pct": None,
                    "best_sl_pct": None,
                    "updated_at": "2026-05-13T00:00:00Z",
                }
            ],
            "stage_timings": {"service_wall_clock_s": 0.1},
            "summary_hash": "c" * 64,
            "cleanup_evidence": {"worker_recycle_strategy": "disposable child process"},
        }
    output_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
