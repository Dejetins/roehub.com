from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from scripts.rl_trading.stage17_multi_ticker_runtime_load import (
    run_stage17_multi_ticker_runtime_load,
)
from trading.contexts.rl_trading.domain import (
    build_stage18_default_incident_drills_v1,
    summarize_stage18_monitor_only_technical_soak_v1,
)

STAGE18_OUTPUT_SUBDIR = "evaluation_runs/stage18_soak_incident_drills_v1"
PROMPT_PATH = ".codex/agents/generated/rl-trading-agent-platform-v1/18-rl-soak-incident-drills.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stage18-rl-soak-incident-drills")
    parser.add_argument("--stage17-summary", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--candidate-manifest", default=None)
    parser.add_argument("--output-root", default="/opt/roehub/state/rl_trading")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--generated-at-utc", default=None)
    parser.add_argument("--ui-evidence-json", required=True)
    parser.add_argument("--redis-host", default=None)
    parser.add_argument("--redis-port", type=int, default=None)
    parser.add_argument("--redis-db", type=int, default=None)
    parser.add_argument("--redis-auth-env", default=None)
    parser.add_argument("--stream-prefix", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--scan-limit", type=int, default=2000)
    parser.add_argument("--iterations-per-scenario", type=int, default=1)
    parser.add_argument("--max-feed-lag-seconds", type=float, default=300.0)
    parser.add_argument("--market", action="append", default=None)
    parser.add_argument("--execution-stream", action="append", default=None)
    parser.add_argument("--allow-fixture-manifest-hash", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_stage18_rl_soak_incident_drills(args=args)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                    "status": "blocked",
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "summary_hash": payload["summary_hash"],
                "summary_path": payload["summary_path"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "accepted" else 2


def run_stage18_rl_soak_incident_drills(*, args: argparse.Namespace) -> dict[str, object]:
    generated_at = _generated_at(args.generated_at_utc)
    run_id = args.run_id or _default_run_id(generated_at)
    stage17_summary = _load_or_run_stage17_summary(args=args, generated_at=generated_at)
    ui_evidence = _load_json_mapping_from_text(args.ui_evidence_json, "ui_evidence_json")
    prompt_sha256 = hashlib.sha256(Path(PROMPT_PATH).read_bytes()).hexdigest()
    summary = summarize_stage18_monitor_only_technical_soak_v1(
        stage17_summary=stage17_summary,
        incident_drills=build_stage18_default_incident_drills_v1(),
        ui_evidence=ui_evidence,
        generated_at_utc=generated_at,
        prompt_path=PROMPT_PATH,
        prompt_sha256=prompt_sha256,
        git_revision=_git_revision(),
        run_id=run_id,
    )
    output_path = _write_summary(
        summary=summary,
        output_root=Path(args.output_root),
        run_id=run_id,
    )
    return {**summary, "summary_path": str(output_path)}


def _load_or_run_stage17_summary(
    *, args: argparse.Namespace, generated_at: str
) -> Mapping[str, object]:
    if args.stage17_summary:
        stage17_summary = _load_json_mapping(Path(args.stage17_summary))
        return {**stage17_summary, "summary_path": str(Path(args.stage17_summary))}
    if not args.config or not args.candidate_manifest:
        raise ValueError(
            "either --stage17-summary or --config with --candidate-manifest is required"
        )
    stage17_args = argparse.Namespace(
        allow_fixture_manifest_hash=args.allow_fixture_manifest_hash,
        candidate_manifest=args.candidate_manifest,
        config=args.config,
        execution_stream=args.execution_stream,
        generated_at_utc=generated_at,
        iterations_per_scenario=args.iterations_per_scenario,
        market=args.market,
        max_feed_lag_seconds=args.max_feed_lag_seconds,
        output_root=args.output_root,
        redis_auth_env=args.redis_auth_env,
        redis_db=args.redis_db,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        run_id=f"{args.run_id or _default_run_id(generated_at)}_stage17_input",
        scan_limit=args.scan_limit,
        stream_prefix=args.stream_prefix,
        window_size=args.window_size,
    )
    return run_stage17_multi_ticker_runtime_load(args=stage17_args)


def _write_summary(*, summary: Mapping[str, object], output_root: Path, run_id: str) -> Path:
    output_dir = output_root / STAGE18_OUTPUT_SUBDIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stage18_soak_incident_drills_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(Mapping[str, Any], payload)


def _load_json_mapping_from_text(value: str, field: str) -> Mapping[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return cast(Mapping[str, Any], payload)


def _generated_at(value: str | None) -> str:
    if value:
        return _parse_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_run_id(generated_at_utc: str) -> str:
    return f"stage18_soak_{generated_at_utc.replace('-', '').replace(':', '').lower()}"


def _parse_utc(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
