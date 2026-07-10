from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, LiteralString, cast
from urllib.request import urlopen

import psycopg
import yaml
from redis import Redis

MODEL_VERSION = "stage08k_roehub_native_best_3e033951"
POLICY_ID = "stage08k_long_only_hold_1m_monitor_v1"
EVIDENCE_ROOT = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/" "stage18a_stage08k_monitor_only_runtime_v1"
)
EXECUTION_STREAMS = (
    "execution.requests.v1",
    "execution.requests.retry.v1",
    "execution.requests.dlq.v1",
)
RUNTIME_CONFIG_PATH = Path("/opt/roehub/app/configs/prod/rl_trading_ml_runtime.yaml")


@dataclass(frozen=True, slots=True)
class PhaseRequirements:
    duration_seconds: int
    instrument_count: int
    require_decision: bool
    require_virtual_exit: bool


PHASE_REQUIREMENTS = {
    "one_ticker_1h": PhaseRequirements(
        duration_seconds=3_600,
        instrument_count=1,
        require_decision=False,
        require_virtual_exit=False,
    ),
    "five_ticker_24h": PhaseRequirements(
        duration_seconds=86_400,
        instrument_count=5,
        require_decision=False,
        require_virtual_exit=False,
    ),
    "twenty_ticker_7d": PhaseRequirements(
        duration_seconds=604_800,
        instrument_count=20,
        require_decision=True,
        require_virtual_exit=True,
    ),
}
PREVIOUS_PHASE = {
    "one_ticker_1h": None,
    "five_ticker_24h": "one_ticker_1h",
    "twenty_ticker_7d": "five_ticker_24h",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stage18a-rl-monitor-evidence")
    parser.add_argument("--mode", choices=("baseline", "final"), required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--phase", choices=tuple(PHASE_REQUIREMENTS), default=None)
    parser.add_argument("--previous-summary", default=None)
    parser.add_argument("--minimum-seconds", type=int, default=None)
    args = parser.parse_args(argv)
    if args.mode == "baseline":
        path = create_baseline(
            requested_phase=args.phase,
            previous_summary_path=(Path(args.previous_summary) if args.previous_summary else None),
        )
        print(json.dumps({"baseline_path": str(path)}, sort_keys=True))
        return 0
    if not args.baseline:
        parser.error("--baseline is required for final mode")
    baseline_path = Path(args.baseline)
    baseline = _read_json(baseline_path)
    phase = str(baseline["rollout_phase"])
    if args.phase is not None and args.phase != phase:
        parser.error("--phase must match the baseline rollout_phase")
    requirements = _phase_requirements(phase)
    minimum_seconds = max(args.minimum_seconds or 0, requirements.duration_seconds)
    summary_path, accepted = create_final_summary(
        baseline_path=baseline_path, minimum_seconds=minimum_seconds
    )
    print(
        json.dumps(
            {"accepted": accepted, "summary_path": str(summary_path)},
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


def create_baseline(
    *,
    requested_phase: str | None = None,
    previous_summary_path: Path | None = None,
) -> Path:
    now = datetime.now(UTC)
    runtime = _runtime_config_snapshot()
    phase = str(runtime["rollout_phase"])
    if requested_phase is not None and requested_phase != phase:
        raise ValueError("requested phase does not match production runtime config")
    requirements = _phase_requirements(phase)
    if len(runtime["instrument_keys"]) != requirements.instrument_count:
        raise ValueError("production runtime instrument count does not match rollout phase")
    previous_summary = _load_previous_summary(
        phase=phase,
        previous_summary_path=previous_summary_path,
    )
    run_id = f"stage18a_{phase}_{now.strftime('%Y%m%dT%H%M%SZ')}"
    run_root = EVIDENCE_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    payload = {
        "artifact_kind": f"stage18a_{phase}_baseline_v1",
        "previous_summary_path": (
            str(previous_summary_path) if previous_summary_path is not None else None
        ),
        "previous_summary_sha256": (
            _sha256(previous_summary_path)
            if previous_summary_path is not None and previous_summary is not None
            else None
        ),
        **capture_snapshot(recorded_at=now),
    }
    path = run_root / "baseline.json"
    _write_json(path, payload)
    return path


def create_final_summary(*, baseline_path: Path, minimum_seconds: int) -> tuple[Path, bool]:
    baseline = _read_json(baseline_path)
    final = capture_snapshot(recorded_at=datetime.now(UTC))
    summary = assess_window(
        baseline=baseline,
        final=final,
        minimum_seconds=minimum_seconds,
    )
    summary_without_hash = {
        "artifact_kind": f"stage18a_{baseline['rollout_phase']}_summary_v1",
        "baseline_path": str(baseline_path),
        **summary,
    }
    summary_hash = _payload_hash(summary_without_hash)
    payload = {**summary_without_hash, "summary_hash": summary_hash}
    path = baseline_path.parent / "summary.json"
    _write_json(path, payload)
    return path, payload["status"] == "accepted"


def capture_snapshot(*, recorded_at: datetime) -> dict[str, Any]:
    runtime = _runtime_config_snapshot()
    database_counts = _database_counts()
    process_pid = _process_pid()
    redis_client = Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
    stream_lengths = {name: int(cast(int, redis_client.xlen(name))) for name in EXECUTION_STREAMS}
    metrics_text = urlopen("http://127.0.0.1:9213/metrics", timeout=5).read().decode()
    health = json.loads(urlopen("http://127.0.0.1:9213/health/ready", timeout=5).read())
    prometheus_health = _prometheus_target_health()
    return {
        "database_counts": database_counts,
        "health": health,
        "log_json_valid": _latest_log_is_json(),
        "metrics": _parse_metrics(metrics_text),
        "model_version_id": MODEL_VERSION,
        "policy_id": POLICY_ID,
        "process_rss_mb": _process_rss_mb(process_pid),
        "process_pid": process_pid,
        "prometheus_target_health": prometheus_health,
        "proof_boundary": "post_main_production_runtime_proof",
        "recorded_at_utc": _format_utc(recorded_at),
        "revision": _checkout_revision(),
        "rollout_phase": runtime["rollout_phase"],
        "instrument_keys": runtime["instrument_keys"],
        "runtime_config_sha256": runtime["runtime_config_sha256"],
        "runtime_policy_sha256": _sha256(
            Path(
                "/opt/roehub/app/src/trading/contexts/rl_trading/domain/"
                "stage08k_monitor_policy.py"
            )
        ),
        "stream_lengths": stream_lengths,
    }


def assess_window(
    *, baseline: dict[str, Any], final: dict[str, Any], minimum_seconds: int
) -> dict[str, Any]:
    started = _parse_utc(str(baseline["recorded_at_utc"]))
    finished = _parse_utc(str(final["recorded_at_utc"]))
    elapsed_seconds = (finished - started).total_seconds()
    baseline_metrics = dict(baseline.get("metrics", {}))
    final_metrics = dict(final.get("metrics", {}))
    phase = str(baseline["rollout_phase"])
    requirements = _phase_requirements(phase)
    candle_delta = _metric_sum_matching(
        final_metrics,
        "rl_trading_inference_candles_total",
        ('result="processed"',),
    ) - _metric_sum_matching(
        baseline_metrics,
        "rl_trading_inference_candles_total",
        ('result="processed"',),
    )
    error_delta = _metric_sum(final_metrics, "rl_trading_inference_errors_total") - _metric_sum(
        baseline_metrics, "rl_trading_inference_errors_total"
    )
    safety_delta = _metric_sum(
        final_metrics, "rl_trading_inference_safety_breaches_total"
    ) - _metric_sum(baseline_metrics, "rl_trading_inference_safety_breaches_total")
    decision_delta = _metric_sum(
        final_metrics, "rl_trading_inference_decisions_total"
    ) - _metric_sum(baseline_metrics, "rl_trading_inference_decisions_total")
    valid_virtual_exit_delta = _metric_sum_matching(
        final_metrics,
        "rl_trading_inference_virtual_exits_total",
        ('valid="true"',),
    ) - _metric_sum_matching(
        baseline_metrics,
        "rl_trading_inference_virtual_exits_total",
        ('valid="true"',),
    )
    invalid_virtual_exit_delta = _metric_sum_matching(
        final_metrics,
        "rl_trading_inference_virtual_exits_total",
        ('valid="false"',),
    ) - _metric_sum_matching(
        baseline_metrics,
        "rl_trading_inference_virtual_exits_total",
        ('valid="false"',),
    )
    virtual_pnl_delta = _metric_value(
        final_metrics, "rl_trading_inference_virtual_realized_pnl_quote"
    ) - _metric_value(baseline_metrics, "rl_trading_inference_virtual_realized_pnl_quote")
    database_deltas = {
        key: int(final["database_counts"][key]) - int(baseline["database_counts"][key])
        for key in (
            "source_events",
            "open_long_source_events",
            "close_source_events",
            "intents",
            "orders",
        )
    }
    stream_deltas = {
        key: int(final["stream_lengths"][key]) - int(baseline["stream_lengths"][key])
        for key in EXECUTION_STREAMS
    }
    required_candles = requirements.instrument_count * max(1, (minimum_seconds // 60) - 5)
    checks = {
        "duration_reached": elapsed_seconds >= minimum_seconds,
        "enough_closed_candles": candle_delta >= required_candles,
        "errors_zero": error_delta == 0,
        "health_ready": final.get("health", {}).get("ready") is True,
        "instrument_count_exact": len(final.get("instrument_keys", ()))
        == requirements.instrument_count,
        "instrument_set_unchanged": final.get("instrument_keys") == baseline.get("instrument_keys"),
        "intents_zero": database_deltas["intents"] == 0,
        "log_json_valid": final.get("log_json_valid") is True,
        "model_loaded": _metric_value(final_metrics, "rl_trading_inference_model_loaded") == 1,
        "orders_zero": database_deltas["orders"] == 0,
        "phase_unchanged": final.get("rollout_phase") == phase,
        "prometheus_target_up": final.get("prometheus_target_health") == "up",
        "process_unchanged": final.get("process_pid") == baseline.get("process_pid"),
        "revision_unchanged": final.get("revision") == baseline.get("revision"),
        "rss_within_budget": float(final.get("process_rss_mb", 0.0)) <= 4_096.0,
        "runtime_policy_unchanged": final.get("runtime_policy_sha256")
        == baseline.get("runtime_policy_sha256"),
        "runtime_config_unchanged": final.get("runtime_config_sha256")
        == baseline.get("runtime_config_sha256"),
        "safety_breaches_zero": safety_delta == 0,
        "dispatch_stream_growth_zero": all(value == 0 for value in stream_deltas.values()),
        "invalid_virtual_exits_zero": invalid_virtual_exit_delta == 0,
        "decision_observed_when_required": (
            not requirements.require_decision
            or (decision_delta > 0 and database_deltas["source_events"] > 0)
        ),
        "virtual_exit_observed_when_required": (
            not requirements.require_virtual_exit
            or (
                valid_virtual_exit_delta > 0
                and database_deltas["open_long_source_events"] > 0
                and database_deltas["close_source_events"] > 0
                and math.isfinite(virtual_pnl_delta)
            )
        ),
    }
    return {
        "checks": checks,
        "database_deltas": database_deltas,
        "elapsed_seconds": elapsed_seconds,
        "final": final,
        "metric_deltas": {
            "candles_total": candle_delta,
            "decisions_total": decision_delta,
            "errors_total": error_delta,
            "invalid_virtual_exits_total": invalid_virtual_exit_delta,
            "safety_breaches_total": safety_delta,
            "valid_virtual_exits_total": valid_virtual_exit_delta,
            "virtual_realized_pnl_quote": virtual_pnl_delta,
        },
        "minimum_seconds": minimum_seconds,
        "status": "accepted" if all(checks.values()) else "blocked",
        "stream_deltas": stream_deltas,
    }


def _database_counts() -> dict[str, int]:
    queries: dict[str, LiteralString] = {
        "source_events": """
            SELECT count(*) FROM execution_source_events
            WHERE source_type = 'ml_agent_decision'
              AND source_ref_json ->> 'model_version_id' = %s
        """,
        "open_long_source_events": """
            SELECT count(*) FROM execution_source_events
            WHERE source_type = 'ml_agent_decision'
              AND source_ref_json ->> 'model_version_id' = %s
              AND source_ref_json ->> 'action' = 'open_long'
        """,
        "close_source_events": """
            SELECT count(*) FROM execution_source_events
            WHERE source_type = 'ml_agent_decision'
              AND source_ref_json ->> 'model_version_id' = %s
              AND source_ref_json ->> 'action' = 'close'
        """,
        "intents": """
            SELECT count(*) FROM execution_intents i
            JOIN execution_source_events e ON e.source_event_id = i.source_event_id
            WHERE e.source_type = 'ml_agent_decision'
              AND e.source_ref_json ->> 'model_version_id' = %s
        """,
        "orders": """
            SELECT count(*) FROM execution_orders o
            JOIN execution_intents i ON i.intent_id = o.intent_id
            JOIN execution_source_events e ON e.source_event_id = i.source_event_id
            WHERE e.source_type = 'ml_agent_decision'
              AND e.source_ref_json ->> 'model_version_id' = %s
        """,
    }
    with psycopg.connect(os.environ["STRATEGY_PG_DSN"]) as connection:
        with connection.cursor() as cursor:
            counts: dict[str, int] = {}
            for name, query in queries.items():
                cursor.execute(query, (MODEL_VERSION,))
                row = cursor.fetchone()
                counts[name] = int(row[0]) if row is not None else 0
    return counts


def _prometheus_target_health() -> str:
    payload = json.loads(
        urlopen("http://127.0.0.1:9090/api/v1/targets?state=active", timeout=5).read()
    )
    for target in payload.get("data", {}).get("activeTargets", []):
        if target.get("labels", {}).get("job") == "rl-trading-inference":
            return str(target.get("health", "unknown"))
    return "missing"


def _latest_log_is_json() -> bool:
    paths = (
        Path("/Users/daniildegtyarev/Library/Logs/roehub/rl-trading-inference.err.log"),
        Path("/Users/daniildegtyarev/Library/Logs/roehub/rl-trading-inference.out.log"),
    )
    for path in paths:
        if not path.is_file():
            continue
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        for line in reversed(lines[-20:]):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "timestamp_utc" in payload:
                return True
    return False


def _process_rss_mb(pid: int) -> float:
    rss_kb = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
    return float(rss_kb.strip()) / 1_024.0


def _process_pid() -> int:
    pid = subprocess.check_output(
        ["pgrep", "-f", "apps.worker.rl_trading_inference.main.main serve"],
        text=True,
    ).splitlines()[0]
    return int(pid)


def _parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("rl_trading_inference_"):
            continue
        name, value = line.rsplit(" ", 1)
        metrics[name] = float(value)
    return metrics


def _metric_sum(metrics: dict[str, Any], prefix: str) -> float:
    return sum(float(value) for key, value in metrics.items() if key.startswith(prefix))


def _metric_sum_matching(metrics: dict[str, Any], prefix: str, fragments: tuple[str, ...]) -> float:
    return sum(
        float(value)
        for key, value in metrics.items()
        if key.startswith(prefix) and all(fragment in key for fragment in fragments)
    )


def _metric_value(metrics: dict[str, Any], name: str) -> float:
    return float(metrics.get(name, 0.0))


def _checkout_revision() -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            "/Users/daniildegtyarev/Projects/roehub.com",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def _runtime_config_snapshot() -> dict[str, Any]:
    payload = yaml.safe_load(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("inference"), dict):
        raise ValueError("invalid production RL runtime config")
    inference = payload["inference"]
    instruments = inference.get("instruments")
    if not isinstance(instruments, list):
        raise ValueError("production RL runtime instruments must be a list")
    instrument_keys = sorted(
        str(row["instrument_key"])
        for row in instruments
        if isinstance(row, dict) and row.get("instrument_key")
    )
    if len(instrument_keys) != len(instruments) or len(set(instrument_keys)) != len(
        instrument_keys
    ):
        raise ValueError("production RL runtime instruments are invalid or duplicated")
    return {
        "instrument_keys": instrument_keys,
        "rollout_phase": str(inference.get("rollout_phase", "")),
        "runtime_config_sha256": _sha256(RUNTIME_CONFIG_PATH),
    }


def _phase_requirements(phase: str) -> PhaseRequirements:
    try:
        return PHASE_REQUIREMENTS[phase]
    except KeyError as error:
        raise ValueError(f"unsupported Stage 18A rollout phase: {phase}") from error


def _load_previous_summary(
    *, phase: str, previous_summary_path: Path | None
) -> dict[str, Any] | None:
    expected_previous = PREVIOUS_PHASE[phase]
    if expected_previous is None:
        if previous_summary_path is not None:
            raise ValueError("one_ticker_1h must not provide a previous summary")
        return None
    if previous_summary_path is None:
        raise ValueError(f"{phase} requires an accepted {expected_previous} summary")
    resolved = previous_summary_path.resolve(strict=True)
    try:
        resolved.relative_to(EVIDENCE_ROOT.resolve(strict=True))
    except ValueError as error:
        raise ValueError("previous summary must be under the Stage 18A evidence root") from error
    payload = _read_json(resolved)
    _validate_previous_summary_payload(payload, expected_phase=expected_previous)
    return payload


def _validate_previous_summary_payload(payload: dict[str, Any], *, expected_phase: str) -> None:
    summary_hash = str(payload.get("summary_hash", ""))
    without_hash = {key: value for key, value in payload.items() if key != "summary_hash"}
    if not summary_hash or _payload_hash(without_hash) != summary_hash:
        raise ValueError("previous summary hash validation failed")
    final = payload.get("final")
    actual_phase = final.get("rollout_phase") if isinstance(final, dict) else None
    if actual_phase != expected_phase:
        raise ValueError("previous summary rollout phase mismatch")
    requirements = _phase_requirements(expected_phase)
    if payload.get("status") != "accepted":
        raise ValueError("previous summary is not accepted")
    if float(payload.get("elapsed_seconds", 0.0)) < requirements.duration_seconds:
        raise ValueError("previous summary duration is insufficient")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("previous summary checks are not all accepted")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence payload must be a mapping")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("evidence timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
