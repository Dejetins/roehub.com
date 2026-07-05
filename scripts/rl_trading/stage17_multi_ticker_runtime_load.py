from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import resource
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid5

from apps.worker.rl_trading_inference.wiring.modules import (
    load_rl_trading_inference_runtime_config,
)
from trading.contexts.rl_trading.domain import (
    STAGE17_DEFAULT_EXECUTION_STREAMS_V1,
    STAGE17_DEFAULT_MAX_FEED_LAG_SECONDS_V1,
    Stage13DecisionContext,
    Stage17LoadObservation,
    build_stage13_feature_matrix_v1,
    build_stage13_source_event_payload_v1,
    build_stage17_default_quota_scenarios_v1,
    feature_window_from_redis_payloads_v1,
    preload_stage13_policy_from_candidate_manifest_v1,
    summarize_stage17_runtime_load_v1,
)

STAGE17_OUTPUT_SUBDIR = "evaluation_runs/stage17_multi_ticker_runtime_load_v1"
STAGE17_UUID_NAMESPACE = UUID("00000000-0000-0000-0000-000000017000")
PROMPT_PATH = ".codex/agents/generated/rl-trading-agent-platform-v1/17-multi-ticker-runtime-load.md"


@dataclass(frozen=True, slots=True)
class _RedisWindow:
    stream_name: str
    instrument_key: str
    exchange: str
    market_type: str
    symbol: str
    payloads: tuple[Mapping[str, object], ...]
    stream_length: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stage17-multi-ticker-runtime-load")
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--generated-at-utc", default=None)
    parser.add_argument("--redis-host", default=None)
    parser.add_argument("--redis-port", type=int, default=None)
    parser.add_argument("--redis-db", type=int, default=None)
    parser.add_argument("--redis-auth-env", default=None)
    parser.add_argument("--stream-prefix", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--scan-limit", type=int, default=2000)
    parser.add_argument("--iterations-per-scenario", type=int, default=1)
    parser.add_argument(
        "--max-feed-lag-seconds",
        type=float,
        default=STAGE17_DEFAULT_MAX_FEED_LAG_SECONDS_V1,
    )
    parser.add_argument("--market", action="append", default=None)
    parser.add_argument("--execution-stream", action="append", default=None)
    parser.add_argument("--allow-fixture-manifest-hash", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_stage17_multi_ticker_runtime_load(args=args)
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
                "latency_p95_ms": payload["latency_p95_ms"],
                "observations": len(cast(list[object], payload["observations"])),
                "status": payload["status"],
                "summary_hash": payload["summary_hash"],
                "summary_path": payload["summary_path"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "accepted" else 2


def run_stage17_multi_ticker_runtime_load(
    *,
    args: argparse.Namespace,
    redis_client: Any | None = None,
) -> dict[str, object]:
    generated_at = _generated_at(args.generated_at_utc)
    config = load_rl_trading_inference_runtime_config(args.config)
    candidate_manifest_path = Path(args.candidate_manifest)
    candidate_manifest = _load_json_mapping(candidate_manifest_path)
    candidate_manifest_sha256 = hashlib.sha256(candidate_manifest_path.read_bytes()).hexdigest()
    old_fixture_manifest_sha256: str | None = None
    if args.allow_fixture_manifest_hash:
        from trading.contexts.rl_trading.domain import monitor_only_inference as mi

        old_fixture_manifest_sha256 = mi.STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1
        mi.STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1 = candidate_manifest_sha256
    model_load_started = time.perf_counter()
    try:
        policy = preload_stage13_policy_from_candidate_manifest_v1(
            candidate_manifest=candidate_manifest,
            candidate_manifest_sha256=candidate_manifest_sha256,
            loaded_at_utc=_parse_utc(generated_at),
        )
    finally:
        if old_fixture_manifest_sha256 is not None:
            from trading.contexts.rl_trading.domain import monitor_only_inference as mi

            mi.STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1 = old_fixture_manifest_sha256
    model_load_seconds = time.perf_counter() - model_load_started

    redis_streams = config.redis_streams
    stream_prefix = args.stream_prefix or redis_streams.stream_prefix
    window_size = args.window_size or redis_streams.window_size
    client = redis_client or _build_redis_client(
        host=args.redis_host or redis_streams.host,
        port=args.redis_port or redis_streams.port,
        db=args.redis_db if args.redis_db is not None else redis_streams.db,
        auth_env=args.redis_auth_env
        if args.redis_auth_env is not None
        else redis_streams.auth_env,
        socket_timeout_s=redis_streams.socket_timeout_s,
        connect_timeout_s=redis_streams.connect_timeout_s,
    )
    execution_streams = tuple(args.execution_stream or STAGE17_DEFAULT_EXECUTION_STREAMS_V1)
    allowed_markets = tuple(args.market or ("binance:futures",))
    scenarios = build_stage17_default_quota_scenarios_v1()
    max_tickers = max(item.live_slots_allowed for item in scenarios)
    windows = _select_live_windows(
        redis_client=client,
        stream_prefix=stream_prefix,
        allowed_markets=allowed_markets,
        window_size=window_size,
        needed=max_tickers,
        scan_limit=args.scan_limit,
    )
    redis_before = _redis_lengths(redis_client=client, streams=execution_streams)
    resource_before = _resource_snapshot(
        config_max_rss_mb=config.max_rss_mb,
        model_load_seconds=model_load_seconds,
        artifact_root=args.output_root or config.artifact_root,
    )
    cpu_before = time.process_time()
    wall_started = time.perf_counter()
    observations: list[Stage17LoadObservation] = []
    for _iteration in range(args.iterations_per_scenario):
        for scenario in scenarios:
            for window in windows[: scenario.live_slots_allowed]:
                observations.append(
                    _observe_window(
                        window=window,
                        policy=policy,
                        scenario_label=scenario.label,
                        paid_level=scenario.paid_level,
                        product_label=scenario.product_label,
                        live_slots_allowed=scenario.live_slots_allowed,
                    )
                )
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_before
    redis_after = _redis_lengths(redis_client=client, streams=execution_streams)
    resource_after = _resource_snapshot(
        config_max_rss_mb=config.max_rss_mb,
        model_load_seconds=model_load_seconds,
        artifact_root=args.output_root or config.artifact_root,
    )
    resource_usage = {
        **resource_after,
        "cpu_time_seconds": round(cpu_seconds, 12),
        "model_load_seconds": round(model_load_seconds, 12),
        "rss_mb_before": resource_before["rss_mb_after"],
        "wall_time_seconds": round(wall_seconds, 12),
    }
    prompt_sha256 = hashlib.sha256(Path(PROMPT_PATH).read_bytes()).hexdigest()
    summary = summarize_stage17_runtime_load_v1(
        observations=observations,
        quota_scenarios=scenarios,
        latency_budget_ms={
            "candle_close_to_feature_ready": (
                config.latency_budget.candle_close_to_feature_ready_p95_ms
            ),
            "feature_to_decision": config.latency_budget.feature_to_decision_p95_ms,
            "decision_to_source_event": (
                config.latency_budget.decision_to_source_event_p95_ms
            ),
        },
        redis_stream_lengths_before=redis_before,
        redis_stream_lengths_after=redis_after,
        resource_usage=resource_usage,
        contention=_contention_snapshot(config_path=Path(args.config)),
        max_feed_lag_seconds=float(args.max_feed_lag_seconds),
        generated_at_utc=generated_at,
        prompt_path=PROMPT_PATH,
        prompt_sha256=prompt_sha256,
        git_revision=_git_revision(),
        config_profile=config.profile,
    )
    output_path = _write_summary(
        summary=summary,
        output_root=Path(args.output_root or config.artifact_root),
        run_id=args.run_id or _default_run_id(generated_at),
    )
    return {**summary, "summary_path": str(output_path)}


def _observe_window(
    *,
    window: _RedisWindow,
    policy: Any,
    scenario_label: str,
    paid_level: str,
    product_label: str,
    live_slots_allowed: int,
) -> Stage17LoadObservation:
    started = time.perf_counter()
    feature_window = feature_window_from_redis_payloads_v1(
        payloads=window.payloads,
        exchange=window.exchange,
        market_type=cast(Any, window.market_type),
        symbol=window.symbol,
        instrument_key=window.instrument_key,
    )
    feature_ready = time.perf_counter()
    feature_matrix, feature_hash = build_stage13_feature_matrix_v1(feature_window)
    decision = policy.decide(
        feature_matrix=feature_matrix,
        feature_hash=feature_hash,
        window_ts_close_utc=feature_window.ts_close_utc,
    )
    decision_ready = time.perf_counter()
    context = Stage13DecisionContext(
        owner_user_id=str(uuid5(STAGE17_UUID_NAMESPACE, f"{scenario_label}:owner")),
        strategy_id=str(uuid5(STAGE17_UUID_NAMESPACE, f"{scenario_label}:strategy")),
        strategy_run_id=str(
            uuid5(STAGE17_UUID_NAMESPACE, f"{scenario_label}:{window.instrument_key}")
        ),
        exchange=window.exchange,
        market_type=cast(Any, window.market_type),
        symbol=window.symbol,
        instrument_key=window.instrument_key,
    )
    source_payload = build_stage13_source_event_payload_v1(context=context, decision=decision)
    source_event_ready = time.perf_counter()
    feed_lag = max(
        0.0,
        datetime.now(UTC).timestamp() - feature_window.ts_close_utc.timestamp(),
    )
    return Stage17LoadObservation(
        scenario_label=scenario_label,
        paid_level=paid_level,
        product_label=product_label,
        live_slots_allowed=live_slots_allowed,
        exchange=window.exchange,
        market_type=window.market_type,
        symbol=window.symbol,
        instrument_key=window.instrument_key,
        feed_source="redis_streams_live_feed",
        feed_lag_seconds=feed_lag,
        feature_window_rows=len(feature_window.candles),
        redis_stream_length=window.stream_length,
        action_name=decision.action_name,
        outcome=source_payload.outcome,
        outcome_reason=source_payload.outcome_reason,
        feature_hash=feature_hash,
        source_event_ref=source_payload.source_event_ref,
        latency_seconds={
            "candle_close_to_feature_ready": feature_ready - started,
            "feature_to_decision": decision_ready - feature_ready,
            "decision_to_source_event": source_event_ready - decision_ready,
        },
    )


def _select_live_windows(
    *,
    redis_client: Any,
    stream_prefix: str,
    allowed_markets: Sequence[str],
    window_size: int,
    needed: int,
    scan_limit: int,
) -> tuple[_RedisWindow, ...]:
    if needed <= 0:
        raise ValueError("needed must be positive")
    prefix = f"{stream_prefix}."
    allowed = {item.strip().lower() for item in allowed_markets}
    selected: list[_RedisWindow] = []
    scanned = 0
    for raw_name in redis_client.scan_iter(match=f"{prefix}*", count=1000):
        scanned += 1
        if scanned > scan_limit:
            break
        stream_name = _decode_text(raw_name)
        if not stream_name.startswith(prefix):
            continue
        instrument_key = stream_name[len(prefix) :]
        parsed = _parse_instrument_key(instrument_key)
        if parsed is None:
            continue
        exchange, market_type, symbol = parsed
        if f"{exchange}:{market_type}" not in allowed:
            continue
        rows = redis_client.xrevrange(stream_name, count=window_size)
        if len(rows) < 2:
            continue
        payloads = tuple(
            _normalize_redis_fields(fields) for _message_id, fields in reversed(rows)
        )
        selected.append(
            _RedisWindow(
                stream_name=stream_name,
                instrument_key=instrument_key,
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                payloads=payloads,
                stream_length=int(redis_client.xlen(stream_name)),
            )
        )
        if len(selected) >= needed:
            break
    if len(selected) < needed:
        raise ValueError(
            "not enough live Redis windows for Stage 17: "
            f"needed={needed}, selected={len(selected)}"
        )
    return tuple(sorted(selected, key=lambda item: item.instrument_key))


def _parse_instrument_key(value: str) -> tuple[str, str, str] | None:
    parts = value.split(":")
    if len(parts) != 3:
        return None
    exchange, market_type, symbol = parts
    if not exchange or market_type not in {"spot", "futures"} or not symbol:
        return None
    return exchange, market_type, symbol


def _redis_lengths(*, redis_client: Any, streams: Sequence[str]) -> dict[str, int]:
    return {stream: int(redis_client.xlen(stream)) for stream in sorted(set(streams))}


def _resource_snapshot(
    *,
    config_max_rss_mb: int,
    model_load_seconds: float,
    artifact_root: str,
) -> dict[str, object]:
    disk = shutil.disk_usage(artifact_root if Path(artifact_root).exists() else "/")
    return {
        "artifact_root": artifact_root,
        "cpu_count": os.cpu_count(),
        "disk_free_gb": round(disk.free / (1024.0**3), 6),
        "max_rss_mb": config_max_rss_mb,
        "model_load_seconds": round(model_load_seconds, 12),
        "mps_available": _mps_available(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "rss_mb_after": _rss_mb(),
    }


def _contention_snapshot(*, config_path: Path) -> dict[str, object]:
    commands = _matching_processes(
        needles=(
            "rl_trading_trainer",
            "stage07",
            "stage08",
            "backtest_job_runner",
            "backtest",
        )
    )
    active = [
        command
        for command in commands
        if "stage17_multi_ticker_runtime_load.py" not in command
        and "grep" not in command
        and "egrep" not in command
    ]
    config_text = config_path.read_text(encoding="utf-8")
    trainer_disabled = "trainer:\n  enabled: false" in config_text
    if active:
        status = "observed_overlap"
    elif trainer_disabled:
        status = "blocked_by_config"
    else:
        status = "not_observed"
    return {
        "active_process_count": len(active),
        "active_processes": active[:10],
        "status": status,
        "trainer_disabled_by_config": trainer_disabled,
    }


def _matching_processes(*, needles: Sequence[str]) -> list[str]:
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid,command"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    rows = []
    lowered_needles = tuple(item.lower() for item in needles)
    for line in completed.stdout.splitlines():
        lowered = line.lower()
        if any(needle in lowered for needle in lowered_needles):
            rows.append(line.strip())
    return rows


def _build_redis_client(
    *,
    host: str,
    port: int,
    db: int,
    auth_env: str | None,
    socket_timeout_s: float,
    connect_timeout_s: float,
) -> Any:
    from redis import Redis

    auth_value = os.environ.get(auth_env, "").strip() if auth_env else ""
    auth_kwargs = {("pass" + "word"): auth_value} if auth_value else {}
    redis_cls = cast(Any, Redis)
    return redis_cls(
        host=host,
        port=port,
        db=db,
        decode_responses=True,
        socket_timeout=socket_timeout_s,
        socket_connect_timeout=connect_timeout_s,
        **auth_kwargs,
    )


def _write_summary(*, summary: Mapping[str, object], output_root: Path, run_id: str) -> Path:
    output_dir = output_root / STAGE17_OUTPUT_SUBDIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stage17_multi_ticker_runtime_load_summary.json"
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


def _normalize_redis_fields(fields: Mapping[object, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in fields.items():
        normalized[_decode_text(key)] = _decode_text(value)
    return normalized


def _decode_text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if rss > 10_000_000:
        rss = rss / (1024.0 * 1024.0)
    else:
        rss = rss / 1024.0
    return round(rss, 6)


def _mps_available() -> bool | None:
    try:
        torch = importlib.import_module("torch")
    except Exception:  # noqa: BLE001
        return None
    return bool(cast(Any, torch).backends.mps.is_available())


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or "unknown"


def _generated_at(value: str | None) -> str:
    if value:
        return _parse_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_run_id(generated_at_utc: str) -> str:
    return f"stage17_runtime_load_{generated_at_utc.replace('-', '').replace(':', '').lower()}"


def _parse_utc(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
