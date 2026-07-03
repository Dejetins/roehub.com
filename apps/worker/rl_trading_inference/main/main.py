from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

from prometheus_client import CollectorRegistry

from apps.worker.rl_trading_inference.wiring.modules import (
    RlTradingInferenceHttpServer,
    RlTradingInferenceMetrics,
    load_rl_trading_inference_runtime_config,
)
from trading.contexts.live_execution.adapters.outbound import (
    InMemoryExecutionIntentRepository,
    SystemLiveExecutionClock,
)
from trading.contexts.live_execution.application import ExecutionIngressService
from trading.contexts.rl_trading.adapters.outbound import LiveExecutionRlInferenceProducer
from trading.contexts.rl_trading.domain import (
    STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
    STAGE13_SOURCE_EVENT_OUTCOME_V1,
    RlFeatureCandle,
    Stage13DecisionContext,
    Stage13FeatureWindow,
    build_stage13_feature_matrix_v1,
    compare_stage13_train_live_feature_parity_v1,
    feature_window_from_redis_payloads_v1,
    offline_feature_window_from_candles_v1,
    preload_stage13_policy_from_candidate_manifest_v1,
)

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-trading-inference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--config", required=True)

    parity = subparsers.add_parser("parity")
    parity.add_argument("--live-window-json", required=True)
    parity.add_argument("--offline-window-json", required=True)

    canary = subparsers.add_parser("canary-once")
    canary.add_argument("--candidate-manifest", required=True)
    canary.add_argument("--feature-window-json", required=True)
    canary.add_argument("--owner-user-id", required=True)
    canary.add_argument("--strategy-id", required=True)
    canary.add_argument("--strategy-run-id", required=True)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--config", required=True)
    serve.add_argument("--duration-seconds", type=float, default=0.0)
    serve.add_argument("--metrics-port", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "status":
            return _run_status(config_path=args.config)
        if args.command == "parity":
            return _run_parity(
                live_window_json=args.live_window_json,
                offline_window_json=args.offline_window_json,
            )
        if args.command == "canary-once":
            return _run_canary_once(
                candidate_manifest=args.candidate_manifest,
                feature_window_json=args.feature_window_json,
                owner_user_id=args.owner_user_id,
                strategy_id=args.strategy_id,
                strategy_run_id=args.strategy_run_id,
            )
        if args.command == "serve":
            return _run_serve(
                config_path=args.config,
                duration_seconds=args.duration_seconds,
                metrics_port=args.metrics_port,
            )
    except Exception:  # noqa: BLE001
        log.exception("rl-trading-inference command failed")
        return 1
    return 2


def _run_status(*, config_path: str) -> int:
    config = load_rl_trading_inference_runtime_config(config_path)
    print(json.dumps(config.readiness_payload(), ensure_ascii=True, sort_keys=True))
    return 0


def _run_parity(*, live_window_json: str, offline_window_json: str) -> int:
    live_window = _load_feature_window(Path(live_window_json))
    offline_window = _load_feature_window(Path(offline_window_json))
    result = compare_stage13_train_live_feature_parity_v1(
        live_window=live_window,
        offline_window=offline_window,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("status") == "accepted" else 2


def _run_canary_once(
    *,
    candidate_manifest: str,
    feature_window_json: str,
    owner_user_id: str,
    strategy_id: str,
    strategy_run_id: str,
) -> int:
    started_at = time.perf_counter()
    manifest_path = Path(candidate_manifest)
    manifest = _load_json_mapping(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    policy = preload_stage13_policy_from_candidate_manifest_v1(
        candidate_manifest=manifest,
        candidate_manifest_sha256=manifest_sha256,
        loaded_at_utc=datetime.now(UTC),
    )
    window = _load_feature_window(Path(feature_window_json))
    feature_ready_at = time.perf_counter()
    feature_matrix, feature_hash = build_stage13_feature_matrix_v1(window)
    decision = policy.decide(
        feature_matrix=feature_matrix,
        feature_hash=feature_hash,
        window_ts_close_utc=window.ts_close_utc,
    )
    decision_ready_at = time.perf_counter()

    repository = InMemoryExecutionIntentRepository()
    ingress = ExecutionIngressService(repository=repository, clock=SystemLiveExecutionClock())
    producer = LiveExecutionRlInferenceProducer(ingress_service=ingress, repository=repository)
    event = producer.record_monitor_only_decision(
        context=Stage13DecisionContext(
            owner_user_id=owner_user_id,
            strategy_id=strategy_id,
            strategy_run_id=strategy_run_id,
            exchange=window.exchange,
            market_type=window.market_type,
            symbol=window.symbol,
            instrument_key=window.instrument_key,
        ),
        decision=decision,
    )
    source_event_ready_at = time.perf_counter()
    result = {
        "action": decision.action_name,
        "decision_id": decision.decision_id,
        "feature_hash": feature_hash,
        "intents_created": len(repository.intents),
        "model_version_id": decision.model_version_id,
        "outcome": event.outcome,
        "outcome_reason": event.outcome_reason,
        "source_event_ref": event.source_event_ref,
        "source_events_created": len(repository.source_events),
        "source_type": event.source_type,
        "latency_seconds": {
            "candle_close_to_feature_ready": round(feature_ready_at - started_at, 12),
            "feature_to_decision": round(decision_ready_at - feature_ready_at, 12),
            "decision_to_source_event": round(source_event_ready_at - decision_ready_at, 12),
        },
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if _canary_result_accepted(result) else 2


def _run_serve(
    *,
    config_path: str,
    duration_seconds: float,
    metrics_port: int | None,
) -> int:
    config = load_rl_trading_inference_runtime_config(config_path)
    metrics = RlTradingInferenceMetrics(registry=CollectorRegistry())
    readiness = config.readiness_payload()
    metrics.set_readiness(
        ready=bool(readiness["ready"]),
        degraded_reasons=cast(list[str], readiness["degraded_reasons"]),
    )
    port = config.metrics_port if metrics_port is None else metrics_port
    server = RlTradingInferenceHttpServer(metrics=metrics, port=port)
    server.start()
    try:
        if duration_seconds > 0:
            time.sleep(duration_seconds)
            return 0
        while True:
            time.sleep(60.0)
    finally:
        server.stop()


def _canary_result_accepted(result: Mapping[str, object]) -> bool:
    return (
        result.get("outcome") == STAGE13_SOURCE_EVENT_OUTCOME_V1
        and result.get("outcome_reason") == STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1
        and result.get("intents_created") == 0
    )


def _load_feature_window(path: Path) -> Stage13FeatureWindow:
    payload = _load_json_mapping(path)
    exchange = str(payload["exchange"])
    market_type = cast(str, payload["market_type"])
    symbol = str(payload["symbol"])
    instrument_key = str(payload["instrument_key"])
    if "payloads" in payload:
        payloads = payload["payloads"]
        if not isinstance(payloads, list):
            raise ValueError("payloads must be a list")
        return feature_window_from_redis_payloads_v1(
            payloads=[_json_mapping(item, "payloads[]") for item in payloads],
            exchange=exchange,
            market_type=cast(Any, market_type),
            symbol=symbol,
            instrument_key=instrument_key,
        )
    candles = payload.get("candles")
    if not isinstance(candles, list):
        raise ValueError("feature window JSON must contain payloads or candles")
    parsed_candles = tuple(
        _feature_candle_from_json(_json_mapping(item, "candles[]")) for item in candles
    )
    return offline_feature_window_from_candles_v1(
        candles=parsed_candles,
        exchange=exchange,
        market_type=cast(Any, market_type),
        symbol=symbol,
        instrument_key=instrument_key,
        ts_open_utc=_parse_utc(str(payload["ts_open"])),
        ts_close_utc=_parse_utc(str(payload["ts_close"])),
    )


def _feature_candle_from_json(payload: Mapping[str, object]) -> RlFeatureCandle:
    return RlFeatureCandle(
        open=_required_float(payload, "open"),
        high=_required_float(payload, "high"),
        low=_required_float(payload, "low"),
        close=_required_float(payload, "close"),
        volume_base=_required_float(payload, "volume_base"),
        volume_quote=_optional_float(payload, "volume_quote"),
        trades_count=_optional_int(payload, "trades_count"),
    )


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _json_mapping(payload, str(path))


def _json_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must contain a JSON object")
    return cast(Mapping[str, Any], value)


def _parse_utc(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("feature window timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _required_float(payload: Mapping[str, object], field: str) -> float:
    return float(str(payload[field]))


def _optional_float(payload: Mapping[str, object], field: str) -> float | None:
    value = payload.get(field)
    return None if value is None else float(str(value))


def _optional_int(payload: Mapping[str, object], field: str) -> int | None:
    value = payload.get(field)
    return None if value is None else int(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
