from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, cast
from uuid import UUID

from prometheus_client import CollectorRegistry

from apps.worker.rl_trading_inference.wiring.modules import (
    RedisRlClosedCandleStream,
    RedisRlFeatureWindowReader,
    RlTradingInferenceHttpServer,
    RlTradingInferenceMetrics,
    Stage08kMonitorWorker,
    load_rl_trading_inference_runtime_config,
)
from trading.contexts.live_execution.adapters.outbound import (
    InMemoryExecutionIntentRepository,
    InMemoryPaperAccountingRepository,
    PostgresExecutionIntentRepository,
    RedisExecutionDispatchTransport,
    RedisExecutionDispatchTransportConfig,
    SystemLiveExecutionClock,
)
from trading.contexts.live_execution.application import (
    CapitalReservationPaperAccountingService,
    ExecutionDispatchConfig,
    ExecutionDispatchResult,
    ExecutionDispatchService,
    ExecutionIngressService,
)
from trading.contexts.live_execution.application.ports import (
    ExecutionDispatchPublishResult,
    ExecutionDispatchTransport,
    ExecutionIntentRepository,
)
from trading.contexts.live_execution.domain import ExecutionIntent, ExecutionRiskContext
from trading.contexts.rl_trading.adapters.outbound import LiveExecutionRlInferenceProducer
from trading.contexts.rl_trading.adapters.outbound.persistence.file_monitor_state import (
    FileStage08kMonitorStateStore,
)
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
    preload_stage08k_monitor_policy_v1,
    preload_stage13_policy_from_candidate_manifest_v1,
)
from trading.contexts.strategy.adapters.outbound import PsycopgStrategyPostgresGateway
from trading.shared_kernel.primitives import OrganizationId

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonLogFormatter())
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        force=True,
    )


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)


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
    canary.add_argument("--organization-id", required=True)
    canary.add_argument("--owner-user-id", required=True)
    canary.add_argument("--strategy-id", required=True)
    canary.add_argument("--strategy-run-id", required=True)

    paper = subparsers.add_parser("paper-once")
    paper.add_argument("--candidate-manifest", required=True)
    paper.add_argument("--feature-window-json", required=True)
    paper.add_argument("--organization-id", required=True)
    paper.add_argument("--owner-user-id", required=True)
    paper.add_argument("--strategy-id", required=True)
    paper.add_argument("--strategy-run-id", required=True)
    paper.add_argument("--quote-notional", required=True)
    paper.add_argument("--reference-price", required=True)

    testnet = subparsers.add_parser("testnet-once")
    testnet.add_argument("--candidate-manifest", required=True)
    testnet.add_argument("--feature-window-json", required=True)
    testnet.add_argument("--organization-id", required=True)
    testnet.add_argument("--owner-user-id", required=True)
    testnet.add_argument("--strategy-id", required=True)
    testnet.add_argument("--strategy-run-id", required=True)
    testnet.add_argument("--exchange-connection-id", required=True)
    testnet.add_argument("--quote-notional", required=True)
    testnet.add_argument("--quantity", default=None)
    testnet.add_argument("--dispatch-mode", choices=("memory", "redis"), default="memory")
    testnet.add_argument("--postgres-dsn-env", default="STRATEGY_PG_DSN")
    testnet.add_argument("--redis-host", default="127.0.0.1")
    testnet.add_argument("--redis-port", type=int, default=6379)
    testnet.add_argument("--redis-db", type=int, default=0)
    testnet.add_argument("--redis-password-env", default="ROEHUB_REDIS_PASSWORD")
    testnet.add_argument("--redis-socket-timeout-s", type=float, default=2.0)
    testnet.add_argument("--redis-connect-timeout-s", type=float, default=2.0)
    testnet.add_argument("--request-stream", default="execution.requests.v1")
    testnet.add_argument("--retry-stream", default="execution.requests.retry.v1")
    testnet.add_argument("--dlq-stream", default="execution.requests.dlq.v1")
    testnet.add_argument("--consumer-group", default="exchange-execution.v1")
    testnet.add_argument("--backpressure-max-stream-length", type=int, default=10_000)

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
                organization_id=args.organization_id,
                owner_user_id=args.owner_user_id,
                strategy_id=args.strategy_id,
                strategy_run_id=args.strategy_run_id,
            )
        if args.command == "paper-once":
            return _run_paper_once(
                candidate_manifest=args.candidate_manifest,
                feature_window_json=args.feature_window_json,
                organization_id=args.organization_id,
                owner_user_id=args.owner_user_id,
                strategy_id=args.strategy_id,
                strategy_run_id=args.strategy_run_id,
                quote_notional=args.quote_notional,
                reference_price=args.reference_price,
            )
        if args.command == "testnet-once":
            return _run_testnet_once(
                candidate_manifest=args.candidate_manifest,
                feature_window_json=args.feature_window_json,
                organization_id=args.organization_id,
                owner_user_id=args.owner_user_id,
                strategy_id=args.strategy_id,
                strategy_run_id=args.strategy_run_id,
                exchange_connection_id=args.exchange_connection_id,
                quote_notional=args.quote_notional,
                quantity=args.quantity,
                dispatch_mode=args.dispatch_mode,
                postgres_dsn_env=args.postgres_dsn_env,
                redis_host=args.redis_host,
                redis_port=args.redis_port,
                redis_db=args.redis_db,
                redis_password_env=args.redis_password_env,
                redis_socket_timeout_s=args.redis_socket_timeout_s,
                redis_connect_timeout_s=args.redis_connect_timeout_s,
                request_stream=args.request_stream,
                retry_stream=args.retry_stream,
                dlq_stream=args.dlq_stream,
                consumer_group=args.consumer_group,
                backpressure_max_stream_length=args.backpressure_max_stream_length,
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
    organization_id: str,
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
        organization_id=OrganizationId.from_string(organization_id),
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


def _run_paper_once(
    *,
    candidate_manifest: str,
    feature_window_json: str,
    organization_id: str,
    owner_user_id: str,
    strategy_id: str,
    strategy_run_id: str,
    quote_notional: str,
    reference_price: str,
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

    execution_repository = InMemoryExecutionIntentRepository()
    paper_repository = InMemoryPaperAccountingRepository()
    clock = SystemLiveExecutionClock()
    ingress = ExecutionIngressService(repository=execution_repository, clock=clock)
    paper_accounting = CapitalReservationPaperAccountingService(
        repository=paper_repository,
        account_projection_repository=None,
        clock=clock,
    )
    producer = LiveExecutionRlInferenceProducer(
        ingress_service=ingress,
        repository=execution_repository,
    )
    context = Stage13DecisionContext(
        owner_user_id=owner_user_id,
        strategy_id=strategy_id,
        strategy_run_id=strategy_run_id,
        exchange=window.exchange,
        market_type=window.market_type,
        symbol=window.symbol,
        instrument_key=window.instrument_key,
    )
    risk_context = _paper_risk_context()
    first = producer.record_paper_decision(
        organization_id=OrganizationId.from_string(organization_id),
        context=context,
        decision=decision,
        risk_context=risk_context,
        paper_accounting_service=paper_accounting,
        quote_notional=Decimal(quote_notional),
        reference_price=Decimal(reference_price),
    )
    replay = producer.record_paper_decision(
        organization_id=OrganizationId.from_string(organization_id),
        context=context,
        decision=decision,
        risk_context=risk_context,
        paper_accounting_service=paper_accounting,
        quote_notional=Decimal(quote_notional),
        reference_price=Decimal(reference_price),
    )
    source_event_ready_at = time.perf_counter()
    parity = _paper_parity_payload(
        quote_notional=Decimal(quote_notional),
        reference_price=Decimal(reference_price),
        result=first,
    )
    result = {
        "action": decision.action_name,
        "decision_id": decision.decision_id,
        "duplicate_replay": replay.duplicate,
        "feature_hash": feature_hash,
        "intents_created": len(execution_repository.intents),
        "model_version_id": decision.model_version_id,
        "outcome": first.event.outcome,
        "outcome_reason": first.event.outcome_reason,
        "paper_accounting_created": len(paper_repository.accounting),
        "paper_fills_created": len(paper_repository.fills),
        "paper_orders_created": len(paper_repository.orders),
        "risk_reason": first.intent.risk_reason if first.intent is not None else None,
        "risk_status": first.intent.risk_status if first.intent is not None else None,
        "simulator_parity": parity,
        "source_event_ref": first.event.source_event_ref,
        "source_events_created": len(execution_repository.source_events),
        "source_type": first.event.source_type,
        "latency_seconds": {
            "candle_close_to_feature_ready": round(feature_ready_at - started_at, 12),
            "feature_to_decision": round(decision_ready_at - feature_ready_at, 12),
            "decision_to_source_event": round(source_event_ready_at - decision_ready_at, 12),
        },
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if _paper_result_accepted(result) else 2


def _run_testnet_once(
    *,
    candidate_manifest: str,
    feature_window_json: str,
    organization_id: str,
    owner_user_id: str,
    strategy_id: str,
    strategy_run_id: str,
    exchange_connection_id: str,
    quote_notional: str,
    quantity: str | None,
    dispatch_mode: str,
    postgres_dsn_env: str,
    redis_host: str,
    redis_port: int,
    redis_db: int,
    redis_password_env: str | None,
    redis_socket_timeout_s: float,
    redis_connect_timeout_s: float,
    request_stream: str,
    retry_stream: str,
    dlq_stream: str,
    consumer_group: str,
    backpressure_max_stream_length: int,
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

    execution_repository, dispatch_transport = _build_testnet_execution_runtime(
        dispatch_mode=dispatch_mode,
        postgres_dsn_env=postgres_dsn_env,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        redis_password_env=redis_password_env,
        redis_socket_timeout_s=redis_socket_timeout_s,
        redis_connect_timeout_s=redis_connect_timeout_s,
        request_stream=request_stream,
        retry_stream=retry_stream,
        dlq_stream=dlq_stream,
        consumer_group=consumer_group,
    )
    clock = SystemLiveExecutionClock()
    ingress = ExecutionIngressService(repository=execution_repository, clock=clock)
    producer = LiveExecutionRlInferenceProducer(
        ingress_service=ingress,
        repository=execution_repository,
    )
    dispatch = ExecutionDispatchService(
        repository=execution_repository,
        transport=dispatch_transport,
        clock=clock,
        config=ExecutionDispatchConfig(
            backpressure_max_stream_length=backpressure_max_stream_length,
        ),
    )
    context = Stage13DecisionContext(
        owner_user_id=owner_user_id,
        strategy_id=strategy_id,
        strategy_run_id=strategy_run_id,
        exchange=window.exchange,
        market_type=window.market_type,
        symbol=window.symbol,
        instrument_key=window.instrument_key,
    )
    first = producer.record_testnet_decision(
        organization_id=OrganizationId.from_string(organization_id),
        context=context,
        decision=decision,
        risk_context=_testnet_risk_context(),
        exchange_connection_id=UUID(exchange_connection_id),
        quote_notional=Decimal(quote_notional),
        quantity=Decimal(quantity) if quantity is not None else None,
    )
    intent_ready_at = time.perf_counter()
    dispatch_result = dispatch.dispatch_intent(intent=first.intent) if first.intent else None
    dispatch_ready_at = time.perf_counter()
    replay = producer.record_testnet_decision(
        organization_id=OrganizationId.from_string(organization_id),
        context=context,
        decision=decision,
        risk_context=_testnet_risk_context(),
        exchange_connection_id=UUID(exchange_connection_id),
        quote_notional=Decimal(quote_notional),
        quantity=Decimal(quantity) if quantity is not None else None,
    )
    replay_dispatch_result = (
        dispatch.dispatch_intent(intent=replay.intent) if replay.intent is not None else None
    )
    result = {
        "action": decision.action_name,
        "decision_id": decision.decision_id,
        "dispatch": _dispatch_payload(dispatch_result),
        "dispatch_mode": dispatch_mode,
        "duplicate_dispatch": _dispatch_payload(replay_dispatch_result),
        "duplicate_replay": replay.duplicate,
        "exchange_connection_id": exchange_connection_id,
        "feature_hash": feature_hash,
        "intent_id": str(first.intent.intent_id) if first.intent is not None else None,
        "intent_status": (
            dispatch_result.intent.status
            if dispatch_result is not None
            else first.intent.status
            if first.intent is not None
            else None
        ),
        "model_version_id": decision.model_version_id,
        "outcome": first.event.outcome,
        "outcome_reason": first.event.outcome_reason,
        "risk_reason": first.intent.risk_reason if first.intent is not None else None,
        "risk_status": first.intent.risk_status if first.intent is not None else None,
        "source_event_ref": first.event.source_event_ref,
        "source_event_id": str(first.event.source_event_id),
        "source_type": first.event.source_type,
        "memory_counts": _memory_repository_counts(execution_repository, dispatch_transport),
        "latency_seconds": {
            "candle_close_to_feature_ready": round(feature_ready_at - started_at, 12),
            "feature_to_decision": round(decision_ready_at - feature_ready_at, 12),
            "decision_to_intent": round(intent_ready_at - decision_ready_at, 12),
            "intent_to_dispatch_attempt": round(dispatch_ready_at - intent_ready_at, 12),
        },
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if _testnet_result_accepted(result) else 2


def _run_serve(
    *,
    config_path: str,
    duration_seconds: float,
    metrics_port: int | None,
) -> int:
    config = load_rl_trading_inference_runtime_config(config_path)
    metrics = RlTradingInferenceMetrics(registry=CollectorRegistry())
    readiness = config.readiness_payload()
    initial_reasons = cast(list[str], readiness["degraded_reasons"])
    if config.enabled:
        initial_reasons = [*initial_reasons, "model_not_loaded"]
    metrics.set_readiness(
        ready=False,
        degraded_reasons=initial_reasons,
    )
    port = config.metrics_port if metrics_port is None else metrics_port
    server = RlTradingInferenceHttpServer(metrics=metrics, port=port)
    server.start()
    try:
        if not config.enabled:
            if duration_seconds > 0:
                time.sleep(duration_seconds)
                return 0
            while True:
                time.sleep(60.0)
        dsn = os.environ.get(config.postgres_dsn_env, "").strip()
        if not dsn:
            raise ValueError(f"{config.postgres_dsn_env} is required for monitor source events")
        policy = preload_stage08k_monitor_policy_v1(
            artifacts=config.artifacts,
            policy_config=config.monitor_policy,
            torch_num_threads=config.torch_num_threads,
            torch_num_interop_threads=config.torch_num_interop_threads,
        )
        metrics.set_model_loaded(loaded=True)
        gateway = PsycopgStrategyPostgresGateway(dsn=dsn)
        repository = PostgresExecutionIntentRepository(gateway=gateway)
        producer = LiveExecutionRlInferenceProducer(
            ingress_service=ExecutionIngressService(
                repository=repository,
                clock=SystemLiveExecutionClock(),
            ),
            repository=repository,
        )
        instrument_keys = tuple(row.instrument_key for row in config.instruments)
        stream = RedisRlClosedCandleStream.from_runtime_config(
            config=config.redis_streams,
            instrument_keys=instrument_keys,
        )
        window_reader = RedisRlFeatureWindowReader.from_runtime_config(
            config=config.redis_streams,
        )
        state_store = FileStage08kMonitorStateStore(path=config.state_path)
        worker = Stage08kMonitorWorker(
            instruments=config.instruments,
            operator_context=config.operator_context,
            stream=stream,
            window_reader=window_reader,
            policy=policy,
            producer=producer,
            state_store=state_store,
            metrics=metrics,
        )
        metrics.set_readiness(ready=True, degraded_reasons=[])
        log.info(
            "stage08k monitor started model_version_id=%s policy_id=%s instruments=%s",
            policy.model_version_id,
            config.monitor_policy.policy_id,
            len(config.instruments),
        )
        worker.run(duration_seconds=duration_seconds)
        return 0
    finally:
        server.stop()


def _canary_result_accepted(result: Mapping[str, object]) -> bool:
    return (
        result.get("outcome") == STAGE13_SOURCE_EVENT_OUTCOME_V1
        and result.get("outcome_reason") == STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1
        and result.get("intents_created") == 0
    )


def _paper_result_accepted(result: Mapping[str, object]) -> bool:
    parity = result.get("simulator_parity")
    return (
        result.get("source_type") == "ml_agent_decision"
        and result.get("outcome") == "risk_rejected"
        and result.get("outcome_reason") == "paper_no_exchange_submit"
        and result.get("risk_status") == "rejected"
        and result.get("risk_reason") == "paper_no_exchange_submit"
        and result.get("duplicate_replay") is True
        and result.get("source_events_created") == 1
        and result.get("intents_created") == 1
        and result.get("paper_orders_created") == 1
        and result.get("paper_fills_created") == 1
        and result.get("paper_accounting_created") == 1
        and isinstance(parity, Mapping)
        and parity.get("status") == "accepted"
    )


def _testnet_result_accepted(result: Mapping[str, object]) -> bool:
    if result.get("source_type") != "ml_agent_decision":
        return False
    if result.get("duplicate_replay") is not True:
        return False
    if result.get("intent_id") is None:
        reason = str(result.get("outcome_reason") or "")
        return result.get("outcome") == "no_intent" and reason.startswith("testnet_")
    dispatch = result.get("dispatch")
    duplicate_dispatch = result.get("duplicate_dispatch")
    return (
        result.get("outcome") == "intent_created"
        and result.get("risk_status") == "accepted"
        and result.get("risk_reason") == "risk_gate_accepted"
        and isinstance(dispatch, Mapping)
        and dispatch.get("result") == "dispatched"
        and isinstance(duplicate_dispatch, Mapping)
        and duplicate_dispatch.get("result") == "duplicate"
    )


def _build_testnet_execution_runtime(
    *,
    dispatch_mode: str,
    postgres_dsn_env: str,
    redis_host: str,
    redis_port: int,
    redis_db: int,
    redis_password_env: str | None,
    redis_socket_timeout_s: float,
    redis_connect_timeout_s: float,
    request_stream: str,
    retry_stream: str,
    dlq_stream: str,
    consumer_group: str,
) -> tuple[ExecutionIntentRepository, ExecutionDispatchTransport]:
    if dispatch_mode == "memory":
        return InMemoryExecutionIntentRepository(), _MemoryExecutionDispatchTransport(
            request_stream=request_stream,
        )
    if dispatch_mode != "redis":
        raise ValueError("dispatch_mode must be memory or redis")
    dsn = os.environ.get(postgres_dsn_env, "").strip()
    if not dsn:
        raise ValueError(f"{postgres_dsn_env} is required for redis dispatch mode")
    gateway = PsycopgStrategyPostgresGateway(dsn=dsn)
    repository = PostgresExecutionIntentRepository(gateway=gateway)
    transport = RedisExecutionDispatchTransport(
        config=RedisExecutionDispatchTransportConfig(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password_env=(redis_password_env or None),
            socket_timeout_s=redis_socket_timeout_s,
            connect_timeout_s=redis_connect_timeout_s,
            request_stream=request_stream,
            retry_stream=retry_stream,
            dlq_stream=dlq_stream,
            consumer_group=consumer_group,
        ),
        environ=os.environ,
    )
    return repository, transport


def _dispatch_payload(result: ExecutionDispatchResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "intent_status": result.intent.status,
        "message_id": result.intent.dispatch_redis_message_id,
        "reason": result.reason,
        "result": result.result,
        "stream_name": result.intent.dispatch_stream_name,
    }


def _memory_repository_counts(
    repository: ExecutionIntentRepository,
    transport: ExecutionDispatchTransport,
) -> dict[str, int] | None:
    if not isinstance(repository, InMemoryExecutionIntentRepository):
        return None
    message_count = (
        transport.message_count
        if isinstance(transport, _MemoryExecutionDispatchTransport)
        else 0
    )
    return {
        "dispatch_messages": message_count,
        "intents": len(repository.intents),
        "source_events": len(repository.source_events),
    }


def _paper_risk_context() -> ExecutionRiskContext:
    return ExecutionRiskContext(
        organization_ownership_verified=True,
        account_ownership_verified=True,
        exchange_connection_active=True,
        secret_custody_ready=True,
        source_authorized=True,
        strategy_live_profile_ready=True,
        strategy_run_active=True,
        market_data_state="ready",
        position_ownership_active=True,
        capital_reservation_active=True,
        capital_reservation_sufficient=True,
        paper_accounting_ready=True,
        paper_no_exchange_submit=True,
        ml_agent_policy_active=True,
        kill_switch_open=True,
        environment_policy_allows=True,
        max_order_size_ok=True,
        daily_limit_ok=True,
    )


def _testnet_risk_context() -> ExecutionRiskContext:
    # The standalone CLI has no trusted ownership, account-state, limit, or
    # kill-switch resolver. Persist the source event, but fail closed before
    # dispatch instead of promoting command-line input into risk authority.
    return ExecutionRiskContext()


class _MemoryExecutionDispatchTransport:
    def __init__(self, *, request_stream: str) -> None:
        self._request_stream = request_stream
        self._group_ready = False
        self._messages: list[dict[str, str]] = []

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def ensure_request_group(self) -> None:
        self._group_ready = True

    def request_stream_length(self) -> int:
        return len(self._messages)

    def publish_request(
        self, *, intent: ExecutionIntent, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        self._messages.append({"intent_id": str(intent.intent_id)})
        return ExecutionDispatchPublishResult(
            stream_name=self._request_stream,
            message_id=f"1-{len(self._messages)}",
        )

    def publish_retry(
        self, *, intent: ExecutionIntent, reason: str, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        return ExecutionDispatchPublishResult(
            stream_name=f"{self._request_stream}.retry",
            message_id=f"1-{len(self._messages)}",
        )

    def publish_dlq(
        self, *, intent: ExecutionIntent, reason: str, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        return ExecutionDispatchPublishResult(
            stream_name=f"{self._request_stream}.dlq",
            message_id=f"1-{len(self._messages)}",
        )

    def ack_after_durable_state_change(self, *, stream_name: str, message_id: str) -> None:
        return None


def _paper_parity_payload(
    *,
    quote_notional: Decimal,
    reference_price: Decimal,
    result: object,
) -> dict[str, object]:
    accounting = getattr(result, "accounting", None)
    if accounting is None:
        return {"status": "blocked", "reason": "paper_accounting_missing"}
    expected_quantity = (quote_notional / reference_price).quantize(Decimal("0.00000001"))
    expected_fee = (quote_notional * Decimal("10") / Decimal("10000")).quantize(
        Decimal("0.00000001")
    )
    expected_equity = (quote_notional - expected_fee).quantize(Decimal("0.00000001"))
    diffs = {
        "equity": abs(accounting.equity - expected_equity),
        "fee_total": abs(accounting.fee_total - expected_fee),
        "position_quantity": abs(accounting.position_quantity - expected_quantity),
    }
    max_abs_diff = max(diffs.values())
    return {
        "abs_diff": {key: str(value) for key, value in sorted(diffs.items())},
        "max_abs_diff": str(max_abs_diff),
        "status": "accepted" if max_abs_diff == Decimal("0") else "blocked",
        "tolerance": "0",
    }


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
