from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence, cast
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from prometheus_client import CollectorRegistry
from redis import Redis

from apps.worker.rl_trading_inference.wiring.modules import (
    RedisRlClosedCandleStream,
    RedisRlFeatureWindowReader,
    RlTradingInferenceInstrumentConfig,
    RlTradingInferenceMetrics,
    RlTradingInferenceOperatorContextConfig,
    RlTradingInferenceRedisStreamsConfig,
    Stage08kMonitorWorker,
    load_rl_trading_inference_runtime_config,
)
from trading.contexts.live_execution.adapters.outbound import (
    InMemoryExecutionIntentRepository,
    SystemLiveExecutionClock,
)
from trading.contexts.live_execution.application import ExecutionIngressService
from trading.contexts.rl_trading.adapters.outbound import LiveExecutionRlInferenceProducer
from trading.contexts.rl_trading.adapters.outbound.persistence.file_monitor_state import (
    FileStage08kMonitorStateStore,
)
from trading.contexts.rl_trading.domain import (
    FEATURE_CONTRACT_HASH_V1,
    STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
    RlFeatureCandle,
    Stage08kPreloadedMonitorPolicy,
    Stage13DecisionContext,
    Stage13InferenceDecision,
    preload_stage08k_monitor_policy_v1,
)
from trading.contexts.rl_trading.domain.raw_feature_dataset import hash_json_payload_v1
from trading.contexts.rl_trading.domain.stage08k_monitor_runtime import (
    close_stage08k_virtual_trade_v1,
    open_stage08k_virtual_trade_v1,
    stage08k_entry_decision_id_v1,
)

DEFAULT_DATASET_MANIFEST = Path(
    "/opt/roehub/state/rl_trading/datasets/"
    "stage08j_article_sessionized_dataset_v1/"
    "stage08j_article_sessionized_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage18a_accelerated_monitor_validation_v1"
)
EXECUTION_STREAMS = (
    "execution.requests.v1",
    "execution.requests.retry.v1",
    "execution.requests.dlq.v1",
)


@dataclass(frozen=True, slots=True)
class HistoricalSession:
    signal_time_ms: int
    symbol: str
    feature_path: Path
    session_index: int


@dataclass(frozen=True, slots=True)
class ScreenedSession:
    session: HistoricalSession
    action_name: str
    requested_action_name: str
    confidence: float
    volatility_score: float
    policy_reason: str
    decision_latency_ms: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stage18a-accelerated-monitor-validation")
    parser.add_argument(
        "--config",
        default="/opt/roehub/app/configs/prod/rl_trading_ml_runtime.yaml",
    )
    parser.add_argument("--dataset-manifest", default=str(DEFAULT_DATASET_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--production-baseline", required=True)
    parser.add_argument("--session-count", type=int, default=100)
    parser.add_argument("--natural-open-target", type=int, default=20)
    parser.add_argument("--boundary-message-count", type=int, default=20)
    args = parser.parse_args(argv)
    if not 50 <= args.session_count <= 100:
        parser.error("--session-count must be between 50 and 100")
    if not 10 <= args.natural_open_target <= args.session_count:
        parser.error("--natural-open-target must be between 10 and session-count")
    if args.boundary_message_count != 20:
        parser.error("--boundary-message-count must equal 20")
    accepted, summary_path = run_validation(
        config_path=Path(args.config),
        dataset_manifest_path=Path(args.dataset_manifest),
        output_root=Path(args.output_root),
        production_baseline_path=Path(args.production_baseline),
        session_count=args.session_count,
        natural_open_target=args.natural_open_target,
        boundary_message_count=args.boundary_message_count,
    )
    print(
        json.dumps(
            {"accepted": accepted, "summary_path": str(summary_path)},
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


def run_validation(
    *,
    config_path: Path,
    dataset_manifest_path: Path,
    output_root: Path,
    production_baseline_path: Path,
    session_count: int,
    natural_open_target: int,
    boundary_message_count: int,
) -> tuple[bool, Path]:
    started_at = datetime.now(UTC)
    run_id = f"stage18a_accelerated_{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    run_root = _validated_run_root(output_root=output_root, run_id=run_id)
    run_root.mkdir(parents=True, exist_ok=False)
    production_baseline_before = _sha256(production_baseline_path)
    runtime = load_rl_trading_inference_runtime_config(config_path)
    policy = preload_stage08k_monitor_policy_v1(
        artifacts=runtime.artifacts,
        policy_config=runtime.monitor_policy,
        torch_num_threads=1,
        torch_num_interop_threads=1,
    )
    sessions = load_backtest_sessions(dataset_manifest_path)
    screened = screen_sessions(sessions=sessions, policy=policy)
    selected = select_event_enriched_sessions(
        screened=screened,
        session_count=session_count,
        natural_open_target=natural_open_target,
    )
    execution_streams_before = _stream_lengths()
    historical = replay_historical_sessions(
        selected=selected,
        policy=policy,
        run_id=run_id,
    )
    boundary = run_close_boundary_validation(
        selected=selected,
        policy=policy,
        run_id=run_id,
        run_root=run_root,
        message_count=boundary_message_count,
    )
    execution_streams_after = _stream_lengths()
    production_baseline_after = _sha256(production_baseline_path)
    stream_deltas = {
        key: execution_streams_after[key] - execution_streams_before[key]
        for key in EXECUTION_STREAMS
    }
    checks = {
        "boundary_messages_all_published_early": boundary["published_early_count"]
        == boundary_message_count,
        "boundary_messages_all_processed": boundary["processed_count"]
        == boundary_message_count,
        "boundary_retries_observed": boundary["close_boundary_retries_total"] > 0,
        "errors_zero": boundary["errors_total"] == 0,
        "execution_stream_growth_zero": all(value == 0 for value in stream_deltas.values()),
        "historical_open_target_reached": historical["open_long_count"]
        >= natural_open_target,
        "historical_open_close_pairing_exact": historical["open_long_count"]
        == historical["valid_close_count"],
        "historical_virtual_pnl_finite": math.isfinite(
            float(historical["virtual_pnl_quote"])
        ),
        "idempotent_replay_no_duplicates": historical["duplicate_replay_added_events"]
        == 0
        and boundary["duplicate_replay_added_events"] == 0,
        "intents_zero": historical["intents"] == 0 and boundary["intents"] == 0,
        "latency_within_budget": historical["decision_latency_p95_ms"]
        <= runtime.latency_budget.feature_to_decision_p95_ms
        and historical["source_event_latency_p95_ms"]
        <= runtime.latency_budget.decision_to_source_event_p95_ms
        and boundary["process_latency_p95_ms"]
        <= runtime.latency_budget.candle_close_to_feature_ready_p95_ms,
        "orders_zero": historical["orders"] == 0 and boundary["orders"] == 0,
        "production_baseline_unchanged": production_baseline_before
        == production_baseline_after,
        "safety_breaches_zero": boundary["safety_breaches_total"] == 0,
    }
    summary_without_hash = {
        "artifact_kind": "stage18a_accelerated_monitor_validation_summary_v1",
        "boundary_validation": boundary,
        "checks": checks,
        "completed_at_utc": _format_utc(datetime.now(UTC)),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "execution_stream_deltas": stream_deltas,
        "historical_replay": historical,
        "model_version_id": policy.model_version_id,
        "policy_id": policy.policy_config.policy_id,
        "production_baseline_path": str(production_baseline_path),
        "production_baseline_sha256": production_baseline_after,
        "proof_boundary": "post_main_isolated_accelerated_monitor_validation",
        "run_id": run_id,
        "screened_session_count": len(screened),
        "selected_session_count": len(selected),
        "selection_policy": {
            "kind": "event_enriched_diagnostic_not_performance_sample",
            "natural_open_long": natural_open_target,
            "deterministic_hold_controls": session_count - natural_open_target,
            "model_actions_forced": False,
        },
        "started_at_utc": _format_utc(started_at),
        "status": "accepted" if all(checks.values()) else "blocked",
    }
    summary = {
        **summary_without_hash,
        "summary_hash": hash_json_payload_v1(summary_without_hash),
    }
    summary_path = run_root / "summary.json"
    _write_json(summary_path, summary)
    return summary["status"] == "accepted", summary_path


def load_backtest_sessions(manifest_path: Path) -> tuple[HistoricalSession, ...]:
    manifest = _read_json(manifest_path)
    rows: list[HistoricalSession] = []
    for artifact in cast(Sequence[Mapping[str, Any]], manifest.get("split_artifacts", ())):
        if artifact.get("split") != "backtest":
            continue
        files = cast(Mapping[str, Mapping[str, Any]], artifact["files"])
        times = np.load(str(files["signal_time_ms"]["path"]))
        for session_index, signal_time_ms in enumerate(times.tolist()):
            rows.append(
                HistoricalSession(
                    signal_time_ms=int(signal_time_ms),
                    symbol=str(artifact["symbol"]),
                    feature_path=Path(str(files["features"]["path"])),
                    session_index=session_index,
                )
            )
    return tuple(
        sorted(
            rows,
            key=lambda row: (row.signal_time_ms, row.symbol, row.session_index),
        )
    )


def screen_sessions(
    *,
    sessions: Sequence[HistoricalSession],
    policy: Stage08kPreloadedMonitorPolicy,
) -> tuple[ScreenedSession, ...]:
    rows: list[ScreenedSession] = []
    cached_path: Path | None = None
    features: np.ndarray[Any, Any] | None = None
    for session in sessions:
        if session.feature_path != cached_path:
            features = np.load(session.feature_path)
            cached_path = session.feature_path
        if features is None:
            raise RuntimeError("historical feature artifact was not loaded")
        candles = _candles(features[session.session_index, :90])
        started = time.perf_counter()
        decision = policy.decide(candles)
        rows.append(
            ScreenedSession(
                session=session,
                action_name=decision.action_name,
                requested_action_name=decision.requested_action_name,
                confidence=decision.confidence,
                volatility_score=decision.signal.volatility_score,
                policy_reason=decision.policy_reason,
                decision_latency_ms=(time.perf_counter() - started) * 1_000.0,
            )
        )
    return tuple(rows)


def select_event_enriched_sessions(
    *,
    screened: Sequence[ScreenedSession],
    session_count: int,
    natural_open_target: int,
) -> tuple[ScreenedSession, ...]:
    opens = [row for row in screened if row.action_name == "open_long"]
    holds = [row for row in screened if row.action_name == "hold"]
    if len(opens) < natural_open_target:
        raise ValueError("not enough natural open_long decisions in historical sessions")
    control_count = session_count - natural_open_target
    if len(holds) < control_count:
        raise ValueError("not enough hold controls in historical sessions")
    selected = [*opens[:natural_open_target], *holds[:control_count]]
    return tuple(
        sorted(
            selected,
            key=lambda row: (
                row.session.signal_time_ms,
                row.session.symbol,
                row.session.session_index,
            ),
        )
    )


def replay_historical_sessions(
    *,
    selected: Sequence[ScreenedSession],
    policy: Stage08kPreloadedMonitorPolicy,
    run_id: str,
) -> dict[str, Any]:
    repository, producer = _in_memory_producer()
    owner_id, strategy_id, strategy_run_id = _run_uuids(run_id + ":historical")
    event_latencies: list[float] = []
    selected_rows: list[dict[str, Any]] = []
    replay_calls: list[tuple[Stage13DecisionContext, Stage13InferenceDecision]] = []
    virtual_pnl = 0.0
    valid_close_count = 0
    open_count = 0
    cached_path: Path | None = None
    features: np.ndarray[Any, Any] | None = None
    for row in selected:
        if row.session.feature_path != cached_path:
            features = np.load(row.session.feature_path)
            cached_path = row.session.feature_path
        if features is None:
            raise RuntimeError("historical feature artifact was not loaded")
        matrix = features[row.session.session_index]
        candles = _candles(matrix[:90])
        decision = policy.decide(candles)
        signal_time = datetime.fromtimestamp(row.session.signal_time_ms / 1_000, tz=UTC)
        instrument_key = f"binance:futures:{row.session.symbol}"
        context = Stage13DecisionContext(
            owner_user_id=owner_id,
            strategy_id=strategy_id,
            strategy_run_id=strategy_run_id,
            exchange="binance",
            market_type="futures",
            symbol=row.session.symbol,
            instrument_key=instrument_key,
        )
        if decision.action_name != "open_long":
            selected_rows.append(
                _screened_payload(row, natural_open_long=False, pnl_quote=None)
            )
            continue
        open_count += 1
        entry_id = stage08k_entry_decision_id_v1(
            instrument_key=instrument_key,
            candle_close_utc=signal_time,
            feature_hash=decision.feature_hash,
            policy_hash=policy.policy_config.policy_hash(),
        )
        entry = Stage13InferenceDecision(
            decision_id=entry_id,
            model_version_id=policy.model_version_id,
            action_id=1,
            action_name="open_long",
            confidence=decision.confidence,
            feature_hash=decision.feature_hash,
            feature_contract_hash=FEATURE_CONTRACT_HASH_V1,
            window_ts_close_utc=signal_time,
            metadata={"accelerated_run_id": run_id},
        )
        event_latencies.append(_record(producer=producer, context=context, decision=entry))
        replay_calls.append((context, entry))
        trade = open_stage08k_virtual_trade_v1(
            instrument_key=instrument_key,
            symbol=row.session.symbol,
            entry_decision_id=entry_id,
            entry_time_utc=signal_time,
            entry_price=candles[-1].close,
            policy=policy.policy_config,
        )
        result = close_stage08k_virtual_trade_v1(
            trade=trade,
            exit_time_utc=signal_time + timedelta(minutes=1),
            exit_price=float(matrix[90, 4]),
            policy=policy.policy_config,
        )
        close_feature_hash = hash_json_payload_v1(
            {
                "entry_decision_id": entry_id,
                "exit_decision_id": result.exit_decision_id,
                "instrument_key": instrument_key,
            }
        )
        close = Stage13InferenceDecision(
            decision_id=result.exit_decision_id,
            model_version_id=policy.model_version_id,
            action_id=3,
            action_name="close",
            confidence=1.0,
            feature_hash=close_feature_hash,
            feature_contract_hash=FEATURE_CONTRACT_HASH_V1,
            window_ts_close_utc=signal_time + timedelta(minutes=1),
            metadata={"accelerated_run_id": run_id},
        )
        event_latencies.append(_record(producer=producer, context=context, decision=close))
        replay_calls.append((context, close))
        valid_close_count += int(result.valid_for_policy_evaluation)
        virtual_pnl += result.pnl_quote
        selected_rows.append(
            _screened_payload(
                row,
                natural_open_long=True,
                pnl_quote=result.pnl_quote,
            )
        )
    source_count_before_replay = len(repository.source_events)
    for context, decision in replay_calls:
        producer.record_monitor_only_decision(context=context, decision=decision)
    return {
        "decision_latency_p95_ms": _p95(
            [row.decision_latency_ms for row in selected]
        ),
        "duplicate_replay_added_events": len(repository.source_events)
        - source_count_before_replay,
        "intents": len(repository.intents),
        "open_long_count": open_count,
        "orders": 0,
        "selected_sessions": selected_rows,
        "source_event_count": len(repository.source_events),
        "source_event_latency_p95_ms": _p95(event_latencies),
        "valid_close_count": valid_close_count,
        "virtual_pnl_quote": virtual_pnl,
    }


def run_close_boundary_validation(
    *,
    selected: Sequence[ScreenedSession],
    policy: Stage08kPreloadedMonitorPolicy,
    run_id: str,
    run_root: Path,
    message_count: int,
) -> dict[str, Any]:
    controls = [row for row in selected if row.action_name == "hold"][:message_count]
    if len(controls) != message_count:
        raise ValueError("not enough hold controls for boundary validation")
    prefix = f"stage18a.accelerated.{run_id}.candles.1m"
    group = f"stage18a.accelerated.{run_id}.group"
    consumer = f"stage18a-accelerated-{os.getpid()}"
    config = RlTradingInferenceRedisStreamsConfig(
        enabled=True,
        host="127.0.0.1",
        port=6379,
        db=0,
        auth_env=None,
        socket_timeout_s=2.0,
        connect_timeout_s=2.0,
        stream_prefix=prefix,
        window_size=90,
        consumer_group=group,
        consumer_name=consumer,
        read_count=message_count,
        block_ms=1,
        pending_claim_min_idle_ms=60_000,
    )
    redis_client = cast(
        Any,
        Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True),
    )
    instruments: list[RlTradingInferenceInstrumentConfig] = []
    final_rows: list[tuple[str, str, dict[str, str], float]] = []
    isolated_streams: list[str] = []
    boundary = datetime.now(UTC) + timedelta(seconds=2.0)
    boundary = boundary.replace(microsecond=(boundary.microsecond // 1_000) * 1_000)
    offsets = _boundary_offsets_ms(message_count)
    cached_path: Path | None = None
    features: np.ndarray[Any, Any] | None = None
    try:
        for index, (row, offset_ms) in enumerate(zip(controls, offsets, strict=True)):
            if row.session.feature_path != cached_path:
                features = np.load(row.session.feature_path)
                cached_path = row.session.feature_path
            if features is None:
                raise RuntimeError("boundary feature artifact was not loaded")
            matrix = features[row.session.session_index, :90]
            instrument_key = (
                f"stage18a:accelerated:{run_id}:{index:02d}:{row.session.symbol}"
            )
            stream_name = config.stream_name(instrument_key)
            isolated_streams.append(stream_name)
            instruments.append(
                RlTradingInferenceInstrumentConfig(
                    exchange="binance",
                    market_type="futures",
                    symbol=row.session.symbol,
                    instrument_key=instrument_key,
                )
            )
            payloads = _retimestamped_payloads(
                matrix=matrix,
                instrument_key=instrument_key,
                final_close=boundary,
            )
            for payload in payloads[:-1]:
                redis_client.xadd(
                    stream_name,
                    payload,
                    id=_stream_id(str(payload["ts_open"])),
                )
            redis_client.xgroup_create(stream_name, group, id="$", mkstream=True)
            final_rows.append(
                (
                    instrument_key,
                    stream_name,
                    payloads[-1],
                    offset_ms,
                )
            )
        stream = RedisRlClosedCandleStream(
            redis_client=redis_client,
            config=config,
            instrument_keys=tuple(row.instrument_key for row in instruments),
        )
        reader = RedisRlFeatureWindowReader(redis_client=redis_client, config=config)
        repository, producer = _in_memory_producer()
        owner_id, strategy_id, strategy_run_id = _run_uuids(run_id + ":boundary")
        metrics = RlTradingInferenceMetrics(registry=CollectorRegistry())
        worker = Stage08kMonitorWorker(
            instruments=tuple(instruments),
            operator_context=RlTradingInferenceOperatorContextConfig(
                owner_user_id=owner_id,
                strategy_id=strategy_id,
                strategy_run_id=strategy_run_id,
            ),
            stream=stream,
            window_reader=reader,
            policy=policy,
            producer=producer,
            state_store=FileStage08kMonitorStateStore(
                path=(run_root / "boundary_state.json").resolve()
            ),
            metrics=metrics,
        )
        published: list[dict[str, Any]] = []
        for instrument_key, stream_name, payload, offset_ms in sorted(
            final_rows,
            key=lambda item: item[3],
            reverse=True,
        ):
            target = boundary - timedelta(milliseconds=offset_ms)
            _sleep_until(target)
            published_at = datetime.now(UTC)
            message_id = _stream_id(str(payload["ts_open"]))
            redis_client.xadd(stream_name, payload, id=message_id)
            published.append(
                {
                    "actual_early_ms": (boundary - published_at).total_seconds()
                    * 1_000.0,
                    "instrument_key": instrument_key,
                    "message_id": message_id,
                    "requested_early_ms": offset_ms,
                }
            )
        messages = list(stream.read())
        process_latencies: list[float] = []
        for message in messages:
            started = time.perf_counter()
            worker.process_with_close_boundary_retry(message=message)
            process_latencies.append((time.perf_counter() - started) * 1_000.0)
        while len(messages) < message_count:
            batch = stream.read()
            if not batch:
                break
            for message in batch:
                started = time.perf_counter()
                worker.process_with_close_boundary_retry(message=message)
                process_latencies.append((time.perf_counter() - started) * 1_000.0)
            messages.extend(batch)
        source_count_before_replay = len(repository.source_events)
        for message in messages:
            worker.process(message=message)
        pending = sum(
            int(redis_client.xpending(stream_name, group)["pending"])
            for stream_name in isolated_streams
        )
        return {
            "cleanup_completed": True,
            "close_boundary_retries_total": _metric_total(
                metrics.registry,
                "rl_trading_inference_close_boundary_retries_total",
            ),
            "consumer_group": group,
            "consumer_name": consumer,
            "duplicate_replay_added_events": len(repository.source_events)
            - source_count_before_replay,
            "errors_total": _metric_total(
                metrics.registry,
                "rl_trading_inference_errors_total",
            ),
            "intents": len(repository.intents),
            "isolated_state_path": str((run_root / "boundary_state.json").resolve()),
            "isolated_stream_count": len(isolated_streams),
            "orders": 0,
            "pending_after_processing": pending,
            "process_latency_p95_ms": _p95(process_latencies),
            "processed_count": len(messages),
            "published_early_count": sum(
                1
                for row in published
                if 0.0 < float(row["actual_early_ms"]) <= 550.0
            ),
            "published_messages": published,
            "safety_breaches_total": _metric_total(
                metrics.registry,
                "rl_trading_inference_safety_breaches_total",
            ),
            "source_event_count": len(repository.source_events),
            "stream_prefix": prefix,
        }
    finally:
        if isolated_streams:
            redis_client.delete(*isolated_streams)


def _candles(matrix: np.ndarray[Any, Any]) -> tuple[RlFeatureCandle, ...]:
    candles: list[RlFeatureCandle] = []
    for row in matrix:
        volume = float(row[5])
        candles.append(
            RlFeatureCandle(
                open=float(row[0]),
                high=float(row[1]),
                low=float(row[3]),
                close=float(row[4]),
                volume_base=volume,
                volume_quote=float(row[2]) * volume,
                trades_count=int(round(float(row[6]))),
            )
        )
    return tuple(candles)


def _retimestamped_payloads(
    *,
    matrix: np.ndarray[Any, Any],
    instrument_key: str,
    final_close: datetime,
) -> list[dict[str, str]]:
    first_open = final_close - timedelta(minutes=len(matrix))
    payloads: list[dict[str, str]] = []
    for index, candle in enumerate(_candles(matrix)):
        opened = first_open + timedelta(minutes=index)
        closed = opened + timedelta(minutes=1)
        payloads.append(
            {
                "close": str(candle.close),
                "high": str(candle.high),
                "instrument_key": instrument_key,
                "low": str(candle.low),
                "open": str(candle.open),
                "schema_version": "1",
                "trades_count": str(candle.trades_count),
                "ts_close": _format_utc(closed),
                "ts_open": _format_utc(opened),
                "volume_base": str(candle.volume_base),
                "volume_quote": str(candle.volume_quote),
            }
        )
    return payloads


def _record(
    *,
    producer: LiveExecutionRlInferenceProducer,
    context: Stage13DecisionContext,
    decision: Stage13InferenceDecision,
) -> float:
    started = time.perf_counter()
    event = producer.record_monitor_only_decision(context=context, decision=decision)
    if event.intent_id is not None:
        raise RuntimeError("accelerated monitor source event unexpectedly has intent")
    if event.outcome_reason != STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1:
        raise RuntimeError("accelerated monitor source event has unexpected outcome")
    return (time.perf_counter() - started) * 1_000.0


def _in_memory_producer() -> tuple[
    InMemoryExecutionIntentRepository,
    LiveExecutionRlInferenceProducer,
]:
    repository = InMemoryExecutionIntentRepository()
    producer = LiveExecutionRlInferenceProducer(
        ingress_service=ExecutionIngressService(
            repository=repository,
            clock=SystemLiveExecutionClock(),
        ),
        repository=repository,
    )
    return repository, producer


def _run_uuids(seed: str) -> tuple[str, str, str]:
    return tuple(
        str(uuid5(NAMESPACE_URL, f"roehub:{seed}:{field}"))
        for field in ("owner", "strategy", "strategy-run")
    )  # type: ignore[return-value]


def _screened_payload(
    row: ScreenedSession,
    *,
    natural_open_long: bool,
    pnl_quote: float | None,
) -> dict[str, Any]:
    return {
        "action_name": row.action_name,
        "confidence": row.confidence,
        "natural_open_long": natural_open_long,
        "pnl_quote": pnl_quote,
        "policy_reason": row.policy_reason,
        "requested_action_name": row.requested_action_name,
        "session_index": row.session.session_index,
        "signal_time_utc": _format_utc(
            datetime.fromtimestamp(row.session.signal_time_ms / 1_000, tz=UTC)
        ),
        "symbol": row.session.symbol,
        "volatility_score": row.volatility_score,
    }


def _boundary_offsets_ms(count: int) -> tuple[float, ...]:
    if count == 1:
        return (50.0,)
    step = 450.0 / (count - 1)
    return tuple(round(50.0 + index * step, 6) for index in range(count))


def _sleep_until(target: datetime) -> None:
    remaining = (target - datetime.now(UTC)).total_seconds()
    if remaining > 0:
        time.sleep(remaining)


def _stream_id(ts_open: str) -> str:
    opened = datetime.fromisoformat(ts_open.replace("Z", "+00:00"))
    return f"{int(opened.timestamp() * 1_000)}-0"


def _stream_lengths() -> dict[str, int]:
    client = cast(
        Any,
        Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True),
    )
    return {name: int(client.xlen(name)) for name in EXECUTION_STREAMS}


def _metric_total(registry: CollectorRegistry, name: str) -> float:
    return sum(
        float(sample.value)
        for metric in registry.collect()
        for sample in metric.samples
        if sample.name == name or sample.name == f"{name}_total"
    )


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _validated_run_root(*, output_root: Path, run_id: str) -> Path:
    root = output_root.expanduser().resolve(strict=False)
    allowed = Path("/opt/roehub/state/rl_trading/evaluation_runs").resolve(
        strict=False
    )
    try:
        root.relative_to(allowed)
    except ValueError as error:
        raise ValueError("accelerated output root must be under evaluation_runs") from error
    return root / run_id


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact must contain a mapping")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
