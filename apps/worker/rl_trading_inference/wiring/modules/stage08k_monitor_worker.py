from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import replace

from trading.contexts.rl_trading.adapters.outbound import LiveExecutionRlInferenceProducer
from trading.contexts.rl_trading.adapters.outbound.persistence.file_monitor_state import (
    FileStage08kMonitorStateStore,
)
from trading.contexts.rl_trading.domain import (
    FEATURE_CONTRACT_HASH_V1,
    STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
    Stage08kPreloadedMonitorPolicy,
    Stage13DecisionContext,
    Stage13FeatureWindow,
    Stage13InferenceDecision,
    Stage13MonitorOnlyInferenceError,
)
from trading.contexts.rl_trading.domain.raw_feature_dataset import hash_json_payload_v1
from trading.contexts.rl_trading.domain.stage08k_monitor_runtime import (
    Stage08kPendingVirtualTrade,
    close_stage08k_virtual_trade_v1,
    open_stage08k_virtual_trade_v1,
    stage08k_entry_decision_id_v1,
)
from trading.shared_kernel.primitives import OrganizationId

from .rl_trading_inference import (
    RedisRlClosedCandleStream,
    RedisRlFeatureWindowReader,
    RlTradingInferenceInstrumentConfig,
    RlTradingInferenceMetrics,
    RlTradingInferenceOperatorContextConfig,
    RlTradingRedisCandleMessage,
)

log = logging.getLogger(__name__)

_CLOSE_BOUNDARY_RETRY_ATTEMPTS = 20
_CLOSE_BOUNDARY_RETRY_SECONDS = 0.05


class Stage08kMonitorWorker:
    def __init__(
        self,
        *,
        instruments: tuple[RlTradingInferenceInstrumentConfig, ...],
        operator_context: RlTradingInferenceOperatorContextConfig,
        stream: RedisRlClosedCandleStream,
        window_reader: RedisRlFeatureWindowReader,
        policy: Stage08kPreloadedMonitorPolicy,
        producer: LiveExecutionRlInferenceProducer,
        state_store: FileStage08kMonitorStateStore,
        metrics: RlTradingInferenceMetrics,
        close_boundary_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._instruments = {row.instrument_key: row for row in instruments}
        if not self._instruments:
            raise ValueError("Stage08kMonitorWorker requires instruments")
        self._operator_context = operator_context
        self._stream = stream
        self._window_reader = window_reader
        self._policy = policy
        self._producer = producer
        self._state_store = state_store
        self._metrics = metrics
        self._close_boundary_sleep = close_boundary_sleep
        self._metrics.set_pending_virtual_positions(len(state_store.all_pending()))

    def run(self, *, duration_seconds: float = 0.0) -> None:
        started = time.monotonic()
        while duration_seconds <= 0.0 or (time.monotonic() - started) < duration_seconds:
            messages = self._stream.read()
            for message in messages:
                try:
                    self.process_with_close_boundary_retry(message=message)
                except Exception as error:  # noqa: BLE001
                    self._metrics.observe_error(
                        operation="process_candle",
                        reason=type(error).__name__,
                    )
                    log.exception(
                        "stage08k monitor candle failed instrument_key=%s message_id=%s",
                        message.instrument_key,
                        message.message_id,
                    )

    def process_with_close_boundary_retry(
        self, *, message: RlTradingRedisCandleMessage
    ) -> None:
        for attempt in range(_CLOSE_BOUNDARY_RETRY_ATTEMPTS):
            try:
                self.process(message=message)
                return
            except Stage13MonitorOnlyInferenceError as error:
                should_retry = (
                    error.reason == "redis_window_contains_unclosed_candle"
                    and attempt + 1 < _CLOSE_BOUNDARY_RETRY_ATTEMPTS
                )
                if not should_retry:
                    raise
                self._metrics.observe_close_boundary_retry()
                self._close_boundary_sleep(_CLOSE_BOUNDARY_RETRY_SECONDS)

    def process(self, *, message: RlTradingRedisCandleMessage) -> None:
        instrument = self._instruments.get(message.instrument_key)
        if instrument is None:
            raise ValueError("monitor message instrument is outside allowlist")
        feature_started = time.perf_counter()
        window = self._window_reader.read_window_at_message(
            exchange=instrument.exchange,
            market_type=instrument.market_type,
            symbol=instrument.symbol,
            instrument_key=instrument.instrument_key,
            message_id=message.message_id,
        )
        last_processed = self._state_store.last_processed_close_utc(
            instrument_key=instrument.instrument_key
        )
        if last_processed is not None and window.ts_close_utc <= last_processed:
            self._stream.ack(message=message)
            self._metrics.observe_candle(
                result="duplicate_or_stale",
                candle_close_unixtime=window.ts_close_utc.timestamp(),
            )
            return
        if len(window.candles) < 90:
            self._metrics.observe_session(eligible=False, reason="insufficient_history")
            self._state_store.commit_processed(
                instrument_key=instrument.instrument_key,
                candle_close_utc=window.ts_close_utc,
                pending_trade=self._state_store.get(
                    instrument_key=instrument.instrument_key
                ),
            )
            self._stream.ack(message=message)
            self._metrics.observe_candle(
                result="insufficient_history",
                candle_close_unixtime=window.ts_close_utc.timestamp(),
            )
            return
        pending = self._close_pending_if_due(window=window, instrument=instrument)
        decision_started = time.perf_counter()
        monitor_decision = self._policy.decide(window.candles)
        if monitor_decision.action_name == "open_long" and pending is not None:
            monitor_decision = replace(
                monitor_decision,
                action_id=0,
                action_name="hold",
                policy_reason="virtual_position_already_open",
            )
        self._metrics.observe_segment_latency(
            segment="candle_close_to_feature_ready",
            seconds=decision_started - feature_started,
        )
        self._metrics.observe_session(
            eligible=monitor_decision.signal.eligible,
            reason=monitor_decision.signal.reason,
        )
        if monitor_decision.signal.eligible:
            decision_id = stage08k_entry_decision_id_v1(
                instrument_key=instrument.instrument_key,
                candle_close_utc=window.ts_close_utc,
                feature_hash=monitor_decision.feature_hash,
                policy_hash=self._policy.policy_config.policy_hash(),
            )
            decision = Stage13InferenceDecision(
                decision_id=decision_id,
                model_version_id=self._policy.model_version_id,
                action_id=monitor_decision.action_id,
                action_name=monitor_decision.action_name,
                confidence=monitor_decision.confidence,
                feature_hash=monitor_decision.feature_hash,
                feature_contract_hash=FEATURE_CONTRACT_HASH_V1,
                window_ts_close_utc=window.ts_close_utc,
                metadata={
                    "monitor_context": _monitor_context(
                        {
                            "phase": "entry",
                            "policy": self._policy.policy_config.policy_id,
                            "reason": monitor_decision.policy_reason,
                            "requested": monitor_decision.requested_action_name,
                            "volatility_score": monitor_decision.signal.volatility_score,
                        }
                    )
                },
            )
            source_started = time.perf_counter()
            event = self._producer.record_monitor_only_decision(
                organization_id=OrganizationId.from_string(
                    self._operator_context.organization_id
                ),
                context=self._decision_context(instrument=instrument),
                decision=decision,
            )
            self._assert_monitor_event_safe(event_intent_id=event.intent_id)
            source_finished = time.perf_counter()
            self._metrics.observe_segment_latency(
                segment="feature_to_decision",
                seconds=source_started - decision_started,
            )
            self._metrics.observe_segment_latency(
                segment="decision_to_source_event",
                seconds=source_finished - source_started,
            )
            self._metrics.observe_decision(
                outcome="no_intent",
                reason=STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
            )
            self._metrics.observe_action(
                requested_action=monitor_decision.requested_action_name,
                effective_action=monitor_decision.action_name,
                result=monitor_decision.policy_reason,
            )
            if monitor_decision.action_name == "open_long":
                pending = open_stage08k_virtual_trade_v1(
                    instrument_key=instrument.instrument_key,
                    symbol=instrument.symbol,
                    entry_decision_id=decision_id,
                    entry_time_utc=window.ts_close_utc,
                    entry_price=window.candles[-1].close,
                    policy=self._policy.policy_config,
                )
        self._state_store.commit_processed(
            instrument_key=instrument.instrument_key,
            candle_close_utc=window.ts_close_utc,
            pending_trade=pending,
        )
        self._metrics.set_pending_virtual_positions(len(self._state_store.all_pending()))
        self._stream.ack(message=message)
        self._metrics.observe_candle(
            result="processed",
            candle_close_unixtime=window.ts_close_utc.timestamp(),
        )

    def _close_pending_if_due(
        self,
        *,
        window: Stage13FeatureWindow,
        instrument: RlTradingInferenceInstrumentConfig,
    ) -> Stage08kPendingVirtualTrade | None:
        pending = self._state_store.get(instrument_key=instrument.instrument_key)
        if pending is None or window.ts_close_utc < pending.expected_exit_time_utc:
            return pending
        result = close_stage08k_virtual_trade_v1(
            trade=pending,
            exit_time_utc=window.ts_close_utc,
            exit_price=window.candles[-1].close,
            policy=self._policy.policy_config,
        )
        feature_hash = hash_json_payload_v1(
            {
                "entry_decision_id": pending.entry_decision_id,
                "exit_decision_id": result.exit_decision_id,
                "instrument_key": instrument.instrument_key,
            }
        )
        decision = Stage13InferenceDecision(
            decision_id=result.exit_decision_id,
            model_version_id=self._policy.model_version_id,
            action_id=3,
            action_name="close",
            confidence=1.0,
            feature_hash=feature_hash,
            feature_contract_hash=FEATURE_CONTRACT_HASH_V1,
            window_ts_close_utc=window.ts_close_utc,
            metadata={
                "monitor_context": _monitor_context(
                    {
                        "entry": result.entry_decision_id,
                        "hold_seconds": result.hold_seconds,
                        "net_return": result.net_return,
                        "phase": "close",
                        "pnl_quote": result.pnl_quote,
                        "reason": result.reason,
                        "valid": result.valid_for_policy_evaluation,
                    }
                )
            },
        )
        event = self._producer.record_monitor_only_decision(
            organization_id=OrganizationId.from_string(
                self._operator_context.organization_id
            ),
            context=self._decision_context(instrument=instrument),
            decision=decision,
        )
        self._assert_monitor_event_safe(event_intent_id=event.intent_id)
        self._metrics.observe_virtual_exit(
            valid=result.valid_for_policy_evaluation,
            reason=result.reason,
            pnl_quote=result.pnl_quote,
        )
        self._metrics.observe_decision(
            outcome="no_intent",
            reason=STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
        )
        return None

    def _decision_context(
        self, *, instrument: RlTradingInferenceInstrumentConfig
    ) -> Stage13DecisionContext:
        return Stage13DecisionContext(
            owner_user_id=self._operator_context.owner_user_id,
            strategy_id=self._operator_context.strategy_id,
            strategy_run_id=self._operator_context.strategy_run_id,
            exchange=instrument.exchange,
            market_type=instrument.market_type,
            symbol=instrument.symbol,
            instrument_key=instrument.instrument_key,
        )

    def _assert_monitor_event_safe(self, *, event_intent_id: object) -> None:
        if event_intent_id is None:
            return
        self._metrics.observe_safety_breach(reason="monitor_event_has_intent")
        raise RuntimeError("monitor-only source event unexpectedly references an intent")


def _monitor_context(payload: Mapping[str, object]) -> str:
    compact = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    if len(compact) > 256:
        raise ValueError("monitor_context exceeds execution source event limit")
    return compact
