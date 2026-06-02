from __future__ import annotations

from trading.contexts.live_execution.application import (
    ExecutionIngressService,
    ExecutionIntentRepository,
    RecordExecutionSourceEventCommand,
)
from trading.contexts.strategy.application.ports import StrategyExecutionProducer
from trading.contexts.strategy.domain.entities import StrategySignal


class LiveExecutionStrategySignalProducer(StrategyExecutionProducer):
    def __init__(
        self,
        *,
        ingress_service: ExecutionIngressService,
        repository: ExecutionIntentRepository,
    ) -> None:
        if ingress_service is None:  # type: ignore[truthy-bool]
            raise ValueError("LiveExecutionStrategySignalProducer requires ingress_service")
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("LiveExecutionStrategySignalProducer requires repository")
        self._ingress_service = ingress_service
        self._repository = repository

    def record_signal(self, *, signal: StrategySignal) -> None:
        result = self._ingress_service.record_source_event(
            command=RecordExecutionSourceEventCommand(
                owner_user_id=signal.owner_user_id,
                source_type="strategy_signal",
                source_event_ref=str(signal.signal_id),
                source_ref_json={
                    "strategy_id": str(signal.strategy_id),
                    "strategy_run_id": str(signal.strategy_run_id),
                    "signal_id": str(signal.signal_id),
                    "mode": signal.mode,
                    "action": signal.signal_action,
                },
                strategy_signal_id=signal.signal_id,
                idempotency_key=_signal_idempotency_key(signal=signal),
            )
        )
        if signal.mode == "monitor_only" or signal.outcome != "signal":
            self._repository.update_source_event_outcome(
                owner_user_id=signal.owner_user_id,
                source_event_id=result.event.source_event_id,
                outcome="no_intent",
                outcome_reason=signal.reason_code,
                intent_id=None,
            )


def _signal_idempotency_key(*, signal: StrategySignal) -> str:
    return "|".join(
        (
            "strategy_signal",
            str(signal.strategy_id),
            str(signal.strategy_run_id),
            str(signal.signal_id),
            signal.instrument_key,
            signal.signal_action,
            signal.side or "none",
            signal.bar_ts_open.isoformat(),
        )
    )
