from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from trading.contexts.live_execution.application import (
    CreateExecutionIntentCommand,
    ExecutionIngressService,
    ExecutionIntentRepository,
    RecordExecutionSourceEventCommand,
)
from trading.contexts.live_execution.domain import (
    PAPER_VIRTUAL_EXCHANGE_CONNECTION_ID,
    ExecutionRiskContext,
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
                organization_id=signal.organization_id,
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
        if signal.mode == "paper" and signal.outcome == "signal":
            self._ingress_service.create_intent(
                command=CreateExecutionIntentCommand(
                    organization_id=signal.organization_id,
                    owner_user_id=signal.owner_user_id,
                    source_event_id=result.event.source_event_id,
                    idempotency_key=_signal_intent_idempotency_key(signal=signal),
                    exchange_connection_id=_expected_exchange_connection_id(signal=signal),
                    market_type=signal.market_type,
                    instrument_key=signal.instrument_key,
                    order_type="market",
                    side=signal.side or "buy",
                    quantity=None,
                    quote_notional=_expected_quote_notional(signal=signal),
                    limit_price=None,
                    advanced_order_flags={},
                    risk_context=_paper_no_dispatch_context(),
                )
            )
            return
        if signal.mode == "monitor_only" or signal.outcome != "signal":
            self._repository.update_source_event_outcome(
                organization_id=signal.organization_id,
                owner_user_id=signal.owner_user_id,
                source_event_id=result.event.source_event_id,
                outcome="no_intent",
                outcome_reason=signal.reason_code,
                intent_id=None,
            )


def _signal_intent_idempotency_key(*, signal: StrategySignal) -> str:
    return f"{_signal_idempotency_key(signal=signal)}|paper-intent"


def _expected_quote_notional(*, signal: StrategySignal) -> Decimal:
    raw_value = signal.expected_order_json.get("quote_notional")
    return Decimal(str(raw_value))


def _expected_exchange_connection_id(*, signal: StrategySignal) -> UUID:
    raw_value = signal.expected_order_json.get("exchange_connection_id")
    if raw_value is None:
        return PAPER_VIRTUAL_EXCHANGE_CONNECTION_ID
    return UUID(str(raw_value))


def _paper_no_dispatch_context() -> ExecutionRiskContext:
    return ExecutionRiskContext(
        organization_ownership_verified=True,
        account_ownership_verified=True,
        exchange_connection_active=True,
        secret_custody_ready=True,
        source_authorized=True,
        strategy_variant_compatible=True,
        market_data_state="ready",
        strategy_binding_active=True,
        strategy_live_profile_ready=True,
        strategy_run_active=True,
        exchange_config_verified=False,
        account_state_fresh=False,
        position_ownership_active=True,
        capital_reservation_active=True,
        capital_reservation_sufficient=True,
        paper_accounting_ready=True,
        paper_no_exchange_submit=True,
        kill_switch_open=True,
        environment_policy_allows=True,
        max_order_size_ok=True,
        daily_limit_ok=True,
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
