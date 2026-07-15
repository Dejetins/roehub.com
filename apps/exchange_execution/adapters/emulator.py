from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import time
from uuid import NAMESPACE_URL, uuid5

from trading.contexts.live_execution.application.ports import ExchangeOrderAdapterError
from trading.contexts.live_execution.domain import (
    ExchangeExecutionConnection,
    ExchangeOrderCancelResult,
    ExchangeOrderCommand,
    ExchangeOrderStatusResult,
    ExchangeOrderSubmitResult,
    ExchangePrivateStreamSession,
)


def _module_revision_hash(provider_id: str) -> str:
    module_bytes = Path(__file__).read_bytes()
    return hashlib.sha256(provider_id.encode("utf-8") + b"\0" + module_bytes).hexdigest()


@dataclass(slots=True)
class ExchangeExecutionEmulatorAdapter:
    exchange_name: str
    scripted_outcomes: dict[str, str] = field(default_factory=dict)
    provider_id: str = "core:exchange-emulator"
    provider_version: str = "v1"
    provider_kind: str = "core"
    revision_hash: str = _module_revision_hash("core:exchange-emulator")
    _orders: dict[str, str] = field(default_factory=dict)

    def server_time_ms(self) -> int:
        return int(time() * 1_000)

    def submit_order(
        self, *, command: ExchangeOrderCommand, credential: object
    ) -> ExchangeOrderSubmitResult:
        _ = credential
        outcome = self.scripted_outcomes.get(command.client_order_id, "accepted")
        exchange_order_id = f"emu-{command.client_order_id}"
        if outcome == "timeout_before_accept":
            raise ExchangeOrderAdapterError(
                reason="emulator_timeout_before_accept",
                unknown_state=True,
            )
        self._orders[command.client_order_id] = exchange_order_id
        if outcome == "timeout_after_accept":
            raise ExchangeOrderAdapterError(
                reason="emulator_timeout_after_accept",
                unknown_state=True,
            )
        now = datetime.now(tz=UTC)
        return ExchangeOrderSubmitResult(
            exchange_order_id=exchange_order_id,
            exchange_status="accepted",
            submitted_at=now,
            latency_ms=0.1,
            metadata={"provider": "exchange-emulator", "effect": "no-external-side-effect"},
        )

    def get_order_status(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderStatusResult:
        _ = credential
        present = self._orders.get(command.client_order_id) == exchange_order_id
        return self._status(exchange_order_id=exchange_order_id if present else "")

    def get_order_status_by_client_order_id(
        self,
        *,
        command: ExchangeOrderCommand,
        client_order_id: str,
        credential: object,
    ) -> ExchangeOrderStatusResult:
        _ = (command, credential)
        return self._status(exchange_order_id=self._orders.get(client_order_id, ""))

    def cancel_order(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderCancelResult:
        _ = credential
        self._orders.pop(command.client_order_id, None)
        return ExchangeOrderCancelResult(
            exchange_order_id=exchange_order_id,
            exchange_status="cancelled",
            cancelled_at=datetime.now(tz=UTC),
            latency_ms=0.1,
            metadata={"provider": "exchange-emulator", "effect": "no-external-side-effect"},
        )

    def ensure_private_stream_session(
        self, *, connection: ExchangeExecutionConnection
    ) -> ExchangePrivateStreamSession:
        now = datetime.now(tz=UTC)
        return ExchangePrivateStreamSession(
            session_id=uuid5(NAMESPACE_URL, f"emulator:{connection.connection_id}"),
            organization_id=connection.organization_id,
            exchange_name=connection.exchange_name,
            environment=connection.environment,
            market_type=connection.market_type,
            status="ready",
            status_reason="emulator_private_state_ready",
            opened_at=now,
            keepalive_at=now,
            expires_at=now + timedelta(minutes=5),
            metadata={"provider": "exchange-emulator"},
        )

    @staticmethod
    def _status(*, exchange_order_id: str) -> ExchangeOrderStatusResult:
        now = datetime.now(tz=UTC)
        return ExchangeOrderStatusResult(
            exchange_order_id=exchange_order_id,
            exchange_status="accepted" if exchange_order_id else "not_found",
            checked_at=now,
            latency_ms=0.1,
            metadata={"provider": "exchange-emulator", "lookup": "client_order_id"},
            lookup_outcome="found" if exchange_order_id else "confirmed_absent",
        )
