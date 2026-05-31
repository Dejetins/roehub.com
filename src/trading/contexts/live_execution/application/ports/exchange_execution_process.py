from __future__ import annotations

from typing import Protocol

from trading.contexts.live_execution.domain import (
    ExchangeExecutionProcessHeartbeat,
    ExchangeExecutionRequestObservation,
)


class ExchangeExecutionProcessRepository(Protocol):
    def record_heartbeat(
        self, *, heartbeat: ExchangeExecutionProcessHeartbeat
    ) -> ExchangeExecutionProcessHeartbeat: ...

    def record_request_observation(
        self, *, observation: ExchangeExecutionRequestObservation
    ) -> ExchangeExecutionRequestObservation: ...
