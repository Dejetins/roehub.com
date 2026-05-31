from __future__ import annotations

from trading.contexts.live_execution.application.ports import (
    ExchangeExecutionProcessRepository,
)
from trading.contexts.live_execution.domain import (
    ExchangeExecutionProcessHeartbeat,
    ExchangeExecutionRequestObservation,
)


class InMemoryExchangeExecutionProcessRepository(ExchangeExecutionProcessRepository):
    def __init__(self) -> None:
        self.heartbeats: dict[str, ExchangeExecutionProcessHeartbeat] = {}
        self.observations: list[ExchangeExecutionRequestObservation] = []

    def record_heartbeat(
        self, *, heartbeat: ExchangeExecutionProcessHeartbeat
    ) -> ExchangeExecutionProcessHeartbeat:
        self.heartbeats[heartbeat.service_id] = heartbeat
        return heartbeat

    def record_request_observation(
        self, *, observation: ExchangeExecutionRequestObservation
    ) -> ExchangeExecutionRequestObservation:
        self.observations.append(observation)
        return observation
