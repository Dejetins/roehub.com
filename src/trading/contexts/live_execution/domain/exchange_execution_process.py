from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Mapping
from uuid import UUID

ExchangeExecutionDependencyStatus = Literal["ready", "degraded", "not_ready"]
ExchangeExecutionProcessStatus = Literal["ready", "degraded", "not_ready"]
ExchangeExecutionAdapterMode = Literal["disabled", "testnet"]
ExchangeExecutionObservationStatus = Literal[
    "adapter_disabled",
    "adapter_error",
    "guard_rejected",
    "quarantined",
    "skipped",
    "testnet_submitted",
]


@dataclass(frozen=True, slots=True)
class ExchangeExecutionDependencyHealth:
    name: str
    status: ExchangeExecutionDependencyStatus
    reason: str
    metadata: Mapping[str, int | float | str]


@dataclass(frozen=True, slots=True)
class ExchangeExecutionHealthSnapshot:
    service_id: str
    status: ExchangeExecutionProcessStatus
    status_reason: str
    adapter_mode: ExchangeExecutionAdapterMode
    checked_at: datetime
    dependencies: tuple[ExchangeExecutionDependencyHealth, ...]


@dataclass(frozen=True, slots=True)
class ExchangeExecutionProcessHeartbeat:
    service_id: str
    status: ExchangeExecutionProcessStatus
    status_reason: str
    adapter_mode: ExchangeExecutionAdapterMode
    started_at: datetime
    heartbeat_at: datetime
    request_stream: str
    consumer_group: str
    consumer_name: str
    metadata: Mapping[str, int | float | str]


@dataclass(frozen=True, slots=True)
class ExchangeExecutionRequestObservation:
    observation_id: UUID
    service_id: str
    intent_id: UUID | None
    stream_name: str
    redis_message_id: str
    status: ExchangeExecutionObservationStatus
    status_reason: str
    adapter_mode: ExchangeExecutionAdapterMode
    observed_at: datetime
    metadata: Mapping[str, int | float | str]
