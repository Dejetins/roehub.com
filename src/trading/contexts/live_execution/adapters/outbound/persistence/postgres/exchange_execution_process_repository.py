from __future__ import annotations

import json
from typing import Mapping

from trading.contexts.live_execution.application.ports import (
    ExchangeExecutionProcessRepository,
)
from trading.contexts.live_execution.domain import (
    ExchangeExecutionProcessHeartbeat,
    ExchangeExecutionRequestObservation,
)
from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)


class PostgresExchangeExecutionProcessRepository(ExchangeExecutionProcessRepository):
    def __init__(self, *, gateway: StrategyPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresExchangeExecutionProcessRepository requires gateway")
        self._gateway = gateway

    def record_heartbeat(
        self, *, heartbeat: ExchangeExecutionProcessHeartbeat
    ) -> ExchangeExecutionProcessHeartbeat:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO exchange_execution_process_heartbeats
            (
                service_id, status, status_reason, adapter_mode, started_at,
                heartbeat_at, request_stream, consumer_group, consumer_name,
                metadata_json
            )
            VALUES
            (
                %(service_id)s, %(status)s, %(status_reason)s, %(adapter_mode)s,
                %(started_at)s, %(heartbeat_at)s, %(request_stream)s,
                %(consumer_group)s, %(consumer_name)s, %(metadata_json)s::jsonb
            )
            ON CONFLICT (service_id) DO UPDATE
            SET status = EXCLUDED.status,
                status_reason = EXCLUDED.status_reason,
                adapter_mode = EXCLUDED.adapter_mode,
                started_at = EXCLUDED.started_at,
                heartbeat_at = EXCLUDED.heartbeat_at,
                request_stream = EXCLUDED.request_stream,
                consumer_group = EXCLUDED.consumer_group,
                consumer_name = EXCLUDED.consumer_name,
                metadata_json = EXCLUDED.metadata_json
            RETURNING service_id
            """,
            parameters=_heartbeat_params(heartbeat),
        )
        if row is None:
            return heartbeat
        return heartbeat

    def record_request_observation(
        self, *, observation: ExchangeExecutionRequestObservation
    ) -> ExchangeExecutionRequestObservation:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO exchange_execution_request_observations
            (
                observation_id, service_id, intent_id, stream_name,
                redis_message_id, status, status_reason, adapter_mode,
                observed_at, metadata_json
            )
            VALUES
            (
                %(observation_id)s, %(service_id)s, %(intent_id)s, %(stream_name)s,
                %(redis_message_id)s, %(status)s, %(status_reason)s,
                %(adapter_mode)s, %(observed_at)s, %(metadata_json)s::jsonb
            )
            ON CONFLICT (stream_name, redis_message_id, status) DO NOTHING
            RETURNING observation_id
            """,
            parameters=_observation_params(observation),
        )
        if row is None:
            return observation
        return observation


def _heartbeat_params(heartbeat: ExchangeExecutionProcessHeartbeat) -> dict[str, object]:
    return {
        "service_id": heartbeat.service_id,
        "status": heartbeat.status,
        "status_reason": heartbeat.status_reason,
        "adapter_mode": heartbeat.adapter_mode,
        "started_at": heartbeat.started_at,
        "heartbeat_at": heartbeat.heartbeat_at,
        "request_stream": heartbeat.request_stream,
        "consumer_group": heartbeat.consumer_group,
        "consumer_name": heartbeat.consumer_name,
        "metadata_json": _metadata_json(heartbeat.metadata),
    }


def _observation_params(observation: ExchangeExecutionRequestObservation) -> dict[str, object]:
    return {
        "observation_id": str(observation.observation_id),
        "service_id": observation.service_id,
        "intent_id": str(observation.intent_id) if observation.intent_id is not None else None,
        "stream_name": observation.stream_name,
        "redis_message_id": observation.redis_message_id,
        "status": observation.status,
        "status_reason": observation.status_reason,
        "adapter_mode": observation.adapter_mode,
        "observed_at": observation.observed_at,
        "metadata_json": _metadata_json(observation.metadata),
    }


def _metadata_json(metadata: Mapping[str, int | float | str]) -> str:
    return json.dumps(dict(metadata), sort_keys=True)
