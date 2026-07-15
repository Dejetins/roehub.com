from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID, uuid4

import pytest
from redis.exceptions import ConnectionError, ResponseError

from trading.contexts.live_execution.adapters.outbound.redis import (
    RedisExecutionDispatchTransport,
    RedisExecutionDispatchTransportConfig,
)
from trading.contexts.live_execution.application.ports import (
    ExecutionDispatchPoisonMessageError,
    ExecutionDispatchUnavailableError,
)
from trading.contexts.live_execution.domain import ExecutionIntent
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("00000000-0000-4000-8000-000000012200"))


class _FakeRedis:
    def __init__(self) -> None:
        self.groups: list[tuple[str, str]] = []
        self.messages: dict[str, list[dict[str, str]]] = {}
        self.acks: list[tuple[str, str, str]] = []
        self.fail_xadd = False

    def xgroup_create(
        self, *, name: str, groupname: str, id: str, mkstream: bool  # noqa: A002
    ) -> None:
        _ = id, mkstream
        if (name, groupname) in self.groups:
            raise ResponseError("BUSYGROUP Consumer Group name already exists")
        self.groups.append((name, groupname))

    def xlen(self, name: str) -> int:
        return len(self.messages.get(name, []))

    def xadd(self, *, name: str, fields: Mapping[str, Any]) -> str:
        if self.fail_xadd:
            raise ConnectionError("redis down")
        payload = {str(key): str(value) for key, value in fields.items()}
        self.messages.setdefault(name, []).append(payload)
        return f"{len(self.messages[name])}-0"

    def xack(self, stream_name: str, group_name: str, message_id: str) -> int:
        self.acks.append((stream_name, group_name, message_id))
        return 1


def test_redis_dispatch_transport_publishes_request_retry_dlq_and_acks() -> None:
    redis = _FakeRedis()
    transport = RedisExecutionDispatchTransport(
        config=_config(),
        environ={},
        redis_client=redis,  # type: ignore[arg-type]
    )
    intent = _intent()

    transport.ensure_request_group()
    transport.ensure_request_group()
    request = transport.publish_request(intent=intent, attempt_count=1)
    retry = transport.publish_retry(intent=intent, reason="dispatch_backpressure", attempt_count=1)
    dlq = transport.publish_dlq(intent=intent, reason="poison", attempt_count=1)
    transport.ack_after_durable_state_change(
        stream_name="execution.requests.v1",
        message_id=request.message_id,
    )

    assert redis.groups == [("execution.requests.v1", "exchange-execution.v1")]
    assert request.stream_name == "execution.requests.v1"
    assert retry.stream_name == "execution.requests.retry.v1"
    assert dlq.stream_name == "execution.requests.dlq.v1"
    assert redis.messages["execution.requests.v1"][0]["intent_id"] == str(intent.intent_id)
    assert redis.messages["execution.requests.v1"][0]["idempotency_key_hash"] == "a" * 64
    assert redis.messages["execution.requests.retry.v1"][0]["retry_reason"] == (
        "dispatch_backpressure"
    )
    assert redis.messages["execution.requests.dlq.v1"][0]["quarantine_reason"] == "poison"
    assert redis.acks == [("execution.requests.v1", "exchange-execution.v1", "1-0")]


def test_redis_dispatch_transport_maps_redis_errors_to_unavailable() -> None:
    redis = _FakeRedis()
    redis.fail_xadd = True
    transport = RedisExecutionDispatchTransport(
        config=_config(),
        environ={},
        redis_client=redis,  # type: ignore[arg-type]
    )

    with pytest.raises(ExecutionDispatchUnavailableError) as error_info:
        transport.publish_request(intent=_intent(), attempt_count=1)

    assert error_info.value.reason == "ConnectionError"


def test_redis_dispatch_transport_rejects_poison_payload_before_xadd() -> None:
    intent = _intent(instrument_key="")
    transport = RedisExecutionDispatchTransport(
        config=_config(),
        environ={},
        redis_client=_FakeRedis(),  # type: ignore[arg-type]
    )

    with pytest.raises(ExecutionDispatchPoisonMessageError) as error_info:
        transport.publish_request(intent=intent, attempt_count=1)

    assert error_info.value.reason == "dispatch_payload_missing_instrument_key"


def _config() -> RedisExecutionDispatchTransportConfig:
    return RedisExecutionDispatchTransportConfig(
        host="localhost",
        port=6379,
        db=0,
        password_env=None,
        socket_timeout_s=1.0,
        connect_timeout_s=1.0,
    )


def _intent(*, instrument_key: str = "binance:spot:BTCUSDT") -> ExecutionIntent:
    return ExecutionIntent(
        intent_id=uuid4(),
        source_event_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000012201"),
        source_type="ops_test",
        strategy_signal_id=None,
        exchange_connection_id=UUID("00000000-0000-0000-0000-000000012301"),
        market_type="spot",
        instrument_key=instrument_key,
        side="buy",
        order_type="market",
        quantity=Decimal("0.01"),
        quote_notional=None,
        limit_price=None,
        status="dispatching",
        status_reason="dispatch_publish_pending",
        risk_status="accepted",
        risk_reason="risk_gate_accepted",
        idempotency_key_hash="a" * 64,
        created_at=datetime(2026, 5, 31, 16, 30, tzinfo=UTC),
        dispatch_attempt_count=1,
    )
