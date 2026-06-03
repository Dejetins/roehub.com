from __future__ import annotations

from typing import Any, Mapping

from redis.exceptions import ResponseError

from trading.contexts.live_execution.adapters.outbound.redis import (
    RedisExchangeExecutionConsumer,
    RedisExecutionDispatchTransportConfig,
)


class _FakeRedis:
    def __init__(self) -> None:
        self.groups: list[tuple[str, str]] = []
        self.reads: list[Mapping[str, str]] = []
        self.messages: dict[str, list[tuple[str, dict[str, str]]]] = {
            "execution.requests.v1": [
                ("1-0", {"intent_id": "00000000-0000-0000-0000-000000000001"})
            ]
        }
        self.acks: list[tuple[str, str, str]] = []

    def xgroup_create(
        self, *, name: str, groupname: str, id: str, mkstream: bool  # noqa: A002
    ) -> None:
        _ = id, mkstream
        if (name, groupname) in self.groups:
            raise ResponseError("BUSYGROUP Consumer Group name already exists")
        self.groups.append((name, groupname))

    def xlen(self, name: str) -> int:
        return len(self.messages.get(name, []))

    def xpending(self, stream_name: str, group_name: str) -> Mapping[str, int]:
        _ = stream_name, group_name
        return {"pending": 0}

    def time(self) -> tuple[int, int]:
        return (1_780_000_000, 0)

    def xreadgroup(
        self,
        *,
        groupname: str,
        consumername: str,
        streams: Mapping[str, str],
        count: int,
        block: int,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        _ = groupname, consumername, block
        self.reads.append(streams)
        stream_name = next(iter(streams))
        return [(stream_name, self.messages.get(stream_name, [])[:count])]

    def xadd(self, *, name: str, fields: Mapping[str, Any]) -> str:
        payload = {str(key): str(value) for key, value in fields.items()}
        messages = self.messages.setdefault(name, [])
        message_id = f"{len(messages) + 1}-0"
        messages.append((message_id, payload))
        return message_id

    def xack(self, stream_name: str, group_name: str, message_id: str) -> int:
        self.acks.append((stream_name, group_name, message_id))
        return 1


def test_redis_exchange_execution_consumer_reads_dlqs_and_acks() -> None:
    redis = _FakeRedis()
    consumer = RedisExchangeExecutionConsumer(
        config=_config(),
        consumer_name="exchange-execution-test",
        environ={},
        redis_client=redis,  # type: ignore[arg-type]
    )

    consumer.ensure_request_group()
    messages = consumer.read_new_requests(count=10, block_ms=0)
    pending_messages = consumer.read_pending_requests(count=10)
    dlq = consumer.publish_dlq(message=messages[0], reason="intent_not_found")
    consumer.ack_after_durable_state_change(
        stream_name=messages[0].stream_name,
        message_id=messages[0].message_id,
    )

    assert redis.groups == [("execution.requests.v1", "exchange-execution.v1")]
    assert redis.reads == [
        {"execution.requests.v1": ">"},
        {"execution.requests.v1": "0"},
    ]
    assert messages[0].payload["intent_id"] == "00000000-0000-0000-0000-000000000001"
    assert pending_messages[0].payload["intent_id"] == (
        "00000000-0000-0000-0000-000000000001"
    )
    assert dlq.stream_name == "execution.requests.dlq.v1"
    assert redis.messages["execution.requests.dlq.v1"][0][1]["quarantine_reason"] == (
        "intent_not_found"
    )
    assert redis.acks == [("execution.requests.v1", "exchange-execution.v1", "1-0")]


def _config() -> RedisExecutionDispatchTransportConfig:
    return RedisExecutionDispatchTransportConfig(
        host="localhost",
        port=6379,
        db=0,
        password_env=None,
        socket_timeout_s=1.0,
        connect_timeout_s=1.0,
    )
