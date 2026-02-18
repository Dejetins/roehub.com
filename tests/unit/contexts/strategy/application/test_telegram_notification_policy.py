from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from trading.contexts.strategy.application import (
    StrategyTelegramNotificationEventV1,
    TelegramNotificationPolicy,
)
from trading.shared_kernel.primitives import UserId


def test_telegram_notification_policy_filters_fixed_event_types_and_builds_plain_text() -> None:
    """
    Ensure policy routes only fixed v1 event types and renders deterministic plain-text messages.
    """
    policy = TelegramNotificationPolicy(failed_debounce_seconds=600)
    ts = datetime(2026, 2, 17, 11, 0, tzinfo=timezone.utc)

    signal = policy.build_notification(
        event=_event(
            ts=ts,
            event_type="signal",
            payload_json={"signal": "LONG"},
        )
    )
    trade_open = policy.build_notification(
        event=_event(
            ts=ts,
            event_type="trade_open",
            payload_json={"side": "BUY", "price": "50000.10"},
        )
    )
    trade_close = policy.build_notification(
        event=_event(
            ts=ts,
            event_type="trade_close",
            payload_json={"side": "SELL", "price": "50120.50"},
        )
    )
    failed = policy.build_notification(
        event=_event(
            ts=ts,
            event_type="failed",
            payload_json={"error": "network disconnected"},
        )
    )
    unknown = policy.build_notification(
        event=_event(
            ts=ts,
            event_type="run_state_changed",
            payload_json={},
        )
    )

    assert signal is not None
    assert signal.message_text == (
        "SIGNAL | strategy_id=00000000-0000-0000-0000-00000000a501 "
        "| run_id=00000000-0000-0000-0000-00000000b501 "
        "| instrument=binance:spot:BTCUSDT | timeframe=1m | signal=LONG"
    )
    assert trade_open is not None
    assert trade_open.message_text == (
        "TRADE OPEN | strategy_id=00000000-0000-0000-0000-00000000a501 "
        "| run_id=00000000-0000-0000-0000-00000000b501 "
        "| instrument=binance:spot:BTCUSDT | timeframe=1m | side=BUY | price=50000.10"
    )
    assert trade_close is not None
    assert trade_close.message_text == (
        "TRADE CLOSE | strategy_id=00000000-0000-0000-0000-00000000a501 "
        "| run_id=00000000-0000-0000-0000-00000000b501 "
        "| instrument=binance:spot:BTCUSDT | timeframe=1m | side=SELL | price=50120.50"
    )
    assert failed is not None
    assert failed.message_text == (
        "FAILED | strategy_id=00000000-0000-0000-0000-00000000a501 "
        "| run_id=00000000-0000-0000-0000-00000000b501 "
        "| error=network disconnected"
    )
    assert unknown is None


def test_telegram_notification_policy_debounces_failed_events_by_normalized_error_text() -> None:
    """
    Ensure failed debounce uses deterministic normalized-error key and 600-second window.
    """
    policy = TelegramNotificationPolicy(failed_debounce_seconds=600)
    base_ts = datetime(2026, 2, 17, 12, 0, tzinfo=timezone.utc)

    first = policy.build_notification(
        event=_event(
            ts=base_ts,
            event_type="failed",
            payload_json={"error": "  transport   timeout  "},
        )
    )
    second = policy.build_notification(
        event=_event(
            ts=base_ts + timedelta(seconds=300),
            event_type="failed",
            payload_json={"error": "transport timeout"},
        )
    )
    third = policy.build_notification(
        event=_event(
            ts=base_ts + timedelta(seconds=601),
            event_type="failed",
            payload_json={"error": "transport timeout"},
        )
    )
    different_strategy = policy.build_notification(
        event=_event(
            ts=base_ts + timedelta(seconds=100),
            strategy_id=UUID("00000000-0000-0000-0000-00000000A999"),
            event_type="failed",
            payload_json={"error": "transport timeout"},
        )
    )

    assert first is not None
    assert second is None
    assert third is not None
    assert different_strategy is not None


def _event(
    *,
    ts: datetime,
    event_type: str,
    payload_json: dict[str, str],
    strategy_id: UUID = UUID("00000000-0000-0000-0000-00000000A501"),
) -> StrategyTelegramNotificationEventV1:
    """
    Build deterministic strategy Telegram event fixture.

    Args:
        ts: Event timestamp.
        event_type: Event type string.
        payload_json: Event payload mapping.
        strategy_id: Strategy identifier override.
    Returns:
        StrategyTelegramNotificationEventV1: Telegram event fixture.
    Assumptions:
        Fixture uses stable user/run identifiers to keep assertions deterministic.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return StrategyTelegramNotificationEventV1(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000005501"),
        ts=ts,
        strategy_id=strategy_id,
        run_id=UUID("00000000-0000-0000-0000-00000000B501"),
        event_type=event_type,
        instrument_key="binance:spot:BTCUSDT",
        timeframe="1m",
        payload_json=payload_json,
    )
