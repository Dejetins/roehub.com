from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from trading.contexts.strategy.adapters.outbound import (
    LogOnlyTelegramNotifier,
    TelegramNotifierHooks,
)
from trading.contexts.strategy.application import StrategyTelegramNotificationV1
from trading.shared_kernel.primitives import UserId


class _ResolverStub:
    """
    Chat binding resolver stub returning configured result or raising configured error.
    """

    def __init__(self, *, chat_id: int | None = None, error: Exception | None = None) -> None:
        """
        Initialize resolver stub behavior.

        Args:
            chat_id: Chat id returned by resolver.
            error: Optional exception raised by resolver call.
        Returns:
            None.
        Assumptions:
            Exactly one resolver method is used by notifier adapter.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._chat_id = chat_id
        self._error = error

    def find_confirmed_chat_id(self, *, user_id: UserId) -> int | None:
        """
        Return configured chat id or raise configured exception.

        Args:
            user_id: Strategy owner identifier.
        Returns:
            int | None: Configured chat id.
        Assumptions:
            User id value itself is not validated by this stub.
        Raises:
            Exception: Configured resolver exception.
        Side Effects:
            None.
        """
        _ = user_id
        if self._error is not None:
            raise self._error
        return self._chat_id


class _HooksProbe:
    """
    Hook callbacks probe for notifier sent/error/skipped counters.
    """

    def __init__(self) -> None:
        """
        Initialize zeroed callback counters.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            One notify call increments at most one callback.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.sent = 0
        self.errors = 0
        self.skipped = 0

    def on_sent(self) -> None:
        """
        Increase sent callback counter.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Called on successful notification flow.
        Raises:
            None.
        Side Effects:
            Mutates probe state.
        """
        self.sent += 1

    def on_error(self) -> None:
        """
        Increase error callback counter.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Called when resolver/notifier flow raises unexpected exception.
        Raises:
            None.
        Side Effects:
            Mutates probe state.
        """
        self.errors += 1

    def on_skipped(self) -> None:
        """
        Increase skipped callback counter.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Called when confirmed chat binding is missing.
        Raises:
            None.
        Side Effects:
            Mutates probe state.
        """
        self.skipped += 1


def test_log_only_telegram_notifier_counts_sent_notification() -> None:
    """
    Ensure log-only notifier counts successful notification when confirmed chat exists.
    """
    hooks = _HooksProbe()
    notifier = LogOnlyTelegramNotifier(
        chat_binding_resolver=_ResolverStub(chat_id=777),
        hooks=TelegramNotifierHooks(
            on_notify_sent=hooks.on_sent,
            on_notify_error=hooks.on_error,
            on_notify_skipped=hooks.on_skipped,
        ),
    )

    notifier.notify(notification=_notification())

    assert hooks.sent == 1
    assert hooks.errors == 0
    assert hooks.skipped == 0


def test_log_only_telegram_notifier_counts_skipped_notification_without_binding() -> None:
    """
    Ensure log-only notifier skips notification when confirmed chat binding is absent.
    """
    hooks = _HooksProbe()
    notifier = LogOnlyTelegramNotifier(
        chat_binding_resolver=_ResolverStub(chat_id=None),
        hooks=TelegramNotifierHooks(
            on_notify_sent=hooks.on_sent,
            on_notify_error=hooks.on_error,
            on_notify_skipped=hooks.on_skipped,
        ),
    )

    notifier.notify(notification=_notification())

    assert hooks.sent == 0
    assert hooks.errors == 0
    assert hooks.skipped == 1


def test_log_only_telegram_notifier_keeps_best_effort_on_resolver_error() -> None:
    """
    Ensure log-only notifier never raises and counts error on resolver failure.
    """
    hooks = _HooksProbe()
    notifier = LogOnlyTelegramNotifier(
        chat_binding_resolver=_ResolverStub(error=RuntimeError("resolver down")),
        hooks=TelegramNotifierHooks(
            on_notify_sent=hooks.on_sent,
            on_notify_error=hooks.on_error,
            on_notify_skipped=hooks.on_skipped,
        ),
    )

    notifier.notify(notification=_notification())

    assert hooks.sent == 0
    assert hooks.errors == 1
    assert hooks.skipped == 0


def _notification() -> StrategyTelegramNotificationV1:
    """
    Build deterministic Telegram notification fixture.

    Args:
        None.
    Returns:
        StrategyTelegramNotificationV1: Notification fixture.
    Assumptions:
        Fixture values satisfy Telegram notification DTO invariants.
    Raises:
        ValueError: If fixture violates DTO validation.
    Side Effects:
        None.
    """
    return StrategyTelegramNotificationV1(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000006101"),
        ts=datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc),
        strategy_id=UUID("00000000-0000-0000-0000-00000000A610"),
        run_id=UUID("00000000-0000-0000-0000-00000000B610"),
        event_type="failed",
        instrument_key="binance:spot:BTCUSDT",
        timeframe="1m",
        message_text="FAILED | strategy_id=... | run_id=... | error=boom",
    )
