from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.strategy.adapters.outbound import (
    TelegramBotApiNotifier,
    TelegramBotApiNotifierConfig,
    TelegramNotifierHooks,
)
from trading.contexts.strategy.application import StrategyTelegramNotificationV1
from trading.shared_kernel.primitives import UserId


class _ResolverStub:
    """
    Chat binding resolver stub returning configured chat id.
    """

    def __init__(self, *, chat_id: int | None) -> None:
        """
        Initialize resolver stub.

        Args:
            chat_id: Chat id value returned by resolver.
        Returns:
            None.
        Assumptions:
            Resolver returns deterministic value configured by test case.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._chat_id = chat_id

    def find_confirmed_chat_id(self, *, user_id: UserId) -> int | None:
        """
        Return configured chat id.

        Args:
            user_id: Strategy owner identifier.
        Returns:
            int | None: Configured chat id.
        Assumptions:
            User id itself is not validated by this stub.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = user_id
        return self._chat_id


class _ResponseStub:
    """
    HTTP response stub for Telegram API session tests.
    """

    def __init__(self, *, status_code: int, payload: Any, text: str = "") -> None:
        """
        Initialize response stub.

        Args:
            status_code: HTTP status code.
            payload: JSON payload returned by `json()`.
            text: Raw response text.
        Returns:
            None.
        Assumptions:
            Payload shape is controlled by test scenario.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.status_code = status_code
        self._payload = payload
        self._text = text

    def json(self) -> Any:
        """
        Return configured JSON payload.

        Args:
            None.
        Returns:
            Any: Configured payload.
        Assumptions:
            Caller handles non-object payloads.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._payload

    @property
    def text(self) -> str:
        """
        Return configured response text.

        Args:
            None.
        Returns:
            str: Raw response body text.
        Assumptions:
            Response text can be empty.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._text


class _SessionStub:
    """
    HTTP session stub recording `post` calls and returning configured response/error.
    """

    def __init__(
        self,
        *,
        response: _ResponseStub | None = None,
        error: Exception | None = None,
    ) -> None:
        """
        Initialize session stub behavior.

        Args:
            response: Response returned by `post`.
            error: Optional exception raised by `post`.
        Returns:
            None.
        Assumptions:
            One response object is sufficient per notify test call.
        Raises:
            None.
        Side Effects:
            Creates mutable call log.
        """
        self._response = response if response is not None else _ResponseStub(
            status_code=200,
            payload={"ok": True},
        )
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        *,
        url: str,
        json: Mapping[str, str],
        timeout: float,
    ) -> _ResponseStub:
        """
        Record call and return configured response or raise configured exception.

        Args:
            url: Request URL.
            json: JSON payload.
            timeout: Request timeout.
        Returns:
            _ResponseStub: Configured response.
        Assumptions:
            Notifier passes string-only JSON payload.
        Raises:
            Exception: Configured transport error.
        Side Effects:
            Appends request snapshot to call log.
        """
        self.calls.append({"url": url, "json": dict(json), "timeout": timeout})
        if self._error is not None:
            raise self._error
        return self._response


class _HooksProbe:
    """
    Probe callbacks for notifier sent/error/skipped counters.
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
            Callback is invoked on successful API response.
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
            Callback is invoked on transport/API errors.
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
            Callback is invoked when confirmed chat binding is absent.
        Raises:
            None.
        Side Effects:
            Mutates probe state.
        """
        self.skipped += 1


def test_telegram_bot_api_notifier_sends_message_for_confirmed_chat_binding() -> None:
    """
    Ensure notifier posts Telegram `sendMessage` payload when confirmed chat binding exists.
    """
    hooks = _HooksProbe()
    session = _SessionStub(response=_ResponseStub(status_code=200, payload={"ok": True}))
    notifier = TelegramBotApiNotifier(
        config=TelegramBotApiNotifierConfig(
            bot_token="secret_token",
            api_base_url="https://api.telegram.org",
            send_timeout_s=2.0,
        ),
        chat_binding_resolver=_ResolverStub(chat_id=555),
        session=session,
        hooks=TelegramNotifierHooks(
            on_notify_sent=hooks.on_sent,
            on_notify_error=hooks.on_error,
            on_notify_skipped=hooks.on_skipped,
        ),
    )

    notifier.notify(notification=_notification())

    assert len(session.calls) == 1
    assert session.calls[0]["url"] == "https://api.telegram.org/botsecret_token/sendMessage"
    assert session.calls[0]["json"] == {
        "chat_id": "555",
        "text": "FAILED | strategy_id=... | run_id=... | error=boom",
    }
    assert session.calls[0]["timeout"] == 2.0
    assert hooks.sent == 1
    assert hooks.errors == 0
    assert hooks.skipped == 0


def test_telegram_bot_api_notifier_skips_when_chat_binding_is_missing() -> None:
    """
    Ensure notifier skips HTTP call when confirmed chat binding is absent.
    """
    hooks = _HooksProbe()
    session = _SessionStub(response=_ResponseStub(status_code=200, payload={"ok": True}))
    notifier = TelegramBotApiNotifier(
        config=TelegramBotApiNotifierConfig(
            bot_token="secret_token",
            api_base_url="https://api.telegram.org",
            send_timeout_s=2.0,
        ),
        chat_binding_resolver=_ResolverStub(chat_id=None),
        session=session,
        hooks=TelegramNotifierHooks(
            on_notify_sent=hooks.on_sent,
            on_notify_error=hooks.on_error,
            on_notify_skipped=hooks.on_skipped,
        ),
    )

    notifier.notify(notification=_notification())

    assert session.calls == []
    assert hooks.sent == 0
    assert hooks.errors == 0
    assert hooks.skipped == 1


def test_telegram_bot_api_notifier_counts_error_for_non_200_response() -> None:
    """
    Ensure notifier counts error hook on non-200 HTTP response status.
    """
    hooks = _HooksProbe()
    session = _SessionStub(
        response=_ResponseStub(status_code=503, payload={"ok": False}, text="unavailable")
    )
    notifier = TelegramBotApiNotifier(
        config=TelegramBotApiNotifierConfig(
            bot_token="secret_token",
            api_base_url="https://api.telegram.org",
            send_timeout_s=2.0,
        ),
        chat_binding_resolver=_ResolverStub(chat_id=555),
        session=session,
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


def test_telegram_bot_api_notifier_counts_error_for_transport_exception() -> None:
    """
    Ensure notifier keeps best-effort semantics and counts error on transport exception.
    """
    hooks = _HooksProbe()
    session = _SessionStub(error=RuntimeError("network down"))
    notifier = TelegramBotApiNotifier(
        config=TelegramBotApiNotifierConfig(
            bot_token="secret_token",
            api_base_url="https://api.telegram.org",
            send_timeout_s=2.0,
        ),
        chat_binding_resolver=_ResolverStub(chat_id=555),
        session=session,
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
        user_id=UserId.from_string("00000000-0000-0000-0000-000000006201"),
        ts=datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc),
        strategy_id=UUID("00000000-0000-0000-0000-00000000A620"),
        run_id=UUID("00000000-0000-0000-0000-00000000B620"),
        event_type="failed",
        instrument_key="binance:spot:BTCUSDT",
        timeframe="1m",
        message_text="FAILED | strategy_id=... | run_id=... | error=boom",
    )
