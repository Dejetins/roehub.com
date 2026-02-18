from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, cast

import requests

from trading.contexts.strategy.application.ports import (
    ConfirmedTelegramChatBindingResolver,
    StrategyTelegramNotificationV1,
    TelegramNotifier,
)

from .telegram_notifier_hooks import TelegramNotifierHooks

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TelegramBotApiNotifierConfig:
    """
    TelegramBotApiNotifierConfig — runtime settings for Telegram Bot API notifier adapter.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/prod/strategy_live_runner.yaml
    """

    bot_token: str
    api_base_url: str
    send_timeout_s: float

    def __post_init__(self) -> None:
        """
        Validate Telegram notifier config invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Bot token is provided through environment and must never be empty.
        Raises:
            ValueError: If one of config values is invalid.
        Side Effects:
            None.
        """
        normalized_token = self.bot_token.strip()
        normalized_api_base = self.api_base_url.strip()
        if not normalized_token:
            raise ValueError("TelegramBotApiNotifierConfig.bot_token must be non-empty")
        if not normalized_api_base:
            raise ValueError("TelegramBotApiNotifierConfig.api_base_url must be non-empty")
        if not normalized_api_base.startswith(("https://", "http://")):
            raise ValueError(
                "TelegramBotApiNotifierConfig.api_base_url must start with http:// or https://"
            )
        if self.send_timeout_s <= 0:
            raise ValueError("TelegramBotApiNotifierConfig.send_timeout_s must be > 0")
        object.__setattr__(self, "bot_token", normalized_token)
        object.__setattr__(self, "api_base_url", normalized_api_base.rstrip("/"))


class TelegramHttpResponse(Protocol):
    """
    TelegramHttpResponse — minimal HTTP response contract used by Telegram bot notifier.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/messaging/telegram/
        telegram_bot_api_notifier.py
      - tests/unit/contexts/strategy/adapters/test_telegram_bot_api_notifier.py
    """

    status_code: int

    def json(self) -> Any:
        """
        Parse HTTP response payload as JSON object.

        Args:
            None.
        Returns:
            Any: Parsed payload.
        Assumptions:
            Telegram Bot API returns JSON object payload for `sendMessage`.
        Raises:
            Exception: Adapter-specific JSON parsing errors.
        Side Effects:
            None.
        """
        ...

    @property
    def text(self) -> str:
        """
        Return raw response text for diagnostics.

        Args:
            None.
        Returns:
            str: Raw response body text.
        Assumptions:
            Text may be empty when parsing fails.
        Raises:
            None.
        Side Effects:
            None.
        """
        ...


class TelegramHttpSession(Protocol):
    """
    TelegramHttpSession — minimal HTTP session contract for Telegram notifier testability.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/messaging/telegram/
        telegram_bot_api_notifier.py
      - tests/unit/contexts/strategy/adapters/test_telegram_bot_api_notifier.py
    """

    def post(
        self,
        *,
        url: str,
        json: Mapping[str, str],
        timeout: float,
    ) -> TelegramHttpResponse:
        """
        Execute HTTP POST request.

        Args:
            url: Full request URL.
            json: JSON payload mapping.
            timeout: Request timeout seconds.
        Returns:
            TelegramHttpResponse: HTTP response object.
        Assumptions:
            Timeout is always positive and provided by caller.
        Raises:
            Exception: Transport-level failures.
        Side Effects:
            Performs outbound HTTP request.
        """
        ...


class TelegramBotApiNotifier(TelegramNotifier):
    """
    TelegramBotApiNotifier — best-effort notifier adapter for Telegram Bot API `sendMessage`.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/telegram_notifier.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - tests/unit/contexts/strategy/adapters/test_telegram_bot_api_notifier.py
    """

    def __init__(
        self,
        *,
        config: TelegramBotApiNotifierConfig,
        chat_binding_resolver: ConfirmedTelegramChatBindingResolver,
        session: TelegramHttpSession | None = None,
        hooks: TelegramNotifierHooks | None = None,
    ) -> None:
        """
        Initialize Telegram Bot API notifier dependencies.

        Args:
            config: Validated Telegram notifier config.
            chat_binding_resolver: Resolver for confirmed Telegram chat id.
            session: Optional injected HTTP session for tests.
            hooks: Optional metrics callbacks.
        Returns:
            None.
        Assumptions:
            Adapter is called from single live-runner process in Strategy v1.
        Raises:
            ValueError: If required dependencies are missing.
        Side Effects:
            Creates `requests.Session` when no custom session is injected.
        """
        if config is None:  # type: ignore[truthy-bool]
            raise ValueError("TelegramBotApiNotifier requires config")
        if chat_binding_resolver is None:  # type: ignore[truthy-bool]
            raise ValueError("TelegramBotApiNotifier requires chat_binding_resolver")
        self._config = config
        self._chat_binding_resolver = chat_binding_resolver
        self._session = (
            session
            if session is not None
            else cast(TelegramHttpSession, requests.Session())
        )
        self._hooks = hooks if hooks is not None else TelegramNotifierHooks()

    def notify(self, *, notification: StrategyTelegramNotificationV1) -> None:
        """
        Send one Telegram message in best-effort mode.

        Args:
            notification: Telegram notification payload.
        Returns:
            None.
        Assumptions:
            Runtime delivery errors must never bubble into strategy live-runner pipeline.
        Raises:
            None.
        Side Effects:
            Reads confirmed chat binding and performs one outbound HTTP request on success path.
        """
        try:
            chat_id = self._chat_binding_resolver.find_confirmed_chat_id(
                user_id=notification.user_id
            )
            if chat_id is None:
                _emit_hook(self._hooks.on_notify_skipped)
                log.warning(
                    (
                        "strategy telegram notification skipped "
                        "reason=no_confirmed_chat_binding event_type=%s strategy_id=%s run_id=%s"
                    ),
                    notification.event_type,
                    notification.strategy_id,
                    notification.run_id,
                )
                return

            response = self._session.post(
                url=_send_message_url(
                    api_base_url=self._config.api_base_url,
                    bot_token=self._config.bot_token,
                ),
                json={
                    "chat_id": str(chat_id),
                    "text": notification.message_text,
                },
                timeout=self._config.send_timeout_s,
            )
            if response.status_code != 200:
                _emit_hook(self._hooks.on_notify_error)
                log.warning(
                    (
                        "strategy telegram notification failed "
                        "status_code=%s event_type=%s strategy_id=%s run_id=%s body=%s"
                    ),
                    response.status_code,
                    notification.event_type,
                    notification.strategy_id,
                    notification.run_id,
                    _response_excerpt(response=response),
                )
                return

            payload = _parse_json_object(response=response)
            if payload.get("ok") is not True:
                _emit_hook(self._hooks.on_notify_error)
                log.warning(
                    (
                        "strategy telegram notification failed "
                        "reason=telegram_api_error event_type=%s strategy_id=%s run_id=%s body=%s"
                    ),
                    notification.event_type,
                    notification.strategy_id,
                    notification.run_id,
                    payload,
                )
                return

            _emit_hook(self._hooks.on_notify_sent)
        except Exception:  # noqa: BLE001
            _emit_hook(self._hooks.on_notify_error)
            log.exception(
                (
                    "strategy telegram notification failed "
                    "event_type=%s strategy_id=%s run_id=%s"
                ),
                notification.event_type,
                notification.strategy_id,
                notification.run_id,
            )


def _send_message_url(*, api_base_url: str, bot_token: str) -> str:
    """
    Build Telegram Bot API `sendMessage` URL.

    Args:
        api_base_url: Telegram API base URL.
        bot_token: Telegram bot token.
    Returns:
        str: Full API URL.
    Assumptions:
        Input values are validated and non-empty.
    Raises:
        None.
    Side Effects:
        None.
    """
    return f"{api_base_url}/bot{bot_token}/sendMessage"


def _parse_json_object(*, response: TelegramHttpResponse) -> dict[str, Any]:
    """
    Parse HTTP response body into JSON object mapping.

    Args:
        response: HTTP response object.
    Returns:
        dict[str, Any]: Parsed payload, empty dict when payload is not object.
    Assumptions:
        Telegram API returns JSON body for all response codes.
    Raises:
        Exception: Underlying JSON parser errors from response object.
    Side Effects:
        None.
    """
    payload = response.json()
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _response_excerpt(*, response: TelegramHttpResponse) -> str:
    """
    Build compact response body excerpt for warning logs.

    Args:
        response: HTTP response object.
    Returns:
        str: Truncated response text.
    Assumptions:
        Response text can be empty.
    Raises:
        None.
    Side Effects:
        None.
    """
    text = response.text
    if not text:
        return ""
    return text[:300]


def _emit_hook(callback: Callable[[], None] | None) -> None:
    """
    Execute optional hook callback.

    Args:
        callback: Callback without arguments.
    Returns:
        None.
    Assumptions:
        Hook callbacks are lightweight counter increments.
    Raises:
        None.
    Side Effects:
        Executes callback when provided.
    """
    if callback is not None:
        callback()
