from __future__ import annotations

import logging
from typing import Callable

from trading.contexts.strategy.application.ports import (
    ConfirmedTelegramChatBindingResolver,
    StrategyTelegramNotificationV1,
    TelegramNotifier,
)

from .telegram_notifier_hooks import TelegramNotifierHooks

log = logging.getLogger(__name__)


class LogOnlyTelegramNotifier(TelegramNotifier):
    """
    LogOnlyTelegramNotifier — dev/test adapter that logs Telegram notifications without HTTP send.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/telegram_notifier.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/dev/strategy_live_runner.yaml
    """

    def __init__(
        self,
        *,
        chat_binding_resolver: ConfirmedTelegramChatBindingResolver,
        hooks: TelegramNotifierHooks | None = None,
    ) -> None:
        """
        Initialize log-only notifier dependencies.

        Args:
            chat_binding_resolver: Resolver for confirmed Telegram chat id.
            hooks: Optional metrics callbacks.
        Returns:
            None.
        Assumptions:
            Adapter is used in dev/test environments for side-effect free observability.
        Raises:
            ValueError: If resolver dependency is missing.
        Side Effects:
            None.
        """
        if chat_binding_resolver is None:  # type: ignore[truthy-bool]
            raise ValueError("LogOnlyTelegramNotifier requires chat_binding_resolver")
        self._chat_binding_resolver = chat_binding_resolver
        self._hooks = hooks if hooks is not None else TelegramNotifierHooks()

    def notify(self, *, notification: StrategyTelegramNotificationV1) -> None:
        """
        Log notification payload in best-effort mode.

        Args:
            notification: Telegram notification payload.
        Returns:
            None.
        Assumptions:
            Missing confirmed chat binding is a skip, not an error.
        Raises:
            None.
        Side Effects:
            Emits structured logs and optional metrics hooks.
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

            log.info(
                (
                    "strategy telegram log-only notification "
                    "chat_id=%s event_type=%s strategy_id=%s run_id=%s message=%s"
                ),
                chat_id,
                notification.event_type,
                notification.strategy_id,
                notification.run_id,
                notification.message_text,
            )
            _emit_hook(self._hooks.on_notify_sent)
        except Exception:  # noqa: BLE001
            _emit_hook(self._hooks.on_notify_error)
            log.exception(
                (
                    "strategy telegram log-only notification failed "
                    "event_type=%s strategy_id=%s run_id=%s"
                ),
                notification.event_type,
                notification.strategy_id,
                notification.run_id,
            )


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
