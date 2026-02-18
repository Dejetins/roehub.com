from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class TelegramNotifierHooks:
    """
    TelegramNotifierHooks — optional callbacks for strategy Telegram notifier counters.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - src/trading/contexts/strategy/adapters/outbound/messaging/telegram/
        log_only_telegram_notifier.py
      - src/trading/contexts/strategy/adapters/outbound/messaging/telegram/
        telegram_bot_api_notifier.py
    """

    on_notify_sent: Callable[[], None] | None = None
    on_notify_error: Callable[[], None] | None = None
    on_notify_skipped: Callable[[], None] | None = None
