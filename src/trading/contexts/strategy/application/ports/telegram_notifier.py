from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol
from uuid import UUID

from trading.shared_kernel.primitives import UserId, UtcTimestamp

TELEGRAM_NOTIFICATION_EVENT_TYPES_V1: tuple[str, ...] = (
    "signal",
    "trade_open",
    "trade_close",
    "failed",
)


@dataclass(frozen=True, slots=True)
class StrategyTelegramNotificationEventV1:
    """
    StrategyTelegramNotificationEventV1 — normalized strategy event candidate for Telegram policy.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/telegram_notification_policy.py
      - src/trading/contexts/strategy/application/services/live_runner.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
    """

    user_id: UserId
    ts: datetime
    strategy_id: UUID
    run_id: UUID
    event_type: str
    instrument_key: str
    timeframe: str
    payload_json: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate Telegram event candidate invariants and normalize UTC timestamp precision.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `event_type` may contain values outside fixed Telegram policy allowlist.
        Raises:
            ValueError: If one of required text fields is empty.
        Side Effects:
            Normalizes `ts` to UTC with millisecond precision and copies payload mapping.
        """
        _require_text(value=self.event_type, field_name="event_type", allow_empty=False)
        _require_text(value=self.instrument_key, field_name="instrument_key", allow_empty=False)
        _require_text(value=self.timeframe, field_name="timeframe", allow_empty=False)

        normalized_ts = UtcTimestamp(self.ts).value
        object.__setattr__(self, "ts", normalized_ts)
        object.__setattr__(self, "payload_json", dict(self.payload_json))


@dataclass(frozen=True, slots=True)
class StrategyTelegramNotificationV1:
    """
    StrategyTelegramNotificationV1 — outbound Telegram notification payload after policy filtering.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/telegram_notification_policy.py
      - src/trading/contexts/strategy/adapters/outbound/messaging/telegram/
        log_only_telegram_notifier.py
      - src/trading/contexts/strategy/adapters/outbound/messaging/telegram/
        telegram_bot_api_notifier.py
    """

    user_id: UserId
    ts: datetime
    strategy_id: UUID
    run_id: UUID
    event_type: str
    instrument_key: str
    timeframe: str
    message_text: str

    def __post_init__(self) -> None:
        """
        Validate filtered Telegram notification invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Notification message text is plain UTF-8 text without markup requirements.
        Raises:
            ValueError: If event type is not allowed or message fields are invalid.
        Side Effects:
            Normalizes `ts` to UTC with millisecond precision.
        """
        if self.event_type not in TELEGRAM_NOTIFICATION_EVENT_TYPES_V1:
            raise ValueError(
                "StrategyTelegramNotificationV1.event_type must match fixed v1 allowlist"
            )
        _require_text(value=self.instrument_key, field_name="instrument_key", allow_empty=False)
        _require_text(value=self.timeframe, field_name="timeframe", allow_empty=False)
        _require_text(value=self.message_text, field_name="message_text", allow_empty=False)

        normalized_ts = UtcTimestamp(self.ts).value
        object.__setattr__(self, "ts", normalized_ts)


class ConfirmedTelegramChatBindingResolver(Protocol):
    """
    ConfirmedTelegramChatBindingResolver — ACL port for identity confirmed Telegram chat binding.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/acl/identity/
        confirmed_telegram_chat_binding_resolver.py
      - migrations/postgres/0001_identity_v1.sql
    """

    def find_confirmed_chat_id(self, *, user_id: UserId) -> int | None:
        """
        Resolve confirmed Telegram chat id for strategy owner.

        Args:
            user_id: Strategy owner identifier.
        Returns:
            int | None: Confirmed chat id or `None` when no confirmed binding exists.
        Assumptions:
            Resolver implementation applies deterministic row ordering when multiple bindings exist.
        Raises:
            Exception: Adapter implementations may raise storage mapping/runtime errors.
        Side Effects:
            Reads identity-side storage.
        """
        ...


class TelegramNotifier(Protocol):
    """
    TelegramNotifier — best-effort outbound port for Strategy Telegram notifications.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - src/trading/contexts/strategy/adapters/outbound/messaging/telegram/
        log_only_telegram_notifier.py
      - src/trading/contexts/strategy/adapters/outbound/messaging/telegram/
        telegram_bot_api_notifier.py
    """

    def notify(self, *, notification: StrategyTelegramNotificationV1) -> None:
        """
        Send one Telegram notification with best-effort semantics.

        Args:
            notification: Filtered notification payload produced by Telegram policy.
        Returns:
            None.
        Assumptions:
            Implementations must never let runtime delivery failures break strategy pipeline.
        Raises:
            Exception: Implementations may raise only on invalid usage/configuration.
        Side Effects:
            May execute outbound IO and emit logs/metrics in concrete adapters.
        """
        ...


@dataclass(frozen=True, slots=True)
class NoOpTelegramNotifier(TelegramNotifier):
    """
    NoOpTelegramNotifier — disabled notifier adapter preserving live-runner behavior.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - tests/unit/contexts/strategy/application/test_strategy_live_runner.py
    """

    def notify(self, *, notification: StrategyTelegramNotificationV1) -> None:
        """
        Ignore notification requests without any side effects.

        Args:
            notification: Notification payload ignored by no-op adapter.
        Returns:
            None.
        Assumptions:
            No-op adapter is used when Telegram integration is disabled.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = notification


def _require_text(*, value: Any, field_name: str, allow_empty: bool) -> None:
    """
    Validate strict string field contract for Telegram policy DTOs.

    Args:
        value: Candidate field value.
        field_name: Field name used in deterministic error messages.
        allow_empty: Whether empty string is accepted.
    Returns:
        None.
    Assumptions:
        Telegram DTO fields that participate in routing and message rendering must be strings.
    Raises:
        ValueError: If value is non-string or empty when disallowed.
    Side Effects:
        None.
    """
    if type(value) is not str:
        raise ValueError(f"{field_name} must be string")
    if allow_empty:
        return
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty string")
