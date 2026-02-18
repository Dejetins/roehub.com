from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping

from trading.contexts.strategy.application.ports.telegram_notifier import (
    TELEGRAM_NOTIFICATION_EVENT_TYPES_V1,
    StrategyTelegramNotificationEventV1,
    StrategyTelegramNotificationV1,
)
from trading.shared_kernel.primitives import UtcTimestamp

_DEFAULT_FAILED_DEBOUNCE_SECONDS = 600


class TelegramNotificationPolicy:
    """
    TelegramNotificationPolicy — Strategy v1 Telegram filtering, debounce,
    and message rendering policy.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/telegram_notifier.py
      - src/trading/contexts/strategy/application/services/live_runner.py
      - tests/unit/contexts/strategy/application/test_telegram_notification_policy.py
    """

    def __init__(self, *, failed_debounce_seconds: int = _DEFAULT_FAILED_DEBOUNCE_SECONDS) -> None:
        """
        Initialize Telegram notification policy state and debounce window.

        Args:
            failed_debounce_seconds: Debounce window for repeated `failed` events.
        Returns:
            None.
        Assumptions:
            Debounce state is process-local in Strategy v1 worker deployment model.
        Raises:
            ValueError: If debounce window is negative.
        Side Effects:
            Creates mutable in-memory debounce state map.
        """
        if failed_debounce_seconds < 0:
            raise ValueError("TelegramNotificationPolicy.failed_debounce_seconds must be >= 0")
        self._failed_debounce = timedelta(seconds=failed_debounce_seconds)
        self._failed_sent_at_by_key: dict[tuple[str, str, str], datetime] = {}

    def build_notification(
        self,
        *,
        event: StrategyTelegramNotificationEventV1,
    ) -> StrategyTelegramNotificationV1 | None:
        """
        Build filtered Telegram notification payload for one strategy event.

        Args:
            event: Strategy event candidate.
        Returns:
            StrategyTelegramNotificationV1 | None: Notification payload or `None` when skipped.
        Assumptions:
            Only fixed v1 event types are routed into Telegram notifications.
        Raises:
            ValueError: If event value normalization fails.
        Side Effects:
            Updates in-memory debounce timestamps for allowed `failed` notifications.
        """
        normalized_event_type = event.event_type.strip().lower()
        if normalized_event_type not in TELEGRAM_NOTIFICATION_EVENT_TYPES_V1:
            return None

        message_text: str
        if normalized_event_type == "failed":
            normalized_error = _normalize_error_text(
                text=_payload_text(
                    payload=event.payload_json,
                    key="error",
                    fallback="unknown_error",
                )
            )
            if self._is_failed_debounced(
                user_id=str(event.user_id),
                strategy_id=str(event.strategy_id),
                normalized_error=normalized_error,
                ts=event.ts,
            ):
                return None
            message_text = _render_failed_message(
                strategy_id=str(event.strategy_id),
                run_id=str(event.run_id),
                error_text=normalized_error,
            )
        elif normalized_event_type == "signal":
            message_text = _render_signal_message(
                strategy_id=str(event.strategy_id),
                run_id=str(event.run_id),
                instrument_key=event.instrument_key,
                timeframe=event.timeframe,
                signal_text=_payload_text(
                    payload=event.payload_json,
                    key="signal",
                    fallback="n/a",
                ),
            )
        elif normalized_event_type == "trade_open":
            message_text = _render_trade_open_message(
                strategy_id=str(event.strategy_id),
                run_id=str(event.run_id),
                instrument_key=event.instrument_key,
                timeframe=event.timeframe,
                side_text=_payload_text(
                    payload=event.payload_json,
                    key="side",
                    fallback="n/a",
                ),
                price_text=_payload_text(
                    payload=event.payload_json,
                    key="price",
                    fallback="n/a",
                ),
            )
        else:
            message_text = _render_trade_close_message(
                strategy_id=str(event.strategy_id),
                run_id=str(event.run_id),
                instrument_key=event.instrument_key,
                timeframe=event.timeframe,
                side_text=_payload_text(
                    payload=event.payload_json,
                    key="side",
                    fallback="n/a",
                ),
                price_text=_payload_text(
                    payload=event.payload_json,
                    key="price",
                    fallback="n/a",
                ),
            )

        return StrategyTelegramNotificationV1(
            user_id=event.user_id,
            ts=event.ts,
            strategy_id=event.strategy_id,
            run_id=event.run_id,
            event_type=normalized_event_type,
            instrument_key=event.instrument_key,
            timeframe=event.timeframe,
            message_text=message_text,
        )

    def _is_failed_debounced(
        self,
        *,
        user_id: str,
        strategy_id: str,
        normalized_error: str,
        ts: datetime,
    ) -> bool:
        """
        Check whether failed notification should be debounced for deterministic key.

        Args:
            user_id: Strategy owner identifier string.
            strategy_id: Strategy identifier string.
            normalized_error: Normalized error text.
            ts: Event timestamp.
        Returns:
            bool: `True` when notification should be skipped by debounce window.
        Assumptions:
            Debounce key is `(user_id, strategy_id, normalized_error_text)` in v1 contract.
        Raises:
            ValueError: If timestamp normalization fails.
        Side Effects:
            Stores timestamp for accepted failed notification key.
        """
        key = (user_id, strategy_id, normalized_error)
        now_value = UtcTimestamp(ts).value
        previous = self._failed_sent_at_by_key.get(key)
        if previous is not None and now_value < previous + self._failed_debounce:
            return True
        self._failed_sent_at_by_key[key] = now_value
        return False


def _render_failed_message(*, strategy_id: str, run_id: str, error_text: str) -> str:
    """
    Render deterministic plain-text message for `failed` strategy event.

    Args:
        strategy_id: Strategy identifier.
        run_id: Run identifier.
        error_text: Normalized error text.
    Returns:
        str: Formatted Telegram message text.
    Assumptions:
        Message format follows architecture contract examples for Strategy notifier v1.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _join_message_parts(
        (
            "FAILED",
            f"strategy_id={strategy_id}",
            f"run_id={run_id}",
            f"error={error_text}",
        )
    )


def _render_signal_message(
    *,
    strategy_id: str,
    run_id: str,
    instrument_key: str,
    timeframe: str,
    signal_text: str,
) -> str:
    """
    Render deterministic plain-text message for `signal` strategy event.

    Args:
        strategy_id: Strategy identifier.
        run_id: Run identifier.
        instrument_key: Instrument key.
        timeframe: Strategy timeframe code.
        signal_text: Signal text from event payload.
    Returns:
        str: Formatted Telegram message text.
    Assumptions:
        Message format follows architecture contract examples for Strategy notifier v1.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _join_message_parts(
        (
            "SIGNAL",
            f"strategy_id={strategy_id}",
            f"run_id={run_id}",
            f"instrument={instrument_key}",
            f"timeframe={timeframe}",
            f"signal={signal_text}",
        )
    )


def _render_trade_open_message(
    *,
    strategy_id: str,
    run_id: str,
    instrument_key: str,
    timeframe: str,
    side_text: str,
    price_text: str,
) -> str:
    """
    Render deterministic plain-text message for `trade_open` strategy event.

    Args:
        strategy_id: Strategy identifier.
        run_id: Run identifier.
        instrument_key: Instrument key.
        timeframe: Strategy timeframe code.
        side_text: Trade side from event payload.
        price_text: Trade price from event payload.
    Returns:
        str: Formatted Telegram message text.
    Assumptions:
        Message format follows architecture contract examples for Strategy notifier v1.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _join_message_parts(
        (
            "TRADE OPEN",
            f"strategy_id={strategy_id}",
            f"run_id={run_id}",
            f"instrument={instrument_key}",
            f"timeframe={timeframe}",
            f"side={side_text}",
            f"price={price_text}",
        )
    )


def _render_trade_close_message(
    *,
    strategy_id: str,
    run_id: str,
    instrument_key: str,
    timeframe: str,
    side_text: str,
    price_text: str,
) -> str:
    """
    Render deterministic plain-text message for `trade_close` strategy event.

    Args:
        strategy_id: Strategy identifier.
        run_id: Run identifier.
        instrument_key: Instrument key.
        timeframe: Strategy timeframe code.
        side_text: Trade side from event payload.
        price_text: Trade price from event payload.
    Returns:
        str: Formatted Telegram message text.
    Assumptions:
        Message format follows architecture contract examples for Strategy notifier v1.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _join_message_parts(
        (
            "TRADE CLOSE",
            f"strategy_id={strategy_id}",
            f"run_id={run_id}",
            f"instrument={instrument_key}",
            f"timeframe={timeframe}",
            f"side={side_text}",
            f"price={price_text}",
        )
    )


def _join_message_parts(parts: tuple[str, ...]) -> str:
    """
    Join message parts into deterministic plain-text Telegram message.

    Args:
        parts: Ordered message parts.
    Returns:
        str: Pipe-separated message string.
    Assumptions:
        Input order is already deterministic at caller side.
    Raises:
        None.
    Side Effects:
        None.
    """
    return " | ".join(parts)


def _normalize_error_text(*, text: str) -> str:
    """
    Normalize error text for debounce key and outbound message stability.

    Args:
        text: Raw error text.
    Returns:
        str: Trimmed text with collapsed internal whitespace.
    Assumptions:
        Empty values are normalized into `unknown_error`.
    Raises:
        None.
    Side Effects:
        None.
    """
    collapsed = " ".join(text.split())
    if collapsed:
        return collapsed
    return "unknown_error"


def _payload_text(*, payload: Mapping[str, Any], key: str, fallback: str) -> str:
    """
    Read payload field as deterministic trimmed string.

    Args:
        payload: Event payload mapping.
        key: Payload field name.
        fallback: Value used when payload key is missing or blank.
    Returns:
        str: Deterministic string value.
    Assumptions:
        Non-string payload values are converted with `str(value)`.
    Raises:
        None.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        return fallback
    if type(value) is str:
        normalized = value.strip()
        return normalized if normalized else fallback
    normalized = str(value).strip()
    return normalized if normalized else fallback
