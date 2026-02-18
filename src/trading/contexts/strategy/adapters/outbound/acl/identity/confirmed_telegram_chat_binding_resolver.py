from __future__ import annotations

from typing import Any, Mapping

from trading.contexts.strategy.adapters.outbound.persistence.postgres import StrategyPostgresGateway
from trading.contexts.strategy.application.ports import ConfirmedTelegramChatBindingResolver
from trading.shared_kernel.primitives import UserId


class PostgresConfirmedTelegramChatBindingResolver(ConfirmedTelegramChatBindingResolver):
    """
    PostgresConfirmedTelegramChatBindingResolver — ACL adapter resolving confirmed Telegram chat.

    Docs:
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - migrations/postgres/0001_identity_v1.sql
      - src/trading/contexts/strategy/application/ports/telegram_notifier.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
    """

    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        channels_table: str = "identity_telegram_channels",
    ) -> None:
        """
        Initialize resolver with SQL gateway and target identity channels table.

        Args:
            gateway: Strategy SQL gateway used by worker wiring.
            channels_table: Identity Telegram channels table name.
        Returns:
            None.
        Assumptions:
            Strategy worker and identity tables are available in the same Postgres DSN.
        Raises:
            ValueError: If dependencies are invalid.
        Side Effects:
            None.
        """
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresConfirmedTelegramChatBindingResolver requires gateway")
        normalized_table = channels_table.strip()
        if not normalized_table:
            raise ValueError(
                "PostgresConfirmedTelegramChatBindingResolver requires non-empty channels_table"
            )
        self._gateway = gateway
        self._channels_table = normalized_table

    def find_confirmed_chat_id(self, *, user_id: UserId) -> int | None:
        """
        Read one confirmed Telegram chat id for user with deterministic tie-break ordering.

        Args:
            user_id: Strategy owner identifier.
        Returns:
            int | None: Confirmed chat id, or `None` when no confirmed row exists.
        Assumptions:
            Ordering is `confirmed_at DESC NULLS LAST, chat_id ASC` per v1 contract.
        Raises:
            ValueError: If SQL row shape contains invalid `chat_id`.
        Side Effects:
            Executes one SQL SELECT query.
        """
        query = f"""
        SELECT chat_id
        FROM {self._channels_table}
        WHERE user_id = %(user_id)s
          AND is_confirmed = TRUE
        ORDER BY confirmed_at DESC NULLS LAST, chat_id ASC
        LIMIT 1
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={"user_id": str(user_id)},
        )
        if row is None:
            return None
        return _read_chat_id(row=row)


def _read_chat_id(*, row: Mapping[str, Any]) -> int:
    """
    Map SQL row into Telegram chat id integer with deterministic validation.

    Args:
        row: SQL row mapping with `chat_id` column.
    Returns:
        int: Non-zero chat id value.
    Assumptions:
        Telegram chat id may be positive or negative, but never boolean/zero.
    Raises:
        ValueError: If `chat_id` is missing, non-integer, or zero.
    Side Effects:
        None.
    """
    try:
        raw_chat_id = row["chat_id"]
    except KeyError as error:
        raise ValueError(
            "PostgresConfirmedTelegramChatBindingResolver row misses chat_id"
        ) from error
    if isinstance(raw_chat_id, bool):
        raise ValueError(
            "PostgresConfirmedTelegramChatBindingResolver chat_id must not be bool"
        )
    try:
        chat_id = int(raw_chat_id)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "PostgresConfirmedTelegramChatBindingResolver chat_id must be integer"
        ) from error
    if chat_id == 0:
        raise ValueError("PostgresConfirmedTelegramChatBindingResolver chat_id must be non-zero")
    return chat_id
