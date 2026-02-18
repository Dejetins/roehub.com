from __future__ import annotations

from typing import Any, Mapping

import pytest

from trading.contexts.strategy.adapters.outbound import (
    PostgresConfirmedTelegramChatBindingResolver,
)
from trading.shared_kernel.primitives import UserId


class _GatewayStub:
    """
    SQL gateway stub recording last query and returning configured row.
    """

    def __init__(self, *, row: Mapping[str, Any] | None) -> None:
        """
        Initialize gateway stub payload.

        Args:
            row: Row returned by `fetch_one`.
        Returns:
            None.
        Assumptions:
            Stub is used for deterministic resolver tests only.
        Raises:
            None.
        Side Effects:
            Stores mutable call arguments for assertions.
        """
        self._row = row
        self.last_query = ""
        self.last_parameters: dict[str, Any] = {}

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Return configured row and record call arguments.

        Args:
            query: SQL query text.
            parameters: SQL bind parameters.
        Returns:
            Mapping[str, Any] | None: Configured row.
        Assumptions:
            Resolver issues exactly one `fetch_one` call.
        Raises:
            None.
        Side Effects:
            Records query text and parameters.
        """
        self.last_query = query
        self.last_parameters = dict(parameters)
        return self._row

    def fetch_all(
        self,
        *,
        query: str,
        parameters: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        """
        Return empty rows for protocol compatibility in tests.

        Args:
            query: SQL query text.
            parameters: SQL bind parameters.
        Returns:
            tuple[Mapping[str, Any], ...]: Empty row set.
        Assumptions:
            Method is never used by resolver under test.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = (query, parameters)
        return ()

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        """
        No-op execute method for protocol compatibility in tests.

        Args:
            query: SQL query text.
            parameters: SQL bind parameters.
        Returns:
            None.
        Assumptions:
            Method is never used by resolver under test.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = (query, parameters)


def test_postgres_confirmed_telegram_chat_binding_resolver_reads_confirmed_chat_id() -> None:
    """
    Ensure resolver uses deterministic SQL ordering and maps confirmed chat id.
    """
    gateway = _GatewayStub(row={"chat_id": 123456789})
    resolver = PostgresConfirmedTelegramChatBindingResolver(gateway=gateway)
    user_id = UserId.from_string("00000000-0000-0000-0000-000000006001")

    chat_id = resolver.find_confirmed_chat_id(user_id=user_id)

    assert chat_id == 123456789
    assert gateway.last_parameters == {"user_id": str(user_id)}
    assert "FROM identity_telegram_channels" in gateway.last_query
    assert "is_confirmed = TRUE" in gateway.last_query
    assert "ORDER BY confirmed_at DESC NULLS LAST, chat_id ASC" in gateway.last_query


def test_postgres_confirmed_telegram_chat_binding_resolver_returns_none_when_missing() -> None:
    """
    Ensure resolver returns None when no confirmed chat binding row exists.
    """
    gateway = _GatewayStub(row=None)
    resolver = PostgresConfirmedTelegramChatBindingResolver(gateway=gateway)
    user_id = UserId.from_string("00000000-0000-0000-0000-000000006002")

    chat_id = resolver.find_confirmed_chat_id(user_id=user_id)

    assert chat_id is None


def test_postgres_confirmed_telegram_chat_binding_resolver_rejects_invalid_chat_id() -> None:
    """
    Ensure resolver rejects malformed chat id mapping values.
    """
    gateway = _GatewayStub(row={"chat_id": 0})
    resolver = PostgresConfirmedTelegramChatBindingResolver(gateway=gateway)
    user_id = UserId.from_string("00000000-0000-0000-0000-000000006003")

    with pytest.raises(ValueError):
        resolver.find_confirmed_chat_id(user_id=user_id)
