from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from trading.contexts.market_data.adapters.outbound.persistence.postgres import (
    PostgresInstrumentSelectionRepository,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, OrganizationId, Symbol, UserId


class _Gateway:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Mapping[str, Any]]] = []
        self.catalog_row: Mapping[str, Any] | None = None

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        _ = parameters
        if "market_data_catalog_refresh_state" in query:
            return self.catalog_row
        if "strategy_variant_compatibility_checks" in query:
            return None
        raise AssertionError(f"unexpected fetch_one query: {query}")

    def fetch_all(
        self, *, query: str, parameters: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], ...]:
        _ = parameters
        if "market_data_instrument_selections" in query and "WITH explicit" not in query:
            return ()
        if "WITH explicit_selections" in query:
            return ()
        raise AssertionError(f"unexpected fetch_all query: {query}")

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        self.executed.append((query, dict(parameters)))


def test_selection_audits_and_global_reader_uses_only_explicit_or_strategy_sources() -> None:
    gateway = _Gateway()
    repository = PostgresInstrumentSelectionRepository(gateway=gateway)
    organization_id = OrganizationId.from_string("00000000-0000-0000-0000-000000000001")
    actor_user_id = UserId.from_string("00000000-0000-0000-0000-000000000101")
    instrument_id = InstrumentId(MarketId(2), Symbol("BTCUSDT"))
    now = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)

    repository.select(
        organization_id=organization_id,
        actor_user_id=actor_user_id,
        instrument_id=instrument_id,
        now=now,
    )

    assert repository.list_global_effective() == (instrument_id,)
    assert repository.list_enabled_tradable() == (instrument_id,)
    assert len(gateway.executed) == 1
    assert "market_data_instrument_selections" in gateway.executed[0][0]
    assert "market_data_instrument_selection_audit_events" in gateway.executed[0][0]
    assert "ref_instruments" not in gateway.executed[0][0]


def test_catalog_state_is_stale_after_freshness_window_and_failure_is_redacted() -> None:
    gateway = _Gateway()
    repository = PostgresInstrumentSelectionRepository(gateway=gateway)
    now = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    gateway.catalog_row = {
        "state": "fresh",
        "refreshed_at": now - timedelta(minutes=31),
    }

    assert repository.catalog_state(market_id=MarketId(2), now=now) == "stale"

    repository.mark_catalog_failed(market_ids=(MarketId(2),), now=now)

    query, parameters = gateway.executed[-1]
    assert "market_data_catalog_refresh_state" in query
    assert parameters["state"] == "failed"
    assert parameters["last_error_code"] == "catalog_refresh_failed"
