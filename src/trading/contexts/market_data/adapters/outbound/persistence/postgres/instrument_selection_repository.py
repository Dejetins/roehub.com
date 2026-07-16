from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Mapping, Sequence
from uuid import uuid4

from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, OrganizationId, Symbol, UserId


@dataclass(frozen=True, slots=True)
class InstrumentSelectionRecord:
    organization_id: OrganizationId
    instrument_id: InstrumentId
    selected_by_user_id: UserId
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class InstrumentHistoryBound:
    instrument_id: InstrumentId
    expected_start_at: datetime
    confirmed_at: datetime


class PostgresInstrumentSelectionRepository:
    """Organization-scoped intent with a separate global effective reader."""

    def __init__(self, *, gateway: BacktestPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresInstrumentSelectionRepository requires gateway")
        self._gateway = gateway

    def list_for_organization(
        self, *, organization_id: OrganizationId
    ) -> tuple[InstrumentSelectionRecord, ...]:
        rows = self._gateway.fetch_all(
            query="""
                SELECT organization_id, market_id, symbol, selected_by_user_id,
                       created_at, updated_at
                FROM market_data_instrument_selections
                WHERE organization_id = %(organization_id)s
                ORDER BY market_id ASC, symbol ASC
            """,
            parameters={"organization_id": str(organization_id)},
        )
        return tuple(_record(row) for row in rows)

    def select(
        self,
        *,
        organization_id: OrganizationId,
        actor_user_id: UserId,
        instrument_id: InstrumentId,
        now: datetime,
    ) -> None:
        normalized_now = _utc(now)
        parameters = _selection_parameters(
            organization_id=organization_id,
            actor_user_id=actor_user_id,
            instrument_id=instrument_id,
            now=normalized_now,
        )
        self._gateway.execute(
            query="""
                WITH selection AS (
                    INSERT INTO market_data_instrument_selections
                    (organization_id, market_id, symbol, selected_by_user_id,
                     created_at, updated_at)
                    VALUES
                    (%(organization_id)s, %(market_id)s, %(symbol)s,
                     %(actor_user_id)s, %(now)s, %(now)s)
                    ON CONFLICT (organization_id, market_id, symbol) DO UPDATE
                    SET selected_by_user_id = EXCLUDED.selected_by_user_id,
                        updated_at = EXCLUDED.updated_at
                    RETURNING 1
                )
                INSERT INTO market_data_instrument_selection_audit_events
                (event_id, organization_id, actor_user_id, market_id, symbol, action, created_at)
                SELECT %(event_id)s, %(organization_id)s, %(actor_user_id)s,
                       %(market_id)s, %(symbol)s, 'selected', %(now)s
                FROM selection
            """,
            parameters={**parameters, "event_id": str(uuid4())},
        )

    def unselect(
        self,
        *,
        organization_id: OrganizationId,
        actor_user_id: UserId,
        instrument_id: InstrumentId,
        now: datetime,
    ) -> None:
        parameters = _selection_parameters(
            organization_id=organization_id,
            actor_user_id=actor_user_id,
            instrument_id=instrument_id,
            now=_utc(now),
        )
        self._gateway.execute(
            query="""
                WITH removed AS (
                    DELETE FROM market_data_instrument_selections
                    WHERE organization_id = %(organization_id)s
                      AND market_id = %(market_id)s
                      AND symbol = %(symbol)s
                    RETURNING 1
                )
                INSERT INTO market_data_instrument_selection_audit_events
                (event_id, organization_id, actor_user_id, market_id, symbol, action, created_at)
                SELECT %(event_id)s, %(organization_id)s, %(actor_user_id)s,
                       %(market_id)s, %(symbol)s, 'unselected', %(now)s
                FROM removed
            """,
            parameters={**parameters, "event_id": str(uuid4())},
        )

    def is_strategy_pinned(
        self, *, organization_id: OrganizationId, instrument_id: InstrumentId
    ) -> bool:
        row = self._gateway.fetch_one(
            query="""
                SELECT 1
                FROM strategy_runs AS runs
                JOIN strategy_variant_compatibility_checks AS checks
                  ON checks.organization_id = runs.organization_id
                 AND checks.strategy_id = runs.strategy_id
                WHERE runs.organization_id = %(organization_id)s
                  AND runs.state IN ('starting', 'warming_up', 'running', 'stopping')
                  AND checks.compatibility_state = 'launchable'
                  AND checks.instrument_key = %(instrument_key)s
                LIMIT 1
            """,
            parameters={
                "organization_id": str(organization_id),
                "instrument_key": _instrument_key(instrument_id),
            },
        )
        return row is not None

    def list_global_effective(self) -> Sequence[InstrumentId]:
        rows = self._gateway.fetch_all(
            query="""
                WITH explicit_selections AS (
                    SELECT market_id, symbol
                    FROM market_data_instrument_selections
                ), strategy_pins AS (
                    SELECT
                        CASE split_part(checks.instrument_key, ':', 1)
                            WHEN 'binance' THEN CASE split_part(checks.instrument_key, ':', 2)
                                WHEN 'spot' THEN 1 WHEN 'futures' THEN 2 END
                            WHEN 'bybit' THEN CASE split_part(checks.instrument_key, ':', 2)
                                WHEN 'spot' THEN 3 WHEN 'futures' THEN 4 END
                        END AS market_id,
                        split_part(checks.instrument_key, ':', 3) AS symbol
                    FROM strategy_runs AS runs
                    JOIN strategy_variant_compatibility_checks AS checks
                      ON checks.organization_id = runs.organization_id
                     AND checks.strategy_id = runs.strategy_id
                    WHERE runs.state IN ('starting', 'warming_up', 'running', 'stopping')
                      AND checks.compatibility_state = 'launchable'
                )
                SELECT market_id, symbol FROM explicit_selections
                UNION
                SELECT market_id, symbol FROM strategy_pins WHERE market_id IS NOT NULL
                ORDER BY market_id ASC, symbol ASC
            """,
            parameters={},
        )
        if not rows:
            return (InstrumentId(MarketId(2), Symbol("BTCUSDT")),)
        return tuple(
            InstrumentId(MarketId(int(row["market_id"])), Symbol(str(row["symbol"])))
            for row in rows
        )

    def list_enabled_tradable(self) -> Sequence[InstrumentId]:
        """Expose the global effective collector set through the worker read port."""
        return self.list_global_effective()

    def catalog_state(self, *, market_id: MarketId, now: datetime) -> str:
        """Return a fail-closed freshness state without exposing provider details."""
        row = self._gateway.fetch_one(
            query="""
                SELECT state, refreshed_at
                FROM market_data_catalog_refresh_state
                WHERE market_id = %(market_id)s
            """,
            parameters={"market_id": market_id.value},
        )
        if row is None:
            return "stale"
        state = str(row.get("state", "stale"))
        refreshed_at = row.get("refreshed_at")
        if state == "fresh" and isinstance(refreshed_at, datetime):
            if _utc(refreshed_at) < _utc(now) - timedelta(minutes=30):
                return "stale"
        return state if state in {"fresh", "stale", "failed"} else "failed"

    def mark_catalog_fresh(
        self, *, market_ids: Sequence[MarketId], now: datetime
    ) -> None:
        self._upsert_catalog_state(
            market_ids=market_ids,
            state="fresh",
            refreshed_at=_utc(now),
            last_error_code=None,
        )

    def mark_catalog_failed(
        self, *, market_ids: Sequence[MarketId], now: datetime
    ) -> None:
        self._upsert_catalog_state(
            market_ids=market_ids,
            state="failed",
            refreshed_at=None,
            last_error_code="catalog_refresh_failed",
            now=_utc(now),
        )

    def record_history_bound(
        self,
        *,
        instrument_id: InstrumentId,
        expected_start_at: datetime,
        confirmed_at: datetime,
    ) -> None:
        self._gateway.execute(
            query="""
                INSERT INTO market_data_instrument_history_bounds
                (market_id, symbol, expected_start_at, confirmed_at)
                VALUES
                (%(market_id)s, %(symbol)s, %(expected_start_at)s, %(confirmed_at)s)
                ON CONFLICT (market_id, symbol) DO UPDATE
                SET expected_start_at = EXCLUDED.expected_start_at,
                    confirmed_at = EXCLUDED.confirmed_at
            """,
            parameters={
                "market_id": instrument_id.market_id.value,
                "symbol": str(instrument_id.symbol),
                "expected_start_at": _utc(expected_start_at),
                "confirmed_at": _utc(confirmed_at),
            },
        )

    def list_history_bounds(
        self, *, instrument_ids: Sequence[InstrumentId]
    ) -> Mapping[tuple[int, str], InstrumentHistoryBound]:
        if not instrument_ids:
            return {}
        requested = {
            (instrument.market_id.value, str(instrument.symbol))
            for instrument in instrument_ids
        }
        rows = self._gateway.fetch_all(
            query="""
                SELECT market_id, symbol, expected_start_at, confirmed_at
                FROM market_data_instrument_history_bounds
                WHERE market_id = ANY(%(market_ids)s)
            """,
            parameters={"market_ids": sorted({market_id for market_id, _ in requested})},
        )
        bounds: dict[tuple[int, str], InstrumentHistoryBound] = {}
        for row in rows:
            key = (int(row["market_id"]), str(row["symbol"]))
            if key not in requested:
                continue
            instrument_id = InstrumentId(MarketId(key[0]), Symbol(key[1]))
            bounds[key] = InstrumentHistoryBound(
                instrument_id=instrument_id,
                expected_start_at=_utc(row["expected_start_at"]),
                confirmed_at=_utc(row["confirmed_at"]),
            )
        return bounds

    def _upsert_catalog_state(
        self,
        *,
        market_ids: Sequence[MarketId],
        state: str,
        refreshed_at: datetime | None,
        last_error_code: str | None,
        now: datetime | None = None,
    ) -> None:
        updated_at = now or refreshed_at
        if updated_at is None:
            raise ValueError("catalog refresh state requires updated_at")
        for market_id in market_ids:
            self._gateway.execute(
                query="""
                    INSERT INTO market_data_catalog_refresh_state
                    (market_id, state, refreshed_at, last_error_code, updated_at)
                    VALUES
                    (%(market_id)s, %(state)s, %(refreshed_at)s,
                     %(last_error_code)s, %(updated_at)s)
                    ON CONFLICT (market_id) DO UPDATE
                    SET state = EXCLUDED.state,
                        refreshed_at = EXCLUDED.refreshed_at,
                        last_error_code = EXCLUDED.last_error_code,
                        updated_at = EXCLUDED.updated_at
                """,
                parameters={
                    "market_id": market_id.value,
                    "state": state,
                    "refreshed_at": refreshed_at,
                    "last_error_code": last_error_code,
                    "updated_at": updated_at,
                },
            )


def _selection_parameters(
    *,
    organization_id: OrganizationId,
    actor_user_id: UserId,
    instrument_id: InstrumentId,
    now: datetime,
) -> dict[str, object]:
    return {
        "organization_id": str(organization_id),
        "actor_user_id": str(actor_user_id),
        "market_id": instrument_id.market_id.value,
        "symbol": str(instrument_id.symbol),
        "now": now,
    }


def _record(row: Mapping[str, object]) -> InstrumentSelectionRecord:
    return InstrumentSelectionRecord(
        organization_id=OrganizationId.from_string(str(row["organization_id"])),
        instrument_id=InstrumentId(
            MarketId(int(str(row["market_id"]))),
            Symbol(str(row["symbol"])),
        ),
        selected_by_user_id=UserId.from_string(str(row["selected_by_user_id"])),
        created_at=_utc(row["created_at"]),
        updated_at=_utc(row["updated_at"]),
    )


def _instrument_key(instrument_id: InstrumentId) -> str:
    market_code = {
        1: "binance:spot",
        2: "binance:futures",
        3: "bybit:spot",
        4: "bybit:futures",
    }.get(instrument_id.market_id.value)
    if market_code is None:
        raise ValueError(
            f"unsupported market id for strategy pin lookup: {instrument_id.market_id}"
        )
    return f"{market_code}:{instrument_id.symbol}"


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("instrument selection timestamps must be datetime values")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
