from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping
from uuid import UUID, uuid4

from trading.contexts.rl_trading.domain.live_entitlements import (
    RlLiveTickerEntitlementSnapshot,
    RlLiveTickerIdentity,
    RlLiveTickerMode,
    evaluate_rl_live_ticker_entitlement,
    resolve_rl_live_ticker_limit,
)
from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.shared_kernel.primitives import UserId


class PostgresRlLiveTickerEntitlementRepository:
    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        override_table_name: str = "rl_live_ticker_entitlement_overrides",
        activation_table_name: str = "rl_live_ticker_activations",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresRlLiveTickerEntitlementRepository requires gateway")
        self._gateway = gateway
        self._override_table_name = _table_name(value=override_table_name)
        self._activation_table_name = _table_name(value=activation_table_name)

    def snapshot(
        self,
        *,
        owner_user_id: UserId,
        paid_level: str,
        mode: RlLiveTickerMode,
        requested_ticker: RlLiveTickerIdentity | None = None,
    ) -> RlLiveTickerEntitlementSnapshot:
        return evaluate_rl_live_ticker_entitlement(
            paid_level=paid_level,
            mode=mode,
            requested_ticker=requested_ticker,
            active_tickers=self._active_tickers(owner_user_id=owner_user_id),
            override_live_slots_allowed=self._override_limit(owner_user_id=owner_user_id),
        )

    def sync_profile(
        self,
        *,
        owner_user_id: UserId,
        paid_level: str,
        strategy_id: UUID,
        live_profile_id: UUID,
        mode: RlLiveTickerMode,
        requested_ticker: RlLiveTickerIdentity | None,
        profile_ready: bool,
        observed_at: datetime,
    ) -> RlLiveTickerEntitlementSnapshot:
        if mode != "live" or not profile_ready or requested_ticker is None:
            self._deactivate_profile(
                owner_user_id=owner_user_id,
                strategy_id=strategy_id,
                observed_at=observed_at,
            )
            return self.snapshot(
                owner_user_id=owner_user_id,
                paid_level=paid_level,
                mode=mode,
                requested_ticker=requested_ticker,
            )

        limit = resolve_rl_live_ticker_limit(paid_level=paid_level)
        self._gateway.fetch_one(
            query=f"""
            WITH owner_lock AS (
                SELECT pg_advisory_xact_lock(hashtext(%(owner_user_id)s))
            ),
            deactivated_previous_profile_ticker AS (
                UPDATE {self._activation_table_name}
                   SET active = FALSE,
                       deactivated_at = %(observed_at)s,
                       updated_at = %(observed_at)s
                 WHERE owner_user_id = %(owner_user_id)s::uuid
                   AND strategy_id = %(strategy_id)s::uuid
                   AND active = TRUE
                   AND NOT (
                       exchange_name = %(exchange_name)s
                       AND market_type = %(market_type)s
                       AND symbol = %(symbol)s
                   )
                 RETURNING activation_id
            ),
            override_limit AS (
                SELECT live_slots_allowed
                  FROM {self._override_table_name}
                 WHERE owner_user_id = %(owner_user_id)s::uuid
                   AND active = TRUE
                 LIMIT 1
            ),
            effective_limit AS (
                SELECT COALESCE(
                    (SELECT live_slots_allowed FROM override_limit),
                    %(paid_level_live_slots_allowed)s
                )::integer AS live_slots_allowed
            ),
            active_before AS (
                SELECT COUNT(*)::integer AS live_slots_used
                  FROM (
                      SELECT DISTINCT exchange_name, market_type, symbol
                        FROM {self._activation_table_name}
                       WHERE owner_user_id = %(owner_user_id)s::uuid
                         AND active = TRUE
                  ) active_distinct
            ),
            existing_requested AS (
                SELECT activation_id
                  FROM {self._activation_table_name}
                 WHERE owner_user_id = %(owner_user_id)s::uuid
                   AND exchange_name = %(exchange_name)s
                   AND market_type = %(market_type)s
                   AND symbol = %(symbol)s
                   AND active = TRUE
                 LIMIT 1
            ),
            inserted AS (
                INSERT INTO {self._activation_table_name}
                (
                    activation_id,
                    owner_user_id,
                    strategy_id,
                    live_profile_id,
                    exchange_name,
                    market_type,
                    symbol,
                    mode,
                    active,
                    activated_at,
                    deactivated_at,
                    created_at,
                    updated_at
                )
                SELECT
                    %(activation_id)s::uuid,
                    %(owner_user_id)s::uuid,
                    %(strategy_id)s::uuid,
                    %(live_profile_id)s::uuid,
                    %(exchange_name)s,
                    %(market_type)s,
                    %(symbol)s,
                    'live',
                    TRUE,
                    %(observed_at)s,
                    NULL,
                    %(observed_at)s,
                    %(observed_at)s
                  FROM active_before, effective_limit, owner_lock
                 WHERE EXISTS (SELECT 1 FROM existing_requested)
                    OR active_before.live_slots_used < effective_limit.live_slots_allowed
                ON CONFLICT (owner_user_id, exchange_name, market_type, symbol)
                    WHERE active
                DO UPDATE SET
                    strategy_id = EXCLUDED.strategy_id,
                    live_profile_id = EXCLUDED.live_profile_id,
                    mode = 'live',
                    updated_at = EXCLUDED.updated_at
                RETURNING activation_id
            )
            SELECT
                (SELECT live_slots_used FROM active_before) AS live_slots_used_before,
                (SELECT live_slots_allowed FROM effective_limit) AS live_slots_allowed,
                EXISTS (SELECT 1 FROM inserted) AS activation_recorded
            """,
            parameters={
                "activation_id": str(uuid4()),
                "owner_user_id": str(owner_user_id),
                "strategy_id": str(strategy_id),
                "live_profile_id": str(live_profile_id),
                "exchange_name": requested_ticker.exchange_name,
                "market_type": requested_ticker.market_type,
                "symbol": requested_ticker.symbol,
                "observed_at": observed_at,
                "paid_level_live_slots_allowed": limit.live_slots_allowed,
            },
        )
        return self.snapshot(
            owner_user_id=owner_user_id,
            paid_level=paid_level,
            mode=mode,
            requested_ticker=requested_ticker,
        )

    def _override_limit(self, *, owner_user_id: UserId) -> int | None:
        row = self._gateway.fetch_one(
            query=f"""
            SELECT live_slots_allowed
              FROM {self._override_table_name}
             WHERE owner_user_id = %(owner_user_id)s
               AND active = TRUE
             LIMIT 1
            """,
            parameters={"owner_user_id": str(owner_user_id)},
        )
        if row is None:
            return None
        return int(row["live_slots_allowed"])

    def _active_tickers(self, *, owner_user_id: UserId) -> tuple[RlLiveTickerIdentity, ...]:
        rows = self._gateway.fetch_all(
            query=f"""
            SELECT DISTINCT owner_user_id, exchange_name, market_type, symbol
              FROM {self._activation_table_name}
             WHERE owner_user_id = %(owner_user_id)s
               AND active = TRUE
             ORDER BY exchange_name, market_type, symbol
            """,
            parameters={"owner_user_id": str(owner_user_id)},
        )
        return tuple(_row_to_identity(row=row) for row in rows)

    def _deactivate_profile(
        self,
        *,
        owner_user_id: UserId,
        strategy_id: UUID,
        observed_at: datetime,
    ) -> None:
        self._gateway.execute(
            query=f"""
            UPDATE {self._activation_table_name}
               SET active = FALSE,
                   deactivated_at = %(observed_at)s,
                   updated_at = %(observed_at)s
             WHERE owner_user_id = %(owner_user_id)s
               AND strategy_id = %(strategy_id)s
               AND active = TRUE
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "strategy_id": str(strategy_id),
                "observed_at": observed_at,
            },
        )


def _row_to_identity(*, row: Mapping[str, Any]) -> RlLiveTickerIdentity:
    return RlLiveTickerIdentity(
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        exchange_name=str(row["exchange_name"]),
        market_type=str(row["market_type"]),
        symbol=str(row["symbol"]),
    )


def _table_name(*, value: str) -> str:
    normalized = value.strip()
    if not normalized.replace("_", "").isalnum():
        raise ValueError("table name contains unsupported characters")
    return normalized
