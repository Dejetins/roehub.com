from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, Mapping
from uuid import UUID

from trading.contexts.live_execution.application.ports import (
    ExchangeAccountProjectionRepository,
)
from trading.contexts.live_execution.domain import (
    AccountConfigGuardResult,
    ExchangeAccountProjection,
    ExchangeBalanceSnapshot,
    ExchangeInstrumentFilterSnapshot,
    ExchangeOpenOrderSnapshot,
    ExchangePositionSnapshot,
    ExpectedInstrumentConfig,
)
from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class PostgresExchangeAccountProjectionRepository(ExchangeAccountProjectionRepository):
    def __init__(self, *, gateway: StrategyPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresExchangeAccountProjectionRepository requires gateway")
        self._gateway = gateway

    def record_projection(
        self, *, projection: ExchangeAccountProjection
    ) -> ExchangeAccountProjection:
        self._gateway.execute(
            query="""
            INSERT INTO exchange_account_snapshots
            (
                account_snapshot_id, organization_id, owner_user_id, exchange_connection_id,
                exchange_name, market_type, environment, account_mode,
                source_hash, sync_status, sync_reason, observed_at, synced_at,
                balance_count, position_count, open_order_count, filter_count,
                metadata_json
            )
            VALUES
            (
                %(account_snapshot_id)s, %(organization_id)s, %(owner_user_id)s,
                %(exchange_connection_id)s,
                %(exchange_name)s, %(market_type)s, %(environment)s, %(account_mode)s,
                %(source_hash)s, %(sync_status)s, %(sync_reason)s, %(observed_at)s,
                %(synced_at)s, %(balance_count)s, %(position_count)s,
                %(open_order_count)s, %(filter_count)s, %(metadata_json)s::jsonb
            )
            ON CONFLICT (account_snapshot_id) DO NOTHING
            """,
            parameters=_projection_params(projection=projection),
        )
        for balance in projection.balances:
            self._gateway.execute(
                query="""
                INSERT INTO exchange_balance_snapshots
                (
                    balance_snapshot_id, account_snapshot_id, organization_id, owner_user_id,
                    exchange_connection_id, asset, free, locked, total, observed_at
                )
                VALUES
                (
                    gen_random_uuid(), %(account_snapshot_id)s, %(organization_id)s,
                    %(owner_user_id)s,
                    %(exchange_connection_id)s, %(asset)s, %(free)s, %(locked)s,
                    %(total)s, %(observed_at)s
                )
                """,
                parameters={**_projection_keys(projection=projection), **_balance_params(balance)},
            )
        for position in projection.positions:
            self._gateway.execute(
                query="""
                INSERT INTO exchange_position_snapshots
                (
                    position_snapshot_id, account_snapshot_id, organization_id, owner_user_id,
                    exchange_connection_id, instrument_key, side, quantity,
                    entry_price, leverage, margin_mode, position_mode, observed_at
                )
                VALUES
                (
                    gen_random_uuid(), %(account_snapshot_id)s, %(organization_id)s,
                    %(owner_user_id)s,
                    %(exchange_connection_id)s, %(instrument_key)s, %(side)s,
                    %(quantity)s, %(entry_price)s, %(leverage)s, %(margin_mode)s,
                    %(position_mode)s, %(observed_at)s
                )
                """,
                parameters={
                    **_projection_keys(projection=projection),
                    **_position_params(position),
                },
            )
        for order in projection.open_orders:
            self._gateway.execute(
                query="""
                INSERT INTO exchange_open_order_snapshots
                (
                    open_order_snapshot_id, account_snapshot_id, organization_id, owner_user_id,
                    exchange_connection_id, instrument_key, exchange_order_ref,
                    side, order_type, quantity, price, status, observed_at
                )
                VALUES
                (
                    gen_random_uuid(), %(account_snapshot_id)s, %(organization_id)s,
                    %(owner_user_id)s,
                    %(exchange_connection_id)s, %(instrument_key)s,
                    %(exchange_order_ref)s, %(side)s, %(order_type)s, %(quantity)s,
                    %(price)s, %(status)s, %(observed_at)s
                )
                """,
                parameters={**_projection_keys(projection=projection), **_order_params(order)},
            )
        for item in projection.instrument_filters:
            self._gateway.execute(
                query="""
                INSERT INTO exchange_instrument_filter_snapshots
                (
                    filter_snapshot_id, account_snapshot_id, organization_id, owner_user_id,
                    exchange_connection_id, instrument_key, tick_size, step_size,
                    min_qty, min_notional, max_leverage, observed_at
                )
                VALUES
                (
                    gen_random_uuid(), %(account_snapshot_id)s, %(organization_id)s,
                    %(owner_user_id)s,
                    %(exchange_connection_id)s, %(instrument_key)s, %(tick_size)s,
                    %(step_size)s, %(min_qty)s, %(min_notional)s, %(max_leverage)s,
                    %(observed_at)s
                )
                """,
                parameters={**_projection_keys(projection=projection), **_filter_params(item)},
            )
        return projection

    def record_config_guard_result(
        self, *, result: AccountConfigGuardResult
    ) -> AccountConfigGuardResult:
        self._gateway.execute(
            query="""
            INSERT INTO exchange_account_config_guard_results
            (
                config_guard_result_id, account_snapshot_id, organization_id, owner_user_id,
                exchange_connection_id, instrument_key, market_type, status,
                reason_codes_json, requirement_json, checked_at
            )
            VALUES
            (
                %(config_guard_result_id)s, %(account_snapshot_id)s, %(organization_id)s,
                %(owner_user_id)s, %(exchange_connection_id)s, %(instrument_key)s,
                %(market_type)s, %(status)s, %(reason_codes_json)s::jsonb,
                %(requirement_json)s::jsonb, %(checked_at)s
            )
            ON CONFLICT (config_guard_result_id) DO NOTHING
            """,
            parameters=_guard_params(result=result),
        )
        return result

    def get_latest_projection(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
    ) -> ExchangeAccountProjection | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT *
            FROM exchange_account_snapshots
            WHERE organization_id = %(organization_id)s
              AND owner_user_id = %(owner_user_id)s
              AND exchange_connection_id = %(exchange_connection_id)s
            ORDER BY observed_at DESC, synced_at DESC, account_snapshot_id DESC
            LIMIT 1
            """,
            parameters={
                "organization_id": str(organization_id),
                "owner_user_id": str(owner_user_id),
                "exchange_connection_id": str(exchange_connection_id),
            },
        )
        if row is None:
            return None
        account_snapshot_id = UUID(str(row["account_snapshot_id"]))
        return ExchangeAccountProjection(
            account_snapshot_id=account_snapshot_id,
            organization_id=OrganizationId.from_string(str(row["organization_id"])),
            owner_user_id=UserId.from_string(str(row["owner_user_id"])),
            exchange_connection_id=UUID(str(row["exchange_connection_id"])),
            exchange_name=str(row["exchange_name"]),
            market_type=str(row["market_type"]),
            environment=str(row["environment"]),
            account_mode=str(row["account_mode"]),
            balances=self._load_balances(
                organization_id=organization_id,
                account_snapshot_id=account_snapshot_id,
            ),
            positions=self._load_positions(
                organization_id=organization_id,
                account_snapshot_id=account_snapshot_id,
            ),
            open_orders=self._load_open_orders(
                organization_id=organization_id,
                account_snapshot_id=account_snapshot_id,
            ),
            instrument_filters=self._load_filters(
                organization_id=organization_id,
                account_snapshot_id=account_snapshot_id,
            ),
            source_hash=str(row["source_hash"]),
            observed_at=_utc(row["observed_at"]),
            synced_at=_utc(row["synced_at"]),
            sync_status=str(row["sync_status"]),  # type: ignore[arg-type]
            sync_reason=str(row["sync_reason"]),
            metadata=dict(row.get("metadata_json") or {}),
        )

    def get_latest_config_guard_result(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        instrument_key: str,
        market_type: str,
    ) -> AccountConfigGuardResult | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT *
            FROM exchange_account_config_guard_results
            WHERE organization_id = %(organization_id)s
              AND owner_user_id = %(owner_user_id)s
              AND exchange_connection_id = %(exchange_connection_id)s
              AND instrument_key = %(instrument_key)s
              AND market_type = %(market_type)s
            ORDER BY checked_at DESC, config_guard_result_id DESC
            LIMIT 1
            """,
            parameters={
                "organization_id": str(organization_id),
                "owner_user_id": str(owner_user_id),
                "exchange_connection_id": str(exchange_connection_id),
                "instrument_key": instrument_key,
                "market_type": market_type,
            },
        )
        if row is None:
            return None
        requirement = _requirement_from_json(row.get("requirement_json") or {})
        return AccountConfigGuardResult(
            config_guard_result_id=UUID(str(row["config_guard_result_id"])),
            account_snapshot_id=(
                UUID(str(row["account_snapshot_id"]))
                if row.get("account_snapshot_id") is not None
                else None
            ),
            organization_id=OrganizationId.from_string(str(row["organization_id"])),
            owner_user_id=UserId.from_string(str(row["owner_user_id"])),
            exchange_connection_id=UUID(str(row["exchange_connection_id"])),
            instrument_key=str(row["instrument_key"]),
            market_type=str(row["market_type"]),
            status=str(row["status"]),  # type: ignore[arg-type]
            reason_codes=tuple(row.get("reason_codes_json") or ()),
            checked_at=_utc(row["checked_at"]),
            requirement=requirement,
        )

    def _load_balances(
        self, *, organization_id: OrganizationId, account_snapshot_id: UUID
    ) -> tuple[ExchangeBalanceSnapshot, ...]:
        rows = self._gateway.fetch_all(
            query="""
            SELECT asset, free, locked, total
            FROM exchange_balance_snapshots
            WHERE organization_id = %(organization_id)s
              AND account_snapshot_id = %(account_snapshot_id)s
            ORDER BY asset
            """,
            parameters={
                "organization_id": str(organization_id),
                "account_snapshot_id": str(account_snapshot_id),
            },
        )
        return tuple(
            ExchangeBalanceSnapshot(
                asset=str(row["asset"]),
                free=_decimal(row["free"]),
                locked=_decimal(row["locked"]),
                total=_decimal_or_none(row.get("total")),
            )
            for row in rows
        )

    def _load_positions(
        self, *, organization_id: OrganizationId, account_snapshot_id: UUID
    ) -> tuple[ExchangePositionSnapshot, ...]:
        rows = self._gateway.fetch_all(
            query="""
            SELECT instrument_key, side, quantity, entry_price, leverage,
                   margin_mode, position_mode
            FROM exchange_position_snapshots
            WHERE organization_id = %(organization_id)s
              AND account_snapshot_id = %(account_snapshot_id)s
            ORDER BY instrument_key, side
            """,
            parameters={
                "organization_id": str(organization_id),
                "account_snapshot_id": str(account_snapshot_id),
            },
        )
        return tuple(
            ExchangePositionSnapshot(
                instrument_key=str(row["instrument_key"]),
                side=str(row["side"]),  # type: ignore[arg-type]
                quantity=_decimal(row["quantity"]),
                entry_price=_decimal_or_none(row.get("entry_price")),
                leverage=_decimal_or_none(row.get("leverage")),
                margin_mode=_str_or_none(row.get("margin_mode")),
                position_mode=_str_or_none(row.get("position_mode")),
            )
            for row in rows
        )

    def _load_open_orders(
        self, *, organization_id: OrganizationId, account_snapshot_id: UUID
    ) -> tuple[ExchangeOpenOrderSnapshot, ...]:
        rows = self._gateway.fetch_all(
            query="""
            SELECT instrument_key, exchange_order_ref, side, order_type, quantity, price, status
            FROM exchange_open_order_snapshots
            WHERE organization_id = %(organization_id)s
              AND account_snapshot_id = %(account_snapshot_id)s
            ORDER BY instrument_key, exchange_order_ref
            """,
            parameters={
                "organization_id": str(organization_id),
                "account_snapshot_id": str(account_snapshot_id),
            },
        )
        return tuple(
            ExchangeOpenOrderSnapshot(
                instrument_key=str(row["instrument_key"]),
                exchange_order_ref=str(row["exchange_order_ref"]),
                side=str(row["side"]),  # type: ignore[arg-type]
                order_type=str(row["order_type"]),
                quantity=_decimal(row["quantity"]),
                price=_decimal_or_none(row.get("price")),
                status=str(row["status"]),
            )
            for row in rows
        )

    def _load_filters(
        self, *, organization_id: OrganizationId, account_snapshot_id: UUID
    ) -> tuple[ExchangeInstrumentFilterSnapshot, ...]:
        rows = self._gateway.fetch_all(
            query="""
            SELECT instrument_key, tick_size, step_size, min_qty, min_notional, max_leverage
            FROM exchange_instrument_filter_snapshots
            WHERE organization_id = %(organization_id)s
              AND account_snapshot_id = %(account_snapshot_id)s
            ORDER BY instrument_key
            """,
            parameters={
                "organization_id": str(organization_id),
                "account_snapshot_id": str(account_snapshot_id),
            },
        )
        return tuple(
            ExchangeInstrumentFilterSnapshot(
                instrument_key=str(row["instrument_key"]),
                tick_size=_decimal_or_none(row.get("tick_size")),
                step_size=_decimal_or_none(row.get("step_size")),
                min_qty=_decimal_or_none(row.get("min_qty")),
                min_notional=_decimal_or_none(row.get("min_notional")),
                max_leverage=_decimal_or_none(row.get("max_leverage")),
            )
            for row in rows
        )


def _projection_params(*, projection: ExchangeAccountProjection) -> dict[str, Any]:
    return {
        **_projection_keys(projection=projection),
        "exchange_name": projection.exchange_name,
        "market_type": projection.market_type,
        "environment": projection.environment,
        "account_mode": projection.account_mode,
        "source_hash": projection.source_hash,
        "sync_status": projection.sync_status,
        "sync_reason": projection.sync_reason,
        "synced_at": _utc(projection.synced_at),
        "balance_count": len(projection.balances),
        "position_count": len(projection.positions),
        "open_order_count": len(projection.open_orders),
        "filter_count": len(projection.instrument_filters),
        "metadata_json": json.dumps(projection.metadata, sort_keys=True),
    }


def _projection_keys(*, projection: ExchangeAccountProjection) -> dict[str, Any]:
    return {
        "account_snapshot_id": str(projection.account_snapshot_id),
        "organization_id": str(projection.organization_id),
        "owner_user_id": str(projection.owner_user_id),
        "exchange_connection_id": str(projection.exchange_connection_id),
        "observed_at": _utc(projection.observed_at),
    }


def _balance_params(balance: ExchangeBalanceSnapshot) -> dict[str, Any]:
    return {
        "asset": balance.asset,
        "free": balance.free,
        "locked": balance.locked,
        "total": balance.total,
    }


def _position_params(position: ExchangePositionSnapshot) -> dict[str, Any]:
    return {
        "instrument_key": position.instrument_key,
        "side": position.side,
        "quantity": position.quantity,
        "entry_price": position.entry_price,
        "leverage": position.leverage,
        "margin_mode": position.margin_mode,
        "position_mode": position.position_mode,
    }


def _order_params(order: ExchangeOpenOrderSnapshot) -> dict[str, Any]:
    return {
        "instrument_key": order.instrument_key,
        "exchange_order_ref": order.exchange_order_ref,
        "side": order.side,
        "order_type": order.order_type,
        "quantity": order.quantity,
        "price": order.price,
        "status": order.status,
    }


def _filter_params(item: ExchangeInstrumentFilterSnapshot) -> dict[str, Any]:
    return {
        "instrument_key": item.instrument_key,
        "tick_size": item.tick_size,
        "step_size": item.step_size,
        "min_qty": item.min_qty,
        "min_notional": item.min_notional,
        "max_leverage": item.max_leverage,
    }


def _guard_params(*, result: AccountConfigGuardResult) -> dict[str, Any]:
    return {
        "config_guard_result_id": str(result.config_guard_result_id),
        "account_snapshot_id": (
            str(result.account_snapshot_id) if result.account_snapshot_id else None
        ),
        "organization_id": str(result.organization_id),
        "owner_user_id": str(result.owner_user_id),
        "exchange_connection_id": str(result.exchange_connection_id),
        "instrument_key": result.instrument_key,
        "market_type": result.market_type,
        "status": result.status,
        "reason_codes_json": json.dumps(result.reason_codes, sort_keys=True),
        "requirement_json": json.dumps(_requirement_to_json(result.requirement), sort_keys=True),
        "checked_at": _utc(result.checked_at),
    }


def _requirement_to_json(requirement: ExpectedInstrumentConfig) -> dict[str, str | None]:
    return {
        "instrument_key": requirement.instrument_key,
        "market_type": requirement.market_type,
        "side": requirement.side,
        "expected_margin_mode": requirement.expected_margin_mode,
        "expected_position_mode": requirement.expected_position_mode,
        "required_leverage": _decimal_to_str(requirement.required_leverage),
        "order_notional": _decimal_to_str(requirement.order_notional),
        "required_balance_asset": requirement.required_balance_asset,
        "min_notional": _decimal_to_str(requirement.min_notional),
        "tick_size": _decimal_to_str(requirement.tick_size),
        "step_size": _decimal_to_str(requirement.step_size),
    }


def _requirement_from_json(payload: Mapping[str, Any]) -> ExpectedInstrumentConfig:
    return ExpectedInstrumentConfig(
        instrument_key=str(payload.get("instrument_key") or ""),
        market_type=str(payload.get("market_type") or ""),
        side=_side_or_none(payload.get("side")),
        expected_margin_mode=_str_or_none(payload.get("expected_margin_mode")),
        expected_position_mode=_str_or_none(payload.get("expected_position_mode")),
        required_leverage=_decimal_or_none(payload.get("required_leverage")),
        order_notional=_decimal_or_none(payload.get("order_notional")),
        required_balance_asset=_str_or_none(payload.get("required_balance_asset")),
        min_notional=_decimal_or_none(payload.get("min_notional")),
        tick_size=_decimal_or_none(payload.get("tick_size")),
        step_size=_decimal_or_none(payload.get("step_size")),
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _decimal_to_str(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    return raw or None


def _side_or_none(value: Any) -> Literal["long", "short"] | None:
    raw = _str_or_none(value)
    if raw == "long":
        return "long"
    if raw == "short":
        return "short"
    return None
