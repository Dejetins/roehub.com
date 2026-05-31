from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, cast
from uuid import UUID, uuid4

from apps.api.exchange_control_client import (
    ExchangeControlAccountStateSnapshot,
    ExchangeControlClient,
    build_exchange_control_client_from_environ,
)
from trading.contexts.live_execution.adapters.outbound.persistence.postgres import (
    PostgresExchangeAccountProjectionRepository,
)
from trading.contexts.live_execution.adapters.outbound.time import SystemLiveExecutionClock
from trading.contexts.live_execution.application import ExchangeAccountProjectionService
from trading.contexts.live_execution.domain import (
    ExchangeAccountProjection,
    ExchangeBalanceSnapshot,
    ExchangeInstrumentFilterSnapshot,
    ExchangeOpenOrderSnapshot,
    ExchangePositionSnapshot,
    ExpectedInstrumentConfig,
)
from trading.contexts.strategy.adapters.outbound import PsycopgStrategyPostgresGateway
from trading.shared_kernel.primitives import UserId


class _ExchangeControlAccountStateReader:
    def __init__(
        self,
        *,
        client: ExchangeControlClient,
        instrument_keys: tuple[str, ...],
    ) -> None:
        self._client = client
        self._instrument_keys = instrument_keys

    def read_account_projection(
        self, *, owner_user_id: UserId, exchange_connection_id: UUID
    ) -> ExchangeAccountProjection:
        snapshot = self._client.read_account_state(
            owner_user_id=str(owner_user_id),
            connection_id=str(exchange_connection_id),
            instrument_keys=self._instrument_keys,
        )
        synced_at = datetime.now(UTC)
        return _projection_from_exchange_control(
            owner_user_id=owner_user_id,
            exchange_connection_id=exchange_connection_id,
            snapshot=snapshot,
            synced_at=synced_at,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sync a sanitized exchange account-state projection through exchange-control."
        )
    )
    parser.add_argument("--owner-user-id", required=True)
    parser.add_argument("--exchange-connection-id", required=True)
    parser.add_argument("--instrument-key", action="append", default=[])
    parser.add_argument("--min-notional", default="0")
    args = parser.parse_args(argv)

    client = build_exchange_control_client_from_environ(environ=os.environ)
    if client is None:
        raise RuntimeError("exchange-control internal client is not configured")
    dsn = _postgres_dsn(environ=os.environ)
    service = ExchangeAccountProjectionService(
        repository=PostgresExchangeAccountProjectionRepository(
            gateway=PsycopgStrategyPostgresGateway(dsn=dsn)
        ),
        clock=SystemLiveExecutionClock(),
    )
    instrument_keys = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in args.instrument_key
            if str(item).strip()
        )
    )
    requirements = tuple(
        ExpectedInstrumentConfig(
            instrument_key=instrument_key,
            market_type=_market_type_from_instrument_key(instrument_key),
            min_notional=Decimal(str(args.min_notional)),
        )
        for instrument_key in instrument_keys
    )
    projection = service.sync_connection(
        owner_user_id=UserId.from_string(args.owner_user_id),
        exchange_connection_id=UUID(args.exchange_connection_id),
        reader=_ExchangeControlAccountStateReader(
            client=client,
            instrument_keys=instrument_keys,
        ),
        requirements=requirements,
    )
    print(
        json.dumps(
            {
                "status": projection.sync_status,
                "reason": projection.sync_reason,
                "account_snapshot_id": str(projection.account_snapshot_id),
                "exchange_connection_id": str(projection.exchange_connection_id),
                "exchange_name": projection.exchange_name,
                "market_type": projection.market_type,
                "environment": projection.environment,
                "balance_count": len(projection.balances),
                "position_count": len(projection.positions),
                "open_order_count": len(projection.open_orders),
                "filter_count": len(projection.instrument_filters),
                "source_hash": projection.source_hash,
                "observed_at": projection.observed_at.isoformat(),
                "synced_at": projection.synced_at.isoformat(),
            },
            sort_keys=True,
        )
    )
    return 0


def _projection_from_exchange_control(
    *,
    owner_user_id: UserId,
    exchange_connection_id: UUID,
    snapshot: ExchangeControlAccountStateSnapshot,
    synced_at: datetime,
) -> ExchangeAccountProjection:
    return ExchangeAccountProjection(
        account_snapshot_id=uuid4(),
        owner_user_id=owner_user_id,
        exchange_connection_id=exchange_connection_id,
        exchange_name=snapshot.exchange_name,
        market_type=snapshot.market_type,
        environment=snapshot.environment,
        account_mode=snapshot.account_mode,
        balances=tuple(
            ExchangeBalanceSnapshot(
                asset=item.asset,
                free=item.free,
                locked=item.locked,
                total=item.total,
            )
            for item in snapshot.balances
        ),
        positions=tuple(
            ExchangePositionSnapshot(
                instrument_key=item.instrument_key,
                side=_position_side(item.side),
                quantity=item.quantity,
                entry_price=item.entry_price,
                leverage=item.leverage,
                margin_mode=item.margin_mode,
                position_mode=item.position_mode,
            )
            for item in snapshot.positions
        ),
        open_orders=tuple(
            ExchangeOpenOrderSnapshot(
                instrument_key=item.instrument_key,
                exchange_order_ref=item.exchange_order_ref,
                side=_order_side(item.side),
                order_type=item.order_type,
                quantity=item.quantity,
                price=item.price,
                status=item.status,
            )
            for item in snapshot.open_orders
        ),
        instrument_filters=tuple(
            ExchangeInstrumentFilterSnapshot(
                instrument_key=item.instrument_key,
                tick_size=item.tick_size,
                step_size=item.step_size,
                min_qty=item.min_qty,
                min_notional=item.min_notional,
                max_leverage=item.max_leverage,
            )
            for item in snapshot.instrument_filters
        ),
        source_hash=snapshot.source_hash,
        observed_at=snapshot.observed_at,
        synced_at=synced_at,
        sync_status=cast(Literal["fresh", "degraded"], snapshot.sync_status),
        sync_reason=snapshot.sync_reason,
        metadata={"source": "exchange_control_account_state"},
    )


def _position_side(value: str) -> Literal["long", "short", "net"]:
    if value in {"long", "short", "net"}:
        return cast(Literal["long", "short", "net"], value)
    return "net"


def _order_side(value: str) -> Literal["buy", "sell"]:
    return "sell" if value == "sell" else "buy"


def _market_type_from_instrument_key(value: str) -> str:
    parts = value.split(":")
    if len(parts) == 3 and parts[1].strip():
        return parts[1].strip()
    return "spot"


def _postgres_dsn(*, environ: Mapping[str, str]) -> str:
    for key in ("STRATEGY_PG_DSN", "POSTGRES_DSN", "IDENTITY_PG_DSN"):
        value = environ.get(key, "").strip()
        if value:
            return value
    raise RuntimeError("Postgres DSN is required via STRATEGY_PG_DSN/POSTGRES_DSN")


if __name__ == "__main__":
    raise SystemExit(main())
