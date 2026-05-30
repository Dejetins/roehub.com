from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.contexts.strategy.application.ports.repositories import StrategySignalRepository
from trading.contexts.strategy.domain.entities import StrategySignal
from trading.contexts.strategy.domain.errors import StrategyStorageError
from trading.shared_kernel.primitives import UserId


class PostgresStrategySignalRepository(StrategySignalRepository):
    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        table_name: str = "strategy_signals",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresStrategySignalRepository requires gateway")
        normalized_table = table_name.strip()
        if not normalized_table:
            raise ValueError("PostgresStrategySignalRepository requires table")
        self._gateway = gateway
        self._table_name = normalized_table

    def record(self, *, signal: StrategySignal) -> StrategySignal:
        row = self._gateway.fetch_one(
            query=f"""
            INSERT INTO {self._table_name}
            (
                signal_id,
                owner_user_id,
                strategy_id,
                strategy_run_id,
                live_profile_id,
                mode,
                instrument_key,
                market_type,
                timeframe,
                bar_ts_open,
                bar_ts_close,
                signal_action,
                side,
                outcome,
                reason_code,
                reference_price,
                confidence,
                expected_order_json,
                source_message_id,
                evaluator_version,
                created_at
            )
            VALUES
            (
                %(signal_id)s,
                %(owner_user_id)s,
                %(strategy_id)s,
                %(strategy_run_id)s,
                %(live_profile_id)s,
                %(mode)s,
                %(instrument_key)s,
                %(market_type)s,
                %(timeframe)s,
                %(bar_ts_open)s,
                %(bar_ts_close)s,
                %(signal_action)s,
                %(side)s,
                %(outcome)s,
                %(reason_code)s,
                %(reference_price)s,
                %(confidence)s,
                %(expected_order_json)s::jsonb,
                %(source_message_id)s,
                %(evaluator_version)s,
                %(created_at)s
            )
            ON CONFLICT (signal_id) DO NOTHING
            RETURNING
                signal_id,
                owner_user_id,
                strategy_id,
                strategy_run_id,
                live_profile_id,
                mode,
                instrument_key,
                market_type,
                timeframe,
                bar_ts_open,
                bar_ts_close,
                signal_action,
                side,
                outcome,
                reason_code,
                reference_price,
                confidence,
                expected_order_json,
                source_message_id,
                evaluator_version,
                created_at
            """,
            parameters=_signal_parameters(signal=signal),
        )
        if row is None:
            row = self._gateway.fetch_one(
                query=f"""
                SELECT
                    signal_id,
                    owner_user_id,
                    strategy_id,
                    strategy_run_id,
                    live_profile_id,
                    mode,
                    instrument_key,
                    market_type,
                    timeframe,
                    bar_ts_open,
                    bar_ts_close,
                    signal_action,
                    side,
                    outcome,
                    reason_code,
                    reference_price,
                    confidence,
                    expected_order_json,
                    source_message_id,
                    evaluator_version,
                    created_at
                FROM {self._table_name}
                WHERE signal_id = %(signal_id)s
                """,
                parameters={"signal_id": str(signal.signal_id)},
            )
        if row is None:
            raise StrategyStorageError("PostgresStrategySignalRepository.record returned no row")
        return _map_signal(row=row)

    def list_latest_for_strategy(
        self,
        *,
        owner_user_id: UserId,
        strategy_id: UUID,
        limit: int,
    ) -> tuple[StrategySignal, ...]:
        bounded_limit = max(0, min(int(limit), 100))
        rows = self._gateway.fetch_all(
            query=f"""
            SELECT
                signal_id,
                owner_user_id,
                strategy_id,
                strategy_run_id,
                live_profile_id,
                mode,
                instrument_key,
                market_type,
                timeframe,
                bar_ts_open,
                bar_ts_close,
                signal_action,
                side,
                outcome,
                reason_code,
                reference_price,
                confidence,
                expected_order_json,
                source_message_id,
                evaluator_version,
                created_at
            FROM {self._table_name}
            WHERE owner_user_id = %(owner_user_id)s
              AND strategy_id = %(strategy_id)s
            ORDER BY created_at DESC, signal_id DESC
            LIMIT %(limit)s
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "strategy_id": str(strategy_id),
                "limit": bounded_limit,
            },
        )
        return tuple(_map_signal(row=row) for row in rows)


def _signal_parameters(*, signal: StrategySignal) -> dict[str, object]:
    return {
        "signal_id": str(signal.signal_id),
        "owner_user_id": str(signal.owner_user_id),
        "strategy_id": str(signal.strategy_id),
        "strategy_run_id": str(signal.strategy_run_id),
        "live_profile_id": str(signal.live_profile_id) if signal.live_profile_id else None,
        "mode": signal.mode,
        "instrument_key": signal.instrument_key,
        "market_type": signal.market_type,
        "timeframe": signal.timeframe,
        "bar_ts_open": signal.bar_ts_open,
        "bar_ts_close": signal.bar_ts_close,
        "signal_action": signal.signal_action,
        "side": signal.side,
        "outcome": signal.outcome,
        "reason_code": signal.reason_code,
        "reference_price": signal.reference_price,
        "confidence": signal.confidence,
        "expected_order_json": json.dumps(
            signal.expected_order_json,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
        "source_message_id": signal.source_message_id,
        "evaluator_version": signal.evaluator_version,
        "created_at": signal.created_at,
    }


def _map_signal(*, row: Mapping[str, Any]) -> StrategySignal:
    return StrategySignal(
        signal_id=UUID(str(row["signal_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        strategy_id=UUID(str(row["strategy_id"])),
        strategy_run_id=UUID(str(row["strategy_run_id"])),
        live_profile_id=(
            UUID(str(row["live_profile_id"])) if row["live_profile_id"] is not None else None
        ),
        mode=str(row["mode"]),  # type: ignore[arg-type]
        instrument_key=str(row["instrument_key"]),
        market_type=str(row["market_type"]),
        timeframe=str(row["timeframe"]),
        bar_ts_open=_normalize_datetime(value=row["bar_ts_open"]),
        bar_ts_close=_normalize_datetime(value=row["bar_ts_close"]),
        signal_action=str(row["signal_action"]),  # type: ignore[arg-type]
        side=str(row["side"]) if row["side"] is not None else None,  # type: ignore[arg-type]
        outcome=str(row["outcome"]),  # type: ignore[arg-type]
        reason_code=str(row["reason_code"]),
        reference_price=_decimal(value=row["reference_price"]),
        confidence=_decimal(value=row["confidence"]) if row["confidence"] is not None else None,
        expected_order_json=_json_object(value=row["expected_order_json"]),
        source_message_id=str(row["source_message_id"]),
        evaluator_version=str(row["evaluator_version"]),
        created_at=_normalize_datetime(value=row["created_at"]),
    )


def _decimal(*, value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _json_object(*, value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        loaded = json.loads(value)
        if isinstance(loaded, Mapping):
            return dict(loaded)
    raise StrategyStorageError("strategy signal expected_order_json is invalid")


def _normalize_datetime(*, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise StrategyStorageError("strategy signal datetime is invalid")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
