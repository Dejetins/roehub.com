from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.contexts.strategy.application.ports.repositories import (
    StrategyBacktestVariantProvenanceRepository,
)
from trading.contexts.strategy.domain.entities import (
    Strategy,
    StrategyBacktestVariantProvenance,
)
from trading.contexts.strategy.domain.errors import StrategyStorageError
from trading.shared_kernel.primitives import UserId


class PostgresStrategyBacktestVariantProvenanceRepository(
    StrategyBacktestVariantProvenanceRepository
):
    """
    Postgres repository for atomic Strategy + backtest variant provenance creation.
    """

    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        strategies_table: str = "strategy_strategies",
        provenance_table: str = "strategy_backtest_variant_provenance",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresStrategyBacktestVariantProvenanceRepository requires gateway")
        self._gateway = gateway
        self._strategies_table = strategies_table.strip()
        self._provenance_table = provenance_table.strip()
        if not self._strategies_table:
            raise ValueError("strategies_table must be non-empty")
        if not self._provenance_table:
            raise ValueError("provenance_table must be non-empty")

    def find_by_idempotency_key(
        self,
        *,
        user_id: UserId,
        idempotency_key_hash: str,
    ) -> StrategyBacktestVariantProvenance | None:
        query = f"""
        SELECT
            {_PROVENANCE_SELECT_COLUMNS}
        FROM {self._provenance_table}
        WHERE user_id = %(user_id)s
          AND idempotency_key_hash = %(idempotency_key_hash)s
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "user_id": str(user_id),
                "idempotency_key_hash": idempotency_key_hash,
            },
        )
        return None if row is None else _map_provenance_row(row=row)

    def find_by_source_variant(
        self,
        *,
        user_id: UserId,
        source_job_id: UUID,
        source_variant_key: str,
        strategy_spec_hash: str,
        launch_request_hash: str,
    ) -> StrategyBacktestVariantProvenance | None:
        query = f"""
        SELECT
            {_PROVENANCE_SELECT_COLUMNS}
        FROM {self._provenance_table}
        WHERE user_id = %(user_id)s
          AND source_job_id = %(source_job_id)s
          AND source_variant_key = %(source_variant_key)s
          AND strategy_spec_hash = %(strategy_spec_hash)s
          AND launch_request_hash = %(launch_request_hash)s
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "user_id": str(user_id),
                "source_job_id": str(source_job_id),
                "source_variant_key": source_variant_key,
                "strategy_spec_hash": strategy_spec_hash,
                "launch_request_hash": launch_request_hash,
            },
        )
        return None if row is None else _map_provenance_row(row=row)

    def create_with_strategy(
        self,
        *,
        strategy: Strategy,
        provenance: StrategyBacktestVariantProvenance,
    ) -> StrategyBacktestVariantProvenance:
        spec_json = strategy.spec.to_json()
        query = f"""
        WITH inserted_strategy AS (
            INSERT INTO {self._strategies_table}
            (
                strategy_id,
                user_id,
                name,
                instrument_id,
                instrument_key,
                market_type,
                symbol,
                timeframe,
                indicators_json,
                spec_json,
                created_at,
                is_deleted
            )
            VALUES
            (
                %(strategy_id)s,
                %(user_id)s,
                %(name)s,
                %(instrument_id)s::jsonb,
                %(instrument_key)s,
                %(market_type)s,
                %(symbol)s,
                %(timeframe)s,
                %(indicators_json)s::jsonb,
                %(spec_json)s::jsonb,
                %(created_at)s,
                %(is_deleted)s
            )
            RETURNING strategy_id
        )
        INSERT INTO {self._provenance_table}
        (
            strategy_id,
            user_id,
            source_job_id,
            source_variant_key,
            source_variant_hash,
            source_indicator_variant_hash,
            backtest_request_hash,
            backtest_result_config_hash,
            strategy_spec_hash,
            launch_request_hash,
            idempotency_key_hash,
            created_at,
            metadata_json
        )
        SELECT
            inserted_strategy.strategy_id,
            %(user_id)s,
            %(source_job_id)s,
            %(source_variant_key)s,
            %(source_variant_hash)s,
            %(source_indicator_variant_hash)s,
            %(backtest_request_hash)s,
            %(backtest_result_config_hash)s,
            %(strategy_spec_hash)s,
            %(launch_request_hash)s,
            %(idempotency_key_hash)s,
            %(created_at)s,
            %(metadata_json)s::jsonb
        FROM inserted_strategy
        RETURNING
            {_PROVENANCE_SELECT_COLUMNS}
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "strategy_id": str(strategy.strategy_id),
                "user_id": str(strategy.user_id),
                "name": strategy.name,
                "instrument_id": _json_dumps(payload=strategy.spec.instrument_id.as_dict()),
                "instrument_key": strategy.spec.instrument_key,
                "market_type": strategy.spec.market_type,
                "symbol": str(strategy.spec.instrument_id.symbol),
                "timeframe": strategy.spec.timeframe.code,
                "indicators_json": _json_dumps(payload=list(strategy.spec.indicators)),
                "spec_json": _json_dumps(payload=spec_json),
                "created_at": strategy.created_at,
                "is_deleted": strategy.is_deleted,
                "source_job_id": str(provenance.source_job_id),
                "source_variant_key": provenance.source_variant_key,
                "source_variant_hash": provenance.source_variant_hash,
                "source_indicator_variant_hash": provenance.source_indicator_variant_hash,
                "backtest_request_hash": provenance.backtest_request_hash,
                "backtest_result_config_hash": provenance.backtest_result_config_hash,
                "strategy_spec_hash": provenance.strategy_spec_hash,
                "launch_request_hash": provenance.launch_request_hash,
                "idempotency_key_hash": provenance.idempotency_key_hash,
                "metadata_json": _json_dumps(payload=provenance.metadata_json),
            },
        )
        if row is None:
            raise StrategyStorageError(
                "PostgresStrategyBacktestVariantProvenanceRepository."
                "create_with_strategy returned no row"
            )
        return _map_provenance_row(row=row)


_PROVENANCE_SELECT_COLUMNS = """
            strategy_id,
            user_id,
            source_job_id,
            source_variant_key,
            source_variant_hash,
            source_indicator_variant_hash,
            backtest_request_hash,
            backtest_result_config_hash,
            strategy_spec_hash,
            launch_request_hash,
            idempotency_key_hash,
            created_at,
            metadata_json
"""


def _map_provenance_row(*, row: Mapping[str, Any]) -> StrategyBacktestVariantProvenance:
    try:
        return StrategyBacktestVariantProvenance(
            strategy_id=UUID(str(row["strategy_id"])),
            user_id=UserId.from_string(str(row["user_id"])),
            source_job_id=UUID(str(row["source_job_id"])),
            source_variant_key=str(row["source_variant_key"]),
            source_variant_hash=str(row["source_variant_hash"]),
            source_indicator_variant_hash=(
                str(row["source_indicator_variant_hash"])
                if row.get("source_indicator_variant_hash") is not None
                else None
            ),
            backtest_request_hash=str(row["backtest_request_hash"]),
            backtest_result_config_hash=str(row["backtest_result_config_hash"]),
            strategy_spec_hash=str(row["strategy_spec_hash"]),
            launch_request_hash=str(row["launch_request_hash"]),
            idempotency_key_hash=str(row["idempotency_key_hash"]),
            created_at=_ensure_utc_datetime(value=row["created_at"]),
            metadata_json=_json_mapping(value=row.get("metadata_json")),
        )
    except Exception as error:  # noqa: BLE001
        raise StrategyStorageError("Failed to map strategy backtest provenance row") from error


def _ensure_utc_datetime(*, value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise StrategyStorageError("provenance created_at must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _json_mapping(*, value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise StrategyStorageError("provenance metadata_json must be JSON object")


def _json_dumps(*, payload: Mapping[str, Any] | list[Mapping[str, Any]]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
