from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.contexts.strategy.application.ports.repositories import (
    StrategyVariantScenarioMatrixRepository,
)
from trading.contexts.strategy.application.use_cases.scenario_matrix import (
    StrategyVariantScenarioMatrixReport,
    StrategyVariantScenarioMatrixRow,
)


class PostgresStrategyVariantScenarioMatrixRepository(
    StrategyVariantScenarioMatrixRepository
):
    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        table: str = "strategy_variant_scenario_matrix_rows",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresStrategyVariantScenarioMatrixRepository requires gateway")
        self._gateway = gateway
        self._table = table.strip()
        if not self._table:
            raise ValueError("scenario matrix table name must be non-empty")

    def record(
        self, *, report: StrategyVariantScenarioMatrixReport
    ) -> StrategyVariantScenarioMatrixReport:
        for row in report.rows:
            self._gateway.execute(
                query=f"""
                INSERT INTO {self._table}
                (
                    scenario_matrix_row_id,
                    owner_user_id,
                    source_job_id,
                    source_variant_key,
                    variant_hash,
                    strategy_spec_hash,
                    scenario_key,
                    mode,
                    market_type,
                    symbol,
                    entry_sizing,
                    risk_mode,
                    direction,
                    backtest_risk_mode,
                    backtest_direction_mode,
                    scenario_state,
                    scenario_reason_codes_json,
                    order_capability,
                    order_capability_reason_codes_json,
                    compatibility_check_id,
                    market_data_requirement_id,
                    compatibility_state,
                    compatibility_reason_codes_json,
                    market_data_state,
                    market_data_reason_codes_json,
                    checked_at
                )
                VALUES
                (
                    %(scenario_matrix_row_id)s,
                    %(owner_user_id)s,
                    %(source_job_id)s,
                    %(source_variant_key)s,
                    %(variant_hash)s,
                    %(strategy_spec_hash)s,
                    %(scenario_key)s,
                    %(mode)s,
                    %(market_type)s,
                    %(symbol)s,
                    %(entry_sizing)s,
                    %(risk_mode)s,
                    %(direction)s,
                    %(backtest_risk_mode)s,
                    %(backtest_direction_mode)s,
                    %(scenario_state)s,
                    %(scenario_reason_codes_json)s::jsonb,
                    %(order_capability)s,
                    %(order_capability_reason_codes_json)s::jsonb,
                    %(compatibility_check_id)s,
                    %(market_data_requirement_id)s,
                    %(compatibility_state)s,
                    %(compatibility_reason_codes_json)s::jsonb,
                    %(market_data_state)s,
                    %(market_data_reason_codes_json)s::jsonb,
                    %(checked_at)s
                )
                ON CONFLICT (owner_user_id, source_job_id, source_variant_key, scenario_key)
                DO UPDATE SET
                    scenario_matrix_row_id = EXCLUDED.scenario_matrix_row_id,
                    variant_hash = EXCLUDED.variant_hash,
                    strategy_spec_hash = EXCLUDED.strategy_spec_hash,
                    scenario_state = EXCLUDED.scenario_state,
                    scenario_reason_codes_json = EXCLUDED.scenario_reason_codes_json,
                    order_capability = EXCLUDED.order_capability,
                    order_capability_reason_codes_json =
                        EXCLUDED.order_capability_reason_codes_json,
                    compatibility_check_id = EXCLUDED.compatibility_check_id,
                    market_data_requirement_id = EXCLUDED.market_data_requirement_id,
                    compatibility_state = EXCLUDED.compatibility_state,
                    compatibility_reason_codes_json = EXCLUDED.compatibility_reason_codes_json,
                    market_data_state = EXCLUDED.market_data_state,
                    market_data_reason_codes_json = EXCLUDED.market_data_reason_codes_json,
                    checked_at = EXCLUDED.checked_at
                """,
                parameters=_params(row=row),
            )
        return report


def _params(*, row: StrategyVariantScenarioMatrixRow) -> dict[str, Any]:
    return {
        "scenario_matrix_row_id": str(row.scenario_matrix_row_id),
        "owner_user_id": str(row.owner_user_id),
        "source_job_id": str(row.source_job_id),
        "source_variant_key": row.source_variant_key,
        "variant_hash": row.variant_hash,
        "strategy_spec_hash": row.strategy_spec_hash,
        "scenario_key": row.scenario_key,
        "mode": row.mode,
        "market_type": row.market_type,
        "symbol": row.symbol,
        "entry_sizing": row.entry_sizing,
        "risk_mode": row.risk_mode,
        "direction": row.direction,
        "backtest_risk_mode": row.backtest_risk_mode,
        "backtest_direction_mode": row.backtest_direction_mode,
        "scenario_state": row.scenario_state,
        "scenario_reason_codes_json": json.dumps(
            list(row.scenario_reason_codes), sort_keys=True
        ),
        "order_capability": row.order_capability,
        "order_capability_reason_codes_json": json.dumps(
            list(row.order_capability_reason_codes), sort_keys=True
        ),
        "compatibility_check_id": str(row.compatibility_check_id),
        "market_data_requirement_id": str(row.market_data_requirement_id),
        "compatibility_state": row.compatibility_state,
        "compatibility_reason_codes_json": json.dumps(
            list(row.compatibility_reason_codes), sort_keys=True
        ),
        "market_data_state": row.market_data_state,
        "market_data_reason_codes_json": json.dumps(
            list(row.market_data_reason_codes), sort_keys=True
        ),
        "checked_at": _utc(row.checked_at),
    }


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
