from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.live_execution.application.ports import (
    PaperScenarioCoverageRepository,
)
from trading.contexts.live_execution.domain import PaperScenarioCoverageResult
from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.shared_kernel.primitives import UserId


class PostgresPaperScenarioCoverageRepository(PaperScenarioCoverageRepository):
    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        table: str = "strategy_paper_scenario_coverage_results",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresPaperScenarioCoverageRepository requires gateway")
        self._gateway = gateway
        self._table = table.strip()
        if not self._table:
            raise ValueError("paper coverage table name must be non-empty")

    def record(
        self, *, result: PaperScenarioCoverageResult
    ) -> PaperScenarioCoverageResult:
        row = self._gateway.fetch_one(
            query=f"""
            INSERT INTO {self._table}
            (
                coverage_result_id, owner_user_id, scenario_matrix_row_id,
                scenario_key, source_job_id, source_variant_key, mode, market_type,
                symbol, entry_sizing, risk_mode, direction, coverage_state,
                coverage_reason, strategy_id, live_profile_id, strategy_run_id,
                strategy_signal_id, source_event_id, intent_id, paper_order_id,
                paper_fill_id, accounting_id, fee_model, funding_model,
                pnl_complete, no_exchange_dispatch, checked_at
            )
            VALUES
            (
                %(coverage_result_id)s, %(owner_user_id)s, %(scenario_matrix_row_id)s,
                %(scenario_key)s, %(source_job_id)s, %(source_variant_key)s,
                %(mode)s, %(market_type)s, %(symbol)s, %(entry_sizing)s,
                %(risk_mode)s, %(direction)s, %(coverage_state)s,
                %(coverage_reason)s, %(strategy_id)s, %(live_profile_id)s,
                %(strategy_run_id)s, %(strategy_signal_id)s, %(source_event_id)s,
                %(intent_id)s, %(paper_order_id)s, %(paper_fill_id)s,
                %(accounting_id)s, %(fee_model)s, %(funding_model)s,
                %(pnl_complete)s, %(no_exchange_dispatch)s, %(checked_at)s
            )
            ON CONFLICT (owner_user_id, scenario_key)
            DO UPDATE SET
                coverage_result_id = EXCLUDED.coverage_result_id,
                scenario_matrix_row_id = EXCLUDED.scenario_matrix_row_id,
                source_job_id = EXCLUDED.source_job_id,
                source_variant_key = EXCLUDED.source_variant_key,
                mode = EXCLUDED.mode,
                market_type = EXCLUDED.market_type,
                symbol = EXCLUDED.symbol,
                entry_sizing = EXCLUDED.entry_sizing,
                risk_mode = EXCLUDED.risk_mode,
                direction = EXCLUDED.direction,
                coverage_state = EXCLUDED.coverage_state,
                coverage_reason = EXCLUDED.coverage_reason,
                strategy_id = EXCLUDED.strategy_id,
                live_profile_id = EXCLUDED.live_profile_id,
                strategy_run_id = EXCLUDED.strategy_run_id,
                strategy_signal_id = EXCLUDED.strategy_signal_id,
                source_event_id = EXCLUDED.source_event_id,
                intent_id = EXCLUDED.intent_id,
                paper_order_id = EXCLUDED.paper_order_id,
                paper_fill_id = EXCLUDED.paper_fill_id,
                accounting_id = EXCLUDED.accounting_id,
                fee_model = EXCLUDED.fee_model,
                funding_model = EXCLUDED.funding_model,
                pnl_complete = EXCLUDED.pnl_complete,
                no_exchange_dispatch = EXCLUDED.no_exchange_dispatch,
                checked_at = EXCLUDED.checked_at
            RETURNING *
            """,
            parameters=_params(result=result),
        )
        if row is None:
            raise ValueError("paper coverage upsert returned no row")
        return _map_result(row)

    def get_latest_by_scenario_key(
        self, *, owner_user_id: UserId, scenario_key: str
    ) -> PaperScenarioCoverageResult | None:
        row = self._gateway.fetch_one(
            query=f"""
            SELECT *
            FROM {self._table}
            WHERE owner_user_id = %(owner_user_id)s
              AND scenario_key = %(scenario_key)s
            ORDER BY checked_at DESC, coverage_result_id DESC
            LIMIT 1
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "scenario_key": scenario_key,
            },
        )
        return _map_result(row) if row is not None else None


def _params(*, result: PaperScenarioCoverageResult) -> dict[str, object]:
    return {
        "coverage_result_id": str(result.coverage_result_id),
        "owner_user_id": str(result.owner_user_id),
        "scenario_matrix_row_id": str(result.scenario_matrix_row_id),
        "scenario_key": result.scenario_key,
        "source_job_id": str(result.source_job_id),
        "source_variant_key": result.source_variant_key,
        "mode": result.mode,
        "market_type": result.market_type,
        "symbol": result.symbol,
        "entry_sizing": result.entry_sizing,
        "risk_mode": result.risk_mode,
        "direction": result.direction,
        "coverage_state": result.coverage_state,
        "coverage_reason": result.coverage_reason,
        "strategy_id": _uuid_param(result.strategy_id),
        "live_profile_id": _uuid_param(result.live_profile_id),
        "strategy_run_id": _uuid_param(result.strategy_run_id),
        "strategy_signal_id": _uuid_param(result.strategy_signal_id),
        "source_event_id": _uuid_param(result.source_event_id),
        "intent_id": _uuid_param(result.intent_id),
        "paper_order_id": _uuid_param(result.paper_order_id),
        "paper_fill_id": _uuid_param(result.paper_fill_id),
        "accounting_id": _uuid_param(result.accounting_id),
        "fee_model": result.fee_model,
        "funding_model": result.funding_model,
        "pnl_complete": result.pnl_complete,
        "no_exchange_dispatch": result.no_exchange_dispatch,
        "checked_at": _utc(result.checked_at),
    }


def _map_result(row: Mapping[str, Any]) -> PaperScenarioCoverageResult:
    return PaperScenarioCoverageResult(
        coverage_result_id=UUID(str(row["coverage_result_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        scenario_matrix_row_id=UUID(str(row["scenario_matrix_row_id"])),
        scenario_key=str(row["scenario_key"]),
        source_job_id=UUID(str(row["source_job_id"])),
        source_variant_key=str(row["source_variant_key"]),
        mode=str(row["mode"]),
        market_type=str(row["market_type"]),
        symbol=str(row["symbol"]),
        entry_sizing=str(row["entry_sizing"]),
        risk_mode=str(row["risk_mode"]),
        direction=str(row["direction"]),
        coverage_state=str(row["coverage_state"]),  # type: ignore[arg-type]
        coverage_reason=str(row["coverage_reason"]),
        strategy_id=_uuid_optional(row.get("strategy_id")),
        live_profile_id=_uuid_optional(row.get("live_profile_id")),
        strategy_run_id=_uuid_optional(row.get("strategy_run_id")),
        strategy_signal_id=_uuid_optional(row.get("strategy_signal_id")),
        source_event_id=_uuid_optional(row.get("source_event_id")),
        intent_id=_uuid_optional(row.get("intent_id")),
        paper_order_id=_uuid_optional(row.get("paper_order_id")),
        paper_fill_id=_uuid_optional(row.get("paper_fill_id")),
        accounting_id=_uuid_optional(row.get("accounting_id")),
        fee_model=_str_or_none(row.get("fee_model")),
        funding_model=_str_or_none(row.get("funding_model")),
        pnl_complete=bool(row["pnl_complete"]),
        no_exchange_dispatch=bool(row["no_exchange_dispatch"]),
        checked_at=_utc(row["checked_at"]),
    )


def _uuid_param(value: UUID | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _uuid_optional(value: object | None) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _str_or_none(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
