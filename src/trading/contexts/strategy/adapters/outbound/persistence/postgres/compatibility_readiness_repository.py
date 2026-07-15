from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.contexts.strategy.application.ports.repositories import (
    StrategyCompatibilityReadinessRepository,
)
from trading.contexts.strategy.application.use_cases.compatibility_readiness import (
    StrategyCompatibilityReadinessReport,
)


class PostgresStrategyCompatibilityReadinessRepository(
    StrategyCompatibilityReadinessRepository
):
    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        checks_table: str = "strategy_variant_compatibility_checks",
        requirements_table: str = "market_data_subscription_requirements",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresStrategyCompatibilityReadinessRepository requires gateway")
        self._gateway = gateway
        self._checks_table = checks_table.strip()
        self._requirements_table = requirements_table.strip()
        if not self._checks_table or not self._requirements_table:
            raise ValueError("compatibility readiness table names must be non-empty")

    def record(
        self, *, report: StrategyCompatibilityReadinessReport
    ) -> StrategyCompatibilityReadinessReport:
        self._gateway.execute(
            query=f"""
            INSERT INTO {self._checks_table}
            (
                compatibility_check_id,
                organization_id,
                owner_user_id,
                strategy_id,
                source_job_id,
                source_variant_key,
                strategy_spec_hash,
                instrument_key,
                market_type,
                timeframe,
                compatibility_state,
                reason_codes_json,
                checked_at
            )
            VALUES
            (
                %(compatibility_check_id)s,
                %(organization_id)s,
                %(owner_user_id)s,
                %(strategy_id)s,
                %(source_job_id)s,
                %(source_variant_key)s,
                %(strategy_spec_hash)s,
                %(instrument_key)s,
                %(market_type)s,
                %(timeframe)s,
                %(compatibility_state)s,
                %(compatibility_reason_codes)s::jsonb,
                %(checked_at)s
            )
            ON CONFLICT (compatibility_check_id) DO NOTHING
            """,
            parameters=_params(report=report),
        )
        self._gateway.execute(
            query=f"""
            INSERT INTO {self._requirements_table}
            (
                market_data_requirement_id,
                compatibility_check_id,
                organization_id,
                owner_user_id,
                strategy_id,
                source_job_id,
                source_variant_key,
                instrument_key,
                market_type,
                timeframe,
                readiness_state,
                reason_codes_json,
                stream_name,
                stream_length,
                last_message_id,
                last_observed_at,
                age_seconds,
                checked_at
            )
            VALUES
            (
                %(market_data_requirement_id)s,
                %(compatibility_check_id)s,
                %(organization_id)s,
                %(owner_user_id)s,
                %(strategy_id)s,
                %(source_job_id)s,
                %(source_variant_key)s,
                %(instrument_key)s,
                %(market_type)s,
                %(timeframe)s,
                %(market_data_state)s,
                %(market_data_reason_codes)s::jsonb,
                %(market_data_stream_name)s,
                %(market_data_stream_length)s,
                %(market_data_last_message_id)s,
                %(market_data_last_observed_at)s,
                %(market_data_age_seconds)s,
                %(checked_at)s
            )
            ON CONFLICT (market_data_requirement_id) DO NOTHING
            """,
            parameters=_params(report=report),
        )
        return report


def _params(*, report: StrategyCompatibilityReadinessReport) -> dict[str, Any]:
    return {
        "compatibility_check_id": str(report.compatibility_check_id),
        "market_data_requirement_id": str(report.market_data_requirement_id),
        "organization_id": str(report.organization_id),
        "owner_user_id": str(report.owner_user_id),
        "strategy_id": str(report.strategy_id) if report.strategy_id is not None else None,
        "source_job_id": str(report.source_job_id) if report.source_job_id is not None else None,
        "source_variant_key": report.source_variant_key,
        "strategy_spec_hash": report.strategy_spec_hash,
        "instrument_key": report.instrument_key,
        "market_type": report.market_type,
        "timeframe": report.timeframe,
        "compatibility_state": report.compatibility_state,
        "compatibility_reason_codes": json.dumps(
            list(report.compatibility_reason_codes), sort_keys=True
        ),
        "market_data_state": report.market_data_state,
        "market_data_reason_codes": json.dumps(
            list(report.market_data_reason_codes), sort_keys=True
        ),
        "market_data_stream_name": report.market_data_stream_name,
        "market_data_stream_length": report.market_data_stream_length,
        "market_data_last_message_id": report.market_data_last_message_id,
        "market_data_last_observed_at": _utc_or_none(report.market_data_last_observed_at),
        "market_data_age_seconds": report.market_data_age_seconds,
        "checked_at": _utc_or_none(report.checked_at),
    }


def _utc_or_none(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
