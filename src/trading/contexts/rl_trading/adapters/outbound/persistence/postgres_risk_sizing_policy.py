from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID, uuid4

from trading.contexts.rl_trading.domain.risk_sizing_policy import (
    RlRiskSizingPolicyConfig,
    RlRiskSizingPolicyKey,
    RlRiskSizingPolicyRecord,
    validate_rl_risk_sizing_policy,
)
from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class PostgresRlRiskSizingPolicyRepository:
    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        policy_table_name: str = "rl_risk_sizing_policies",
        audit_table_name: str = "rl_risk_sizing_policy_audit_events",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresRlRiskSizingPolicyRepository requires gateway")
        self._gateway = gateway
        self._policy_table_name = _table_name(value=policy_table_name)
        self._audit_table_name = _table_name(value=audit_table_name)

    def get_policy(self, *, key: RlRiskSizingPolicyKey) -> RlRiskSizingPolicyRecord | None:
        row = self._gateway.fetch_one(
            query=f"""
            SELECT *
              FROM {self._policy_table_name}
             WHERE organization_id = %(organization_id)s
               AND owner_user_id = %(owner_user_id)s
               AND strategy_id = %(strategy_id)s
               AND exchange_name = %(exchange_name)s
               AND market_type = %(market_type)s
               AND symbol = %(symbol)s
             ORDER BY updated_at DESC
             LIMIT 1
            """,
            parameters=_key_parameters(key=key),
        )
        if row is None:
            return None
        return _row_to_record(row=row)

    def upsert_policy(
        self,
        *,
        key: RlRiskSizingPolicyKey,
        config: RlRiskSizingPolicyConfig,
        observed_at: datetime,
    ) -> RlRiskSizingPolicyRecord:
        validation = validate_rl_risk_sizing_policy(config=config)
        policy_id = uuid4()
        row = self._gateway.fetch_one(
            query=f"""
            WITH upserted AS (
                INSERT INTO {self._policy_table_name}
                (
                    policy_id,
                    organization_id,
                    owner_user_id,
                    strategy_id,
                    exchange_name,
                    market_type,
                    symbol,
                    active,
                    sizing_method,
                    base_quote_notional,
                    max_position_notional,
                    max_daily_loss_notional,
                    max_drawdown_pct,
                    max_turnover_notional,
                    max_exposure_notional,
                    min_expected_pnl_pct,
                    min_confidence,
                    take_profit_pct,
                    stop_loss_pct,
                    trailing_stop_pct,
                    validation_status,
                    validation_reasons,
                    synthetic_exit_rules_json,
                    created_at,
                    updated_at
                )
                VALUES (
                    %(policy_id)s::uuid,
                    %(organization_id)s::uuid,
                    %(owner_user_id)s::uuid,
                    %(strategy_id)s::uuid,
                    %(exchange_name)s,
                    %(market_type)s,
                    %(symbol)s,
                    %(active)s,
                    %(sizing_method)s,
                    %(base_quote_notional)s,
                    %(max_position_notional)s,
                    %(max_daily_loss_notional)s,
                    %(max_drawdown_pct)s,
                    %(max_turnover_notional)s,
                    %(max_exposure_notional)s,
                    %(min_expected_pnl_pct)s,
                    %(min_confidence)s,
                    %(take_profit_pct)s,
                    %(stop_loss_pct)s,
                    %(trailing_stop_pct)s,
                    %(validation_status)s,
                    %(validation_reasons)s,
                    %(synthetic_exit_rules_json)s::jsonb,
                    %(observed_at)s,
                    %(observed_at)s
                )
                ON CONFLICT (
                    organization_id,
                    owner_user_id,
                    strategy_id,
                    exchange_name,
                    market_type,
                    symbol
                )
                DO UPDATE SET
                    active = EXCLUDED.active,
                    sizing_method = EXCLUDED.sizing_method,
                    base_quote_notional = EXCLUDED.base_quote_notional,
                    max_position_notional = EXCLUDED.max_position_notional,
                    max_daily_loss_notional = EXCLUDED.max_daily_loss_notional,
                    max_drawdown_pct = EXCLUDED.max_drawdown_pct,
                    max_turnover_notional = EXCLUDED.max_turnover_notional,
                    max_exposure_notional = EXCLUDED.max_exposure_notional,
                    min_expected_pnl_pct = EXCLUDED.min_expected_pnl_pct,
                    min_confidence = EXCLUDED.min_confidence,
                    take_profit_pct = EXCLUDED.take_profit_pct,
                    stop_loss_pct = EXCLUDED.stop_loss_pct,
                    trailing_stop_pct = EXCLUDED.trailing_stop_pct,
                    validation_status = EXCLUDED.validation_status,
                    validation_reasons = EXCLUDED.validation_reasons,
                    synthetic_exit_rules_json = EXCLUDED.synthetic_exit_rules_json,
                    updated_at = EXCLUDED.updated_at
                RETURNING *
            ),
            audit AS (
                INSERT INTO {self._audit_table_name}
                (
                    event_id,
                    policy_id,
                    organization_id,
                    owner_user_id,
                    strategy_id,
                    exchange_name,
                    market_type,
                    symbol,
                    event_type,
                    validation_status,
                    validation_reasons,
                    changes_json,
                    created_at
                )
                SELECT
                    %(event_id)s::uuid,
                    policy_id,
                    organization_id,
                    owner_user_id,
                    strategy_id,
                    exchange_name,
                    market_type,
                    symbol,
                    'upsert',
                    validation_status,
                    validation_reasons,
                    %(changes_json)s::jsonb,
                    %(observed_at)s
                  FROM upserted
                RETURNING event_id
            )
            SELECT * FROM upserted
            """,
            parameters={
                **_key_parameters(key=key),
                "policy_id": str(policy_id),
                "event_id": str(uuid4()),
                "active": config.active,
                "sizing_method": config.sizing_method,
                "base_quote_notional": config.base_quote_notional,
                "max_position_notional": config.max_position_notional,
                "max_daily_loss_notional": config.max_daily_loss_notional,
                "max_drawdown_pct": config.max_drawdown_pct,
                "max_turnover_notional": config.max_turnover_notional,
                "max_exposure_notional": config.max_exposure_notional,
                "min_expected_pnl_pct": config.min_expected_pnl_pct,
                "min_confidence": config.min_confidence,
                "take_profit_pct": config.take_profit_pct,
                "stop_loss_pct": config.stop_loss_pct,
                "trailing_stop_pct": config.trailing_stop_pct,
                "validation_status": validation.status,
                "validation_reasons": list(validation.reasons),
                "synthetic_exit_rules_json": json.dumps(
                    [rule.as_payload() for rule in validation.synthetic_exit_rules],
                    sort_keys=True,
                ),
                "changes_json": json.dumps(
                    {"kind": "rl_risk_sizing_policy_upsert_v1"},
                    sort_keys=True,
                ),
                "observed_at": observed_at,
            },
        )
        if row is None:
            raise RuntimeError("rl_risk_sizing_policy_upsert_returned_no_row")
        return _row_to_record(row=row)


def _row_to_record(*, row: Mapping[str, Any]) -> RlRiskSizingPolicyRecord:
    key = RlRiskSizingPolicyKey(
        organization_id=OrganizationId.from_string(str(row["organization_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        strategy_id=UUID(str(row["strategy_id"])),
        exchange_name=str(row["exchange_name"]),
        market_type=str(row["market_type"]),
        symbol=str(row["symbol"]),
    )
    config = RlRiskSizingPolicyConfig(
        active=bool(row["active"]),
        sizing_method=str(row["sizing_method"]),  # type: ignore[arg-type]
        base_quote_notional=_decimal(row["base_quote_notional"]),
        max_position_notional=_decimal(row["max_position_notional"]),
        max_daily_loss_notional=_decimal(row["max_daily_loss_notional"]),
        max_drawdown_pct=_decimal(row["max_drawdown_pct"]),
        max_turnover_notional=_decimal(row["max_turnover_notional"]),
        max_exposure_notional=_decimal(row["max_exposure_notional"]),
        min_expected_pnl_pct=_decimal(row["min_expected_pnl_pct"]),
        min_confidence=_optional_decimal(row["min_confidence"]),
        take_profit_pct=_optional_decimal(row["take_profit_pct"]),
        stop_loss_pct=_optional_decimal(row["stop_loss_pct"]),
        trailing_stop_pct=_optional_decimal(row["trailing_stop_pct"]),
    )
    return RlRiskSizingPolicyRecord(
        policy_id=UUID(str(row["policy_id"])),
        key=key,
        config=config,
        validation=validate_rl_risk_sizing_policy(config=config),
        created_at=row["created_at"],  # type: ignore[arg-type]
        updated_at=row["updated_at"],  # type: ignore[arg-type]
    )


def _key_parameters(*, key: RlRiskSizingPolicyKey) -> dict[str, str]:
    return {
        "organization_id": str(key.organization_id),
        "owner_user_id": str(key.owner_user_id),
        "strategy_id": str(key.strategy_id),
        "exchange_name": key.exchange_name,
        "market_type": key.market_type,
        "symbol": key.symbol,
    }


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _table_name(*, value: str) -> str:
    normalized = value.strip()
    if not normalized.replace("_", "").isalnum():
        raise ValueError("table name contains unsupported characters")
    return normalized
