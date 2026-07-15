from __future__ import annotations

from trading.contexts.live_execution.application.ports import (
    ExecutionRiskContextQuery,
    ExecutionRiskContextResolutionError,
    ExecutionRiskContextResolver,
)
from trading.contexts.live_execution.domain import ExecutionRiskContext
from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)


class PostgresExecutionRiskContextResolver(ExecutionRiskContextResolver):
    """Resolve durable ownership facts and fail closed for unwired policies."""

    def __init__(self, *, gateway: StrategyPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresExecutionRiskContextResolver requires gateway")
        self._gateway = gateway

    def resolve(self, *, query: ExecutionRiskContextQuery) -> ExecutionRiskContext:
        source = self._gateway.fetch_one(
            query="""
            SELECT source_type
            FROM execution_source_events
            WHERE organization_id = %(organization_id)s
              AND owner_user_id = %(owner_user_id)s
              AND source_event_id = %(source_event_id)s
            """,
            parameters={
                "organization_id": str(query.organization_id),
                "owner_user_id": str(query.owner_user_id),
                "source_event_id": str(query.source_event_id),
            },
        )
        if source is None:
            raise ExecutionRiskContextResolutionError(reason="source_event_not_found")

        connection = self._gateway.fetch_one(
            query="""
            SELECT
                connection.connection_id,
                connection.exchange_name,
                connection.market_type,
                connection.environment,
                connection.status,
                connection.active_credential_version_id,
                credential.status AS credential_status
            FROM exchange_connections AS connection
            LEFT JOIN exchange_credential_versions AS credential
              ON credential.organization_id = connection.organization_id
             AND credential.credential_version_id =
                    connection.active_credential_version_id
            WHERE connection.organization_id = %(organization_id)s
              AND connection.owner_user_id = %(owner_user_id)s
              AND connection.connection_id = %(exchange_connection_id)s
            """,
            parameters={
                "organization_id": str(query.organization_id),
                "owner_user_id": str(query.owner_user_id),
                "exchange_connection_id": str(query.exchange_connection_id),
            },
        )
        if connection is None:
            raise ExecutionRiskContextResolutionError(reason="account_ownership_mismatch")

        exchange_name = str(connection["exchange_name"]).strip().casefold()
        market_type = str(connection["market_type"]).strip().casefold()
        instrument_exchange = query.instrument_key.split(":", maxsplit=1)[0].casefold()
        return ExecutionRiskContext(
            organization_ownership_verified=True,
            account_ownership_verified=True,
            exchange_connection_active=str(connection["status"]) == "active",
            secret_custody_ready=(
                connection["active_credential_version_id"] is not None
                and str(connection["credential_status"]) == "active"
            ),
            source_authorized=True,
            exchange_config_verified=(
                exchange_name == instrument_exchange and market_type == query.market_type
            ),
            account_state_fresh=False,
            kill_switch_open=False,
            environment_policy_allows=str(connection["environment"]) == "testnet",
            max_order_size_ok=False,
            daily_limit_ok=False,
        )
