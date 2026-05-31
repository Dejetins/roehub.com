"""Add testnet execution order and private stream ledgers."""

from __future__ import annotations

from alembic import op

revision = "20260531_0027"
down_revision = "20260531_0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE exchange_execution_process_heartbeats
        DROP CONSTRAINT IF EXISTS exchange_execution_process_heartbeats_adapter_mode_chk
        """
    )
    op.execute(
        """
        ALTER TABLE exchange_execution_process_heartbeats
        ADD CONSTRAINT exchange_execution_process_heartbeats_adapter_mode_chk
            CHECK (adapter_mode IN ('disabled', 'testnet'))
        """
    )
    op.execute(
        """
        ALTER TABLE exchange_execution_request_observations
        DROP CONSTRAINT IF EXISTS exchange_execution_request_observations_status_chk
        """
    )
    op.execute(
        """
        ALTER TABLE exchange_execution_request_observations
        ADD CONSTRAINT exchange_execution_request_observations_status_chk
            CHECK (
                status IN (
                    'adapter_disabled',
                    'adapter_error',
                    'guard_rejected',
                    'quarantined',
                    'skipped',
                    'testnet_submitted'
                )
            )
        """
    )
    op.execute(
        """
        ALTER TABLE exchange_execution_request_observations
        DROP CONSTRAINT IF EXISTS exchange_execution_request_observations_adapter_mode_chk
        """
    )
    op.execute(
        """
        ALTER TABLE exchange_execution_request_observations
        ADD CONSTRAINT exchange_execution_request_observations_adapter_mode_chk
            CHECK (adapter_mode IN ('disabled', 'testnet'))
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_orders (
            order_id UUID PRIMARY KEY,
            intent_id UUID NOT NULL UNIQUE REFERENCES execution_intents(intent_id),
            owner_user_id UUID NOT NULL,
            exchange_connection_id UUID NOT NULL,
            exchange_name TEXT NOT NULL,
            environment TEXT NOT NULL,
            market_type TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            quantity NUMERIC(28, 12) NULL,
            quote_notional NUMERIC(28, 12) NULL,
            limit_price NUMERIC(28, 12) NULL,
            client_order_id TEXT NOT NULL,
            exchange_order_id TEXT NULL,
            status TEXT NOT NULL,
            status_reason TEXT NOT NULL,
            submitted_at TIMESTAMPTZ NULL,
            cancel_requested_at TIMESTAMPTZ NULL,
            cancelled_at TIMESTAMPTZ NULL,
            last_checked_at TIMESTAMPTZ NULL,
            adapter_attempt_count INTEGER NOT NULL DEFAULT 0,
            latency_ms DOUBLE PRECISION NULL,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT execution_orders_exchange_chk CHECK (exchange_name IN ('binance', 'bybit')),
            CONSTRAINT execution_orders_environment_chk CHECK (environment IN ('testnet')),
            CONSTRAINT execution_orders_market_type_chk CHECK (market_type IN ('spot', 'futures')),
            CONSTRAINT execution_orders_side_chk CHECK (side IN ('buy', 'sell')),
            CONSTRAINT execution_orders_order_type_chk CHECK (order_type IN ('market', 'limit')),
            CONSTRAINT execution_orders_status_chk CHECK (
                status IN (
                    'guard_rejected',
                    'submit_pending',
                    'submitted',
                    'status_checked',
                    'cancelled',
                    'adapter_error',
                    'unknown'
                )
            ),
            CONSTRAINT execution_orders_reason_chk CHECK (char_length(trim(status_reason)) > 0),
            CONSTRAINT execution_orders_client_order_chk
                CHECK (char_length(trim(client_order_id)) > 0),
            CONSTRAINT execution_orders_metadata_chk CHECK (jsonb_typeof(metadata_json) = 'object')
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_orders_owner_created
            ON execution_orders (owner_user_id, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_orders_exchange_status
            ON execution_orders (exchange_name, environment, status, updated_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS exchange_private_stream_sessions (
            session_id UUID PRIMARY KEY,
            exchange_connection_id UUID NOT NULL,
            exchange_name TEXT NOT NULL,
            environment TEXT NOT NULL,
            market_type TEXT NOT NULL,
            status TEXT NOT NULL,
            status_reason TEXT NOT NULL,
            opened_at TIMESTAMPTZ NOT NULL,
            keepalive_at TIMESTAMPTZ NULL,
            expires_at TIMESTAMPTZ NULL,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT exchange_private_stream_sessions_exchange_chk
                CHECK (exchange_name IN ('binance', 'bybit')),
            CONSTRAINT exchange_private_stream_sessions_environment_chk
                CHECK (environment IN ('testnet')),
            CONSTRAINT exchange_private_stream_sessions_market_type_chk
                CHECK (market_type IN ('spot', 'futures')),
            CONSTRAINT exchange_private_stream_sessions_status_chk
                CHECK (status IN ('ready', 'degraded', 'not_ready')),
            CONSTRAINT exchange_private_stream_sessions_reason_chk
                CHECK (char_length(trim(status_reason)) > 0),
            CONSTRAINT exchange_private_stream_sessions_metadata_chk
                CHECK (jsonb_typeof(metadata_json) = 'object'),
            CONSTRAINT exchange_private_stream_sessions_unique
                UNIQUE (exchange_connection_id, exchange_name, market_type, environment)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_exchange_private_stream_sessions_status
            ON exchange_private_stream_sessions (exchange_name, status, updated_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_exchange_private_stream_sessions_status")
    op.execute("DROP TABLE IF EXISTS exchange_private_stream_sessions")
    op.execute("DROP INDEX IF EXISTS idx_execution_orders_exchange_status")
    op.execute("DROP INDEX IF EXISTS idx_execution_orders_owner_created")
    op.execute("DROP TABLE IF EXISTS execution_orders")
    op.execute(
        """
        ALTER TABLE exchange_execution_request_observations
        DROP CONSTRAINT IF EXISTS exchange_execution_request_observations_adapter_mode_chk
        """
    )
    op.execute(
        """
        ALTER TABLE exchange_execution_request_observations
        ADD CONSTRAINT exchange_execution_request_observations_adapter_mode_chk
            CHECK (adapter_mode IN ('disabled'))
        """
    )
    op.execute(
        """
        ALTER TABLE exchange_execution_request_observations
        DROP CONSTRAINT IF EXISTS exchange_execution_request_observations_status_chk
        """
    )
    op.execute(
        """
        ALTER TABLE exchange_execution_request_observations
        ADD CONSTRAINT exchange_execution_request_observations_status_chk
            CHECK (status IN ('adapter_disabled', 'quarantined', 'skipped'))
        """
    )
    op.execute(
        """
        ALTER TABLE exchange_execution_process_heartbeats
        DROP CONSTRAINT IF EXISTS exchange_execution_process_heartbeats_adapter_mode_chk
        """
    )
    op.execute(
        """
        ALTER TABLE exchange_execution_process_heartbeats
        ADD CONSTRAINT exchange_execution_process_heartbeats_adapter_mode_chk
            CHECK (adapter_mode IN ('disabled'))
        """
    )
