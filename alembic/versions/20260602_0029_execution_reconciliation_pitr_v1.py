"""Add execution fill reconciliation and PITR ledgers."""

from __future__ import annotations

from alembic import op

revision = "20260602_0029"
down_revision = "20260531_0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE execution_orders
        DROP CONSTRAINT IF EXISTS execution_orders_status_chk
        """
    )
    op.execute(
        """
        ALTER TABLE execution_orders
        ADD CONSTRAINT execution_orders_status_chk CHECK (
            status IN (
                'guard_rejected',
                'submit_pending',
                'submitted',
                'status_checked',
                'cancelled',
                'adapter_error',
                'unknown',
                'reconciled'
            )
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_order_events (
            event_id UUID PRIMARY KEY,
            order_id UUID NOT NULL REFERENCES execution_orders(order_id),
            intent_id UUID NOT NULL,
            owner_user_id UUID NOT NULL,
            event_type TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            provider_order_id TEXT NULL,
            provider_event_id TEXT NULL,
            provider_event_id_key TEXT GENERATED ALWAYS AS
                (COALESCE(provider_event_id, event_type)) STORED,
            observed_at TIMESTAMPTZ NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            CONSTRAINT execution_order_events_type_chk CHECK (
                event_type IN (
                    'guard_rejected',
                    'submit_pending',
                    'submitted',
                    'status_checked',
                    'cancelled',
                    'adapter_error',
                    'private_stream_backfill',
                    'reconciled'
                )
            ),
            CONSTRAINT execution_order_events_reason_chk
                CHECK (char_length(trim(reason)) > 0),
            CONSTRAINT execution_order_events_metadata_chk
                CHECK (jsonb_typeof(metadata_json) = 'object'),
            CONSTRAINT ux_execution_order_events_dedupe
                UNIQUE (order_id, event_type, provider_event_id_key)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_order_events_owner_observed
            ON execution_order_events (owner_user_id, observed_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_fills (
            fill_id UUID PRIMARY KEY,
            order_id UUID NOT NULL REFERENCES execution_orders(order_id),
            intent_id UUID NOT NULL,
            owner_user_id UUID NOT NULL,
            provider_trade_id TEXT NOT NULL,
            price NUMERIC(28, 12) NOT NULL,
            quantity NUMERIC(28, 12) NOT NULL,
            fee_amount NUMERIC(28, 12) NOT NULL DEFAULT 0,
            fee_asset TEXT NOT NULL,
            filled_at TIMESTAMPTZ NOT NULL,
            liquidity TEXT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            CONSTRAINT execution_fills_quantity_chk CHECK (quantity > 0),
            CONSTRAINT execution_fills_price_chk CHECK (price > 0),
            CONSTRAINT execution_fills_fee_chk CHECK (fee_amount >= 0),
            CONSTRAINT execution_fills_trade_id_chk
                CHECK (char_length(trim(provider_trade_id)) > 0),
            CONSTRAINT execution_fills_metadata_chk
                CHECK (jsonb_typeof(metadata_json) = 'object'),
            CONSTRAINT ux_execution_fills_order_trade
                UNIQUE (order_id, provider_trade_id)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_fills_owner_filled
            ON execution_fills (owner_user_id, filled_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_funding_events (
            funding_event_id UUID PRIMARY KEY,
            order_id UUID NOT NULL REFERENCES execution_orders(order_id),
            intent_id UUID NOT NULL,
            owner_user_id UUID NOT NULL,
            provider_event_id TEXT NOT NULL,
            amount NUMERIC(28, 12) NOT NULL,
            asset TEXT NOT NULL,
            funding_at TIMESTAMPTZ NOT NULL,
            reason TEXT NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            CONSTRAINT execution_funding_events_provider_chk
                CHECK (char_length(trim(provider_event_id)) > 0),
            CONSTRAINT execution_funding_events_asset_chk
                CHECK (char_length(trim(asset)) > 0),
            CONSTRAINT execution_funding_events_reason_chk
                CHECK (char_length(trim(reason)) > 0),
            CONSTRAINT execution_funding_events_metadata_chk
                CHECK (jsonb_typeof(metadata_json) = 'object'),
            CONSTRAINT ux_execution_funding_events_order_provider
                UNIQUE (order_id, provider_event_id)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_funding_events_owner_funding
            ON execution_funding_events (owner_user_id, funding_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_reconciliation_runs (
            reconciliation_run_id UUID PRIMARY KEY,
            order_id UUID NOT NULL REFERENCES execution_orders(order_id),
            intent_id UUID NOT NULL,
            owner_user_id UUID NOT NULL,
            exchange_name TEXT NOT NULL,
            environment TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            local_status TEXT NOT NULL,
            provider_status TEXT NULL,
            fill_count INTEGER NOT NULL DEFAULT 0,
            funding_event_count INTEGER NOT NULL DEFAULT 0,
            started_at TIMESTAMPTZ NOT NULL,
            completed_at TIMESTAMPTZ NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            CONSTRAINT execution_reconciliation_runs_status_chk
                CHECK (status IN ('matched', 'mismatch', 'pending', 'failed')),
            CONSTRAINT execution_reconciliation_runs_counts_chk
                CHECK (fill_count >= 0 AND funding_event_count >= 0),
            CONSTRAINT execution_reconciliation_runs_reason_chk
                CHECK (char_length(trim(reason)) > 0),
            CONSTRAINT execution_reconciliation_runs_metadata_chk
                CHECK (jsonb_typeof(metadata_json) = 'object')
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_reconciliation_runs_order_completed
            ON execution_reconciliation_runs (order_id, completed_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_ledger_retention_policies (
            policy_name TEXT PRIMARY KEY,
            table_name TEXT NOT NULL,
            partition_key TEXT NOT NULL,
            retention_days INTEGER NOT NULL,
            archive_before_purge BOOLEAN NOT NULL,
            pitr_required BOOLEAN NOT NULL,
            checked_at TIMESTAMPTZ NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            CONSTRAINT execution_ledger_retention_policy_days_chk
                CHECK (retention_days >= 30),
            CONSTRAINT execution_ledger_retention_policy_status_chk
                CHECK (status IN ('configured', 'verified', 'blocked')),
            CONSTRAINT execution_ledger_retention_policy_reason_chk
                CHECK (char_length(trim(reason)) > 0)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_ledger_pitr_drills (
            drill_id UUID PRIMARY KEY,
            target_time TIMESTAMPTZ NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            verified_at TIMESTAMPTZ NOT NULL,
            row_counts_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            CONSTRAINT execution_ledger_pitr_drills_status_chk
                CHECK (status IN ('verified', 'blocked', 'failed')),
            CONSTRAINT execution_ledger_pitr_drills_reason_chk
                CHECK (char_length(trim(reason)) > 0),
            CONSTRAINT execution_ledger_pitr_drills_rows_chk
                CHECK (jsonb_typeof(row_counts_json) = 'object'),
            CONSTRAINT execution_ledger_pitr_drills_metadata_chk
                CHECK (jsonb_typeof(metadata_json) = 'object')
        )
        """
    )
    op.execute(
        """
        INSERT INTO execution_ledger_retention_policies
            (
                policy_name, table_name, partition_key, retention_days,
                archive_before_purge, pitr_required, checked_at, status, reason
            )
        VALUES
            (
                'execution_orders_money_ledger_v1',
                'execution_orders',
                'created_at',
                2555,
                TRUE,
                TRUE,
                now(),
                'configured',
                'money_ledger_retention_policy_configured'
            ),
            (
                'execution_order_events_money_ledger_v1',
                'execution_order_events',
                'observed_at',
                2555,
                TRUE,
                TRUE,
                now(),
                'configured',
                'money_ledger_retention_policy_configured'
            ),
            (
                'execution_fills_money_ledger_v1',
                'execution_fills',
                'filled_at',
                2555,
                TRUE,
                TRUE,
                now(),
                'configured',
                'money_ledger_retention_policy_configured'
            ),
            (
                'execution_funding_events_money_ledger_v1',
                'execution_funding_events',
                'funding_at',
                2555,
                TRUE,
                TRUE,
                now(),
                'configured',
                'money_ledger_retention_policy_configured'
            ),
            (
                'execution_reconciliation_runs_money_ledger_v1',
                'execution_reconciliation_runs',
                'completed_at',
                2555,
                TRUE,
                TRUE,
                now(),
                'configured',
                'money_ledger_retention_policy_configured'
            )
        ON CONFLICT (policy_name) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS execution_ledger_pitr_drills")
    op.execute("DROP TABLE IF EXISTS execution_ledger_retention_policies")
    op.execute("DROP INDEX IF EXISTS idx_execution_reconciliation_runs_order_completed")
    op.execute("DROP TABLE IF EXISTS execution_reconciliation_runs")
    op.execute("DROP INDEX IF EXISTS idx_execution_funding_events_owner_funding")
    op.execute("DROP TABLE IF EXISTS execution_funding_events")
    op.execute("DROP INDEX IF EXISTS idx_execution_fills_owner_filled")
    op.execute("DROP TABLE IF EXISTS execution_fills")
    op.execute("DROP INDEX IF EXISTS idx_execution_order_events_owner_observed")
    op.execute("DROP TABLE IF EXISTS execution_order_events")
    op.execute(
        """
        ALTER TABLE execution_orders
        DROP CONSTRAINT IF EXISTS execution_orders_status_chk
        """
    )
    op.execute(
        """
        ALTER TABLE execution_orders
        ADD CONSTRAINT execution_orders_status_chk CHECK (
            status IN (
                'guard_rejected',
                'submit_pending',
                'submitted',
                'status_checked',
                'cancelled',
                'adapter_error',
                'unknown'
            )
        )
        """
    )
