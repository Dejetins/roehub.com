"""Add exchange account projection and verify-only config guard snapshots."""

from __future__ import annotations

from alembic import op

revision = "20260531_0020"
down_revision = "20260531_0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS exchange_account_snapshots (
            account_snapshot_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            exchange_connection_id UUID NOT NULL,
            exchange_name TEXT NOT NULL,
            market_type TEXT NOT NULL,
            environment TEXT NOT NULL,
            account_mode TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            sync_status TEXT NOT NULL,
            sync_reason TEXT NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL,
            balance_count BIGINT NOT NULL DEFAULT 0,
            position_count BIGINT NOT NULL DEFAULT 0,
            open_order_count BIGINT NOT NULL DEFAULT 0,
            filter_count BIGINT NOT NULL DEFAULT 0,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            CONSTRAINT exchange_account_snapshots_source_hash_chk
                CHECK (source_hash ~ '^[0-9a-f]{64}$'),
            CONSTRAINT exchange_account_snapshots_status_chk
                CHECK (sync_status IN ('fresh', 'degraded')),
            CONSTRAINT exchange_account_snapshots_counts_chk
                CHECK (
                    balance_count >= 0
                    AND position_count >= 0
                    AND open_order_count >= 0
                    AND filter_count >= 0
                ),
            CONSTRAINT exchange_account_snapshots_metadata_chk
                CHECK (jsonb_typeof(metadata_json) = 'object')
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS exchange_balance_snapshots (
            balance_snapshot_id UUID PRIMARY KEY,
            account_snapshot_id UUID NOT NULL
                REFERENCES exchange_account_snapshots(account_snapshot_id)
                ON DELETE CASCADE,
            owner_user_id UUID NOT NULL,
            exchange_connection_id UUID NOT NULL,
            asset TEXT NOT NULL,
            free NUMERIC NOT NULL,
            locked NUMERIC NOT NULL,
            total NUMERIC NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT exchange_balance_snapshots_asset_chk
                CHECK (char_length(trim(asset)) > 0)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS exchange_position_snapshots (
            position_snapshot_id UUID PRIMARY KEY,
            account_snapshot_id UUID NOT NULL
                REFERENCES exchange_account_snapshots(account_snapshot_id)
                ON DELETE CASCADE,
            owner_user_id UUID NOT NULL,
            exchange_connection_id UUID NOT NULL,
            instrument_key TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity NUMERIC NOT NULL,
            entry_price NUMERIC NULL,
            leverage NUMERIC NULL,
            margin_mode TEXT NULL,
            position_mode TEXT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT exchange_position_snapshots_side_chk
                CHECK (side IN ('long', 'short', 'net')),
            CONSTRAINT exchange_position_snapshots_instrument_chk
                CHECK (char_length(trim(instrument_key)) > 0)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS exchange_open_order_snapshots (
            open_order_snapshot_id UUID PRIMARY KEY,
            account_snapshot_id UUID NOT NULL
                REFERENCES exchange_account_snapshots(account_snapshot_id)
                ON DELETE CASCADE,
            owner_user_id UUID NOT NULL,
            exchange_connection_id UUID NOT NULL,
            instrument_key TEXT NOT NULL,
            exchange_order_ref TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            quantity NUMERIC NOT NULL,
            price NUMERIC NULL,
            status TEXT NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT exchange_open_order_snapshots_side_chk
                CHECK (side IN ('buy', 'sell')),
            CONSTRAINT exchange_open_order_snapshots_ref_chk
                CHECK (char_length(trim(exchange_order_ref)) > 0)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS exchange_instrument_filter_snapshots (
            filter_snapshot_id UUID PRIMARY KEY,
            account_snapshot_id UUID NOT NULL
                REFERENCES exchange_account_snapshots(account_snapshot_id)
                ON DELETE CASCADE,
            owner_user_id UUID NOT NULL,
            exchange_connection_id UUID NOT NULL,
            instrument_key TEXT NOT NULL,
            tick_size NUMERIC NULL,
            step_size NUMERIC NULL,
            min_qty NUMERIC NULL,
            min_notional NUMERIC NULL,
            max_leverage NUMERIC NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT exchange_instrument_filter_snapshots_instrument_chk
                CHECK (char_length(trim(instrument_key)) > 0)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS exchange_account_config_guard_results (
            config_guard_result_id UUID PRIMARY KEY,
            account_snapshot_id UUID NULL
                REFERENCES exchange_account_snapshots(account_snapshot_id)
                ON DELETE SET NULL,
            owner_user_id UUID NOT NULL,
            exchange_connection_id UUID NOT NULL,
            instrument_key TEXT NOT NULL,
            market_type TEXT NOT NULL,
            status TEXT NOT NULL,
            reason_codes_json JSONB NOT NULL,
            requirement_json JSONB NOT NULL,
            checked_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT exchange_account_config_guard_results_status_chk
                CHECK (status IN ('verified', 'mismatch', 'degraded')),
            CONSTRAINT exchange_account_config_guard_results_reason_chk
                CHECK (jsonb_typeof(reason_codes_json) = 'array'),
            CONSTRAINT exchange_account_config_guard_results_requirement_chk
                CHECK (jsonb_typeof(requirement_json) = 'object')
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_exchange_account_snapshots_owner_connection_observed
            ON exchange_account_snapshots (
                owner_user_id, exchange_connection_id, observed_at DESC
            )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_exchange_config_guard_owner_connection_checked
            ON exchange_account_config_guard_results (
                owner_user_id, exchange_connection_id, instrument_key, market_type,
                checked_at DESC
            )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS exchange_account_config_guard_results")
    op.execute("DROP TABLE IF EXISTS exchange_instrument_filter_snapshots")
    op.execute("DROP TABLE IF EXISTS exchange_open_order_snapshots")
    op.execute("DROP TABLE IF EXISTS exchange_position_snapshots")
    op.execute("DROP TABLE IF EXISTS exchange_balance_snapshots")
    op.execute("DROP TABLE IF EXISTS exchange_account_snapshots")
