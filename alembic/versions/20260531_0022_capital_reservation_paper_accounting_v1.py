"""Add capital reservation and paper accounting ledgers."""

from __future__ import annotations

from alembic import op

revision = "20260531_0022"
down_revision = "20260531_0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_capital_reservations (
            reservation_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            exchange_connection_id UUID NOT NULL,
            strategy_id UUID NOT NULL,
            live_profile_id UUID NULL,
            strategy_run_id UUID NOT NULL,
            asset TEXT NOT NULL,
            requested_amount NUMERIC(28, 12) NOT NULL,
            reserved_amount NUMERIC(28, 12) NOT NULL,
            state TEXT NOT NULL,
            source_account_snapshot_id UUID NULL,
            acquired_at TIMESTAMPTZ NOT NULL,
            released_at TIMESTAMPTZ NULL,
            reason TEXT NOT NULL,
            fee_model TEXT NOT NULL,
            funding_model TEXT NOT NULL,
            pnl_complete BOOLEAN NOT NULL DEFAULT FALSE,
            CONSTRAINT strategy_capital_reservations_asset_chk
                CHECK (char_length(trim(asset)) > 0),
            CONSTRAINT strategy_capital_reservations_amount_chk
                CHECK (requested_amount >= 0 AND reserved_amount >= 0),
            CONSTRAINT strategy_capital_reservations_state_chk
                CHECK (state IN ('reserved', 'released', 'rejected', 'stale_requires_repair')),
            CONSTRAINT strategy_capital_reservations_reason_chk
                CHECK (char_length(trim(reason)) > 0),
            CONSTRAINT strategy_capital_reservations_released_at_chk
                CHECK (
                    (state IN ('released', 'rejected') AND released_at IS NOT NULL)
                    OR (state NOT IN ('released', 'rejected'))
                )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_capital_reservations_run
            ON strategy_capital_reservations (owner_user_id, strategy_run_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_capital_reservations_active
            ON strategy_capital_reservations (owner_user_id, exchange_connection_id, asset)
            WHERE state = 'reserved'
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_orders (
            paper_order_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            strategy_id UUID NOT NULL,
            strategy_run_id UUID NOT NULL,
            reservation_id UUID NOT NULL,
            source_signal_id UUID NOT NULL,
            instrument_key TEXT NOT NULL,
            market_type TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            quantity NUMERIC(28, 12) NOT NULL,
            quote_notional NUMERIC(28, 12) NOT NULL,
            reference_price NUMERIC(28, 12) NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT paper_orders_signal_unique UNIQUE (source_signal_id),
            CONSTRAINT paper_orders_market_type_chk CHECK (market_type IN ('spot', 'futures')),
            CONSTRAINT paper_orders_side_chk CHECK (side IN ('buy', 'sell')),
            CONSTRAINT paper_orders_type_chk CHECK (order_type IN ('market')),
            CONSTRAINT paper_orders_status_chk CHECK (status IN ('filled', 'rejected')),
            CONSTRAINT paper_orders_quantity_chk CHECK (quantity >= 0 AND quote_notional >= 0)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_paper_orders_run
            ON paper_orders (owner_user_id, strategy_run_id, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_fills (
            paper_fill_id UUID PRIMARY KEY,
            paper_order_id UUID NOT NULL,
            owner_user_id UUID NOT NULL,
            strategy_id UUID NOT NULL,
            strategy_run_id UUID NOT NULL,
            instrument_key TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity NUMERIC(28, 12) NOT NULL,
            fill_price NUMERIC(28, 12) NOT NULL,
            quote_notional NUMERIC(28, 12) NOT NULL,
            fee_amount NUMERIC(28, 12) NOT NULL,
            fee_asset TEXT NOT NULL,
            funding_amount NUMERIC(28, 12) NOT NULL,
            funding_asset TEXT NOT NULL,
            filled_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT paper_fills_order_unique UNIQUE (paper_order_id),
            CONSTRAINT paper_fills_side_chk CHECK (side IN ('buy', 'sell')),
            CONSTRAINT paper_fills_amount_chk
                CHECK (quantity >= 0 AND quote_notional >= 0 AND fee_amount >= 0)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_paper_accounting (
            accounting_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            strategy_id UUID NOT NULL,
            strategy_run_id UUID NOT NULL,
            reservation_id UUID NOT NULL,
            paper_fill_id UUID NOT NULL,
            instrument_key TEXT NOT NULL,
            market_type TEXT NOT NULL,
            position_quantity NUMERIC(28, 12) NOT NULL,
            average_entry_price NUMERIC(28, 12) NULL,
            reserved_budget NUMERIC(28, 12) NOT NULL,
            cash_balance NUMERIC(28, 12) NOT NULL,
            equity NUMERIC(28, 12) NOT NULL,
            realized_pnl NUMERIC(28, 12) NOT NULL,
            unrealized_pnl NUMERIC(28, 12) NOT NULL,
            fee_total NUMERIC(28, 12) NOT NULL,
            funding_total NUMERIC(28, 12) NOT NULL,
            fee_model TEXT NOT NULL,
            funding_model TEXT NOT NULL,
            pnl_complete BOOLEAN NOT NULL,
            completeness_reason TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT strategy_paper_accounting_fill_unique UNIQUE (paper_fill_id),
            CONSTRAINT strategy_paper_accounting_market_type_chk
                CHECK (market_type IN ('spot', 'futures')),
            CONSTRAINT strategy_paper_accounting_reason_chk
                CHECK (char_length(trim(completeness_reason)) > 0)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_paper_accounting_latest
            ON strategy_paper_accounting (owner_user_id, strategy_id, created_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_strategy_paper_accounting_latest")
    op.execute("DROP TABLE IF EXISTS strategy_paper_accounting")
    op.execute("DROP TABLE IF EXISTS paper_fills")
    op.execute("DROP INDEX IF EXISTS idx_paper_orders_run")
    op.execute("DROP TABLE IF EXISTS paper_orders")
    op.execute("DROP INDEX IF EXISTS idx_strategy_capital_reservations_active")
    op.execute("DROP INDEX IF EXISTS idx_strategy_capital_reservations_run")
    op.execute("DROP TABLE IF EXISTS strategy_capital_reservations")
