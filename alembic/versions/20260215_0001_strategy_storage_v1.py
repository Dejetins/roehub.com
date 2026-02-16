"""Create Strategy v1 tables for immutable specs, runs, and append-only events."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260215_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Apply Strategy v1 storage schema.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Migration is additive and safe on fresh environments and existing production baseline.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Creates strategy tables, constraints, and indexes.
    """
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_strategies (
            strategy_id UUID PRIMARY KEY,
            user_id UUID NOT NULL,
            name TEXT NOT NULL,
            instrument_id JSONB NOT NULL,
            instrument_key TEXT NOT NULL,
            market_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            indicators_json JSONB NOT NULL,
            spec_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            CONSTRAINT strategy_strategies_market_type_chk
                CHECK (market_type IN ('spot', 'futures')),
            CONSTRAINT strategy_strategies_timeframe_chk
                CHECK (timeframe IN ('1m', '5m', '15m', '1h', '4h', '1d')),
            CONSTRAINT strategy_strategies_spec_schema_version_chk
                CHECK ((spec_json ->> 'schema_version')::INTEGER = 1),
            CONSTRAINT strategy_strategies_spec_kind_chk
                CHECK ((spec_json ->> 'spec_kind') = 'roehub.strategy.v1'),
            CONSTRAINT strategy_strategies_instrument_id_shape_chk
                CHECK (
                    jsonb_typeof(instrument_id) = 'object'
                    AND instrument_id ? 'market_id'
                    AND instrument_id ? 'symbol'
                )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_strategies_user_created
            ON strategy_strategies (user_id, created_at, strategy_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_strategies_tags
            ON strategy_strategies (user_id, symbol, market_type, timeframe)
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_runs (
            run_id UUID PRIMARY KEY,
            user_id UUID NOT NULL,
            strategy_id UUID NOT NULL REFERENCES strategy_strategies (strategy_id),
            state TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            stopped_at TIMESTAMPTZ NULL,
            checkpoint_ts_open TIMESTAMPTZ NULL,
            last_error TEXT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT strategy_runs_state_chk
                CHECK (
                    state IN ('starting', 'warming_up', 'running', 'stopping', 'stopped', 'failed')
                )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_runs_strategy_started
            ON strategy_runs (strategy_id, started_at, run_id)
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS strategy_runs_one_active
            ON strategy_runs (strategy_id)
            WHERE state IN ('starting', 'warming_up', 'running', 'stopping')
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_events (
            event_id UUID PRIMARY KEY,
            user_id UUID NOT NULL,
            strategy_id UUID NOT NULL REFERENCES strategy_strategies (strategy_id),
            run_id UUID NULL REFERENCES strategy_runs (run_id),
            ts TIMESTAMPTZ NOT NULL,
            event_type TEXT NOT NULL,
            payload_json JSONB NOT NULL,
            CONSTRAINT strategy_events_event_type_nonempty_chk
                CHECK (char_length(trim(event_type)) > 0),
            CONSTRAINT strategy_events_payload_shape_chk
                CHECK (jsonb_typeof(payload_json) = 'object')
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_events_strategy_ts
            ON strategy_events (strategy_id, ts, event_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_events_run_ts
            ON strategy_events (run_id, ts, event_id)
            WHERE run_id IS NOT NULL
        """
    )


def downgrade() -> None:
    """
    Revert Strategy v1 storage schema.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Downgrade order drops dependent tables first.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Drops strategy events, runs, and strategies tables.
    """
    op.execute("DROP TABLE IF EXISTS strategy_events")
    op.execute("DROP TABLE IF EXISTS strategy_runs")
    op.execute("DROP TABLE IF EXISTS strategy_strategies")
