"""Add execution source events and intent ledger."""

from __future__ import annotations

from alembic import op

revision = "20260531_0023"
down_revision = "20260531_0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_source_events (
            source_event_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            source_type TEXT NOT NULL,
            source_event_ref TEXT NOT NULL,
            source_ref_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            strategy_signal_id UUID NULL,
            idempotency_key_hash TEXT NOT NULL,
            outcome TEXT NOT NULL,
            outcome_reason TEXT NOT NULL,
            intent_id UUID NULL,
            received_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT execution_source_events_source_type_chk CHECK (
                source_type IN (
                    'strategy_signal',
                    'manual_request',
                    'ml_agent_decision',
                    'ops_test'
                )
            ),
            CONSTRAINT execution_source_events_ref_chk
                CHECK (char_length(trim(source_event_ref)) > 0),
            CONSTRAINT execution_source_events_source_ref_json_chk
                CHECK (jsonb_typeof(source_ref_json) = 'object'),
            CONSTRAINT execution_source_events_strategy_signal_chk CHECK (
                (source_type = 'strategy_signal' AND strategy_signal_id IS NOT NULL)
                OR (source_type <> 'strategy_signal' AND strategy_signal_id IS NULL)
            ),
            CONSTRAINT execution_source_events_idempotency_hash_chk
                CHECK (idempotency_key_hash ~ '^[0-9a-f]{64}$'),
            CONSTRAINT execution_source_events_outcome_chk CHECK (
                outcome IN (
                    'recorded',
                    'intent_created',
                    'order_model_rejected',
                    'no_intent'
                )
            ),
            CONSTRAINT execution_source_events_reason_chk
                CHECK (char_length(trim(outcome_reason)) > 0),
            CONSTRAINT execution_source_events_idempotency_unique
                UNIQUE (owner_user_id, source_type, idempotency_key_hash)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_source_events_received
            ON execution_source_events (owner_user_id, received_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_source_events_strategy_signal
            ON execution_source_events (strategy_signal_id)
            WHERE strategy_signal_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_intents (
            intent_id UUID PRIMARY KEY,
            source_event_id UUID NOT NULL REFERENCES execution_source_events(source_event_id),
            owner_user_id UUID NOT NULL,
            source_type TEXT NOT NULL,
            strategy_signal_id UUID NULL,
            exchange_connection_id UUID NOT NULL,
            market_type TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            quantity NUMERIC(28, 12) NULL,
            quote_notional NUMERIC(28, 12) NULL,
            limit_price NUMERIC(28, 12) NULL,
            status TEXT NOT NULL,
            status_reason TEXT NOT NULL,
            risk_status TEXT NOT NULL,
            risk_reason TEXT NOT NULL,
            idempotency_key_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT execution_intents_source_type_chk CHECK (
                source_type IN (
                    'strategy_signal',
                    'manual_request',
                    'ml_agent_decision',
                    'ops_test'
                )
            ),
            CONSTRAINT execution_intents_strategy_signal_chk CHECK (
                (source_type = 'strategy_signal' AND strategy_signal_id IS NOT NULL)
                OR (source_type <> 'strategy_signal' AND strategy_signal_id IS NULL)
            ),
            CONSTRAINT execution_intents_market_type_chk CHECK (market_type IN ('spot', 'futures')),
            CONSTRAINT execution_intents_instrument_chk
                CHECK (char_length(trim(instrument_key)) > 0),
            CONSTRAINT execution_intents_side_chk CHECK (side IN ('buy', 'sell')),
            CONSTRAINT execution_intents_order_type_chk CHECK (order_type IN ('market', 'limit')),
            CONSTRAINT execution_intents_size_chk CHECK (
                (quantity IS NOT NULL AND quantity > 0)
                OR (quote_notional IS NOT NULL AND quote_notional > 0)
            ),
            CONSTRAINT execution_intents_limit_price_chk CHECK (
                (order_type = 'limit' AND limit_price IS NOT NULL AND limit_price > 0)
                OR (order_type = 'market' AND limit_price IS NULL)
            ),
            CONSTRAINT execution_intents_status_chk CHECK (status IN ('recorded')),
            CONSTRAINT execution_intents_risk_status_chk CHECK (risk_status IN ('not_evaluated')),
            CONSTRAINT execution_intents_reason_chk CHECK (
                char_length(trim(status_reason)) > 0
                AND char_length(trim(risk_reason)) > 0
            ),
            CONSTRAINT execution_intents_idempotency_hash_chk
                CHECK (idempotency_key_hash ~ '^[0-9a-f]{64}$'),
            CONSTRAINT execution_intents_idempotency_unique
                UNIQUE (owner_user_id, idempotency_key_hash)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_intents_source_event
            ON execution_intents (source_event_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_intents_created
            ON execution_intents (owner_user_id, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_intents_strategy_signal
            ON execution_intents (strategy_signal_id)
            WHERE strategy_signal_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_execution_intents_strategy_signal")
    op.execute("DROP INDEX IF EXISTS idx_execution_intents_created")
    op.execute("DROP INDEX IF EXISTS idx_execution_intents_source_event")
    op.execute("DROP TABLE IF EXISTS execution_intents")
    op.execute("DROP INDEX IF EXISTS idx_execution_source_events_strategy_signal")
    op.execute("DROP INDEX IF EXISTS idx_execution_source_events_received")
    op.execute("DROP TABLE IF EXISTS execution_source_events")
