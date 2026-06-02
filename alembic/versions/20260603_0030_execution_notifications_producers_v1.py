"""Add execution producer notifications outbox."""

from __future__ import annotations

from alembic import op

revision = "20260603_0030"
down_revision = "20260602_0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE execution_source_events
        DROP CONSTRAINT IF EXISTS execution_source_events_outcome_chk
        """
    )
    op.execute(
        """
        ALTER TABLE execution_source_events
        ADD CONSTRAINT execution_source_events_outcome_chk CHECK (
            outcome IN (
                'recorded',
                'intent_created',
                'order_model_rejected',
                'no_intent',
                'risk_rejected',
                'submitted',
                'filled',
                'cancelled',
                'failed',
                'reconciliation_required',
                'handoff_failed'
            )
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_notification_outbox (
            notification_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            source_type TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            reason TEXT NOT NULL,
            source_event_id UUID NULL REFERENCES execution_source_events(source_event_id),
            source_event_key UUID GENERATED ALWAYS AS
                (COALESCE(source_event_id, '00000000-0000-0000-0000-000000000000'::uuid))
                STORED,
            intent_id UUID NULL REFERENCES execution_intents(intent_id),
            intent_key UUID GENERATED ALWAYS AS
                (COALESCE(intent_id, '00000000-0000-0000-0000-000000000000'::uuid))
                STORED,
            order_id UUID NULL REFERENCES execution_orders(order_id),
            order_key UUID GENERATED ALWAYS AS
                (COALESCE(order_id, '00000000-0000-0000-0000-000000000000'::uuid))
                STORED,
            strategy_signal_id UUID NULL,
            labels_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMPTZ NOT NULL,
            sent_at TIMESTAMPTZ NULL,
            CONSTRAINT execution_notification_source_type_chk CHECK (
                source_type IN (
                    'strategy_signal',
                    'manual_request',
                    'ml_agent_decision',
                    'ops_test'
                )
            ),
            CONSTRAINT execution_notification_event_type_chk CHECK (
                event_type IN (
                    'producer_rejected',
                    'producer_fill',
                    'producer_unknown',
                    'producer_kill_switch',
                    'producer_terminal'
                )
            ),
            CONSTRAINT execution_notification_severity_chk CHECK (
                severity IN ('info', 'warning', 'critical')
            ),
            CONSTRAINT execution_notification_status_chk CHECK (
                status IN ('pending', 'sent', 'failed')
            ),
            CONSTRAINT execution_notification_reason_chk
                CHECK (char_length(trim(reason)) > 0),
            CONSTRAINT execution_notification_labels_chk
                CHECK (jsonb_typeof(labels_json) = 'object'),
            CONSTRAINT execution_notification_outbox_dedupe
                UNIQUE (
                    owner_user_id,
                    event_type,
                    source_event_key,
                    intent_key,
                    order_key,
                    reason
                )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_notification_owner_created
            ON execution_notification_outbox (owner_user_id, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_notification_source_event
            ON execution_notification_outbox (source_event_id)
            WHERE source_event_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_notification_intent
            ON execution_notification_outbox (intent_id)
            WHERE intent_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_execution_notification_intent")
    op.execute("DROP INDEX IF EXISTS idx_execution_notification_source_event")
    op.execute("DROP INDEX IF EXISTS idx_execution_notification_owner_created")
    op.execute("DROP TABLE IF EXISTS execution_notification_outbox")
    op.execute(
        """
        ALTER TABLE execution_source_events
        DROP CONSTRAINT IF EXISTS execution_source_events_outcome_chk
        """
    )
    op.execute(
        """
        ALTER TABLE execution_source_events
        ADD CONSTRAINT execution_source_events_outcome_chk CHECK (
            outcome IN (
                'recorded',
                'intent_created',
                'order_model_rejected',
                'no_intent'
            )
        )
        """
    )
