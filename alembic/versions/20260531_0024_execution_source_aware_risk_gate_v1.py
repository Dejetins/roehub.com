"""Add execution source-aware risk gate state."""

from __future__ import annotations

from alembic import op

revision = "20260531_0024"
down_revision = "20260531_0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE execution_intents DROP CONSTRAINT IF EXISTS execution_intents_status_chk"
    )
    op.execute(
        """
        ALTER TABLE execution_intents
            ADD CONSTRAINT execution_intents_status_chk
            CHECK (status IN ('recorded', 'accepted', 'rejected'))
        """
    )
    op.execute(
        "ALTER TABLE execution_intents DROP CONSTRAINT IF EXISTS execution_intents_risk_status_chk"
    )
    op.execute(
        """
        ALTER TABLE execution_intents
            ADD CONSTRAINT execution_intents_risk_status_chk
            CHECK (risk_status IN ('not_evaluated', 'accepted', 'rejected'))
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_risk_audit_events (
            event_id UUID PRIMARY KEY,
            intent_id UUID NOT NULL REFERENCES execution_intents(intent_id),
            source_event_id UUID NOT NULL REFERENCES execution_source_events(source_event_id),
            owner_user_id UUID NOT NULL,
            source_type TEXT NOT NULL,
            event_type TEXT NOT NULL,
            risk_status TEXT NOT NULL,
            risk_reason TEXT NOT NULL,
            check_name TEXT NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT execution_risk_audit_events_source_type_chk CHECK (
                source_type IN (
                    'strategy_signal',
                    'manual_request',
                    'ml_agent_decision',
                    'ops_test'
                )
            ),
            CONSTRAINT execution_risk_audit_events_event_type_chk CHECK (
                event_type IN ('risk_gate_accepted', 'risk_gate_rejected')
            ),
            CONSTRAINT execution_risk_audit_events_risk_status_chk CHECK (
                risk_status IN ('accepted', 'rejected')
            ),
            CONSTRAINT execution_risk_audit_events_reason_chk CHECK (
                char_length(trim(risk_reason)) > 0
                AND char_length(trim(check_name)) > 0
                AND jsonb_typeof(metadata_json) = 'object'
            )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_risk_audit_events_intent
            ON execution_risk_audit_events (intent_id, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_risk_audit_events_owner_created
            ON execution_risk_audit_events (owner_user_id, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_risk_audit_events_reason
            ON execution_risk_audit_events (risk_status, risk_reason)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_execution_risk_audit_events_reason")
    op.execute("DROP INDEX IF EXISTS idx_execution_risk_audit_events_owner_created")
    op.execute("DROP INDEX IF EXISTS idx_execution_risk_audit_events_intent")
    op.execute("DROP TABLE IF EXISTS execution_risk_audit_events")
    op.execute(
        "ALTER TABLE execution_intents DROP CONSTRAINT IF EXISTS execution_intents_status_chk"
    )
    op.execute(
        """
        ALTER TABLE execution_intents
            ADD CONSTRAINT execution_intents_status_chk
            CHECK (status IN ('recorded'))
        """
    )
    op.execute(
        "ALTER TABLE execution_intents DROP CONSTRAINT IF EXISTS execution_intents_risk_status_chk"
    )
    op.execute(
        """
        ALTER TABLE execution_intents
            ADD CONSTRAINT execution_intents_risk_status_chk
            CHECK (risk_status IN ('not_evaluated'))
        """
    )
