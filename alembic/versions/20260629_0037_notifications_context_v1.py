"""Add provider-neutral notifications context tables."""

from __future__ import annotations

from alembic import op

revision = "20260629_0037"
down_revision = "20260618_0036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS notification_events (
            event_id UUID PRIMARY KEY,
            owner_user_id UUID NULL,
            recipient_kind TEXT NOT NULL,
            source_context TEXT NOT NULL,
            source_event_type TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            scope_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            dedupe_key TEXT NOT NULL,
            occurred_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT notification_events_recipient_kind_chk
                CHECK (recipient_kind IN ('user', 'admin', 'both')),
            CONSTRAINT notification_events_owner_required_chk
                CHECK (
                    (recipient_kind = 'admin' AND owner_user_id IS NULL)
                    OR (recipient_kind IN ('user', 'both') AND owner_user_id IS NOT NULL)
                ),
            CONSTRAINT notification_events_source_context_chk
                CHECK (
                    source_context IN (
                        'strategy',
                        'live_execution',
                        'rl_trading',
                        'market_data',
                        'ops',
                        'identity',
                        'notifications'
                    )
                ),
            CONSTRAINT notification_events_category_chk
                CHECK (
                    category IN (
                        'strategy_run_failed',
                        'strategy_signal',
                        'trade_fill',
                        'execution_rejected',
                        'execution_terminal',
                        'execution_unknown',
                        'kill_switch',
                        'portfolio_report',
                        'stats_response',
                        'system_alert',
                        'admin_critical',
                        'admin_alert',
                        'admin_report'
                    )
                ),
            CONSTRAINT notification_events_severity_chk
                CHECK (severity IN ('info', 'warning', 'critical')),
            CONSTRAINT notification_events_scope_json_chk
                CHECK (jsonb_typeof(scope_json) = 'object'),
            CONSTRAINT notification_events_payload_json_chk
                CHECK (jsonb_typeof(payload_json) = 'object'),
            CONSTRAINT notification_events_dedupe_key_chk
                CHECK (char_length(trim(dedupe_key)) BETWEEN 16 AND 240),
            CONSTRAINT notification_events_dedupe_unique
                UNIQUE (dedupe_key)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_events_owner_created
            ON notification_events (owner_user_id, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_events_category_created
            ON notification_events (category, created_at DESC)
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS notification_routes (
            route_id UUID PRIMARY KEY,
            recipient_kind TEXT NOT NULL,
            owner_user_id UUID NULL,
            channel_key TEXT NOT NULL,
            provider_key TEXT NOT NULL,
            mode TEXT NOT NULL,
            category_filter TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
            scope_filter_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            schedule_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            recipient_address_ref TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT notification_routes_recipient_kind_chk
                CHECK (recipient_kind IN ('user', 'admin')),
            CONSTRAINT notification_routes_owner_required_chk
                CHECK (
                    (recipient_kind = 'admin' AND owner_user_id IS NULL)
                    OR (recipient_kind = 'user' AND owner_user_id IS NOT NULL)
                ),
            CONSTRAINT notification_routes_channel_key_chk
                CHECK (channel_key IN ('telegram', 'email', 'webhook', 'push', 'in_app')),
            CONSTRAINT notification_routes_provider_key_chk
                CHECK (
                    provider_key IN (
                        'telegram_bot_api',
                        'log_only',
                        'fake',
                        'email',
                        'webhook',
                        'push'
                    )
                ),
            CONSTRAINT notification_routes_mode_chk
                CHECK (mode IN ('off', 'critical_only', 'trades', 'signals', 'reports', 'all')),
            CONSTRAINT notification_routes_status_chk
                CHECK (status IN ('active', 'paused', 'requires_rebind', 'disabled')),
            CONSTRAINT notification_routes_scope_filter_json_chk
                CHECK (jsonb_typeof(scope_filter_json) = 'object'),
            CONSTRAINT notification_routes_schedule_json_chk
                CHECK (jsonb_typeof(schedule_json) = 'object'),
            CONSTRAINT notification_routes_address_ref_not_raw_chk
                CHECK (
                    char_length(trim(recipient_address_ref)) BETWEEN 8 AND 180
                    AND recipient_address_ref !~* '(token|secret|password|cookie|authorization)'
                ),
            CONSTRAINT notification_routes_unique_active_ref
                UNIQUE (
                    recipient_kind,
                    owner_user_id,
                    channel_key,
                    provider_key,
                    recipient_address_ref
                )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_routes_owner_channel
            ON notification_routes (owner_user_id, channel_key, status)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_routes_admin
            ON notification_routes (provider_key, status)
            WHERE recipient_kind = 'admin'
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS notification_report_runs (
            report_run_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            report_type TEXT NOT NULL,
            period_start TIMESTAMPTZ NOT NULL,
            period_end TIMESTAMPTZ NOT NULL,
            scope_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            quality_status TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            dedupe_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            rendered_at TIMESTAMPTZ NULL,
            finished_at TIMESTAMPTZ NULL,
            CONSTRAINT notification_report_runs_type_chk
                CHECK (report_type IN ('portfolio_weekly', 'portfolio_monthly', 'stats_on_demand')),
            CONSTRAINT notification_report_runs_period_chk
                CHECK (period_start < period_end),
            CONSTRAINT notification_report_runs_scope_json_chk
                CHECK (jsonb_typeof(scope_json) = 'object'),
            CONSTRAINT notification_report_runs_quality_status_chk
                CHECK (quality_status IN ('complete', 'partial', 'unavailable')),
            CONSTRAINT notification_report_runs_status_chk
                CHECK (status IN ('pending', 'rendered', 'sent', 'failed', 'suppressed')),
            CONSTRAINT notification_report_runs_dedupe_key_chk
                CHECK (char_length(trim(dedupe_key)) BETWEEN 16 AND 240),
            CONSTRAINT notification_report_runs_dedupe_unique
                UNIQUE (dedupe_key)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_report_runs_owner_period
            ON notification_report_runs (owner_user_id, report_type, period_start DESC)
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS notification_deliveries (
            delivery_id UUID PRIMARY KEY,
            event_id UUID NULL REFERENCES notification_events(event_id),
            report_run_id UUID NULL REFERENCES notification_report_runs(report_run_id),
            command_id UUID NULL,
            route_id UUID NOT NULL REFERENCES notification_routes(route_id),
            provider_key TEXT NOT NULL,
            channel_key TEXT NOT NULL,
            recipient_address_ref TEXT NOT NULL,
            template_key TEXT NOT NULL,
            rendered_payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TIMESTAMPTZ NULL,
            lease_until TIMESTAMPTZ NULL,
            last_error_code TEXT NULL,
            provider_message_id TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            sent_at TIMESTAMPTZ NULL,
            CONSTRAINT notification_deliveries_one_source_chk
                CHECK (num_nonnulls(event_id, report_run_id, command_id) = 1),
            CONSTRAINT notification_deliveries_provider_key_chk
                CHECK (
                    provider_key IN (
                        'telegram_bot_api',
                        'log_only',
                        'fake',
                        'email',
                        'webhook',
                        'push'
                    )
                ),
            CONSTRAINT notification_deliveries_channel_key_chk
                CHECK (channel_key IN ('telegram', 'email', 'webhook', 'push', 'in_app')),
            CONSTRAINT notification_deliveries_payload_json_chk
                CHECK (jsonb_typeof(rendered_payload_json) = 'object'),
            CONSTRAINT notification_deliveries_status_chk
                CHECK (
                    status IN (
                        'pending',
                        'claimed',
                        'sent',
                        'failed',
                        'retry',
                        'dead_letter',
                        'suppressed',
                        'unknown'
                    )
                ),
            CONSTRAINT notification_deliveries_attempt_count_chk
                CHECK (attempt_count >= 0),
            CONSTRAINT notification_deliveries_address_ref_not_raw_chk
                CHECK (
                    char_length(trim(recipient_address_ref)) BETWEEN 8 AND 180
                    AND recipient_address_ref !~* '(token|secret|password|cookie|authorization)'
                )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_deliveries_pending
            ON notification_deliveries (status, next_attempt_at, created_at)
            WHERE status IN ('pending', 'retry', 'unknown')
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_deliveries_route_created
            ON notification_deliveries (route_id, created_at DESC)
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS notification_delivery_attempts (
            attempt_id UUID PRIMARY KEY,
            delivery_id UUID NOT NULL REFERENCES notification_deliveries(delivery_id),
            provider_key TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            finished_at TIMESTAMPTZ NULL,
            status TEXT NOT NULL,
            http_status INTEGER NULL,
            error_code TEXT NULL,
            retry_after_seconds INTEGER NULL,
            redacted_request_hash TEXT NULL,
            redacted_response_hash TEXT NULL,
            CONSTRAINT notification_delivery_attempts_status_chk
                CHECK (
                    status IN (
                        'pending',
                        'claimed',
                        'sent',
                        'failed',
                        'retry',
                        'dead_letter',
                        'suppressed',
                        'unknown'
                    )
                ),
            CONSTRAINT notification_delivery_attempts_http_status_chk
                CHECK (http_status IS NULL OR http_status BETWEEN 100 AND 599),
            CONSTRAINT notification_delivery_attempts_retry_after_chk
                CHECK (retry_after_seconds IS NULL OR retry_after_seconds >= 0),
            CONSTRAINT notification_delivery_attempts_request_hash_chk
                CHECK (
                    redacted_request_hash IS NULL
                    OR redacted_request_hash ~ '^[a-f0-9]{64}$'
                ),
            CONSTRAINT notification_delivery_attempts_response_hash_chk
                CHECK (
                    redacted_response_hash IS NULL
                    OR redacted_response_hash ~ '^[a-f0-9]{64}$'
                )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_delivery_attempts_delivery
            ON notification_delivery_attempts (delivery_id, started_at DESC)
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS notification_telegram_updates (
            telegram_update_id BIGINT PRIMARY KEY,
            received_at TIMESTAMPTZ NOT NULL,
            chat_id_ref TEXT NOT NULL,
            owner_user_id UUID NULL,
            command_name TEXT NULL,
            command_args_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            status TEXT NOT NULL DEFAULT 'pending',
            idempotency_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            handled_at TIMESTAMPTZ NULL,
            CONSTRAINT notification_telegram_updates_chat_ref_not_raw_chk
                CHECK (
                    char_length(trim(chat_id_ref)) BETWEEN 8 AND 180
                    AND chat_id_ref !~* '(token|secret|password|cookie|authorization)'
                ),
            CONSTRAINT notification_telegram_updates_command_args_json_chk
                CHECK (jsonb_typeof(command_args_json) = 'object'),
            CONSTRAINT notification_telegram_updates_status_chk
                CHECK (status IN ('pending', 'handled', 'ignored', 'failed', 'dead_letter')),
            CONSTRAINT notification_telegram_updates_idempotency_key_chk
                CHECK (char_length(trim(idempotency_key)) BETWEEN 16 AND 240),
            CONSTRAINT notification_telegram_updates_idempotency_unique
                UNIQUE (idempotency_key)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_telegram_updates_status
            ON notification_telegram_updates (status, received_at)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_notification_telegram_updates_status")
    op.execute("DROP TABLE IF EXISTS notification_telegram_updates")
    op.execute("DROP INDEX IF EXISTS idx_notification_delivery_attempts_delivery")
    op.execute("DROP TABLE IF EXISTS notification_delivery_attempts")
    op.execute("DROP INDEX IF EXISTS idx_notification_deliveries_route_created")
    op.execute("DROP INDEX IF EXISTS idx_notification_deliveries_pending")
    op.execute("DROP TABLE IF EXISTS notification_deliveries")
    op.execute("DROP INDEX IF EXISTS idx_notification_report_runs_owner_period")
    op.execute("DROP TABLE IF EXISTS notification_report_runs")
    op.execute("DROP INDEX IF EXISTS idx_notification_routes_admin")
    op.execute("DROP INDEX IF EXISTS idx_notification_routes_owner_channel")
    op.execute("DROP TABLE IF EXISTS notification_routes")
    op.execute("DROP INDEX IF EXISTS idx_notification_events_category_created")
    op.execute("DROP INDEX IF EXISTS idx_notification_events_owner_created")
    op.execute("DROP TABLE IF EXISTS notification_events")
