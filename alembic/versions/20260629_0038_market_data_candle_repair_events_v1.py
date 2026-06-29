"""Add Market Data live-tail candle repair audit events."""

from __future__ import annotations

from alembic import op

revision = "20260629_0038"
down_revision = "20260629_0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS market_data_candle_repair_events (
            event_id UUID PRIMARY KEY,
            correlation_id TEXT NOT NULL,
            market_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            range_start_ts_open TIMESTAMPTZ NOT NULL,
            range_end_ts_open TIMESTAMPTZ NOT NULL,
            status TEXT NOT NULL,
            sources_attempted_json JSONB NOT NULL,
            restored_ts_opens_json JSONB NOT NULL,
            missing_ts_opens_json JSONB NOT NULL,
            error_code TEXT NULL,
            error_summary TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT market_data_candle_repair_events_correlation_chk
                CHECK (char_length(trim(correlation_id)) BETWEEN 1 AND 160),
            CONSTRAINT market_data_candle_repair_events_market_chk
                CHECK (market_id > 0 AND market_id <= 65535),
            CONSTRAINT market_data_candle_repair_events_symbol_chk
                CHECK (char_length(trim(symbol)) > 0),
            CONSTRAINT market_data_candle_repair_events_instrument_key_chk
                CHECK (char_length(trim(instrument_key)) > 0),
            CONSTRAINT market_data_candle_repair_events_range_chk
                CHECK (range_start_ts_open < range_end_ts_open),
            CONSTRAINT market_data_candle_repair_events_status_chk
                CHECK (
                    status IN (
                        'attempted',
                        'succeeded',
                        'miss',
                        'failed',
                        'circuit_open',
                        'rate_limited'
                    )
                ),
            CONSTRAINT market_data_candle_repair_events_sources_shape_chk
                CHECK (jsonb_typeof(sources_attempted_json) = 'array'),
            CONSTRAINT market_data_candle_repair_events_restored_shape_chk
                CHECK (jsonb_typeof(restored_ts_opens_json) = 'array'),
            CONSTRAINT market_data_candle_repair_events_missing_shape_chk
                CHECK (jsonb_typeof(missing_ts_opens_json) = 'array'),
            CONSTRAINT market_data_candle_repair_events_error_code_chk
                CHECK (
                    error_code IS NULL
                    OR error_code ~ '^[a-z0-9][a-z0-9_:-]{0,95}$'
                ),
            CONSTRAINT market_data_candle_repair_events_error_summary_chk
                CHECK (
                    error_summary IS NULL
                    OR (
                        char_length(trim(error_summary)) BETWEEN 1 AND 240
                        AND error_summary !~* (
                            '(api[_-]?key|authorization|bearer|cookie|dsn|'
                            || 'password|secret|token)'
                        )
                    )
                )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_market_data_candle_repair_events_correlation
            ON market_data_candle_repair_events (correlation_id, created_at ASC, event_id ASC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_market_data_candle_repair_events_instrument_created
            ON market_data_candle_repair_events (instrument_key, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_market_data_candle_repair_events_status_created
            ON market_data_candle_repair_events (status, created_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS market_data_candle_repair_events")
