BEGIN;

CREATE TABLE market_data_instrument_selections (
    organization_id UUID NOT NULL
        REFERENCES identity_organizations(organization_id) ON DELETE CASCADE,
    market_id SMALLINT NOT NULL CHECK (market_id > 0),
    symbol TEXT NOT NULL CHECK (symbol ~ '^[A-Z0-9][A-Z0-9._-]{1,63}$'),
    selected_by_user_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (organization_id, market_id, symbol),
    CONSTRAINT market_data_instrument_selections_actor_fk
        FOREIGN KEY (organization_id, selected_by_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    CONSTRAINT market_data_instrument_selections_time_chk CHECK (updated_at >= created_at)
);

CREATE INDEX idx_market_data_instrument_selections_market
    ON market_data_instrument_selections (market_id, symbol, organization_id);

CREATE TABLE market_data_catalog_refresh_state (
    market_id SMALLINT PRIMARY KEY CHECK (market_id > 0),
    state TEXT NOT NULL CHECK (state IN ('fresh', 'stale', 'failed')),
    refreshed_at TIMESTAMPTZ NULL,
    last_error_code TEXT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT market_data_catalog_refresh_state_freshness_chk CHECK (
        (state = 'fresh' AND refreshed_at IS NOT NULL AND last_error_code IS NULL)
        OR (state = 'stale' AND refreshed_at IS NOT NULL)
        OR (state = 'failed' AND last_error_code IS NOT NULL)
    ),
    CONSTRAINT market_data_catalog_refresh_state_error_chk CHECK (
        last_error_code IS NULL OR last_error_code ~ '^[a-z0-9][a-z0-9._-]{0,127}$'
    )
);

CREATE TABLE market_data_instrument_history_bounds (
    market_id SMALLINT NOT NULL CHECK (market_id > 0),
    symbol TEXT NOT NULL CHECK (symbol ~ '^[A-Z0-9][A-Z0-9._-]{1,63}$'),
    expected_start_at TIMESTAMPTZ NOT NULL,
    confirmed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (market_id, symbol),
    CONSTRAINT market_data_instrument_history_bounds_time_chk
        CHECK (confirmed_at >= expected_start_at)
);

CREATE TABLE market_data_instrument_selection_audit_events (
    event_id UUID PRIMARY KEY,
    organization_id UUID NOT NULL
        REFERENCES identity_organizations(organization_id) ON DELETE RESTRICT,
    actor_user_id UUID NOT NULL,
    market_id SMALLINT NOT NULL CHECK (market_id > 0),
    symbol TEXT NOT NULL CHECK (symbol ~ '^[A-Z0-9][A-Z0-9._-]{1,63}$'),
    action TEXT NOT NULL CHECK (action IN ('selected', 'unselected')),
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT market_data_selection_audit_actor_fk
        FOREIGN KEY (organization_id, actor_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT
);

CREATE INDEX idx_market_data_selection_audit_org_created
    ON market_data_instrument_selection_audit_events (
        organization_id, created_at DESC, event_id DESC
    );

COMMIT;
