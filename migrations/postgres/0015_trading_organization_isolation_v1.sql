BEGIN;

-- Stage 10 is greenfield-only. Trading ownership must never be inferred from
-- legacy users, account identifiers, strategy identifiers, or idempotency keys.
DO $$
DECLARE
    populated_table TEXT;
    has_rows BOOLEAN;
BEGIN
    FOREACH populated_table IN ARRAY ARRAY[
        'exchange_credential_versions',
        'strategy_runs',
        'strategy_events',
        'strategy_live_profiles',
        'strategy_signals',
        'strategy_variant_compatibility_checks',
        'market_data_subscription_requirements',
        'strategy_variant_scenario_matrix_rows',
        'strategy_paper_scenario_coverage_results',
        'rl_risk_sizing_policies',
        'rl_risk_sizing_policy_audit_events',
        'rl_live_ticker_entitlement_overrides',
        'rl_live_ticker_activations',
        'strategy_position_ownership',
        'exchange_balance_snapshots',
        'exchange_open_order_snapshots',
        'exchange_instrument_filter_snapshots',
        'exchange_account_config_guard_results',
        'strategy_capital_reservations',
        'paper_orders',
        'paper_fills',
        'strategy_paper_accounting',
        'execution_source_events',
        'execution_intents',
        'execution_risk_audit_events',
        'execution_notification_outbox',
        'exchange_execution_request_observations',
        'execution_orders',
        'exchange_private_stream_sessions',
        'execution_order_events',
        'execution_fills',
        'execution_funding_events',
        'execution_reconciliation_runs'
    ]
    LOOP
        IF to_regclass('public.' || populated_table) IS NULL THEN
            RAISE EXCEPTION 'required greenfield trading table is missing: %',
                populated_table;
        END IF;
        EXECUTE format('SELECT EXISTS (SELECT 1 FROM %I)', populated_table)
            INTO STRICT has_rows;
        IF has_rows THEN
            RAISE EXCEPTION
                'trading organization schema requires empty greenfield table: %',
                populated_table;
        END IF;
    END LOOP;
END
$$;

ALTER TABLE strategy_strategies
    ADD CONSTRAINT strategy_strategies_org_user_id_unique
        UNIQUE (organization_id, user_id, strategy_id);

ALTER TABLE exchange_connections
    ADD CONSTRAINT exchange_connections_org_owner_id_unique
        UNIQUE (organization_id, owner_user_id, connection_id);

ALTER TABLE exchange_credential_versions
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT exchange_credential_versions_org_connection_fk
        FOREIGN KEY (organization_id, connection_id)
        REFERENCES exchange_connections(organization_id, connection_id) ON DELETE CASCADE,
    ADD CONSTRAINT exchange_credential_versions_org_id_unique
        UNIQUE (organization_id, credential_version_id);

ALTER TABLE exchange_connections
    DROP CONSTRAINT exchange_connections_active_credential_version_id_fk,
    ADD CONSTRAINT exchange_connections_org_active_credential_fk
        FOREIGN KEY (organization_id, active_credential_version_id)
        REFERENCES exchange_credential_versions(organization_id, credential_version_id)
        DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE strategy_runs
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT strategy_runs_org_member_fk
        FOREIGN KEY (organization_id, user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_runs_org_strategy_fk
        FOREIGN KEY (organization_id, user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT strategy_runs_org_id_unique
        UNIQUE (organization_id, run_id),
    ADD CONSTRAINT strategy_runs_org_user_id_unique
        UNIQUE (organization_id, user_id, run_id);

DROP INDEX strategy_runs_one_active;
CREATE UNIQUE INDEX strategy_runs_one_active
    ON strategy_runs (organization_id, strategy_id)
    WHERE state IN ('starting', 'warming_up', 'running', 'stopping');

ALTER TABLE strategy_events
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT strategy_events_org_member_fk
        FOREIGN KEY (organization_id, user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_events_org_strategy_fk
        FOREIGN KEY (organization_id, user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT strategy_events_org_run_fk
        FOREIGN KEY (organization_id, user_id, run_id)
        REFERENCES strategy_runs(organization_id, user_id, run_id) ON DELETE CASCADE;

ALTER TABLE strategy_live_profiles
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT strategy_live_profiles_owner_strategy_unique,
    ADD CONSTRAINT strategy_live_profiles_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_live_profiles_org_strategy_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT strategy_live_profiles_org_connection_fk
        FOREIGN KEY (organization_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, connection_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_live_profiles_org_owner_strategy_unique
        UNIQUE (organization_id, owner_user_id, strategy_id),
    ADD CONSTRAINT strategy_live_profiles_org_id_unique
        UNIQUE (organization_id, profile_id),
    ADD CONSTRAINT strategy_live_profiles_org_owner_id_unique
        UNIQUE (organization_id, owner_user_id, profile_id);

ALTER TABLE strategy_signals
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT strategy_signals_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_signals_org_strategy_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT strategy_signals_org_run_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_run_id)
        REFERENCES strategy_runs(organization_id, user_id, run_id) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_signals_org_profile_fk
        FOREIGN KEY (organization_id, live_profile_id)
        REFERENCES strategy_live_profiles(organization_id, profile_id)
        ON DELETE SET NULL (live_profile_id),
    ADD CONSTRAINT strategy_signals_org_id_unique
        UNIQUE (organization_id, signal_id),
    ADD CONSTRAINT strategy_signals_org_owner_id_unique
        UNIQUE (organization_id, owner_user_id, signal_id);

ALTER TABLE strategy_variant_compatibility_checks
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT strategy_compatibility_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_compatibility_org_strategy_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT strategy_compatibility_org_job_fk
        FOREIGN KEY (organization_id, owner_user_id, source_job_id)
        REFERENCES backtest_jobs(organization_id, user_id, job_id)
        ON DELETE SET NULL (source_job_id),
    ADD CONSTRAINT strategy_compatibility_org_id_unique
        UNIQUE (organization_id, compatibility_check_id);

ALTER TABLE market_data_subscription_requirements
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT market_data_requirements_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT market_data_requirements_org_compatibility_fk
        FOREIGN KEY (organization_id, compatibility_check_id)
        REFERENCES strategy_variant_compatibility_checks(
            organization_id,
            compatibility_check_id
        ) ON DELETE CASCADE,
    ADD CONSTRAINT market_data_requirements_org_strategy_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT market_data_requirements_org_job_fk
        FOREIGN KEY (organization_id, owner_user_id, source_job_id)
        REFERENCES backtest_jobs(organization_id, user_id, job_id)
        ON DELETE SET NULL (source_job_id),
    ADD CONSTRAINT market_data_requirements_org_id_unique
        UNIQUE (organization_id, market_data_requirement_id);

ALTER TABLE strategy_variant_scenario_matrix_rows
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT strategy_variant_scenario_matrix_unique_row,
    ADD CONSTRAINT strategy_scenario_matrix_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_scenario_matrix_org_job_fk
        FOREIGN KEY (organization_id, owner_user_id, source_job_id)
        REFERENCES backtest_jobs(organization_id, user_id, job_id) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_scenario_matrix_org_compatibility_fk
        FOREIGN KEY (organization_id, compatibility_check_id)
        REFERENCES strategy_variant_compatibility_checks(
            organization_id,
            compatibility_check_id
        ) ON DELETE SET NULL (compatibility_check_id),
    ADD CONSTRAINT strategy_scenario_matrix_org_requirement_fk
        FOREIGN KEY (organization_id, market_data_requirement_id)
        REFERENCES market_data_subscription_requirements(
            organization_id,
            market_data_requirement_id
        ) ON DELETE SET NULL (market_data_requirement_id),
    ADD CONSTRAINT strategy_scenario_matrix_org_id_unique
        UNIQUE (organization_id, scenario_matrix_row_id),
    ADD CONSTRAINT strategy_scenario_matrix_org_owner_id_unique
        UNIQUE (organization_id, owner_user_id, scenario_matrix_row_id),
    ADD CONSTRAINT strategy_scenario_matrix_org_unique_row
        UNIQUE (
            organization_id,
            owner_user_id,
            source_job_id,
            source_variant_key,
            scenario_key
        );

DROP INDEX uq_rl_risk_sizing_policies_scope;
ALTER TABLE rl_risk_sizing_policies
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT rl_risk_policies_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT rl_risk_policies_org_strategy_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT rl_risk_policies_org_id_unique
        UNIQUE (organization_id, policy_id),
    ADD CONSTRAINT rl_risk_policies_org_scope_unique
        UNIQUE (
            organization_id,
            owner_user_id,
            strategy_id,
            exchange_name,
            market_type,
            symbol
        );

ALTER TABLE rl_risk_sizing_policy_audit_events
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT rl_risk_policy_audit_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT rl_risk_policy_audit_org_policy_fk
        FOREIGN KEY (organization_id, policy_id)
        REFERENCES rl_risk_sizing_policies(organization_id, policy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT rl_risk_policy_audit_org_strategy_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE;

ALTER TABLE rl_live_ticker_entitlement_overrides
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT rl_live_ticker_entitlement_overrides_pkey,
    ADD CONSTRAINT rl_live_ticker_overrides_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE CASCADE,
    ADD CONSTRAINT rl_live_ticker_overrides_org_owner_pk
        PRIMARY KEY (organization_id, owner_user_id);

DROP INDEX uq_rl_live_ticker_activations_active_owner_ticker;
ALTER TABLE rl_live_ticker_activations
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT rl_live_ticker_activations_strategy_id_fkey,
    DROP CONSTRAINT rl_live_ticker_activations_live_profile_id_fkey,
    ADD CONSTRAINT rl_live_ticker_activations_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE CASCADE,
    ADD CONSTRAINT rl_live_ticker_activations_org_strategy_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT rl_live_ticker_activations_org_profile_fk
        FOREIGN KEY (organization_id, owner_user_id, live_profile_id)
        REFERENCES strategy_live_profiles(organization_id, owner_user_id, profile_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT rl_live_ticker_activations_org_id_unique
        UNIQUE (organization_id, activation_id);

CREATE UNIQUE INDEX uq_rl_live_ticker_activations_active_owner_ticker
    ON rl_live_ticker_activations (
        organization_id,
        owner_user_id,
        exchange_name,
        market_type,
        symbol
    )
    WHERE active;

ALTER TABLE strategy_position_ownership
    ADD CONSTRAINT strategy_position_ownership_org_run_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_run_id)
        REFERENCES strategy_runs(organization_id, user_id, run_id) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_position_ownership_org_profile_fk
        FOREIGN KEY (organization_id, live_profile_id)
        REFERENCES strategy_live_profiles(organization_id, profile_id)
        ON DELETE SET NULL (live_profile_id),
    ADD CONSTRAINT strategy_position_ownership_org_id_unique
        UNIQUE (organization_id, ownership_id);

DROP INDEX strategy_position_ownership_one_blocking;
CREATE UNIQUE INDEX strategy_position_ownership_one_blocking
    ON strategy_position_ownership (
        organization_id,
        owner_user_id,
        exchange_connection_id,
        market_type,
        instrument_key
    )
    WHERE state IN ('reserved', 'active', 'releasing', 'stale_requires_repair');

ALTER TABLE exchange_account_snapshots
    ADD CONSTRAINT exchange_account_snapshots_org_owner_connection_id_unique
        UNIQUE (
            organization_id,
            owner_user_id,
            exchange_connection_id,
            account_snapshot_id
        );

ALTER TABLE exchange_balance_snapshots
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT exchange_balance_snapshots_org_account_fk
        FOREIGN KEY (
            organization_id,
            owner_user_id,
            exchange_connection_id,
            account_snapshot_id
        ) REFERENCES exchange_account_snapshots(
            organization_id,
            owner_user_id,
            exchange_connection_id,
            account_snapshot_id
        ) ON DELETE CASCADE;

ALTER TABLE exchange_open_order_snapshots
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT exchange_open_order_snapshots_org_account_fk
        FOREIGN KEY (
            organization_id,
            owner_user_id,
            exchange_connection_id,
            account_snapshot_id
        ) REFERENCES exchange_account_snapshots(
            organization_id,
            owner_user_id,
            exchange_connection_id,
            account_snapshot_id
        ) ON DELETE CASCADE;

ALTER TABLE exchange_instrument_filter_snapshots
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT exchange_instrument_filters_org_account_fk
        FOREIGN KEY (
            organization_id,
            owner_user_id,
            exchange_connection_id,
            account_snapshot_id
        ) REFERENCES exchange_account_snapshots(
            organization_id,
            owner_user_id,
            exchange_connection_id,
            account_snapshot_id
        ) ON DELETE CASCADE;

ALTER TABLE exchange_account_config_guard_results
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT exchange_config_guard_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT exchange_config_guard_org_connection_fk
        FOREIGN KEY (organization_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, connection_id) ON DELETE RESTRICT,
    ADD CONSTRAINT exchange_config_guard_org_account_fk
        FOREIGN KEY (organization_id, account_snapshot_id)
        REFERENCES exchange_account_snapshots(organization_id, account_snapshot_id)
        ON DELETE SET NULL (account_snapshot_id);

ALTER TABLE strategy_capital_reservations
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT strategy_capital_reservations_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_capital_reservations_org_strategy_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT strategy_capital_reservations_org_run_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_run_id)
        REFERENCES strategy_runs(organization_id, user_id, run_id) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_capital_reservations_org_id_unique
        UNIQUE (organization_id, reservation_id);

ALTER TABLE paper_orders
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT paper_orders_signal_unique,
    ADD CONSTRAINT paper_orders_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT paper_orders_org_strategy_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT paper_orders_org_run_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_run_id)
        REFERENCES strategy_runs(organization_id, user_id, run_id) ON DELETE CASCADE,
    ADD CONSTRAINT paper_orders_org_reservation_fk
        FOREIGN KEY (organization_id, reservation_id)
        REFERENCES strategy_capital_reservations(organization_id, reservation_id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT paper_orders_org_signal_unique
        UNIQUE (organization_id, source_signal_id),
    ADD CONSTRAINT paper_orders_org_id_unique
        UNIQUE (organization_id, paper_order_id),
    ADD CONSTRAINT paper_orders_org_owner_id_unique
        UNIQUE (organization_id, owner_user_id, paper_order_id);

ALTER TABLE paper_fills
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT paper_fills_order_unique,
    ADD CONSTRAINT paper_fills_org_order_fk
        FOREIGN KEY (organization_id, paper_order_id)
        REFERENCES paper_orders(organization_id, paper_order_id) ON DELETE CASCADE,
    ADD CONSTRAINT paper_fills_org_order_unique
        UNIQUE (organization_id, paper_order_id),
    ADD CONSTRAINT paper_fills_org_id_unique
        UNIQUE (organization_id, paper_fill_id);

ALTER TABLE strategy_paper_accounting
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT strategy_paper_accounting_fill_unique,
    ADD CONSTRAINT strategy_paper_accounting_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_paper_accounting_org_strategy_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_accounting_org_run_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_run_id)
        REFERENCES strategy_runs(organization_id, user_id, run_id) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_accounting_org_reservation_fk
        FOREIGN KEY (organization_id, reservation_id)
        REFERENCES strategy_capital_reservations(organization_id, reservation_id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_paper_accounting_org_fill_fk
        FOREIGN KEY (organization_id, paper_fill_id)
        REFERENCES paper_fills(organization_id, paper_fill_id) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_accounting_org_fill_unique
        UNIQUE (organization_id, paper_fill_id),
    ADD CONSTRAINT strategy_paper_accounting_org_id_unique
        UNIQUE (organization_id, accounting_id),
    ADD CONSTRAINT strategy_paper_accounting_org_owner_id_unique
        UNIQUE (organization_id, owner_user_id, accounting_id);

ALTER TABLE execution_source_events
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT execution_source_events_idempotency_unique,
    ADD CONSTRAINT execution_source_events_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT execution_source_events_org_signal_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_signal_id)
        REFERENCES strategy_signals(organization_id, owner_user_id, signal_id)
        ON DELETE SET NULL (strategy_signal_id),
    ADD CONSTRAINT execution_source_events_org_idempotency_unique
        UNIQUE (organization_id, owner_user_id, source_type, idempotency_key_hash),
    ADD CONSTRAINT execution_source_events_org_id_unique
        UNIQUE (organization_id, source_event_id),
    ADD CONSTRAINT execution_source_events_org_owner_id_unique
        UNIQUE (organization_id, owner_user_id, source_event_id);

ALTER TABLE execution_intents
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT execution_intents_idempotency_unique,
    ADD CONSTRAINT execution_intents_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT execution_intents_org_source_fk
        FOREIGN KEY (organization_id, owner_user_id, source_event_id)
        REFERENCES execution_source_events(organization_id, owner_user_id, source_event_id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT execution_intents_org_connection_fk
        FOREIGN KEY (organization_id, owner_user_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, owner_user_id, connection_id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT execution_intents_org_signal_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_signal_id)
        REFERENCES strategy_signals(organization_id, owner_user_id, signal_id)
        ON DELETE SET NULL (strategy_signal_id),
    ADD CONSTRAINT execution_intents_org_idempotency_unique
        UNIQUE (organization_id, owner_user_id, idempotency_key_hash),
    ADD CONSTRAINT execution_intents_org_id_unique
        UNIQUE (organization_id, intent_id),
    ADD CONSTRAINT execution_intents_org_owner_id_unique
        UNIQUE (organization_id, owner_user_id, intent_id);

ALTER TABLE execution_risk_audit_events
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT execution_risk_audit_org_intent_fk
        FOREIGN KEY (organization_id, owner_user_id, intent_id)
        REFERENCES execution_intents(organization_id, owner_user_id, intent_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT execution_risk_audit_org_source_fk
        FOREIGN KEY (organization_id, owner_user_id, source_event_id)
        REFERENCES execution_source_events(organization_id, owner_user_id, source_event_id)
        ON DELETE CASCADE;

ALTER TABLE execution_orders
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT execution_orders_intent_id_key,
    ADD CONSTRAINT execution_orders_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT execution_orders_org_intent_fk
        FOREIGN KEY (organization_id, owner_user_id, intent_id)
        REFERENCES execution_intents(organization_id, owner_user_id, intent_id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT execution_orders_org_connection_fk
        FOREIGN KEY (organization_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, connection_id) ON DELETE RESTRICT,
    ADD CONSTRAINT execution_orders_org_intent_unique
        UNIQUE (organization_id, intent_id),
    ADD CONSTRAINT execution_orders_org_id_unique
        UNIQUE (organization_id, order_id);

-- Successfully resolved ingress observations carry the organization and are
-- constrained to an intent in that organization. Quarantined or malformed
-- messages may keep organization_id NULL because no trustworthy owner exists.
ALTER TABLE exchange_execution_request_observations
    ADD COLUMN organization_id UUID NULL,
    ADD CONSTRAINT exchange_execution_observations_org_intent_fk
        FOREIGN KEY (organization_id, intent_id)
        REFERENCES execution_intents(organization_id, intent_id) ON DELETE CASCADE;

ALTER TABLE exchange_private_stream_sessions
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT exchange_private_stream_sessions_unique,
    ADD CONSTRAINT exchange_private_stream_sessions_org_connection_fk
        FOREIGN KEY (organization_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, connection_id) ON DELETE CASCADE,
    ADD CONSTRAINT exchange_private_stream_sessions_org_unique
        UNIQUE (
            organization_id,
            exchange_connection_id,
            exchange_name,
            market_type,
            environment
        );

ALTER TABLE execution_order_events
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT execution_order_events_org_order_fk
        FOREIGN KEY (organization_id, order_id)
        REFERENCES execution_orders(organization_id, order_id) ON DELETE CASCADE,
    ADD CONSTRAINT execution_order_events_org_intent_fk
        FOREIGN KEY (organization_id, intent_id)
        REFERENCES execution_intents(organization_id, intent_id) ON DELETE CASCADE;

ALTER TABLE execution_fills
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT execution_fills_org_order_fk
        FOREIGN KEY (organization_id, order_id)
        REFERENCES execution_orders(organization_id, order_id) ON DELETE CASCADE,
    ADD CONSTRAINT execution_fills_org_intent_fk
        FOREIGN KEY (organization_id, intent_id)
        REFERENCES execution_intents(organization_id, intent_id) ON DELETE CASCADE;

ALTER TABLE execution_funding_events
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT execution_funding_events_org_order_fk
        FOREIGN KEY (organization_id, order_id)
        REFERENCES execution_orders(organization_id, order_id) ON DELETE CASCADE,
    ADD CONSTRAINT execution_funding_events_org_intent_fk
        FOREIGN KEY (organization_id, intent_id)
        REFERENCES execution_intents(organization_id, intent_id) ON DELETE CASCADE;

ALTER TABLE execution_reconciliation_runs
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT execution_reconciliation_org_order_fk
        FOREIGN KEY (organization_id, order_id)
        REFERENCES execution_orders(organization_id, order_id) ON DELETE CASCADE,
    ADD CONSTRAINT execution_reconciliation_org_intent_fk
        FOREIGN KEY (organization_id, intent_id)
        REFERENCES execution_intents(organization_id, intent_id) ON DELETE CASCADE;

ALTER TABLE execution_notification_outbox
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT execution_notification_outbox_dedupe,
    ADD CONSTRAINT execution_notification_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT execution_notification_org_source_fk
        FOREIGN KEY (organization_id, source_event_id)
        REFERENCES execution_source_events(organization_id, source_event_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT execution_notification_org_intent_fk
        FOREIGN KEY (organization_id, intent_id)
        REFERENCES execution_intents(organization_id, intent_id) ON DELETE CASCADE,
    ADD CONSTRAINT execution_notification_org_order_fk
        FOREIGN KEY (organization_id, order_id)
        REFERENCES execution_orders(organization_id, order_id) ON DELETE CASCADE,
    ADD CONSTRAINT execution_notification_org_signal_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_signal_id)
        REFERENCES strategy_signals(organization_id, owner_user_id, signal_id)
        ON DELETE SET NULL (strategy_signal_id),
    ADD CONSTRAINT execution_notification_org_dedupe
        UNIQUE (
            organization_id,
            owner_user_id,
            event_type,
            source_event_key,
            intent_key,
            order_key,
            reason
        );

ALTER TABLE strategy_paper_scenario_coverage_results
    ADD COLUMN organization_id UUID NOT NULL,
    DROP CONSTRAINT strategy_paper_coverage_unique_scenario,
    ADD CONSTRAINT strategy_paper_coverage_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_paper_coverage_org_scenario_fk
        FOREIGN KEY (organization_id, owner_user_id, scenario_matrix_row_id)
        REFERENCES strategy_variant_scenario_matrix_rows(
            organization_id,
            owner_user_id,
            scenario_matrix_row_id
        ) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_coverage_org_job_fk
        FOREIGN KEY (organization_id, owner_user_id, source_job_id)
        REFERENCES backtest_jobs(organization_id, user_id, job_id) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_coverage_org_strategy_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, user_id, strategy_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_coverage_org_profile_fk
        FOREIGN KEY (organization_id, owner_user_id, live_profile_id)
        REFERENCES strategy_live_profiles(organization_id, owner_user_id, profile_id)
        ON DELETE SET NULL (live_profile_id),
    ADD CONSTRAINT strategy_paper_coverage_org_run_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_run_id)
        REFERENCES strategy_runs(organization_id, user_id, run_id) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_coverage_org_signal_fk
        FOREIGN KEY (organization_id, owner_user_id, strategy_signal_id)
        REFERENCES strategy_signals(organization_id, owner_user_id, signal_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_coverage_org_source_fk
        FOREIGN KEY (organization_id, owner_user_id, source_event_id)
        REFERENCES execution_source_events(organization_id, owner_user_id, source_event_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_coverage_org_intent_fk
        FOREIGN KEY (organization_id, owner_user_id, intent_id)
        REFERENCES execution_intents(organization_id, owner_user_id, intent_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_coverage_org_order_fk
        FOREIGN KEY (organization_id, owner_user_id, paper_order_id)
        REFERENCES paper_orders(organization_id, owner_user_id, paper_order_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_coverage_org_fill_fk
        FOREIGN KEY (organization_id, paper_fill_id)
        REFERENCES paper_fills(organization_id, paper_fill_id) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_coverage_org_accounting_fk
        FOREIGN KEY (organization_id, owner_user_id, accounting_id)
        REFERENCES strategy_paper_accounting(
            organization_id,
            owner_user_id,
            accounting_id
        ) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_paper_coverage_org_unique_scenario
        UNIQUE (organization_id, owner_user_id, scenario_key);

CREATE INDEX idx_strategy_runs_org_user_started
    ON strategy_runs (organization_id, user_id, started_at DESC, run_id DESC);
CREATE INDEX idx_strategy_events_org_user_strategy_ts
    ON strategy_events (organization_id, user_id, strategy_id, ts DESC, event_id DESC);
CREATE INDEX idx_strategy_signals_org_owner_strategy_created
    ON strategy_signals (
        organization_id,
        owner_user_id,
        strategy_id,
        created_at DESC,
        signal_id DESC
    );
CREATE INDEX idx_strategy_compatibility_org_owner_checked
    ON strategy_variant_compatibility_checks (
        organization_id,
        owner_user_id,
        checked_at DESC
    );
CREATE INDEX idx_strategy_scenario_matrix_org_owner_checked
    ON strategy_variant_scenario_matrix_rows (
        organization_id,
        owner_user_id,
        checked_at DESC
    );
CREATE INDEX idx_strategy_paper_coverage_org_owner_checked
    ON strategy_paper_scenario_coverage_results (
        organization_id,
        owner_user_id,
        checked_at DESC
    );
CREATE INDEX idx_rl_risk_policies_org_owner_updated
    ON rl_risk_sizing_policies (
        organization_id,
        owner_user_id,
        updated_at DESC
    );
CREATE INDEX idx_rl_risk_policy_audit_org_policy_created
    ON rl_risk_sizing_policy_audit_events (
        organization_id,
        policy_id,
        created_at DESC
    );

CREATE INDEX idx_rl_live_ticker_activations_org_owner_active
    ON rl_live_ticker_activations (
        organization_id,
        owner_user_id,
        active,
        updated_at DESC
    );

CREATE INDEX idx_rl_live_ticker_activations_org_profile_active
    ON rl_live_ticker_activations (
        organization_id,
        owner_user_id,
        strategy_id,
        active
    );
CREATE INDEX idx_execution_source_events_org_owner_received
    ON execution_source_events (organization_id, owner_user_id, received_at DESC);
CREATE INDEX idx_execution_intents_org_owner_created
    ON execution_intents (organization_id, owner_user_id, created_at DESC);
CREATE INDEX idx_execution_orders_org_owner_created
    ON execution_orders (organization_id, owner_user_id, created_at DESC);
CREATE INDEX idx_exchange_execution_observations_org_intent
    ON exchange_execution_request_observations (
        organization_id,
        intent_id,
        observed_at DESC
    )
    WHERE organization_id IS NOT NULL AND intent_id IS NOT NULL;
CREATE INDEX idx_exchange_private_stream_sessions_org_status
    ON exchange_private_stream_sessions (
        organization_id,
        exchange_name,
        status,
        updated_at DESC
    );
CREATE INDEX idx_strategy_paper_accounting_org_latest
    ON strategy_paper_accounting (
        organization_id,
        owner_user_id,
        strategy_id,
        created_at DESC
    );

COMMIT;
