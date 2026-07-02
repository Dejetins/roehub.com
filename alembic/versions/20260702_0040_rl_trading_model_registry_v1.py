"""Add RL trading model registry metadata tables."""

from __future__ import annotations

from alembic import op

revision = "20260702_0040"
down_revision = "20260702_0039"
branch_labels = None
depends_on = None

_ARTIFACT_ROOT = "/opt/roehub/state/rl_trading/"
_SHA256_CHECK = "~ '^[0-9a-f]{64}$'"
_REASON_CHECK = "~ '^[a-z0-9][a-z0-9_:-]{0,95}$'"


def upgrade() -> None:
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS rl_trading_dataset_versions (
            dataset_version_id TEXT PRIMARY KEY,
            dataset_hash TEXT NOT NULL,
            feature_contract_hash TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            artifact_deleted_at TIMESTAMPTZ NULL,
            missing_reason_code TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT rl_trading_dataset_versions_dataset_hash_chk
                CHECK (dataset_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_dataset_versions_feature_contract_hash_chk
                CHECK (feature_contract_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_dataset_versions_manifest_sha_chk
                CHECK (manifest_sha256 {_SHA256_CHECK}),
            CONSTRAINT rl_trading_dataset_versions_manifest_path_chk
                CHECK (manifest_path LIKE '{_ARTIFACT_ROOT}%'),
            CONSTRAINT rl_trading_dataset_versions_status_chk CHECK (
                status IN (
                    'building',
                    'qa_failed',
                    'accepted',
                    'missing_artifact',
                    'superseded'
                )
            ),
            CONSTRAINT rl_trading_dataset_versions_metadata_json_chk
                CHECK (jsonb_typeof(metadata_json) = 'object'),
            CONSTRAINT rl_trading_dataset_versions_missing_reason_chk CHECK (
                missing_reason_code IS NULL OR missing_reason_code {_REASON_CHECK}
            )
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS rl_trading_training_runs (
            training_run_id TEXT PRIMARY KEY,
            dataset_version_id TEXT NOT NULL REFERENCES rl_trading_dataset_versions(
                dataset_version_id
            ),
            model_family TEXT NOT NULL,
            run_config_hash TEXT NOT NULL,
            run_manifest_path TEXT NOT NULL,
            run_manifest_sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            metrics_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            artifact_deleted_at TIMESTAMPTZ NULL,
            missing_reason_code TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT rl_trading_training_runs_config_hash_chk
                CHECK (run_config_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_training_runs_manifest_sha_chk
                CHECK (run_manifest_sha256 {_SHA256_CHECK}),
            CONSTRAINT rl_trading_training_runs_manifest_path_chk
                CHECK (run_manifest_path LIKE '{_ARTIFACT_ROOT}%'),
            CONSTRAINT rl_trading_training_runs_status_chk CHECK (
                status IN ('planned', 'running', 'failed', 'completed', 'rejected', 'candidate')
            ),
            CONSTRAINT rl_trading_training_runs_metrics_json_chk
                CHECK (jsonb_typeof(metrics_json) = 'object'),
            CONSTRAINT rl_trading_training_runs_missing_reason_chk CHECK (
                missing_reason_code IS NULL OR missing_reason_code {_REASON_CHECK}
            )
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS rl_trading_model_versions (
            model_version_id TEXT PRIMARY KEY,
            training_run_id TEXT NOT NULL REFERENCES rl_trading_training_runs(training_run_id),
            dataset_version_id TEXT NOT NULL REFERENCES rl_trading_dataset_versions(
                dataset_version_id
            ),
            model_family TEXT NOT NULL,
            feature_contract_hash TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            model_state_hash TEXT NOT NULL,
            checkpoint_path TEXT NOT NULL,
            checkpoint_sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            producer TEXT NOT NULL DEFAULT 'roehub_trainer_service',
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            artifact_deleted_at TIMESTAMPTZ NULL,
            missing_reason_code TEXT NULL,
            accepted_at TIMESTAMPTZ NULL,
            replaced_by_model_version_id TEXT NULL REFERENCES rl_trading_model_versions(
                model_version_id
            ),
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT rl_trading_model_versions_feature_contract_hash_chk
                CHECK (feature_contract_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_model_versions_dataset_hash_chk
                CHECK (dataset_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_model_versions_state_hash_chk
                CHECK (model_state_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_model_versions_checkpoint_sha_chk
                CHECK (checkpoint_sha256 {_SHA256_CHECK}),
            CONSTRAINT rl_trading_model_versions_checkpoint_path_chk
                CHECK (checkpoint_path LIKE '{_ARTIFACT_ROOT}%'),
            CONSTRAINT rl_trading_model_versions_status_chk CHECK (
                status IN (
                    'candidate',
                    'rejected',
                    'accepted_champion',
                    'rollback_candidate',
                    'missing_artifact'
                )
            ),
            CONSTRAINT rl_trading_model_versions_producer_chk
                CHECK (producer = 'roehub_trainer_service'),
            CONSTRAINT rl_trading_model_versions_metadata_json_chk
                CHECK (jsonb_typeof(metadata_json) = 'object'),
            CONSTRAINT rl_trading_model_versions_missing_reason_chk CHECK (
                missing_reason_code IS NULL OR missing_reason_code {_REASON_CHECK}
            )
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_rl_trading_model_versions_one_champion
            ON rl_trading_model_versions (model_family, feature_contract_hash)
            WHERE status = 'accepted_champion'
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS rl_trading_calibration_packs (
            calibration_pack_id TEXT PRIMARY KEY,
            model_version_id TEXT NOT NULL REFERENCES rl_trading_model_versions(
                model_version_id
            ),
            dataset_version_id TEXT NOT NULL REFERENCES rl_trading_dataset_versions(
                dataset_version_id
            ),
            feature_contract_hash TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            calibration_pack_hash TEXT NOT NULL,
            calibration_path TEXT NOT NULL,
            calibration_sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            artifact_deleted_at TIMESTAMPTZ NULL,
            missing_reason_code TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT rl_trading_calibration_packs_feature_contract_hash_chk
                CHECK (feature_contract_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_calibration_packs_dataset_hash_chk
                CHECK (dataset_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_calibration_packs_pack_hash_chk
                CHECK (calibration_pack_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_calibration_packs_calibration_sha_chk
                CHECK (calibration_sha256 {_SHA256_CHECK}),
            CONSTRAINT rl_trading_calibration_packs_calibration_path_chk
                CHECK (calibration_path LIKE '{_ARTIFACT_ROOT}%'),
            CONSTRAINT rl_trading_calibration_packs_status_chk CHECK (
                status IN (
                    'candidate',
                    'accepted',
                    'rejected',
                    'superseded',
                    'missing_artifact'
                )
            ),
            CONSTRAINT rl_trading_calibration_packs_metadata_json_chk
                CHECK (jsonb_typeof(metadata_json) = 'object'),
            CONSTRAINT rl_trading_calibration_packs_missing_reason_chk CHECK (
                missing_reason_code IS NULL OR missing_reason_code {_REASON_CHECK}
            )
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS rl_trading_model_activations (
            activation_id UUID PRIMARY KEY,
            model_family TEXT NOT NULL,
            feature_contract_hash TEXT NOT NULL,
            exchange TEXT NOT NULL,
            market_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            model_version_id TEXT NOT NULL REFERENCES rl_trading_model_versions(
                model_version_id
            ),
            calibration_pack_id TEXT NOT NULL REFERENCES rl_trading_calibration_packs(
                calibration_pack_id
            ),
            dataset_version_id TEXT NOT NULL REFERENCES rl_trading_dataset_versions(
                dataset_version_id
            ),
            activation_state TEXT NOT NULL,
            activation_matrix_hash TEXT NOT NULL,
            is_current BOOLEAN NOT NULL DEFAULT true,
            previous_activation_id UUID NULL REFERENCES rl_trading_model_activations(
                activation_id
            ),
            reason_code TEXT NOT NULL,
            operator_ref_hash TEXT NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            changed_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT rl_trading_model_activations_feature_contract_hash_chk
                CHECK (feature_contract_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_model_activations_exchange_chk
                CHECK (exchange IN ('binance', 'bybit')),
            CONSTRAINT rl_trading_model_activations_market_type_chk
                CHECK (market_type IN ('spot', 'futures')),
            CONSTRAINT rl_trading_model_activations_symbol_chk
                CHECK (symbol = upper(symbol) AND char_length(symbol) > 0),
            CONSTRAINT rl_trading_model_activations_state_chk CHECK (
                activation_state IN (
                    'inactive',
                    'shadow',
                    'monitor_only',
                    'paper',
                    'testnet',
                    'live',
                    'paused',
                    'rolled_back'
                )
            ),
            CONSTRAINT rl_trading_model_activations_matrix_hash_chk
                CHECK (activation_matrix_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_model_activations_reason_chk
                CHECK (reason_code {_REASON_CHECK}),
            CONSTRAINT rl_trading_model_activations_operator_ref_hash_chk
                CHECK (operator_ref_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_model_activations_metadata_json_chk
                CHECK (jsonb_typeof(metadata_json) = 'object')
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_rl_trading_model_activations_current_scope
            ON rl_trading_model_activations (
                model_family,
                feature_contract_hash,
                exchange,
                market_type,
                symbol
            )
            WHERE is_current
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS rl_trading_model_registry_audit_events (
            audit_event_id UUID PRIMARY KEY,
            event_type TEXT NOT NULL,
            model_version_id TEXT NULL REFERENCES rl_trading_model_versions(model_version_id),
            calibration_pack_id TEXT NULL REFERENCES rl_trading_calibration_packs(
                calibration_pack_id
            ),
            dataset_version_id TEXT NULL REFERENCES rl_trading_dataset_versions(
                dataset_version_id
            ),
            previous_activation_id UUID NULL REFERENCES rl_trading_model_activations(
                activation_id
            ),
            next_activation_id UUID NULL REFERENCES rl_trading_model_activations(
                activation_id
            ),
            previous_state TEXT NULL,
            next_state TEXT NULL,
            checkpoint_sha256 TEXT NULL,
            calibration_pack_hash TEXT NULL,
            dataset_hash TEXT NULL,
            reason_code TEXT NOT NULL,
            operator_ref_hash TEXT NOT NULL,
            event_payload_hash TEXT NOT NULL,
            details_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT rl_trading_model_registry_audit_event_type_chk CHECK (
                event_type IN (
                    'candidate_registered',
                    'champion_promoted',
                    'activation_changed',
                    'rollback_selected',
                    'artifact_missing',
                    'cleanup_planned'
                )
            ),
            CONSTRAINT rl_trading_model_registry_audit_checkpoint_sha_chk CHECK (
                checkpoint_sha256 IS NULL OR checkpoint_sha256 {_SHA256_CHECK}
            ),
            CONSTRAINT rl_trading_model_registry_audit_calibration_hash_chk CHECK (
                calibration_pack_hash IS NULL OR calibration_pack_hash {_SHA256_CHECK}
            ),
            CONSTRAINT rl_trading_model_registry_audit_dataset_hash_chk CHECK (
                dataset_hash IS NULL OR dataset_hash {_SHA256_CHECK}
            ),
            CONSTRAINT rl_trading_model_registry_audit_reason_chk
                CHECK (reason_code {_REASON_CHECK}),
            CONSTRAINT rl_trading_model_registry_audit_operator_ref_hash_chk
                CHECK (operator_ref_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_model_registry_audit_payload_hash_chk
                CHECK (event_payload_hash {_SHA256_CHECK}),
            CONSTRAINT rl_trading_model_registry_audit_details_json_chk
                CHECK (jsonb_typeof(details_json) = 'object')
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rl_trading_model_registry_audit_created
            ON rl_trading_model_registry_audit_events (created_at DESC)
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS rl_trading_artifact_lifecycle_policies (
            policy_id TEXT PRIMARY KEY,
            artifact_root TEXT NOT NULL,
            rejected_run_retention_days INTEGER NOT NULL,
            disk_quota_bytes BIGINT NOT NULL,
            disk_watermark_pct NUMERIC(5, 2) NOT NULL,
            cleanup_enabled BOOLEAN NOT NULL DEFAULT false,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT rl_trading_artifact_lifecycle_root_chk
                CHECK (artifact_root = '{_ARTIFACT_ROOT.rstrip("/")}'),
            CONSTRAINT rl_trading_artifact_lifecycle_retention_chk
                CHECK (rejected_run_retention_days > 0),
            CONSTRAINT rl_trading_artifact_lifecycle_quota_chk
                CHECK (disk_quota_bytes > 0),
            CONSTRAINT rl_trading_artifact_lifecycle_watermark_chk
                CHECK (disk_watermark_pct > 0 AND disk_watermark_pct < 100),
            CONSTRAINT rl_trading_artifact_lifecycle_metadata_json_chk
                CHECK (jsonb_typeof(metadata_json) = 'object')
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS rl_trading_artifact_lifecycle_policies")
    op.execute("DROP INDEX IF EXISTS idx_rl_trading_model_registry_audit_created")
    op.execute("DROP TABLE IF EXISTS rl_trading_model_registry_audit_events")
    op.execute("DROP INDEX IF EXISTS idx_rl_trading_model_activations_current_scope")
    op.execute("DROP TABLE IF EXISTS rl_trading_model_activations")
    op.execute("DROP TABLE IF EXISTS rl_trading_calibration_packs")
    op.execute("DROP INDEX IF EXISTS idx_rl_trading_model_versions_one_champion")
    op.execute("DROP TABLE IF EXISTS rl_trading_model_versions")
    op.execute("DROP TABLE IF EXISTS rl_trading_training_runs")
    op.execute("DROP TABLE IF EXISTS rl_trading_dataset_versions")
