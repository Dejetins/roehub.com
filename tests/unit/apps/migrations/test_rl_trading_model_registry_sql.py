from pathlib import Path


def _migration_text() -> str:
    return Path(
        "alembic/versions/20260702_0040_rl_trading_model_registry_v1.py"
    ).read_text(encoding="utf-8")


def test_rl_trading_model_registry_migration_adds_metadata_tables() -> None:
    text = _migration_text()

    assert "CREATE TABLE IF NOT EXISTS rl_trading_dataset_versions" in text
    assert "CREATE TABLE IF NOT EXISTS rl_trading_training_runs" in text
    assert "CREATE TABLE IF NOT EXISTS rl_trading_model_versions" in text
    assert "CREATE TABLE IF NOT EXISTS rl_trading_calibration_packs" in text
    assert "CREATE TABLE IF NOT EXISTS rl_trading_model_activations" in text
    assert "CREATE TABLE IF NOT EXISTS rl_trading_model_registry_audit_events" in text
    assert "CREATE TABLE IF NOT EXISTS rl_trading_artifact_lifecycle_policies" in text
    assert 'down_revision = "20260702_0039"' in text


def test_rl_trading_model_registry_migration_enforces_state_and_hash_contracts() -> None:
    text = _migration_text()

    assert "status IN (" in text
    assert "'accepted_champion'" in text
    assert "'rollback_candidate'" in text
    assert "'missing_artifact'" in text
    assert "'monitor_only'" in text
    assert "'testnet'" in text
    assert "'live'" in text
    assert "~ '^[0-9a-f]{64}$'" in text
    assert "manifest_path LIKE '{_ARTIFACT_ROOT}%'" in text
    assert "checkpoint_path LIKE '{_ARTIFACT_ROOT}%'" in text
    assert "calibration_path LIKE '{_ARTIFACT_ROOT}%'" in text
    assert "producer = 'roehub_trainer_service'" in text
    assert "missing_reason_code IS NULL OR missing_reason_code" in text


def test_rl_trading_model_registry_migration_has_champion_activation_and_policy_guards() -> None:
    text = _migration_text()

    assert "idx_rl_trading_model_versions_one_champion" in text
    assert "WHERE status = 'accepted_champion'" in text
    assert "idx_rl_trading_model_activations_current_scope" in text
    assert "WHERE is_current" in text
    assert "rejected_run_retention_days INTEGER NOT NULL" in text
    assert "CHECK (rejected_run_retention_days > 0)" in text
    assert "disk_quota_bytes BIGINT NOT NULL" in text
    assert "disk_watermark_pct > 0 AND disk_watermark_pct < 100" in text


def test_rl_trading_model_registry_migration_keeps_secrets_and_raw_payloads_out() -> None:
    text = _migration_text().lower()

    assert "raw_payload" not in text
    assert "provider_payload" not in text
    assert "api_secret" not in text
    assert "api_key" not in text
    assert "password" not in text
    assert "token" not in text
    assert "operator_ref_hash" in text
    assert "details_json" in text
