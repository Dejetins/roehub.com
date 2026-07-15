from pathlib import Path


def _sql() -> str:
    return (
        Path(__file__).resolve().parents[4]
        / "migrations"
        / "postgres"
        / "0012_identity_local_auth_v1.sql"
    ).read_text(encoding="utf-8")


def test_local_auth_schema_keeps_only_hashes_and_public_passkey_material() -> None:
    sql = _sql()

    assert "identity_local_accounts" in sql
    assert "identity_webauthn_credentials" in sql
    assert "public_key BYTEA NOT NULL" in sql
    assert "password_hash IS NULL OR password_hash LIKE '$argon2id$%'" in sql
    assert "token_sha256" in sql
    assert "challenge_sha256" in sql
    assert "code_hash LIKE '$argon2id$%'" in sql
    assert "CREATE TABLE identity_local_auth_rate_limits" in sql


def test_local_auth_events_are_append_only_and_subjects_are_redacted() -> None:
    sql = _sql()

    assert "subject_sha256 ~ '^[0-9a-f]{64}$'" in sql
    assert "identity_local_auth_events_immutable" in sql
    assert "BEFORE UPDATE OR DELETE ON identity_local_auth_events" in sql
    assert "outcome IN ('succeeded', 'rejected')" in sql


def test_challenges_are_bounded_single_use_and_purpose_scoped() -> None:
    sql = _sql()

    assert "expires_at > created_at" in sql
    assert "consumed_at IS NULL" in sql
    assert "purpose IN ('bootstrap', 'login', 'register', 'recent_auth')" in sql
    assert "purpose IN ('bootstrap', 'login') AND user_id IS NULL" in sql
    assert "context_json->>'bootstrap_user_id'" in sql
    assert "idx_identity_local_bootstrap_single_active" in sql
