from pathlib import Path


def test_oidc_migration_is_hash_only_and_fail_closed() -> None:
    source = (
        Path(__file__).resolve().parents[4]
        / "migrations/postgres/0013_identity_oidc_provider_v1.sql"
    ).read_text(encoding="utf-8")

    assert "identity_oidc_login_attempts" in source
    assert "state_sha256" in source
    assert "nonce_sha256" in source
    assert "identity_external_identities" in source
    assert "subject_sha256" in source
    assert "UNIQUE (provider_id, issuer, subject_sha256)" in source
    assert "UNIQUE (provider_id, issuer, user_id)" in source
    assert "identity_oidc_auth_events_immutable" in source
    assert "access_token" not in source
    assert "refresh_token" not in source
    assert "id_token TEXT" not in source
