import pytest

from apps.api.wiring.modules.identity import build_identity_api_module, build_identity_router
from trading.contexts.identity.adapters.outbound.security.current_user import (
    RoehubSessionCurrentUser,
)


def test_identity_wiring_fail_fast_in_prod_requires_keycloak_base_url() -> None:
    """
    Verify prod default fail-fast rejects startup when `KEYCLOAK_BASE_URL` is missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Prod environment defaults fail-fast to enabled when override is absent.
    Raises:
        AssertionError: If router wiring does not fail on missing Keycloak base URL in prod.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "prod",
        "KEYCLOAK_REALM": "roehub",
        "KEYCLOAK_CLIENT_ID": "roehub-api",
        "KEYCLOAK_CLIENT_SECRET": "prod-client-secret",
        "KEYCLOAK_REDIRECT_URI": "https://roehub.com/auth/callback",
        "KEYCLOAK_LOGOUT_REDIRECT_URI": "https://roehub.com/login",
        "IDENTITY_EXCHANGE_KEYS_KEK_B64": "cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE=",
    }

    with pytest.raises(ValueError, match="KEYCLOAK_BASE_URL"):
        build_identity_router(environ=environ)


def test_identity_wiring_fail_fast_in_prod_requires_keycloak_client_secret() -> None:
    """
    Verify prod default fail-fast rejects startup when `KEYCLOAK_CLIENT_SECRET` is missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Base Keycloak settings except client secret are configured.
    Raises:
        AssertionError: If router wiring does not fail on missing Keycloak client secret in prod.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "prod",
        "KEYCLOAK_BASE_URL": "https://auth.roehub.com",
        "KEYCLOAK_REALM": "roehub",
        "KEYCLOAK_CLIENT_ID": "roehub-api",
        "KEYCLOAK_REDIRECT_URI": "https://roehub.com/auth/callback",
        "KEYCLOAK_LOGOUT_REDIRECT_URI": "https://roehub.com/login",
        "IDENTITY_EXCHANGE_KEYS_KEK_B64": "cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE=",
    }

    with pytest.raises(ValueError, match="KEYCLOAK_CLIENT_SECRET"):
        build_identity_router(environ=environ)


def test_identity_wiring_fail_fast_override_requires_exchange_keys_kek_b64() -> None:
    """
    Verify explicit `IDENTITY_FAIL_FAST=true` requires exchange keys KEK in non-prod env too.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Keycloak variables are set so fail-fast validation reaches exchange keys KEK check.
    Raises:
        AssertionError: If router wiring does not fail on missing exchange keys KEK.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "dev",
        "IDENTITY_FAIL_FAST": "true",
        "KEYCLOAK_BASE_URL": "http://127.0.0.1:18080",
        "KEYCLOAK_REALM": "roehub",
        "KEYCLOAK_CLIENT_ID": "roehub-api",
        "KEYCLOAK_CLIENT_SECRET": "dev-client-secret",
        "KEYCLOAK_REDIRECT_URI": "http://127.0.0.1:8010/auth/callback",
        "KEYCLOAK_LOGOUT_REDIRECT_URI": "http://127.0.0.1:8010/login",
        "IDENTITY_SESSION_IDLE_TTL_SECONDS": "1800",
        "IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS": "43200",
    }

    with pytest.raises(ValueError, match="IDENTITY_EXCHANGE_KEYS_KEK_B64"):
        build_identity_router(environ=environ)


def test_identity_wiring_does_not_require_notifier_secret_for_api_auth() -> None:
    """
    Verify identity fail-fast wiring succeeds without notifier-specific secret inputs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Notifier secret belongs to strategy scope, not identity/API auth scope.
    Raises:
        AssertionError: If identity router requires Telegram bot token unexpectedly.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "dev",
        "IDENTITY_FAIL_FAST": "true",
        "KEYCLOAK_BASE_URL": "http://127.0.0.1:18080",
        "KEYCLOAK_REALM": "roehub",
        "KEYCLOAK_CLIENT_ID": "roehub-api",
        "KEYCLOAK_CLIENT_SECRET": "dev-client-secret",
        "KEYCLOAK_REDIRECT_URI": "http://127.0.0.1:8010/auth/callback",
        "KEYCLOAK_LOGOUT_REDIRECT_URI": "http://127.0.0.1:8010/login",
        "IDENTITY_SESSION_IDLE_TTL_SECONDS": "1800",
        "IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS": "43200",
        "IDENTITY_EXCHANGE_KEYS_KEK_B64": "cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE=",
    }

    router = build_identity_router(environ=environ)
    paths = {
        str(getattr(route, "path"))
        for route in router.routes
        if hasattr(route, "path")
    }

    assert "/auth/login" in paths


def test_identity_wiring_fail_fast_in_prod_requires_session_idle_ttl() -> None:
    """
    Verify prod default fail-fast rejects startup when session idle TTL is missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Persisted session policy must be configured explicitly in prod.
    Raises:
        AssertionError: If router wiring does not fail on missing session idle TTL.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "prod",
        "KEYCLOAK_BASE_URL": "https://auth.roehub.com",
        "KEYCLOAK_REALM": "roehub",
        "KEYCLOAK_CLIENT_ID": "roehub-api",
        "KEYCLOAK_CLIENT_SECRET": "prod-client-secret",
        "KEYCLOAK_REDIRECT_URI": "https://roehub.com/auth/callback",
        "KEYCLOAK_LOGOUT_REDIRECT_URI": "https://roehub.com/login",
        "IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS": "43200",
        "IDENTITY_EXCHANGE_KEYS_KEK_B64": "cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE=",
    }

    with pytest.raises(ValueError, match="IDENTITY_SESSION_IDLE_TTL_SECONDS"):
        build_identity_router(environ=environ)


def test_identity_wiring_fail_fast_in_prod_requires_session_absolute_ttl() -> None:
    """
    Verify prod default fail-fast rejects startup when session absolute TTL is missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Persisted session policy must define absolute session lifetime in prod.
    Raises:
        AssertionError: If router wiring does not fail on missing session absolute TTL.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "prod",
        "KEYCLOAK_BASE_URL": "https://auth.roehub.com",
        "KEYCLOAK_REALM": "roehub",
        "KEYCLOAK_CLIENT_ID": "roehub-api",
        "KEYCLOAK_CLIENT_SECRET": "prod-client-secret",
        "KEYCLOAK_REDIRECT_URI": "https://roehub.com/auth/callback",
        "KEYCLOAK_LOGOUT_REDIRECT_URI": "https://roehub.com/login",
        "IDENTITY_SESSION_IDLE_TTL_SECONDS": "1800",
        "IDENTITY_EXCHANGE_KEYS_KEK_B64": "cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE=",
    }

    with pytest.raises(ValueError, match="IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS"):
        build_identity_router(environ=environ)


def test_identity_wiring_fail_fast_in_prod_requires_postgres_dsn() -> None:
    """
    Verify prod startup rejects missing Postgres DSN because persisted Roehub sessions
    are mandatory.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Prod runtime must not fall back to in-memory session storage.
    Raises:
        AssertionError: If router wiring allows prod startup without `IDENTITY_PG_DSN`.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "prod",
        "KEYCLOAK_BASE_URL": "https://auth.roehub.com",
        "KEYCLOAK_REALM": "roehub",
        "KEYCLOAK_CLIENT_ID": "roehub-api",
        "KEYCLOAK_CLIENT_SECRET": "prod-client-secret",
        "KEYCLOAK_REDIRECT_URI": "https://roehub.com/auth/callback",
        "KEYCLOAK_LOGOUT_REDIRECT_URI": "https://roehub.com/login",
        "IDENTITY_SESSION_IDLE_TTL_SECONDS": "1800",
        "IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS": "43200",
        "IDENTITY_EXCHANGE_KEYS_KEK_B64": "cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE=",
    }

    with pytest.raises(ValueError, match="IDENTITY_PG_DSN"):
        build_identity_router(environ=environ)


def test_identity_wiring_dev_defaults_allow_missing_keycloak_and_keks_and_expose_routes() -> None:
    """
    Verify dev defaults allow missing Keycloak settings/KEKs and expose target auth routes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Dev fallback KEKs are acceptable when fail-fast is not explicitly enabled.
    Raises:
        AssertionError: If critical identity routes are missing.
    Side Effects:
        None.
    """
    router = build_identity_router(environ={"ROEHUB_ENV": "dev"})
    paths = sorted(
        str(getattr(route, "path"))
        for route in router.routes
        if hasattr(route, "path")
    )

    assert "/auth/login" in paths
    assert "/auth/callback" in paths
    assert "/auth/logout" in paths
    assert "/auth/current-user" in paths
    assert "/exchange-keys" in paths
    assert "/exchange-keys/{key_id}" in paths


def test_identity_wiring_uses_roehub_session_current_user_resolver() -> None:
    """
    Verify wiring builds `RequireCurrentUserDependency` on top of Roehub session resolver.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Final runtime path must resolve principal from local Roehub session storage.
    Raises:
        AssertionError: If current-user resolver type differs from Roehub session adapter.
    Side Effects:
        None.
    """
    module = build_identity_api_module(environ={"ROEHUB_ENV": "dev"})

    resolver = module.current_user_dependency._current_user

    assert isinstance(resolver, RoehubSessionCurrentUser)
    assert module.current_user_dependency._cookie_name == "roehub_session_id"
