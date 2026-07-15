from pathlib import Path

import pytest
from fastapi.routing import APIRoute

from apps.api.wiring.modules.identity import build_identity_api_module, build_identity_router
from trading.contexts.identity.adapters.outbound.security.current_user import (
    RoehubSessionCurrentUser,
)
from trading.platform.secrets import OpenBaoSecretResolver, SecretValue

_DEV_ONLY_EXCHANGE_KEYS_KEK_B64 = "cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE="
_PROD_TEST_EXCHANGE_KEYS_KEK_B64 = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="


def test_oidc_partial_configuration_requires_provider_id() -> None:
    environ = _oidc_env()
    del environ["IDENTITY_OIDC_PROVIDER_ID"]

    with pytest.raises(ValueError, match="IDENTITY_OIDC_PROVIDER_ID"):
        build_identity_router(environ=environ)


def test_oidc_partial_configuration_requires_client_credential() -> None:
    environ = _oidc_env()
    del environ["IDENTITY_OIDC_CLIENT_SECRET_REF"]

    with pytest.raises(ValueError, match="IDENTITY_OIDC_CLIENT_SECRET_REF"):
        build_identity_router(environ=environ)


def test_oidc_timeout_budgets_can_tighten_but_not_expand() -> None:
    environ = _oidc_env()
    environ["IDENTITY_OIDC_OVERALL_TIMEOUT_SECONDS"] = "15.01"

    with pytest.raises(ValueError, match="overall_timeout_seconds"):
        build_identity_router(environ=environ)


def test_identity_wiring_fail_fast_override_requires_exchange_keys_kek_b64() -> None:
    environ = {
        "ROEHUB_ENV": "dev",
        "IDENTITY_FAIL_FAST": "true",
        "IDENTITY_SESSION_IDLE_TTL_SECONDS": "1800",
        "IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS": "43200",
        "IDENTITY_LOCAL_RP_ID": "localhost",
        "IDENTITY_LOCAL_RP_NAME": "Roehub",
        "IDENTITY_LOCAL_ORIGIN": "http://localhost:8000",
    }

    with pytest.raises(ValueError, match="IDENTITY_EXCHANGE_KEYS_KEK_B64"):
        build_identity_router(environ=environ)


def test_identity_wiring_exposes_oidc_and_local_fallback_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = _oidc_env()
    environ.update(
        {
            "IDENTITY_FAIL_FAST": "true",
            "IDENTITY_SESSION_IDLE_TTL_SECONDS": "1800",
            "IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS": "43200",
            "IDENTITY_EXCHANGE_KEYS_KEK_B64": _DEV_ONLY_EXCHANGE_KEYS_KEK_B64,
            "IDENTITY_LOCAL_RP_ID": "localhost",
            "IDENTITY_LOCAL_RP_NAME": "Roehub",
            "IDENTITY_LOCAL_ORIGIN": "http://localhost:8000",
        }
    )
    monkeypatch.setattr(
        OpenBaoSecretResolver,
        "resolve",
        lambda *_args, **_kwargs: SecretValue(_text="fixture-value", version=1),
    )

    paths = {
        route.path
        for route in build_identity_router(environ=environ).routes
        if isinstance(route, APIRoute)
    }

    assert "/auth/oidc/login" in paths
    assert "/auth/oidc/callback" in paths
    assert "/auth/oidc/link" in paths
    assert "/auth/local/passkey/options" in paths


def test_identity_wiring_fail_fast_in_prod_requires_session_idle_ttl() -> None:
    environ = _prod_env()
    del environ["IDENTITY_SESSION_IDLE_TTL_SECONDS"]

    with pytest.raises(ValueError, match="IDENTITY_SESSION_IDLE_TTL_SECONDS"):
        build_identity_router(environ=environ)


def test_identity_wiring_fail_fast_in_prod_requires_session_absolute_ttl() -> None:
    environ = _prod_env()
    del environ["IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS"]

    with pytest.raises(ValueError, match="IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS"):
        build_identity_router(environ=environ)


def test_identity_wiring_fail_fast_in_prod_requires_postgres_dsn() -> None:
    environ = _prod_env()
    del environ["IDENTITY_PG_DSN"]

    with pytest.raises(ValueError, match="IDENTITY_PG_DSN"):
        build_identity_router(environ=environ)


def test_identity_wiring_prod_rejects_dev_only_exchange_keys_kek() -> None:
    environ = _prod_env()
    environ["IDENTITY_EXCHANGE_KEYS_KEK_B64"] = _DEV_ONLY_EXCHANGE_KEYS_KEK_B64

    with pytest.raises(ValueError, match="dev-only KEK"):
        build_identity_router(environ=environ)


def test_identity_wiring_prod_reads_exchange_keys_kek_from_secure_file(
    tmp_path: Path,
) -> None:
    kek_path = tmp_path / "identity-exchange-kek"
    kek_path.write_text(_PROD_TEST_EXCHANGE_KEYS_KEK_B64, encoding="utf-8")
    kek_path.chmod(0o600)
    environ = _prod_env()
    del environ["IDENTITY_EXCHANGE_KEYS_KEK_B64"]
    environ["IDENTITY_EXCHANGE_KEYS_KEK_B64_FILE"] = str(kek_path.resolve())

    build_identity_router(environ=environ)


def test_identity_wiring_rejects_ambiguous_exchange_keys_kek_sources(
    tmp_path: Path,
) -> None:
    kek_path = tmp_path / "identity-exchange-kek"
    kek_path.write_text(_PROD_TEST_EXCHANGE_KEYS_KEK_B64, encoding="utf-8")
    kek_path.chmod(0o600)
    environ = _prod_env()
    environ["IDENTITY_EXCHANGE_KEYS_KEK_B64_FILE"] = str(kek_path.resolve())

    with pytest.raises(ValueError, match="set only one"):
        build_identity_router(environ=environ)


def test_identity_wiring_dev_defaults_expose_only_local_auth() -> None:
    paths = {
        route.path
        for route in build_identity_router(environ={"ROEHUB_ENV": "dev"}).routes
        if isinstance(route, APIRoute)
    }

    assert "/auth/oidc/login" not in paths
    assert "/auth/oidc/callback" not in paths
    assert "/auth/local/status" in paths
    assert "/auth/local/recovery" in paths
    assert "/auth/local/logout" in paths
    assert "/auth/current-user" in paths


def test_identity_wiring_prod_local_auth_does_not_require_oidc() -> None:
    paths = {
        route.path
        for route in build_identity_router(environ=_prod_env()).routes
        if isinstance(route, APIRoute)
    }

    assert "/auth/local/status" in paths
    assert "/auth/oidc/login" not in paths
    assert "/auth/current-user" in paths


def test_identity_wiring_prod_allows_explicit_http_localhost() -> None:
    environ = _prod_env()
    environ.update(
        {
            "IDENTITY_LOCAL_RP_ID": "localhost",
            "IDENTITY_LOCAL_ORIGIN": "http://localhost:8080",
            "IDENTITY_LOCAL_ALLOW_INSECURE_LOCALHOST": "true",
        }
    )

    build_identity_router(environ=environ)


def test_identity_wiring_insecure_localhost_override_rejects_nonlocal_origin() -> None:
    environ = _prod_env()
    environ["IDENTITY_LOCAL_ALLOW_INSECURE_LOCALHOST"] = "true"

    with pytest.raises(ValueError, match="exact http://localhost"):
        build_identity_router(environ=environ)


def test_identity_wiring_uses_roehub_session_current_user_resolver() -> None:
    module = build_identity_api_module(environ={"ROEHUB_ENV": "dev"})

    resolver = module.current_user_dependency._current_user

    assert isinstance(resolver, RoehubSessionCurrentUser)
    assert module.current_user_dependency._cookie_name == "roehub_session_id"


def _oidc_env() -> dict[str, str]:
    return {
        "ROEHUB_ENV": "dev",
        "IDENTITY_OIDC_PROVIDER_ID": "fixture",
        "IDENTITY_OIDC_DISPLAY_NAME": "Fixture Identity",
        "IDENTITY_OIDC_ISSUER": "http://localhost:9010",
        "IDENTITY_OIDC_CLIENT_ID": "roehub-browser",
        "IDENTITY_OIDC_CLIENT_SECRET_REF": ("openbao://kv/roehub/oidc/fixture#client_secret"),
        "IDENTITY_OIDC_REDIRECT_URI": "http://localhost:8000/api/auth/oidc/callback",
        "IDENTITY_OIDC_ALLOW_INSECURE_HTTP": "true",
        "OPENBAO_ADDR": "http://127.0.0.1:8200",
        "ROEHUB_IDENTITY_OPENBAO_TOKEN_FILE": "/tmp/roehub-test-openbao-identity",
        "ROEHUB_OPENBAO_ROOT": "kv/roehub",
    }


def _prod_env() -> dict[str, str]:
    return {
        "ROEHUB_ENV": "prod",
        "IDENTITY_SESSION_IDLE_TTL_SECONDS": "1800",
        "IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS": "43200",
        "IDENTITY_EXCHANGE_KEYS_KEK_B64": _PROD_TEST_EXCHANGE_KEYS_KEK_B64,
        "IDENTITY_PG_DSN": "postgresql://localhost/roehub",
        "IDENTITY_LOCAL_RP_ID": "roehub.example",
        "IDENTITY_LOCAL_RP_NAME": "Roehub",
        "IDENTITY_LOCAL_ORIGIN": "https://roehub.example",
    }
