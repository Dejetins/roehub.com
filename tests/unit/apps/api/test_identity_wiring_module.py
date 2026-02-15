import pytest

from apps.api.wiring.modules.identity import build_identity_router


def test_identity_wiring_fail_fast_in_prod_requires_identity_2fa_kek_b64() -> None:
    """
    Verify prod default fail-fast rejects startup when `IDENTITY_2FA_KEK_B64` is missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Prod environment defaults fail-fast to enabled when override is absent.
    Raises:
        AssertionError: If router wiring does not fail on missing 2FA KEK in prod.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "prod",
        "TELEGRAM_BOT_TOKEN": "prod-telegram-token",
        "IDENTITY_JWT_SECRET": "prod-jwt-secret",
    }

    with pytest.raises(ValueError, match="IDENTITY_2FA_KEK_B64"):
        build_identity_router(environ=environ)


def test_identity_wiring_fail_fast_in_prod_requires_exchange_keys_kek_b64() -> None:
    """
    Verify prod default fail-fast rejects startup when `IDENTITY_EXCHANGE_KEYS_KEK_B64` is missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        2FA KEK is set so failure must be specifically about exchange keys KEK.
    Raises:
        AssertionError: If router wiring does not fail on missing exchange keys KEK in prod.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "prod",
        "TELEGRAM_BOT_TOKEN": "prod-telegram-token",
        "IDENTITY_JWT_SECRET": "prod-jwt-secret",
        "IDENTITY_2FA_KEK_B64": "cm9laHViLWRldi1pZGVudGl0eS0yZmEta2V5LTAwMDE=",
    }

    with pytest.raises(ValueError, match="IDENTITY_EXCHANGE_KEYS_KEK_B64"):
        build_identity_router(environ=environ)


def test_identity_wiring_fail_fast_override_requires_identity_2fa_kek_b64() -> None:
    """
    Verify explicit `IDENTITY_FAIL_FAST=true` requires 2FA KEK even in non-prod environments.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Fail-fast override should enforce strict secret presence checks in any env.
    Raises:
        AssertionError: If router wiring does not fail on missing 2FA KEK with override enabled.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "dev",
        "IDENTITY_FAIL_FAST": "true",
        "TELEGRAM_BOT_TOKEN": "dev-telegram-token",
        "IDENTITY_JWT_SECRET": "dev-jwt-secret",
    }

    with pytest.raises(ValueError, match="IDENTITY_2FA_KEK_B64"):
        build_identity_router(environ=environ)


def test_identity_wiring_fail_fast_override_requires_exchange_keys_kek_b64() -> None:
    """
    Verify explicit `IDENTITY_FAIL_FAST=true` requires exchange keys KEK in non-prod env too.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        2FA KEK is set so fail-fast validation reaches exchange keys KEK check.
    Raises:
        AssertionError: If router wiring does not fail on missing exchange keys KEK.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "dev",
        "IDENTITY_FAIL_FAST": "true",
        "TELEGRAM_BOT_TOKEN": "dev-telegram-token",
        "IDENTITY_JWT_SECRET": "dev-jwt-secret",
        "IDENTITY_2FA_KEK_B64": "cm9laHViLWRldi1pZGVudGl0eS0yZmEta2V5LTAwMDE=",
    }

    with pytest.raises(ValueError, match="IDENTITY_EXCHANGE_KEYS_KEK_B64"):
        build_identity_router(environ=environ)


def test_identity_wiring_dev_defaults_allow_missing_keks_and_exposes_identity_routes() -> None:
    """
    Verify dev defaults allow missing KEKs and wire both 2FA and exchange-keys endpoints.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Dev fallback KEKs are acceptable when fail-fast is not explicitly enabled.
    Raises:
        AssertionError: If critical identity routes are missing or wiring unexpectedly fails.
    Side Effects:
        None.
    """
    router = build_identity_router(environ={"ROEHUB_ENV": "dev"})
    paths = sorted(
        str(getattr(route, "path"))
        for route in router.routes
        if hasattr(route, "path")
    )

    assert "/2fa/setup" in paths
    assert "/2fa/verify" in paths
    assert "/exchange-keys" in paths
    assert "/exchange-keys/{key_id}" in paths
