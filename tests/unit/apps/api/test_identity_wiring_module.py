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
        AssertionError: If router wiring does not fail on missing KEK in prod.
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


def test_identity_wiring_fail_fast_override_requires_identity_2fa_kek_b64() -> None:
    """
    Verify explicit `IDENTITY_FAIL_FAST=true` requires KEK even in non-prod environments.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Fail-fast override should enforce strict secret presence checks in any env.
    Raises:
        AssertionError: If router wiring does not fail on missing KEK with override enabled.
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


def test_identity_wiring_dev_defaults_allow_missing_kek_and_exposes_two_factor_routes() -> None:
    """
    Verify dev defaults allow missing KEK and still wire `/2fa/setup` and `/2fa/verify` routes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Dev fallback KEK is acceptable when fail-fast is not explicitly enabled.
    Raises:
        AssertionError: If routes are missing or wiring unexpectedly fails.
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
