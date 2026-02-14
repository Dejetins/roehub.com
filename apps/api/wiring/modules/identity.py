"""
Composition helpers for identity API module.

Docs: docs/architecture/identity/identity-telegram-login-user-model-v1.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from fastapi import APIRouter

from apps.api.routes import build_identity_router as build_identity_api_router
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.adapters.outbound import (
    Hs256JwtCodec,
    InMemoryIdentityUserRepository,
    JwtCookieCurrentUser,
    PostgresIdentityUserRepository,
    PsycopgIdentityPostgresGateway,
    SystemIdentityClock,
    TelegramLoginWidgetPayloadValidator,
)
from trading.contexts.identity.application import UserRepository
from trading.contexts.identity.application.use_cases import TelegramLoginUseCase

_ENV_NAME_KEY = "ROEHUB_ENV"
_IDENTITY_FAIL_FAST_KEY = "IDENTITY_FAIL_FAST"
_JWT_TTL_DAYS_KEY = "JWT_TTL_DAYS"
_TELEGRAM_BOT_TOKEN_KEY = "TELEGRAM_BOT_TOKEN"
_IDENTITY_JWT_SECRET_KEY = "IDENTITY_JWT_SECRET"
_IDENTITY_PG_DSN_KEY = "IDENTITY_PG_DSN"
_IDENTITY_COOKIE_NAME_KEY = "IDENTITY_COOKIE_NAME"
_IDENTITY_COOKIE_PATH_KEY = "IDENTITY_COOKIE_PATH"
_IDENTITY_COOKIE_SAMESITE_KEY = "IDENTITY_COOKIE_SAMESITE"
_IDENTITY_COOKIE_SECURE_KEY = "IDENTITY_COOKIE_SECURE"
_ALLOWED_ENVS = ("dev", "prod", "test")
_ALLOWED_SAMESITE = ("lax", "none", "strict")


@dataclass(frozen=True, slots=True)
class IdentityRuntimeSettings:
    """
    IdentityRuntimeSettings — runtime policy for identity v1 wiring.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - apps/api/wiring/modules/identity.py
      - apps/api/main/app.py
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
    """

    env_name: str
    fail_fast: bool
    jwt_ttl_days: int
    telegram_bot_token: str
    identity_jwt_secret: str
    postgres_dsn: str
    jwt_cookie_name: str
    jwt_cookie_secure: bool
    jwt_cookie_samesite: Literal["lax", "strict", "none"]
    jwt_cookie_path: str

    def __post_init__(self) -> None:
        """
        Validate identity runtime settings invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Values are normalized by resolver before dataclass construction.
        Raises:
            ValueError: If one of invariants is violated.
        Side Effects:
            None.
        """
        if self.env_name not in _ALLOWED_ENVS:
            raise ValueError(
                f"IdentityRuntimeSettings.env_name must be one of {_ALLOWED_ENVS}, "
                f"got {self.env_name!r}"
            )
        if self.jwt_ttl_days <= 0:
            raise ValueError("IdentityRuntimeSettings.jwt_ttl_days must be > 0")
        if not self.telegram_bot_token:
            raise ValueError("IdentityRuntimeSettings.telegram_bot_token must be non-empty")
        if not self.identity_jwt_secret:
            raise ValueError("IdentityRuntimeSettings.identity_jwt_secret must be non-empty")
        if not self.jwt_cookie_name:
            raise ValueError("IdentityRuntimeSettings.jwt_cookie_name must be non-empty")
        if self.jwt_cookie_samesite not in _ALLOWED_SAMESITE:
            raise ValueError(
                "IdentityRuntimeSettings.jwt_cookie_samesite must be one of "
                f"{_ALLOWED_SAMESITE}, got {self.jwt_cookie_samesite!r}"
            )
        if not self.jwt_cookie_path:
            raise ValueError("IdentityRuntimeSettings.jwt_cookie_path must be non-empty")


def build_identity_router(*, environ: Mapping[str, str]) -> APIRouter:
    """
    Build fully wired identity router from environment settings.

    Docs: docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related: apps.api.routes.identity,
      trading.contexts.identity.adapters.outbound,
      apps.api.main.app

    Args:
        environ: Runtime environment mapping.
    Returns:
        APIRouter: Identity API router with all dependencies wired.
    Assumptions:
        Fail-fast policy and secrets are resolved by `_resolve_identity_runtime_settings`.
    Raises:
        ValueError: If fail-fast settings require missing secrets or invalid values.
    Side Effects:
        None.
    """
    settings = _resolve_identity_runtime_settings(environ=environ)
    clock = SystemIdentityClock()
    user_repository = _build_user_repository(settings=settings)
    telegram_validator = TelegramLoginWidgetPayloadValidator(
        bot_token=settings.telegram_bot_token,
    )
    jwt_codec = Hs256JwtCodec(
        secret_key=settings.identity_jwt_secret,
        clock=clock,
    )

    telegram_login = TelegramLoginUseCase(
        validator=telegram_validator,
        user_repository=user_repository,
        jwt_codec=jwt_codec,
        clock=clock,
        jwt_ttl_days=settings.jwt_ttl_days,
    )

    current_user_port = JwtCookieCurrentUser(
        jwt_codec=jwt_codec,
        user_repository=user_repository,
    )
    current_user_dependency = RequireCurrentUserDependency(
        current_user=current_user_port,
        cookie_name=settings.jwt_cookie_name,
    )

    return build_identity_api_router(
        telegram_login=telegram_login,
        current_user_dependency=current_user_dependency,
        cookie_name=settings.jwt_cookie_name,
        cookie_secure=settings.jwt_cookie_secure,
        cookie_samesite=settings.jwt_cookie_samesite,
        cookie_path=settings.jwt_cookie_path,
    )



def _build_user_repository(*, settings: IdentityRuntimeSettings) -> UserRepository:
    """
    Build user repository adapter based on runtime DSN availability.

    Args:
        settings: Resolved runtime settings.
    Returns:
        UserRepository: Postgres or in-memory adapter.
    Assumptions:
        Postgres DSN is optional in dev/test, in-memory fallback is acceptable for local runs.
    Raises:
        ValueError: If Postgres DSN is malformed for gateway construction.
    Side Effects:
        None.
    """
    if settings.postgres_dsn:
        gateway = PsycopgIdentityPostgresGateway(dsn=settings.postgres_dsn)
        return PostgresIdentityUserRepository(gateway=gateway)
    return InMemoryIdentityUserRepository()



def _resolve_identity_runtime_settings(*, environ: Mapping[str, str]) -> IdentityRuntimeSettings:
    """
    Resolve identity runtime settings with fail-fast policy and defaults.

    Args:
        environ: Runtime environment mapping.
    Returns:
        IdentityRuntimeSettings: Validated normalized settings.
    Assumptions:
        Missing `ROEHUB_ENV` defaults to `dev`.
    Raises:
        ValueError: If env values are invalid or fail-fast policy requires missing secrets.
    Side Effects:
        None.
    """
    env_name = _resolve_env_name(environ=environ)
    fail_fast = _resolve_fail_fast(environ=environ, env_name=env_name)

    jwt_ttl_days = _resolve_positive_int(
        environ=environ,
        key=_JWT_TTL_DAYS_KEY,
        default=7,
    )
    telegram_bot_token = environ.get(_TELEGRAM_BOT_TOKEN_KEY, "").strip()
    identity_jwt_secret = environ.get(_IDENTITY_JWT_SECRET_KEY, "").strip()

    if fail_fast:
        if not telegram_bot_token:
            raise ValueError(
                f"{_TELEGRAM_BOT_TOKEN_KEY} must be set when {_IDENTITY_FAIL_FAST_KEY}=true"
            )
        if not identity_jwt_secret:
            raise ValueError(
                f"{_IDENTITY_JWT_SECRET_KEY} must be set when {_IDENTITY_FAIL_FAST_KEY}=true"
            )

    effective_telegram_token = telegram_bot_token or "dev-telegram-bot-token"
    effective_jwt_secret = identity_jwt_secret or "dev-identity-jwt-secret"

    postgres_dsn = environ.get(_IDENTITY_PG_DSN_KEY, "").strip()
    cookie_name = environ.get(_IDENTITY_COOKIE_NAME_KEY, "roehub_identity_jwt").strip()
    cookie_path = environ.get(_IDENTITY_COOKIE_PATH_KEY, "/").strip()
    cookie_samesite = _resolve_cookie_samesite(environ=environ)
    cookie_secure = _resolve_cookie_secure(environ=environ, env_name=env_name)

    return IdentityRuntimeSettings(
        env_name=env_name,
        fail_fast=fail_fast,
        jwt_ttl_days=jwt_ttl_days,
        telegram_bot_token=effective_telegram_token,
        identity_jwt_secret=effective_jwt_secret,
        postgres_dsn=postgres_dsn,
        jwt_cookie_name=cookie_name,
        jwt_cookie_secure=cookie_secure,
        jwt_cookie_samesite=cookie_samesite,
        jwt_cookie_path=cookie_path,
    )



def _resolve_env_name(*, environ: Mapping[str, str]) -> str:
    """
    Resolve normalized runtime env name for identity wiring.

    Args:
        environ: Runtime environment mapping.
    Returns:
        str: One of `dev`, `prod`, or `test`.
    Assumptions:
        Missing value defaults to `dev`.
    Raises:
        ValueError: If value is outside allowed list.
    Side Effects:
        None.
    """
    raw_env_name = environ.get(_ENV_NAME_KEY, "dev").strip().lower()
    if raw_env_name not in _ALLOWED_ENVS:
        raise ValueError(
            f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}, got {raw_env_name!r}"
        )
    return raw_env_name



def _resolve_fail_fast(*, environ: Mapping[str, str], env_name: str) -> bool:
    """
    Resolve fail-fast policy for identity startup validation.

    Args:
        environ: Runtime environment mapping.
        env_name: Normalized environment name.
    Returns:
        bool: Effective fail-fast flag.
    Assumptions:
        Default is enabled for `prod` and disabled for `dev`/`test`.
    Raises:
        ValueError: If override value is not parseable as boolean.
    Side Effects:
        None.
    """
    default_fail_fast = env_name == "prod"
    raw_override = environ.get(_IDENTITY_FAIL_FAST_KEY, "").strip()
    if not raw_override:
        return default_fail_fast
    return _parse_bool(raw_value=raw_override, key=_IDENTITY_FAIL_FAST_KEY)



def _resolve_cookie_secure(*, environ: Mapping[str, str], env_name: str) -> bool:
    """
    Resolve cookie secure flag from env override or environment default.

    Args:
        environ: Runtime environment mapping.
        env_name: Normalized environment name.
    Returns:
        bool: Effective secure flag.
    Assumptions:
        Secure defaults to true only in `prod`.
    Raises:
        ValueError: If override value is invalid.
    Side Effects:
        None.
    """
    raw_value = environ.get(_IDENTITY_COOKIE_SECURE_KEY, "").strip()
    if not raw_value:
        return env_name == "prod"
    return _parse_bool(raw_value=raw_value, key=_IDENTITY_COOKIE_SECURE_KEY)



def _resolve_cookie_samesite(*, environ: Mapping[str, str]) -> Literal["lax", "strict", "none"]:
    """
    Resolve cookie SameSite mode with deterministic accepted values.

    Args:
        environ: Runtime environment mapping.
    Returns:
        Literal["lax", "strict", "none"]: Effective same-site mode.
    Assumptions:
        Missing value defaults to `lax`.
    Raises:
        ValueError: If provided value is unsupported.
    Side Effects:
        None.
    """
    raw_samesite = environ.get(_IDENTITY_COOKIE_SAMESITE_KEY, "lax").strip().lower()
    if raw_samesite not in _ALLOWED_SAMESITE:
        raise ValueError(
            f"{_IDENTITY_COOKIE_SAMESITE_KEY} must be one of {_ALLOWED_SAMESITE}, "
            f"got {raw_samesite!r}"
        )
    return raw_samesite  # type: ignore[return-value]



def _resolve_positive_int(*, environ: Mapping[str, str], key: str, default: int) -> int:
    """
    Resolve positive integer env setting with fallback default.

    Args:
        environ: Runtime environment mapping.
        key: Environment variable key.
        default: Fallback integer.
    Returns:
        int: Positive integer value.
    Assumptions:
        Empty env value means default should be used.
    Raises:
        ValueError: If value is not parseable or non-positive.
    Side Effects:
        None.
    """
    raw_value = environ.get(key, "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{key} must be integer, got {raw_value!r}") from error
    if parsed <= 0:
        raise ValueError(f"{key} must be > 0, got {parsed}")
    return parsed



def _parse_bool(*, raw_value: str, key: str) -> bool:
    """
    Parse strict boolean env value from known textual literals.

    Args:
        raw_value: Raw env string value.
        key: Env key used in error messages.
    Returns:
        bool: Parsed boolean value.
    Assumptions:
        Accepted true values: `1,true,yes,on`; false values: `0,false,no,off`.
    Raises:
        ValueError: If value is not recognized.
    Side Effects:
        None.
    """
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{key} must be a boolean literal (1/0/true/false/yes/no/on/off), got {raw_value!r}"
    )
