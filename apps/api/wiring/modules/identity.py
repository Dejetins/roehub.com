"""
Composition helpers for identity API module.

Docs:
  - docs/architecture/identity/keycloak-cutover-plan-v1.md
  - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from fastapi import APIRouter

from apps.api.routes import build_identity_router as build_identity_api_router
from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.adapters.outbound import (
    AesGcmEnvelopeExchangeKeysSecretCipher,
    InMemoryIdentityExchangeKeysRepository,
    InMemoryIdentitySessionRepository,
    InMemoryIdentityUserRepository,
    PostgresIdentityExchangeKeysRepository,
    PostgresIdentitySessionRepository,
    PostgresIdentityUserRepository,
    PsycopgIdentityPostgresGateway,
    RoehubSessionCurrentUser,
    SystemIdentityClock,
)
from trading.contexts.identity.application import (
    ExchangeKeysRepository,
    SessionRepository,
    UserRepository,
)
from trading.contexts.identity.application.use_cases import (
    CreateExchangeKeyUseCase,
    DeleteExchangeKeyUseCase,
    ListExchangeKeysUseCase,
)

_ENV_NAME_KEY = "ROEHUB_ENV"
_IDENTITY_FAIL_FAST_KEY = "IDENTITY_FAIL_FAST"
_IDENTITY_EXCHANGE_KEYS_KEK_B64_KEY = "IDENTITY_EXCHANGE_KEYS_KEK_B64"
_IDENTITY_PG_DSN_KEY = "IDENTITY_PG_DSN"
_IDENTITY_SESSION_COOKIE_NAME_KEY = "IDENTITY_SESSION_COOKIE_NAME"
_IDENTITY_SESSION_IDLE_TTL_SECONDS_KEY = "IDENTITY_SESSION_IDLE_TTL_SECONDS"
_IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS_KEY = "IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS"
_KEYCLOAK_BASE_URL_KEY = "KEYCLOAK_BASE_URL"
_KEYCLOAK_REALM_KEY = "KEYCLOAK_REALM"
_KEYCLOAK_CLIENT_ID_KEY = "KEYCLOAK_CLIENT_ID"
_KEYCLOAK_CLIENT_SECRET_KEY = "KEYCLOAK_CLIENT_SECRET"
_KEYCLOAK_REDIRECT_URI_KEY = "KEYCLOAK_REDIRECT_URI"
_KEYCLOAK_LOGOUT_REDIRECT_URI_KEY = "KEYCLOAK_LOGOUT_REDIRECT_URI"
_KEYCLOAK_AUTH_URL_KEY = "KEYCLOAK_AUTH_URL"
_KEYCLOAK_TOKEN_URL_KEY = "KEYCLOAK_TOKEN_URL"
_KEYCLOAK_END_SESSION_URL_KEY = "KEYCLOAK_END_SESSION_URL"
_KEYCLOAK_INTROSPECTION_URL_KEY = "KEYCLOAK_INTROSPECTION_URL"
_DEFAULT_DEV_IDENTITY_EXCHANGE_KEYS_KEK_B64 = "cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE="
_DEFAULT_DEV_KEYCLOAK_BASE_URL = "http://127.0.0.1:18080"
_DEFAULT_DEV_KEYCLOAK_REALM = "roehub"
_DEFAULT_DEV_KEYCLOAK_CLIENT_ID = "roehub-api"
_DEFAULT_DEV_KEYCLOAK_CLIENT_SECRET = "dev-keycloak-client-secret"
_DEFAULT_DEV_KEYCLOAK_REDIRECT_URI = "http://127.0.0.1:8010/auth/callback"
_DEFAULT_DEV_KEYCLOAK_LOGOUT_REDIRECT_URI = "http://127.0.0.1:8010/login"
_DEFAULT_IDENTITY_SESSION_COOKIE_NAME = "roehub_session_id"
_DEFAULT_IDENTITY_SESSION_IDLE_TTL_SECONDS = 1800
_DEFAULT_IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS = 43200
_LEGACY_AUTH_COOKIE_PATH = "/"
_LEGACY_AUTH_COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"
_ALLOWED_ENVS = ("dev", "prod", "test")


@dataclass(frozen=True, slots=True)
class IdentityRuntimeSettings:
    """
    IdentityRuntimeSettings — runtime policy for identity module wiring.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
    Related:
      - apps/api/wiring/modules/identity.py
      - apps/api/main/app.py
      - apps/api/routes/identity.py
    """

    env_name: str
    fail_fast: bool
    keycloak_base_url: str
    keycloak_realm: str
    keycloak_client_id: str
    keycloak_client_secret: str
    keycloak_redirect_uri: str
    keycloak_logout_redirect_uri: str
    keycloak_auth_url: str
    keycloak_token_url: str
    keycloak_end_session_url: str
    keycloak_introspection_url: str
    identity_session_cookie_name: str
    identity_session_idle_ttl_seconds: int
    identity_session_absolute_ttl_seconds: int
    identity_exchange_keys_kek_b64: str
    postgres_dsn: str

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
        if not self.keycloak_base_url:
            raise ValueError("IdentityRuntimeSettings.keycloak_base_url must be non-empty")
        if not self.keycloak_realm:
            raise ValueError("IdentityRuntimeSettings.keycloak_realm must be non-empty")
        if not self.keycloak_client_id:
            raise ValueError("IdentityRuntimeSettings.keycloak_client_id must be non-empty")
        if not self.keycloak_client_secret:
            raise ValueError("IdentityRuntimeSettings.keycloak_client_secret must be non-empty")
        if not self.keycloak_redirect_uri:
            raise ValueError("IdentityRuntimeSettings.keycloak_redirect_uri must be non-empty")
        if not self.keycloak_logout_redirect_uri:
            raise ValueError(
                "IdentityRuntimeSettings.keycloak_logout_redirect_uri must be non-empty"
            )
        if not self.keycloak_auth_url:
            raise ValueError("IdentityRuntimeSettings.keycloak_auth_url must be non-empty")
        if not self.keycloak_token_url:
            raise ValueError("IdentityRuntimeSettings.keycloak_token_url must be non-empty")
        if not self.keycloak_end_session_url:
            raise ValueError(
                "IdentityRuntimeSettings.keycloak_end_session_url must be non-empty"
            )
        if not self.keycloak_introspection_url:
            raise ValueError(
                "IdentityRuntimeSettings.keycloak_introspection_url must be non-empty"
            )
        if not self.identity_session_cookie_name:
            raise ValueError(
                "IdentityRuntimeSettings.identity_session_cookie_name must be non-empty"
            )
        if self.identity_session_idle_ttl_seconds <= 0:
            raise ValueError(
                "IdentityRuntimeSettings.identity_session_idle_ttl_seconds must be > 0"
            )
        if self.identity_session_absolute_ttl_seconds <= 0:
            raise ValueError(
                "IdentityRuntimeSettings.identity_session_absolute_ttl_seconds must be > 0"
            )
        if self.identity_session_absolute_ttl_seconds < self.identity_session_idle_ttl_seconds:
            raise ValueError(
                "IdentityRuntimeSettings.identity_session_absolute_ttl_seconds must be >= "
                "identity_session_idle_ttl_seconds"
            )
        if not self.identity_exchange_keys_kek_b64:
            raise ValueError(
                "IdentityRuntimeSettings.identity_exchange_keys_kek_b64 must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class IdentityApiModule:
    """
    IdentityApiModule — bundled identity router and shared current-user dependency.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
    Related:
      - apps/api/wiring/modules/identity.py
      - apps/api/wiring/modules/strategy.py
      - apps/api/main/app.py
    """

    router: APIRouter
    current_user_dependency: RequireCurrentUserDependency


@dataclass(frozen=True, slots=True)
class _IdentityPersistenceBundle:
    """
    _IdentityPersistenceBundle groups identity repositories built from one storage policy.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
    Related:
      - apps/api/wiring/modules/identity.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/
      - src/trading/contexts/identity/adapters/outbound/persistence/in_memory/
    """

    exchange_keys_repository: ExchangeKeysRepository
    user_repository: UserRepository
    session_repository: SessionRepository


def build_identity_router(*, environ: Mapping[str, str]) -> APIRouter:
    """
    Build fully wired identity router from environment settings.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
    Related:
      - apps/api/routes/identity.py
      - trading.contexts.identity.adapters.outbound
      - apps/api/main/app.py

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
    return build_identity_api_module(environ=environ).router


def build_identity_api_module(*, environ: Mapping[str, str]) -> IdentityApiModule:
    """
    Build bundled identity API module with router and reusable current-user dependency.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - apps/api/wiring/modules/identity.py
      - apps/api/wiring/modules/strategy.py
      - apps/api/main/app.py

    Args:
        environ: Runtime environment mapping.
    Returns:
        IdentityApiModule: Identity router and shared current-user dependency.
    Assumptions:
        Shared dependency is used by both identity and strategy routes in one FastAPI app.
    Raises:
        ValueError: If fail-fast settings require missing secrets or invalid values.
    Side Effects:
        None.
    """
    settings = _resolve_identity_runtime_settings(environ=environ)
    clock = SystemIdentityClock()
    persistence = _build_identity_persistence(settings=settings)

    exchange_keys_secret_cipher = AesGcmEnvelopeExchangeKeysSecretCipher(
        kek_b64=settings.identity_exchange_keys_kek_b64,
    )
    create_exchange_key_use_case = CreateExchangeKeyUseCase(
        repository=persistence.exchange_keys_repository,
        secret_cipher=exchange_keys_secret_cipher,
        clock=clock,
    )
    list_exchange_keys_use_case = ListExchangeKeysUseCase(
        repository=persistence.exchange_keys_repository
    )
    delete_exchange_key_use_case = DeleteExchangeKeyUseCase(
        repository=persistence.exchange_keys_repository,
        clock=clock,
    )

    current_user_port = RoehubSessionCurrentUser(
        session_repository=persistence.session_repository,
        user_repository=persistence.user_repository,
        clock=clock,
    )
    current_user_dependency = RequireCurrentUserDependency(
        current_user=current_user_port,
        cookie_name=settings.identity_session_cookie_name,
    )

    return IdentityApiModule(
        router=build_identity_api_router(
            keycloak_auth_url=settings.keycloak_auth_url,
            keycloak_token_url=settings.keycloak_token_url,
            keycloak_introspection_url=settings.keycloak_introspection_url,
            keycloak_client_id=settings.keycloak_client_id,
            keycloak_client_secret=settings.keycloak_client_secret,
            keycloak_redirect_uri=settings.keycloak_redirect_uri,
            keycloak_logout_redirect_uri=settings.keycloak_logout_redirect_uri,
            current_user_dependency=current_user_dependency,
            user_repository=persistence.user_repository,
            session_repository=persistence.session_repository,
            clock=clock,
            cookie_name=settings.identity_session_cookie_name,
            cookie_secure=settings.env_name == "prod",
            session_idle_ttl_seconds=settings.identity_session_idle_ttl_seconds,
            session_absolute_ttl_seconds=settings.identity_session_absolute_ttl_seconds,
            cookie_samesite=_LEGACY_AUTH_COOKIE_SAMESITE,
            cookie_path=_LEGACY_AUTH_COOKIE_PATH,
            create_exchange_key_use_case=create_exchange_key_use_case,
            list_exchange_keys_use_case=list_exchange_keys_use_case,
            delete_exchange_key_use_case=delete_exchange_key_use_case,
        ),
        current_user_dependency=current_user_dependency,
    )

def _build_identity_persistence(*, settings: IdentityRuntimeSettings) -> _IdentityPersistenceBundle:
    """
    Build identity repositories according to runtime storage policy.

    Args:
        settings: Resolved runtime settings.
    Returns:
        _IdentityPersistenceBundle: Exchange-keys, user, and session repositories.
    Assumptions:
        Prod runtime must use persisted Postgres-backed session storage.
    Raises:
        ValueError: If prod runtime lacks Postgres DSN or gateway construction fails.
    Side Effects:
        None.
    """
    if settings.postgres_dsn:
        gateway = PsycopgIdentityPostgresGateway(dsn=settings.postgres_dsn)
        return _IdentityPersistenceBundle(
            exchange_keys_repository=PostgresIdentityExchangeKeysRepository(gateway=gateway),
            user_repository=PostgresIdentityUserRepository(gateway=gateway),
            session_repository=PostgresIdentitySessionRepository(gateway=gateway),
        )
    if settings.env_name == "prod":
        raise ValueError(
            f"{_IDENTITY_PG_DSN_KEY} must be set in prod for persisted Roehub sessions"
        )
    return _IdentityPersistenceBundle(
        exchange_keys_repository=InMemoryIdentityExchangeKeysRepository(),
        user_repository=InMemoryIdentityUserRepository(),
        session_repository=InMemoryIdentitySessionRepository(),
    )



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
        ValueError: If env values are invalid or fail-fast requires missing secrets.
    Side Effects:
        None.
    """
    env_name = _resolve_env_name(environ=environ)
    fail_fast = _resolve_fail_fast(environ=environ, env_name=env_name)

    keycloak_base_url = _normalize_base_url(
        raw_base_url=environ.get(_KEYCLOAK_BASE_URL_KEY, "").strip()
    )
    keycloak_realm = environ.get(_KEYCLOAK_REALM_KEY, "").strip()
    keycloak_client_id = environ.get(_KEYCLOAK_CLIENT_ID_KEY, "").strip()
    keycloak_client_secret = environ.get(_KEYCLOAK_CLIENT_SECRET_KEY, "").strip()
    keycloak_redirect_uri = environ.get(_KEYCLOAK_REDIRECT_URI_KEY, "").strip()
    keycloak_logout_redirect_uri = environ.get(_KEYCLOAK_LOGOUT_REDIRECT_URI_KEY, "").strip()
    keycloak_auth_url = environ.get(_KEYCLOAK_AUTH_URL_KEY, "").strip()
    keycloak_token_url = environ.get(_KEYCLOAK_TOKEN_URL_KEY, "").strip()
    keycloak_end_session_url = environ.get(_KEYCLOAK_END_SESSION_URL_KEY, "").strip()
    keycloak_introspection_url = environ.get(_KEYCLOAK_INTROSPECTION_URL_KEY, "").strip()
    identity_session_cookie_name = environ.get(_IDENTITY_SESSION_COOKIE_NAME_KEY, "").strip()
    identity_session_idle_ttl_seconds = _read_optional_positive_int(
        raw_value=environ.get(_IDENTITY_SESSION_IDLE_TTL_SECONDS_KEY, "").strip(),
        key=_IDENTITY_SESSION_IDLE_TTL_SECONDS_KEY,
    )
    identity_session_absolute_ttl_seconds = _read_optional_positive_int(
        raw_value=environ.get(_IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS_KEY, "").strip(),
        key=_IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS_KEY,
    )
    identity_exchange_keys_kek_b64 = environ.get(_IDENTITY_EXCHANGE_KEYS_KEK_B64_KEY, "").strip()

    if fail_fast:
        if not keycloak_base_url:
            raise ValueError(
                f"{_KEYCLOAK_BASE_URL_KEY} must be set when {_IDENTITY_FAIL_FAST_KEY}=true"
            )
        if not keycloak_realm:
            raise ValueError(
                f"{_KEYCLOAK_REALM_KEY} must be set when {_IDENTITY_FAIL_FAST_KEY}=true"
            )
        if not keycloak_client_id:
            raise ValueError(
                f"{_KEYCLOAK_CLIENT_ID_KEY} must be set when {_IDENTITY_FAIL_FAST_KEY}=true"
            )
        if not keycloak_client_secret:
            raise ValueError(
                f"{_KEYCLOAK_CLIENT_SECRET_KEY} must be set when {_IDENTITY_FAIL_FAST_KEY}=true"
            )
        if not keycloak_redirect_uri:
            raise ValueError(
                f"{_KEYCLOAK_REDIRECT_URI_KEY} must be set when {_IDENTITY_FAIL_FAST_KEY}=true"
            )
        if not keycloak_logout_redirect_uri:
            raise ValueError(
                f"{_KEYCLOAK_LOGOUT_REDIRECT_URI_KEY} must be set when "
                f"{_IDENTITY_FAIL_FAST_KEY}=true"
            )
        if identity_session_idle_ttl_seconds is None:
            raise ValueError(
                f"{_IDENTITY_SESSION_IDLE_TTL_SECONDS_KEY} must be set when "
                f"{_IDENTITY_FAIL_FAST_KEY}=true"
            )
        if identity_session_absolute_ttl_seconds is None:
            raise ValueError(
                f"{_IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS_KEY} must be set when "
                f"{_IDENTITY_FAIL_FAST_KEY}=true"
            )
        if not identity_exchange_keys_kek_b64:
            raise ValueError(
                f"{_IDENTITY_EXCHANGE_KEYS_KEK_B64_KEY} must be set when "
                f"{_IDENTITY_FAIL_FAST_KEY}=true"
            )

    effective_keycloak_base_url = keycloak_base_url or _DEFAULT_DEV_KEYCLOAK_BASE_URL
    effective_keycloak_realm = keycloak_realm or _DEFAULT_DEV_KEYCLOAK_REALM
    effective_keycloak_client_id = keycloak_client_id or _DEFAULT_DEV_KEYCLOAK_CLIENT_ID
    effective_keycloak_client_secret = (
        keycloak_client_secret or _DEFAULT_DEV_KEYCLOAK_CLIENT_SECRET
    )
    effective_keycloak_redirect_uri = keycloak_redirect_uri or _DEFAULT_DEV_KEYCLOAK_REDIRECT_URI
    effective_keycloak_logout_redirect_uri = (
        keycloak_logout_redirect_uri or _DEFAULT_DEV_KEYCLOAK_LOGOUT_REDIRECT_URI
    )
    effective_keycloak_auth_url = keycloak_auth_url or _build_keycloak_oidc_endpoint(
        base_url=effective_keycloak_base_url,
        realm=effective_keycloak_realm,
        suffix="auth",
    )
    effective_keycloak_token_url = keycloak_token_url or _build_keycloak_oidc_endpoint(
        base_url=effective_keycloak_base_url,
        realm=effective_keycloak_realm,
        suffix="token",
    )
    effective_keycloak_end_session_url = (
        keycloak_end_session_url
        or _build_keycloak_oidc_endpoint(
            base_url=effective_keycloak_base_url,
            realm=effective_keycloak_realm,
            suffix="logout",
        )
    )
    effective_keycloak_introspection_url = (
        keycloak_introspection_url
        or _build_keycloak_oidc_endpoint(
            base_url=effective_keycloak_base_url,
            realm=effective_keycloak_realm,
            suffix="token/introspect",
        )
    )
    effective_identity_session_cookie_name = (
        identity_session_cookie_name or _DEFAULT_IDENTITY_SESSION_COOKIE_NAME
    )
    effective_identity_session_idle_ttl_seconds = (
        identity_session_idle_ttl_seconds or _DEFAULT_IDENTITY_SESSION_IDLE_TTL_SECONDS
    )
    effective_identity_session_absolute_ttl_seconds = (
        identity_session_absolute_ttl_seconds or _DEFAULT_IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS
    )
    effective_exchange_keys_kek_b64 = (
        identity_exchange_keys_kek_b64 or _DEFAULT_DEV_IDENTITY_EXCHANGE_KEYS_KEK_B64
    )

    postgres_dsn = environ.get(_IDENTITY_PG_DSN_KEY, "").strip()

    return IdentityRuntimeSettings(
        env_name=env_name,
        fail_fast=fail_fast,
        keycloak_base_url=effective_keycloak_base_url,
        keycloak_realm=effective_keycloak_realm,
        keycloak_client_id=effective_keycloak_client_id,
        keycloak_client_secret=effective_keycloak_client_secret,
        keycloak_redirect_uri=effective_keycloak_redirect_uri,
        keycloak_logout_redirect_uri=effective_keycloak_logout_redirect_uri,
        keycloak_auth_url=effective_keycloak_auth_url,
        keycloak_token_url=effective_keycloak_token_url,
        keycloak_end_session_url=effective_keycloak_end_session_url,
        keycloak_introspection_url=effective_keycloak_introspection_url,
        identity_session_cookie_name=effective_identity_session_cookie_name,
        identity_session_idle_ttl_seconds=effective_identity_session_idle_ttl_seconds,
        identity_session_absolute_ttl_seconds=effective_identity_session_absolute_ttl_seconds,
        identity_exchange_keys_kek_b64=effective_exchange_keys_kek_b64,
        postgres_dsn=postgres_dsn,
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



def _normalize_base_url(*, raw_base_url: str) -> str:
    """
    Normalize Keycloak base URL by trimming whitespace and trailing slash.

    Args:
        raw_base_url: Raw base URL value from environment mapping.
    Returns:
        str: Normalized base URL without trailing slash.
    Assumptions:
        Empty input is allowed and returned as empty string.
    Raises:
        None.
    Side Effects:
        None.
    """
    return raw_base_url.strip().rstrip("/")



def _build_keycloak_oidc_endpoint(*, base_url: str, realm: str, suffix: str) -> str:
    """
    Build deterministic Keycloak OIDC endpoint URL from base/realm/suffix.

    Args:
        base_url: Normalized Keycloak server base URL.
        realm: Keycloak realm name.
        suffix: OIDC endpoint suffix inside `protocol/openid-connect`.
    Returns:
        str: Fully qualified endpoint URL.
    Assumptions:
        `base_url`, `realm`, and `suffix` are non-empty normalized strings.
    Raises:
        ValueError: If one of input parts is empty.
    Side Effects:
        None.
    """
    normalized_base_url = base_url.strip().rstrip("/")
    normalized_realm = realm.strip().strip("/")
    normalized_suffix = suffix.strip().strip("/")
    if not normalized_base_url:
        raise ValueError("_build_keycloak_oidc_endpoint requires non-empty base_url")
    if not normalized_realm:
        raise ValueError("_build_keycloak_oidc_endpoint requires non-empty realm")
    if not normalized_suffix:
        raise ValueError("_build_keycloak_oidc_endpoint requires non-empty suffix")
    return (
        f"{normalized_base_url}/realms/{normalized_realm}/protocol/openid-connect/"
        f"{normalized_suffix}"
    )



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


def _read_optional_positive_int(*, raw_value: str, key: str) -> int | None:
    """
    Parse optional positive integer env value.

    Args:
        raw_value: Raw env string value.
        key: Env key used in deterministic error messages.
    Returns:
        int | None: Parsed positive integer or `None` when value is blank.
    Assumptions:
        Blank values mean "use resolver default" outside fail-fast environments.
    Raises:
        ValueError: If value is non-numeric or not strictly positive.
    Side Effects:
        None.
    """
    normalized = raw_value.strip()
    if not normalized:
        return None
    try:
        parsed_value = int(normalized)
    except ValueError as error:
        raise ValueError(f"{key} must be a positive integer, got {raw_value!r}") from error
    if parsed_value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {raw_value!r}")
    return parsed_value
