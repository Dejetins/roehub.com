"""
Composition helpers for identity API module.

Docs:
  - docs/architecture/identity/oidc-authentication-provider-v1.md
  - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping
from urllib.parse import urlparse

from fastapi import APIRouter

from apps.api.monitoring import PrometheusOidcProviderMetrics
from apps.api.routes import build_identity_router as build_identity_api_router
from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.adapters.outbound import (
    AesGcmEnvelopeExchangeKeysSecretCipher,
    HttpOidcAuthenticationProvider,
    InMemoryAccountSettingsRepository,
    InMemoryIdentityExchangeKeysRepository,
    InMemoryIdentitySessionRepository,
    InMemoryIdentityUserRepository,
    InMemoryLocalAuthRepository,
    InMemoryOidcIdentityRepository,
    InMemoryOrganizationRepository,
    PostgresAccountSettingsRepository,
    PostgresIdentityExchangeKeysRepository,
    PostgresIdentitySessionRepository,
    PostgresIdentityUserRepository,
    PostgresLocalAuthRepository,
    PostgresOidcIdentityRepository,
    PostgresOrganizationRepository,
    PsycopgIdentityPostgresGateway,
    RoehubSessionCurrentUser,
    SystemIdentityClock,
)
from trading.contexts.identity.application import (
    AccountSettingsRepository,
    ExchangeKeysRepository,
    IdentityClock,
    LocalAuthRepository,
    OidcIdentityRepository,
    OrganizationRepository,
    SessionRepository,
    UserRepository,
)
from trading.contexts.identity.application.use_cases import (
    CreateExchangeKeyUseCase,
    DeleteExchangeKeyUseCase,
    ListExchangeKeysUseCase,
    LocalAuthService,
    OidcAuthenticationService,
    OrganizationAccessService,
)
from trading.platform.secrets import (
    OpenBaoSecretResolver,
    SecretKind,
    SecretReference,
    SecretReferenceError,
    SecureTokenFile,
)

_ENV_NAME_KEY = "ROEHUB_ENV"
_IDENTITY_FAIL_FAST_KEY = "IDENTITY_FAIL_FAST"
_IDENTITY_EXCHANGE_KEYS_KEK_B64_KEY = "IDENTITY_EXCHANGE_KEYS_KEK_B64"
_IDENTITY_EXCHANGE_KEYS_KEK_B64_FILE_KEY = "IDENTITY_EXCHANGE_KEYS_KEK_B64_FILE"
_IDENTITY_PG_DSN_KEY = "IDENTITY_PG_DSN"
_IDENTITY_SESSION_COOKIE_NAME_KEY = "IDENTITY_SESSION_COOKIE_NAME"
_IDENTITY_SESSION_IDLE_TTL_SECONDS_KEY = "IDENTITY_SESSION_IDLE_TTL_SECONDS"
_IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS_KEY = "IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS"
_IDENTITY_LOCAL_RP_ID_KEY = "IDENTITY_LOCAL_RP_ID"
_IDENTITY_LOCAL_RP_NAME_KEY = "IDENTITY_LOCAL_RP_NAME"
_IDENTITY_LOCAL_ORIGIN_KEY = "IDENTITY_LOCAL_ORIGIN"
_IDENTITY_LOCAL_ALLOW_INSECURE_LOCALHOST_KEY = "IDENTITY_LOCAL_ALLOW_INSECURE_LOCALHOST"
_OIDC_PROVIDER_ID_KEY = "IDENTITY_OIDC_PROVIDER_ID"
_OIDC_DISPLAY_NAME_KEY = "IDENTITY_OIDC_DISPLAY_NAME"
_OIDC_ISSUER_KEY = "IDENTITY_OIDC_ISSUER"
_OIDC_CLIENT_ID_KEY = "IDENTITY_OIDC_CLIENT_ID"
_OIDC_CLIENT_REFERENCE_KEY = "IDENTITY_OIDC_CLIENT_SECRET_REF"
_OIDC_REDIRECT_URI_KEY = "IDENTITY_OIDC_REDIRECT_URI"
_OIDC_CONNECT_TIMEOUT_KEY = "IDENTITY_OIDC_CONNECT_TIMEOUT_SECONDS"
_OIDC_RESPONSE_TIMEOUT_KEY = "IDENTITY_OIDC_RESPONSE_TIMEOUT_SECONDS"
_OIDC_OVERALL_TIMEOUT_KEY = "IDENTITY_OIDC_OVERALL_TIMEOUT_SECONDS"
_OIDC_ALLOW_INSECURE_HTTP_KEY = "IDENTITY_OIDC_ALLOW_INSECURE_HTTP"
_OPENBAO_ADDRESS_KEY = "OPENBAO_ADDR"
_OPENBAO_CREDENTIAL_PATH_KEY = "ROEHUB_IDENTITY_OPENBAO_TOKEN_FILE"
_OPENBAO_ROOT_KEY = "ROEHUB_OPENBAO_ROOT"
_DEFAULT_DEV_IDENTITY_EXCHANGE_KEYS_KEK_B64 = "cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE="
_DEFAULT_IDENTITY_SESSION_COOKIE_NAME = "roehub_session_id"
_DEFAULT_IDENTITY_SESSION_IDLE_TTL_SECONDS = 1800
_DEFAULT_IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS = 43200
_DEFAULT_IDENTITY_LOCAL_RP_ID = "localhost"
_DEFAULT_IDENTITY_LOCAL_RP_NAME = "Roehub"
_DEFAULT_IDENTITY_LOCAL_ORIGIN = "http://localhost:8000"
_LEGACY_AUTH_COOKIE_PATH = "/"
_LEGACY_AUTH_COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"
_ALLOWED_ENVS = ("dev", "prod", "test")


@dataclass(frozen=True, slots=True)
class IdentityRuntimeSettings:
    """
    IdentityRuntimeSettings — runtime policy for identity module wiring.

    Docs:
      - docs/architecture/identity/oidc-authentication-provider-v1.md
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
    Related:
      - apps/api/wiring/modules/identity.py
      - apps/api/main/app.py
      - apps/api/routes/identity.py
    """

    env_name: str
    fail_fast: bool
    oidc_enabled: bool
    oidc_provider_id: str
    oidc_display_name: str
    oidc_issuer: str
    oidc_client_id: str
    oidc_client_reference: str
    oidc_redirect_uri: str
    oidc_connect_timeout_seconds: float
    oidc_response_timeout_seconds: float
    oidc_overall_timeout_seconds: float
    oidc_allow_insecure_http: bool
    openbao_address: str
    openbao_credential_path: str
    openbao_root: str
    identity_session_cookie_name: str
    identity_session_idle_ttl_seconds: int
    identity_session_absolute_ttl_seconds: int
    identity_local_rp_id: str
    identity_local_rp_name: str
    identity_local_origin: str
    identity_local_allow_insecure_localhost: bool
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
        if self.oidc_enabled:
            required_oidc_values = (
                self.oidc_provider_id,
                self.oidc_display_name,
                self.oidc_issuer,
                self.oidc_client_id,
                self.oidc_client_reference,
                self.openbao_address,
                self.openbao_credential_path,
                self.openbao_root,
                self.oidc_redirect_uri,
            )
            if not all(required_oidc_values):
                raise ValueError("Enabled OIDC provider settings must be non-empty")
            if self.env_name == "prod" and self.oidc_allow_insecure_http:
                raise ValueError("OIDC insecure HTTP is forbidden in prod")
            for value, maximum, name in (
                (self.oidc_connect_timeout_seconds, 3.0, "connect_timeout_seconds"),
                (self.oidc_response_timeout_seconds, 10.0, "response_timeout_seconds"),
                (self.oidc_overall_timeout_seconds, 15.0, "overall_timeout_seconds"),
            ):
                if value <= 0 or value > maximum:
                    raise ValueError(f"OIDC {name} must be in (0, {maximum}]")
            try:
                SecretReference.parse(
                    self.oidc_client_reference,
                    expected_root=self.openbao_root,
                    expected_kind=SecretKind.OIDC,
                )
                SecureTokenFile(Path(self.openbao_credential_path))
            except (SecretReferenceError, ValueError) as error:
                raise ValueError("OIDC OpenBao reference configuration is invalid") from error
        if not self.identity_session_cookie_name:
            raise ValueError(
                "IdentityRuntimeSettings.identity_session_cookie_name must be non-empty"
            )
        if not self.identity_local_rp_id or "://" in self.identity_local_rp_id:
            raise ValueError("IdentityRuntimeSettings.identity_local_rp_id is invalid")
        if not self.identity_local_rp_name:
            raise ValueError("IdentityRuntimeSettings.identity_local_rp_name must be non-empty")
        parsed_local_origin = urlparse(self.identity_local_origin)
        if parsed_local_origin.scheme not in {"http", "https"} or not parsed_local_origin.hostname:
            raise ValueError("IdentityRuntimeSettings.identity_local_origin is invalid")
        if parsed_local_origin.hostname != self.identity_local_rp_id:
            raise ValueError("Identity local origin host must match WebAuthn RP id")
        if self.env_name == "prod" and parsed_local_origin.scheme != "https":
            local_http_allowed = (
                self.identity_local_allow_insecure_localhost
                and parsed_local_origin.scheme == "http"
                and parsed_local_origin.hostname == "localhost"
                and self.identity_local_rp_id == "localhost"
            )
            if not local_http_allowed:
                raise ValueError("Identity local auth requires HTTPS in prod")
        if self.identity_local_allow_insecure_localhost and not (
            parsed_local_origin.scheme == "http"
            and parsed_local_origin.hostname == "localhost"
            and self.identity_local_rp_id == "localhost"
        ):
            raise ValueError(
                "IDENTITY_LOCAL_ALLOW_INSECURE_LOCALHOST requires exact "
                "http://localhost local auth"
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
      - docs/architecture/identity/oidc-authentication-provider-v1.md
    Related:
      - apps/api/wiring/modules/identity.py
      - apps/api/wiring/modules/strategy.py
      - apps/api/main/app.py
    """

    router: APIRouter
    current_user_dependency: RequireCurrentUserDependency
    organization_repository: OrganizationRepository
    organization_access_service: OrganizationAccessService
    clock: IdentityClock


@dataclass(frozen=True, slots=True)
class _IdentityPersistenceBundle:
    """
    _IdentityPersistenceBundle groups identity repositories built from one storage policy.

    Docs:
      - docs/architecture/identity/oidc-authentication-provider-v1.md
    Related:
      - apps/api/wiring/modules/identity.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/
      - src/trading/contexts/identity/adapters/outbound/persistence/in_memory/
    """

    exchange_keys_repository: ExchangeKeysRepository
    account_settings_repository: AccountSettingsRepository
    user_repository: UserRepository
    session_repository: SessionRepository
    organization_repository: OrganizationRepository
    local_auth_repository: LocalAuthRepository
    oidc_identity_repository: OidcIdentityRepository


def build_identity_router(*, environ: Mapping[str, str]) -> APIRouter:
    """
    Build fully wired identity router from environment settings.

    Docs:
      - docs/architecture/identity/oidc-authentication-provider-v1.md
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
      - docs/architecture/identity/oidc-authentication-provider-v1.md
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
    organization_access_service = OrganizationAccessService(
        repository=persistence.organization_repository
    )
    local_auth_service = LocalAuthService(
        repository=persistence.local_auth_repository,
        user_repository=persistence.user_repository,
        session_repository=persistence.session_repository,
        clock=clock,
        rp_id=settings.identity_local_rp_id,
        rp_name=settings.identity_local_rp_name,
        expected_origin=settings.identity_local_origin,
        session_idle_ttl_seconds=settings.identity_session_idle_ttl_seconds,
        session_absolute_ttl_seconds=settings.identity_session_absolute_ttl_seconds,
    )
    oidc_authentication_service: OidcAuthenticationService | None = None
    if settings.oidc_enabled:
        oidc_credential_resolver = OpenBaoSecretResolver(
            address=settings.openbao_address,
            token_source=SecureTokenFile(Path(settings.openbao_credential_path)),
            secret_root=settings.openbao_root,
        )
        provider = HttpOidcAuthenticationProvider(
            provider_id=settings.oidc_provider_id,
            display_name=settings.oidc_display_name,
            issuer=settings.oidc_issuer,
            client_id=settings.oidc_client_id,
            client_credential_source=lambda: oidc_credential_resolver.resolve(
                settings.oidc_client_reference,
                expected_kind=SecretKind.OIDC,
            ).reveal_text(),
            redirect_uri=settings.oidc_redirect_uri,
            connect_timeout_seconds=settings.oidc_connect_timeout_seconds,
            response_timeout_seconds=settings.oidc_response_timeout_seconds,
            overall_timeout_seconds=settings.oidc_overall_timeout_seconds,
            allow_insecure_http=settings.oidc_allow_insecure_http,
            metrics=PrometheusOidcProviderMetrics(),
        )
        oidc_authentication_service = OidcAuthenticationService(
            provider=provider,
            repository=persistence.oidc_identity_repository,
            session_repository=persistence.session_repository,
            clock=clock,
            session_idle_ttl_seconds=settings.identity_session_idle_ttl_seconds,
            session_absolute_ttl_seconds=settings.identity_session_absolute_ttl_seconds,
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
            current_user_dependency=current_user_dependency,
            audit_events_repository=persistence.account_settings_repository,
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
            organization_access_service=organization_access_service,
            local_auth_service=local_auth_service,
            oidc_authentication_service=oidc_authentication_service,
        ),
        current_user_dependency=current_user_dependency,
        organization_repository=persistence.organization_repository,
        organization_access_service=organization_access_service,
        clock=clock,
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
            account_settings_repository=PostgresAccountSettingsRepository(gateway=gateway),
            user_repository=PostgresIdentityUserRepository(gateway=gateway),
            session_repository=PostgresIdentitySessionRepository(gateway=gateway),
            organization_repository=PostgresOrganizationRepository(dsn=settings.postgres_dsn),
            local_auth_repository=PostgresLocalAuthRepository(dsn=settings.postgres_dsn),
            oidc_identity_repository=PostgresOidcIdentityRepository(dsn=settings.postgres_dsn),
        )
    if settings.env_name == "prod":
        raise ValueError(
            f"{_IDENTITY_PG_DSN_KEY} must be set in prod for persisted Roehub sessions"
        )
    user_repository = InMemoryIdentityUserRepository()
    session_repository = InMemoryIdentitySessionRepository()
    organization_repository = InMemoryOrganizationRepository()
    return _IdentityPersistenceBundle(
        exchange_keys_repository=InMemoryIdentityExchangeKeysRepository(),
        account_settings_repository=InMemoryAccountSettingsRepository(),
        user_repository=user_repository,
        session_repository=session_repository,
        organization_repository=organization_repository,
        local_auth_repository=InMemoryLocalAuthRepository(
            user_repository=user_repository,
            organization_repository=organization_repository,
        ),
        oidc_identity_repository=InMemoryOidcIdentityRepository(
            user_repository=user_repository,
            organization_repository=organization_repository,
        ),
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

    oidc_provider_id = environ.get(_OIDC_PROVIDER_ID_KEY, "").strip().lower()
    oidc_display_name = environ.get(_OIDC_DISPLAY_NAME_KEY, "").strip()
    oidc_issuer = environ.get(_OIDC_ISSUER_KEY, "").strip().rstrip("/")
    oidc_client_id = environ.get(_OIDC_CLIENT_ID_KEY, "").strip()
    oidc_client_reference = environ.get(_OIDC_CLIENT_REFERENCE_KEY, "").strip()
    openbao_address = environ.get(_OPENBAO_ADDRESS_KEY, "").strip()
    openbao_credential_path = environ.get(_OPENBAO_CREDENTIAL_PATH_KEY, "").strip()
    openbao_root = environ.get(_OPENBAO_ROOT_KEY, "").strip()
    oidc_redirect_uri = environ.get(_OIDC_REDIRECT_URI_KEY, "").strip()
    oidc_connect_timeout_seconds = _read_optional_positive_float(
        raw_value=environ.get(_OIDC_CONNECT_TIMEOUT_KEY, "").strip(),
        key=_OIDC_CONNECT_TIMEOUT_KEY,
    )
    oidc_response_timeout_seconds = _read_optional_positive_float(
        raw_value=environ.get(_OIDC_RESPONSE_TIMEOUT_KEY, "").strip(),
        key=_OIDC_RESPONSE_TIMEOUT_KEY,
    )
    oidc_overall_timeout_seconds = _read_optional_positive_float(
        raw_value=environ.get(_OIDC_OVERALL_TIMEOUT_KEY, "").strip(),
        key=_OIDC_OVERALL_TIMEOUT_KEY,
    )
    raw_allow_insecure = environ.get(_OIDC_ALLOW_INSECURE_HTTP_KEY, "").strip()
    oidc_allow_insecure_http = (
        _parse_bool(raw_value=raw_allow_insecure, key=_OIDC_ALLOW_INSECURE_HTTP_KEY)
        if raw_allow_insecure
        else False
    )
    identity_session_cookie_name = environ.get(_IDENTITY_SESSION_COOKIE_NAME_KEY, "").strip()
    identity_session_idle_ttl_seconds = _read_optional_positive_int(
        raw_value=environ.get(_IDENTITY_SESSION_IDLE_TTL_SECONDS_KEY, "").strip(),
        key=_IDENTITY_SESSION_IDLE_TTL_SECONDS_KEY,
    )
    identity_session_absolute_ttl_seconds = _read_optional_positive_int(
        raw_value=environ.get(_IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS_KEY, "").strip(),
        key=_IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS_KEY,
    )
    identity_local_rp_id = environ.get(_IDENTITY_LOCAL_RP_ID_KEY, "").strip().lower()
    identity_local_rp_name = environ.get(_IDENTITY_LOCAL_RP_NAME_KEY, "").strip()
    identity_local_origin = environ.get(_IDENTITY_LOCAL_ORIGIN_KEY, "").strip().rstrip("/")
    raw_allow_insecure_localhost = environ.get(
        _IDENTITY_LOCAL_ALLOW_INSECURE_LOCALHOST_KEY, ""
    ).strip()
    identity_local_allow_insecure_localhost = (
        _parse_bool(
            raw_value=raw_allow_insecure_localhost,
            key=_IDENTITY_LOCAL_ALLOW_INSECURE_LOCALHOST_KEY,
        )
        if raw_allow_insecure_localhost
        else False
    )
    identity_exchange_keys_kek_b64 = _resolve_identity_exchange_keys_kek(environ=environ)

    explicit_oidc_settings = {
        _OIDC_PROVIDER_ID_KEY: oidc_provider_id,
        _OIDC_DISPLAY_NAME_KEY: oidc_display_name,
        _OIDC_ISSUER_KEY: oidc_issuer,
        _OIDC_CLIENT_ID_KEY: oidc_client_id,
        _OIDC_CLIENT_REFERENCE_KEY: oidc_client_reference,
        _OPENBAO_ADDRESS_KEY: openbao_address,
        _OPENBAO_CREDENTIAL_PATH_KEY: openbao_credential_path,
        _OPENBAO_ROOT_KEY: openbao_root,
        _OIDC_REDIRECT_URI_KEY: oidc_redirect_uri,
    }
    oidc_enabled = any(explicit_oidc_settings.values())
    if oidc_enabled:
        for setting_name, setting_value in explicit_oidc_settings.items():
            if not setting_value:
                raise ValueError(f"{setting_name} must be set when OIDC is enabled")

    if fail_fast:
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
        if not identity_local_rp_id:
            raise ValueError(
                f"{_IDENTITY_LOCAL_RP_ID_KEY} must be set when {_IDENTITY_FAIL_FAST_KEY}=true"
            )
        if not identity_local_rp_name:
            raise ValueError(
                f"{_IDENTITY_LOCAL_RP_NAME_KEY} must be set when {_IDENTITY_FAIL_FAST_KEY}=true"
            )
        if not identity_local_origin:
            raise ValueError(
                f"{_IDENTITY_LOCAL_ORIGIN_KEY} must be set when {_IDENTITY_FAIL_FAST_KEY}=true"
            )
    if env_name == "prod" and identity_exchange_keys_kek_b64 == (
        _DEFAULT_DEV_IDENTITY_EXCHANGE_KEYS_KEK_B64
    ):
        raise ValueError(
            f"{_IDENTITY_EXCHANGE_KEYS_KEK_B64_KEY} must not use the dev-only KEK "
            "when ROEHUB_ENV=prod"
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
    effective_identity_local_rp_id = identity_local_rp_id or _DEFAULT_IDENTITY_LOCAL_RP_ID
    effective_identity_local_rp_name = identity_local_rp_name or _DEFAULT_IDENTITY_LOCAL_RP_NAME
    effective_identity_local_origin = identity_local_origin or _DEFAULT_IDENTITY_LOCAL_ORIGIN
    effective_exchange_keys_kek_b64 = (
        identity_exchange_keys_kek_b64 or _DEFAULT_DEV_IDENTITY_EXCHANGE_KEYS_KEK_B64
    )

    postgres_dsn = environ.get(_IDENTITY_PG_DSN_KEY, "").strip()

    return IdentityRuntimeSettings(
        env_name=env_name,
        fail_fast=fail_fast,
        oidc_enabled=oidc_enabled,
        oidc_provider_id=oidc_provider_id,
        oidc_display_name=oidc_display_name,
        oidc_issuer=oidc_issuer,
        oidc_client_id=oidc_client_id,
        oidc_client_reference=oidc_client_reference,
        oidc_redirect_uri=oidc_redirect_uri,
        oidc_connect_timeout_seconds=oidc_connect_timeout_seconds or 3.0,
        oidc_response_timeout_seconds=oidc_response_timeout_seconds or 10.0,
        oidc_overall_timeout_seconds=oidc_overall_timeout_seconds or 15.0,
        oidc_allow_insecure_http=oidc_allow_insecure_http,
        openbao_address=openbao_address,
        openbao_credential_path=openbao_credential_path,
        openbao_root=openbao_root,
        identity_session_cookie_name=effective_identity_session_cookie_name,
        identity_session_idle_ttl_seconds=effective_identity_session_idle_ttl_seconds,
        identity_session_absolute_ttl_seconds=effective_identity_session_absolute_ttl_seconds,
        identity_local_rp_id=effective_identity_local_rp_id,
        identity_local_rp_name=effective_identity_local_rp_name,
        identity_local_origin=effective_identity_local_origin,
        identity_local_allow_insecure_localhost=(identity_local_allow_insecure_localhost),
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
        raise ValueError(f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}, got {raw_env_name!r}")
    return raw_env_name


def _resolve_identity_exchange_keys_kek(*, environ: Mapping[str, str]) -> str:
    raw_value = environ.get(_IDENTITY_EXCHANGE_KEYS_KEK_B64_KEY, "").strip()
    raw_path = environ.get(_IDENTITY_EXCHANGE_KEYS_KEK_B64_FILE_KEY, "").strip()
    if raw_value and raw_path:
        raise ValueError(
            f"set only one of {_IDENTITY_EXCHANGE_KEYS_KEK_B64_KEY} and "
            f"{_IDENTITY_EXCHANGE_KEYS_KEK_B64_FILE_KEY}"
        )
    if not raw_path:
        return raw_value
    return SecureTokenFile(Path(raw_path)).read()


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


def _read_optional_positive_float(*, raw_value: str, key: str) -> float | None:
    normalized = raw_value.strip()
    if not normalized:
        return None
    try:
        parsed_value = float(normalized)
    except ValueError as error:
        raise ValueError(f"{key} must be a positive number, got {raw_value!r}") from error
    if parsed_value <= 0:
        raise ValueError(f"{key} must be a positive number, got {raw_value!r}")
    return parsed_value
