"""
Adapters package for identity bounded context.
"""

from .inbound import (
    CreateExchangeKeyRequest,
    CurrentUserResponse,
    ExchangeKeyResponse,
    RequireCurrentUserDependency,
    build_auth_oidc_router,
    build_exchange_keys_router,
)
from .outbound import (
    AesGcmEnvelopeExchangeKeysSecretCipher,
    IdentityPostgresGateway,
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

__all__ = [
    "AesGcmEnvelopeExchangeKeysSecretCipher",
    "CreateExchangeKeyRequest",
    "CurrentUserResponse",
    "ExchangeKeyResponse",
    "IdentityPostgresGateway",
    "InMemoryIdentityExchangeKeysRepository",
    "InMemoryIdentitySessionRepository",
    "InMemoryIdentityUserRepository",
    "PostgresIdentityExchangeKeysRepository",
    "PostgresIdentitySessionRepository",
    "PostgresIdentityUserRepository",
    "PsycopgIdentityPostgresGateway",
    "RequireCurrentUserDependency",
    "RoehubSessionCurrentUser",
    "SystemIdentityClock",
    "build_auth_oidc_router",
    "build_exchange_keys_router",
]
