"""
Adapters package for identity bounded context.
"""

from .inbound import (
    CurrentUserResponse,
    RequireCurrentUserDependency,
    TelegramLoginRequest,
    TelegramLoginResponse,
    build_auth_telegram_router,
)
from .outbound import (
    Hs256JwtCodec,
    IdentityPostgresGateway,
    InMemoryIdentityUserRepository,
    JwtCookieCurrentUser,
    PostgresIdentityUserRepository,
    PsycopgIdentityPostgresGateway,
    SystemIdentityClock,
    TelegramLoginWidgetPayloadValidator,
)

__all__ = [
    "CurrentUserResponse",
    "Hs256JwtCodec",
    "IdentityPostgresGateway",
    "InMemoryIdentityUserRepository",
    "JwtCookieCurrentUser",
    "PostgresIdentityUserRepository",
    "PsycopgIdentityPostgresGateway",
    "RequireCurrentUserDependency",
    "SystemIdentityClock",
    "TelegramLoginRequest",
    "TelegramLoginResponse",
    "TelegramLoginWidgetPayloadValidator",
    "build_auth_telegram_router",
]
