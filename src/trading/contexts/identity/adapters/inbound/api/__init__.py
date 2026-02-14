from .deps import RequireCurrentUserDependency
from .routes import (
    CurrentUserResponse,
    TelegramLoginRequest,
    TelegramLoginResponse,
    build_auth_telegram_router,
)

__all__ = [
    "CurrentUserResponse",
    "RequireCurrentUserDependency",
    "TelegramLoginRequest",
    "TelegramLoginResponse",
    "build_auth_telegram_router",
]
