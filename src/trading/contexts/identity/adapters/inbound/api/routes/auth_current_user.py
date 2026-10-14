from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from trading.contexts.identity.adapters.inbound.api.deps.current_user import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal


class CurrentUserResponse(BaseModel):
    """Protected endpoint response for the authenticated local Roehub user."""

    user_id: str


def build_auth_current_user_router(
    *,
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    """Build the provider-neutral current-user route for local authentication."""
    router = APIRouter()

    @router.get("/auth/current-user", response_model=CurrentUserResponse)
    def get_auth_current_user(
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> CurrentUserResponse:
        return CurrentUserResponse(
            user_id=str(principal.user_id),
        )

    return router
