from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from trading.contexts.identity.adapters.inbound.api.deps.current_user import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal


class CurrentUserResponse(BaseModel):
    """Protected endpoint response for the authenticated local Roehub user."""

    user_id: str
    paid_level: Literal["free", "base", "pro", "ultra"]


def build_auth_current_user_router(
    *,
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    """Build the provider-neutral current-user route shared by local and OIDC auth."""
    router = APIRouter()

    @router.get("/auth/current-user", response_model=CurrentUserResponse)
    def get_auth_current_user(
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> CurrentUserResponse:
        return CurrentUserResponse(
            user_id=str(principal.user_id),
            paid_level=cast(
                Literal["free", "base", "pro", "ultra"],
                str(principal.paid_level),
            ),
        )

    return router
