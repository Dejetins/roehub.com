"""Closed storage capability at the authenticated installation control boundary."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from trading.contexts.backtest_artifacts.application.services.management import (
    Coordinate,
    Schedule,
    StrictModel,
)


class StorageCommand(StrictModel):
    action: Literal["inspect", "validate", "configure", "cleanup_preview", "cleanup_confirm"]
    root_id: str = Field(default="backtest-primary", pattern=r"^[a-z][a-z0-9-]{0,63}$")
    expected_version: int = Field(ge=0, strict=True)
    budget_bytes: int | None = Field(default=None, ge=1, strict=True)
    schedule: Schedule | None = None
    validation_id: UUID | None = None
    preview_id: UUID | None = None
    coordinate: Coordinate | None = None
    slot: Literal["slot_a", "slot_b"] | None = None

    @model_validator(mode="after")
    def semantics(self):
        if self.action == "configure" and (
            self.validation_id is None or self.budget_bytes is None or self.schedule is None
        ):
            raise ValueError("validated_configuration_required")
        if self.action == "cleanup_preview" and (self.coordinate is None or self.slot is None):
            raise ValueError("cleanup_target_required")
        if self.action == "cleanup_confirm" and self.preview_id is None:
            raise ValueError("preview_required")
        return self


class StorageEnvelope(StrictModel):
    organization_id: UUID
    actor_id: UUID
    idempotency_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
    command: StorageCommand


class StorageResult(StrictModel):
    state: Literal[
        "ready",
        "validated",
        "configured",
        "relocation_required",
        "preview",
        "cleaned",
        "busy",
        "preview_stale",
        "unknown",
    ]
    root_id: str = Field(max_length=64)
    control_version: int
    root_revision: int
    path: str | None = Field(default=None, max_length=4096)
    free_bytes: int | None = Field(default=None, ge=0)
    used_bytes: int | None = Field(default=None, ge=0)
    budget_bytes: int | None = Field(default=None, ge=0)
    validation_id: UUID | None = None
    preview_id: UUID | None = None
    expires_at: datetime | None = None
    eligible_bytes: int | None = Field(default=None, ge=0)
    schedule: Schedule | None = None
    cleanup_policy: Literal["explicit_inactive_unpinned"] = "explicit_inactive_unpinned"
