"""Closed browser contracts; worker snapshots are never accepted or serialized."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from trading.contexts.backtest_artifacts.application.services.management import (
    Coordinate,
    Profile,
    Schedule,
    StrictModel,
)

State = Literal["waiting_for_source", "queued", "running", "retry_wait", "ready", "error"]


class PolicyPatch(StrictModel):
    profile: Profile | None = None
    schedule: Schedule | None = None
    mode: Literal["on_demand", "precompute"] | None = None

    @model_validator(mode="after")
    def fields(self):
        if not self.model_fields_set or (self.profile is not None and self.mode is not None):
            raise ValueError("explicit_nonconflicting_patch_required")
        if "profile" in self.model_fields_set and self.profile is None:
            raise ValueError("profile_required")
        if "mode" in self.model_fields_set and self.mode is None:
            raise ValueError("mode_required")
        return self


class CommandItem(StrictModel):
    coordinate: Coordinate
    expected_version: int = Field(ge=0, strict=True)
    patch: PolicyPatch | None = None


class Command(StrictModel):
    items: tuple[CommandItem, ...] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def unique(self):
        if len({i.coordinate.key for i in self.items}) != len(self.items):
            raise ValueError("duplicate_coordinate")
        return self


class SinglePolicy(StrictModel):
    expected_version: int = Field(ge=0, strict=True)
    patch: PolicyPatch


class CommandOutcome(StrictModel):
    coordinate: Coordinate
    request_id: UUID | None
    state: State | None
    outcome: Literal["submitted", "policy_updated"] = "submitted"
    control_version: int


class CommandResult(StrictModel):
    command_id: UUID
    items: tuple[CommandOutcome, ...] = Field(max_length=50)


class ProgressView(StrictModel):
    phase: str = Field(max_length=64)
    timeframe: str | None = Field(default=None, max_length=8)
    completed_units: int | None = Field(default=None, ge=0)
    total_units: int | None = Field(default=None, ge=0)
    unit: str = Field(max_length=32)
    progress_epoch: int = Field(ge=0)
    observed_at: datetime | None = None


class RequestView(StrictModel):
    request_id: UUID
    coordinate: Coordinate
    kind: Literal["base", "derivatives"]
    state: State
    reason_code: str | None = Field(default=None, max_length=64)
    created_at: datetime
    progress: tuple[ProgressView, ...] = Field(default=(), max_length=256)


class EventView(StrictModel):
    state: State
    reason_code: str | None = Field(default=None, max_length=64)
    observed_at: datetime
    progress: tuple[ProgressView, ...] = Field(default=(), max_length=256)


class TimeframeView(StrictModel):
    timeframe: str = Field(max_length=8)
    state: str = Field(max_length=32)
    closed_bars: int | None = None
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    missing_bars: int | None = Field(default=None, ge=0)
    retry_after_source_end_ms: int | None = None


class Detail(StrictModel):
    coordinate: Coordinate
    control_version: int
    profile: Profile
    schedule: Schedule
    inherits_schedule: bool
    next_run: datetime
    last_success: datetime | None = None
    source_through: datetime | None = None
    covered_through: datetime | None = None
    base_state: str = Field(max_length=32)
    timeframes: tuple[TimeframeView, ...] = Field(default=(), max_length=11)
    base_request: RequestView | None = None
    derivative_request: RequestView | None = None
    size_bytes: int | None = None
    location_label: Literal["Backtest artifacts"] = "Backtest artifacts"


class InventoryPage(StrictModel):
    snapshot_id: UUID
    observed_at: datetime
    items: tuple[Detail, ...] = Field(max_length=100)
    next_cursor: str | None = Field(default=None, max_length=256)


class RequestPage(StrictModel):
    items: tuple[RequestView, ...] = Field(max_length=50)
    next_cursor: UUID | None = None


class Events(StrictModel):
    items: tuple[EventView, ...] = Field(max_length=50)


class Defaults(StrictModel):
    profile: Profile
    schedule: Schedule
    base_timeframes: tuple[str, ...] = Field(max_length=11)
    unavailable_indicator_ids: tuple[str, ...] = Field(max_length=200)
    can_mutate: bool


class SafeStorage(StrictModel):
    location_label: Literal["Backtest artifacts"] = "Backtest artifacts"
    state: Literal["ready", "paused", "unavailable", "storage_unconfigured"]
