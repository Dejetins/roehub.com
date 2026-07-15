from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Literal, Mapping, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROEHUB_DATA_FRAME_CONTRACT = "RoehubDataFrame/v1"
ROEHUB_PANEL_CONTRIBUTION_CONTRACT = "RoehubPanelContribution/v1"
ROEHUB_APP_CONTRIBUTION_CONTRACT = "RoehubAppContribution/v1"

MAX_DATA_FRAME_ROWS = 1_000
MAX_DATA_FRAME_BYTES = 1_048_576
MAX_DATA_FRAME_POINTS = 5_000
MAX_DATA_SOURCE_TIMEOUT_MS = 5_000

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]{0,127}$")
_SECRET_IDENTIFIER_RE = re.compile(
    r"(?:^|[_.-])(?:authorization|cookie|credential|password|secret|token|api[_-]?key)(?:$|[_.-])",
    re.IGNORECASE,
)

DataFrameScalar: TypeAlias = str | int | float | bool | None


class DataFrameUnit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "currency",
        "percent",
        "ratio",
        "quantity",
        "duration",
        "timestamp",
        "count",
        "custom",
    ]
    symbol: str = Field(min_length=1, max_length=24)
    scale: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_finite_scale(self) -> DataFrameUnit:
        if not math.isfinite(self.scale):
            raise ValueError("unit scale must be finite")
        return self


class DataFrameColumn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=160)
    data_type: Literal["timestamp", "number", "integer", "string", "boolean"]
    role: Literal["dimension", "measure"]
    unit: DataFrameUnit | None = None
    nullable: bool = True

    @model_validator(mode="after")
    def validate_identifier_and_unit(self) -> DataFrameColumn:
        if _IDENTIFIER_RE.fullmatch(self.key) is None:
            raise ValueError("column key is not a portable identifier")
        if _SECRET_IDENTIFIER_RE.search(self.key):
            raise ValueError("secret-shaped columns are forbidden")
        if self.data_type == "timestamp" and (
            self.unit is None or self.unit.kind != "timestamp"
        ):
            raise ValueError("timestamp columns require an explicit timestamp unit")
        return self


class DataFrameFreshness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["fresh", "stale", "unknown"]
    observed_at: datetime | None = None
    age_seconds: int | None = Field(default=None, ge=0)
    max_age_seconds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_observation_timestamp(self) -> DataFrameFreshness:
        if self.observed_at is not None and self.observed_at.tzinfo is None:
            raise ValueError("freshness observed_at must be timezone-aware")
        return self


class DataFrameNotice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["info", "warning"]
    code: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,127}$")
    message: str = Field(min_length=1, max_length=500)


class DataFrameBoundedError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,127}$")
    message: str = Field(min_length=1, max_length=500)
    retryable: bool
    field: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_field(self) -> DataFrameBoundedError:
        if self.field is None:
            return self
        if _IDENTIFIER_RE.fullmatch(self.field) is None:
            raise ValueError("bounded error field must be a portable identifier")
        if _SECRET_IDENTIFIER_RE.search(self.field):
            raise ValueError("secret-shaped bounded error fields are forbidden")
        return self


class DataFrameMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_label: str = Field(min_length=1, max_length=160)
    query_label: str = Field(min_length=1, max_length=160)
    generated_at: datetime
    attributes: dict[str, DataFrameScalar] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def reject_secret_metadata(self) -> DataFrameMetadata:
        if self.generated_at.tzinfo is None:
            raise ValueError("metadata generated_at must be timezone-aware")
        for key, value in self.attributes.items():
            if _IDENTIFIER_RE.fullmatch(key) is None:
                raise ValueError("metadata keys must be portable identifiers")
            if _SECRET_IDENTIFIER_RE.search(key):
                raise ValueError("secret-shaped metadata is forbidden")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("metadata numbers must be finite")
        return self


class RoehubDataFrame(BaseModel):
    """Library-independent, bounded table contract used by host render adapters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["RoehubDataFrame/v1"] = ROEHUB_DATA_FRAME_CONTRACT
    frame_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{2,127}$")
    title: str = Field(min_length=1, max_length=200)
    columns: list[DataFrameColumn] = Field(min_length=1, max_length=64)
    rows: list[dict[str, DataFrameScalar]] = Field(max_length=MAX_DATA_FRAME_ROWS)
    metadata: DataFrameMetadata
    freshness: DataFrameFreshness
    notices: list[DataFrameNotice] = Field(default_factory=list, max_length=32)
    partial: bool = False
    errors: list[DataFrameBoundedError] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_tabular_shape(self) -> RoehubDataFrame:
        columns_by_key = {column.key: column for column in self.columns}
        if len(columns_by_key) != len(self.columns):
            raise ValueError("column keys must be unique")
        expected_keys = set(columns_by_key)
        for row in self.rows:
            if set(row) != expected_keys:
                raise ValueError("each row must contain exactly the declared columns")
            for key, value in row.items():
                _validate_cell(column=columns_by_key[key], value=value)
        if self.errors and not self.partial:
            raise ValueError("bounded errors require partial=true")
        return self


class DataSourceFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]{0,127}$")
    operator: Literal["eq", "in", "gt", "gte", "lt", "lte"]
    value: DataFrameScalar | list[DataFrameScalar]

    @model_validator(mode="after")
    def validate_filter(self) -> DataSourceFilter:
        if _SECRET_IDENTIFIER_RE.search(self.field):
            raise ValueError("secret-shaped filter fields are forbidden")
        if self.operator == "in":
            if not isinstance(self.value, list) or not 1 <= len(self.value) <= 100:
                raise ValueError("in filter requires between 1 and 100 values")
        elif isinstance(self.value, list):
            raise ValueError("only the in operator accepts a value list")
        values = self.value if isinstance(self.value, list) else [self.value]
        if any(isinstance(value, float) and not math.isfinite(value) for value in values):
            raise ValueError("filter numbers must be finite")
        return self


class DataSourceQueryRequest(BaseModel):
    """Public query payload; organization scope is deliberately absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["DataSourceQuery/v1"] = "DataSourceQuery/v1"
    dataset: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]{0,127}$")
    dimensions: list[str] = Field(min_length=1, max_length=16)
    measures: list[str] = Field(min_length=1, max_length=16)
    filters: list[DataSourceFilter] = Field(default_factory=list, max_length=32)
    row_limit: int = Field(default=200, ge=1, le=MAX_DATA_FRAME_ROWS)
    byte_limit: int = Field(default=262_144, ge=1_024, le=MAX_DATA_FRAME_BYTES)
    point_limit: int = Field(default=1_000, ge=1, le=MAX_DATA_FRAME_POINTS)
    timeout_ms: int = Field(default=3_000, ge=50, le=MAX_DATA_SOURCE_TIMEOUT_MS)
    read_only: Literal[True] = True

    @model_validator(mode="after")
    def validate_field_selection(self) -> DataSourceQueryRequest:
        fields = self.dimensions + self.measures
        if len(set(fields)) != len(fields):
            raise ValueError("dimensions and measures must be unique")
        for field in fields:
            if _IDENTIFIER_RE.fullmatch(field) is None:
                raise ValueError("query fields must be portable identifiers")
            if _SECRET_IDENTIFIER_RE.search(field):
                raise ValueError("secret-shaped query fields are forbidden")
        return self


class PanelDataSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: UUID
    query: DataSourceQueryRequest


class PanelPresentation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x_column: str | None = Field(default=None, max_length=128)
    y_columns: list[str] = Field(default_factory=list, max_length=6)
    table_columns: list[str] = Field(min_length=1, max_length=24)
    default_view: Literal["visual", "table"] = "visual"

    @model_validator(mode="after")
    def validate_columns(self) -> PanelPresentation:
        columns = [
            *([] if self.x_column is None else [self.x_column]),
            *self.y_columns,
            *self.table_columns,
        ]
        if any(_IDENTIFIER_RE.fullmatch(column) is None for column in columns):
            raise ValueError("presentation columns must be portable identifiers")
        if len(set(self.y_columns)) != len(self.y_columns):
            raise ValueError("presentation y columns must be unique")
        if len(set(self.table_columns)) != len(self.table_columns):
            raise ValueError("presentation table columns must be unique")
        return self


class RoehubPanelContribution(BaseModel):
    """Host-owned panel declaration. Code, markup, and script URLs are not representable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["RoehubPanelContribution/v1"] = (
        ROEHUB_PANEL_CONTRIBUTION_CONTRACT
    )
    contribution_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,127}$")
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=500)
    renderer: Literal[
        "trading-time-series",
        "analytics-series",
        "analytics-table",
        "research-summary",
    ]
    source: PanelDataSource
    presentation: PanelPresentation

    @model_validator(mode="after")
    def validate_renderer_presentation(self) -> RoehubPanelContribution:
        dimensions = set(self.source.query.dimensions)
        measures = set(self.source.query.measures)
        selected = dimensions | measures
        if self.renderer in {"trading-time-series", "analytics-series"} and (
            self.presentation.x_column is None or not self.presentation.y_columns
        ):
            raise ValueError("series renderers require x and y columns")
        if (
            self.presentation.x_column is not None
            and self.presentation.x_column not in dimensions
        ):
            raise ValueError("panel x column must be a selected query dimension")
        if not set(self.presentation.y_columns).issubset(measures):
            raise ValueError("panel y columns must be selected query measures")
        if not set(self.presentation.table_columns).issubset(selected):
            raise ValueError("panel table columns must be selected query fields")
        return self


class AppContributionSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,127}$")
    title: str = Field(min_length=1, max_length=160)
    panel_contribution_ids: list[str] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_panel_references(self) -> AppContributionSection:
        if len(set(self.panel_contribution_ids)) != len(self.panel_contribution_ids):
            raise ValueError("panel contribution references must be unique")
        if any(
            re.fullmatch(r"^[a-z][a-z0-9_.-]{2,127}$", contribution_id) is None
            for contribution_id in self.panel_contribution_ids
        ):
            raise ValueError("panel contribution references must be portable identifiers")
        return self


class RoehubAppContribution(BaseModel):
    """A composition of declarative panels, never plugin-owned executable UI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["RoehubAppContribution/v1"] = ROEHUB_APP_CONTRIBUTION_CONTRACT
    contribution_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,127}$")
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=500)
    sections: list[AppContributionSection] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_sections(self) -> RoehubAppContribution:
        section_ids = [section.section_id for section in self.sections]
        if len(set(section_ids)) != len(section_ids):
            raise ValueError("app contribution section ids must be unique")
        return self


def dataframe_point_count(frame: RoehubDataFrame) -> int:
    measure_count = sum(column.role == "measure" for column in frame.columns)
    return len(frame.rows) * measure_count


def _validate_cell(*, column: DataFrameColumn, value: DataFrameScalar) -> None:
    if value is None:
        if not column.nullable:
            raise ValueError(f"column {column.key!r} is not nullable")
        return
    valid = {
        "timestamp": isinstance(value, str) and _is_aware_timestamp(value),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
    }[column.data_type]
    if not valid:
        raise ValueError(f"cell type does not match column {column.key!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"cell number must be finite for column {column.key!r}")


def _is_aware_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def redact_data_frame(frame: RoehubDataFrame) -> RoehubDataFrame:
    """Redact token-like text everywhere it could cross into the browser."""

    return frame.model_copy(
        update={
            "title": _redact_text(frame.title),
            "columns": [
                column.model_copy(
                    update={
                        "label": _redact_text(column.label),
                        "unit": (
                            column.unit.model_copy(
                                update={"symbol": _redact_text(column.unit.symbol)}
                            )
                            if column.unit is not None
                            else None
                        ),
                    }
                )
                for column in frame.columns
            ],
            "rows": [
                {
                    key: _redact_text(value) if isinstance(value, str) else value
                    for key, value in row.items()
                }
                for row in frame.rows
            ],
            "metadata": frame.metadata.model_copy(
                update={
                    "source_label": _redact_text(frame.metadata.source_label),
                    "query_label": _redact_text(frame.metadata.query_label),
                    "attributes": {
                        key: _redact_text(value) if isinstance(value, str) else value
                        for key, value in frame.metadata.attributes.items()
                    },
                }
            ),
            "notices": [
                notice.model_copy(update={"message": _redact_text(notice.message)})
                for notice in frame.notices
            ],
            "errors": [
                error.model_copy(
                    update={
                        "message": _redact_text(error.message),
                        "field": (
                            _redact_text(error.field) if error.field is not None else None
                        ),
                    }
                )
                for error in frame.errors
            ],
        }
    )


def _redact_text(value: str) -> str:
    redacted = re.sub(
        r"(?i)\b(?:bearer|token|password|secret|api[_-]?key)\s*[:=]\s*[^\s,;]+",
        "[REDACTED]",
        value,
    )
    return redacted[:500]


def data_frame_json_payload(frame: RoehubDataFrame) -> Mapping[str, Any]:
    return frame.model_dump(mode="json")
