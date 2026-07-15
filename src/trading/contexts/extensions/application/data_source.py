from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import UUID, uuid4

from pydantic import ValidationError

from trading.contexts.extensions.application.ports import (
    DataSourceAuthorization,
    DataSourceAuthorizationError,
    DataSourceGatewayError,
    DataSourceInvoker,
    PluginRepository,
)
from trading.contexts.extensions.domain import PluginEvent
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.integration import (
    DataSourceQueryRequest,
    RoehubDataFrame,
    dataframe_point_count,
    redact_data_frame,
)


class DataSourceQueryError(ValueError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DataSourceQueryService:
    """Session-scoped read-only data-source query boundary with hard response budgets."""

    def __init__(
        self,
        *,
        repository: PluginRepository,
        authorization: DataSourceAuthorization,
        invoker: DataSourceInvoker,
    ) -> None:
        self._repository = repository
        self._authorization = authorization
        self._invoker = invoker

    async def query(
        self,
        *,
        principal: CurrentUserPrincipal,
        instance_id: UUID,
        request: DataSourceQueryRequest,
        now: datetime | None = None,
    ) -> RoehubDataFrame:
        query_time = _utc(now or datetime.now(UTC))
        try:
            installation_id, organization_id = self._authorization.resolve_read_scope(
                principal=principal
            )
        except DataSourceAuthorizationError as error:
            raise DataSourceQueryError(
                code=error.code,
                message="Data-source organization scope is unavailable",
            ) from error
        instance = self._repository.get_instance(instance_id=instance_id)
        if instance is None or instance.organization_id != organization_id:
            raise DataSourceQueryError(
                code="data_source.not_found",
                message="Data-source instance is not found",
            )
        plugin_installation = self._repository.get_plugin_installation_by_id(
            plugin_installation_id=instance.plugin_installation_id
        )
        if (
            plugin_installation is None
            or plugin_installation.organization_id != organization_id
            or plugin_installation.installation_id != installation_id
            or plugin_installation.status != "enabled"
            or instance.status != "enabled"
            or "data.read" not in plugin_installation.granted_permissions
        ):
            raise DataSourceQueryError(
                code="data_source.capability_forbidden",
                message="Data-source read capability is not available",
            )
        package = self._repository.get_package(package_id=plugin_installation.package_id)
        manifest_spec = package.manifest.get("spec") if package is not None else None
        if (
            package is None
            or not isinstance(manifest_spec, Mapping)
            or manifest_spec.get("type") != "data-source"
        ):
            raise DataSourceQueryError(
                code="data_source.capability_forbidden",
                message="Plugin package is not a data source",
            )
        payload: Mapping[str, object] = {
            "contract": request.contract,
            "dataset": request.dataset,
            "dimensions": list(request.dimensions),
            "measures": list(request.measures),
            "filters": [item.model_dump(mode="json") for item in request.filters],
            "limits": {
                "rows": request.row_limit,
                "bytes": request.byte_limit,
                "points": request.point_limit,
                "timeout_ms": request.timeout_ms,
            },
            "read_only": True,
        }
        try:
            async with asyncio.timeout(request.timeout_ms / 1000):
                response = await self._invoker.query(
                    organization_id=organization_id,
                    instance_id=instance_id,
                    payload=payload,
                    timeout_seconds=request.timeout_ms / 1000,
                    response_byte_limit=request.byte_limit,
                )
        except TimeoutError as error:
            raise DataSourceQueryError(
                code="data_source.query_timeout",
                message="Data-source query exceeded its time budget",
            ) from error
        except DataSourceGatewayError as error:
            raise DataSourceQueryError(
                code=error.code,
                message="Data-source gateway rejected the query",
            ) from error
        frame = _validate_plugin_response(response=response)
        _enforce_frame_bounds(frame=frame, request=request)
        redacted = redact_data_frame(frame)
        self._repository.record_event(
            event=PluginEvent(
                event_id=uuid4(),
                installation_id=installation_id,
                organization_id=organization_id,
                actor_user_id=principal.user_id,
                event_type="plugin.data_source.queried",
                target_type="plugin_instance",
                target_id=str(instance_id),
                outcome="succeeded",
                metadata={
                    "dataset": request.dataset,
                    "rows": str(len(redacted.rows)),
                    "points": str(dataframe_point_count(redacted)),
                    "partial": str(redacted.partial).lower(),
                },
                created_at=query_time,
            )
        )
        return redacted


def _validate_plugin_response(*, response: Mapping[str, Any]) -> RoehubDataFrame:
    if (
        response.get("contract") != "PluginResponse/v1alpha1"
        or response.get("status") not in {"succeeded", "partial"}
        or not isinstance(response.get("frame"), Mapping)
    ):
        raise DataSourceQueryError(
            code="data_source.response_invalid",
            message="Data-source response does not contain RoehubDataFrame/v1",
        )
    try:
        frame = RoehubDataFrame.model_validate(response["frame"])
    except ValidationError as error:
        raise DataSourceQueryError(
            code="data_source.response_invalid",
            message="Data-source frame is invalid",
        ) from error
    if (response["status"] == "partial") != frame.partial:
        raise DataSourceQueryError(
            code="data_source.response_invalid",
            message="Data-source partial status is inconsistent",
        )
    return frame


def _enforce_frame_bounds(
    *, frame: RoehubDataFrame, request: DataSourceQueryRequest
) -> None:
    expected_columns = set(request.dimensions) | set(request.measures)
    columns_by_key = {column.key: column for column in frame.columns}
    actual_columns = set(columns_by_key)
    if actual_columns != expected_columns:
        raise DataSourceQueryError(
            code="data_source.response_fields_mismatch",
            message="Data-source returned fields outside the bounded query",
        )
    if any(
        columns_by_key[key].role != "dimension" for key in request.dimensions
    ) or any(columns_by_key[key].role != "measure" for key in request.measures):
        raise DataSourceQueryError(
            code="data_source.response_fields_mismatch",
            message="Data-source returned roles outside the bounded query",
        )
    if len(frame.rows) > request.row_limit:
        raise DataSourceQueryError(
            code="data_source.response_too_large",
            message="Data-source exceeded the row limit",
        )
    if len(frame.rows) * len(request.measures) > request.point_limit:
        raise DataSourceQueryError(
            code="data_source.response_too_large",
            message="Data-source exceeded the point limit",
        )
    frame_bytes = len(
        json.dumps(
            frame.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if frame_bytes > request.byte_limit:
        raise DataSourceQueryError(
            code="data_source.response_too_large",
            message="Data-source exceeded the byte limit",
        )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("data-source timestamp must be timezone-aware")
    return value.astimezone(UTC)
