from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import UUID, uuid4

import pytest

from trading.contexts.extensions.adapters import InMemoryPluginRepository
from trading.contexts.extensions.application import (
    DataSourceQueryError,
    DataSourceQueryService,
)
from trading.contexts.extensions.domain import (
    PluginInstallation,
    PluginInstance,
    PluginPackage,
)
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.integration import DataSourceQueryRequest
from trading.shared_kernel.primitives import (
    InstallationId,
    OrganizationId,
    PaidLevel,
    UserId,
)


class _Authorization:
    def __init__(
        self, *, installation_id: InstallationId, organization_id: OrganizationId
    ) -> None:
        self.installation_id = installation_id
        self.organization_id = organization_id

    def resolve_read_scope(
        self, *, principal: CurrentUserPrincipal
    ) -> tuple[InstallationId, OrganizationId]:
        _ = principal
        return self.installation_id, self.organization_id


class _Invoker:
    def __init__(self, *, response: Mapping[str, Any]) -> None:
        self.response = response
        self.calls: list[Mapping[str, object]] = []

    async def query(
        self,
        *,
        organization_id: OrganizationId,
        instance_id: UUID,
        payload: Mapping[str, object],
        timeout_seconds: float,
        response_byte_limit: int,
    ) -> Mapping[str, Any]:
        _ = organization_id, instance_id, timeout_seconds, response_byte_limit
        self.calls.append(payload)
        return self.response


class _HangingInvoker:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def query(
        self,
        *,
        organization_id: OrganizationId,
        instance_id: UUID,
        payload: Mapping[str, object],
        timeout_seconds: float,
        response_byte_limit: int,
    ) -> Mapping[str, Any]:
        _ = organization_id, instance_id, payload, timeout_seconds, response_byte_limit
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("hanging invoker unexpectedly completed")


def _frame(
    *,
    extra_column: bool = False,
    measure_as_dimension: bool = False,
    secret_text: bool = False,
) -> dict[str, Any]:
    columns: list[dict[str, Any]] = [
        {
            "key": "timestamp",
            "label": "Time",
            "data_type": "timestamp",
            "role": "dimension",
            "unit": {"kind": "timestamp", "symbol": "UTC", "scale": 1.0},
            "nullable": False,
        },
        {
            "key": "pnl",
            "label": "PnL",
            "data_type": "number",
            "role": "dimension" if measure_as_dimension else "measure",
            "unit": {"kind": "currency", "symbol": "USD", "scale": 1.0},
            "nullable": False,
        },
    ]
    row: dict[str, Any] = {"timestamp": "2026-07-13T10:00:00Z", "pnl": 12.5}
    if secret_text:
        columns.append(
            {
                "key": "note",
                "label": "=".join(("token", "column-material")),
                "data_type": "string",
                "role": "dimension",
                "unit": None,
                "nullable": False,
            }
        )
        row["note"] = "=".join(("password", "row-material"))
    if extra_column:
        columns.append(
            {
                "key": "unexpected",
                "label": "Unexpected",
                "data_type": "string",
                "role": "dimension",
                "unit": None,
                "nullable": False,
            }
        )
        row["unexpected"] = "blocked"
    return {
        "contract": "RoehubDataFrame/v1",
        "frame_id": "fixture.frame",
        "title": (
            "=".join(("secret", "title-material"))
            if secret_text
            else "Fixture PnL"
        ),
        "columns": columns,
        "rows": [row],
        "metadata": {
            "source_label": (
                "=".join(("api_key", "source-material"))
                if secret_text
                else "External PostgreSQL fixture"
            ),
            "query_label": "PnL by minute",
            "generated_at": "2026-07-13T10:00:00Z",
            "attributes": (
                {"description": "=".join(("token", "metadata-material"))}
                if secret_text
                else {}
            ),
        },
        "freshness": {
            "status": "fresh",
            "observed_at": "2026-07-13T10:00:00Z",
            "age_seconds": 0,
            "max_age_seconds": 60,
        },
        "notices": [
            {
                "level": "warning",
                "code": "fixture.partial",
                "message": "=".join(("token", "fixture-material")),
            }
        ],
        "partial": True,
        "errors": [
            {
                "code": "fixture.segment_unavailable",
                "message": "One sanitized segment is unavailable",
                "retryable": True,
                "field": None,
            }
        ],
    }


def _fixture(
    *, invoker: Any, authorization_organization_id: OrganizationId | None = None
) -> tuple[DataSourceQueryService, CurrentUserPrincipal, UUID, InMemoryPluginRepository]:
    now = datetime(2026, 7, 13, tzinfo=UTC)
    installation_id = InstallationId(uuid4())
    organization_id = OrganizationId(uuid4())
    repository = InMemoryPluginRepository()
    package = PluginPackage(
        package_id=uuid4(),
        installation_id=installation_id,
        plugin_id="fixture.external-db",
        version="0.1.0",
        package_digest="1" * 64,
        image_reference="fixture/external-db:0.1.0",
        image_digest="sha256:" + "1" * 64,
        publisher_key_id=None,
        publisher_public_key_b64=None,
        publisher_key_fingerprint_sha256=None,
        manifest={"spec": {"type": "data-source"}},
        created_at=now,
    )
    repository.register_package(package=package, actor_user_id=UserId(uuid4()))
    plugin_installation = PluginInstallation(
        plugin_installation_id=uuid4(),
        installation_id=installation_id,
        organization_id=organization_id,
        plugin_id=package.plugin_id,
        package_id=package.package_id,
        previous_package_id=None,
        granted_permissions=("data.read",),
        status="enabled",
        created_at=now,
        updated_at=now,
    )
    instance = PluginInstance(
        instance_id=uuid4(),
        plugin_installation_id=plugin_installation.plugin_installation_id,
        installation_id=installation_id,
        organization_id=organization_id,
        name="External fixture",
        config={},
        config_revision=1,
        status="enabled",
        created_at=now,
        updated_at=now,
    )
    repository.install_package(plugin_installation=plugin_installation, instance=instance)
    service = DataSourceQueryService(
        repository=repository,
        authorization=_Authorization(
            installation_id=installation_id,
            organization_id=authorization_organization_id or organization_id,
        ),
        invoker=invoker,
    )
    principal = CurrentUserPrincipal(
        user_id=UserId(uuid4()),
        paid_level=PaidLevel("free"),
        session_created_at=now,
    )
    return service, principal, instance.instance_id, repository


def _request(**overrides: Any) -> DataSourceQueryRequest:
    payload: dict[str, Any] = {
        "dataset": "portfolio.pnl",
        "dimensions": ["timestamp"],
        "measures": ["pnl"],
    }
    payload.update(overrides)
    return DataSourceQueryRequest.model_validate(payload)


@pytest.mark.asyncio
async def test_query_is_read_only_bounded_redacted_and_audited() -> None:
    response_frame = _frame(secret_text=True)
    invoker = _Invoker(
        response={
            "contract": "PluginResponse/v1alpha1",
            "status": "partial",
            "frame": response_frame,
        }
    )
    service, principal, instance_id, repository = _fixture(invoker=invoker)

    frame = await service.query(
        principal=principal,
        instance_id=instance_id,
        request=_request(dimensions=["timestamp", "note"]),
    )

    assert frame.partial is True
    assert frame.title == "[REDACTED]"
    assert frame.columns[2].label == "[REDACTED]"
    assert frame.rows[0]["note"] == "[REDACTED]"
    assert frame.metadata.source_label == "[REDACTED]"
    assert frame.metadata.attributes["description"] == "[REDACTED]"
    assert frame.notices[0].message == "[REDACTED]"
    assert invoker.calls[0]["read_only"] is True
    assert "organization_id" not in invoker.calls[0]
    instance = repository.get_instance(instance_id=instance_id)
    assert instance is not None
    events = repository.list_events(
        organization_id=instance.organization_id,
        limit=10,
    )
    assert events[0].event_type == "plugin.data_source.queried"
    assert events[0].metadata["rows"] == "1"


@pytest.mark.asyncio
async def test_cross_organization_instance_is_hidden_before_invocation() -> None:
    invoker = _Invoker(response={})
    service, principal, instance_id, _repository = _fixture(
        invoker=invoker,
        authorization_organization_id=OrganizationId(uuid4()),
    )

    with pytest.raises(DataSourceQueryError) as error:
        await service.query(
            principal=principal,
            instance_id=instance_id,
            request=_request(),
        )

    assert error.value.code == "data_source.not_found"
    assert invoker.calls == []


@pytest.mark.asyncio
async def test_unrequested_fields_are_rejected_after_plugin_response() -> None:
    invoker = _Invoker(
        response={
            "contract": "PluginResponse/v1alpha1",
            "status": "partial",
            "frame": _frame(extra_column=True),
        }
    )
    service, principal, instance_id, _repository = _fixture(invoker=invoker)

    with pytest.raises(DataSourceQueryError) as error:
        await service.query(
            principal=principal,
            instance_id=instance_id,
            request=_request(),
        )

    assert error.value.code == "data_source.response_fields_mismatch"


@pytest.mark.asyncio
async def test_measure_role_cannot_bypass_point_limit() -> None:
    invoker = _Invoker(
        response={
            "contract": "PluginResponse/v1alpha1",
            "status": "partial",
            "frame": _frame(measure_as_dimension=True),
        }
    )
    service, principal, instance_id, _repository = _fixture(invoker=invoker)

    with pytest.raises(DataSourceQueryError) as error:
        await service.query(
            principal=principal,
            instance_id=instance_id,
            request=_request(point_limit=1),
        )

    assert error.value.code == "data_source.response_fields_mismatch"


@pytest.mark.asyncio
async def test_timeout_and_caller_cancellation_propagate_to_invoker() -> None:
    timeout_invoker = _HangingInvoker()
    service, principal, instance_id, _repository = _fixture(invoker=timeout_invoker)
    with pytest.raises(DataSourceQueryError) as timeout_error:
        await service.query(
            principal=principal,
            instance_id=instance_id,
            request=_request(timeout_ms=50),
        )
    assert timeout_error.value.code == "data_source.query_timeout"
    assert timeout_invoker.cancelled.is_set()

    cancellation_invoker = _HangingInvoker()
    service, principal, instance_id, _repository = _fixture(
        invoker=cancellation_invoker
    )
    task = asyncio.create_task(
        service.query(
            principal=principal,
            instance_id=instance_id,
            request=_request(timeout_ms=5000),
        )
    )
    await cancellation_invoker.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancellation_invoker.cancelled.is_set()
