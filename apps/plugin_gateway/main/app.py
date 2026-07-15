from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import UUID

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from trading.integration import PluginRpcClient, PluginRpcError, PluginServiceIdentitySigner


@dataclass(frozen=True, slots=True)
class PluginGatewayRegistration:
    organization_id: UUID
    instance_id: UUID
    package_digest: str
    package_version: str
    runtime_base_url: str
    granted_capabilities: frozenset[str]


class CapabilityRequest(BaseModel):
    organization_id: UUID
    instance_id: UUID
    payload: dict[str, Any] = Field(default_factory=dict)


class CapabilityResponse(BaseModel):
    contract: str = "PluginGatewayResponse/v1alpha1"
    result: dict[str, Any]


def build_plugin_gateway_app(
    *,
    registrations: Mapping[UUID, PluginGatewayRegistration],
    signer: PluginServiceIdentitySigner,
) -> FastAPI:
    """Build typed routes only; arbitrary execute, shell, mount, and env are absent."""

    app = FastAPI(title="Roehub Plugin Gateway", version="v1alpha1")

    def client_for(*, request: CapabilityRequest, capability: str) -> PluginRpcClient:
        registration = registrations.get(request.instance_id)
        if (
            registration is None
            or registration.organization_id != request.organization_id
            or capability not in registration.granted_capabilities
        ):
            raise HTTPException(status_code=403, detail="plugin capability is not granted")
        return PluginRpcClient(
            base_url=registration.runtime_base_url,
            signer=signer,
            organization_id=registration.organization_id,
            instance_id=registration.instance_id,
            package_digest=registration.package_digest,
            package_version=registration.package_version,
            granted_capabilities=registration.granted_capabilities,
        )

    @app.post(
        "/internal/plugin-rpc/v1alpha1/data-source/query",
        response_model=CapabilityResponse,
    )
    def query_data(request: CapabilityRequest) -> CapabilityResponse:
        client = client_for(request=request, capability="data.read")
        try:
            try:
                result = client.query_data(request=request.payload, now=datetime.now(UTC))
            except PluginRpcError as error:
                if error.code == "plugin.rpc_response_too_large":
                    raise HTTPException(
                        status_code=413,
                        detail="plugin response exceeded its byte budget",
                    ) from error
                raise HTTPException(status_code=502, detail="plugin RPC failed") from error
            return CapabilityResponse(result=dict(result))
        finally:
            client.close()

    @app.post(
        "/internal/plugin-rpc/v1alpha1/panel/describe",
        response_model=CapabilityResponse,
    )
    def describe_panel(request: CapabilityRequest) -> CapabilityResponse:
        client = client_for(request=request, capability="panel.describe")
        try:
            result = client.describe_panel(now=datetime.now(UTC))
            return CapabilityResponse(result=dict(result))
        finally:
            client.close()

    @app.post(
        "/internal/plugin-rpc/v1alpha1/app/action",
        response_model=CapabilityResponse,
    )
    def app_action(
        request: CapabilityRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
    ) -> CapabilityResponse:
        client = client_for(request=request, capability="app.action")
        try:
            result = client.invoke_app_action(
                request=request.payload,
                idempotency_key=idempotency_key,
                now=datetime.now(UTC),
            )
            return CapabilityResponse(result=dict(result))
        finally:
            client.close()

    @app.post(
        "/internal/plugin-rpc/v1alpha1/notification-provider/send",
        response_model=CapabilityResponse,
    )
    def send_notification(
        request: CapabilityRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
    ) -> CapabilityResponse:
        client = client_for(request=request, capability="notification.send")
        try:
            result = client.send_notification(
                request=request.payload,
                idempotency_key=idempotency_key,
                now=datetime.now(UTC),
            )
            return CapabilityResponse(result=dict(result))
        finally:
            client.close()

    return app


def build_empty_runtime_plugin_gateway_app() -> FastAPI:
    """Build a fail-closed clean-install gateway before plugin activation is configured."""

    app = build_plugin_gateway_app(
        registrations={},
        signer=PluginServiceIdentitySigner(
            private_key=Ed25519PrivateKey.generate(),
            key_id="clean-install-no-active-plugins",
        ),
    )

    @app.get("/health/live", include_in_schema=False)
    def health_live() -> dict[str, object]:
        return {"live": True}

    @app.get("/health/ready", include_in_schema=False)
    def health_ready() -> dict[str, object]:
        return {
            "ready": True,
            "status": "degraded",
            "reason": "no_active_plugin_registrations",
            "external_effects_enabled": False,
        }

    return app


app = build_empty_runtime_plugin_gateway_app()
