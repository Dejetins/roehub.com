from __future__ import annotations

import json
from typing import Any, Mapping, cast
from uuid import UUID

import httpx

from trading.contexts.extensions.application.ports import DataSourceGatewayError
from trading.shared_kernel.primitives import OrganizationId


class HttpPluginGatewayDataSourceInvoker:
    """Bounded internal client; browser credentials and data-source secrets never cross it."""

    def __init__(
        self,
        *,
        gateway_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not gateway_url.startswith(("http://", "https://")):
            raise ValueError("plugin gateway URL must use http or https")
        self._gateway_url = gateway_url.rstrip("/")
        self._transport = transport

    async def query(
        self,
        *,
        organization_id: OrganizationId,
        instance_id: UUID,
        payload: Mapping[str, object],
        timeout_seconds: float,
        response_byte_limit: int,
    ) -> Mapping[str, Any]:
        hard_response_limit = response_byte_limit + 65_536
        try:
            async with httpx.AsyncClient(
                base_url=self._gateway_url,
                timeout=timeout_seconds,
                transport=self._transport,
            ) as client:
                async with client.stream(
                    "POST",
                    "/internal/plugin-rpc/v1alpha1/data-source/query",
                    json={
                        "organization_id": str(organization_id),
                        "instance_id": str(instance_id),
                        "payload": dict(payload),
                    },
                ) as response:
                    if response.status_code == 403:
                        raise DataSourceGatewayError(
                            code="data_source.capability_forbidden"
                        )
                    if response.status_code == 413:
                        raise DataSourceGatewayError(
                            code="data_source.response_too_large"
                        )
                    if response.status_code >= 400:
                        raise DataSourceGatewayError(
                            code="data_source.gateway_unavailable"
                        )
                    content_length = response.headers.get("content-length")
                    if content_length is not None:
                        try:
                            declared_length = int(content_length)
                        except ValueError as error:
                            raise DataSourceGatewayError(
                                code="data_source.response_invalid"
                            ) from error
                        if declared_length < 0:
                            raise DataSourceGatewayError(
                                code="data_source.response_invalid"
                            )
                        if declared_length > hard_response_limit:
                            raise DataSourceGatewayError(
                                code="data_source.response_too_large"
                            )
                    body_bytes = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body_bytes) + len(chunk) > hard_response_limit:
                            raise DataSourceGatewayError(
                                code="data_source.response_too_large"
                            )
                        body_bytes.extend(chunk)
        except DataSourceGatewayError:
            raise
        except httpx.TimeoutException as error:
            raise DataSourceGatewayError(code="data_source.query_timeout") from error
        except httpx.HTTPError as error:
            raise DataSourceGatewayError(code="data_source.gateway_unavailable") from error
        try:
            body = json.loads(body_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DataSourceGatewayError(code="data_source.response_invalid") from error
        if (
            not isinstance(body, dict)
            or body.get("contract") != "PluginGatewayResponse/v1alpha1"
            or not isinstance(body.get("result"), dict)
        ):
            raise DataSourceGatewayError(code="data_source.response_invalid")
        return cast(Mapping[str, Any], body["result"])
