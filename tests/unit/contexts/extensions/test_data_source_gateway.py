from __future__ import annotations

from uuid import uuid4

import httpx
import pytest

from trading.contexts.extensions.adapters import HttpPluginGatewayDataSourceInvoker
from trading.contexts.extensions.application.ports import DataSourceGatewayError
from trading.shared_kernel.primitives import OrganizationId


@pytest.mark.asyncio
async def test_gateway_invoker_stops_streaming_at_hard_response_limit() -> None:
    class ChunkedStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.yielded = 0

        async def __aiter__(self):  # type: ignore[no-untyped-def]
            for _ in range(3):
                self.yielded += 1
                yield b"x" * 40_000

    stream = ChunkedStream()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    invoker = HttpPluginGatewayDataSourceInvoker(
        gateway_url="http://plugin-gateway:8080",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(DataSourceGatewayError) as error:
        await invoker.query(
            organization_id=OrganizationId(uuid4()),
            instance_id=uuid4(),
            payload={"contract": "DataSourceQuery/v1", "read_only": True},
            timeout_seconds=1,
            response_byte_limit=1024,
        )

    assert error.value.code == "data_source.response_too_large"
    assert stream.yielded == 2
