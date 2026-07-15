from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trading.integration import (
    PluginRpcClient,
    PluginRpcError,
    PluginServiceIdentitySigner,
    PluginServiceIdentityVerifier,
)


def test_short_lived_identity_is_scoped_and_rpc_negotiates_protocol() -> None:
    now = datetime(2026, 7, 13, tzinfo=UTC)
    signing_key = Ed25519PrivateKey.generate()
    signer = PluginServiceIdentitySigner(private_key=signing_key, key_id="gateway-fixture")
    verifier = PluginServiceIdentityVerifier(
        public_keys={"gateway-fixture": signing_key.public_key()}
    )
    organization_id = uuid4()
    instance_id = uuid4()
    package_digest = "1" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Roehub-Plugin-Protocol"] == "roehub.plugin.rpc/v1alpha1"
        scheme, identity = request.headers["Authorization"].split(" ", 1)
        assert scheme == "RoehubPluginIdentity"
        verifier.verify(
            identity=identity,
            expected_organization_id=organization_id,
            expected_instance_id=instance_id,
            expected_package_digest=package_digest,
            expected_package_version="0.1.0",
            expected_capability="data.read",
            now=now,
        )
        return httpx.Response(
            200,
            headers={"X-Roehub-Plugin-Protocol": "roehub.plugin.rpc/v1alpha1"},
            json={"contract": "PluginResponse/v1alpha1", "status": "ready"},
        )

    client = PluginRpcClient(
        base_url="http://plugin-runtime:8080",
        signer=signer,
        organization_id=organization_id,
        instance_id=instance_id,
        package_digest=package_digest,
        package_version="0.1.0",
        granted_capabilities=frozenset({"data.read"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        assert client.query_data(request={"symbol": "FIXTURE"}, now=now)["status"] == "ready"
    finally:
        client.close()

    identity = signer.issue(
        organization_id=organization_id,
        instance_id=instance_id,
        package_digest=package_digest,
        package_version="0.1.0",
        capability="data.read",
        now=now,
    )
    with pytest.raises(PluginRpcError) as error:
        verifier.verify(
            identity=identity,
            expected_organization_id=uuid4(),
            expected_instance_id=instance_id,
            expected_package_digest=package_digest,
            expected_package_version="0.1.0",
            expected_capability="data.read",
            now=now + timedelta(seconds=1),
        )
    assert error.value.code == "plugin.identity_scope_mismatch"

    replayed_identity = signer.issue(
        organization_id=organization_id,
        instance_id=instance_id,
        package_digest=package_digest,
        package_version="0.1.0",
        capability="data.read",
        now=now,
    )
    verifier.verify(
        identity=replayed_identity,
        expected_organization_id=organization_id,
        expected_instance_id=instance_id,
        expected_package_digest=package_digest,
        expected_package_version="0.1.0",
        expected_capability="data.read",
        now=now,
    )
    with pytest.raises(PluginRpcError) as replay_error:
        verifier.verify(
            identity=replayed_identity,
            expected_organization_id=organization_id,
            expected_instance_id=instance_id,
            expected_package_digest=package_digest,
            expected_package_version="0.1.0",
            expected_capability="data.read",
            now=now,
        )
    assert replay_error.value.code == "plugin.identity_replayed"


def test_rpc_client_has_no_generic_execute_surface() -> None:
    assert not hasattr(PluginRpcClient, "execute")


def test_rpc_client_stops_streaming_before_response_byte_budget_is_exceeded() -> None:
    class ChunkedStream(httpx.SyncByteStream):
        def __init__(self) -> None:
            self.yielded = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            for _ in range(3):
                self.yielded += 1
                yield b"x" * 40_000

    stream = ChunkedStream()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-Roehub-Plugin-Protocol": "roehub.plugin.rpc/v1alpha1"},
            stream=stream,
        )

    now = datetime(2026, 7, 13, tzinfo=UTC)
    signer = PluginServiceIdentitySigner(
        private_key=Ed25519PrivateKey.generate(),
        key_id="gateway-fixture",
    )
    client = PluginRpcClient(
        base_url="http://plugin-runtime:8080",
        signer=signer,
        organization_id=uuid4(),
        instance_id=uuid4(),
        package_digest="1" * 64,
        package_version="0.1.0",
        granted_capabilities=frozenset({"data.read"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(PluginRpcError) as error:
            client.query_data(
                request={
                    "limits": {
                        "rows": 1,
                        "bytes": 1024,
                        "points": 1,
                        "timeout_ms": 100,
                    }
                },
                now=now,
            )
    finally:
        client.close()

    assert error.value.code == "plugin.rpc_response_too_large"
    assert stream.yielded == 2
