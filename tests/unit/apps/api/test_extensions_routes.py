from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes.extensions import build_extensions_router
from apps.api.wiring.modules.extensions import build_extensions_api_module
from apps.cli.commands.plugins import PluginsCli
from trading.contexts.extensions.adapters import InMemoryPluginRepository
from trading.contexts.extensions.application import PluginBundleValidator, PluginLifecycleService
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.integration import RoehubDataFrame
from trading.shared_kernel.primitives import InstallationId, OrganizationId, PaidLevel, UserId


class _Authorization:
    def __init__(self, installation_id: InstallationId) -> None:
        self.installation_id = installation_id

    def require_manage(
        self, *, principal: CurrentUserPrincipal, organization_id: OrganizationId
    ) -> InstallationId:
        _ = principal, organization_id
        return self.installation_id

    def require_read(
        self, *, principal: CurrentUserPrincipal, organization_id: OrganizationId
    ) -> InstallationId:
        _ = principal, organization_id
        return self.installation_id


class _DataSourceService:
    async def query(self, **_kwargs: object) -> RoehubDataFrame:
        return RoehubDataFrame.model_validate(
            {
                "contract": "RoehubDataFrame/v1",
                "frame_id": "fixture.frame",
                "title": "Fixture",
                "columns": [
                    {
                        "key": "timestamp",
                        "label": "Time",
                        "data_type": "timestamp",
                        "role": "dimension",
                        "unit": {"kind": "timestamp", "symbol": "UTC", "scale": 1},
                        "nullable": False,
                    },
                    {
                        "key": "pnl",
                        "label": "PnL",
                        "data_type": "number",
                        "role": "measure",
                        "unit": {"kind": "currency", "symbol": "USD", "scale": 1},
                        "nullable": False,
                    },
                ],
                "rows": [{"timestamp": "2026-07-13T10:00:00Z", "pnl": 1.0}],
                "metadata": {
                    "source_label": "Fixture",
                    "query_label": "PnL",
                    "generated_at": "2026-07-13T10:00:00Z",
                    "attributes": {},
                },
                "freshness": {
                    "status": "fresh",
                    "observed_at": "2026-07-13T10:00:00Z",
                    "age_seconds": 0,
                    "max_age_seconds": 60,
                },
                "notices": [],
                "partial": False,
                "errors": [],
            }
        )


def test_extensions_api_is_async_idempotent_and_has_no_generic_execute(tmp_path: Path) -> None:
    bundle = tmp_path / "fixture-bundle"
    assert PluginsCli(environ={}).run(["init", str(bundle)]) == 0
    repo_root = Path(__file__).resolve().parents[4]
    validator = PluginBundleValidator(
        schema_path=repo_root
        / "schemas/plugins/roehub-plugin-manifest-v1alpha1.schema.json",
        trusted_publisher_keys={},
        roehub_version="0.1.0",
        supported_architectures=frozenset({"linux/amd64", "linux/arm64"}),
        allow_unsigned_development=True,
        trading_mode="paper",
    )
    service = PluginLifecycleService(
        repository=InMemoryPluginRepository(),
        authorization=_Authorization(InstallationId(uuid4())),
    )
    principal = CurrentUserPrincipal(
        user_id=UserId(uuid4()),
        paid_level=PaidLevel("free"),
        session_created_at=datetime.now(UTC),
    )

    def current_user() -> CurrentUserPrincipal:
        return principal

    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_extensions_router(
            service=service,
            validator=validator,
            bundle_spool_root=tmp_path,
            current_user_dependency=current_user,  # type: ignore[arg-type]
        )
    )
    client = TestClient(app)
    organization_id = uuid4()
    request = {
        "bundle_id": bundle.name,
        "instance_name": "Fixture",
        "permissions": ["data.read"],
        "config": {},
    }
    headers = {
        "Idempotency-Key": "extensions-api-0001",
        "Origin": "http://testserver",
    }

    first = client.post(
        f"/api/v1/organizations/{organization_id}/plugins/installations",
        json=request,
        headers=headers,
    )
    second = client.post(
        f"/api/v1/organizations/{organization_id}/plugins/installations",
        json=request,
        headers=headers,
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["operation_id"] == second.json()["operation_id"]
    assert first.json()["status"] == "pending"
    paths = app.openapi()["paths"]
    assert all("/execute" not in path for path in paths)
    assert all("trust" not in path for path in paths)

    csrf_rejected = client.post(
        f"/api/v1/organizations/{organization_id}/plugins/installations",
        json=request,
        headers={"Idempotency-Key": "extensions-api-0002"},
    )
    assert csrf_rejected.status_code == 403


def test_data_source_query_api_derives_scope_from_session_and_is_read_only(
    tmp_path: Path,
) -> None:
    principal = CurrentUserPrincipal(
        user_id=UserId(uuid4()),
        paid_level=PaidLevel("free"),
        session_created_at=datetime.now(UTC),
    )

    def current_user() -> CurrentUserPrincipal:
        return principal

    validator = PluginBundleValidator(
        schema_path=Path(__file__).resolve().parents[4]
        / "schemas/plugins/roehub-plugin-manifest-v1alpha1.schema.json",
        trusted_publisher_keys={},
        roehub_version="0.1.0",
        supported_architectures=frozenset({"linux/amd64", "linux/arm64"}),
        allow_unsigned_development=True,
        trading_mode="paper",
    )
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_extensions_router(
            service=PluginLifecycleService(
                repository=InMemoryPluginRepository(),
                authorization=_Authorization(InstallationId(uuid4())),
            ),
            validator=validator,
            bundle_spool_root=tmp_path,
            current_user_dependency=current_user,  # type: ignore[arg-type]
            data_source_service=_DataSourceService(),  # type: ignore[arg-type]
        )
    )
    client = TestClient(app)
    instance_id = uuid4()
    payload = {
        "contract": "DataSourceQuery/v1",
        "dataset": "portfolio.pnl",
        "dimensions": ["timestamp"],
        "measures": ["pnl"],
        "read_only": True,
    }

    response = client.post(
        f"/api/v1/plugins/data-sources/{instance_id}:query",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["contract"] == "RoehubDataFrame/v1"
    paths = app.openapi()["paths"]
    query_path = "/api/v1/plugins/data-sources/{instance_id}:query"
    assert query_path in paths
    assert "organization_id" not in json.dumps(paths[query_path])

    write_attempt = client.post(
        f"/api/v1/plugins/data-sources/{instance_id}:query",
        json={**payload, "read_only": False},
    )
    assert write_attempt.status_code == 422


def test_extensions_api_fails_closed_without_production_trust_configuration() -> None:
    with pytest.raises(
        ValueError,
        match="ROEHUB_PLUGIN_PUBLISHER_KEYS_FILE is required in prod",
    ):
        build_extensions_api_module(
            environ={"ROEHUB_ENV": "prod"},
            current_user_dependency=lambda: None,  # type: ignore[arg-type]
            organization_repository=object(),  # type: ignore[arg-type]
        )


def test_extensions_api_fails_closed_without_production_database(tmp_path: Path) -> None:
    public_key = Ed25519PrivateKey.generate().public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    publisher_keys = tmp_path / "publisher-keys.json"
    publisher_keys.write_text(
        json.dumps(
            {
                "contract": "PluginPublisherKeys/v1alpha1",
                "keys": {"fixture-key": base64.b64encode(public_key).decode("ascii")},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="EXTENSIONS_PG_DSN is required in prod"):
        build_extensions_api_module(
            environ={
                "ROEHUB_ENV": "prod",
                "ROEHUB_PLUGIN_PUBLISHER_KEYS_FILE": str(publisher_keys),
            },
            current_user_dependency=lambda: None,  # type: ignore[arg-type]
            organization_repository=object(),  # type: ignore[arg-type]
        )
