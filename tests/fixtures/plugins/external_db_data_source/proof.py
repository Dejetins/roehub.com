from __future__ import annotations

import base64
import hashlib
import json
import shutil
import socket
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

import httpx
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.common.errors import register_api_error_handlers
from apps.api.routes.extensions import build_extensions_router
from apps.plugin_gateway.main.app import (
    PluginGatewayRegistration,
    build_plugin_gateway_app,
)
from trading.contexts.extensions.adapters import (
    HttpPluginGatewayDataSourceInvoker,
    IdentityPluginAuthorization,
    InMemoryPluginRepository,
)
from trading.contexts.extensions.application import (
    DataSourceQueryService,
    PluginBundleValidator,
    PluginLifecycleService,
    canonical_package_digest,
    sign_package_digest,
)
from trading.contexts.extensions.domain import (
    PluginInstallation,
    PluginInstance,
    PluginPackage,
    ValidatedPluginBundle,
)
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryOrganizationRepository,
)
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.integration import (
    PluginRpcClient,
    PluginRpcError,
    PluginServiceIdentitySigner,
    RoehubAppContribution,
    RoehubDataFrame,
    RoehubPanelContribution,
)
from trading.shared_kernel.primitives import (
    InstallationId,
    OrganizationId,
    PaidLevel,
    UserId,
)

_POSTGRES_IMAGE = (
    "postgres:16@sha256:be01cf82fc7dbba824acf0a82e150b4b360f3ff93c6631d7844af431e841a95c"
)


class Stage13ProofError(RuntimeError):
    pass


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise Stage13ProofError(
            f"command failed without publishing captured output: {command[0]}"
        )
    return result


def _wait_for_database(*, container: str, cwd: Path) -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        result = _run(
            ["docker", "exec", container, "pg_isready", "-U", "postgres"],
            cwd=cwd,
            check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(0.5)
    raise Stage13ProofError("external PostgreSQL fixture did not become ready")


def _artifact(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _build_bundle(
    *,
    root: Path,
    image_reference: str,
    image_digest: str,
    architecture: str,
    signing_key: Ed25519PrivateKey,
) -> Path:
    root.mkdir()
    (root / "config.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "LICENSE").write_text("Apache-2.0 fixture declaration\n", encoding="utf-8")
    (root / "sbom.spdx.json").write_text(
        json.dumps(
            {
                "spdxVersion": "SPDX-2.3",
                "dataLicense": "CC0-1.0",
                "SPDXID": "SPDXRef-DOCUMENT",
                "name": "roehub-stage13-external-data-source-fixture",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "apiVersion": "roehub.io/v1alpha1",
        "kind": "Plugin",
        "metadata": {
            "id": "stage13.external-data-source",
            "version": "0.1.0",
            "publisher": "stage13.publisher",
        },
        "spec": {
            "type": "data-source",
            "pluginApi": "v1alpha1",
            "rpc": {"version": "roehub.plugin.rpc/v1alpha1", "port": 8080},
            "image": {
                "reference": image_reference,
                "digest": image_digest,
                "architectures": [architecture],
            },
            "compatibility": {"roehubMin": "0.1.0", "roehubMaxExclusive": "0.2.0"},
            "permissions": [
                {"capability": "data.read"},
                {"capability": "panel.describe"},
            ],
            "configSchema": _artifact(root / "config.schema.json"),
            "license": {**_artifact(root / "LICENSE"), "spdx": "Apache-2.0"},
            "sbom": _artifact(root / "sbom.spdx.json"),
            "runtime": {
                "nonRootUid": 10001,
                "readOnlyRootFilesystem": True,
                "noNewPrivileges": True,
                "resources": {"cpus": 0.5, "memoryMb": 192, "pids": 64},
                "egress": [{"host": "stage13-external-db", "port": 5432}],
            },
        },
        "signature": {
            "algorithm": "Ed25519",
            "keyId": "stage13-publisher",
            "value": "placeholder",
        },
    }
    signature = manifest["signature"]
    assert isinstance(signature, dict)
    signature["value"] = sign_package_digest(
        private_key=signing_key,
        package_digest=canonical_package_digest(manifest),
    )
    (root / "roehub.plugin.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return root


def _docker_inspect(image: str, *, cwd: Path) -> Mapping[str, Any]:
    result = _run(["docker", "image", "inspect", image], cwd=cwd)
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise Stage13ProofError("Docker image inspection has invalid shape")
    return payload[0]


def _query_payload(*, row_limit: int, dataset: str = "portfolio.pnl") -> dict[str, Any]:
    return {
        "contract": "DataSourceQuery/v1",
        "dataset": dataset,
        "dimensions": ["timestamp"],
        "measures": ["pnl", "drawdown"],
        "filters": [],
        "limits": {
            "rows": row_limit,
            "bytes": 262144,
            "points": 1000,
            "timeout_ms": 100 if dataset == "portfolio.slow" else 3000,
        },
        "read_only": True,
    }


def _public_query_payload(
    *,
    dataset: str,
    row_limit: int,
    byte_limit: int = 262_144,
    point_limit: int = 1_000,
    timeout_ms: int = 3_000,
) -> dict[str, Any]:
    return {
        "contract": "DataSourceQuery/v1",
        "dataset": dataset,
        "dimensions": ["timestamp"],
        "measures": ["pnl", "drawdown"],
        "filters": [],
        "row_limit": row_limit,
        "byte_limit": byte_limit,
        "point_limit": point_limit,
        "timeout_ms": timeout_ms,
        "read_only": True,
    }


def _host_chain_proof(
    *,
    repo_root: Path,
    identity_repository: InMemoryOrganizationRepository,
    owner_user_id: UserId,
    foreign_user_id: UserId,
    installation_id: InstallationId,
    plugin_repository: InMemoryPluginRepository,
    validator: PluginBundleValidator,
    validated: ValidatedPluginBundle,
    instance_id: UUID,
    organization_id: OrganizationId,
    runtime_base_url: str,
    gateway_signer: PluginServiceIdentitySigner,
) -> dict[str, str]:
    authorization = IdentityPluginAuthorization(repository=identity_repository)
    now = datetime.now(UTC)
    package = PluginPackage(
        package_id=uuid4(),
        installation_id=installation_id,
        plugin_id=validated.manifest.plugin_id,
        version=validated.manifest.version,
        package_digest=validated.manifest.package_digest,
        image_reference=validated.manifest.image_reference,
        image_digest=validated.manifest.image_digest,
        publisher_key_id=validated.manifest.publisher_key_id,
        publisher_public_key_b64=validated.publisher_public_key_b64,
        publisher_key_fingerprint_sha256=(
            validated.publisher_key_fingerprint_sha256
        ),
        manifest=validated.manifest.raw,
        created_at=now,
    )
    plugin_repository.register_package(package=package, actor_user_id=owner_user_id)
    plugin_installation = PluginInstallation(
        plugin_installation_id=uuid4(),
        installation_id=package.installation_id,
        organization_id=organization_id,
        plugin_id=package.plugin_id,
        package_id=package.package_id,
        previous_package_id=None,
        granted_permissions=("data.read", "panel.describe"),
        status="enabled",
        created_at=now,
        updated_at=now,
    )
    instance = PluginInstance(
        instance_id=instance_id,
        plugin_installation_id=plugin_installation.plugin_installation_id,
        installation_id=package.installation_id,
        organization_id=organization_id,
        name="Stage 13 external database",
        config={},
        config_revision=1,
        status="enabled",
        created_at=now,
        updated_at=now,
    )
    plugin_repository.install_package(
        plugin_installation=plugin_installation,
        instance=instance,
    )
    gateway_app = build_plugin_gateway_app(
        registrations={
            instance_id: PluginGatewayRegistration(
                organization_id=organization_id.value,
                instance_id=instance_id,
                package_digest=validated.manifest.package_digest,
                package_version=validated.manifest.version,
                runtime_base_url=runtime_base_url,
                granted_capabilities=frozenset({"data.read", "panel.describe"}),
            )
        },
        signer=gateway_signer,
    )
    data_source_service = DataSourceQueryService(
        repository=plugin_repository,
        authorization=authorization,
        invoker=HttpPluginGatewayDataSourceInvoker(
            gateway_url="http://stage13-plugin-gateway",
            transport=httpx.ASGITransport(app=gateway_app),
        ),
    )
    lifecycle_service = PluginLifecycleService(
        repository=plugin_repository,
        authorization=authorization,
    )
    active_principal = {
        "value": CurrentUserPrincipal(
            user_id=owner_user_id,
            paid_level=PaidLevel("free"),
            session_created_at=now,
        )
    }

    def current_user() -> CurrentUserPrincipal:
        return active_principal["value"]

    api_app = FastAPI()
    register_api_error_handlers(app=api_app)
    api_app.include_router(
        build_extensions_router(
            service=lifecycle_service,
            validator=validator,
            bundle_spool_root=repo_root,
            current_user_dependency=current_user,  # type: ignore[arg-type]
            data_source_service=data_source_service,
        )
    )
    query_path = f"/api/v1/plugins/data-sources/{instance_id}:query"
    with TestClient(api_app) as client:
        bounded = client.post(
            query_path,
            json=_public_query_payload(dataset="portfolio.pnl", row_limit=2),
        )
        if bounded.status_code != 200 or len(bounded.json().get("rows", [])) != 2:
            raise Stage13ProofError("host API did not enforce bounded data-source query")

        oversized = client.post(
            query_path,
            json=_public_query_payload(
                dataset="portfolio.oversized",
                row_limit=1,
                byte_limit=1_024,
                point_limit=2,
            ),
        )
        if (
            oversized.status_code != 502
            or oversized.json().get("error", {}).get("code")
            != "data_source.response_too_large"
        ):
            raise Stage13ProofError("host chain accepted an oversized plugin stream")

        started = time.monotonic()
        non_cooperative = client.post(
            query_path,
            json=_public_query_payload(
                dataset="portfolio.ignore-timeout",
                row_limit=1,
                timeout_ms=100,
            ),
        )
        elapsed = time.monotonic() - started
        if (
            non_cooperative.status_code != 504
            or non_cooperative.json().get("error", {}).get("code")
            != "data_source.query_timeout"
            or elapsed >= 1.0
        ):
            raise Stage13ProofError("host timeout did not bound a non-cooperative plugin")

        active_principal["value"] = CurrentUserPrincipal(
            user_id=foreign_user_id,
            paid_level=PaidLevel("free"),
            session_created_at=now,
        )
        foreign = client.post(
            query_path,
            json=_public_query_payload(dataset="portfolio.pnl", row_limit=1),
        )
        if (
            foreign.status_code != 404
            or foreign.json().get("error", {}).get("code") != "data_source.not_found"
        ):
            raise Stage13ProofError("host API exposed a cross-organization instance")

    return {
        "api_service_gateway_plugin_chain": "passed",
        "stream_byte_limit": "passed",
        "non_cooperative_timeout": "passed",
        "session_scope_denial": "passed",
    }


def run_proof(repo_root: Path) -> dict[str, object]:
    if shutil.which("docker") is None:
        raise Stage13ProofError("docker executable is unavailable")
    suffix = uuid4().hex[:8]
    data_network = f"roehub-stage13-data-{suffix}"
    gateway_network = f"roehub-stage13-gateway-{suffix}"
    database_container = f"roehub-stage13-db-{suffix}"
    plugin_container = f"roehub-stage13-plugin-{suffix}"
    plugin_image = f"roehub-stage13-data-source:{suffix}"
    fixture_root = repo_root / "tests/fixtures/plugins/external_db_data_source"
    owner_user_id = UserId(uuid4())
    foreign_user_id = UserId(uuid4())
    identity_repository = InMemoryOrganizationRepository()
    installation, organization = identity_repository.bootstrap_installation(
        owner_user_id=owner_user_id,
        installation_name="Stage 13 proof",
        organization_slug="stage13-owner",
        organization_name="Stage 13 owner",
        created_at=datetime.now(UTC),
    )
    foreign_organization = identity_repository.create_organization(
        actor_user_id=foreign_user_id,
        slug="stage13-foreign",
        display_name="Stage 13 foreign",
        created_at=datetime.now(UTC),
    )
    organization_id = organization.organization_id
    foreign_organization_id = foreign_organization.organization_id
    instance_id = uuid4()
    gateway_key = Ed25519PrivateKey.generate()
    publisher_key = Ed25519PrivateKey.generate()
    cleanup = False
    image_id = ""
    host_chain: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="roehub-stage13-") as temporary:
        temporary_root = Path(temporary)
        image_context = temporary_root / "image"
        image_context.mkdir()
        shutil.copy2(fixture_root / "Dockerfile", image_context / "Dockerfile")
        shutil.copy2(fixture_root / "server.py", image_context / "server.py")
        gateway_public = gateway_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        (image_context / "fixture-config.json").write_text(
            json.dumps(
                {
                    "organization_id": str(organization_id),
                    "instance_id": str(instance_id),
                    "public_key_b64": base64.b64encode(gateway_public).decode("ascii"),
                    "database_dsn": (
                        "postgresql://stage13_reader@stage13-external-db/postgres"
                    ),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        try:
            _run(
                ["docker", "network", "create", "--internal", data_network],
                cwd=repo_root,
            )
            _run(
                [
                    "docker",
                    "network",
                    "create",
                    "--driver",
                    "bridge",
                    "--opt",
                    "com.docker.network.bridge.enable_ip_masquerade=false",
                    gateway_network,
                ],
                cwd=repo_root,
            )
            _run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    database_container,
                    "--network",
                    data_network,
                    "--network-alias",
                    "stage13-external-db",
                    "--env",
                    "POSTGRES_HOST_AUTH_METHOD=trust",
                    _POSTGRES_IMAGE,
                ],
                cwd=repo_root,
            )
            _wait_for_database(container=database_container, cwd=repo_root)
            setup_sql = f"""
                CREATE TABLE stage13_portfolio_points (
                    organization_id uuid NOT NULL,
                    observed_at timestamptz NOT NULL,
                    pnl numeric NOT NULL,
                    drawdown numeric NOT NULL
                );
                INSERT INTO stage13_portfolio_points VALUES
                    ('{organization_id}', '2026-07-13T10:00:00Z', 10.0, -1.0),
                    ('{organization_id}', '2026-07-13T10:05:00Z', 20.0, -2.0),
                    ('{organization_id}', '2026-07-13T10:10:00Z', 30.0, -3.0),
                    ('{foreign_organization_id}', '2026-07-13T10:00:00Z', 999999.0, -99.0),
                    ('{foreign_organization_id}', '2026-07-13T10:05:00Z', 999999.0, -99.0);
                CREATE ROLE stage13_reader LOGIN;
                ALTER ROLE stage13_reader SET default_transaction_read_only = on;
                GRANT CONNECT ON DATABASE postgres TO stage13_reader;
                GRANT USAGE ON SCHEMA public TO stage13_reader;
                GRANT SELECT ON stage13_portfolio_points TO stage13_reader;
            """
            _run(
                [
                    "docker",
                    "exec",
                    "-i",
                    database_container,
                    "psql",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-U",
                    "postgres",
                ],
                cwd=repo_root,
                input_text=setup_sql,
            )
            write_attempt = _run(
                [
                    "docker",
                    "exec",
                    database_container,
                    "psql",
                    "-h",
                    "127.0.0.1",
                    "-U",
                    "stage13_reader",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-c",
                    "UPDATE stage13_portfolio_points SET pnl = 0",
                ],
                cwd=repo_root,
                check=False,
            )
            if write_attempt.returncode == 0:
                raise Stage13ProofError("read-only external database role accepted a write")

            _run(["docker", "build", "--tag", plugin_image, str(image_context)], cwd=repo_root)
            image = _docker_inspect(plugin_image, cwd=repo_root)
            image_id = str(image["Id"])
            architecture = "linux/" + str(image["Architecture"])
            bundle = _build_bundle(
                root=temporary_root / "bundle",
                image_reference=plugin_image,
                image_digest=image_id,
                architecture=architecture,
                signing_key=publisher_key,
            )
            validator = PluginBundleValidator(
                schema_path=repo_root
                / "schemas/plugins/roehub-plugin-manifest-v1alpha1.schema.json",
                trusted_publisher_keys={"stage13-publisher": publisher_key.public_key()},
                roehub_version="0.1.0",
                supported_architectures=frozenset({"linux/amd64", "linux/arm64"}),
                trading_mode="testnet",
            )
            validated = validator.validate(bundle)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_probe:
                port_probe.bind(("127.0.0.1", 0))
                port = int(port_probe.getsockname()[1])
            _run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    plugin_container,
                    "--network",
                    gateway_network,
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--memory",
                    "192m",
                    "--cpus",
                    "0.5",
                    "--pids-limit",
                    "64",
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,size=16m",
                    "--user",
                    "10001:10001",
                    "--publish",
                    f"127.0.0.1:{port}:8080",
                    image_id,
                    "--package-digest",
                    validated.manifest.package_digest,
                    "--package-version",
                    validated.manifest.version,
                ],
                cwd=repo_root,
            )
            _run(
                [
                    "docker",
                    "network",
                    "connect",
                    "--alias",
                    "stage13-external-db",
                    data_network,
                    plugin_container,
                ],
                cwd=repo_root,
            )
            signer = PluginServiceIdentitySigner(
                private_key=gateway_key,
                key_id="stage13-gateway",
            )
            client = PluginRpcClient(
                base_url=f"http://127.0.0.1:{port}",
                signer=signer,
                organization_id=organization_id.value,
                instance_id=instance_id,
                package_digest=validated.manifest.package_digest,
                package_version=validated.manifest.version,
                granted_capabilities=frozenset({"data.read", "panel.describe"}),
                timeout_seconds=4.0,
            )
            try:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        if client.health(now=datetime.now(UTC))["status"] == "ready":
                            break
                    except PluginRpcError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.25)
                response = client.query_data(
                    request=_query_payload(row_limit=2),
                    now=datetime.now(UTC),
                )
                frame = RoehubDataFrame.model_validate(response["frame"])
                pnl_values = [row["pnl"] for row in frame.rows]
                if len(frame.rows) != 2 or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or value >= 1000
                    for value in pnl_values
                ):
                    raise Stage13ProofError("organization or row limit isolation failed")
                described = client.describe_panel(now=datetime.now(UTC))
                RoehubPanelContribution.model_validate(described["panel"])
                RoehubAppContribution.model_validate(described["app"])
                slow_started = time.monotonic()
                try:
                    client.query_data(
                        request=_query_payload(
                            row_limit=2,
                            dataset="portfolio.slow",
                        ),
                        now=datetime.now(UTC),
                    )
                except PluginRpcError as error:
                    if error.code != "plugin.rpc_rejected":
                        raise
                else:
                    raise Stage13ProofError("slow query was not cancelled")
                if time.monotonic() - slow_started >= 1.0:
                    raise Stage13ProofError("slow query exceeded cancellation budget")
            finally:
                client.close()

            host_chain = _host_chain_proof(
                repo_root=repo_root,
                identity_repository=identity_repository,
                owner_user_id=owner_user_id,
                foreign_user_id=foreign_user_id,
                installation_id=installation.installation_id,
                plugin_repository=InMemoryPluginRepository(),
                validator=validator,
                validated=validated,
                instance_id=instance_id,
                organization_id=organization_id,
                runtime_base_url=f"http://127.0.0.1:{port}",
                gateway_signer=signer,
            )

            foreign_client = PluginRpcClient(
                base_url=f"http://127.0.0.1:{port}",
                signer=signer,
                organization_id=foreign_organization_id.value,
                instance_id=instance_id,
                package_digest=validated.manifest.package_digest,
                package_version=validated.manifest.version,
                granted_capabilities=frozenset({"data.read"}),
                timeout_seconds=2.0,
            )
            try:
                try:
                    foreign_client.query_data(
                        request=_query_payload(row_limit=2),
                        now=datetime.now(UTC),
                    )
                except PluginRpcError as error:
                    if error.code != "plugin.rpc_rejected":
                        raise
                else:
                    raise Stage13ProofError("cross-organization identity was accepted")
            finally:
                foreign_client.close()

            active_sleep = _run(
                [
                    "docker",
                    "exec",
                    database_container,
                    "psql",
                    "-U",
                    "postgres",
                    "-tAc",
                    (
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE pid <> pg_backend_pid() AND state = 'active' "
                        "AND query LIKE '%pg_sleep%'"
                    ),
                ],
                cwd=repo_root,
            )
            if active_sleep.stdout.strip() != "0":
                raise Stage13ProofError("cancelled external query remained active")
        finally:
            _run(["docker", "rm", "-f", plugin_container], cwd=repo_root, check=False)
            _run(["docker", "rm", "-f", database_container], cwd=repo_root, check=False)
            _run(
                ["docker", "network", "rm", data_network],
                cwd=repo_root,
                check=False,
            )
            _run(
                ["docker", "network", "rm", gateway_network],
                cwd=repo_root,
                check=False,
            )
            if image_id:
                _run(["docker", "image", "rm", "-f", image_id], cwd=repo_root, check=False)
            cleanup = (
                _run(
                    ["docker", "container", "inspect", plugin_container],
                    cwd=repo_root,
                    check=False,
                ).returncode
                != 0
                and _run(
                    ["docker", "container", "inspect", database_container],
                    cwd=repo_root,
                    check=False,
                ).returncode
                != 0
                and _run(
                    ["docker", "network", "inspect", data_network],
                    cwd=repo_root,
                    check=False,
                ).returncode
                != 0
                and _run(
                    ["docker", "network", "inspect", gateway_network],
                    cwd=repo_root,
                    check=False,
                ).returncode
                != 0
                and (
                    not image_id
                    or _run(
                        ["docker", "image", "inspect", image_id],
                        cwd=repo_root,
                        check=False,
                    ).returncode
                    != 0
                )
            )
    return {
        "schema": "io.roehub.data-source-panel-proof/v1",
        "status": "passed",
        "signed_bundle": "passed",
        "image_digest_binding": "passed",
        "external_database": "passed",
        "two_organization_isolation": "passed",
        "read_only_role": "passed",
        "row_limit": "passed",
        "timeout_cancellation": "passed",
        "declarative_contributions": "passed",
        "cleanup": "passed" if cleanup else "failed",
        **host_chain,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    try:
        payload = run_proof(repo_root)
    except Exception as error:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema": "io.roehub.data-source-panel-proof/v1",
                    "status": "failed",
                    "error_type": type(error).__name__,
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
