from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import APIRouter

from apps.api.routes.extensions import build_extensions_router
from trading.contexts.extensions.adapters import (
    HttpPluginGatewayDataSourceInvoker,
    IdentityPluginAuthorization,
    InMemoryPluginRepository,
    PostgresPluginRepository,
)
from trading.contexts.extensions.application import (
    DataSourceQueryService,
    PluginBundleValidator,
    PluginLifecycleService,
    load_publisher_key_file,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.application.ports import OrganizationRepository

_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class ExtensionsApiModule:
    router: APIRouter
    service: PluginLifecycleService
    data_source_service: DataSourceQueryService


def build_extensions_api_module(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
    organization_repository: OrganizationRepository,
) -> ExtensionsApiModule:
    env_name = environ.get("ROEHUB_ENV", "dev")
    publisher_key_path = environ.get("ROEHUB_PLUGIN_PUBLISHER_KEYS_FILE", "")
    unsigned_development = (
        environ.get("ROEHUB_PLUGIN_UNSIGNED_DEVELOPMENT", "false") == "true"
    )
    if env_name == "prod" and unsigned_development:
        raise ValueError("unsigned plugin development mode is forbidden in prod")
    if env_name == "prod" and not publisher_key_path:
        raise ValueError("ROEHUB_PLUGIN_PUBLISHER_KEYS_FILE is required in prod")
    publisher_keys = (
        load_publisher_key_file(Path(publisher_key_path))
        if publisher_key_path
        else {}
    )
    validator = PluginBundleValidator(
        schema_path=_REPO_ROOT
        / "schemas/plugins/roehub-plugin-manifest-v1alpha1.schema.json",
        trusted_publisher_keys=publisher_keys,
        roehub_version="0.1.0",
        supported_architectures=frozenset({"linux/amd64", "linux/arm64"}),
        allow_unsigned_development=unsigned_development,
        trading_mode=environ.get("ROEHUB_TRADING_MODE", "paper"),
    )
    postgres_dsn = environ.get("EXTENSIONS_PG_DSN", "").strip()
    if env_name == "prod" and not postgres_dsn:
        raise ValueError("EXTENSIONS_PG_DSN is required in prod")
    repository = (
        PostgresPluginRepository(dsn=postgres_dsn)
        if postgres_dsn
        else InMemoryPluginRepository()
    )
    authorization = IdentityPluginAuthorization(repository=organization_repository)
    service = PluginLifecycleService(
        repository=repository,
        authorization=authorization,
        trusted_publisher_fingerprints={
            key_id: hashlib.sha256(
                public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
            ).hexdigest()
            for key_id, public_key in publisher_keys.items()
        },
        allow_unsigned_development=unsigned_development,
        trading_mode=environ.get("ROEHUB_TRADING_MODE", "paper"),
    )
    spool_root = Path(
        environ.get(
            "ROEHUB_PLUGIN_BUNDLE_SPOOL_ROOT", "/var/lib/roehub/plugin-bundles"
        )
    )
    data_source_service = DataSourceQueryService(
        repository=repository,
        authorization=authorization,
        invoker=HttpPluginGatewayDataSourceInvoker(
            gateway_url=environ.get(
                "ROEHUB_PLUGIN_GATEWAY_URL", "http://plugin-gateway:8080"
            )
        ),
    )
    return ExtensionsApiModule(
        router=build_extensions_router(
            service=service,
            validator=validator,
            bundle_spool_root=spool_root,
            current_user_dependency=current_user_dependency,
            data_source_service=data_source_service,
        ),
        service=service,
        data_source_service=data_source_service,
    )
