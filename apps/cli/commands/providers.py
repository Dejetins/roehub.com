from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping
from uuid import UUID

from trading.contexts.notifications.adapters.outbound.persistence.postgres import (
    PostgresNotificationProviderRepository,
    PsycopgNotificationPostgresGateway,
)
from trading.contexts.notifications.application import (
    AddNotificationProviderCommand,
    InstallNotificationProviderPackageCommand,
    NotificationProviderAdministrationService,
)
from trading.shared_kernel.primitives import OrganizationId

_POSTGRES_DSN_ENV = "NOTIFICATIONS_PG_DSN"


class ProvidersCli:
    def __init__(self, *, environ: Mapping[str, str] | None = None) -> None:
        self._environ = os.environ if environ is None else environ

    def run(self, argv: list[str]) -> int:
        parser = argparse.ArgumentParser(prog="roehubctl providers")
        subparsers = parser.add_subparsers(dest="action", required=True)
        install = subparsers.add_parser("install")
        install.add_argument("--descriptor-file", type=Path, required=True)
        add = subparsers.add_parser("add")
        add.add_argument("--instance-id", type=UUID, required=True)
        add.add_argument("--package-id", type=UUID, required=True)
        add.add_argument("--provider-key", required=True)
        add.add_argument("--scope", choices=("installation", "organization"), required=True)
        add.add_argument("--organization-id", type=UUID)
        add.add_argument("--display-name", required=True)
        add.add_argument("--config-json", default="{}")
        add.add_argument("--secret-ref")
        args = parser.parse_args(argv)
        dsn = self._environ.get(_POSTGRES_DSN_ENV, "").strip()
        if not dsn:
            parser.error(f"{_POSTGRES_DSN_ENV} is required")
        service = NotificationProviderAdministrationService(
            repository=PostgresNotificationProviderRepository(
                gateway=PsycopgNotificationPostgresGateway(dsn=dsn)
            )
        )
        if args.action == "install":
            descriptor = _load_descriptor(path=args.descriptor_file, parser=parser)
            package = service.install_provider_package(
                command=InstallNotificationProviderPackageCommand(
                    provider_key=_required_text(descriptor, "provider_key", parser),
                    display_name=_required_text(descriptor, "display_name", parser),
                    package_version=_required_text(
                        descriptor, "package_version", parser
                    ),
                    config_schema=_required_mapping(
                        descriptor, "config_schema", parser
                    ),
                    channels=_required_strings(descriptor, "channels", parser),
                    templates=_required_strings(descriptor, "templates", parser),
                    error_codes=_required_strings(descriptor, "error_codes", parser),
                ),
                now=datetime.now(UTC),
            )
            print(
                json.dumps(
                    {
                        "schema": "io.roehub.notification-provider-package/v1",
                        "package_id": str(package.package_id),
                        "provider_key": package.descriptor.provider_key,
                        "contract_version": package.descriptor.contract_version,
                        "package_version": package.descriptor.package_version,
                        "built_in": package.built_in,
                    },
                    sort_keys=True,
                )
            )
            return 0
        try:
            config_json = json.loads(args.config_json)
        except json.JSONDecodeError as error:
            parser.error(f"--config-json is invalid: {error}")
        if not isinstance(config_json, dict):
            parser.error("--config-json must be an object")
        organization_id = (
            OrganizationId(args.organization_id)
            if args.organization_id is not None
            else None
        )
        instance = service.add_provider(
            command=AddNotificationProviderCommand(
                instance_id=args.instance_id,
                package_id=args.package_id,
                provider_key=args.provider_key,
                scope=args.scope,
                organization_id=organization_id,
                display_name=args.display_name,
                config_json=config_json,
                secret_ref=args.secret_ref,
            ),
            now=datetime.now(UTC),
        )
        print(
            json.dumps(
                {
                    "schema": "io.roehub.notification-provider-instance/v1",
                    "instance_id": str(instance.instance_id),
                    "provider_key": instance.provider_key,
                    "scope": instance.scope,
                    "organization_id": (
                        str(instance.organization_id)
                        if instance.organization_id is not None
                        else None
                    ),
                    "status": instance.status,
                },
                sort_keys=True,
            )
        )
        return 0


def _load_descriptor(
    *, path: Path, parser: argparse.ArgumentParser
) -> dict[str, object]:
    try:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"--descriptor-file is invalid: {error}")
    if not isinstance(payload, dict):
        parser.error("--descriptor-file must contain an object")
    return payload


def _required_text(
    payload: Mapping[str, object], key: str, parser: argparse.ArgumentParser
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        parser.error(f"descriptor {key} must be non-empty text")
    return value.strip()


def _required_mapping(
    payload: Mapping[str, object], key: str, parser: argparse.ArgumentParser
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        parser.error(f"descriptor {key} must be an object")
    return value


def _required_strings(
    payload: Mapping[str, object], key: str, parser: argparse.ArgumentParser
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        parser.error(f"descriptor {key} must be a non-empty string array")
    return tuple(item.strip() for item in value if isinstance(item, str))
