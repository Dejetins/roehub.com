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
    PostgresNotificationTelegramBindingStore,
    PsycopgNotificationPostgresGateway,
)
from trading.contexts.notifications.application import (
    ConnectTelegramProviderCommand,
    NotificationProviderAdministrationService,
    NotificationTelegramBindingService,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

_POSTGRES_DSN_ENV = "NOTIFICATIONS_PG_DSN"


class TelegramCli:
    def __init__(self, *, environ: Mapping[str, str] | None = None) -> None:
        self._environ = os.environ if environ is None else environ

    def run(self, argv: list[str]) -> int:
        parser = argparse.ArgumentParser(prog="roehubctl telegram")
        subparsers = parser.add_subparsers(dest="action", required=True)
        connect = subparsers.add_parser("connect")
        connect.add_argument("--provider-instance-id", type=UUID, required=True)
        connect.add_argument("--organization-id", type=UUID, required=True)
        connect.add_argument("--owner-user-id", type=UUID, required=True)
        connect.add_argument("--output-file", type=Path, required=True)
        args = parser.parse_args(argv)
        dsn = self._environ.get(_POSTGRES_DSN_ENV, "").strip()
        if not dsn:
            parser.error(f"{_POSTGRES_DSN_ENV} is required")
        output_file = args.output_file.expanduser().resolve()
        if output_file.exists():
            parser.error("output file already exists")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        gateway = PsycopgNotificationPostgresGateway(dsn=dsn)
        organization_id = OrganizationId(args.organization_id)
        administration = NotificationProviderAdministrationService(
            repository=PostgresNotificationProviderRepository(gateway=gateway)
        )
        administration.connect_telegram(
            command=ConnectTelegramProviderCommand(
                provider_instance_id=args.provider_instance_id,
                organization_id=organization_id,
            )
        )
        binding_service = NotificationTelegramBindingService(
            store=PostgresNotificationTelegramBindingStore(gateway=gateway),
            organization_id=organization_id,
            provider_instance_id=args.provider_instance_id,
        )
        binding = binding_service.create_binding_code(
            owner_user_id=UserId(args.owner_user_id),
            now=datetime.now(UTC),
        )
        descriptor = os.open(output_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(binding.code)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "schema": "io.roehub.telegram-connect/v1",
                    "provider_instance_id": str(args.provider_instance_id),
                    "organization_id": str(organization_id),
                    "output_file": str(output_file),
                    "expires_at": binding.expires_at.isoformat(),
                    "status": "pending_confirmation",
                },
                sort_keys=True,
            )
        )
        return 0
