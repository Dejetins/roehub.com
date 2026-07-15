from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from trading.contexts.identity.adapters.outbound import PostgresLocalAuthRepository

_TICKET_TTL = timedelta(minutes=15)
_POSTGRES_DSN_ENV = "IDENTITY_PG_DSN"


class LocalAuthBootstrapCli:
    """Issue one local, single-use bootstrap ticket without exposing it in logs."""

    def __init__(self, *, environ: Mapping[str, str] | None = None) -> None:
        self._environ = os.environ if environ is None else environ

    def run(self, argv: list[str]) -> int:
        parser = argparse.ArgumentParser(prog="roehubctl local-auth-bootstrap")
        parser.add_argument("--output-file", type=Path, required=True)
        args = parser.parse_args(argv)
        dsn = self._environ.get(_POSTGRES_DSN_ENV, "").strip()
        if not dsn:
            parser.error(f"{_POSTGRES_DSN_ENV} is required")
        output_file = args.output_file.expanduser().resolve()
        if output_file.exists():
            parser.error("output file already exists")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        one_time_value = secrets.token_urlsafe(32)
        repository = PostgresLocalAuthRepository(dsn=dsn)
        repository.issue_bootstrap_ticket(
            token_sha256=hashlib.sha256(one_time_value.encode()).hexdigest(),
            created_at=now,
            expires_at=now + _TICKET_TTL,
        )
        descriptor = os.open(
            output_file,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(one_time_value)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "schema": "io.roehub.local-auth-bootstrap/v1alpha1",
                    "status": "issued",
                    "output_file": str(output_file),
                    "expires_at": (now + _TICKET_TTL).isoformat(),
                },
                sort_keys=True,
            )
        )
        return 0


def main() -> int:
    return LocalAuthBootstrapCli().run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
