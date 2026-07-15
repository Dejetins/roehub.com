"""Provision one disposable invited user through the OIDC application boundary."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Sequence

from trading.contexts.identity.adapters.outbound import (
    PostgresIdentitySessionRepository,
    PostgresOidcIdentityRepository,
    PsycopgIdentityPostgresGateway,
    SystemIdentityClock,
)
from trading.contexts.identity.application.ports import VerifiedExternalIdentity
from trading.contexts.identity.application.use_cases import OidcAuthenticationService

_ISSUER = "https://stage23.invalid.example"
_PROVIDER_ID = "stage23-disposable-oidc"


@dataclass(frozen=True, slots=True)
class _DisposableProvider:
    email: str
    subject: str

    @property
    def provider_id(self) -> str:
        return _PROVIDER_ID

    @property
    def issuer(self) -> str:
        return _ISSUER

    @property
    def display_name(self) -> str:
        return "Stage 23 disposable OIDC"

    def authorization_url(
        self,
        *,
        state: str,
        nonce: str,
        code_challenge: str,
    ) -> str:
        del state, nonce, code_challenge
        return f"{_ISSUER}/authorize"

    def exchange_code(
        self,
        *,
        code: str,
        code_verifier: str,
        expected_nonce_sha256: str,
    ) -> VerifiedExternalIdentity:
        del code, code_verifier, expected_nonce_sha256
        return VerifiedExternalIdentity(
            issuer=_ISSUER,
            subject=self.subject,
            email=self.email,
            email_verified=True,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--fixture-id", choices=("a", "b"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dsn = os.environ.get("IDENTITY_PG_DSN", "").strip()
    if not dsn:
        raise RuntimeError("IDENTITY_PG_DSN is required")
    gateway = PsycopgIdentityPostgresGateway(dsn=dsn)
    service = OidcAuthenticationService(
        provider=_DisposableProvider(
            email=args.email,
            subject=f"stage23-disposable-subject-{args.fixture_id}",
        ),
        repository=PostgresOidcIdentityRepository(dsn=dsn),
        session_repository=PostgresIdentitySessionRepository(gateway=gateway),
        clock=SystemIdentityClock(),
        session_idle_ttl_seconds=1800,
        session_absolute_ttl_seconds=43200,
    )
    start = service.begin_login(next_path="/admin")
    result = service.complete(
        attempt_id=start.attempt_id,
        state=start.state,
        code=f"stage23-code-{args.fixture_id}",
        callback_user_id=None,
    )
    if not result.provisioned or result.accepted_invitation_count != 1:
        raise RuntimeError("disposable OIDC invitation provisioning did not complete")
    print(
        json.dumps(
            {
                "accepted_invitation_count": result.accepted_invitation_count,
                "fixture_id": args.fixture_id,
                "provisioned": result.provisioned,
                "schema": "io.roehub.stage23-disposable-oidc/v1alpha1",
                "user_id": str(result.session.user_id) if result.session is not None else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
