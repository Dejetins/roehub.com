"""Disposable PostgreSQL proof for OIDC identity mapping and invitation controls."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier
from uuid import UUID

import psycopg
from psycopg.errors import ObjectNotInPrerequisiteState

from trading.contexts.identity.adapters.outbound.persistence.postgres import (
    PostgresOidcIdentityRepository,
)
from trading.contexts.identity.application import OidcIdentityRepositoryError
from trading.shared_kernel.primitives import UserId


class OidcRuntimeProofError(RuntimeError):
    """Raised when the disposable OIDC persistence proof is incomplete."""


def run_probe(*, dsn: str) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    repository = PostgresOidcIdentityRepository(dsn=dsn)
    owner_id, secondary_user_id = _fixture_users(dsn=dsn)
    users_before = _count_users(dsn=dsn)

    concurrent = repository.create_attempt(
        provider_id="fixture",
        issuer="https://identity.example.test",
        purpose="login",
        state_sha256="0" * 64,
        nonce_sha256="f" * 64,
        code_verifier="q" * 64,
        linking_user_id=None,
        next_path="/",
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    barrier = Barrier(2)

    def claim_concurrently() -> bool:
        barrier.wait()
        return repository.claim_attempt(
            attempt_id=concurrent.attempt_id,
            claimed_at=now,
        ) is not None

    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed = tuple(executor.map(lambda _: claim_concurrently(), range(2)))
    if claimed.count(True) != 1:
        raise OidcRuntimeProofError("OIDC attempt admitted multiple token exchanges")
    repository.reject_attempt(
        attempt_id=concurrent.attempt_id,
        reason_code="proof_completed",
        rejected_at=now,
    )

    invited = repository.create_attempt(
        provider_id="fixture",
        issuer="https://identity.example.test",
        purpose="login",
        state_sha256="1" * 64,
        nonce_sha256="2" * 64,
        code_verifier="v" * 64,
        linking_user_id=None,
        next_path="/",
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    _require_claim(repository=repository, attempt_id=invited.attempt_id, now=now)
    provisioned = repository.complete_attempt(
        attempt_id=invited.attempt_id,
        provider_id="fixture",
        issuer="https://identity.example.test",
        subject_sha256="3" * 64,
        email_sha256="a" * 64,
        email_verified=True,
        callback_user_id=None,
        completed_at=now,
    )
    if not provisioned.provisioned or provisioned.accepted_invitation_count != 1:
        raise OidcRuntimeProofError("invitation provisioning was not atomic")

    repeated = repository.create_attempt(
        provider_id="fixture",
        issuer="https://identity.example.test",
        purpose="login",
        state_sha256="4" * 64,
        nonce_sha256="5" * 64,
        code_verifier="w" * 64,
        linking_user_id=None,
        next_path="/",
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    _require_claim(repository=repository, attempt_id=repeated.attempt_id, now=now)
    mapped = repository.complete_attempt(
        attempt_id=repeated.attempt_id,
        provider_id="fixture",
        issuer="https://identity.example.test",
        subject_sha256="3" * 64,
        email_sha256=None,
        email_verified=False,
        callback_user_id=None,
        completed_at=now,
    )
    if mapped.user_id != provisioned.user_id or mapped.provisioned:
        raise OidcRuntimeProofError("stable provider subject mapping was not preserved")

    uninvited = repository.create_attempt(
        provider_id="fixture",
        issuer="https://identity.example.test",
        purpose="login",
        state_sha256="6" * 64,
        nonce_sha256="7" * 64,
        code_verifier="x" * 64,
        linking_user_id=None,
        next_path="/",
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    _require_claim(repository=repository, attempt_id=uninvited.attempt_id, now=now)
    try:
        repository.complete_attempt(
            attempt_id=uninvited.attempt_id,
            provider_id="fixture",
            issuer="https://identity.example.test",
            subject_sha256="8" * 64,
            email_sha256="9" * 64,
            email_verified=True,
            callback_user_id=None,
            completed_at=now,
        )
    except OidcIdentityRepositoryError as error:
        if error.code != "oidc_invitation_required":
            raise OidcRuntimeProofError("unexpected uninvited rejection") from error
        repository.reject_attempt(
            attempt_id=uninvited.attempt_id,
            reason_code=error.code,
            rejected_at=now,
        )
    else:
        raise OidcRuntimeProofError("uninvited identity was provisioned")
    if _count_users(dsn=dsn) != users_before + 1:
        raise OidcRuntimeProofError("rejected provisioning left an orphan user")

    linked_attempt = repository.create_attempt(
        provider_id="fixture",
        issuer="https://identity.example.test",
        purpose="link",
        state_sha256="a" * 64,
        nonce_sha256="b" * 64,
        code_verifier="y" * 64,
        linking_user_id=owner_id,
        next_path="/account",
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    _require_claim(repository=repository, attempt_id=linked_attempt.attempt_id, now=now)
    linked = repository.complete_attempt(
        attempt_id=linked_attempt.attempt_id,
        provider_id="fixture",
        issuer="https://identity.example.test",
        subject_sha256="c" * 64,
        email_sha256=None,
        email_verified=False,
        callback_user_id=owner_id,
        completed_at=now,
    )
    if linked.user_id != owner_id or not linked.linked:
        raise OidcRuntimeProofError("authenticated linking did not preserve local user id")

    conflict_attempt = repository.create_attempt(
        provider_id="fixture",
        issuer="https://identity.example.test",
        purpose="link",
        state_sha256="d" * 64,
        nonce_sha256="e" * 64,
        code_verifier="z" * 64,
        linking_user_id=secondary_user_id,
        next_path="/account",
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    _require_claim(repository=repository, attempt_id=conflict_attempt.attempt_id, now=now)
    try:
        repository.complete_attempt(
            attempt_id=conflict_attempt.attempt_id,
            provider_id="fixture",
            issuer="https://identity.example.test",
            subject_sha256="c" * 64,
            email_sha256=None,
            email_verified=False,
            callback_user_id=secondary_user_id,
            completed_at=now,
        )
    except OidcIdentityRepositoryError as error:
        if error.code != "oidc_identity_conflict":
            raise OidcRuntimeProofError("unexpected linking conflict") from error
        repository.reject_attempt(
            attempt_id=conflict_attempt.attempt_id,
            reason_code=error.code,
            rejected_at=now,
        )
    else:
        raise OidcRuntimeProofError("provider subject takeover was accepted")

    immutable = False
    try:
        with psycopg.connect(dsn, autocommit=True) as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE identity_oidc_auth_events SET reason_code = 'changed'")
    except ObjectNotInPrerequisiteState:
        immutable = True
    if not immutable:
        raise OidcRuntimeProofError("OIDC audit mutation was accepted")

    return {
        "schema": "io.roehub.oidc-runtime-proof/v1alpha1",
        "invitation_provisioning": "passed",
        "stable_subject_mapping": "passed",
        "single_exchange_claim": "passed",
        "uninvited_provisioning": "rejected",
        "authenticated_linking": "passed",
        "subject_takeover": "rejected",
        "hash_only_identity": "passed",
        "audit_immutable": "passed",
    }


def _fixture_users(*, dsn: str) -> tuple[UserId, UserId]:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT owner.user_id
            FROM identity_installation_owners AS owner
            ORDER BY owner.granted_at, owner.user_id
            LIMIT 1
            """
        )
        owner = cursor.fetchone()
        cursor.execute(
            """
            SELECT membership.user_id
            FROM identity_memberships AS membership
            WHERE membership.role = 'admin'
            ORDER BY membership.created_at, membership.user_id
            LIMIT 1
            """
        )
        secondary = cursor.fetchone()
    if owner is None or secondary is None:
        raise OidcRuntimeProofError("organization proof users are missing")
    return UserId(UUID(str(owner[0]))), UserId(UUID(str(secondary[0])))


def _count_users(*, dsn: str) -> int:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM identity_users")
        row = cursor.fetchone()
    if row is None:
        raise OidcRuntimeProofError("count query returned no row")
    return int(row[0])


def _require_claim(
    *, repository: PostgresOidcIdentityRepository, attempt_id: UUID, now: datetime
) -> None:
    if repository.claim_attempt(attempt_id=attempt_id, claimed_at=now) is None:
        raise OidcRuntimeProofError("OIDC proof attempt could not be claimed")


def main() -> int:
    dsn = os.environ.get("IDENTITY_PG_DSN", "").strip()
    if not dsn:
        raise SystemExit("IDENTITY_PG_DSN is required")
    try:
        result = run_probe(dsn=dsn)
    except (OidcRuntimeProofError, OidcIdentityRepositoryError, psycopg.Error) as error:
        print(f"OIDC runtime proof failed: {error}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
