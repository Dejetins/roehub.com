"""Disposable PostgreSQL proof for local-auth persistence and redaction invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import psycopg
from argon2 import PasswordHasher
from psycopg.errors import ObjectNotInPrerequisiteState, UniqueViolation

from trading.contexts.identity.adapters.outbound.persistence.postgres import (
    PostgresLocalAuthRepository,
)
from trading.contexts.identity.application.ports import LocalPasskey
from trading.shared_kernel.primitives import UserId


class LocalAuthRuntimeProofError(RuntimeError):
    """Raised when local-auth persistence evidence is incomplete."""


def run_probe(*, dsn: str, bootstrap_file: Path) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    bootstrap_value = bootstrap_file.read_text(encoding="utf-8").strip()
    if not bootstrap_value:
        raise LocalAuthRuntimeProofError("bootstrap file is empty")
    bootstrap_digest = hashlib.sha256(bootstrap_value.encode()).hexdigest()
    repository = PostgresLocalAuthRepository(dsn=dsn)
    if repository.find_bootstrap_ticket(token_sha256=bootstrap_digest, now=now) is None:
        raise LocalAuthRuntimeProofError("bootstrap ticket was not persisted by CLI")

    owner_id = UserId(uuid4())
    hasher = PasswordHasher()
    account_digest = hasher.hash(secrets.token_urlsafe(24))
    recovery_digest = hasher.hash(secrets.token_urlsafe(24))
    credential_id = secrets.token_urlsafe(24)
    with psycopg.connect(dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT token_sha256 = %s, token_sha256 = %s
                FROM identity_local_bootstrap_tickets
                WHERE consumed_at IS NULL
                """,
                (bootstrap_digest, bootstrap_value),
            )
            digest_match, raw_match = cursor.fetchone() or (False, True)
            if not digest_match or raw_match:
                raise LocalAuthRuntimeProofError("bootstrap storage is not hash-only")
            cursor.execute(
                """
                INSERT INTO identity_users (
                    user_id, telegram_user_id, paid_level, created_at,
                    last_login_at, is_deleted, keycloak_subject
                ) VALUES (%s, NULL, 'free', %s, %s, FALSE, NULL)
                """,
                (str(owner_id), now, now),
            )
            cursor.execute(
                """
                INSERT INTO identity_local_accounts (
                    user_id, username, display_name, password_hash, created_at, updated_at
                ) VALUES (%s, 'stage06-owner', 'Stage 06 Owner', %s, %s, %s)
                """,
                (str(owner_id), account_digest, now, now),
            )
            cursor.execute(
                """
                INSERT INTO identity_local_recovery_codes (
                    recovery_code_id, user_id, code_hash, created_at
                ) VALUES (%s, %s, %s, %s)
                """,
                (str(uuid4()), str(owner_id), recovery_digest, now),
            )

        duplicate_ticket_rejected = False
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO identity_local_bootstrap_tickets (
                        ticket_id, token_sha256, created_at, expires_at
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (str(uuid4()), "f" * 64, now, now + timedelta(minutes=5)),
                )
        except UniqueViolation:
            duplicate_ticket_rejected = True

    if not duplicate_ticket_rejected:
        raise LocalAuthRuntimeProofError("multiple active bootstrap tickets were accepted")

    challenge_bytes = secrets.token_bytes(32)
    challenge = repository.create_challenge(
        purpose="register",
        challenge_sha256=hashlib.sha256(challenge_bytes).hexdigest(),
        user_id=owner_id,
        context={},
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    passkey = LocalPasskey(
        credential_id=credential_id,
        user_id=owner_id,
        public_key=b"stage06-public-key-proof-material",
        sign_count=0,
        transports=("internal",),
        created_at=now,
    )
    repository.add_passkey_and_consume_challenge(
        challenge_id=challenge.challenge_id,
        passkey=passkey,
        completed_at=now,
    )
    auth_challenge = repository.create_challenge(
        purpose="recent_auth",
        challenge_sha256=hashlib.sha256(secrets.token_bytes(32)).hexdigest(),
        user_id=owner_id,
        context={},
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    repository.finish_passkey_authentication(
        challenge_id=auth_challenge.challenge_id,
        credential_id=credential_id,
        new_sign_count=1,
        completed_at=now,
    )
    stored_passkey = repository.find_passkey(credential_id=credential_id)
    if stored_passkey is None or stored_passkey.sign_count != 1:
        raise LocalAuthRuntimeProofError("passkey counter was not persisted")

    recovery = repository.list_recovery_code_hashes(user_id=owner_id)
    if len(recovery) != 1 or not repository.consume_recovery_code(
        recovery_code_id=recovery[0].recovery_code_id,
        user_id=owner_id,
        consumed_at=now,
    ):
        raise LocalAuthRuntimeProofError("one-time recovery consumption failed")
    if repository.consume_recovery_code(
        recovery_code_id=recovery[0].recovery_code_id,
        user_id=owner_id,
        consumed_at=now,
    ):
        raise LocalAuthRuntimeProofError("recovery code replay was accepted")

    subject_digest = hashlib.sha256(b"stage06-subject").hexdigest()
    for _attempt in range(5):
        repository.record_auth_failure(subject_sha256=subject_digest, now=now)
    if not repository.is_rate_limited(subject_sha256=subject_digest, now=now):
        raise LocalAuthRuntimeProofError("rate-limit lockout was not persisted")
    repository.record_auth_event(
        user_id=owner_id,
        subject_sha256=subject_digest,
        action="local_auth.login",
        outcome="succeeded",
        reason_code="completed",
        created_at=now,
    )

    immutable_event = False
    try:
        with psycopg.connect(dsn, autocommit=True) as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE identity_local_auth_events SET reason_code = 'changed'")
    except ObjectNotInPrerequisiteState:
        immutable_event = True
    if not immutable_event:
        raise LocalAuthRuntimeProofError("local auth audit mutation was accepted")

    account = repository.find_account_by_user_id(user_id=owner_id)
    if account is None or account.username != "stage06-owner":
        raise LocalAuthRuntimeProofError("local account lookup failed")
    return {
        "schema": "io.roehub.local-auth-runtime-proof/v1alpha1",
        "bootstrap_hash_only": "passed",
        "single_active_bootstrap": "passed",
        "passkey_counter": "passed",
        "recovery_replay": "rejected",
        "rate_limit": "passed",
        "audit_immutable": "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-file", type=Path, required=True)
    args = parser.parse_args()
    dsn = os.environ.get("IDENTITY_PG_DSN", "").strip()
    if not dsn:
        raise SystemExit("IDENTITY_PG_DSN is required")
    try:
        result = run_probe(dsn=dsn, bootstrap_file=args.bootstrap_file)
    except (LocalAuthRuntimeProofError, OSError, psycopg.Error) as error:
        print(f"local auth runtime proof failed: {error}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
