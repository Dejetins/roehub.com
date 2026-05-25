from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, cast
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from apps.api.exchange_control_client import (
    ExchangeConnectionCommandResult,
    ExchangeControlClientError,
    HttpExchangeControlClient,
)
from trading.contexts.identity.adapters.outbound import (
    PostgresAccountSettingsRepository,
    PsycopgIdentityPostgresGateway,
    SystemIdentityClock,
)
from trading.contexts.identity.application.use_cases.account_settings import (
    AccountSettingsUseCase,
)
from trading.shared_kernel.primitives import UserId

DEFAULT_LABEL_PREFIXES = ("stage08_", "e2e_", "smoke_")
DEFAULT_CLEANUP_SOURCE = "stage09d"
DEFAULT_LIMIT = 100
_DSN_ENV = "IDENTITY_PG_DSN"
_BASE_URL_ENV = "ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL"
_TOKEN_ENV = "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN"
_TIMEOUT_ENV = "ROEHUB_EXCHANGE_CONTROL_INTERNAL_TIMEOUT_SECONDS"


@dataclass(frozen=True, slots=True)
class ExchangeConnectionCleanupCandidate:
    connection_id: UUID
    owner_user_id: UserId
    exchange_name: str
    market_type: str
    environment: str
    label: str
    status: str
    created_at: datetime
    disabled_at: datetime


@dataclass(frozen=True, slots=True)
class ExchangeConnectionCleanupResult:
    candidate: ExchangeConnectionCleanupCandidate
    result: str
    status: str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExchangeConnectionArchivedAuditRepairCandidate:
    connection_id: UUID
    owner_user_id: UserId
    exchange_name: str
    market_type: str
    environment: str
    label: str
    status: str
    status_reason: str | None
    created_at: datetime
    disabled_at: datetime
    archived_at: datetime


class ExchangeConnectionArchiveClient(Protocol):
    def archive_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        cleanup_source: str | None = None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult: ...


class ExchangeConnectionArchiveAuditRecorder(Protocol):
    def record_exchange_connection_archive(
        self,
        *,
        owner_user_id: UserId,
        connection_id: str,
        exchange_name: str,
        market_type: str,
        environment: str,
        previous_status: str,
        new_status: str,
        reason: str,
    ) -> None: ...


def select_cleanup_candidates(
    *,
    rows: Iterable[Mapping[str, object]],
    label_prefixes: Sequence[str] = DEFAULT_LABEL_PREFIXES,
) -> tuple[ExchangeConnectionCleanupCandidate, ...]:
    candidates = []
    normalized_prefixes = _normalize_label_prefixes(label_prefixes)
    for row in rows:
        label = row.get("label")
        status = str(row.get("status") or "")
        disabled_at = row.get("disabled_at")
        archived_at = row.get("archived_at")
        if not isinstance(label, str) or not label:
            continue
        if not _matches_label_prefix(label=label, label_prefixes=normalized_prefixes):
            continue
        if status != "disabled" or disabled_at is None or archived_at is not None:
            continue
        candidates.append(
            ExchangeConnectionCleanupCandidate(
                connection_id=UUID(str(row["connection_id"])),
                owner_user_id=UserId.from_string(str(row["owner_user_id"])),
                exchange_name=str(row["exchange_name"]),
                market_type=str(row["market_type"]),
                environment=str(row["environment"]),
                label=label,
                status=status,
                created_at=_coerce_datetime(row["created_at"]),
                disabled_at=_coerce_datetime(disabled_at),
            )
        )
    candidates.sort(key=lambda item: (item.created_at, str(item.connection_id)))
    return tuple(candidates)


def load_cleanup_candidates(
    *,
    dsn: str,
    label_prefixes: Sequence[str] = DEFAULT_LABEL_PREFIXES,
    owner_user_id: str | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> tuple[ExchangeConnectionCleanupCandidate, ...]:
    normalized_prefixes = _normalize_label_prefixes(label_prefixes)
    parameters: dict[str, object] = {
        "limit": limit,
    }
    prefix_predicates = []
    for index, prefix in enumerate(normalized_prefixes):
        key = f"prefix_{index}"
        parameters[key] = prefix
        prefix_predicates.append(f"left(label, char_length(%({key})s)) = %({key})s")
    predicates = [
        "status = 'disabled'",
        "disabled_at IS NOT NULL",
        "archived_at IS NULL",
        "label IS NOT NULL",
        f"({' OR '.join(prefix_predicates)})",
    ]
    if owner_user_id:
        parameters["owner_user_id"] = str(UserId.from_string(owner_user_id))
        predicates.append("owner_user_id = %(owner_user_id)s")
    if created_after is not None:
        parameters["created_after"] = created_after
        predicates.append("created_at >= %(created_after)s")
    if created_before is not None:
        parameters["created_before"] = created_before
        predicates.append("created_at < %(created_before)s")

    query = f"""
        SELECT
            connection_id,
            owner_user_id,
            exchange_name,
            market_type,
            environment,
            label,
            status,
            created_at,
            disabled_at,
            archived_at
        FROM exchange_connections
        WHERE {' AND '.join(predicates)}
        ORDER BY created_at ASC, connection_id ASC
        LIMIT %(limit)s
    """
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(cast(Any, query), parameters)
            rows = cursor.fetchall()
    return select_cleanup_candidates(
        rows=cast(Iterable[Mapping[str, object]], rows),
        label_prefixes=normalized_prefixes,
    )


def load_archived_audit_repair_candidates(
    *,
    dsn: str,
    label_prefixes: Sequence[str] = DEFAULT_LABEL_PREFIXES,
    owner_user_id: str | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> tuple[ExchangeConnectionArchivedAuditRepairCandidate, ...]:
    normalized_prefixes = _normalize_label_prefixes(label_prefixes)
    parameters: dict[str, object] = {
        "limit": limit,
    }
    prefix_predicates = []
    for index, prefix in enumerate(normalized_prefixes):
        key = f"prefix_{index}"
        parameters[key] = prefix
        prefix_predicates.append(f"left(c.label, char_length(%({key})s)) = %({key})s")
    predicates = [
        "c.status = 'archived'",
        "c.disabled_at IS NOT NULL",
        "c.archived_at IS NOT NULL",
        "c.label IS NOT NULL",
        f"({' OR '.join(prefix_predicates)})",
        """
        NOT EXISTS (
            SELECT 1
            FROM identity_audit_events AS audit
            WHERE audit.owner_user_id = c.owner_user_id
              AND audit.event_type = 'exchange_connection_archived'
              AND audit.metadata_json->>'connection_id' = c.connection_id::text
        )
        """,
    ]
    if owner_user_id:
        parameters["owner_user_id"] = str(UserId.from_string(owner_user_id))
        predicates.append("c.owner_user_id = %(owner_user_id)s")
    if created_after is not None:
        parameters["created_after"] = created_after
        predicates.append("c.created_at >= %(created_after)s")
    if created_before is not None:
        parameters["created_before"] = created_before
        predicates.append("c.created_at < %(created_before)s")

    query = f"""
        SELECT
            c.connection_id,
            c.owner_user_id,
            c.exchange_name,
            c.market_type,
            c.environment,
            c.label,
            c.status,
            c.status_reason,
            c.created_at,
            c.disabled_at,
            c.archived_at
        FROM exchange_connections AS c
        WHERE {' AND '.join(predicates)}
        ORDER BY c.archived_at ASC, c.connection_id ASC
        LIMIT %(limit)s
    """
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(cast(Any, query), parameters)
            rows = cursor.fetchall()
    return tuple(
        ExchangeConnectionArchivedAuditRepairCandidate(
            connection_id=UUID(str(row["connection_id"])),
            owner_user_id=UserId.from_string(str(row["owner_user_id"])),
            exchange_name=str(row["exchange_name"]),
            market_type=str(row["market_type"]),
            environment=str(row["environment"]),
            label=str(row["label"]),
            status=str(row["status"]),
            status_reason=(
                None if row.get("status_reason") is None else str(row["status_reason"])
            ),
            created_at=_coerce_datetime(row["created_at"]),
            disabled_at=_coerce_datetime(row["disabled_at"]),
            archived_at=_coerce_datetime(row["archived_at"]),
        )
        for row in cast(Iterable[Mapping[str, object]], rows)
    )


def execute_cleanup(
    *,
    candidates: Sequence[ExchangeConnectionCleanupCandidate],
    archive_client: ExchangeConnectionArchiveClient,
    audit_recorder: ExchangeConnectionArchiveAuditRecorder,
    source: str,
) -> tuple[ExchangeConnectionCleanupResult, ...]:
    normalized_source = normalize_cleanup_source(source)
    results: list[ExchangeConnectionCleanupResult] = []
    for candidate in candidates:
        try:
            archived = archive_client.archive_connection(
                owner_user_id=str(candidate.owner_user_id),
                connection_id=str(candidate.connection_id),
                cleanup_source=normalized_source,
                request_id=f"stage09d-cleanup-{redacted_uuid(candidate.connection_id)}",
            )
        except ExchangeControlClientError as error:
            results.append(
                ExchangeConnectionCleanupResult(
                    candidate=candidate,
                    result="rejected",
                    status=None,
                    error=str(error),
                )
            )
            continue
        audit_recorder.record_exchange_connection_archive(
            owner_user_id=candidate.owner_user_id,
            connection_id=str(candidate.connection_id),
            exchange_name=candidate.exchange_name,
            market_type=candidate.market_type,
            environment=candidate.environment,
            previous_status="disabled",
            new_status=archived.status,
            reason=archived.status_reason or "stage09d_cleanup",
        )
        results.append(
            ExchangeConnectionCleanupResult(
                candidate=candidate,
                result="archived",
                status=archived.status,
            )
        )
    return tuple(results)


def repair_archived_audit_events(
    *,
    candidates: Sequence[ExchangeConnectionArchivedAuditRepairCandidate],
    audit_recorder: ExchangeConnectionArchiveAuditRecorder,
) -> int:
    for candidate in candidates:
        audit_recorder.record_exchange_connection_archive(
            owner_user_id=candidate.owner_user_id,
            connection_id=str(candidate.connection_id),
            exchange_name=candidate.exchange_name,
            market_type=candidate.market_type,
            environment=candidate.environment,
            previous_status="disabled",
            new_status="archived",
            reason=candidate.status_reason or "stage09d_cleanup_audit_repair",
        )
    return len(candidates)


def redacted_uuid(value: UUID) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def normalize_cleanup_source(value: str) -> str:
    stripped = value.strip().lower()
    if not stripped:
        return DEFAULT_CLEANUP_SOURCE
    normalized = "".join(
        character
        if character.isascii()
        and (character.isalnum() or character in {"_", "-"})
        else "_"
        for character in stripped
    )
    return normalized[:40] or DEFAULT_CLEANUP_SOURCE


def summarize_candidates(
    *,
    candidates: Sequence[ExchangeConnectionCleanupCandidate],
    source: str,
    dry_run: bool,
) -> dict[str, object]:
    return {
        "mode": "dry-run" if dry_run else "execute",
        "source": normalize_cleanup_source(source),
        "count": len(candidates),
        "items": [
            {
                "connection_ref": redacted_uuid(candidate.connection_id),
                "owner_ref": redacted_uuid(candidate.owner_user_id.value),
                "label_prefix": _matched_prefix(candidate.label, DEFAULT_LABEL_PREFIXES),
                "exchange_name": candidate.exchange_name,
                "market_type": candidate.market_type,
                "environment": candidate.environment,
                "status": candidate.status,
                "created_at": candidate.created_at.isoformat(),
                "disabled_at": candidate.disabled_at.isoformat(),
            }
            for candidate in candidates
        ],
    }


def summarize_results(
    *,
    results: Sequence[ExchangeConnectionCleanupResult],
    source: str,
) -> dict[str, object]:
    return {
        "mode": "execute",
        "source": normalize_cleanup_source(source),
        "count": len(results),
        "items": [
            {
                "connection_ref": redacted_uuid(result.candidate.connection_id),
                "owner_ref": redacted_uuid(result.candidate.owner_user_id.value),
                "label_prefix": _matched_prefix(
                    result.candidate.label,
                    DEFAULT_LABEL_PREFIXES,
                ),
                "exchange_name": result.candidate.exchange_name,
                "market_type": result.candidate.market_type,
                "environment": result.candidate.environment,
                "result": result.result,
                "status": result.status,
                "error": result.error,
            }
            for result in results
        ],
    }


def summarize_audit_repairs(
    *,
    candidates: Sequence[ExchangeConnectionArchivedAuditRepairCandidate],
) -> dict[str, object]:
    return {
        "mode": "audit-repair",
        "count": len(candidates),
        "items": [
            {
                "connection_ref": redacted_uuid(candidate.connection_id),
                "owner_ref": redacted_uuid(candidate.owner_user_id.value),
                "label_prefix": _matched_prefix(candidate.label, DEFAULT_LABEL_PREFIXES),
                "exchange_name": candidate.exchange_name,
                "market_type": candidate.market_type,
                "environment": candidate.environment,
                "status": candidate.status,
                "archived_at": candidate.archived_at.isoformat(),
            }
            for candidate in candidates
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or execute Stage 09D exchange connection cleanup through "
            "the supported archive lifecycle path."
        )
    )
    parser.add_argument("--dsn", default=os.environ.get(_DSN_ENV, ""))
    parser.add_argument("--exchange-control-url", default=os.environ.get(_BASE_URL_ENV, ""))
    parser.add_argument("--exchange-control-token", default=os.environ.get(_TOKEN_ENV, ""))
    parser.add_argument(
        "--exchange-control-timeout",
        type=float,
        default=float(os.environ.get(_TIMEOUT_ENV, "2.0")),
    )
    parser.add_argument("--source", default=DEFAULT_CLEANUP_SOURCE)
    parser.add_argument(
        "--label-prefix",
        dest="label_prefixes",
        action="append",
        default=None,
        help="Eligible label prefix. Defaults to stage08_, e2e_, smoke_.",
    )
    parser.add_argument("--owner-user-id", default=None)
    parser.add_argument("--created-after", type=_parse_datetime, default=None)
    parser.add_argument("--created-before", type=_parse_datetime, default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true", default=False)
    parser.add_argument(
        "--repair-archived-audit",
        action="store_true",
        default=False,
        help=(
            "When executing, record missing archive audit events for already "
            "archived eligible cleanup rows. This does not change lifecycle state."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dsn.strip():
        raise SystemExit(f"Postgres DSN is required via --dsn or ${_DSN_ENV}.")
    label_prefixes = tuple(args.label_prefixes or DEFAULT_LABEL_PREFIXES)
    dry_run = not bool(args.execute)
    candidates = load_cleanup_candidates(
        dsn=args.dsn,
        label_prefixes=label_prefixes,
        owner_user_id=args.owner_user_id,
        created_after=args.created_after,
        created_before=args.created_before,
        limit=args.limit,
    )
    if dry_run:
        print(
            json.dumps(
                summarize_candidates(
                    candidates=candidates,
                    source=args.source,
                    dry_run=True,
                )
            )
        )
        return 0
    if not args.exchange_control_url.strip() or not args.exchange_control_token.strip():
        raise SystemExit(
            "Execution requires --exchange-control-url and --exchange-control-token."
        )
    archive_client = HttpExchangeControlClient(
        base_url=args.exchange_control_url,
        internal_api_token=args.exchange_control_token,
        timeout_seconds=args.exchange_control_timeout,
    )
    audit_recorder = AccountSettingsUseCase(
        repository=PostgresAccountSettingsRepository(
            gateway=PsycopgIdentityPostgresGateway(dsn=args.dsn)
        ),
        clock=SystemIdentityClock(),
    )
    results = execute_cleanup(
        candidates=candidates,
        archive_client=archive_client,
        audit_recorder=audit_recorder,
        source=args.source,
    )
    payload = summarize_results(results=results, source=args.source)
    if args.repair_archived_audit:
        repair_candidates = load_archived_audit_repair_candidates(
            dsn=args.dsn,
            label_prefixes=label_prefixes,
            owner_user_id=args.owner_user_id,
            created_after=args.created_after,
            created_before=args.created_before,
            limit=args.limit,
        )
        repair_archived_audit_events(
            candidates=repair_candidates,
            audit_recorder=audit_recorder,
        )
        payload["audit_repairs"] = summarize_audit_repairs(
            candidates=repair_candidates,
        )
    print(json.dumps(payload))
    return 0 if all(result.result == "archived" for result in results) else 1


def _normalize_label_prefixes(label_prefixes: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(prefix.strip() for prefix in label_prefixes if prefix.strip())
    if not normalized:
        raise ValueError("At least one cleanup label prefix is required.")
    return normalized


def _matches_label_prefix(*, label: str, label_prefixes: Sequence[str]) -> bool:
    return any(label.startswith(prefix) for prefix in label_prefixes)


def _matched_prefix(label: str, label_prefixes: Sequence[str]) -> str:
    for prefix in label_prefixes:
        if label.startswith(prefix):
            return prefix
    return "unknown"


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
