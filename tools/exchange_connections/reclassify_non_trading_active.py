from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from apps.api.exchange_control_client import (
    ExchangeConnectionCommandResult,
    ExchangeControlClientError,
    HttpExchangeControlClient,
)
from trading.contexts.exchange_control.application.connections import (
    RECLASSIFIED_NON_TRADING_STATUS_REASON,
    trading_capability_summary,
)
from trading.contexts.identity.adapters.outbound import (
    PostgresAccountSettingsRepository,
    PsycopgIdentityPostgresGateway,
    SystemIdentityClock,
)
from trading.contexts.identity.application.use_cases.account_settings import (
    AccountSettingsUseCase,
)
from trading.platform.secrets import SecureCredentialFile
from trading.shared_kernel.primitives import UserId

DEFAULT_SOURCE = "stage10d"
DEFAULT_LIMIT = 200
_DSN_ENV = "IDENTITY_PG_DSN"
_ALT_DSN_ENV = "ROEHUB_PG_DSN"
_BASE_URL_ENV = "ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL"
_TOKEN_FILE_ENV = "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN_FILE"
_TIMEOUT_ENV = "ROEHUB_EXCHANGE_CONTROL_INTERNAL_TIMEOUT_SECONDS"


@dataclass(frozen=True, slots=True)
class ExchangeConnectionReclassificationCandidate:
    connection_id: UUID
    owner_user_id: UserId
    exchange_name: str
    market_type: str
    environment: str
    label: str | None
    status: str
    status_reason: str | None
    effective_capability: str
    connection_readiness: str
    connection_readiness_reason: str
    reasons: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ExchangeConnectionReclassificationResult:
    candidate: ExchangeConnectionReclassificationCandidate
    result: str
    status: str | None
    status_reason: str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExchangeConnectionReclassificationAuditRepairCandidate:
    connection_id: UUID
    owner_user_id: UserId
    exchange_name: str
    market_type: str
    environment: str
    label: str | None
    status: str
    status_reason: str
    connection_readiness_reason: str
    created_at: datetime
    disabled_at: datetime


class ExchangeConnectionReclassificationClient(Protocol):
    def disable_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        status_reason: str | None = None,
        reclassification_source: str | None = None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult: ...


class ExchangeConnectionReclassificationAuditRecorder(Protocol):
    def record_exchange_connection_reclassification(
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
        source: str,
    ) -> None: ...


def select_reclassification_candidates(
    *,
    rows: Iterable[Mapping[str, object]],
) -> tuple[ExchangeConnectionReclassificationCandidate, ...]:
    candidates: list[ExchangeConnectionReclassificationCandidate] = []
    for row in rows:
        if str(row.get("status") or "") != "active":
            continue
        if row.get("archived_at") is not None:
            continue
        summary = _summary_dict(row.get("permission_summary_json"))
        computed = trading_capability_summary(
            status="active",
            status_reason=_optional_string(row.get("status_reason")),
            validation_status=_summary_string(
                summary=summary,
                key="validation_status",
                default="skipped_external_validation",
            ),
            validation_reason=_summary_optional_string(
                summary=summary,
                key="validation_reason",
            ),
            ip_restriction_status=str(row.get("ip_restriction_status") or "unknown"),
            exchange_permissions=_summary_string(
                summary=summary,
                key="exchange_permissions",
                default="unknown",
            ),
        )
        effective_capability = _summary_string(
            summary=summary,
            key="effective_capability",
            default=str(computed["effective_capability"]),
        )
        connection_readiness = _summary_string(
            summary=summary,
            key="connection_readiness",
            default=str(computed["connection_readiness"]),
        )
        readiness_reason = _summary_string(
            summary=summary,
            key="connection_readiness_reason",
            default=str(computed["connection_readiness_reason"]),
        )
        reasons = _candidate_reasons(
            summary=summary,
            effective_capability=effective_capability,
            connection_readiness=connection_readiness,
            readiness_reason=readiness_reason,
        )
        if not reasons:
            continue
        candidates.append(
            ExchangeConnectionReclassificationCandidate(
                connection_id=UUID(str(row["connection_id"])),
                owner_user_id=UserId.from_string(str(row["owner_user_id"])),
                exchange_name=str(row["exchange_name"]),
                market_type=str(row["market_type"]),
                environment=str(row["environment"]),
                label=_optional_string(row.get("label")),
                status="active",
                status_reason=_optional_string(row.get("status_reason")),
                effective_capability=effective_capability,
                connection_readiness=connection_readiness,
                connection_readiness_reason=readiness_reason,
                reasons=tuple(reasons),
                created_at=_coerce_datetime(row["created_at"]),
            )
        )
    candidates.sort(key=lambda item: (item.created_at, str(item.connection_id)))
    return tuple(candidates)


def load_reclassification_candidates(
    *,
    dsn: str,
    owner_user_id: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> tuple[ExchangeConnectionReclassificationCandidate, ...]:
    parameters: dict[str, object] = {"limit": limit}
    predicates = ["status = 'active'", "archived_at IS NULL"]
    if owner_user_id:
        parameters["owner_user_id"] = str(UserId.from_string(owner_user_id))
        predicates.append("owner_user_id = %(owner_user_id)s")
    query = f"""
        SELECT
            connection_id,
            owner_user_id,
            exchange_name,
            market_type,
            environment,
            label,
            status,
            status_reason,
            permission_summary_json,
            ip_restriction_status,
            created_at,
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
    return select_reclassification_candidates(
        rows=cast(Iterable[Mapping[str, object]], rows)
    )


def load_reclassification_audit_repair_candidates(
    *,
    dsn: str,
    owner_user_id: str | None = None,
    limit: int = DEFAULT_LIMIT,
    missing_audit_only: bool = True,
) -> tuple[ExchangeConnectionReclassificationAuditRepairCandidate, ...]:
    parameters: dict[str, object] = {"limit": limit}
    predicates = [
        "c.status = 'disabled'",
        "c.status_reason = %(status_reason)s",
        "c.disabled_at IS NOT NULL",
    ]
    parameters["status_reason"] = RECLASSIFIED_NON_TRADING_STATUS_REASON
    if missing_audit_only:
        predicates.append(
            """
            NOT EXISTS (
                SELECT 1
                FROM identity_audit_events AS audit
                WHERE audit.owner_user_id = c.owner_user_id
                  AND audit.event_type = 'exchange_connection_reclassified'
                  AND audit.metadata_json->>'connection_id' = c.connection_id::text
            )
            """
        )
    if owner_user_id:
        parameters["owner_user_id"] = str(UserId.from_string(owner_user_id))
        predicates.append("c.owner_user_id = %(owner_user_id)s")
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
            c.permission_summary_json,
            c.created_at,
            c.disabled_at
        FROM exchange_connections AS c
        WHERE {' AND '.join(predicates)}
        ORDER BY c.disabled_at ASC, c.connection_id ASC
        LIMIT %(limit)s
    """
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(cast(Any, query), parameters)
            rows = cursor.fetchall()
    return tuple(
        ExchangeConnectionReclassificationAuditRepairCandidate(
            connection_id=UUID(str(row["connection_id"])),
            owner_user_id=UserId.from_string(str(row["owner_user_id"])),
            exchange_name=str(row["exchange_name"]),
            market_type=str(row["market_type"]),
            environment=str(row["environment"]),
            label=_optional_string(row.get("label")),
            status=str(row["status"]),
            status_reason=str(row["status_reason"]),
            connection_readiness_reason=_summary_string(
                summary=_summary_dict(row.get("permission_summary_json")),
                key="connection_readiness_reason",
                default="reclassified",
            ),
            created_at=_coerce_datetime(row["created_at"]),
            disabled_at=_coerce_datetime(row["disabled_at"]),
        )
        for row in cast(Iterable[Mapping[str, object]], rows)
    )


def execute_reclassification(
    *,
    candidates: Sequence[ExchangeConnectionReclassificationCandidate],
    client: ExchangeConnectionReclassificationClient,
    audit_recorder: ExchangeConnectionReclassificationAuditRecorder,
    source: str,
) -> tuple[ExchangeConnectionReclassificationResult, ...]:
    normalized_source = normalize_source(source)
    results: list[ExchangeConnectionReclassificationResult] = []
    for candidate in candidates:
        try:
            disabled = client.disable_connection(
                owner_user_id=str(candidate.owner_user_id),
                connection_id=str(candidate.connection_id),
                status_reason=RECLASSIFIED_NON_TRADING_STATUS_REASON,
                reclassification_source=normalized_source,
                request_id=f"stage10d-reclassify-{redacted_uuid(candidate.connection_id)}",
            )
        except ExchangeControlClientError as error:
            results.append(
                ExchangeConnectionReclassificationResult(
                    candidate=candidate,
                    result="rejected",
                    status=None,
                    status_reason=None,
                    error=str(error),
                )
            )
            continue
        audit_recorder.record_exchange_connection_reclassification(
            owner_user_id=candidate.owner_user_id,
            connection_id=str(candidate.connection_id),
            exchange_name=candidate.exchange_name,
            market_type=candidate.market_type,
            environment=candidate.environment,
            previous_status="active",
            new_status=disabled.status,
            reason=disabled.connection_readiness_reason,
            source=normalized_source,
        )
        results.append(
            ExchangeConnectionReclassificationResult(
                candidate=candidate,
                result="reclassified",
                status=disabled.status,
                status_reason=disabled.status_reason,
            )
        )
    return tuple(results)


def repair_reclassification_audit_events(
    *,
    candidates: Sequence[ExchangeConnectionReclassificationAuditRepairCandidate],
    audit_recorder: ExchangeConnectionReclassificationAuditRecorder,
    source: str,
) -> int:
    normalized_source = normalize_source(source)
    for candidate in candidates:
        audit_recorder.record_exchange_connection_reclassification(
            owner_user_id=candidate.owner_user_id,
            connection_id=str(candidate.connection_id),
            exchange_name=candidate.exchange_name,
            market_type=candidate.market_type,
            environment=candidate.environment,
            previous_status="active",
            new_status=candidate.status,
            reason=candidate.connection_readiness_reason,
            source=normalized_source,
        )
    return len(candidates)


def emit_reclassification_metrics_for_repaired_rows(
    *,
    candidates: Sequence[ExchangeConnectionReclassificationAuditRepairCandidate],
    client: ExchangeConnectionReclassificationClient,
    source: str,
) -> int:
    normalized_source = normalize_source(source)
    emitted = 0
    for candidate in candidates:
        try:
            client.disable_connection(
                owner_user_id=str(candidate.owner_user_id),
                connection_id=str(candidate.connection_id),
                status_reason=RECLASSIFIED_NON_TRADING_STATUS_REASON,
                reclassification_source=normalized_source,
                request_id=f"stage10d-metric-{redacted_uuid(candidate.connection_id)}",
            )
        except ExchangeControlClientError:
            continue
        emitted += 1
    return emitted


def summarize_candidates(
    *,
    candidates: Sequence[ExchangeConnectionReclassificationCandidate],
    source: str,
) -> dict[str, object]:
    return {
        "mode": "dry-run",
        "source": normalize_source(source),
        "candidate_count": len(candidates),
        "safety": "physical hard delete запрещен",
        "items": [_candidate_summary(candidate) for candidate in candidates],
    }


def summarize_results(
    *,
    results: Sequence[ExchangeConnectionReclassificationResult],
    source: str,
    audit_repairs: Sequence[ExchangeConnectionReclassificationAuditRepairCandidate] = (),
) -> dict[str, object]:
    return {
        "mode": "execute",
        "source": normalize_source(source),
        "candidate_count": len(results),
        "reclassified_count": sum(1 for result in results if result.result == "reclassified"),
        "audit_repair_count": len(audit_repairs),
        "safety": "physical hard delete запрещен",
        "items": [
            {
                **_candidate_summary(result.candidate),
                "result": result.result,
                "status": result.status,
                "status_reason": result.status_reason,
                "error": result.error,
            }
            for result in results
        ],
        "audit_repairs": [
            {
                "connection_ref": redacted_uuid(candidate.connection_id),
                "owner_ref": redacted_uuid(candidate.owner_user_id.value),
                "label_ref": (
                    hashlib.sha256(candidate.label.encode("utf-8")).hexdigest()[:12]
                    if candidate.label
                    else None
                ),
                "exchange_name": candidate.exchange_name,
                "market_type": candidate.market_type,
                "environment": candidate.environment,
                "status": candidate.status,
                "status_reason": candidate.status_reason,
                "reason": candidate.connection_readiness_reason,
                "disabled_at": candidate.disabled_at.isoformat(),
            }
            for candidate in audit_repairs
        ],
    }


def normalize_source(value: str) -> str:
    stripped = value.strip().lower()
    if not stripped:
        return DEFAULT_SOURCE
    normalized = "".join(
        character
        if character.isascii()
        and (character.isalnum() or character in {"_", "-"})
        else "_"
        for character in stripped
    )
    return normalized[:40] or DEFAULT_SOURCE


def redacted_uuid(value: UUID) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or execute Stage 10D active non-trading exchange connection "
            "reclassification through the supported disable lifecycle path."
        )
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get(_DSN_ENV) or os.environ.get(_ALT_DSN_ENV, ""),
    )
    parser.add_argument("--exchange-control-url", default=os.environ.get(_BASE_URL_ENV, ""))
    parser.add_argument(
        "--exchange-control-token-file",
        default=os.environ.get(_TOKEN_FILE_ENV, ""),
    )
    parser.add_argument(
        "--exchange-control-timeout",
        type=float,
        default=float(os.environ.get(_TIMEOUT_ENV, "2.0")),
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--owner-user-id", default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--json", action="store_true", default=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dsn.strip():
        raise SystemExit(f"Postgres DSN is required via --dsn, ${_DSN_ENV}, or ${_ALT_DSN_ENV}.")
    dry_run = not bool(args.execute)
    candidates = load_reclassification_candidates(
        dsn=args.dsn,
        owner_user_id=args.owner_user_id,
        limit=args.limit,
    )
    if dry_run:
        print(json.dumps(summarize_candidates(candidates=candidates, source=args.source)))
        return 0
    if not args.exchange_control_url.strip() or not args.exchange_control_token_file.strip():
        raise SystemExit(
            "Execution requires --exchange-control-url and --exchange-control-token-file."
        )
    client = HttpExchangeControlClient(
        base_url=args.exchange_control_url,
        internal_api_credential=SecureCredentialFile(
            Path(args.exchange_control_token_file).expanduser().resolve()
        ),
        timeout_seconds=args.exchange_control_timeout,
    )
    audit_recorder = AccountSettingsUseCase(
        repository=PostgresAccountSettingsRepository(
            gateway=PsycopgIdentityPostgresGateway(dsn=args.dsn)
        ),
        clock=SystemIdentityClock(),
    )
    results = execute_reclassification(
        candidates=candidates,
        client=client,
        audit_recorder=audit_recorder,
        source=args.source,
    )
    audit_repairs = load_reclassification_audit_repair_candidates(
        dsn=args.dsn,
        owner_user_id=args.owner_user_id,
        limit=args.limit,
    )
    repair_reclassification_audit_events(
        candidates=audit_repairs,
        audit_recorder=audit_recorder,
        source=args.source,
    )
    metric_repairs = load_reclassification_audit_repair_candidates(
        dsn=args.dsn,
        owner_user_id=args.owner_user_id,
        limit=args.limit,
        missing_audit_only=False,
    )
    metric_repair_count = emit_reclassification_metrics_for_repaired_rows(
        candidates=metric_repairs,
        client=client,
        source=args.source,
    )
    payload = summarize_results(
        results=results,
        source=args.source,
        audit_repairs=audit_repairs,
    )
    payload["metric_repair_count"] = metric_repair_count
    print(
        json.dumps(payload)
    )
    return 0 if all(result.result == "reclassified" for result in results) else 1


def _candidate_reasons(
    *,
    summary: Mapping[str, object],
    effective_capability: str,
    connection_readiness: str,
    readiness_reason: str,
) -> list[str]:
    reasons: list[str] = []
    validation_status = _summary_string(
        summary=summary,
        key="validation_status",
        default="",
    )
    exchange_permissions = _summary_string(
        summary=summary,
        key="exchange_permissions",
        default="unknown",
    )
    effective_permissions = _summary_string(
        summary=summary,
        key="effective_permissions",
        default="none",
    )
    if validation_status == "permission_mismatch":
        reasons.append("permission_mismatch")
    if effective_permissions == "read":
        reasons.append("effective_permissions=read")
    if exchange_permissions == "read":
        reasons.append("exchange_permissions=read")
    if effective_capability != "trading":
        reasons.append(f"effective_capability={effective_capability}")
    if connection_readiness != "ready_for_trading":
        reasons.append(f"connection_readiness={connection_readiness}")
    if reasons and readiness_reason and readiness_reason not in reasons:
        reasons.append(readiness_reason)
    if (
        reasons
        and readiness_reason == "read_only_not_supported"
        and "read_only_not_supported" not in reasons
    ):
        reasons.append("read_only_not_supported")
    return reasons


def _candidate_summary(
    candidate: ExchangeConnectionReclassificationCandidate,
) -> dict[str, object]:
    return {
        "connection_ref": redacted_uuid(candidate.connection_id),
        "owner_ref": redacted_uuid(candidate.owner_user_id.value),
        "label_ref": (
            hashlib.sha256(candidate.label.encode("utf-8")).hexdigest()[:12]
            if candidate.label
            else None
        ),
        "exchange_name": candidate.exchange_name,
        "market_type": candidate.market_type,
        "environment": candidate.environment,
        "status": candidate.status,
        "status_reason": candidate.status_reason,
        "effective_capability": candidate.effective_capability,
        "connection_readiness": candidate.connection_readiness,
        "connection_readiness_reason": candidate.connection_readiness_reason,
        "reasons": list(candidate.reasons),
        "created_at": candidate.created_at.isoformat(),
    }


def _summary_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _summary_string(
    *,
    summary: Mapping[str, object],
    key: str,
    default: str,
) -> str:
    value = summary.get(key)
    if isinstance(value, str) and value:
        return value
    return default


def _summary_optional_string(
    *,
    summary: Mapping[str, object],
    key: str,
) -> str | None:
    value = summary.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
