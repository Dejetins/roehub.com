from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from uuid import UUID

from apps.api.exchange_control_client import ExchangeConnectionCommandResult
from tools.exchange_connection_cleanup import (
    ExchangeConnectionCleanupCandidate,
    execute_cleanup,
    normalize_cleanup_source,
    select_cleanup_candidates,
    summarize_candidates,
)
from trading.shared_kernel.primitives import UserId


def test_cleanup_candidate_selection_is_conservative() -> None:
    disabled_at = datetime(2026, 5, 26, 10, 0, tzinfo=timezone.utc)
    rows = [
        _row(
            connection_id="00000000-0000-0000-0000-000000000001",
            label="stage08_old_disabled",
            status="disabled",
            disabled_at=disabled_at,
        ),
        _row(
            connection_id="00000000-0000-0000-0000-000000000002",
            label="e2e_active_must_not_archive",
            status="active",
            disabled_at=None,
        ),
        _row(
            connection_id="00000000-0000-0000-0000-000000000003",
            label="manual_disabled",
            status="disabled",
            disabled_at=disabled_at,
        ),
        _row(
            connection_id="00000000-0000-0000-0000-000000000004",
            label="smoke_already_archived",
            status="disabled",
            disabled_at=disabled_at,
            archived_at=datetime(2026, 5, 26, 11, 0, tzinfo=timezone.utc),
        ),
        _row(
            connection_id="00000000-0000-0000-0000-000000000005",
            label="e2e_ready",
            status="disabled",
            disabled_at=disabled_at,
        ),
    ]

    candidates = select_cleanup_candidates(rows=rows)

    assert [str(candidate.connection_id) for candidate in candidates] == [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000005",
    ]
    evidence = summarize_candidates(candidates=candidates, source="stage09d", dry_run=True)
    assert evidence["mode"] == "dry-run"
    assert evidence["count"] == 2
    assert "00000000-0000-0000-0000-000000000001" not in str(evidence)


def test_execute_cleanup_archives_through_client_and_records_audit() -> None:
    candidate = ExchangeConnectionCleanupCandidate(
        connection_id=UUID("00000000-0000-0000-0000-000000000101"),
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000201"),
        exchange_name="binance",
        market_type="spot",
        environment="testnet",
        label="e2e_stage09d_candidate",
        status="disabled",
        created_at=datetime(2026, 5, 26, 9, 0, tzinfo=timezone.utc),
        disabled_at=datetime(2026, 5, 26, 10, 0, tzinfo=timezone.utc),
    )
    archive_client = _RecordingArchiveClient()
    audit_recorder = _RecordingAuditRecorder()

    results = execute_cleanup(
        candidates=(candidate,),
        archive_client=archive_client,
        audit_recorder=audit_recorder,
        source="stage09d",
    )

    assert [result.result for result in results] == ["archived"]
    assert archive_client.calls == [
        {
            "owner_user_id": "00000000-0000-0000-0000-000000000201",
            "connection_id": "00000000-0000-0000-0000-000000000101",
            "cleanup_source": "stage09d",
        }
    ]
    assert audit_recorder.events == [
        {
            "owner_user_id": "00000000-0000-0000-0000-000000000201",
            "connection_id": "00000000-0000-0000-0000-000000000101",
            "event": "exchange_connection_archived",
            "previous_status": "disabled",
            "new_status": "archived",
            "reason": "user_archived",
        }
    ]


def test_cleanup_source_is_bounded_for_metrics() -> None:
    assert normalize_cleanup_source("Stage 09D Cleanup!") == "stage_09d_cleanup_"
    assert normalize_cleanup_source("") == "stage09d"
    assert len(normalize_cleanup_source("x" * 80)) == 40


def _row(
    *,
    connection_id: str,
    label: str,
    status: str,
    disabled_at: datetime | None,
    archived_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "connection_id": connection_id,
        "owner_user_id": "00000000-0000-0000-0000-000000000901",
        "exchange_name": "binance",
        "market_type": "spot",
        "environment": "testnet",
        "label": label,
        "status": status,
        "created_at": datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc),
        "disabled_at": disabled_at,
        "archived_at": archived_at,
    }


class _RecordingArchiveClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def archive_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        cleanup_source: str | None = None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        _ = request_id
        self.calls.append(
            {
                "owner_user_id": owner_user_id,
                "connection_id": connection_id,
                "cleanup_source": cleanup_source or "",
            }
        )
        return replace(
            _command_result(),
            connection_id=connection_id,
            status="archived",
            status_reason="user_archived",
        )


class _RecordingAuditRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, str]] = []

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
    ) -> None:
        _ = exchange_name, market_type, environment
        self.events.append(
            {
                "owner_user_id": str(owner_user_id),
                "connection_id": connection_id,
                "event": "exchange_connection_archived",
                "previous_status": previous_status,
                "new_status": new_status,
                "reason": reason,
            }
        )


def _command_result() -> ExchangeConnectionCommandResult:
    now = datetime(2026, 5, 26, 10, 0, tzinfo=timezone.utc)
    return ExchangeConnectionCommandResult(
        connection_id="00000000-0000-0000-0000-000000000101",
        credential_version_id="00000000-0000-0000-0000-000000000301",
        exchange_name="binance",
        market_type="spot",
        environment="testnet",
        label="e2e_stage09d_candidate",
        permissions="read",
        requested_permissions="read",
        exchange_permissions="unknown",
        effective_permissions="none",
        permission_warnings=(),
        api_key="****1234",
        status="archived",
        status_reason="user_archived",
        validation_status="skipped_external_validation",
        validation_reason="not_validated",
        ip_restriction_status="unknown",
        last_validated_at=None,
        created_at=now,
        updated_at=now,
        disabled_at=now,
        archived_at=now,
    )
