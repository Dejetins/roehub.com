from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from uuid import UUID

from apps.api.exchange_control_client import ExchangeConnectionCommandResult
from tools.exchange_connections.reclassify_non_trading_active import (
    ExchangeConnectionReclassificationAuditRepairCandidate,
    ExchangeConnectionReclassificationCandidate,
    execute_reclassification,
    normalize_source,
    repair_reclassification_audit_events,
    select_reclassification_candidates,
    summarize_candidates,
    summarize_results,
)
from trading.contexts.exchange_control.application.connections import (
    RECLASSIFIED_NON_TRADING_STATUS_REASON,
)
from trading.shared_kernel.primitives import UserId


def test_reclassification_selects_active_non_trading_rows_only() -> None:
    rows = [
        _row(
            connection_id="00000000-0000-0000-0000-000000000001",
            summary={
                "validation_status": "valid_trade_enabled",
                "exchange_permissions": "trade",
                "effective_permissions": "trade",
                "effective_capability": "trading",
                "connection_readiness": "ready_for_trading",
                "connection_readiness_reason": "trading_policy_ok",
            },
        ),
        _row(
            connection_id="00000000-0000-0000-0000-000000000002",
            summary={
                "validation_status": "valid_readonly",
                "exchange_permissions": "read",
                "effective_permissions": "read",
                "effective_capability": "none",
                "connection_readiness": "rejected",
                "connection_readiness_reason": "read_only_not_supported",
            },
        ),
        _row(
            connection_id="00000000-0000-0000-0000-000000000003",
            status="disabled",
            summary={
                "validation_status": "valid_readonly",
                "exchange_permissions": "read",
                "effective_capability": "none",
                "connection_readiness": "rejected",
            },
        ),
        _row(
            connection_id="00000000-0000-0000-0000-000000000004",
            summary={
                "validation_status": "permission_mismatch",
                "exchange_permissions": "read",
                "effective_permissions": "read",
            },
        ),
        _row(
            connection_id="00000000-0000-0000-0000-000000000005",
            archived_at=datetime(2026, 5, 27, 11, 0, tzinfo=timezone.utc),
            summary={
                "validation_status": "valid_readonly",
                "exchange_permissions": "read",
                "effective_capability": "none",
                "connection_readiness": "rejected",
            },
        ),
    ]

    candidates = select_reclassification_candidates(rows=rows)

    assert [str(candidate.connection_id) for candidate in candidates] == [
        "00000000-0000-0000-0000-000000000002",
        "00000000-0000-0000-0000-000000000004",
    ]
    assert "read_only_not_supported" in candidates[0].reasons
    assert "permission_mismatch" in candidates[1].reasons
    evidence = summarize_candidates(candidates=candidates, source="Stage 10D!")
    assert evidence["mode"] == "dry-run"
    assert evidence["candidate_count"] == 2
    assert evidence["safety"] == "physical hard delete запрещен"
    assert "00000000-0000-0000-0000-000000000002" not in str(evidence)


def test_reclassification_executes_supported_disable_and_records_audit() -> None:
    candidate = ExchangeConnectionReclassificationCandidate(
        connection_id=UUID("00000000-0000-0000-0000-000000000102"),
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000202"),
        exchange_name="bybit",
        market_type="spot",
        environment="mainnet",
        label="manual_readonly",
        status="active",
        status_reason=None,
        effective_capability="none",
        connection_readiness="rejected",
        connection_readiness_reason="read_only_not_supported",
        reasons=("exchange_permissions=read", "read_only_not_supported"),
        created_at=datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc),
    )
    client = _RecordingReclassificationClient()
    audit_recorder = _RecordingReclassificationAuditRecorder()

    results = execute_reclassification(
        candidates=(candidate,),
        client=client,
        audit_recorder=audit_recorder,
        source="stage10d",
    )

    assert [result.result for result in results] == ["reclassified"]
    assert client.calls == [
        {
            "owner_user_id": "00000000-0000-0000-0000-000000000202",
            "connection_id": "00000000-0000-0000-0000-000000000102",
            "status_reason": RECLASSIFIED_NON_TRADING_STATUS_REASON,
            "reclassification_source": "stage10d",
        }
    ]
    assert audit_recorder.events == [
        {
            "owner_user_id": "00000000-0000-0000-0000-000000000202",
            "connection_id": "00000000-0000-0000-0000-000000000102",
            "event": "exchange_connection_reclassified",
            "previous_status": "active",
            "new_status": "disabled",
            "reason": "read_only_not_supported",
            "source": "stage10d",
        }
    ]
    evidence = summarize_results(results=results, source="stage10d")
    assert evidence["reclassified_count"] == 1
    assert evidence["items"][0]["result"] == "reclassified"  # type: ignore[index]


def test_reclassification_source_is_bounded_for_metrics() -> None:
    assert normalize_source("Stage 10D Reclassification!") == "stage_10d_reclassification_"
    assert normalize_source("") == "stage10d"
    assert len(normalize_source("x" * 80)) == 40


def test_reclassification_audit_repair_records_missing_event_only() -> None:
    candidate = ExchangeConnectionReclassificationAuditRepairCandidate(
        connection_id=UUID("00000000-0000-0000-0000-000000000103"),
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000203"),
        exchange_name="bybit",
        market_type="spot",
        environment="mainnet",
        label="partially_reclassified",
        status="disabled",
        status_reason=RECLASSIFIED_NON_TRADING_STATUS_REASON,
        connection_readiness_reason="read_only_not_supported",
        created_at=datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc),
        disabled_at=datetime(2026, 5, 27, 10, 5, tzinfo=timezone.utc),
    )
    audit_recorder = _RecordingReclassificationAuditRecorder()

    repaired = repair_reclassification_audit_events(
        candidates=(candidate,),
        audit_recorder=audit_recorder,
        source="stage10d",
    )

    assert repaired == 1
    assert audit_recorder.events == [
        {
            "owner_user_id": "00000000-0000-0000-0000-000000000203",
            "connection_id": "00000000-0000-0000-0000-000000000103",
            "event": "exchange_connection_reclassified",
            "previous_status": "active",
            "new_status": "disabled",
            "reason": "read_only_not_supported",
            "source": "stage10d",
        }
    ]


def _row(
    *,
    connection_id: str,
    summary: dict[str, object],
    status: str = "active",
    archived_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "connection_id": connection_id,
        "owner_user_id": "00000000-0000-0000-0000-000000000901",
        "exchange_name": "binance",
        "market_type": "spot",
        "environment": "mainnet",
        "label": "candidate",
        "status": status,
        "status_reason": None,
        "permission_summary_json": summary,
        "ip_restriction_status": "configured",
        "created_at": datetime(2026, 5, 27, 8, 0, tzinfo=timezone.utc),
        "archived_at": archived_at,
    }


class _RecordingReclassificationClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def disable_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        status_reason: str | None = None,
        reclassification_source: str | None = None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        _ = request_id
        self.calls.append(
            {
                "owner_user_id": owner_user_id,
                "connection_id": connection_id,
                "status_reason": status_reason or "",
                "reclassification_source": reclassification_source or "",
            }
        )
        return replace(
            _command_result(),
            connection_id=connection_id,
            status="disabled",
            status_reason=status_reason,
            connection_readiness="rejected",
            connection_readiness_reason="read_only_not_supported",
        )


class _RecordingReclassificationAuditRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, str]] = []

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
    ) -> None:
        _ = exchange_name, market_type, environment
        self.events.append(
            {
                "owner_user_id": str(owner_user_id),
                "connection_id": connection_id,
                "event": "exchange_connection_reclassified",
                "previous_status": previous_status,
                "new_status": new_status,
                "reason": reason,
                "source": source,
            }
        )


def _command_result() -> ExchangeConnectionCommandResult:
    now = datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc)
    return ExchangeConnectionCommandResult(
        connection_id="00000000-0000-0000-0000-000000000102",
        credential_version_id="00000000-0000-0000-0000-000000000302",
        exchange_name="bybit",
        market_type="spot",
        environment="mainnet",
        label="manual_readonly",
        permissions="trade",
        requested_permissions="trade",
        exchange_permissions="read",
        effective_permissions="read",
        permission_warnings=(),
        api_key="****1234",
        status="disabled",
        status_reason=RECLASSIFIED_NON_TRADING_STATUS_REASON,
        validation_status="valid_readonly",
        validation_reason="readonly_key",
        ip_restriction_status="configured",
        last_validated_at=now,
        created_at=now,
        updated_at=now,
        disabled_at=now,
        archived_at=None,
        requested_capability="trading",
        effective_capability="none",
        connection_readiness="rejected",
        connection_readiness_reason="read_only_not_supported",
        permissions_deprecated=True,
    )
