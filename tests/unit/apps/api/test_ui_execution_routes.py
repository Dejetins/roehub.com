from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_ui_execution_router
from apps.api.wiring.modules.research_tenancy import DevelopmentOrganizationScopeResolver
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryExecutionIntentRepository,
)
from trading.contexts.live_execution.application import ExecutionIngressService
from trading.contexts.live_execution.application.ports import (
    ExecutionDispatchPublishResult,
    ExecutionRiskContextQuery,
    ExecutionRiskContextResolver,
    FailClosedExecutionRiskContextResolver,
)
from trading.contexts.live_execution.application.use_cases.execution_dispatch import (
    ExecutionDispatchService,
)
from trading.contexts.live_execution.domain import ExecutionRiskContext
from trading.shared_kernel.primitives import PaidLevel, UserId

_USER_ID = "00000000-0000-0000-0000-000000010501"


class _Clock:
    def __init__(self) -> None:
        self._index = 0

    def now(self) -> datetime:
        value = datetime(2026, 5, 31, 13, 30, tzinfo=UTC) + timedelta(seconds=self._index)
        self._index += 1
        return value


class _CurrentUserDependency:
    def __call__(self, request: Request) -> CurrentUserPrincipal:
        return CurrentUserPrincipal(
            user_id=UserId.from_string(request.headers["x-user-id"]),
            paid_level=PaidLevel.free(),
        )


def test_ui_execution_routes_create_and_dedupe_intent() -> None:
    client = _build_client()

    source_response = client.post(
        "/ui/execution/source-events",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "ops_test",
            "source_event_ref": "ops-stage10-ref",
            "source_ref": {"ops_test_id": "stage10"},
            "idempotency_key": "source-key",
        },
    )
    source_replay = client.post(
        "/ui/execution/source-events",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "ops_test",
            "source_event_ref": "ops-stage10-ref",
            "source_ref": {"ops_test_id": "stage10"},
            "idempotency_key": "source-key",
        },
    )
    intent_payload = {
        "source_event_id": source_response.json()["source_event_id"],
        "idempotency_key": "intent-key",
        "exchange_connection_id": "00000000-0000-0000-0000-000000010601",
        "market_type": "spot",
        "instrument_key": "binance:spot:BTCUSDT",
        "order": {
            "order_type": "limit",
            "side": "buy",
            "quantity": "0.01",
            "limit_price": "10000",
        },
    }
    intent_response = client.post(
        "/ui/execution/intents",
        headers={"x-user-id": _USER_ID},
        json=intent_payload,
    )
    intent_replay = client.post(
        "/ui/execution/intents",
        headers={"x-user-id": _USER_ID},
        json=intent_payload,
    )

    assert source_response.status_code == 201
    assert source_replay.status_code == 200
    assert source_replay.json()["duplicate"] is True
    assert intent_response.status_code == 201
    assert intent_replay.status_code == 200
    assert intent_replay.json()["duplicate"] is True
    assert intent_replay.json()["intent_id"] == intent_response.json()["intent_id"]
    assert intent_response.json()["order_type"] == "limit"
    assert intent_response.json()["status"] == "accepted"
    assert intent_response.json()["status_reason"] == "risk_gate_accepted"


def test_ui_execution_route_dispatches_accepted_intent_when_dispatch_service_is_wired() -> None:
    repository = InMemoryExecutionIntentRepository()
    client = _build_client(repository=repository, dispatch_transport=_DispatchTransport())

    source_response = client.post(
        "/ui/execution/source-events",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "ops_test",
            "source_event_ref": "ops-stage12-ref",
            "source_ref": {"ops_test_id": "stage12"},
            "idempotency_key": "stage12-source-key",
        },
    )
    intent_response = client.post(
        "/ui/execution/intents",
        headers={"x-user-id": _USER_ID},
        json={
            "source_event_id": source_response.json()["source_event_id"],
            "idempotency_key": "stage12-intent-key",
            "exchange_connection_id": "00000000-0000-0000-0000-000000010601",
            "market_type": "spot",
            "instrument_key": "binance:spot:BTCUSDT",
            "order": {
                "order_type": "market",
                "side": "buy",
                "quote_notional": "25",
            },
        },
    )

    assert intent_response.status_code == 201
    assert intent_response.json()["status"] == "dispatched"
    assert intent_response.json()["status_reason"] == "redis_xadd_ok"
    assert intent_response.json()["dispatch_stream_name"] == "execution.requests.v1"
    assert intent_response.json()["dispatch_redis_message_id"] == "1-0"
    assert repository.intents[0].status == "dispatched"


def test_ui_execution_route_fails_closed_when_server_risk_state_is_unavailable() -> None:
    client = _build_client(
        risk_context_resolver=FailClosedExecutionRiskContextResolver()
    )

    source_id = client.post(
        "/ui/execution/source-events",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "ops_test",
            "source_event_ref": "ops-stage11-missing-risk",
            "source_ref": {"ops_test_id": "stage11"},
            "idempotency_key": "stage11-source-key",
        },
    ).json()["source_event_id"]

    response = client.post(
        "/ui/execution/intents",
        headers={"x-user-id": _USER_ID},
        json={
            "source_event_id": source_id,
            "idempotency_key": "stage11-intent-key",
            "exchange_connection_id": "00000000-0000-0000-0000-000000010601",
            "market_type": "spot",
            "instrument_key": "binance:spot:BTCUSDT",
            "order": {
                "order_type": "market",
                "side": "buy",
                "quote_notional": "25",
            },
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["details"]["reason"] == "risk_state_unavailable"


def test_ui_execution_route_rejects_client_supplied_risk_authority() -> None:
    repository = InMemoryExecutionIntentRepository()
    client = _build_client(repository=repository)
    source_id = client.post(
        "/ui/execution/source-events",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "ops_test",
            "source_event_ref": "client-risk-spoof",
            "source_ref": {"ops_test_id": "client-risk-spoof"},
            "idempotency_key": "client-risk-spoof-source",
        },
    ).json()["source_event_id"]

    response = client.post(
        "/ui/execution/intents",
        headers={"x-user-id": _USER_ID},
        json={
            "source_event_id": source_id,
            "idempotency_key": "client-risk-spoof-intent",
            "exchange_connection_id": "00000000-0000-0000-0000-000000010601",
            "market_type": "spot",
            "instrument_key": "binance:spot:BTCUSDT",
            "order": {
                "order_type": "market",
                "side": "buy",
                "quote_notional": "25",
            },
            "risk_context": _accepted_risk_context(),
        },
    )

    assert response.status_code == 422
    assert repository.intents == []


def test_ui_execution_route_rejects_unsupported_order_model() -> None:
    client = _build_client()
    source_id = client.post(
        "/ui/execution/source-events",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "manual_request",
            "source_event_ref": "manual-stage10-ref",
            "source_ref": {"manual_request_id": "manual-stage10"},
            "idempotency_key": "manual-source-key",
        },
    ).json()["source_event_id"]

    response = client.post(
        "/ui/execution/intents",
        headers={"x-user-id": _USER_ID},
        json={
            "source_event_id": source_id,
            "idempotency_key": "manual-intent-key",
            "exchange_connection_id": "00000000-0000-0000-0000-000000010601",
            "market_type": "spot",
            "instrument_key": "binance:spot:BTCUSDT",
            "order": {
                "order_type": "market",
                "side": "buy",
                "quote_notional": "25",
                "take_profit": {"price": "11000"},
            },
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "execution.unsupported_order_model"
    assert response.json()["error"]["details"]["reason"] == "tp_sl_not_supported"


def test_ui_execution_notifications_create_dedupe_and_list() -> None:
    client = _build_client()
    source_id = client.post(
        "/ui/execution/source-events",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "ops_test",
            "source_event_ref": "ops-terminal-ref",
            "source_ref": {"ops_test_id": "stage16"},
            "idempotency_key": "stage16-notification-source-key",
        },
    ).json()["source_event_id"]
    payload = {
        "source_type": "ops_test",
        "event_type": "producer_terminal",
        "severity": "info",
        "reason": "cancelled",
        "source_event_id": source_id,
        "labels": {"exchange": "bybit", "status": "cancelled"},
    }

    created = client.post(
        "/ui/execution/notifications",
        headers={"x-user-id": _USER_ID},
        json=payload,
    )
    replay = client.post(
        "/ui/execution/notifications",
        headers={"x-user-id": _USER_ID},
        json=payload,
    )
    listed = client.get("/ui/execution/notifications", headers={"x-user-id": _USER_ID})

    assert created.status_code == 201
    assert replay.status_code == 200
    assert replay.json()["duplicate"] is True
    assert replay.json()["notification_id"] == created.json()["notification_id"]
    assert listed.status_code == 200
    assert listed.json()["items"][0]["event_type"] == "producer_terminal"


def test_ui_execution_notifications_accept_stage13_event_types() -> None:
    client = _build_client()

    event_types = (
        "producer_signal_rejected",
        "producer_order_rejected",
        "producer_manual_exit",
        "producer_reconciliation_pending",
        "producer_strategy_stopped",
        "producer_strategy_restarted",
        "producer_soak_failed",
        "producer_soak_succeeded",
        "producer_resource_threshold_breached",
    )

    for index, event_type in enumerate(event_types, start=1):
        response = client.post(
            "/ui/execution/notifications",
            headers={"x-user-id": _USER_ID},
            json={
                "source_type": "ops_test",
                "event_type": event_type,
                "severity": (
                    "critical"
                    if event_type in {
                        "producer_reconciliation_pending",
                        "producer_soak_failed",
                        "producer_resource_threshold_breached",
                    }
                    else "info"
                ),
                "reason": f"{event_type}_api_dry_run",
                "labels": {"stage": "13", "row": str(index)},
            },
        )

        assert response.status_code == 201
        assert response.json()["event_type"] == event_type
        assert response.json()["labels"] == {"stage": "13", "row": str(index)}

    listed = client.get("/ui/execution/notifications", headers={"x-user-id": _USER_ID})

    assert listed.status_code == 200
    assert {item["event_type"] for item in listed.json()["items"]} >= set(event_types)


def test_ui_execution_notifications_reject_sensitive_labels() -> None:
    client = _build_client()

    response = client.post(
        "/ui/execution/notifications",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "ops_test",
            "event_type": "producer_unknown",
            "severity": "critical",
            "reason": "adapter_unknown_state_reconciliation_required",
            "labels": {"authorization": "secret"},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "execution.invalid_notification"
    assert response.json()["error"]["details"]["reason"] == "sensitive_notification_label_rejected"


def test_ui_execution_route_requires_strategy_signal_id_for_strategy_sources() -> None:
    client = _build_client()

    response = client.post(
        "/ui/execution/source-events",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "strategy_signal",
            "source_event_ref": "signal-without-id",
            "source_ref": {"strategy_id": str(UUID(int=1))},
            "idempotency_key": "bad-source-key",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "execution.invalid_source_event"
    assert response.json()["error"]["details"]["reason"] == "strategy_signal_id_required"


def _build_client(
    *,
    repository: InMemoryExecutionIntentRepository | None = None,
    dispatch_transport: object | None = None,
    risk_context_resolver: ExecutionRiskContextResolver | None = None,
) -> TestClient:
    intent_repository = repository or InMemoryExecutionIntentRepository()
    clock = _Clock()
    dispatch_service = None
    if dispatch_transport is not None:
        dispatch_service = ExecutionDispatchService(
            repository=intent_repository,
            transport=dispatch_transport,  # type: ignore[arg-type]
            clock=clock,
        )
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_ui_execution_router(
            ingress_service=ExecutionIngressService(
                repository=intent_repository,
                clock=clock,
            ),
            dispatch_service=dispatch_service,
            current_user_dependency=_CurrentUserDependency(),
            organization_scope_resolver=DevelopmentOrganizationScopeResolver(),
            risk_context_resolver=(
                risk_context_resolver or _TrustedRiskContextResolver()
            ),
        )
    )
    return TestClient(app)


class _DispatchTransport:
    def ensure_request_group(self) -> None:
        return None

    def request_stream_length(self) -> int:
        return 0

    def publish_request(self, *, intent, attempt_count: int) -> ExecutionDispatchPublishResult:
        _ = intent, attempt_count
        return ExecutionDispatchPublishResult(
            stream_name="execution.requests.v1",
            message_id="1-0",
        )

    def publish_retry(
        self, *, intent, reason: str, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        _ = intent, reason, attempt_count
        return ExecutionDispatchPublishResult(
            stream_name="execution.requests.retry.v1",
            message_id="2-0",
        )

    def publish_dlq(
        self, *, intent, reason: str, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        _ = intent, reason, attempt_count
        return ExecutionDispatchPublishResult(
            stream_name="execution.requests.dlq.v1",
            message_id="3-0",
        )

    def ack_after_durable_state_change(self, *, stream_name: str, message_id: str) -> None:
        _ = stream_name, message_id


class _TrustedRiskContextResolver:
    def resolve(self, *, query: ExecutionRiskContextQuery) -> ExecutionRiskContext:
        _ = query
        return ExecutionRiskContext(**_accepted_risk_context())  # type: ignore[arg-type]


def _accepted_risk_context() -> dict[str, object]:
    return {
        "organization_ownership_verified": True,
        "account_ownership_verified": True,
        "exchange_connection_active": True,
        "secret_custody_ready": True,
        "source_authorized": True,
        "strategy_variant_compatible": True,
        "market_data_state": "ready",
        "strategy_binding_active": True,
        "strategy_live_profile_ready": True,
        "strategy_run_active": True,
        "exchange_config_verified": True,
        "account_state_fresh": True,
        "position_ownership_active": True,
        "capital_reservation_active": True,
        "capital_reservation_sufficient": True,
        "paper_accounting_ready": True,
        "manual_recent_auth": True,
        "ml_agent_policy_active": True,
        "kill_switch_open": True,
        "environment_policy_allows": True,
        "max_order_size_ok": True,
        "daily_limit_ok": True,
    }
