from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

import pytest

from trading.contexts.notifications.adapters import (
    InMemoryNotificationRepository,
    InMemoryNotificationStatsSourceReader,
)
from trading.contexts.notifications.application import (
    InMemoryNotificationTelegramBindingStore,
    NotificationStatsQueryService,
    NotificationStatsSourceRow,
    NotificationTelegramBindingService,
    TelegramCommandHandler,
    TelegramInboundCommand,
)
from trading.contexts.notifications.application.telegram_commands import (
    TelegramCommandScopeAuthorizer,
)
from trading.platform.secrets import SecretValue
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))
_PROVIDER_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000003")


def _now() -> datetime:
    return datetime(2026, 6, 29, 15, 0, tzinfo=timezone.utc)


def _user_id() -> UserId:
    return UserId(UUID("11111111-1111-4111-8111-111111111111"))


def test_start_binding_uses_hashed_one_time_code_and_idempotent_updates() -> None:
    repository = InMemoryNotificationRepository()
    binding_store = InMemoryNotificationTelegramBindingStore()
    binding_service = NotificationTelegramBindingService(
        store=binding_store,
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
    )
    code_view = binding_service.create_binding_code(owner_user_id=_user_id(), now=_now())
    handler = TelegramCommandHandler(
        repository=repository,
        binding_service=binding_service,
    )
    command = _command(update_id=101, text=f"/start {code_view.code}")

    first = handler.handle(command=command)
    repeated = handler.handle(command=command)
    reuse = handler.handle(command=_command(update_id=102, text=f"/start {code_view.code}"))

    assert first.status == "handled"
    assert first.telegram_update.owner_user_id == _user_id()
    assert first.delivery is not None
    assert repeated.idempotent_replay is True
    assert repeated.delivery is None
    assert reuse.status == "handled"
    assert "already confirmed" in reuse.response_text
    assert len(repository.telegram_updates) == 2
    assert len(repository.deliveries) == 2
    assert code_view.code not in repr(binding_store.binding_codes)
    assert code_view.code not in repr(repository.telegram_updates)
    assert first.telegram_update.command_args_json == {
        "arg_count": 1,
        "arg_0": "<redacted>",
    }


def test_start_retries_after_atomic_response_failure_without_partial_state() -> None:
    repository = _FailOnceCommandResponseRepository()
    binding_store = InMemoryNotificationTelegramBindingStore()
    binding_service = NotificationTelegramBindingService(
        store=binding_store,
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
    )
    code_view = binding_service.create_binding_code(owner_user_id=_user_id(), now=_now())
    handler = TelegramCommandHandler(
        repository=repository,
        binding_service=binding_service,
    )
    command = _command(update_id=151, text=f"/start {code_view.code}")

    with pytest.raises(RuntimeError, match="simulated transaction failure"):
        handler.handle(command=command)

    assert repository.telegram_updates == {}
    assert repository.routes == {}
    assert repository.deliveries == {}
    recovered = handler.handle(command=command)
    assert recovered.status == "handled"
    assert recovered.telegram_update.owner_user_id == _user_id()
    assert recovered.delivery is not None
    assert len(repository.telegram_updates) == 1
    assert len(repository.deliveries) == 1


def test_start_after_unbound_command_uses_a_distinct_user_response_route() -> None:
    repository = InMemoryNotificationRepository()
    binding_store = InMemoryNotificationTelegramBindingStore()
    binding_service = NotificationTelegramBindingService(
        store=binding_store,
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
    )
    code_view = binding_service.create_binding_code(owner_user_id=_user_id(), now=_now())
    handler = TelegramCommandHandler(
        repository=repository,
        binding_service=binding_service,
    )

    unbound = handler.handle(command=_command(update_id=171, text="/stats today"))
    bound = handler.handle(
        command=_command(update_id=172, text=f"/start {code_view.code}")
    )

    assert unbound.status == "failed"
    assert bound.status == "handled"
    assert unbound.delivery is not None
    assert bound.delivery is not None
    assert unbound.delivery.route_id != bound.delivery.route_id


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("/stats today", "Stats for today"),
        ("/stats week", "Stats for week"),
        ("/stats month", "Stats for month"),
        ("/settings", "Telegram settings"),
        ("/critical_only", "critical_only"),
        ("/signals_on", "Signal notifications enabled"),
        ("/signals_off", "Signal notifications disabled"),
        ("/reports weekly on", "Weekly reports turned on"),
        ("/reports monthly off", "Monthly reports turned off"),
    ),
)
def test_bound_command_coverage_creates_command_response_delivery(
    text: str, expected: str
) -> None:
    repository, binding_service = _bound_repository_and_service()
    handler = TelegramCommandHandler(
        repository=repository,
        binding_service=binding_service,
    )

    result = handler.handle(command=_command(update_id=201, text=text))

    assert result.status == "handled"
    assert expected in result.response_text
    assert result.delivery is not None
    assert result.delivery.status == "pending"
    assert result.delivery.rendered_payload_json["command"] == text.split()[0].removeprefix("/")


def test_strategy_and_exchange_scopes_fail_closed_when_unauthorized() -> None:
    repository, binding_service = _bound_repository_and_service()
    handler = TelegramCommandHandler(
        repository=repository,
        binding_service=binding_service,
        scope_authorizer=_ScopeAuthorizer(
            allowed_strategy_refs=frozenset({"owned-strategy"}),
            allowed_exchange_refs=frozenset({"owned-exchange"}),
        ),
    )

    denied_strategy = handler.handle(
        command=_command(update_id=301, text="/strategy foreign-strategy week")
    )
    allowed_strategy = handler.handle(
        command=_command(update_id=302, text="/strategy owned-strategy month")
    )
    denied_exchange = handler.handle(
        command=_command(update_id=303, text="/exchange foreign-exchange today")
    )
    allowed_exchange = handler.handle(
        command=_command(update_id=304, text="/exchange owned-exchange week")
    )

    assert denied_strategy.status == "failed"
    assert "unavailable" in denied_strategy.response_text
    assert allowed_strategy.status == "handled"
    assert denied_exchange.status == "failed"
    assert "unavailable" in denied_exchange.response_text
    assert allowed_exchange.status == "handled"


def test_bound_stats_commands_render_stats_service_snapshot() -> None:
    repository, binding_service = _bound_repository_and_service()
    handler = TelegramCommandHandler(
        repository=repository,
        binding_service=binding_service,
        scope_authorizer=_ScopeAuthorizer(
            allowed_strategy_refs=frozenset({"owned-strategy"}),
            allowed_exchange_refs=frozenset({"owned-exchange"}),
        ),
        stats_query_service=NotificationStatsQueryService(
            source_reader=InMemoryNotificationStatsSourceReader(rows=_stats_rows())
        ),
    )

    portfolio = handler.handle(command=_command(update_id=351, text="/stats today"))
    strategy = handler.handle(
        command=_command(update_id=352, text="/strategy owned-strategy today")
    )
    exchange = handler.handle(
        command=_command(update_id=353, text="/exchange owned-exchange today")
    )

    assert portfolio.status == "handled"
    assert "Portfolio stats for today: complete" in portfolio.response_text
    assert "realized_pnl=12.50" in portfolio.response_text
    assert "Strategy owned-strategy stats for today: complete" in strategy.response_text
    assert "Exchange owned-exchange stats for today: complete" in exchange.response_text


def test_expired_binding_code_fails_closed() -> None:
    repository = InMemoryNotificationRepository()
    binding_service = NotificationTelegramBindingService(
        store=InMemoryNotificationTelegramBindingStore(),
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
        ttl_seconds=1,
    )
    code_view = binding_service.create_binding_code(owner_user_id=_user_id(), now=_now())
    handler = TelegramCommandHandler(
        repository=repository,
        binding_service=binding_service,
    )

    result = handler.handle(
        command=_command(
            update_id=401,
            text=f"/start {code_view.code}",
            received_at=_now() + timedelta(seconds=2),
        )
    )

    assert result.status == "failed"
    assert result.telegram_update.owner_user_id is None
    assert "invalid or expired" in result.response_text


@dataclass(frozen=True, slots=True)
class _ScopeAuthorizer(TelegramCommandScopeAuthorizer):
    allowed_strategy_refs: frozenset[str]
    allowed_exchange_refs: frozenset[str]

    def can_read_strategy(self, *, owner_user_id: UserId, strategy_ref: str) -> bool:
        _ = owner_user_id
        return strategy_ref in self.allowed_strategy_refs

    def can_read_exchange(self, *, owner_user_id: UserId, exchange_ref: str) -> bool:
        _ = owner_user_id
        return exchange_ref in self.allowed_exchange_refs


class _FailOnceCommandResponseRepository(InMemoryNotificationRepository):
    def __init__(self) -> None:
        super().__init__()
        self._fail_once = True

    def record_telegram_command_response(self, **kwargs):  # type: ignore[no-untyped-def]
        if self._fail_once:
            self._fail_once = False
            raise RuntimeError("simulated transaction failure")
        return super().record_telegram_command_response(**kwargs)


def _bound_repository_and_service() -> tuple[
    InMemoryNotificationRepository, NotificationTelegramBindingService
]:
    repository = InMemoryNotificationRepository()
    binding_service = NotificationTelegramBindingService(
        store=InMemoryNotificationTelegramBindingStore(),
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
    )
    code_view = binding_service.create_binding_code(owner_user_id=_user_id(), now=_now())
    binding_service.confirm_binding_code(
        code=code_view.code,
        chat_id_ref="telegram_ref:test:1234",
        now=_now(),
    )
    return repository, binding_service


def _command(
    *, update_id: int, text: str, received_at: datetime | None = None
) -> TelegramInboundCommand:
    return TelegramInboundCommand(
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
        telegram_update_id=update_id,
        chat_id_ref="telegram_ref:test:1234",
        chat_id=SecretValue.from_text("1234"),
        command_text=text,
        received_at=received_at or _now(),
    )


def _stats_rows() -> tuple[NotificationStatsSourceRow, ...]:
    common = {
        "owner_user_id": _user_id(),
        "observed_at": _now(),
        "strategy_ref": "owned-strategy",
        "exchange_ref": "owned-exchange",
    }
    return (
        NotificationStatsSourceRow(
            **common,
            source="strategy_signals",
            signal_count=2,
        ),
        NotificationStatsSourceRow(
            **common,
            source="strategy_paper_accounting",
            realized_pnl=Decimal("12.50"),
            unrealized_pnl=Decimal("2.25"),
            equity=Decimal("1012.50"),
            pnl_complete=True,
        ),
        NotificationStatsSourceRow(
            **common,
            source="execution_fills",
            fill_count=1,
            order_count=1,
            fee_total=Decimal("0.10"),
            funding_total=Decimal("0"),
        ),
        NotificationStatsSourceRow(
            **common,
            source="exchange_account_projection",
            balance_count=3,
            position_count=1,
        ),
    )
