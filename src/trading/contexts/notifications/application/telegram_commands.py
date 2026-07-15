from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TypeGuard
from uuid import NAMESPACE_URL, UUID, uuid5

from trading.contexts.notifications.application.ports import NotificationRepository
from trading.contexts.notifications.application.stats_query import (
    NotificationStatsPeriod,
    NotificationStatsQueryService,
    render_notification_stats_snapshot,
)
from trading.contexts.notifications.application.telegram_binding import (
    NotificationTelegramBindingError,
    NotificationTelegramBindingService,
)
from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationRoute,
    TelegramUpdate,
)
from trading.platform.secrets import SecretValue
from trading.shared_kernel.primitives import OrganizationId, UserId

TelegramCommandStatus = Literal["handled", "ignored", "failed"]


@dataclass(frozen=True, slots=True)
class TelegramInboundCommand:
    organization_id: OrganizationId
    provider_instance_id: UUID
    telegram_update_id: int
    chat_id_ref: str
    chat_id: SecretValue
    command_text: str
    received_at: datetime


@dataclass(frozen=True, slots=True)
class TelegramCommandHandlingResult:
    telegram_update: TelegramUpdate
    status: TelegramCommandStatus
    response_text: str
    delivery: NotificationDelivery | None
    idempotent_replay: bool = False


class TelegramCommandScopeAuthorizer(Protocol):
    def can_read_strategy(self, *, owner_user_id: UserId, strategy_ref: str) -> bool: ...

    def can_read_exchange(self, *, owner_user_id: UserId, exchange_ref: str) -> bool: ...


class DenyAllTelegramCommandScopeAuthorizer:
    def can_read_strategy(self, *, owner_user_id: UserId, strategy_ref: str) -> bool:
        _ = owner_user_id, strategy_ref
        return False

    def can_read_exchange(self, *, owner_user_id: UserId, exchange_ref: str) -> bool:
        _ = owner_user_id, exchange_ref
        return False


class TelegramCommandHandler:
    def __init__(
        self,
        *,
        repository: NotificationRepository,
        binding_service: NotificationTelegramBindingService,
        scope_authorizer: TelegramCommandScopeAuthorizer | None = None,
        stats_query_service: NotificationStatsQueryService | None = None,
    ) -> None:
        self._repository = repository
        self._binding_service = binding_service
        self._scope_authorizer = scope_authorizer or DenyAllTelegramCommandScopeAuthorizer()
        self._stats_query_service = stats_query_service

    def handle(self, *, command: TelegramInboundCommand) -> TelegramCommandHandlingResult:
        existing = self._repository.get_telegram_update(
            organization_id=command.organization_id,
            provider_instance_id=command.provider_instance_id,
            telegram_update_id=command.telegram_update_id
        )
        if existing is not None:
            return TelegramCommandHandlingResult(
                telegram_update=existing,
                status="ignored",
                response_text="Duplicate update ignored.",
                delivery=None,
                idempotent_replay=True,
            )

        parsed = _parse_command(text=command.command_text)
        owner_user_id = self._binding_service.owner_for_chat_ref(
            chat_id_ref=command.chat_id_ref
        )
        status: TelegramCommandStatus = "handled"
        response_text = ""

        if parsed.name == "start":
            status, bound_owner_user_id, response_text = self._handle_start(
                code=parsed.args[0] if parsed.args else "",
                chat_id_ref=command.chat_id_ref,
                chat_id=command.chat_id,
                existing_owner_user_id=owner_user_id,
                now=command.received_at,
            )
            owner_user_id = bound_owner_user_id or owner_user_id
        elif owner_user_id is None:
            status = "failed"
            response_text = (
                "Telegram is not bound. Open Roehub settings and send /start with a fresh code."
            )
        else:
            status, response_text = self._handle_bound_command(
                parsed=parsed,
                owner_user_id=owner_user_id,
                received_at=command.received_at,
            )

        update = TelegramUpdate(
            organization_id=command.organization_id,
            provider_instance_id=command.provider_instance_id,
            telegram_update_id=command.telegram_update_id,
            received_at=command.received_at,
            chat_id_ref=command.chat_id_ref,
            owner_user_id=owner_user_id,
            command_name=parsed.name,
            command_args_json=_command_args_json(
                command_name=parsed.name,
                args=parsed.args,
            ),
            status=status,
            idempotency_key=(
                f"telegram_update:{command.provider_instance_id}:"
                f"{command.telegram_update_id}"
            ),
            created_at=command.received_at,
            handled_at=command.received_at if status == "handled" else None,
        )
        route = _command_response_route(update=update, created_at=command.received_at)
        delivery = _command_response_delivery(
            organization_id=command.organization_id,
            provider_instance_id=command.provider_instance_id,
            update=update,
            route=route,
            response_text=response_text,
            created_at=command.received_at,
        )
        recorded, _recorded_route, recorded_delivery = (
            self._repository.record_telegram_command_response(
                update=update,
                route=route,
                delivery=delivery,
            )
        )
        return TelegramCommandHandlingResult(
            telegram_update=recorded,
            status=status,
            response_text=response_text,
            delivery=recorded_delivery,
        )

    def _handle_start(
        self,
        *,
        code: str,
        chat_id_ref: str,
        chat_id: SecretValue,
        existing_owner_user_id: UserId | None,
        now: datetime,
    ) -> tuple[TelegramCommandStatus, UserId | None, str]:
        if existing_owner_user_id is not None:
            return "handled", existing_owner_user_id, "Telegram binding already confirmed."
        if not code:
            return "failed", None, "Open Roehub settings and send /start with a fresh code."
        try:
            status = self._binding_service.confirm_binding_code(
                code=code,
                chat_id_ref=chat_id_ref,
                chat_id=chat_id,
                now=now,
            )
        except NotificationTelegramBindingError:
            return "failed", None, "Binding code is invalid or expired."
        return "handled", status.owner_user_id, "Telegram binding confirmed."

    def _handle_bound_command(
        self, *, parsed: _ParsedCommand, owner_user_id: UserId, received_at: datetime
    ) -> tuple[TelegramCommandStatus, str]:
        if parsed.name == "stats":
            period = parsed.args[0] if parsed.args else "today"
            if not _is_stats_period(value=period):
                return "failed", "Supported stats periods: today, week, month."
            if self._stats_query_service is not None:
                snapshot = self._stats_query_service.get_portfolio_stats(
                    owner_user_id=owner_user_id,
                    period=period,
                    generated_at=received_at,
                )
                return "handled", render_notification_stats_snapshot(snapshot=snapshot)
            return "handled", f"Stats for {period}: unavailable until stats service is enabled."
        if parsed.name == "strategy":
            if len(parsed.args) < 1:
                return "failed", "Usage: /strategy <id_or_name> [today|week|month]."
            if not self._scope_authorizer.can_read_strategy(
                owner_user_id=owner_user_id, strategy_ref=parsed.args[0]
            ):
                return "failed", "Strategy scope is unavailable."
            period = parsed.args[1] if len(parsed.args) > 1 else "week"
            if not _is_stats_period(value=period):
                return "failed", "Supported stats periods: today, week, month."
            if self._stats_query_service is not None:
                snapshot = self._stats_query_service.get_strategy_stats(
                    owner_user_id=owner_user_id,
                    strategy_ref=parsed.args[0],
                    period=period,
                    generated_at=received_at,
                )
                return "handled", render_notification_stats_snapshot(snapshot=snapshot)
            return (
                "handled",
                f"Strategy stats for {period}: unavailable until stats service is enabled.",
            )
        if parsed.name == "exchange":
            if len(parsed.args) < 1:
                return "failed", "Usage: /exchange <connection> [today|week|month]."
            if not self._scope_authorizer.can_read_exchange(
                owner_user_id=owner_user_id, exchange_ref=parsed.args[0]
            ):
                return "failed", "Exchange scope is unavailable."
            period = parsed.args[1] if len(parsed.args) > 1 else "week"
            if not _is_stats_period(value=period):
                return "failed", "Supported stats periods: today, week, month."
            if self._stats_query_service is not None:
                snapshot = self._stats_query_service.get_exchange_stats(
                    owner_user_id=owner_user_id,
                    exchange_ref=parsed.args[0],
                    period=period,
                    generated_at=received_at,
                )
                return "handled", render_notification_stats_snapshot(snapshot=snapshot)
            return (
                "handled",
                f"Exchange stats for {period}: unavailable until stats service is enabled.",
            )
        if parsed.name == "settings":
            return (
                "handled",
                "Telegram settings: critical_only, signals, and reports toggles available.",
            )
        if parsed.name == "critical_only":
            return "handled", "Telegram mode set to critical_only."
        if parsed.name in {"signals_on", "signals_off"}:
            enabled = "enabled" if parsed.name == "signals_on" else "disabled"
            return "handled", f"Signal notifications {enabled}."
        if parsed.name == "reports":
            valid_schedule = len(parsed.args) == 2 and parsed.args[0] in {
                "weekly",
                "monthly",
            }
            valid_mode = len(parsed.args) == 2 and parsed.args[1] in {"on", "off"}
            if not valid_schedule or not valid_mode:
                return "failed", "Usage: /reports weekly|monthly on|off."
            return "handled", f"{parsed.args[0].title()} reports turned {parsed.args[1]}."
        return "failed", "Unknown command. Supported: /stats, /strategy, /exchange, /settings."


@dataclass(frozen=True, slots=True)
class _ParsedCommand:
    name: str
    args: tuple[str, ...]


def _parse_command(*, text: str) -> _ParsedCommand:
    tokens = tuple(part for part in text.strip().split() if part)
    if not tokens:
        return _ParsedCommand(name="unknown", args=())
    command_name = tokens[0].removeprefix("/").split("@", maxsplit=1)[0].casefold()
    return _ParsedCommand(name=command_name, args=tuple(token.strip() for token in tokens[1:]))


def _is_stats_period(*, value: str) -> TypeGuard[NotificationStatsPeriod]:
    return value in {"today", "week", "month"}


def _command_args_json(
    *, command_name: str, args: tuple[str, ...]
) -> dict[str, object]:
    values: dict[str, object] = {"arg_count": len(args)}
    if command_name == "start":
        if args:
            values["arg_0"] = "<redacted>"
        return values
    for index, arg in enumerate(args[:8]):
        values[f"arg_{index}"] = arg
    return values


def _command_response_delivery(
    *,
    organization_id: OrganizationId,
    provider_instance_id: UUID,
    update: TelegramUpdate,
    route: NotificationRoute,
    response_text: str,
    created_at: datetime,
) -> NotificationDelivery:
    return NotificationDelivery(
        delivery_id=uuid5(
            NAMESPACE_URL,
            (
                "roehub:notifications:telegram-command-delivery:"
                f"{organization_id}:{provider_instance_id}:{update.telegram_update_id}"
            ),
        ),
        organization_id=organization_id,
        provider_instance_id=provider_instance_id,
        event_id=None,
        report_run_id=None,
        command_id=UUID(int=update.telegram_update_id),
        route_id=route.route_id,
        provider_key="telegram_bot_api",
        channel_key="telegram",
        recipient_address_ref=update.chat_id_ref,
        template_key="telegram_command_response",
        rendered_payload_json={
            "text": response_text,
            "command": update.command_name,
            "telegram_update_id": update.telegram_update_id,
        },
        status="pending",
        attempt_count=0,
        created_at=created_at,
    )


def _command_response_route(
    *, update: TelegramUpdate, created_at: datetime
) -> NotificationRoute:
    recipient_identity = (
        f"user:{update.owner_user_id}"
        if update.owner_user_id is not None
        else "admin:unbound"
    )
    route_id = uuid5(
        NAMESPACE_URL,
        (
            "roehub:notifications:telegram-command-response:"
            f"{update.organization_id}:{update.provider_instance_id}:"
            f"{update.chat_id_ref}:{recipient_identity}"
        ),
    )
    return NotificationRoute(
        route_id=route_id,
        organization_id=update.organization_id,
        provider_instance_id=update.provider_instance_id,
        recipient_kind="user" if update.owner_user_id is not None else "admin",
        owner_user_id=update.owner_user_id,
        channel_key="telegram",
        provider_key="telegram_bot_api",
        mode="all",
        category_filter=(),
        scope_filter_json={"source": "telegram_command"},
        schedule_json={},
        recipient_address_ref=update.chat_id_ref,
        status="active",
        created_at=created_at,
        updated_at=created_at,
    )
