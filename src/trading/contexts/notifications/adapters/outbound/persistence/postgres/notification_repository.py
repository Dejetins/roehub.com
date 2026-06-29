from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

from psycopg.types.json import Jsonb

from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationDeliveryAttempt,
    NotificationEvent,
    NotificationReportRun,
    NotificationRoute,
    TelegramUpdate,
)
from trading.shared_kernel.primitives import UserId

from .gateway import NotificationPostgresGateway


class PostgresNotificationRepository:
    def __init__(self, *, gateway: NotificationPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresNotificationRepository requires gateway")
        self._gateway = gateway

    def record_event(self, *, event: NotificationEvent) -> NotificationEvent:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO notification_events
              (event_id, owner_user_id, recipient_kind, source_context, source_event_type,
               category, severity, scope_json, payload_json, dedupe_key, occurred_at, created_at)
            VALUES
              (%(event_id)s, %(owner_user_id)s, %(recipient_kind)s, %(source_context)s,
               %(source_event_type)s, %(category)s, %(severity)s, %(scope_json)s,
               %(payload_json)s, %(dedupe_key)s, %(occurred_at)s, %(created_at)s)
            ON CONFLICT (dedupe_key) DO UPDATE SET dedupe_key = EXCLUDED.dedupe_key
            RETURNING event_id, owner_user_id, recipient_kind, source_context,
                      source_event_type, category, severity, scope_json, payload_json,
                      dedupe_key, occurred_at, created_at
            """,
            parameters={
                "event_id": str(event.event_id),
                "owner_user_id": str(event.owner_user_id) if event.owner_user_id else None,
                "recipient_kind": event.recipient_kind,
                "source_context": event.source_context,
                "source_event_type": event.source_event_type,
                "category": event.category,
                "severity": event.severity,
                "scope_json": Jsonb(dict(event.scope_json)),
                "payload_json": Jsonb(dict(event.payload_json)),
                "dedupe_key": event.dedupe_key,
                "occurred_at": event.occurred_at,
                "created_at": event.created_at,
            },
        )
        return _map_event_row(row=_require_row(row, "notification_events insert"))

    def get_event_by_dedupe_key(self, *, dedupe_key: str) -> NotificationEvent | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT event_id, owner_user_id, recipient_kind, source_context, source_event_type,
                   category, severity, scope_json, payload_json, dedupe_key, occurred_at,
                   created_at
            FROM notification_events
            WHERE dedupe_key = %(dedupe_key)s
            """,
            parameters={"dedupe_key": dedupe_key},
        )
        if row is None:
            return None
        return _map_event_row(row=row)

    def upsert_route(self, *, route: NotificationRoute) -> NotificationRoute:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO notification_routes
              (route_id, recipient_kind, owner_user_id, channel_key, provider_key, mode,
               category_filter, scope_filter_json, schedule_json, recipient_address_ref,
               status, created_at, updated_at)
            VALUES
              (%(route_id)s, %(recipient_kind)s, %(owner_user_id)s, %(channel_key)s,
               %(provider_key)s, %(mode)s, %(category_filter)s, %(scope_filter_json)s,
               %(schedule_json)s, %(recipient_address_ref)s, %(status)s, %(created_at)s,
               %(updated_at)s)
            ON CONFLICT (route_id) DO UPDATE SET
              recipient_kind = EXCLUDED.recipient_kind,
              owner_user_id = EXCLUDED.owner_user_id,
              channel_key = EXCLUDED.channel_key,
              provider_key = EXCLUDED.provider_key,
              mode = EXCLUDED.mode,
              category_filter = EXCLUDED.category_filter,
              scope_filter_json = EXCLUDED.scope_filter_json,
              schedule_json = EXCLUDED.schedule_json,
              recipient_address_ref = EXCLUDED.recipient_address_ref,
              status = EXCLUDED.status,
              updated_at = EXCLUDED.updated_at
            RETURNING route_id, recipient_kind, owner_user_id, channel_key, provider_key,
                      mode, category_filter, scope_filter_json, schedule_json,
                      recipient_address_ref, status, created_at, updated_at
            """,
            parameters=_route_parameters(route=route),
        )
        return _map_route_row(row=_require_row(row, "notification_routes upsert"))

    def get_route(self, *, route_id: UUID) -> NotificationRoute | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT route_id, recipient_kind, owner_user_id, channel_key, provider_key,
                   mode, category_filter, scope_filter_json, schedule_json,
                   recipient_address_ref, status, created_at, updated_at
            FROM notification_routes
            WHERE route_id = %(route_id)s
            """,
            parameters={"route_id": str(route_id)},
        )
        if row is None:
            return None
        return _map_route_row(row=row)

    def list_active_routes(
        self, *, owner_user_id: UserId | None, recipient_kind: str, category: str
    ) -> tuple[NotificationRoute, ...]:
        rows = self._gateway.fetch_all(
            query="""
            SELECT route_id, recipient_kind, owner_user_id, channel_key, provider_key,
                   mode, category_filter, scope_filter_json, schedule_json,
                   recipient_address_ref, status, created_at, updated_at
            FROM notification_routes
            WHERE status = 'active'
              AND recipient_kind = %(recipient_kind)s
              AND (
                    (%(owner_user_id)s IS NULL AND owner_user_id IS NULL)
                    OR owner_user_id = %(owner_user_id)s
                  )
              AND (cardinality(category_filter) = 0 OR %(category)s = ANY(category_filter))
            ORDER BY updated_at DESC, route_id ASC
            """,
            parameters={
                "owner_user_id": str(owner_user_id) if owner_user_id else None,
                "recipient_kind": recipient_kind,
                "category": category,
            },
        )
        return tuple(_map_route_row(row=row) for row in rows)

    def list_active_report_routes(self) -> tuple[NotificationRoute, ...]:
        rows = self._gateway.fetch_all(
            query="""
            SELECT route_id, recipient_kind, owner_user_id, channel_key, provider_key,
                   mode, category_filter, scope_filter_json, schedule_json,
                   recipient_address_ref, status, created_at, updated_at
            FROM notification_routes
            WHERE status = 'active'
              AND recipient_kind = 'user'
              AND owner_user_id IS NOT NULL
              AND mode IN ('reports', 'all')
              AND (
                    cardinality(category_filter) = 0
                    OR 'portfolio_report' = ANY(category_filter)
                  )
            ORDER BY updated_at DESC, route_id ASC
            """,
            parameters={},
        )
        return tuple(_map_route_row(row=row) for row in rows)

    def record_delivery(self, *, delivery: NotificationDelivery) -> NotificationDelivery:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO notification_deliveries
              (delivery_id, event_id, report_run_id, command_id, route_id, provider_key,
               channel_key, recipient_address_ref, template_key, rendered_payload_json,
               status, attempt_count, next_attempt_at, lease_until, last_error_code,
               provider_message_id, created_at, sent_at)
            VALUES
              (%(delivery_id)s, %(event_id)s, %(report_run_id)s, %(command_id)s,
               %(route_id)s, %(provider_key)s, %(channel_key)s, %(recipient_address_ref)s,
               %(template_key)s, %(rendered_payload_json)s, %(status)s, %(attempt_count)s,
               %(next_attempt_at)s, %(lease_until)s, %(last_error_code)s,
               %(provider_message_id)s, %(created_at)s, %(sent_at)s)
            RETURNING delivery_id, event_id, report_run_id, command_id, route_id,
                      provider_key, channel_key, recipient_address_ref, template_key,
                      rendered_payload_json, status, attempt_count, next_attempt_at,
                      lease_until, last_error_code, provider_message_id, created_at, sent_at
            """,
            parameters=_delivery_parameters(delivery=delivery),
        )
        return _map_delivery_row(row=_require_row(row, "notification_deliveries insert"))

    def list_due_deliveries(
        self, *, now: datetime, limit: int
    ) -> tuple[NotificationDelivery, ...]:
        rows = self._gateway.fetch_all(
            query="""
            SELECT delivery_id, event_id, report_run_id, command_id, route_id,
                   provider_key, channel_key, recipient_address_ref, template_key,
                   rendered_payload_json, status, attempt_count, next_attempt_at,
                   lease_until, last_error_code, provider_message_id, created_at, sent_at
            FROM notification_deliveries
            WHERE (
                    status IN ('pending', 'retry')
                    AND (next_attempt_at IS NULL OR next_attempt_at <= %(now)s)
                  )
               OR (
                    status = 'claimed'
                    AND lease_until IS NOT NULL
                    AND lease_until <= %(now)s
                  )
            ORDER BY created_at ASC, delivery_id ASC
            LIMIT %(limit)s
            """,
            parameters={"now": now, "limit": limit},
        )
        return tuple(_map_delivery_row(row=row) for row in rows)

    def update_delivery(self, *, delivery: NotificationDelivery) -> NotificationDelivery:
        row = self._gateway.fetch_one(
            query="""
            UPDATE notification_deliveries SET
              status = %(status)s,
              attempt_count = %(attempt_count)s,
              next_attempt_at = %(next_attempt_at)s,
              lease_until = %(lease_until)s,
              last_error_code = %(last_error_code)s,
              provider_message_id = %(provider_message_id)s,
              sent_at = %(sent_at)s
            WHERE delivery_id = %(delivery_id)s
            RETURNING delivery_id, event_id, report_run_id, command_id, route_id,
                      provider_key, channel_key, recipient_address_ref, template_key,
                      rendered_payload_json, status, attempt_count, next_attempt_at,
                      lease_until, last_error_code, provider_message_id, created_at, sent_at
            """,
            parameters=_delivery_parameters(delivery=delivery),
        )
        return _map_delivery_row(row=_require_row(row, "notification_deliveries update"))

    def count_deliveries_by_status(self, *, status: str) -> int:
        row = self._gateway.fetch_one(
            query="SELECT count(*) AS count FROM notification_deliveries WHERE status = %(status)s",
            parameters={"status": status},
        )
        if row is None:
            return 0
        return int(row["count"])

    def claim_delivery(
        self, *, delivery_id: UUID, lease_until: datetime, now: datetime
    ) -> NotificationDelivery | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE notification_deliveries SET
              status = 'claimed',
              attempt_count = attempt_count + 1,
              lease_until = %(lease_until)s
            WHERE delivery_id = %(delivery_id)s
              AND (
                    (
                      status = 'claimed'
                      AND lease_until IS NOT NULL
                      AND lease_until <= %(now)s
                    )
                    OR (
                      status IN ('pending', 'retry')
                      AND (next_attempt_at IS NULL OR next_attempt_at <= %(now)s)
                    )
                  )
            RETURNING delivery_id, event_id, report_run_id, command_id, route_id,
                      provider_key, channel_key, recipient_address_ref, template_key,
                      rendered_payload_json, status, attempt_count, next_attempt_at,
                      lease_until, last_error_code, provider_message_id, created_at, sent_at
            """,
            parameters={
                "delivery_id": str(delivery_id),
                "lease_until": lease_until,
                "now": now,
            },
        )
        if row is None:
            return None
        return _map_delivery_row(row=row)

    def record_delivery_attempt(
        self, *, attempt: NotificationDeliveryAttempt
    ) -> NotificationDeliveryAttempt:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO notification_delivery_attempts
              (attempt_id, delivery_id, provider_key, started_at, finished_at, status,
               http_status, error_code, retry_after_seconds, redacted_request_hash,
               redacted_response_hash)
            VALUES
              (%(attempt_id)s, %(delivery_id)s, %(provider_key)s, %(started_at)s,
               %(finished_at)s, %(status)s, %(http_status)s, %(error_code)s,
               %(retry_after_seconds)s, %(redacted_request_hash)s,
               %(redacted_response_hash)s)
            RETURNING attempt_id, delivery_id, provider_key, started_at, finished_at,
                      status, http_status, error_code, retry_after_seconds,
                      redacted_request_hash, redacted_response_hash
            """,
            parameters={
                "attempt_id": str(attempt.attempt_id),
                "delivery_id": str(attempt.delivery_id),
                "provider_key": attempt.provider_key,
                "started_at": attempt.started_at,
                "finished_at": attempt.finished_at,
                "status": attempt.status,
                "http_status": attempt.http_status,
                "error_code": attempt.error_code,
                "retry_after_seconds": attempt.retry_after_seconds,
                "redacted_request_hash": attempt.redacted_request_hash,
                "redacted_response_hash": attempt.redacted_response_hash,
            },
        )
        return _map_attempt_row(row=_require_row(row, "notification_delivery_attempts insert"))

    def record_telegram_update(self, *, update: TelegramUpdate) -> TelegramUpdate:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO notification_telegram_updates
              (telegram_update_id, received_at, chat_id_ref, owner_user_id, command_name,
               command_args_json, status, idempotency_key, created_at, handled_at)
            VALUES
              (%(telegram_update_id)s, %(received_at)s, %(chat_id_ref)s,
               %(owner_user_id)s, %(command_name)s, %(command_args_json)s, %(status)s,
               %(idempotency_key)s, %(created_at)s, %(handled_at)s)
            ON CONFLICT (telegram_update_id) DO UPDATE SET
              telegram_update_id = EXCLUDED.telegram_update_id
            RETURNING telegram_update_id, received_at, chat_id_ref, owner_user_id,
                      command_name, command_args_json, status, idempotency_key,
                      created_at, handled_at
            """,
            parameters={
                "telegram_update_id": update.telegram_update_id,
                "received_at": update.received_at,
                "chat_id_ref": update.chat_id_ref,
                "owner_user_id": str(update.owner_user_id) if update.owner_user_id else None,
                "command_name": update.command_name,
                "command_args_json": Jsonb(dict(update.command_args_json)),
                "status": update.status,
                "idempotency_key": update.idempotency_key,
                "created_at": update.created_at,
                "handled_at": update.handled_at,
            },
        )
        return _map_telegram_update_row(
            row=_require_row(row, "notification_telegram_updates insert")
        )

    def get_telegram_update(self, *, telegram_update_id: int) -> TelegramUpdate | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT telegram_update_id, received_at, chat_id_ref, owner_user_id,
                   command_name, command_args_json, status, idempotency_key,
                   created_at, handled_at
            FROM notification_telegram_updates
            WHERE telegram_update_id = %(telegram_update_id)s
            """,
            parameters={"telegram_update_id": telegram_update_id},
        )
        if row is None:
            return None
        return _map_telegram_update_row(row=row)

    def record_report_run(self, *, report_run: NotificationReportRun) -> NotificationReportRun:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO notification_report_runs
              (report_run_id, owner_user_id, report_type, period_start, period_end,
               scope_json, quality_status, status, dedupe_key, created_at, rendered_at,
               finished_at)
            VALUES
              (%(report_run_id)s, %(owner_user_id)s, %(report_type)s,
               %(period_start)s, %(period_end)s, %(scope_json)s, %(quality_status)s,
               %(status)s, %(dedupe_key)s, %(created_at)s, %(rendered_at)s,
               %(finished_at)s)
            ON CONFLICT (dedupe_key) DO UPDATE SET dedupe_key = EXCLUDED.dedupe_key
            RETURNING report_run_id, owner_user_id, report_type, period_start,
                      period_end, scope_json, quality_status, status, dedupe_key,
                      created_at, rendered_at, finished_at
            """,
            parameters={
                "report_run_id": str(report_run.report_run_id),
                "owner_user_id": str(report_run.owner_user_id),
                "report_type": report_run.report_type,
                "period_start": report_run.period_start,
                "period_end": report_run.period_end,
                "scope_json": Jsonb(dict(report_run.scope_json)),
                "quality_status": report_run.quality_status,
                "status": report_run.status,
                "dedupe_key": report_run.dedupe_key,
                "created_at": report_run.created_at,
                "rendered_at": report_run.rendered_at,
                "finished_at": report_run.finished_at,
            },
        )
        return _map_report_run_row(
            row=_require_row(row, "notification_report_runs insert")
        )

    def get_report_run_by_dedupe_key(
        self, *, dedupe_key: str
    ) -> NotificationReportRun | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT report_run_id, owner_user_id, report_type, period_start, period_end,
                   scope_json, quality_status, status, dedupe_key, created_at,
                   rendered_at, finished_at
            FROM notification_report_runs
            WHERE dedupe_key = %(dedupe_key)s
            """,
            parameters={"dedupe_key": dedupe_key},
        )
        if row is None:
            return None
        return _map_report_run_row(row=row)


def _route_parameters(*, route: NotificationRoute) -> dict[str, object]:
    return {
        "route_id": str(route.route_id),
        "recipient_kind": route.recipient_kind,
        "owner_user_id": str(route.owner_user_id) if route.owner_user_id else None,
        "channel_key": route.channel_key,
        "provider_key": route.provider_key,
        "mode": route.mode,
        "category_filter": list(route.category_filter),
        "scope_filter_json": Jsonb(dict(route.scope_filter_json)),
        "schedule_json": Jsonb(dict(route.schedule_json)),
        "recipient_address_ref": route.recipient_address_ref,
        "status": route.status,
        "created_at": route.created_at,
        "updated_at": route.updated_at,
    }


def _delivery_parameters(*, delivery: NotificationDelivery) -> dict[str, object]:
    return {
        "delivery_id": str(delivery.delivery_id),
        "event_id": str(delivery.event_id) if delivery.event_id else None,
        "report_run_id": str(delivery.report_run_id) if delivery.report_run_id else None,
        "command_id": str(delivery.command_id) if delivery.command_id else None,
        "route_id": str(delivery.route_id),
        "provider_key": delivery.provider_key,
        "channel_key": delivery.channel_key,
        "recipient_address_ref": delivery.recipient_address_ref,
        "template_key": delivery.template_key,
        "rendered_payload_json": Jsonb(dict(delivery.rendered_payload_json)),
        "status": delivery.status,
        "attempt_count": delivery.attempt_count,
        "next_attempt_at": delivery.next_attempt_at,
        "lease_until": delivery.lease_until,
        "last_error_code": delivery.last_error_code,
        "provider_message_id": delivery.provider_message_id,
        "created_at": delivery.created_at,
        "sent_at": delivery.sent_at,
    }


def _map_event_row(*, row: Mapping[str, Any]) -> NotificationEvent:
    return NotificationEvent(
        event_id=UUID(str(row["event_id"])),
        owner_user_id=_optional_user_id(row["owner_user_id"]),
        recipient_kind=row["recipient_kind"],
        source_context=row["source_context"],
        source_event_type=str(row["source_event_type"]),
        category=row["category"],
        severity=row["severity"],
        scope_json=_json_mapping(row["scope_json"]),
        payload_json=_json_mapping(row["payload_json"]),
        dedupe_key=str(row["dedupe_key"]),
        occurred_at=row["occurred_at"],
        created_at=row["created_at"],
    )


def _map_route_row(*, row: Mapping[str, Any]) -> NotificationRoute:
    return NotificationRoute(
        route_id=UUID(str(row["route_id"])),
        recipient_kind=row["recipient_kind"],
        owner_user_id=_optional_user_id(row["owner_user_id"]),
        channel_key=row["channel_key"],
        provider_key=row["provider_key"],
        mode=row["mode"],
        category_filter=_string_tuple(row["category_filter"]),
        scope_filter_json=_json_mapping(row["scope_filter_json"]),
        schedule_json=_json_mapping(row["schedule_json"]),
        recipient_address_ref=str(row["recipient_address_ref"]),
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _map_delivery_row(*, row: Mapping[str, Any]) -> NotificationDelivery:
    return NotificationDelivery(
        delivery_id=UUID(str(row["delivery_id"])),
        event_id=_optional_uuid(row["event_id"]),
        report_run_id=_optional_uuid(row["report_run_id"]),
        command_id=_optional_uuid(row["command_id"]),
        route_id=UUID(str(row["route_id"])),
        provider_key=row["provider_key"],
        channel_key=row["channel_key"],
        recipient_address_ref=str(row["recipient_address_ref"]),
        template_key=str(row["template_key"]),
        rendered_payload_json=_json_mapping(row["rendered_payload_json"]),
        status=row["status"],
        attempt_count=int(row["attempt_count"]),
        next_attempt_at=row["next_attempt_at"],
        lease_until=row["lease_until"],
        last_error_code=row["last_error_code"],
        provider_message_id=row["provider_message_id"],
        created_at=row["created_at"],
        sent_at=row["sent_at"],
    )


def _map_attempt_row(*, row: Mapping[str, Any]) -> NotificationDeliveryAttempt:
    return NotificationDeliveryAttempt(
        attempt_id=UUID(str(row["attempt_id"])),
        delivery_id=UUID(str(row["delivery_id"])),
        provider_key=row["provider_key"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        status=row["status"],
        http_status=row["http_status"],
        error_code=row["error_code"],
        retry_after_seconds=row["retry_after_seconds"],
        redacted_request_hash=row["redacted_request_hash"],
        redacted_response_hash=row["redacted_response_hash"],
    )


def _map_telegram_update_row(*, row: Mapping[str, Any]) -> TelegramUpdate:
    return TelegramUpdate(
        telegram_update_id=int(row["telegram_update_id"]),
        received_at=row["received_at"],
        chat_id_ref=str(row["chat_id_ref"]),
        owner_user_id=_optional_user_id(row["owner_user_id"]),
        command_name=row["command_name"],
        command_args_json=_json_mapping(row["command_args_json"]),
        status=row["status"],
        idempotency_key=str(row["idempotency_key"]),
        created_at=row["created_at"],
        handled_at=row["handled_at"],
    )


def _map_report_run_row(*, row: Mapping[str, Any]) -> NotificationReportRun:
    return NotificationReportRun(
        report_run_id=UUID(str(row["report_run_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        report_type=row["report_type"],
        period_start=row["period_start"],
        period_end=row["period_end"],
        scope_json=_json_mapping(row["scope_json"]),
        quality_status=row["quality_status"],
        status=row["status"],
        dedupe_key=str(row["dedupe_key"]),
        created_at=row["created_at"],
        rendered_at=row["rendered_at"],
        finished_at=row["finished_at"],
    )


def _optional_user_id(value: object) -> UserId | None:
    if value is None:
        return None
    return UserId.from_string(str(value))


def _optional_uuid(value: object) -> UUID | None:
    if value is None:
        return None
    return UUID(str(value))


def _json_mapping(value: object) -> Mapping[str, object]:
    parsed: object
    if isinstance(value, Mapping):
        parsed = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        parsed = json.loads(bytes(value).decode("utf-8"))
    elif isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, Mapping):
        raise ValueError("notification JSON field must be an object")
    return dict(parsed)


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("notification category_filter must be an array")
    return tuple(str(item) for item in value)


def _require_row(row: Mapping[str, Any] | None, operation: str) -> Mapping[str, Any]:
    if row is None:
        raise ValueError(f"{operation} returned no row")
    return row
