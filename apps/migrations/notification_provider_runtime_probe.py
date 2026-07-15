"""Disposable PostgreSQL and controlled HTTP proof for notification providers."""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from uuid import UUID, uuid4

import psycopg

from trading.contexts.notifications.adapters import (
    PostgresNotificationProviderRepository,
    PostgresNotificationRepository,
    PostgresNotificationTelegramBindingStore,
    PostgresTelegramRecipientScopeResolver,
    PsycopgNotificationPostgresGateway,
    TelegramBotApiNotificationProvider,
    TelegramNotificationProviderConfig,
)
from trading.contexts.notifications.adapters.inbound import TelegramUpdateMapper
from trading.contexts.notifications.application import (
    NotificationDeliveryReplayService,
    NotificationDispatcher,
    NotificationDispatcherConfig,
    NotificationTelegramBindingService,
    ReplayNotificationDeliveryCommand,
    TelegramCommandHandler,
    TelegramProviderWorker,
)
from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationProviderInstance,
    NotificationRoute,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

_TELEGRAM_PACKAGE_ID = UUID("00000000-0000-4000-8000-000000000103")


class NotificationProviderRuntimeProofError(RuntimeError):
    """Raised when a disposable notification-provider proof is incomplete."""


@dataclass(frozen=True, slots=True)
class _Fixture:
    organizations: tuple[OrganizationId, OrganizationId]
    users: tuple[UserId, UserId]
    instances: tuple[NotificationProviderInstance, NotificationProviderInstance]


@dataclass(slots=True)
class _StubState:
    accepted_delivery_ids: set[str] = field(default_factory=set)
    duplicate_delivery_ids: int = 0
    seen_credentials: set[str] = field(default_factory=set)
    seen_recipients: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _Clock:
    value: datetime

    def now(self) -> datetime:
        return self.value


@dataclass(frozen=True, slots=True)
class _ControlledUpdateSource:
    updates: tuple[Mapping[str, Any], ...]

    def fetch_updates(
        self, *, offset: int, long_poll_timeout_seconds: int
    ) -> tuple[Mapping[str, Any], ...]:
        _ = long_poll_timeout_seconds
        return tuple(
            update
            for update in self.updates
            if isinstance(update.get("update_id"), int)
            and int(update["update_id"]) >= offset
        )


class _CancellationSession:
    def post(self, **kwargs: object) -> object:
        _ = kwargs
        raise asyncio.CancelledError

    def get(self, **kwargs: object) -> object:
        raise AssertionError(f"unexpected cancellation health request: {sorted(kwargs)}")


def run_probe(*, postgres_dsn: str) -> dict[str, object]:
    now = datetime.now(UTC)
    gateway = PsycopgNotificationPostgresGateway(dsn=postgres_dsn)
    provider_repository = PostgresNotificationProviderRepository(gateway=gateway)
    notification_repository = PostgresNotificationRepository(gateway=gateway)
    binding_store = PostgresNotificationTelegramBindingStore(gateway=gateway)
    state = _StubState()
    server, thread, api_base_url = _start_stub(state=state)
    credentials = ("stage11-credential-a", "stage11-credential-b")
    recipients = ("987654321012345", "876543210123456")
    mapper = TelegramUpdateMapper()
    try:
        fixture = _seed_fixture(
            dsn=postgres_dsn,
            provider_repository=provider_repository,
            now=now,
            api_base_url=api_base_url,
        )
        providers = tuple(
            _provider(
                instance=instance,
                credential=credential,
                recipient=recipient,
                api_base_url=api_base_url,
            )
            for instance, credential, recipient in zip(
                fixture.instances, credentials, recipients, strict=True
            )
        )
        _bind_recipients(
            binding_store=binding_store,
            fixture=fixture,
            recipients=recipients,
            mapper=mapper,
            now=now,
        )

        direct_delivery = _delivery(
            route=_route(
                organization_id=fixture.organizations[0],
                owner_user_id=fixture.users[0],
                instance=fixture.instances[0],
                suffix="direct",
                now=now,
            ),
            text="NORMAL",
            now=now,
        )
        first_direct = providers[0].send(delivery=direct_delivery)
        repeated_direct = providers[0].send(delivery=direct_delivery)
        if first_direct.status != "sent" or repeated_direct.status != "sent":
            raise NotificationProviderRuntimeProofError("idempotent delivery was not accepted")

        cross_scope = providers[0].send(
            delivery=replace(
                direct_delivery,
                organization_id=fixture.organizations[1],
            )
        )
        if cross_scope.status != "dead_letter":
            raise NotificationProviderRuntimeProofError("provider cross-scope send was accepted")
        _expect_cross_organization_route_rejection(
            repository=notification_repository,
            organization_id=fixture.organizations[1],
            owner_user_id=fixture.users[1],
            instance=fixture.instances[0],
            now=now,
        )
        _expect_cross_scope_provider_secret_ref_rejection(
            dsn=postgres_dsn,
            organization_id=fixture.organizations[0],
            foreign_organization_id=fixture.organizations[1],
            now=now,
        )

        rate_delivery = _persist_delivery(
            repository=notification_repository,
            route=_route(
                organization_id=fixture.organizations[0],
                owner_user_id=fixture.users[0],
                instance=fixture.instances[0],
                suffix="rate",
                now=now,
            ),
            text="RATE",
            now=now,
        )
        slow_delivery = _persist_delivery(
            repository=notification_repository,
            route=_route(
                organization_id=fixture.organizations[1],
                owner_user_id=fixture.users[1],
                instance=fixture.instances[1],
                suffix="slow",
                now=now,
            ),
            text="SLOW",
            now=now,
        )
        expired_claim = _persist_delivery(
            repository=notification_repository,
            route=_route(
                organization_id=fixture.organizations[1],
                owner_user_id=fixture.users[1],
                instance=fixture.instances[1],
                suffix="shutdown",
                now=now,
            ),
            text="MUST_NOT_SEND",
            now=now,
            status="claimed",
            lease_until=now - timedelta(seconds=1),
        )
        dispatcher = NotificationDispatcher(
            repository=notification_repository,
            providers=providers,
            clock=_Clock(now),
            config=NotificationDispatcherConfig(
                retry_backoff_seconds=4,
                max_retry_backoff_seconds=16,
                retry_jitter_ratio=0.2,
            ),
        )
        batch = dispatcher.drain_once()
        rate_state = _delivery_state(
            dsn=postgres_dsn, delivery_id=rate_delivery.delivery_id
        )
        slow_state = _delivery_state(
            dsn=postgres_dsn, delivery_id=slow_delivery.delivery_id
        )
        shutdown_state = _delivery_state(
            dsn=postgres_dsn, delivery_id=expired_claim.delivery_id
        )
        if batch.retry != 1 or batch.unknown != 2:
            raise NotificationProviderRuntimeProofError("dispatcher state matrix is incomplete")
        if rate_state[0] != "retry" or rate_state[1] != now + timedelta(seconds=2):
            raise NotificationProviderRuntimeProofError("Retry-After was not persisted")
        if slow_state[0] != "unknown":
            raise NotificationProviderRuntimeProofError("post-acceptance timeout was retried")
        if shutdown_state != ("unknown", None, "provider_shutdown"):
            raise NotificationProviderRuntimeProofError("expired claim was blindly retried")

        cancelled_delivery = _persist_delivery(
            repository=notification_repository,
            route=_route(
                organization_id=fixture.organizations[0],
                owner_user_id=fixture.users[0],
                instance=fixture.instances[0],
                suffix="cancelled",
                now=now + timedelta(seconds=1),
            ),
            text="CANCELLED",
            now=now + timedelta(seconds=1),
        )
        cancelling_provider = _provider(
            instance=fixture.instances[0],
            credential=credentials[0],
            recipient=recipients[0],
            api_base_url=api_base_url,
            session=_CancellationSession(),
        )
        try:
            NotificationDispatcher(
                repository=notification_repository,
                providers=(cancelling_provider,),
                clock=_Clock(now + timedelta(seconds=1)),
            ).drain_once()
        except asyncio.CancelledError:
            pass
        else:
            raise NotificationProviderRuntimeProofError(
                "provider cancellation was not propagated"
            )
        if _delivery_state(
            dsn=postgres_dsn,
            delivery_id=cancelled_delivery.delivery_id,
        ) != ("unknown", None, "provider_cancelled"):
            raise NotificationProviderRuntimeProofError(
                "provider cancellation left an in-memory delivery state"
            )

        pre_acceptance = _provider(
            instance=fixture.instances[0],
            credential=credentials[0],
            recipient=recipients[0],
            api_base_url="http://127.0.0.1:1",
        ).send(delivery=replace(direct_delivery, delivery_id=uuid4()))
        if pre_acceptance.status != "retry":
            raise NotificationProviderRuntimeProofError("pre-acceptance failure was not retryable")

        jitter_delivery = _persist_delivery(
            repository=notification_repository,
            route=_route(
                organization_id=fixture.organizations[0],
                owner_user_id=fixture.users[0],
                instance=fixture.instances[0],
                suffix="jitter",
                now=now + timedelta(seconds=1),
            ),
            text="CONNECT",
            now=now + timedelta(seconds=1),
        )
        NotificationDispatcher(
            repository=notification_repository,
            providers=(
                _provider(
                    instance=fixture.instances[0],
                    credential=credentials[0],
                    recipient=recipients[0],
                    api_base_url="http://127.0.0.1:1",
                ),
                providers[1],
            ),
            clock=_Clock(now + timedelta(seconds=1)),
            config=NotificationDispatcherConfig(
                retry_backoff_seconds=4,
                max_retry_backoff_seconds=16,
                retry_jitter_ratio=0.2,
            ),
        ).drain_once()
        jitter_state = _delivery_state(
            dsn=postgres_dsn, delivery_id=jitter_delivery.delivery_id
        )
        if jitter_state[0] != "retry" or jitter_state[1] is None:
            raise NotificationProviderRuntimeProofError("bounded backoff was not persisted")
        jitter_delay = (jitter_state[1] - (now + timedelta(seconds=1))).total_seconds()
        if not 3 <= jitter_delay <= 5:
            raise NotificationProviderRuntimeProofError("bounded jitter is outside its contract")

        no_fallback_delivery = _persist_delivery(
            repository=notification_repository,
            route=_route(
                organization_id=fixture.organizations[1],
                owner_user_id=fixture.users[1],
                instance=fixture.instances[1],
                suffix="no-fallback",
                now=now + timedelta(seconds=2),
            ),
            text="CRITICAL",
            now=now + timedelta(seconds=2),
        )
        NotificationDispatcher(
            repository=notification_repository,
            providers=(providers[0],),
            clock=_Clock(now + timedelta(seconds=2)),
        ).drain_once()
        if _delivery_state(
            dsn=postgres_dsn, delivery_id=no_fallback_delivery.delivery_id
        )[0] != "dead_letter":
            raise NotificationProviderRuntimeProofError("missing provider used a fallback")

        duplicate_update, durable_cursor = _verify_worker_updates(
            repository=notification_repository,
            provider_repository=provider_repository,
            binding_store=binding_store,
            scope_resolver=PostgresTelegramRecipientScopeResolver(gateway=gateway),
            mapper=mapper,
            fixture=fixture,
            recipient=recipients[0],
            now=now,
        )

        replay_delivery_id = uuid4()
        replayed = NotificationDeliveryReplayService(
            repository=notification_repository
        ).replay(
            command=ReplayNotificationDeliveryCommand(
                organization_id=fixture.organizations[1],
                original_delivery_id=slow_delivery.delivery_id,
                replay_delivery_id=replay_delivery_id,
            ),
            now=now + timedelta(seconds=3),
        )
        original_after_replay = notification_repository.get_delivery(
            organization_id=fixture.organizations[1],
            delivery_id=slow_delivery.delivery_id,
        )
        if (
            original_after_replay is None
            or original_after_replay.status != "unknown"
            or replayed.status != "pending"
            or replayed.replayed_from_delivery_id != slow_delivery.delivery_id
        ):
            raise NotificationProviderRuntimeProofError(
                "explicit replay did not preserve unknown source lineage"
            )

        ready_health = providers[0].health()
        degraded_health = _provider(
            instance=fixture.instances[1],
            credential=credentials[1],
            recipient=recipients[1],
            api_base_url="http://127.0.0.1:1",
        ).health()
        if ready_health.status != "ready" or degraded_health.status != "degraded":
            raise NotificationProviderRuntimeProofError("provider health matrix is incomplete")

        _assert_database_secret_boundary(
            dsn=postgres_dsn,
            raw_recipients=recipients,
            fixture=fixture,
        )
        if state.seen_credentials != set(credentials):
            raise NotificationProviderRuntimeProofError("provider credentials crossed instances")
        if state.seen_recipients != set(recipients):
            raise NotificationProviderRuntimeProofError("recipient resolution crossed scopes")
        if state.duplicate_delivery_ids != 1:
            raise NotificationProviderRuntimeProofError("delivery idempotency was not observed")

        command_count = _command_registry_count(
            dsn=postgres_dsn,
            instance_ids=tuple(item.instance_id for item in fixture.instances),
        )
        if command_count != 18:
            raise NotificationProviderRuntimeProofError("command registry is incomplete")

        return {
            "schema": "io.roehub.notification-provider-runtime-proof/v1alpha1",
            "two_organizations_two_instances": "passed",
            "delivery_idempotency": "passed",
            "duplicate_update": duplicate_update,
            "per_organization_secret_refs": "passed",
            "pre_acceptance_failure": "retry",
            "post_acceptance_timeout": "unknown",
            "cancellation": "unknown_persisted_before_propagation",
            "retry_after": "persisted",
            "bounded_backoff_jitter": "passed",
            "shutdown_recovery": "unknown_without_resubmit",
            "provider_health": "ready_and_degraded",
            "cross_organization_write": "rejected",
            "provider_secret_scope": "rejected",
            "critical_fallback": "not_used",
            "durable_cursor": durable_cursor,
            "telegram_command_transaction": "atomic_worker_recovery",
            "explicit_replay": "linked_new_delivery",
            "command_registry_entries": command_count,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _start_stub(
    *, state: _StubState
) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            _ = format, args

        def do_GET(self) -> None:  # noqa: N802
            credential = _credential_from_path(self.path)
            if credential is not None:
                state.seen_credentials.add(credential)
            self._json_response(200, {"ok": True, "result": {"id": 1}})

        def do_POST(self) -> None:  # noqa: N802
            credential = _credential_from_path(self.path)
            if credential is not None:
                state.seen_credentials.add(credential)
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            delivery_id = self.headers.get("X-Roehub-Delivery-Id", "")
            recipient = str(payload.get("chat_id", ""))
            text = str(payload.get("text", ""))
            state.seen_recipients.add(recipient)
            if text == "RATE":
                self._json_response(
                    429,
                    {"ok": False, "parameters": {"retry_after": 2}},
                )
                return
            duplicate = delivery_id in state.accepted_delivery_ids
            if duplicate:
                state.duplicate_delivery_ids += 1
            else:
                state.accepted_delivery_ids.add(delivery_id)
            if text == "SLOW":
                time.sleep(0.25)
            try:
                self._json_response(
                    200,
                    {"ok": True, "result": {"message_id": abs(hash(delivery_id)) % 10000}},
                )
            except BrokenPipeError:
                return

        def _json_response(self, status: int, payload: Mapping[str, object]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = str(server.server_address[0])
    port = int(server.server_address[1])
    return server, thread, f"http://{host}:{port}"


def _credential_from_path(path: str) -> str | None:
    match = re.search(r"/bot([^/]+)/", path)
    return None if match is None else match.group(1)


def _seed_fixture(
    *,
    dsn: str,
    provider_repository: PostgresNotificationProviderRepository,
    now: datetime,
    api_base_url: str,
) -> _Fixture:
    organization_values = (uuid4(), uuid4())
    user_values = (uuid4(), uuid4())
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT installation_id FROM identity_installations WHERE singleton_key = TRUE"
        )
        row = cursor.fetchone()
        if row is None:
            raise NotificationProviderRuntimeProofError("disposable installation is unavailable")
        installation_id = row[0]
        cursor.executemany(
            """
            INSERT INTO identity_users (
                user_id, telegram_user_id, paid_level, created_at,
                last_login_at, is_deleted, keycloak_subject
            ) VALUES (%s, NULL, 'free', %s, %s, FALSE, NULL)
            """,
            [(user_id, now, now) for user_id in user_values],
        )
        cursor.executemany(
            """
            INSERT INTO identity_organizations (
                organization_id, installation_id, slug, display_name, status, created_at
            ) VALUES (%s, %s, %s, %s, 'active', %s)
            """,
            tuple(
                (
                    str(organization_id),
                    installation_id,
                    f"stage11-{index}-{organization_id.hex[:8]}",
                    f"Stage 11 {index}",
                    now,
                )
                for index, organization_id in enumerate(organization_values)
            ),
        )
        cursor.executemany(
            """
            INSERT INTO identity_memberships (
                organization_id, user_id, role, status, created_at, updated_at
            ) VALUES (%s, %s, 'owner', 'active', %s, %s)
            """,
            tuple(
                (organization_id, user_id, now, now)
                for organization_id, user_id in zip(
                    organization_values, user_values, strict=True
                )
            ),
        )

    organizations = (
        OrganizationId(organization_values[0]),
        OrganizationId(organization_values[1]),
    )
    instances = tuple(
        provider_repository.add_instance(
            instance=NotificationProviderInstance(
                instance_id=instance_id,
                package_id=_TELEGRAM_PACKAGE_ID,
                provider_key="telegram_bot_api",
                scope="organization",
                organization_id=organization_id,
                display_name=f"Stage 11 Telegram {index}",
                config_json={"api_base_url": api_base_url},
                secret_ref=(
                    f"openbao://kv/roehub/telegram/providers/{organization_id}/"
                    f"{instance_id}#bot_token"
                ),
                status="active",
                created_at=now,
                updated_at=now,
            )
        )
        for index, (organization_id, instance_id) in enumerate(
            zip(organizations, (uuid4(), uuid4()), strict=True)
        )
    )
    return _Fixture(
        organizations=organizations,
        users=(UserId(user_values[0]), UserId(user_values[1])),
        instances=(instances[0], instances[1]),
    )


def _provider(
    *,
    instance: NotificationProviderInstance,
    credential: str,
    recipient: str,
    api_base_url: str,
    session: Any | None = None,
) -> TelegramBotApiNotificationProvider:
    def resolve_recipient(
        _recipient_ref: str,
        organization_id: OrganizationId,
        provider_instance_id: UUID,
    ) -> str:
        if (
            organization_id != instance.organization_id
            or provider_instance_id != instance.instance_id
        ):
            raise ValueError("recipient scope mismatch")
        return recipient

    return TelegramBotApiNotificationProvider(
        config=TelegramNotificationProviderConfig(
            instance=instance,
            api_base_url=api_base_url,
            connect_timeout_seconds=0.05,
            overall_timeout_seconds=0.1,
        ),
        credential_source=lambda: credential,
        recipient_resolver=resolve_recipient,
        session=session,
    )


def _bind_recipients(
    *,
    binding_store: PostgresNotificationTelegramBindingStore,
    fixture: _Fixture,
    recipients: tuple[str, str],
    mapper: TelegramUpdateMapper,
    now: datetime,
) -> None:
    for organization_id, user_id, instance, recipient in zip(
        fixture.organizations,
        fixture.users,
        fixture.instances,
        recipients,
        strict=True,
    ):
        raw_update = _raw_command_update(
            update_id=0,
            chat_id=int(recipient),
            text="/stats today",
        )
        chat_id_ref = mapper.chat_id_ref_from_update(update=raw_update)
        if chat_id_ref is None:
            raise NotificationProviderRuntimeProofError("controlled chat ref is unavailable")
        binding_store.confirm_chat(
            organization_id=organization_id,
            provider_instance_id=instance.instance_id,
            owner_user_id=user_id,
            chat_id_ref=chat_id_ref,
            recipient_secret_ref=(
                f"openbao://kv/roehub/telegram/recipients/{organization_id}/"
                f"{instance.instance_id}/{user_id}/{uuid4()}#chat_id"
            ),
            confirmed_at=now,
        )


def _route(
    *,
    organization_id: OrganizationId,
    owner_user_id: UserId,
    instance: NotificationProviderInstance,
    suffix: str,
    now: datetime,
) -> NotificationRoute:
    return NotificationRoute(
        route_id=uuid4(),
        organization_id=organization_id,
        provider_instance_id=instance.instance_id,
        recipient_kind="user",
        owner_user_id=owner_user_id,
        channel_key="telegram",
        provider_key="telegram_bot_api",
        mode="all",
        category_filter=(),
        scope_filter_json={"proof": "stage11"},
        schedule_json={},
        recipient_address_ref=f"telegram_ref:{suffix}:masked",
        status="active",
        created_at=now,
        updated_at=now,
    )


def _delivery(
    *,
    route: NotificationRoute,
    text: str,
    now: datetime,
    status: str = "pending",
    lease_until: datetime | None = None,
) -> NotificationDelivery:
    return NotificationDelivery(
        delivery_id=uuid4(),
        organization_id=route.organization_id,
        provider_instance_id=route.provider_instance_id,
        event_id=None,
        report_run_id=None,
        command_id=uuid4(),
        route_id=route.route_id,
        provider_key=route.provider_key,
        channel_key=route.channel_key,
        recipient_address_ref=route.recipient_address_ref,
        template_key="telegram_command_response",
        rendered_payload_json={"text": text, "category": "stats_response"},
        status=status,  # type: ignore[arg-type]
        attempt_count=0,
        lease_until=lease_until,
        created_at=now,
    )


def _persist_delivery(
    *,
    repository: PostgresNotificationRepository,
    route: NotificationRoute,
    text: str,
    now: datetime,
    status: str = "pending",
    lease_until: datetime | None = None,
) -> NotificationDelivery:
    repository.upsert_route(route=route)
    return repository.record_delivery(
        delivery=_delivery(
            route=route,
            text=text,
            now=now,
            status=status,
            lease_until=lease_until,
        )
    )


def _expect_cross_organization_route_rejection(
    *,
    repository: PostgresNotificationRepository,
    organization_id: OrganizationId,
    owner_user_id: UserId,
    instance: NotificationProviderInstance,
    now: datetime,
) -> None:
    try:
        repository.upsert_route(
            route=_route(
                organization_id=organization_id,
                owner_user_id=owner_user_id,
                instance=instance,
                suffix="cross-scope",
                now=now,
            )
        )
    except psycopg.errors.CheckViolation:
        return
    raise NotificationProviderRuntimeProofError("cross-organization route was accepted")


def _expect_cross_scope_provider_secret_ref_rejection(
    *,
    dsn: str,
    organization_id: OrganizationId,
    foreign_organization_id: OrganizationId,
    now: datetime,
) -> None:
    instance_id = uuid4()
    try:
        with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO notification_provider_instances (
                    instance_id, package_id, provider_key, scope, organization_id,
                    display_name, config_json, secret_ref, status, created_at, updated_at
                ) VALUES (
                    %s, %s, 'telegram_bot_api', 'organization', %s,
                    %s, '{}'::jsonb, %s, 'active', %s, %s
                )
                """,
                (
                    instance_id,
                    _TELEGRAM_PACKAGE_ID,
                    str(organization_id),
                    f"Rejected secret scope {instance_id}",
                    (
                        "openbao://kv/roehub/telegram/providers/"
                        f"{foreign_organization_id}/{instance_id}#bot_token"
                    ),
                    now,
                    now,
                ),
            )
    except psycopg.errors.CheckViolation:
        return
    raise NotificationProviderRuntimeProofError(
        "cross-scope provider secret reference was accepted"
    )


def _delivery_state(
    *, dsn: str, delivery_id: UUID
) -> tuple[str, datetime | None, str | None]:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT status, next_attempt_at, last_error_code
            FROM notification_deliveries
            WHERE delivery_id = %s
            """,
            (delivery_id,),
        )
        row = cursor.fetchone()
    if row is None:
        raise NotificationProviderRuntimeProofError("delivery state is unavailable")
    return str(row[0]), row[1], None if row[2] is None else str(row[2])


def _verify_worker_updates(
    *,
    repository: PostgresNotificationRepository,
    provider_repository: PostgresNotificationProviderRepository,
    binding_store: PostgresNotificationTelegramBindingStore,
    scope_resolver: PostgresTelegramRecipientScopeResolver,
    mapper: TelegramUpdateMapper,
    fixture: _Fixture,
    recipient: str,
    now: datetime,
) -> tuple[str, str]:
    instance = fixture.instances[0]
    organization_id = fixture.organizations[0]
    binding_service = NotificationTelegramBindingService(
        store=binding_store,
        organization_id=organization_id,
        provider_instance_id=instance.instance_id,
    )
    handler = TelegramCommandHandler(
        repository=repository,
        binding_service=binding_service,
    )
    updates = (
        _raw_command_update(
            update_id=501,
            chat_id=int(recipient),
            text="/stats today",
        ),
        _raw_command_update(
            update_id=502,
            chat_id=int(recipient),
            text="/settings",
        ),
    )
    first_command = mapper.command_from_update(
        organization_id=organization_id,
        provider_instance_id=instance.instance_id,
        update=updates[0],
        received_at=now,
    )
    if first_command is None:
        raise NotificationProviderRuntimeProofError("controlled command is unavailable")
    seeded = handler.handle(command=first_command)
    if seeded.delivery is None:
        raise NotificationProviderRuntimeProofError("atomic command response is unavailable")
    cursor = provider_repository.get_cursor(provider_instance_id=instance.instance_id)
    if cursor is None or cursor.last_update_id != -1:
        raise NotificationProviderRuntimeProofError("initial cursor is unavailable")

    worker = TelegramProviderWorker(
        provider_instance_id=instance.instance_id,
        organization_id=organization_id,
        provider_repository=provider_repository,
        update_source=_ControlledUpdateSource(updates=updates),
        scope_resolver=scope_resolver,
        command_handler_factory=lambda resolved_organization_id: handler,
        mapper=mapper,
        long_poll_timeout_seconds=1,
    )
    result = worker.run_once(now=now)
    repeated = worker.run_once(now=now + timedelta(seconds=1))
    if (
        result.fetched != 2
        or result.handled != 2
        or result.duplicates != 1
        or result.cursor != 502
        or repeated.fetched != 0
        or repeated.cursor != 502
    ):
        raise NotificationProviderRuntimeProofError(
            "Telegram worker did not recover duplicate update and durable cursor"
        )
    return "worker_recovered_idempotently", "advanced_by_worker"


def _raw_command_update(
    *, update_id: int, chat_id: int, text: str
) -> Mapping[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "chat": {"id": chat_id, "type": "private"},
            "text": text,
        },
    }


def _assert_database_secret_boundary(
    *,
    dsn: str,
    raw_recipients: tuple[str, str],
    fixture: _Fixture,
) -> None:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT organization_id, provider_instance_id, chat_id_ref, recipient_secret_ref
            FROM notification_telegram_recipient_bindings
            WHERE provider_instance_id = ANY(%s::uuid[])
            ORDER BY organization_id
            """,
            ([str(item.instance_id) for item in fixture.instances],),
        )
        rows = cursor.fetchall()
    if len(rows) != 2:
        raise NotificationProviderRuntimeProofError("recipient bindings are incomplete")
    serialized = json.dumps(rows, default=str)
    if any(raw in serialized for raw in raw_recipients):
        raise NotificationProviderRuntimeProofError("raw recipient leaked into PostgreSQL")
    if len({str(row[3]) for row in rows}) != 2:
        raise NotificationProviderRuntimeProofError("recipient secret refs are shared")


def _command_registry_count(*, dsn: str, instance_ids: tuple[UUID, ...]) -> int:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT count(*)
            FROM notification_telegram_command_registry
            WHERE provider_instance_id = ANY(%s::uuid[])
            """,
            ([str(item) for item in instance_ids],),
        )
        row = cursor.fetchone()
    return 0 if row is None else int(row[0])


def _bounded_error_message(error: Exception) -> str:
    message = " ".join(str(error).split())
    message = re.sub(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
        r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b",
        "[uuid]",
        message,
    )
    return message[:240] or type(error).__name__


def main() -> int:
    if os.environ.get("ROEHUB_DISPOSABLE_STORAGE_PROOF") != "1":
        print("notification provider proof failed: disposable proof guard is not enabled")
        return 1
    postgres_dsn = os.environ.get("ROEHUB_STORAGE_POSTGRES_DSN", "").strip()
    if not postgres_dsn:
        print("notification provider proof failed: PostgreSQL DSN is unavailable")
        return 1
    try:
        result = run_probe(postgres_dsn=postgres_dsn)
    except Exception as error:  # noqa: BLE001
        print(f"notification provider proof failed: {_bounded_error_message(error)}")
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
