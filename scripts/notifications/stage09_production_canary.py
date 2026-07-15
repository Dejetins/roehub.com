from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping
from uuid import NAMESPACE_URL, UUID, uuid5

from prometheus_client import generate_latest

from apps.worker.notification_dispatcher.wiring.modules.notification_dispatcher import (
    NotificationDispatcherPrometheusMetrics,
    build_notification_dispatcher,
    load_notification_dispatcher_runtime_config,
    openbao_service_input_presence,
    postgres_dsn_presence,
    resolve_notification_postgres_dsn,
)
from trading.contexts.notifications.adapters import (
    LogOnlyNotificationProvider,
    PostgresNotificationRepository,
    PsycopgNotificationPostgresGateway,
)
from trading.contexts.notifications.application import (
    NotificationSourceRouter,
    synthetic_notification_matrix,
)
from trading.contexts.notifications.domain import NotificationRoute
from trading.shared_kernel.primitives import OrganizationId, UserId

_LOG_PROVIDER_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000001")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    env = os.environ
    organization_id = OrganizationId(args.organization_id)
    owner_user_id = UserId(args.owner_user_id)
    if args.mode == "real-readiness":
        if args.provider_instance_id is None:
            raise SystemExit("--provider-instance-id is required for real-readiness")
        result = run_real_telegram_readiness(
            config_path=config_path,
            environ=env,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            provider_instance_id=args.provider_instance_id,
        )
    else:
        result = run_log_only_matrix(
            config_path=config_path,
            environ=env,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            run_id=args.run_id or _default_run_id(),
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"ok", "blocked"} else 1


def run_log_only_matrix(
    *,
    config_path: Path,
    environ: Mapping[str, str],
    organization_id: OrganizationId,
    owner_user_id: UserId,
    run_id: str,
) -> dict[str, object]:
    runtime_config = load_notification_dispatcher_runtime_config(config_path=config_path)
    gateway = PsycopgNotificationPostgresGateway(
        dsn=resolve_notification_postgres_dsn(environ=environ)
    )
    repository = PostgresNotificationRepository(gateway=gateway)
    now = datetime.now(UTC)
    routes = _stage09_log_only_routes(
        organization_id=organization_id,
        owner_user_id=owner_user_id,
        run_id=run_id,
        now=now,
    )
    router = NotificationSourceRouter()
    categories: set[str] = set()
    delivery_ids: list[str] = []
    event_rows = 0

    for route in routes:
        repository.upsert_route(route=route)

    for fact in synthetic_notification_matrix(
        organization_id=organization_id,
        owner_user_id=owner_user_id,
        now=now,
    ):
        unique_fact = replace(fact, fact_id=f"stage09:{run_id}:{fact.fact_id}")
        flow = router.route(fact=unique_fact, routes=routes, now=now)
        repository.record_event(event=flow.event)
        event_rows += 1
        categories.add(flow.event.category)
        for delivery in flow.deliveries:
            stored = repository.record_delivery(delivery=delivery)
            delivery_ids.append(str(stored.delivery_id))

    metrics = NotificationDispatcherPrometheusMetrics()
    dispatcher = build_notification_dispatcher(
        repository=repository,
        runtime_config=runtime_config,
        providers=(LogOnlyNotificationProvider(),),
        metrics=metrics,
    )
    dispatch_result = dispatcher.drain_once()
    metrics_payload = generate_latest(metrics.registry).decode("utf-8")
    status_counts = {
        status: repository.count_deliveries_by_status(status=status)
        for status in ("pending", "retry", "sent", "unknown", "dead_letter")
    }
    return {
        "status": "ok" if dispatch_result.sent == len(delivery_ids) else "failed",
        "proof": "stage09_log_only_matrix",
        "run_id": run_id,
        "provider_mode": runtime_config.provider_mode,
        "event_rows": event_rows,
        "category_count": len(categories),
        "deliveries": len(delivery_ids),
        "claimed": dispatch_result.claimed,
        "sent": dispatch_result.sent,
        "unknown": dispatch_result.unknown,
        "dead_letter": dispatch_result.dead_letter,
        "categories": sorted(categories),
        "status_counts": status_counts,
        "metric_names_present": all(
            name in metrics_payload
            for name in (
                "notification_dispatcher_deliveries_claimed_total",
                "notification_dispatcher_delivery_results_total",
                "notification_dispatcher_pending_age_seconds",
                "notification_dispatcher_unknown_deliveries",
            )
        ),
    }


def run_real_telegram_readiness(
    *,
    config_path: Path,
    environ: Mapping[str, str],
    organization_id: OrganizationId,
    owner_user_id: UserId,
    provider_instance_id: UUID,
) -> dict[str, object]:
    _ = load_notification_dispatcher_runtime_config(config_path=config_path)
    gateway = PsycopgNotificationPostgresGateway(
        dsn=resolve_notification_postgres_dsn(environ=environ)
    )
    rows = gateway.fetch_all(
        query="""
        SELECT
          EXISTS (
            SELECT 1 FROM notification_provider_instances
            WHERE instance_id = %(provider_instance_id)s
              AND provider_key = 'telegram_bot_api'
              AND status IN ('active', 'degraded')
              AND (organization_id IS NULL OR organization_id = %(organization_id)s)
          ) AS instance_ready,
          EXISTS (
            SELECT 1 FROM notification_telegram_recipient_bindings
            WHERE organization_id = %(organization_id)s
              AND provider_instance_id = %(provider_instance_id)s
              AND owner_user_id = %(owner_user_id)s
              AND status = 'confirmed'
          ) AS recipient_ready
        """,
        parameters={
            "organization_id": str(organization_id),
            "owner_user_id": str(owner_user_id),
            "provider_instance_id": str(provider_instance_id),
        },
    )
    row = rows[0] if rows else {}
    instance_ready = bool(row.get("instance_ready"))
    recipient_ready = bool(row.get("recipient_ready"))
    openbao_presence = openbao_service_input_presence(environ=environ)
    dsn_presence = postgres_dsn_presence(environ=environ)
    ready = (
        instance_ready
        and recipient_ready
        and all(openbao_presence.values())
        and any(dsn_presence.values())
    )
    return {
        "status": "ok" if ready else "blocked",
        "proof": "stage09_real_telegram_readiness",
        "provider_instance_ready": instance_ready,
        "recipient_binding_ready": recipient_ready,
        "openbao_service_inputs_present": all(openbao_presence.values()),
        "postgres_dsn_present": any(dsn_presence.values()),
        "user_confirmation_required": True,
        "blocker": None
        if ready
        else "missing_provider_instance_recipient_binding_or_openbao_input",
    }


def _stage09_log_only_routes(
    *,
    organization_id: OrganizationId,
    owner_user_id: UserId,
    run_id: str,
    now: datetime,
) -> tuple[NotificationRoute, ...]:
    user_route_id = uuid5(NAMESPACE_URL, f"roehub:notifications:stage09:{run_id}:user")
    admin_route_id = uuid5(NAMESPACE_URL, f"roehub:notifications:stage09:{run_id}:admin")
    return (
        NotificationRoute(
            route_id=user_route_id,
            organization_id=organization_id,
            provider_instance_id=_LOG_PROVIDER_INSTANCE_ID,
            recipient_kind="user",
            owner_user_id=owner_user_id,
            channel_key="telegram",
            provider_key="log_only",
            mode="all",
            category_filter=(),
            scope_filter_json={"stage": "09", "run": run_id},
            schedule_json={},
            recipient_address_ref=f"telegram_ref:stage09:user:{run_id[:16]}",
            status="active",
            created_at=now,
            updated_at=now,
        ),
        NotificationRoute(
            route_id=admin_route_id,
            organization_id=organization_id,
            provider_instance_id=_LOG_PROVIDER_INSTANCE_ID,
            recipient_kind="admin",
            owner_user_id=None,
            channel_key="telegram",
            provider_key="log_only",
            mode="all",
            category_filter=("admin_critical", "admin_alert", "admin_report"),
            scope_filter_json={"stage": "09", "run": run_id},
            schedule_json={},
            recipient_address_ref=f"telegram_ref:stage09:admin:{run_id[:16]}",
            status="active",
            created_at=now,
            updated_at=now,
        ),
    )


def _default_run_id() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stage09-production-canary")
    parser.add_argument(
        "--config",
        default="configs/prod/notifications.yaml",
        help="Path to notifications.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=("log-only-matrix", "real-readiness"),
        default="log-only-matrix",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--organization-id", type=UUID, required=True)
    parser.add_argument("--owner-user-id", type=UUID, required=True)
    parser.add_argument("--provider-instance-id", type=UUID)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
