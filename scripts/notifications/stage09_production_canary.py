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
    postgres_dsn_presence,
    resolve_notification_postgres_dsn,
    telegram_credential_presence,
)
from trading.contexts.notifications.adapters import (
    PostgresNotificationRepository,
    PsycopgNotificationPostgresGateway,
)
from trading.contexts.notifications.application import (
    NotificationSourceRouter,
    synthetic_notification_matrix,
)
from trading.contexts.notifications.domain import NotificationRoute
from trading.shared_kernel.primitives import UserId

_OWNER_USER_ID = UserId(UUID("11111111-1111-4111-8111-111111111111"))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    env = os.environ
    if args.mode == "real-readiness":
        result = run_real_telegram_readiness(config_path=config_path, environ=env)
    else:
        result = run_log_only_matrix(
            config_path=config_path,
            environ=env,
            run_id=args.run_id or _default_run_id(),
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"ok", "blocked"} else 1


def run_log_only_matrix(
    *,
    config_path: Path,
    environ: Mapping[str, str],
    run_id: str,
) -> dict[str, object]:
    runtime_config = load_notification_dispatcher_runtime_config(config_path=config_path)
    gateway = PsycopgNotificationPostgresGateway(
        dsn=resolve_notification_postgres_dsn(environ=environ)
    )
    repository = PostgresNotificationRepository(gateway=gateway)
    now = datetime.now(UTC)
    routes = _stage09_log_only_routes(run_id=run_id, now=now)
    router = NotificationSourceRouter()
    categories: set[str] = set()
    delivery_ids: list[str] = []

    for route in routes:
        repository.upsert_route(route=route)

    for fact in synthetic_notification_matrix(owner_user_id=_OWNER_USER_ID, now=now):
        unique_fact = replace(fact, fact_id=f"stage09:{run_id}:{fact.fact_id}")
        flow = router.route(fact=unique_fact, routes=routes, now=now)
        repository.record_event(event=flow.event)
        categories.add(flow.event.category)
        for delivery in flow.deliveries:
            stored = repository.record_delivery(delivery=delivery)
            delivery_ids.append(str(stored.delivery_id))

    metrics = NotificationDispatcherPrometheusMetrics()
    dispatcher = build_notification_dispatcher(
        repository=repository,
        runtime_config=runtime_config,
        environ=environ,
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
        "events": len(categories),
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
) -> dict[str, object]:
    _ = load_notification_dispatcher_runtime_config(config_path=config_path)
    gateway = PsycopgNotificationPostgresGateway(
        dsn=resolve_notification_postgres_dsn(environ=environ)
    )
    rows = gateway.fetch_all(
        query="""
        SELECT provider_key, status, count(*) AS count
        FROM notification_routes
        WHERE recipient_kind = 'admin'
          AND channel_key = 'telegram'
          AND provider_key = 'telegram_bot_api'
          AND status = 'active'
        GROUP BY provider_key, status
        """,
        parameters={},
    )
    active_admin_route_count = sum(int(row["count"]) for row in rows)
    telegram_presence = telegram_credential_presence(environ=environ)
    dsn_presence = postgres_dsn_presence(environ=environ)
    ready = active_admin_route_count > 0 and any(telegram_presence.values())
    return {
        "status": "ok" if ready else "blocked",
        "proof": "stage09_real_telegram_readiness",
        "telegram_token_present": any(telegram_presence.values()),
        "preferred_telegram_token_present": telegram_presence.get(
            "ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN", False
        ),
        "fallback_telegram_token_present": telegram_presence.get("TELEGRAM_BOT_TOKEN", False),
        "postgres_dsn_present": any(dsn_presence.values()),
        "active_admin_telegram_route_count": active_admin_route_count,
        "user_confirmation_required": True,
        "blocker": None
        if ready
        else "missing_admin_route_or_user_confirmed_canary_recipient",
    }


def _stage09_log_only_routes(*, run_id: str, now: datetime) -> tuple[NotificationRoute, ...]:
    user_route_id = uuid5(NAMESPACE_URL, f"roehub:notifications:stage09:{run_id}:user")
    admin_route_id = uuid5(NAMESPACE_URL, f"roehub:notifications:stage09:{run_id}:admin")
    return (
        NotificationRoute(
            route_id=user_route_id,
            recipient_kind="user",
            owner_user_id=_OWNER_USER_ID,
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
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
