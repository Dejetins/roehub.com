"""Reuse the operational observer against explicit local preview dependencies."""

from typing import Mapping
from urllib.parse import urlparse

from apps.monitoring.operational_health import (
    OperationalHealthService,
    OperationalManifest,
    OperationalProbe,
)


def build_preview_observer(environ: Mapping[str, str]) -> OperationalHealthService:
    postgres = urlparse(environ["STRATEGY_PG_DSN"])
    probes = (
        OperationalProbe(
            service_id="api",
            capability="product.web_api",
            kind="http_json",
            target="http://127.0.0.1:20111/health",
            runbook_id="web.api-health-degraded",
            action_ref="diagnostics",
        ),
        OperationalProbe(
            service_id="web",
            capability="product.web_api",
            kind="http_json",
            target="http://127.0.0.1:20120/health/ready",
            runbook_id="web.api-health-degraded",
            action_ref="diagnostics",
        ),
        OperationalProbe(
            service_id="postgresql",
            capability="storage.postgresql",
            kind="tcp_reachability",
            target=f"{postgres.hostname}:{postgres.port or 5432}",
            runbook_id="runtime.service-degraded",
            action_ref="diagnostics",
        ),
        OperationalProbe(
            service_id="clickhouse",
            capability="storage.clickhouse",
            kind="tcp_reachability",
            target=f"{environ['CH_HOST']}:{environ['CH_PORT']}",
            runbook_id="database.clickhouse-degraded",
            action_ref="diagnostics",
        ),
    )
    return OperationalHealthService(
        manifest=OperationalManifest(profile="trading", services=probes)
    )


class PreviewOperationalClient:
    def __init__(self, service: OperationalHealthService):
        self.service = service

    def snapshot(self):
        return self.service.snapshot()

    def events(self):
        return self.service.recent_events()
