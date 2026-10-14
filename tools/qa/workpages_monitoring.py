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
    port = int(environ.get("ROEHUB_PROOF_PORT", "20120"))
    api_port = port + 1 if "ROEHUB_PROOF_PORT" in environ else 20111
    probes = (
        OperationalProbe(
            service_id="api",
            capability="product.web_api",
            kind="http_json",
            target=f"http://127.0.0.1:{api_port}/health",
            runbook_id="web.api-health-degraded",
            action_ref="diagnostics",
        ),
        OperationalProbe(
            service_id="web",
            capability="product.web_api",
            kind="http_json",
            target=f"http://127.0.0.1:{port}/health/ready",
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
