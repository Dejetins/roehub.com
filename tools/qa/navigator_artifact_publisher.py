"""Run the production managed publisher with loopback-only local metrics."""

from __future__ import annotations

import asyncio
import os
import signal

from prometheus_client import start_http_server

from apps.scheduler.backtest_artifact_publisher.wiring.managed_app import (
    ManagedArtifactPublisherApp,
)
from apps.scheduler.backtest_artifact_publisher.wiring.modules.backtest_artifact_publisher import (
    build_backtest_artifact_publisher_app,
)


async def run() -> None:
    if not os.environ.get("ROEHUB_ARTIFACT_MANAGEMENT_INSTALLATION_ID"):
        raise RuntimeError("Explicit artifact installation configuration is required")
    port = int(os.environ.get("ROEHUB_LOCAL_ARTIFACT_METRICS_PORT", "20123"))
    app = build_backtest_artifact_publisher_app(
        config_path=os.environ["ROEHUB_BACKTEST_ARTIFACTS_CONFIG"],
        environ=os.environ,
        metrics_port=port,
    )
    if not isinstance(app, ManagedArtifactPublisherApp):
        raise RuntimeError("The local publisher requires managed composition")
    app.metrics_port = None
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    server, thread = start_http_server(port, addr="127.0.0.1")
    try:
        await app.run(stop)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    asyncio.run(run())
