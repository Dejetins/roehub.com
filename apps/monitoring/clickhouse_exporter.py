"""
Repo-managed ClickHouse exporter for Prometheus.

Docs:
  - docs/runbooks/mac-studio-monitoring-plan.md
  - docs/runbooks/mac-studio-backend-operations.md
Related:
  - infra/docker/docker-compose.backend.yml
  - infra/monitoring/monitoring/prometheus/prometheus.yml
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Final, Iterable, Mapping, Protocol, cast

import httpx
from prometheus_client import REGISTRY, start_http_server
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric

log = logging.getLogger(__name__)

_DEFAULT_EVENTS: Final[tuple[str, ...]] = (
    "InsertedBytes",
    "InsertedRows",
    "Query",
    "SelectQuery",
    "SelectedBytes",
    "SelectedRows",
)
_DEFAULT_METRICS: Final[tuple[str, ...]] = (
    "BackgroundMergesAndMutationsPoolTask",
    "HTTPConnection",
    "Query",
    "TCPConnection",
)


@dataclass(frozen=True, slots=True)
class ClickHouseMetricsSnapshot:
    """
    Immutable ClickHouse scrape snapshot used by Prometheus collector.

    Docs:
      - docs/runbooks/mac-studio-monitoring-plan.md
    Related:
      - apps/monitoring/clickhouse_exporter.py

    Attributes:
        scrape_duration_seconds: End-to-end exporter scrape duration.
        system_events: Selected `system.events` counters keyed by event name.
        system_metrics: Selected `system.metrics` gauges keyed by metric name.
        uptime_seconds: ClickHouse process uptime in seconds.
    """

    scrape_duration_seconds: float
    system_events: Mapping[str, float]
    system_metrics: Mapping[str, float]
    uptime_seconds: float


class ClickHouseMetricsClient(Protocol):
    """
    Protocol for ClickHouse snapshot acquisition used by exporter collector.

    Docs:
      - docs/runbooks/mac-studio-monitoring-plan.md
    Related:
      - apps/monitoring/clickhouse_exporter.py
    """

    def fetch_snapshot(self) -> ClickHouseMetricsSnapshot:
        """
        Fetch one deterministic metrics snapshot from ClickHouse.

        Args:
            None.
        Returns:
            ClickHouseMetricsSnapshot: Latest snapshot payload.
        Raises:
            Exception: If ClickHouse is unavailable or returns invalid response.
        """
        ...


@dataclass(frozen=True, slots=True)
class HttpClickHouseMetricsClient:
    """
    HTTP client that reads ClickHouse system tables via the native HTTP interface.

    Docs:
      - docs/runbooks/mac-studio-monitoring-plan.md
    Related:
      - infra/docker/docker-compose.backend.yml
      - infra/monitoring/monitoring/prometheus/prometheus.yml

    Args:
        database: Default database passed to ClickHouse HTTP interface.
        password: Optional ClickHouse password.
        scrape_uri: Base ClickHouse HTTP URI, for example `http://clickhouse:8123/`.
        timeout_seconds: Request timeout per HTTP round-trip.
        user: ClickHouse user name.
        verify_tls: Whether HTTPS certificate verification stays enabled.
    """

    database: str
    password: str
    scrape_uri: str
    timeout_seconds: float
    user: str
    verify_tls: bool

    def fetch_snapshot(self) -> ClickHouseMetricsSnapshot:
        """
        Fetch selected system metrics, events, and uptime from ClickHouse.

        Args:
            None.
        Returns:
            ClickHouseMetricsSnapshot: Parsed snapshot with deterministic key ordering.
        Assumptions:
            ClickHouse HTTP interface is reachable on `scrape_uri`.
        Raises:
            httpx.HTTPError: On transport or HTTP status failures.
            KeyError: If expected JSON payload shape is missing.
            ValueError: If numeric values cannot be converted to floats.
        Side Effects:
            Opens short-lived HTTP connections to ClickHouse.
        """
        started_at = time.perf_counter()
        auth = (self.user, self.password) if self.user else None
        with httpx.Client(
            auth=auth,
            base_url=self.scrape_uri,
            follow_redirects=False,
            timeout=self.timeout_seconds,
            verify=self.verify_tls,
        ) as client:
            system_metrics = self._query_name_value_map(
                client=client,
                sql=(
                    "SELECT metric, value "
                    "FROM system.metrics "
                    "WHERE metric IN ('BackgroundMergesAndMutationsPoolTask', 'HTTPConnection', "
                    "'Query', 'TCPConnection') "
                    "ORDER BY metric ASC FORMAT JSON"
                ),
                name_key="metric",
            )
            system_events = self._query_name_value_map(
                client=client,
                sql=(
                    "SELECT event, value "
                    "FROM system.events "
                    "WHERE event IN ('InsertedBytes', 'InsertedRows', 'Query', 'SelectQuery', "
                    "'SelectedBytes', 'SelectedRows') "
                    "ORDER BY event ASC FORMAT JSON"
                ),
                name_key="event",
            )
            uptime_seconds = self._query_single_value(
                client=client,
                sql="SELECT uptime() AS value FORMAT JSON",
            )

        return ClickHouseMetricsSnapshot(
            scrape_duration_seconds=time.perf_counter() - started_at,
            system_events=system_events,
            system_metrics=system_metrics,
            uptime_seconds=uptime_seconds,
        )

    def _query_name_value_map(
        self,
        *,
        client: httpx.Client,
        sql: str,
        name_key: str,
    ) -> dict[str, float]:
        """
        Execute one ClickHouse query that returns `(name, value)` rows.

        Args:
            client: Reused HTTP client.
            sql: SQL statement returning JSON rows.
            name_key: JSON field name for metric/event key.
        Returns:
            dict[str, float]: Deterministically ordered metric map.
        Assumptions:
            Query uses `FORMAT JSON` and exposes `data` rows.
        Raises:
            httpx.HTTPError: On HTTP failure.
            KeyError: If JSON payload is malformed.
            ValueError: If values are not numeric.
        Side Effects:
            Performs one HTTP GET to ClickHouse.
        """
        response = client.get(
            "/",
            params={
                "database": self.database,
                "query": sql,
            },
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload["data"]
        metrics: dict[str, float] = {}
        for row in rows:
            name = str(row[name_key])
            metrics[name] = float(row["value"])
        return dict(sorted(metrics.items(), key=lambda item: item[0]))

    def _query_single_value(self, *, client: httpx.Client, sql: str) -> float:
        """
        Execute one ClickHouse query that returns exactly one numeric value.

        Args:
            client: Reused HTTP client.
            sql: SQL statement returning one row with `value`.
        Returns:
            float: Parsed numeric value.
        Assumptions:
            Query result includes one row inside JSON `data`.
        Raises:
            httpx.HTTPError: On HTTP failure.
            KeyError: If JSON payload is malformed.
            ValueError: If value cannot be converted to float.
        Side Effects:
            Performs one HTTP GET to ClickHouse.
        """
        response = client.get(
            "/",
            params={
                "database": self.database,
                "query": sql,
            },
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload["data"]
        if not rows:
            raise ValueError("ClickHouse uptime query returned no rows")
        return float(rows[0]["value"])


@dataclass(slots=True, eq=False)
class ClickHouseExporterCollector:
    """
    Prometheus custom collector that exposes selected ClickHouse service metrics.

    Docs:
      - docs/runbooks/mac-studio-monitoring-plan.md
    Related:
      - infra/monitoring/monitoring/prometheus/prometheus.yml
      - infra/docker/docker-compose.backend.yml

    Args:
        client: Client implementation used to fetch ClickHouse snapshots.
    """

    client: ClickHouseMetricsClient

    def collect(self) -> Iterable[Metric]:
        """
        Yield Prometheus metric families for current ClickHouse snapshot.

        Args:
            None.
        Returns:
            Iterable[object]: Prometheus metric family objects.
        Assumptions:
            Collector must keep `/metrics` endpoint available even when ClickHouse is down.
        Raises:
            None.
        Side Effects:
            Logs warning when ClickHouse scrape fails.
        """
        try:
            snapshot = self.client.fetch_snapshot()
        except Exception as error:  # pragma: no cover - exercised via unit tests
            log.warning("clickhouse exporter scrape failed: %s", error)
            yield self._build_scrape_duration_metric(value=0.0)
            yield self._build_scrape_success_metric(value=0.0)
            return

        yield self._build_scrape_duration_metric(value=snapshot.scrape_duration_seconds)
        yield self._build_scrape_success_metric(value=1.0)
        yield self._build_uptime_metric(value=snapshot.uptime_seconds)
        yield self._build_system_metric_family(values=snapshot.system_metrics)
        yield self._build_system_event_family(values=snapshot.system_events)

    @staticmethod
    def _build_scrape_duration_metric(*, value: float) -> GaugeMetricFamily:
        family = GaugeMetricFamily(
            "clickhouse_exporter_scrape_duration_seconds",
            "Duration of the last ClickHouse exporter scrape in seconds.",
        )
        family.add_metric([], value)
        return family

    @staticmethod
    def _build_scrape_success_metric(*, value: float) -> GaugeMetricFamily:
        family = GaugeMetricFamily(
            "clickhouse_exporter_scrape_success",
            "Whether the last ClickHouse exporter scrape succeeded (1) or failed (0).",
        )
        family.add_metric([], value)
        return family

    @staticmethod
    def _build_system_metric_family(*, values: Mapping[str, float]) -> GaugeMetricFamily:
        family = GaugeMetricFamily(
            "clickhouse_system_metric_value",
            "Selected current-value metrics from ClickHouse system.metrics.",
            labels=["metric"],
        )
        for metric_name, metric_value in values.items():
            family.add_metric([metric_name], metric_value)
        return family

    @staticmethod
    def _build_system_event_family(*, values: Mapping[str, float]) -> CounterMetricFamily:
        family = CounterMetricFamily(
            "clickhouse_system_event_total",
            "Selected cumulative counters from ClickHouse system.events.",
            labels=["event"],
        )
        for event_name, event_value in values.items():
            family.add_metric([event_name], event_value)
        return family

    @staticmethod
    def _build_uptime_metric(*, value: float) -> GaugeMetricFamily:
        family = GaugeMetricFamily(
            "clickhouse_uptime_seconds",
            "Current ClickHouse process uptime in seconds.",
        )
        family.add_metric([], value)
        return family


def _build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for repo-managed ClickHouse exporter service.

    Args:
        None.
    Returns:
        argparse.ArgumentParser: Configured CLI parser.
    Assumptions:
        Environment variables provide production defaults in Docker compose.
    Raises:
        None.
    Side Effects:
        None.
    """
    parser = argparse.ArgumentParser(prog="roehub-clickhouse-exporter")
    parser.add_argument("--host", default=os.getenv("CLICKHOUSE_EXPORTER_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("CLICKHOUSE_EXPORTER_PORT", "9116")),
    )
    parser.add_argument(
        "--scrape-uri",
        default=os.getenv("CLICKHOUSE_EXPORTER_SCRAPE_URI", "http://clickhouse:8123/"),
    )
    parser.add_argument(
        "--database",
        default=os.getenv("CH_DATABASE", "default"),
    )
    parser.add_argument(
        "--user",
        default=os.getenv("CH_USER", ""),
    )
    parser.add_argument(
        "--password",
        default=os.getenv("CH_PASSWORD", ""),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("CLICKHOUSE_EXPORTER_TIMEOUT_SECONDS", "5.0")),
    )
    parser.add_argument(
        "--verify-tls",
        action="store_true",
        default=_parse_bool_env(
            os.getenv("CLICKHOUSE_EXPORTER_VERIFY_TLS", os.getenv("CH_VERIFY", "1"))
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Run repo-managed ClickHouse exporter HTTP server.

    Args:
        argv: Optional CLI arguments without program name.
    Returns:
        int: Process exit code.
    Assumptions:
        Service is supervised by Docker restart policy in production.
    Raises:
        None.
    Side Effects:
        Starts Prometheus HTTP exposition server on configured host/port.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=os.getenv("ROEHUB_LOG_LEVEL", "INFO").upper(),
    )
    args = _build_parser().parse_args(argv)
    collector = ClickHouseExporterCollector(
        client=HttpClickHouseMetricsClient(
            database=args.database,
            password=args.password,
            scrape_uri=args.scrape_uri,
            timeout_seconds=args.timeout_seconds,
            user=args.user,
            verify_tls=args.verify_tls,
        )
    )
    REGISTRY.register(cast(Any, collector))
    start_http_server(port=args.port, addr=args.host)
    log.info(
        "clickhouse exporter listening on %s:%s scraping %s",
        args.host,
        args.port,
        args.scrape_uri,
    )
    while True:
        time.sleep(3600)


def _parse_bool_env(value: str) -> bool:
    """
    Parse deterministic boolean environment variable values.

    Args:
        value: Raw environment string.
    Returns:
        bool: Parsed boolean value.
    Assumptions:
        Supported truthy values are `1`, `true`, `yes`, and `on`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return value.strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "ClickHouseExporterCollector",
    "ClickHouseMetricsSnapshot",
    "HttpClickHouseMetricsClient",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
