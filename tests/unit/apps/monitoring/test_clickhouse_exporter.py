from __future__ import annotations

from typing import cast

from prometheus_client.samples import Sample

from apps.monitoring.clickhouse_exporter import (
    ClickHouseExporterCollector,
    ClickHouseMetricsSnapshot,
)


class _OkClient:
    def fetch_snapshot(self) -> ClickHouseMetricsSnapshot:
        return ClickHouseMetricsSnapshot(
            scrape_duration_seconds=0.125,
            system_events={
                "InsertedRows": 123.0,
                "Query": 456.0,
            },
            system_metrics={
                "HTTPConnection": 7.0,
                "Query": 2.0,
            },
            uptime_seconds=321.0,
        )


class _FailingClient:
    def fetch_snapshot(self) -> ClickHouseMetricsSnapshot:
        raise RuntimeError("boom")


def _metric_samples_by_name(*, collector: ClickHouseExporterCollector) -> dict[str, list[Sample]]:
    """
    Materialize Prometheus samples keyed by metric name for deterministic assertions.

    Args:
        collector: Collector under test.
    Returns:
        dict[str, list[object]]: Mapping of sample name to emitted samples.
    Assumptions:
        Collector yields Prometheus family objects with `.samples`.
    Raises:
        None.
    Side Effects:
        Executes collector `collect()` once.
    """
    samples: dict[str, list[Sample]] = {}
    for family in collector.collect():
        for sample in family.samples:
            typed_sample = cast(Sample, sample)
            samples.setdefault(typed_sample.name, []).append(typed_sample)
    return samples


def test_clickhouse_exporter_emits_successful_snapshot_metric_families() -> None:
    """
    Verify collector emits scrape status, uptime, metrics, and events on success.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Snapshot client returns deterministic values.
    Raises:
        AssertionError: If emitted metric names or values differ from snapshot.
    Side Effects:
        None.
    """
    collector = ClickHouseExporterCollector(client=_OkClient())

    samples = _metric_samples_by_name(collector=collector)

    assert samples["clickhouse_exporter_scrape_success"][0].value == 1.0
    assert samples["clickhouse_exporter_scrape_duration_seconds"][0].value == 0.125
    assert samples["clickhouse_uptime_seconds"][0].value == 321.0
    assert {
        sample.labels["metric"]: sample.value
        for sample in samples["clickhouse_system_metric_value"]
    } == {
        "HTTPConnection": 7.0,
        "Query": 2.0,
    }
    assert {
        sample.labels["event"]: sample.value
        for sample in samples["clickhouse_system_event_total"]
    } == {
        "InsertedRows": 123.0,
        "Query": 456.0,
    }


def test_clickhouse_exporter_reports_failed_scrape_without_crashing() -> None:
    """
    Verify collector emits scrape failure gauge when ClickHouse fetch raises.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Exporter must stay scrapeable even when ClickHouse is unavailable.
    Raises:
        AssertionError: If failure gauges are absent or incorrect.
    Side Effects:
        None.
    """
    collector = ClickHouseExporterCollector(client=_FailingClient())

    samples = _metric_samples_by_name(collector=collector)

    assert samples["clickhouse_exporter_scrape_success"][0].value == 0.0
    assert samples["clickhouse_exporter_scrape_duration_seconds"][0].value == 0.0
