from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from apps.api.wiring.modules.indicators import (
    build_artifact_precompute_indicators_compute,
    build_indicators_registry,
)
from apps.cli.wiring.db.clickhouse import ClickHouseSettingsLoader, _clickhouse_client
from trading.contexts.backtest_artifacts.adapters.outbound import (
    AtomicArtifactAvailabilitySummaryWriterV2,
    AtomicArtifactCurrentPointerWriterV2,
    BacktestArtifactPathBuilderV2,
    PostgresBacktestJobRepository,
    PsycopgBacktestPostgresGateway,
    YamlBacktestGridDefaultsProvider,
    build_backtest_artifacts_runtime_config_hash,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest_artifacts.application.services import (
    ArtifactCoordinatesV2,
    BacktestArtifactAvailabilitySummaryGeneratorV2,
    BacktestArtifactAvailabilitySummaryResultV2,
    BacktestArtifactPrecomputeRunnerV2,
    BacktestArtifactSlotPublisherV2,
    BacktestSignalRulesEngineV2,
    YamlBacktestArtifactLoaderV2,
)
from trading.contexts.backtest_artifacts.application.use_cases import (
    PublishBacktestArtifactsV2Request,
    PublishBacktestArtifactsV2Result,
    PublishBacktestArtifactsV2UseCase,
)
from trading.contexts.indicators.application.services import GridBuilder
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseCanonicalCandleIndexReader,
    ClickHouseCanonicalCandleReader,
    ClickHouseConnectGateway,
)
from trading.platform.time.system_clock import SystemClock  # noqa: F401

log = logging.getLogger(__name__)

PublishUseCaseFactoryV2 = Callable[
    [str | None, Mapping[str, str]],
    PublishBacktestArtifactsV2UseCase,
]
SummaryRegeneratorFactoryV2 = Callable[
    [str | None, Mapping[str, str]],
    "BacktestArtifactAvailabilitySummaryRegeneratorV2",
]


class BacktestArtifactAvailabilitySummaryRegeneratorV2(Protocol):
    def regenerate(self) -> BacktestArtifactAvailabilitySummaryResultV2:
        """Regenerate root-level artifact availability summary."""
        ...


@dataclass(frozen=True, slots=True)
class BacktestArtifactPublishCliArgs:
    """
    Validated CLI arguments for one explicit backtest artifact publish run.

    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/architecture/backtest/README.md
    Related:
      - apps/cli/commands/backtest_artifact_publish.py
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """

    config: str | None
    exchange: str | None
    market_type: str | None
    symbol: str | None
    full_rebuild: bool
    regenerate_summary_only: bool
    report_format: str

    def to_request(self) -> PublishBacktestArtifactsV2Request:
        """
        Convert validated CLI args into the shared publish use-case request DTO.

        Args:
            None.
        Returns:
            PublishBacktestArtifactsV2Request: Explicit single-target publish request.
        Assumptions:
            Path/config resolution stays in CLI wiring, not in the use-case request.
        Raises:
            ValueError: If coordinate literals violate strict artifact token rules.
        Side Effects:
            None.
        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if self.exchange is None or self.market_type is None or self.symbol is None:
            raise ValueError("publish mode requires --exchange, --market-type, and --symbol")
        return PublishBacktestArtifactsV2Request(
            coordinates=ArtifactCoordinatesV2(
                exchange=self.exchange,
                market_type=self.market_type,
                symbol=self.symbol,
            ),
            full_rebuild=self.full_rebuild,
        )


class BacktestArtifactPublishCli:
    """
    Manual CLI entrypoint for the shared backtest artifact publisher use-case.

    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/architecture/backtest/README.md
    Related:
      - apps/cli/main/main.py
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        use_case_factory: PublishUseCaseFactoryV2 | None = None,
        summary_regenerator_factory: SummaryRegeneratorFactoryV2 | None = None,
    ) -> None:
        """
        Store environment and optional factory override for CLI wiring/tests.

        Args:
            environ: Optional explicit environment mapping.
            use_case_factory: Optional override used by unit tests or alternate wiring.
            summary_regenerator_factory: Optional override used by unit tests or alternate wiring.
        Returns:
            None.
        Assumptions:
            Production CLI uses fail-fast default wiring when no factory override is supplied.
        Raises:
            None.
        Side Effects:
            Stores configuration in memory.
        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
          - docs/architecture/backtest/README.md
        Related:
          - tests/unit/apps/cli/test_backtest_artifact_publish_cli.py
        """
        self._environ = dict(environ) if environ is not None else None
        self._use_case_factory = (
            use_case_factory if use_case_factory is not None else _build_publish_use_case_v2
        )
        self._summary_regenerator_factory = (
            summary_regenerator_factory
            if summary_regenerator_factory is not None
            else _build_availability_summary_regenerator_v2
        )

    def run(self, argv: Sequence[str]) -> int:
        """
        Parse CLI args, build the shared publish use-case, and execute one publish run.

        Args:
            argv: Raw CLI arguments without the program name.
        Returns:
            int: Process exit code (`0` success, `1` runtime failure, `2` argument failure).
        Assumptions:
            Artifact config, ClickHouse, indicators defaults, and Postgres pin-guard storage are
            available to the selected environment.
        Raises:
            None.
        Side Effects:
            Reads configs/environment, queries storage, writes artifacts, and prints one report.
        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
        """
        parser = _build_parser()
        ns = parser.parse_args(list(argv))
        try:
            args = self._to_args(ns)
        except Exception as error:  # noqa: BLE001
            log.error("Invalid arguments: %s", error)
            return 2

        try:
            if args.regenerate_summary_only:
                summary_regenerator = self._summary_regenerator_factory(
                    args.config,
                    self._effective_environ(),
                )
                summary_result = summary_regenerator.regenerate()
                print(_render_summary_report_v2(summary_result, report_format=args.report_format))
                return 0

            use_case = self._use_case_factory(args.config, self._effective_environ())
            result = use_case.run(args.to_request())
            summary_regenerator = self._summary_regenerator_factory(
                args.config,
                self._effective_environ(),
            )
            summary_result = summary_regenerator.regenerate()
        except Exception as error:  # noqa: BLE001
            log.exception("Backtest artifact publish failed: %s", error)
            return 1

        print(
            _render_report_v2(
                result=result,
                summary_result=summary_result,
                report_format=args.report_format,
            )
        )
        return 0

    def _effective_environ(self) -> Mapping[str, str]:
        """
        Return the explicit environment override or the current process environment.

        Args:
            None.
        Returns:
            Mapping[str, str]: Environment mapping used by CLI wiring.
        Assumptions:
            Unit tests may supply an isolated environment mapping.
        Raises:
            None.
        Side Effects:
            Reads `os.environ` only when no override is configured.
        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/cli/commands/backtest_artifact_publish.py
        """
        if self._environ is not None:
            return self._environ
        return os.environ

    def _to_args(self, ns: argparse.Namespace) -> BacktestArtifactPublishCliArgs:
        """
        Normalize parsed argparse namespace into the validated CLI args DTO.

        Args:
            ns: Parsed argparse namespace.
        Returns:
            BacktestArtifactPublishCliArgs: Validated immutable CLI args.
        Assumptions:
            `argparse` already enforced required options and report-format choices.
        Raises:
            ValueError: If one string literal is blank after trimming.
        Side Effects:
            None.
        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        config = None if ns.config is None or not str(ns.config).strip() else str(ns.config).strip()
        regenerate_summary_only = bool(ns.regenerate_summary_only)
        exchange = None if ns.exchange is None else str(ns.exchange).strip()
        market_type = None if ns.market_type is None else str(ns.market_type).strip()
        symbol = None if ns.symbol is None else str(ns.symbol).strip()
        if not regenerate_summary_only:
            if not exchange:
                raise ValueError("--exchange must be non-empty")
            if not market_type:
                raise ValueError("--market-type must be non-empty")
            if not symbol:
                raise ValueError("--symbol must be non-empty")
        if regenerate_summary_only and (exchange or market_type or symbol):
            raise ValueError(
                "--regenerate-summary-only must not be combined with symbol publish arguments"
            )
        if regenerate_summary_only and bool(ns.full_rebuild):
            raise ValueError("--regenerate-summary-only must not be combined with --full-rebuild")
        return BacktestArtifactPublishCliArgs(
            config=config,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            full_rebuild=bool(ns.full_rebuild),
            regenerate_summary_only=regenerate_summary_only,
            report_format=str(ns.report_format),
        )


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the manual backtest artifact publisher entrypoint.

    Args:
        None.
    Returns:
        argparse.ArgumentParser: Configured parser instance.
    Assumptions:
        Config path may be omitted to use environment-aware artifact config resolution.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/architecture/backtest/README.md
    Related:
      - apps/cli/main/main.py
    """
    parser = argparse.ArgumentParser(prog="backtest-artifact-publish")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to backtest_artifacts.yaml. Default: "
            "ROEHUB_BACKTEST_ARTIFACTS_CONFIG or configs/<ROEHUB_ENV>/backtest_artifacts.yaml."
        ),
    )
    parser.add_argument("--exchange", default=None, help="Exchange literal, e.g. binance")
    parser.add_argument("--market-type", default=None, help="Market type literal, e.g. spot")
    parser.add_argument("--symbol", default=None, help="Symbol literal, e.g. BTCUSDT")
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Disable inactive-slot reuse and force a full rebuild before publish",
    )
    parser.add_argument(
        "--report-format",
        choices=("json", "text"),
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--regenerate-summary-only",
        action="store_true",
        help=(
            "Regenerate availability_summary.yaml from existing active current/manifest state "
            "without rebuilding one symbol root."
        ),
    )
    return parser


def _build_publish_use_case_v2(
    config_path: str | None,
    environ: Mapping[str, str],
) -> PublishBacktestArtifactsV2UseCase:
    """
    Build the shared publish use-case with fail-fast CLI wiring dependencies.

    Args:
        config_path: Optional explicit artifact config path override.
        environ: Runtime environment mapping.
    Returns:
        PublishBacktestArtifactsV2UseCase: Ready-to-run shared publish orchestration.
    Assumptions:
        CLI wiring should reuse the same adapters/config contracts later planned for scheduler
        integration.
    Raises:
        ValueError: If config/env/DSN/contracts are invalid.
        FileNotFoundError: If required configs are missing.
        RuntimeError: If ClickHouse client dependency is not installed.
    Side Effects:
        Reads configs, builds storage adapters, and warms up indicators compute.
    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/architecture/backtest/README.md
      - docs/runbooks/mac-studio-native-backend-operations.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """
    resolved_config_path = (
        resolve_backtest_artifacts_config_path(environ=environ)
        if config_path is None
        else Path(config_path)
    )
    artifact_runtime_config = load_backtest_artifacts_runtime_config(resolved_config_path)
    path_builder = BacktestArtifactPathBuilderV2(root=artifact_runtime_config.artifact_root_path())
    artifact_loader = YamlBacktestArtifactLoaderV2(path_resolver=path_builder)
    pointer_writer = AtomicArtifactCurrentPointerWriterV2(path_resolver=path_builder)
    clickhouse_settings = ClickHouseSettingsLoader(environ).load()
    clickhouse_gateway = ClickHouseConnectGateway(_clickhouse_client(clickhouse_settings))
    strategy_postgres_dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    if not strategy_postgres_dsn:
        raise ValueError("STRATEGY_PG_DSN is required for backtest-artifact-publish CLI")

    defaults_provider = YamlBacktestGridDefaultsProvider.from_environ(
        environ=environ,
        artifact_config_path=config_path,
    )
    indicator_registry = build_indicators_registry(
        environ=environ,
        artifact_config_path=config_path,
    )
    indicator_compute = build_artifact_precompute_indicators_compute(
        environ=environ,
        artifact_config_path=config_path,
    )
    artifact_config_hash = build_backtest_artifacts_runtime_config_hash(
        config=artifact_runtime_config
    )
    precompute_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=artifact_runtime_config.to_precompute_runtime_settings(
            config_sha256=artifact_config_hash
        ),
        artifact_loader=artifact_loader,
        canonical_candle_reader=ClickHouseCanonicalCandleReader(
            gateway=clickhouse_gateway,
            database=clickhouse_settings.database,
        ),
        defaults_provider=defaults_provider,
        signal_rules_engine=BacktestSignalRulesEngineV2(defaults_provider=defaults_provider),
        indicator_compute=indicator_compute,
        indicator_grid_builder=GridBuilder(registry=indicator_registry),
    )
    slot_publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=artifact_loader,
        current_pointer_writer=pointer_writer,
        job_repository=PostgresBacktestJobRepository(
            gateway=PsycopgBacktestPostgresGateway(dsn=strategy_postgres_dsn)
        ),
    )
    return PublishBacktestArtifactsV2UseCase(
        canonical_candle_index_reader=ClickHouseCanonicalCandleIndexReader(
            gateway=clickhouse_gateway,
            database=clickhouse_settings.database,
        ),
        precompute_runner=precompute_runner,
        slot_publisher=slot_publisher,
        validation_spec=artifact_runtime_config.to_validation_spec(),
    )


def _build_availability_summary_regenerator_v2(
    config_path: str | None,
    environ: Mapping[str, str],
) -> BacktestArtifactAvailabilitySummaryGeneratorV2:
    """
    Build the root-level availability summary regenerator from artifact config only.

    Args:
        config_path: Optional explicit artifact config path override.
        environ: Runtime environment mapping.
    Returns:
        BacktestArtifactAvailabilitySummaryGeneratorV2: Ready-to-run summary generator.
    Assumptions:
        Summary regeneration must not require ClickHouse, Postgres, exchange APIs, or AI context.
    Raises:
        FileNotFoundError: If the artifact config is missing.
        ValueError: If artifact config values are invalid.
    Side Effects:
        Reads artifact config from disk.
    """
    resolved_config_path = (
        resolve_backtest_artifacts_config_path(environ=environ)
        if config_path is None
        else Path(config_path)
    )
    artifact_runtime_config = load_backtest_artifacts_runtime_config(resolved_config_path)
    path_builder = BacktestArtifactPathBuilderV2(root=artifact_runtime_config.artifact_root_path())
    artifact_loader = YamlBacktestArtifactLoaderV2(path_resolver=path_builder)
    return BacktestArtifactAvailabilitySummaryGeneratorV2(
        artifact_root=artifact_runtime_config.artifact_root_path(),
        path_resolver=path_builder,
        artifact_loader=artifact_loader,
        writer=AtomicArtifactAvailabilitySummaryWriterV2(),
    )


def _render_report_v2(
    *,
    result: PublishBacktestArtifactsV2Result,
    summary_result: BacktestArtifactAvailabilitySummaryResultV2,
    report_format: str,
) -> str:
    """
    Render the shared publish result into one CLI output payload.

    Args:
        result: Shared publish result returned by the use-case.
        report_format: Requested output format (`json` or `text`).
    Returns:
        str: Rendered CLI report body.
    Assumptions:
        JSON remains the default because later scheduler integration expects machine-readable
        diagnostics, while text mode should still expose stage-level rebuild breakdowns for manual
        operators.
    Raises:
        ValueError: If report format is unsupported.
    Side Effects:
        None.
    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """
    if report_format == "json":
        payload = dict(result.as_dict())
        payload["availability_summary"] = dict(summary_result.as_dict())
        return json.dumps(payload, ensure_ascii=False)
    if report_format == "text":
        return (
            "backtest-artifact-publish report:\n"
            f"- status: {result.status}\n"
            f"- mode: {result.publish_mode}\n"
            f"- target: {result.coordinates.exchange}/"
            f"{result.coordinates.market_type}/{result.coordinates.symbol}\n"
            f"- previous_slot: {result.previous_active_slot}\n"
            f"- published_slot: {result.published_active_slot}\n"
            f"- slot_generation: {result.published_slot_generation}\n"
            f"- manifest_sha256: {result.published_manifest_sha256}\n"
            f"- asof_date: {result.asof_date}\n"
            f"- published_at_utc: {result.published_at_utc}\n"
            f"- requested_range: {result.requested_start_utc} -> {result.requested_end_utc}\n"
            f"- source_range: {result.source_start_utc} -> {result.source_end_utc}\n"
            f"- source_candle_count: {result.source_candle_count}\n"
            f"- reused_prefix_bars: {result.reused_prefix_bars}\n"
            f"- rewritten_tail_bars: {result.rewritten_tail_bars}\n"
            f"- stage_rebuild_stats: prices(reused={result.stage_rebuild_stats.prices.reused_prefix_bars}, rewritten={result.stage_rebuild_stats.prices.rewritten_tail_bars}), "  # noqa: E501
            f"mappings(reused={result.stage_rebuild_stats.mappings.reused_prefix_bars}, rewritten={result.stage_rebuild_stats.mappings.rewritten_tail_bars}), "  # noqa: E501
            f"signals(reused={result.stage_rebuild_stats.signals.reused_prefix_bars}, rewritten={result.stage_rebuild_stats.signals.rewritten_tail_bars}), "  # noqa: E501
            f"hit_times(reused={result.stage_rebuild_stats.hit_times.reused_prefix_bars}, rewritten={result.stage_rebuild_stats.hit_times.rewritten_tail_bars})\n"  # noqa: E501
            f"- tail_rebuild_bars: prices={result.tail_rebuild_bars.prices}, "
            f"mappings={result.tail_rebuild_bars.mappings}, "
            f"signals={result.tail_rebuild_bars.signals}, "
            f"hit_times={result.tail_rebuild_bars.hit_times}\n"
            f"- signal_manifest_count: {result.validation.signal_manifest_count}\n"
            f"- hit_times_manifest_present: {result.validation.hit_times_manifest_present}\n"
            f"- availability_summary_path: {summary_result.summary_path}\n"
            f"- availability_summary_hash: {summary_result.summary_hash}\n"
            f"- availability_summary_instruments: {summary_result.instrument_count}\n"
        )
    raise ValueError(f"unsupported report format: {report_format!r}")


def _render_summary_report_v2(
    result: BacktestArtifactAvailabilitySummaryResultV2,
    *,
    report_format: str,
) -> str:
    if report_format == "json":
        return json.dumps(result.as_dict(), ensure_ascii=False)
    if report_format == "text":
        return (
            "backtest-artifact-availability-summary report:\n"
            f"- summary_path: {result.summary_path}\n"
            f"- summary_hash: {result.summary_hash}\n"
            f"- generated_at_utc: {result.generated_at_utc}\n"
            f"- instrument_count: {result.instrument_count}\n"
            f"- skipped_count: {result.skipped_count}\n"
            f"- skipped_reasons: {dict(result.skipped_reasons)}\n"
        )
    raise ValueError(f"unsupported report format: {report_format!r}")


__all__ = ["BacktestArtifactPublishCli"]
