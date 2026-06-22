from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import apps.cli.commands.backtest_artifact_publish as publish_module
import apps.cli.main.main as cli_main_module
from apps.cli.commands.backtest_artifact_publish import BacktestArtifactPublishCli
from trading.contexts.backtest_artifacts.application.services import (
    ArtifactCoordinatesV2,
    ArtifactStageRebuildStatsCollectionV2,
    ArtifactStageRebuildStatsV2,
    ArtifactTailRebuildBarsV2,
)
from trading.contexts.backtest_artifacts.application.use_cases import (
    PublishBacktestArtifactsV2Request,
    PublishBacktestArtifactsV2Result,
    PublishBacktestArtifactsV2UseCase,
    PublishBacktestArtifactsV2ValidationSummary,
)


@dataclass(slots=True)
class _FakePublishUseCaseV2:
    """
    Recording fake use-case used to verify CLI parsing and request forwarding.

    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/architecture/backtest/README.md
    Related:
      - apps/cli/commands/backtest_artifact_publish.py
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """

    result: PublishBacktestArtifactsV2Result
    requests: list[PublishBacktestArtifactsV2Request] = field(default_factory=list)

    def run(
        self,
        request: PublishBacktestArtifactsV2Request,
    ) -> PublishBacktestArtifactsV2Result:
        """
        Record one request and return the preconfigured deterministic publish result.

        Args:
            request: Shared publish request constructed by the CLI.
        Returns:
            PublishBacktestArtifactsV2Result: Fixed result payload for CLI assertions.
        Assumptions:
            CLI unit tests verify parsing and rendering rather than artifact I/O.
        Raises:
            None.
        Side Effects:
            Appends the request to in-memory call history.
        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
          - docs/architecture/backtest/README.md
        Related:
          - apps/cli/commands/backtest_artifact_publish.py
        """
        self.requests.append(request)
        return self.result


def test_backtest_artifact_publish_cli_forwards_request_and_renders_json(capsys) -> None:
    """
    Verify CLI parses one explicit target, forwards `--full-rebuild`, and prints JSON output.

    Args:
        capsys: pytest stdout/stderr capture fixture.
    Returns:
        None.
    Assumptions:
        CLI should reuse the shared use-case request DTO without ad-hoc field reshaping.
    Raises:
        AssertionError: If forwarded request fields or rendered JSON drift.
    Side Effects:
        Writes one JSON payload to captured stdout.
    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/architecture/backtest/README.md
    Related:
      - apps/cli/commands/backtest_artifact_publish.py
    """
    fake_use_case = _FakePublishUseCaseV2(result=_sample_publish_result_v2())
    factory_calls: list[tuple[str | None, Mapping[str, str]]] = []

    def _factory(
        config_path: str | None,
        environ: Mapping[str, str],
    ) -> PublishBacktestArtifactsV2UseCase:
        """
        Capture CLI wiring inputs and return the recording fake use-case.

        Args:
            config_path: Optional config path override forwarded by the CLI.
            environ: Effective environment mapping forwarded by the CLI.
        Returns:
            _FakePublishUseCaseV2: Recording fake use-case instance.
        Assumptions:
            CLI should pass the explicit environment override unchanged.
        Raises:
            None.
        Side Effects:
            Records wiring arguments in memory.
        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/cli/commands/backtest_artifact_publish.py
        """
        factory_calls.append((config_path, environ))
        return cast(PublishBacktestArtifactsV2UseCase, fake_use_case)

    cli = BacktestArtifactPublishCli(
        environ={"ROEHUB_ENV": "test"},
        use_case_factory=_factory,
    )

    exit_code = cli.run(
        [
            "--exchange",
            "binance",
            "--market-type",
            "spot",
            "--symbol",
            "BTCUSDT",
            "--full-rebuild",
        ]
    )
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)

    assert exit_code == 0
    assert len(factory_calls) == 1
    assert factory_calls[0][0] is None
    assert factory_calls[0][1]["ROEHUB_ENV"] == "test"
    assert len(fake_use_case.requests) == 1
    assert fake_use_case.requests[0].coordinates == ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )
    assert fake_use_case.requests[0].full_rebuild is True
    assert payload["publish_mode"] == "full_rebuild"
    assert payload["coordinates"] == {
        "exchange": "binance",
        "market_type": "spot",
        "symbol": "BTCUSDT",
    }


def test_backtest_artifact_publish_cli_returns_one_on_invalid_config_factory_error() -> None:
    """
    Verify CLI fails fast with exit code `1` when config/wiring setup raises a runtime error.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Config and dependency validation stays inside CLI wiring, not parser normalization.
    Raises:
        AssertionError: If runtime wiring errors do not map to exit code `1`.
    Side Effects:
        Emits one logged exception during the test run.
    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/runbooks/mac-studio-native-backend-operations.md
    Related:
      - apps/cli/commands/backtest_artifact_publish.py
    """

    def _raising_factory(
        config_path: str | None,
        environ: Mapping[str, str],
    ) -> PublishBacktestArtifactsV2UseCase:
        """
        Raise a deterministic configuration error for CLI failure-path assertions.

        Args:
            config_path: Optional config path override ignored by this helper.
            environ: Environment mapping ignored by this helper.
        Returns:
            _FakePublishUseCaseV2: Never returns successfully.
        Assumptions:
            CLI converts dependency-wiring failures into a non-zero exit code.
        Raises:
            ValueError: Always, with a stable invalid-config message.
        Side Effects:
            None.
        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/cli/commands/backtest_artifact_publish.py
        """
        del config_path, environ
        raise ValueError("invalid config")

    cli = BacktestArtifactPublishCli(
        environ={"ROEHUB_ENV": "test"},
        use_case_factory=_raising_factory,
    )

    exit_code = cli.run(
        [
            "--exchange",
            "binance",
            "--market-type",
            "spot",
            "--symbol",
            "BTCUSDT",
        ]
    )

    assert exit_code == 1


def test_backtest_artifact_publish_cli_renders_text_stage_breakdown(capsys) -> None:
    """
    Verify text output includes the stage-level rebuild breakdown for manual operators.

    Args:
        capsys: pytest stdout/stderr capture fixture.
    Returns:
        None.
    Assumptions:
        Text mode should stay compact but still surface `stage_rebuild_stats` explicitly.
    Raises:
        AssertionError: If text rendering drops the stage-level rebuild summary.
    Side Effects:
        Writes one text report to captured stdout.
    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/architecture/backtest/README.md
    Related:
      - apps/cli/commands/backtest_artifact_publish.py
    """
    fake_use_case = _FakePublishUseCaseV2(result=_sample_publish_result_v2())
    cli = BacktestArtifactPublishCli(
        environ={"ROEHUB_ENV": "test"},
        use_case_factory=lambda config_path, environ: cast(
            PublishBacktestArtifactsV2UseCase,
            fake_use_case,
        ),
    )

    exit_code = cli.run(
        [
            "--exchange",
            "binance",
            "--market-type",
            "spot",
            "--symbol",
            "BTCUSDT",
            "--report-format",
            "text",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "stage_rebuild_stats:" in stdout
    assert "prices(reused=4300, rewritten=20)" in stdout
    assert "hit_times(reused=4290, rewritten=20)" in stdout


def test_cli_main_dispatches_backtest_artifact_publish_command(monkeypatch) -> None:
    """
    Verify `apps.cli.main` routes the explicit command to `BacktestArtifactPublishCli`.

    Args:
        monkeypatch: pytest fixture for patching command construction.
    Returns:
        None.
    Assumptions:
        Main CLI router should forward the remaining argv unchanged.
    Raises:
        AssertionError: If dispatch target or forwarded argv regresses.
    Side Effects:
        Monkeypatches the command class inside `apps.cli.main`.
    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - apps/cli/main/main.py
    Related:
      - apps/cli/commands/backtest_artifact_publish.py
    """
    captured_argv: list[list[str]] = []

    class _FakeCommand:
        """
        Minimal fake command object used to record delegated argv.

        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/cli/main/main.py
        """

        def run(self, argv: list[str]) -> int:
            """
            Record delegated argv and return a deterministic success exit code.

            Args:
                argv: Remaining CLI arguments after command selection.
            Returns:
                int: Always `0`.
            Assumptions:
                Main router should not rewrite arguments for explicit command dispatch.
            Raises:
                None.
            Side Effects:
                Appends forwarded argv to in-memory call history.
            Docs:
              - docs/runbooks/backtest-artifacts-rebuild.md
            Related:
              - apps/cli/main/main.py
            """
            captured_argv.append(argv)
            return 0

    monkeypatch.setattr(cli_main_module, "BacktestArtifactPublishCli", _FakeCommand)

    exit_code = cli_main_module.main(
        [
            "backtest-artifact-publish",
            "--exchange",
            "binance",
            "--market-type",
            "spot",
            "--symbol",
            "BTCUSDT",
        ]
    )

    assert exit_code == 0
    assert captured_argv == [
        [
            "--exchange",
            "binance",
            "--market-type",
            "spot",
            "--symbol",
            "BTCUSDT",
        ]
    ]


def test_build_publish_use_case_v2_uses_explicit_artifact_config_for_indicators_resolution(
    monkeypatch,
) -> None:
    """
    Verify CLI wiring forwards explicit artifact config to all indicators-resolution entrypoints.

    Args:
        monkeypatch: pytest fixture used to replace heavy runtime dependencies with fakes.
    Returns:
        None.
    Assumptions:
        Manual `--config configs/prod/backtest_artifacts.yaml` must resolve the matching
        production indicators catalog even when `ROEHUB_ENV` is unset.
    Raises:
        AssertionError: If CLI wiring drops the explicit artifact-config path.
    Side Effects:
        Monkeypatches the CLI wiring module in memory.
    """
    captured: dict[str, Any] = {}

    class _FakeArtifactRuntimeConfig:
        """
        Minimal artifact runtime config stub for CLI wiring tests.

        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/cli/commands/backtest_artifact_publish.py
        """

        def artifact_root_path(self) -> Path:
            """
            Return a deterministic artifact root path.

            Args:
                None.
            Returns:
                Path: Fixed artifact root path.
            Assumptions:
                Path shape is irrelevant to the indicators-resolution assertion.
            Raises:
                None.
            Side Effects:
                None.
            """
            return Path("/tmp/artifacts")

        def to_precompute_runtime_settings(self, *, config_sha256: str) -> object:
            """
            Return a deterministic runtime settings sentinel.

            Args:
                config_sha256: Stable artifact-config hash from wiring.
            Returns:
                object: Sentinel runtime settings object.
            Assumptions:
                The CLI test only needs the call to succeed.
            Raises:
                None.
            Side Effects:
                Stores the hash for later assertions.
            """
            captured["config_sha256"] = config_sha256
            return object()

        def to_validation_spec(self) -> object:
            """
            Return a deterministic validation spec sentinel.

            Args:
                None.
            Returns:
                object: Sentinel validation spec.
            Assumptions:
                Validation details are outside the scope of this wiring test.
            Raises:
                None.
            Side Effects:
                None.
            """
            return object()

    class _FakeClickHouseSettingsLoader:
        """
        Minimal ClickHouse settings loader stub for CLI wiring tests.

        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/cli/commands/backtest_artifact_publish.py
        """

        def __init__(self, environ: Mapping[str, str]) -> None:
            """
            Store the provided environment mapping for parity with production signature.

            Args:
                environ: Environment mapping forwarded by CLI wiring.
            Returns:
                None.
            Assumptions:
                The fake loader only needs to preserve constructor compatibility.
            Raises:
                None.
            Side Effects:
                Stores the environment mapping in memory.
            """
            self._environ = dict(environ)

        def load(self) -> object:
            """
            Return a deterministic settings object with a database attribute.

            Args:
                None.
            Returns:
                object: Simple namespace-like settings object.
            Assumptions:
                Downstream fake dependencies only read the `database` attribute.
            Raises:
                None.
            Side Effects:
                None.
            """
            return type("Settings", (), {"database": "test_db"})()

    def _capture_defaults_provider(_cls, *, environ, artifact_config_path=None):
        """
        Capture defaults-provider wiring inputs and return a sentinel provider.

        Args:
            _cls: Provider class forwarded by the classmethod descriptor.
            environ: Environment mapping forwarded by CLI wiring.
            artifact_config_path: Explicit artifact-config path forwarded by CLI wiring.
        Returns:
            object: Sentinel defaults provider.
        Assumptions:
            The test verifies resolution wiring, not defaults parsing.
        Raises:
            None.
        Side Effects:
            Stores call arguments in memory.
        """
        captured["defaults_artifact_config_path"] = artifact_config_path
        captured["defaults_environ"] = dict(environ)
        return object()

    def _capture_registry(*, environ, artifact_config_path=None):
        """
        Capture registry wiring inputs and return a sentinel registry.

        Args:
            environ: Environment mapping forwarded by CLI wiring.
            artifact_config_path: Explicit artifact-config path forwarded by CLI wiring.
        Returns:
            object: Sentinel registry.
        Assumptions:
            Registry construction itself is outside the scope of this wiring test.
        Raises:
            None.
        Side Effects:
            Stores call arguments in memory.
        """
        captured["registry_artifact_config_path"] = artifact_config_path
        captured["registry_environ"] = dict(environ)
        return object()

    def _capture_compute(*, environ, artifact_config_path=None, config=None):
        """
        Capture compute wiring inputs and return a sentinel compute adapter.

        Args:
            environ: Environment mapping forwarded by CLI wiring.
            artifact_config_path: Explicit artifact-config path forwarded by CLI wiring.
            config: Optional preloaded config, unused in this test.
        Returns:
            object: Sentinel compute adapter.
        Assumptions:
            Offline compute construction itself is outside this test's scope.
        Raises:
            None.
        Side Effects:
            Stores call arguments in memory.
        """
        del config
        captured["compute_artifact_config_path"] = artifact_config_path
        captured["compute_environ"] = dict(environ)
        return object()

    def _capture_grid_builder(*, registry):
        """
        Capture grid-builder input and return a sentinel grid builder.

        Args:
            registry: Registry instance forwarded by CLI wiring.
        Returns:
            object: Sentinel grid builder.
        Assumptions:
            The grid-builder constructor is not under test here.
        Raises:
            None.
        Side Effects:
            Stores the registry object in memory.
        """
        captured["grid_builder_registry"] = registry
        return object()

    monkeypatch.setattr(
        publish_module,
        "load_backtest_artifacts_runtime_config",
        lambda path: _FakeArtifactRuntimeConfig(),
    )
    monkeypatch.setattr(
        publish_module,
        "BacktestArtifactPathBuilderV2",
        lambda root: object(),
    )
    monkeypatch.setattr(
        publish_module,
        "YamlBacktestArtifactLoaderV2",
        lambda path_resolver: object(),
    )
    monkeypatch.setattr(
        publish_module,
        "AtomicArtifactCurrentPointerWriterV2",
        lambda path_resolver: object(),
    )
    monkeypatch.setattr(
        publish_module,
        "ClickHouseSettingsLoader",
        _FakeClickHouseSettingsLoader,
    )
    monkeypatch.setattr(publish_module, "_clickhouse_client", lambda settings: object())
    monkeypatch.setattr(
        publish_module,
        "ClickHouseConnectGateway",
        lambda client: object(),
    )
    monkeypatch.setattr(publish_module, "SystemClock", lambda: object())
    monkeypatch.setattr(
        publish_module.YamlBacktestGridDefaultsProvider,
        "from_environ",
        classmethod(_capture_defaults_provider),
    )
    monkeypatch.setattr(publish_module, "build_indicators_registry", _capture_registry)
    monkeypatch.setattr(
        publish_module,
        "build_artifact_precompute_indicators_compute",
        _capture_compute,
    )
    monkeypatch.setattr(
        publish_module,
        "build_backtest_artifacts_runtime_config_hash",
        lambda config: "artifact-hash",
    )
    monkeypatch.setattr(
        publish_module,
        "ClickHouseCanonicalCandleReader",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        publish_module,
        "BacktestSignalRulesEngineV2",
        lambda defaults_provider: object(),
    )
    monkeypatch.setattr(publish_module, "GridBuilder", _capture_grid_builder)
    monkeypatch.setattr(
        publish_module,
        "BacktestArtifactPrecomputeRunnerV2",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        publish_module,
        "PsycopgBacktestPostgresGateway",
        lambda dsn: object(),
    )
    monkeypatch.setattr(
        publish_module,
        "PostgresBacktestJobRepository",
        lambda gateway: object(),
    )
    monkeypatch.setattr(
        publish_module,
        "BacktestArtifactSlotPublisherV2",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        publish_module,
        "ClickHouseCanonicalCandleIndexReader",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        publish_module,
        "PublishBacktestArtifactsV2UseCase",
        lambda **kwargs: object(),
    )

    result = publish_module._build_publish_use_case_v2(
        "configs/prod/backtest_artifacts.yaml",
        {"STRATEGY_PG_DSN": "postgresql://test"},
    )

    assert result is not None
    assert captured["defaults_artifact_config_path"] == "configs/prod/backtest_artifacts.yaml"
    assert captured["registry_artifact_config_path"] == "configs/prod/backtest_artifacts.yaml"
    assert captured["compute_artifact_config_path"] == "configs/prod/backtest_artifacts.yaml"


def _sample_publish_result_v2() -> PublishBacktestArtifactsV2Result:
    """
    Build one deterministic publish result payload for CLI rendering assertions.

    Args:
        None.
    Returns:
        PublishBacktestArtifactsV2Result: Stable result DTO with explicit validation summary.
    Assumptions:
        CLI rendering tests need representative scalar/list fields without real filesystem work.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/architecture/backtest/README.md
    Related:
      - apps/cli/commands/backtest_artifact_publish.py
    """
    coordinates = ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )
    return PublishBacktestArtifactsV2Result(
        status="succeeded",
        publish_mode="full_rebuild",
        coordinates=coordinates,
        previous_active_slot="slot_a",
        previous_slot_generation=1,
        previous_manifest_sha256="1" * 64,
        published_active_slot="slot_b",
        published_slot_generation=2,
        published_manifest_sha256="2" * 64,
        asof_date="2026-03-28",
        published_at_utc=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        requested_start_utc="2026-03-26T00:00:00Z",
        requested_end_utc="2026-03-29T00:00:00Z",
        source_start_utc="2026-03-26T00:00:00Z",
        source_end_utc="2026-03-29T00:00:00Z",
        source_candle_count=4320,
        reused_prefix_bars=4300,
        rewritten_tail_bars=20,
        blocking_active_run_count=0,
        validation=PublishBacktestArtifactsV2ValidationSummary(
            slot_manifest_path=None,
            manifest_sha256="2" * 64,
            price_timeframes=("1m", "15m"),
            mapping_timeframes=("15m",),
            signal_artifacts=(("15m", "ma.ema"),),
            signal_manifest_count=1,
            hit_times_manifest_present=True,
            funding_coverage_status=None,
            funding_manifest_hash=None,
            diagnostics_count=0,
        ),
        stage_rebuild_stats=ArtifactStageRebuildStatsCollectionV2(
            prices=ArtifactStageRebuildStatsV2(reused_prefix_bars=4300, rewritten_tail_bars=20),
            mappings=ArtifactStageRebuildStatsV2(reused_prefix_bars=95, rewritten_tail_bars=3),
            signals=ArtifactStageRebuildStatsV2(reused_prefix_bars=94, rewritten_tail_bars=2),
            hit_times=ArtifactStageRebuildStatsV2(
                reused_prefix_bars=4290,
                rewritten_tail_bars=20,
            ),
        ),
        tail_rebuild_bars=ArtifactTailRebuildBarsV2(
            prices=20,
            mappings=3,
            signals=2,
            hit_times=20,
        ),
    )
