from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, cast

import apps.cli.main.main as cli_main_module
from apps.cli.commands.backtest_artifact_publish import BacktestArtifactPublishCli
from trading.contexts.backtest.application.services import (
    ArtifactCoordinatesV2,
    ArtifactStageRebuildStatsCollectionV2,
    ArtifactStageRebuildStatsV2,
    ArtifactTailRebuildBarsV2,
)
from trading.contexts.backtest.application.use_cases import (
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
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
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
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
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
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
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
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
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
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
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
