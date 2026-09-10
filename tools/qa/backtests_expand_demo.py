"""Expand the existing disposable preview; preserve databases and published active slot.

Run only against tools.qa.backtests_client_fixture state. Produces synthetic prices,
not exchange history. API/runner must reload the returned local indicator config.
"""

import json
import math
import tempfile
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import yaml

from tools.qa.backtests_client_fixture import ROOT, STATE
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp


def synthetic_rows(build_rows):
    """Thirty days of deterministic oscillations and trend, with valid OHLC."""

    def price(i):
        return (
            60000
            + 2500 * math.sin(i * 2 * math.pi / 720)
            + 900 * math.sin(i * 2 * math.pi / 179)
            + i * 0.035
        )

    return tuple(
        replace(
            row,
            candle=replace(
                row.candle,
                ts_open=UtcTimestamp(row.candle.ts_open.value - timedelta(days=27)),
                ts_close=UtcTimestamp(row.candle.ts_close.value - timedelta(days=27)),
                open=price(i),
                close=price(i + 1),
                high=max(price(i), price(i + 1)) + 35,
                low=min(price(i), price(i + 1)) - 35,
                volume_base=20 + 5 * math.sin(i / 17),
            ),
        )
        for i, row in enumerate(build_rows(bar_indexes=tuple(range(43200))))
    )


def expand_artifacts(env: dict[str, str], dsn: str, *, canonical_rows=None) -> Path:
    """Build real artifact files from deterministic, disposable canonical input candles."""
    from datetime import UTC, datetime

    from apps.api.wiring.modules.indicators import (
        build_indicators_compute,
        build_indicators_registry,
    )
    from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
        build_artifact_precompute_fixture_v2,
    )
    from tests.unit.contexts.backtest.application.services.v2.test_artifact_precompute_runner_v2 import (  # noqa: E501
        _build_canonical_rows_v2,
        _FakeCanonicalCandleReader,
        _request_v2,
    )
    from trading.contexts.backtest.adapters.outbound import (
        PostgresBacktestJobRepository,
        PsycopgBacktestPostgresGateway,
        YamlBacktestGridDefaultsProvider,
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
        AtomicArtifactCurrentPointerWriterV2,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_precompute_runner import (  # noqa: E501
        BacktestArtifactPrecomputeRunnerV2,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_publisher import (  # noqa: E501
        BacktestArtifactSlotPublisherV2,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.signal_rules_engine_v2 import (
        BacktestSignalRulesEngineV2,
    )
    from trading.contexts.indicators.application.services import GridBuilder

    staging = Path(tempfile.mkdtemp(prefix="expanded-demo-", dir=STATE))
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=staging,
        hit_times_tp_levels_pct=(1.0, 2.0),
        hit_times_sl_levels_pct=(1.0, 2.0),
        validation_price_timeframes=("1m", "15m"),
        validation_mapping_timeframes=("15m",),
        validation_signal_artifacts=(("15m", "ma.ema"),),
        precompute_signal_artifacts=(("15m", "ma.ema"),),
        require_hit_times_manifest=True,
        signal_worker_processes=1,
    )
    from dataclasses import replace

    from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
        BacktestArtifactPathBuilderV2,
    )
    from trading.contexts.backtest.adapters.outbound.config import (
        build_backtest_artifacts_runtime_config_hash,
        load_backtest_artifacts_runtime_config,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (  # noqa: E501
        YamlBacktestArtifactLoaderV2,
    )

    live_config = load_backtest_artifacts_runtime_config(STATE / "backtest_artifacts.yaml")
    builder = BacktestArtifactPathBuilderV2(root=STATE / "artifacts/backtest/v2")
    fixture = replace(
        fixture,
        builder=builder,
        loader=YamlBacktestArtifactLoaderV2(path_resolver=builder),
        runtime_config=live_config,
        runtime_settings=replace(
            fixture.runtime_settings,
            config_sha256=build_backtest_artifacts_runtime_config_hash(config=live_config),
        ),
    )
    defaults = YamlBacktestGridDefaultsProvider.from_environ(environ=env)
    # Only the input candle reader is synthetic. Computation, manifests, validation,
    # publication and the job-blocking check use production implementations.
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_FakeCanonicalCandleReader(
            rows=(synthetic_rows(_build_canonical_rows_v2)
                  if canonical_rows is None else canonical_rows)
        ),
        defaults_provider=defaults,
        signal_rules_engine=BacktestSignalRulesEngineV2(defaults_provider=defaults),
        indicator_compute=build_indicators_compute(environ=env),
        indicator_grid_builder=GridBuilder(registry=build_indicators_registry(environ=env)),
    )
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=fixture.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=fixture.builder),
        job_repository=PostgresBacktestJobRepository(
            gateway=PsycopgBacktestPostgresGateway(dsn=dsn)
        ),
        now_provider=lambda: datetime(2026, 3, 29, 3, 0, tzinfo=UTC),
    )
    precheck = publisher.precheck_publish(fixture.coordinates)
    runner.export_canonical_price_1m(
        replace(
            _request_v2(
                fixture=fixture,
                end_minute=4320,
                asof_date="2026-03-29",
                generated_at_utc="2026-03-29T03:00:00Z",
            ),
            time_range=TimeRange(
                start=UtcTimestamp(datetime(2026, 2, 27, tzinfo=UTC)),
                end=UtcTimestamp(datetime(2026, 3, 29, tzinfo=UTC)),
            ),
            force_full_rebuild=True,
        )
    )
    publisher.publish(
        precheck=precheck,
        validation_spec=fixture.runtime_config.to_validation_spec(),
        asof_date="2026-03-29",
    )
    print(
        ("Synthetic" if canonical_rows is None else "Supplied canonical")
        + " 43200 candles / 10 EMA windows -> "
        "production precompute/validate/local publish: passed",
        flush=True,
    )
    return STATE / "backtest_artifacts.yaml"


if __name__ == "__main__":
    config = yaml.safe_load((ROOT / "configs/test/indicators.yaml").read_text())
    config["defaults"]["ma.ema"]["params"]["window"]["values"] = list(range(5, 51, 5))
    local = STATE / "expanded-indicators.yaml"
    local.write_text(yaml.safe_dump(config))
    env = {
        "ROEHUB_ENV": "test",
        "ROEHUB_INDICATORS_CONFIG": str(local),
        "NUMBA_NUM_THREADS": "1",
        "ROEHUB_NUMBA_NUM_THREADS": "1",
    }
    private = json.loads((STATE / "credentials.json").read_text())
    expand_artifacts(env, private["dsn"])
