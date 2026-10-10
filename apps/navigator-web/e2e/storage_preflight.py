"""Native disposable NPY input assessment for S08 browser proof, no provider."""

import hashlib
import inspect
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import yaml

from tests.unit.contexts.backtest.application.services.v2 import (
    test_backtest_preflight_service as tests,
)
from tests.unit.contexts.backtest.application.services.v2 import (
    test_derived_artifact_materialization as native,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs.artifact_array_loader import (
    FilesystemBacktestArtifactArrayLoader,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs.artifact_context_resolver import (
    FilesystemBacktestArtifactContextResolver,
)
from trading.contexts.backtest.application.services.v2.preflight import BacktestPreflightService


def setup_preflight(directory: Path):
    def mktemp(name):
        path = directory / name
        path.mkdir()
        return path

    fixture, runner, snapshot, _ = inspect.unwrap(native.built)(SimpleNamespace(mktemp=mktemp))
    path = fixture.loader.resolve_slot_manifest_path(fixture.coordinates, snapshot.slot)
    manifest = yaml.safe_load(path.read_text())
    manifest["asof_date"] = (
        datetime.fromtimestamp(
            max(
                p.coverage.close_time_end
                for p in fixture.loader.load_slot_manifest(
                    fixture.coordinates, snapshot.slot
                ).prices
            )
            / 1000,
            UTC,
        )
        .date()
        .isoformat()
    )
    manifest["schema_version"] = 2
    manifest.pop("hit_times", None)
    manifest["signals"]["manifests"] = []
    path.write_text(yaml.safe_dump(manifest))
    fixture.loader.resolve_current_pointer_path(fixture.coordinates).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "active_slot": snapshot.slot,
                "slot_generation": snapshot.generation,
                "asof_date": manifest["asof_date"],
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "published_at_utc": "2026-03-25T02:00:00Z",
            }
        )
    )
    service = BacktestPreflightService(
        defaults_provider=runner.defaults_provider,
        artifact_context_resolver=FilesystemBacktestArtifactContextResolver(
            artifact_loader=fixture.loader
        ),
        runtime_config=replace(
            tests._runtime_config(),
            artifact_config_hash=runner.runtime_settings.config_sha256,
            hit_times_tp_levels_pct=runner.runtime_settings.hit_times_tp_levels_pct,
            hit_times_sl_levels_pct=runner.runtime_settings.hit_times_sl_levels_pct,
        ),
        artifact_array_loader=FilesystemBacktestArtifactArrayLoader(artifact_loader=fixture.loader),
        indicator_grid_builder=runner.indicator_grid_builder,
    )
    request = tests._valid_request()
    request["coordinates"] = {
        "exchange": fixture.coordinates.exchange,
        "market_type": fixture.coordinates.market_type,
        "symbol": fixture.coordinates.symbol,
    }
    request["time_range"] = {
        "start": snapshot.consumed_domains[0].origin_utc[:10] + "T00:00:00Z",
        "end": manifest["asof_date"] + "T00:00:00Z",
    }
    request["indicators"] = [
        {
            "indicator_id": "ma.ema",
            "sources": ["close"],
            "window": {"start": 21, "stop": 21, "step": 1},
        }
    ]
    return service, request, fixture.loader
