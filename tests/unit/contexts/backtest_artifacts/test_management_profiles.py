from copy import deepcopy
from datetime import UTC, date, datetime, time
from pathlib import Path

import pytest

from trading.contexts.backtest_artifacts.adapters.outbound.config.backtest_artifacts_runtime_config import (  # noqa: E501
    load_backtest_artifacts_runtime_config,
)
from trading.contexts.backtest_artifacts.adapters.outbound.config.legacy_management_import import (
    legacy_demand,
)
from trading.contexts.backtest_artifacts.adapters.outbound.defaults import (
    YamlBacktestGridDefaultsProvider,
)
from trading.contexts.backtest_artifacts.adapters.outbound.defaults.management_catalog import (
    ManagementCatalog,
)
from trading.contexts.backtest_artifacts.application.services.management import (
    Coordinate,
    Grid,
    History,
    Profile,
    Schedule,
)
from trading.contexts.indicators.adapters.outbound.registry import YamlIndicatorRegistry
from trading.contexts.indicators.domain.definitions import all_defs


@pytest.fixture(scope="module")
def catalog():
    config = load_backtest_artifacts_runtime_config("configs/dev/backtest_artifacts.yaml")
    return ManagementCatalog(
        defaults=YamlBacktestGridDefaultsProvider.from_yaml(
            config_path=Path("configs/dev/indicators.yaml")
        ),
        registry=YamlIndicatorRegistry.from_yaml(
            defs=all_defs(), config_path="configs/dev/indicators.yaml"
        ),
        config=config,
    )


def test_all_selected_defaults_and_stable_revision(catalog):
    profile = catalog.default_profile()
    assert len(profile.indicators) == 40
    assert len(profile.derivative_timeframes) == 10
    assert profile.mode == "on_demand"
    assert len(profile.tp_levels_pct) == 100
    assert len(profile.sl_levels_pct) == 50
    assert catalog.validate(profile).catalog_revision == catalog.revision


def test_empty_selection_and_sparse_independent_risk(catalog):
    payload = catalog.default_profile().model_dump(mode="json")
    payload.update(
        indicators=[], derivative_timeframes=[], tp_levels_pct=[0.5, 2], sl_levels_pct=[]
    )
    assert catalog.validate(Profile.model_validate(payload)).indicators == ()
    payload["indicators"] = [catalog.default_profile().indicators[0].model_dump(mode="json")]
    with pytest.raises(ValueError, match="profile_timeframes_required"):
        Profile.model_validate(payload)


@pytest.mark.parametrize(
    "mutation", ["unknown", "catalog", "risk", "params", "source", "extra", "duplicate"]
)
def test_invalid_profile_rejected(catalog, mutation):
    payload = deepcopy(catalog.default_profile().model_dump(mode="json"))
    if mutation == "unknown":
        payload["indicators"][0]["indicator_id"] = "foreign.unknown"
    if mutation == "catalog":
        payload["catalog_revision"] = "0" * 64
    if mutation == "risk":
        payload["tp_levels_pct"] = [0.51]
    if mutation == "params":
        payload["indicators"][0]["params"]["window"]["values"] = [999999]
    if mutation == "source":
        payload["indicators"][0]["sources"] = ["foreign"]
    if mutation == "extra":
        payload["base_enabled"] = False
    if mutation == "duplicate":
        payload["indicators"].append(payload["indicators"][0])
    with pytest.raises(ValueError):
        catalog.validate(Profile.model_validate(payload))


@pytest.mark.parametrize("days", [0, 2, 4, 0.5, True, "1"])
def test_cadence_rejects_other_values(days):
    with pytest.raises(ValueError):
        Schedule(interval_days=days, timezone="UTC", local_time=time(2), anchor=date(2026, 1, 1))


@pytest.mark.parametrize("days", [1, 3, 7])
def test_cadence_presets(days):
    assert (
        Schedule(
            interval_days=days,
            timezone="Europe/Moscow",
            local_time=time(2),
            anchor=date(2026, 1, 1),
        ).interval_days
        == days
    )


def test_history_and_grid_bounds():
    with pytest.raises(ValueError):
        History(kind="range", start_utc=datetime(2026, 1, 1), end_utc=datetime(2026, 2, 1))
    with pytest.raises(ValueError):
        History(
            kind="range",
            start_utc=datetime(2026, 2, 1, tzinfo=UTC),
            end_utc=datetime(2026, 1, 1, tzinfo=UTC),
        )
    with pytest.raises(ValueError):
        Grid(mode="range", start=0, stop_incl=1e12, step=0.0001).expand()
    with pytest.raises(ValueError):
        Grid(mode="explicit", values=(float("nan"),)).expand()


def test_legacy_import_preserves_implicit_precompute():
    config = load_backtest_artifacts_runtime_config("configs/dev/backtest_artifacts.yaml")
    _, config_hash, snapshot = legacy_demand(
        config,
        Coordinate(exchange="binance", market_type="spot", symbol="BTCUSDT"),
        defaults=YamlBacktestGridDefaultsProvider.from_yaml(
            config_path="configs/dev/indicators.yaml"
        ),
    )
    assert len(config_hash) == 64
    assert snapshot["signals"] == snapshot["hit_times"] == "precompute"
    assert len(snapshot["signal_artifacts"]) == 400


@pytest.mark.parametrize("value", [True, float("inf"), float("nan")])
def test_boolean_and_nonfinite_grid_values_rejected(value):
    with pytest.raises(ValueError):
        Grid.model_validate({"mode": "explicit", "values": [value]})


def test_boolean_risk_rejected(catalog):
    payload = catalog.default_profile().model_dump(mode="json")
    payload["tp_levels_pct"] = [True]
    with pytest.raises(ValueError):
        Profile.model_validate(payload)


def test_signal_rule_parameters_are_snapshotted_and_bounded(catalog):
    payload = catalog.default_profile().model_dump(mode="json")
    rsi = next(i for i in payload["indicators"] if i["indicator_id"] == "momentum.rsi")
    assert rsi["signal_params"]["long_threshold"]["values"] == [30]
    catalog.validate(Profile.model_validate(payload))
    rsi["signal_params"]["long_threshold"]["values"] = [31]
    with pytest.raises(ValueError, match="parameter_outside_catalog"):
        catalog.validate(Profile.model_validate(payload))
