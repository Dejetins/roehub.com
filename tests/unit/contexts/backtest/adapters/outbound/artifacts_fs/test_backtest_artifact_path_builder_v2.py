from __future__ import annotations

import os
from pathlib import Path

import pytest

from trading.contexts.backtest.adapters.outbound import BacktestArtifactPathBuilderV2
from trading.contexts.backtest.application.services import ArtifactCoordinatesV2


def test_backtest_artifact_path_builder_v2_builds_canonical_paths(tmp_path: Path) -> None:
    """
    Verify the path builder returns the exact R2-01 canonical layout for every artifact family.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        The artifact root may be injected and path resolution must remain side-effect free.
    Raises:
        AssertionError: If one resolved path deviates from the documented R2-01 layout.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    root = tmp_path / "artifacts" / "backtest" / "v2"
    builder = BacktestArtifactPathBuilderV2(root=root)
    coordinates = ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )

    assert builder.symbol_root(coordinates) == root / "binance" / "spot" / "BTCUSDT"
    assert builder.current_pointer_path(coordinates) == (
        root / "binance" / "spot" / "BTCUSDT" / "current.yaml"
    )
    assert builder.slot_root(coordinates, "slot_a") == (
        root / "binance" / "spot" / "BTCUSDT" / "slot_a"
    )
    assert builder.slot_manifest_path(coordinates, "slot_b") == (
        root / "binance" / "spot" / "BTCUSDT" / "slot_b" / "manifest.yaml"
    )

    price_paths = builder.price_paths(coordinates, "slot_a", "1h")
    assert price_paths.open_time == (
        root / "binance" / "spot" / "BTCUSDT" / "slot_a" / "prices" / "1h" / "open_time.i64.npy"
    )
    assert price_paths.close_time == (
        root / "binance" / "spot" / "BTCUSDT" / "slot_a" / "prices" / "1h" / "close_time.i64.npy"
    )
    assert price_paths.ohlcv == (
        root / "binance" / "spot" / "BTCUSDT" / "slot_a" / "prices" / "1h" / "ohlcv.f32.npy"
    )

    signal_paths = builder.signal_paths(coordinates, "slot_b", "15m", "ma.ema")
    assert signal_paths.manifest == (
        root
        / "binance"
        / "spot"
        / "BTCUSDT"
        / "slot_b"
        / "signals"
        / "15m"
        / "ma.ema"
        / "manifest.yaml"
    )
    assert signal_paths.signals == (
        root
        / "binance"
        / "spot"
        / "BTCUSDT"
        / "slot_b"
        / "signals"
        / "15m"
        / "ma.ema"
        / "signals.i8.npy"
    )

    mapping_paths = builder.mapping_paths(coordinates, "slot_a", "2h")
    assert mapping_paths.bar_open_1m_idx == (
        root
        / "binance"
        / "spot"
        / "BTCUSDT"
        / "slot_a"
        / "mappings"
        / "2h"
        / "bar_open_1m_idx.u32.npy"
    )
    assert mapping_paths.bar_close_1m_idx == (
        root
        / "binance"
        / "spot"
        / "BTCUSDT"
        / "slot_a"
        / "mappings"
        / "2h"
        / "bar_close_1m_idx.u32.npy"
    )

    assert builder.hit_times_manifest_path(coordinates, "slot_b") == (
        root / "binance" / "spot" / "BTCUSDT" / "slot_b" / "hit_times" / "1m" / "manifest.yaml"
    )


def test_backtest_artifact_path_builder_v2_returns_byte_stable_paths(tmp_path: Path) -> None:
    """
    Verify repeated identical inputs produce byte-identical path strings and stable slot order.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Deterministic ordering is part of the runtime-facing contract for R2-01.
    Raises:
        AssertionError: If repeated path resolution changes output bytes or slot order.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    builder = BacktestArtifactPathBuilderV2(root=tmp_path / "artifacts" / "backtest" / "v2")
    coordinates = ArtifactCoordinatesV2(
        exchange="bybit",
        market_type="linear",
        symbol="ETHUSDT",
    )

    first = builder.signal_paths(coordinates, "slot_a", "30m", "momentum.rsi").signals
    second = builder.signal_paths(coordinates, "slot_a", "30m", "momentum.rsi").signals

    assert builder.ordered_slots() == ("slot_a", "slot_b")
    assert os.fspath(first).encode("utf-8") == os.fspath(second).encode("utf-8")


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda builder, coordinates: builder.slot_root(coordinates, "slot_c"),
            "artifact slot must be one of",
        ),
        (
            lambda builder, coordinates: builder.price_paths(coordinates, "slot_a", "5m"),
            "artifact price timeframe must be one of",
        ),
        (
            lambda builder, coordinates: builder.signal_paths(
                coordinates,
                "slot_a",
                "15m",
                "../ma.ema",
            ),
            "artifact indicator_id must be a non-empty safe token",
        ),
        (
            lambda builder, coordinates: ArtifactCoordinatesV2(
                exchange="binance",
                market_type="spot",
                symbol="../BTCUSDT",
            ),
            "artifact coordinate symbol must be a non-empty safe token",
        ),
    ],
)
def test_backtest_artifact_path_builder_v2_rejects_invalid_inputs(
    tmp_path: Path,
    factory: object,
    message: str,
) -> None:
    """
    Verify invalid slot, timeframe, and traversal tokens fail fast with stable messages.

    Args:
        tmp_path: pytest temporary path fixture.
        factory: Deferred call that should raise `ValueError`.
        message: Stable error-message fragment expected from the contract.
    Returns:
        None.
    Assumptions:
        The path builder must reject unsafe inputs before any filesystem access occurs.
    Raises:
        AssertionError: If invalid inputs are accepted or the failure message drifts.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    builder = BacktestArtifactPathBuilderV2(root=tmp_path / "artifacts" / "backtest" / "v2")
    coordinates = ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )

    with pytest.raises(ValueError, match=message):
        factory(builder, coordinates)  # type: ignore[misc]
