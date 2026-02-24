from __future__ import annotations

from pathlib import Path


def _read_backtest_ui_asset() -> str:
    """
    Read backtests browser UI asset from repository for static behavior assertions.

    Args:
        None.
    Returns:
        str: Full JavaScript source code used by `/backtests` page.
    Assumptions:
        Asset is committed at `apps/web/dist/backtest_ui.js`.
    Raises:
        OSError: If file cannot be read from workspace.
    Side Effects:
        Reads source file from local filesystem.
    """
    return Path("apps/web/dist/backtest_ui.js").read_text(encoding="utf-8")


def test_backtest_ui_asset_supports_param_axis_modes_and_ma_window_labels() -> None:
    """
    Verify indicator params UI supports explicit/range mode toggles and MA window labels.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        PR1 keeps mode options in front-end code as explicit string literals.
    Raises:
        AssertionError: If axis modes or MA window labels disappear from UI asset.
    Side Effects:
        None.
    """
    source = _read_backtest_ui_asset()

    assert "[\"explicit\", \"range\"]" in source
    assert "mode: \"range\"" in source
    assert "mode: \"explicit\"" in source
    assert "window period" in source
    assert "window grid step" in source


def test_backtest_ui_asset_uses_single_select_source_with_explicit_axis_payload() -> None:
    """
    Verify source axis is rendered as single-select and serialized as explicit one-value axis.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Source input remains part of indicator grid contract (`BacktestAxisSpecRequest`).
    Raises:
        AssertionError: If source UI is not select-based or payload is not explicit single-value.
    Side Effects:
        None.
    """
    source = _read_backtest_ui_asset()

    assert "source-select" in source
    assert "const sourceSelect = document.createElement(\"select\");" in source
    assert "source requires one selected value" in source
    assert "values: [selectedSource]" in source


def test_backtest_ui_asset_fetches_runtime_defaults_and_tracks_fee_dirty_state() -> None:
    """
    Verify UI loads `/api/backtests/runtime-defaults` and avoids fee overwrite after manual edits.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Runtime defaults endpoint path and dirty-tracking literals remain stable in source.
    Raises:
        AssertionError: If runtime-defaults fetch or fee dirty logic is missing.
    Side Effects:
        None.
    """
    source = _read_backtest_ui_asset()

    assert "apiBacktestRuntimeDefaultsPath" in source
    assert "loadRuntimeDefaults" in source
    assert "executionFeeDirty" in source
    assert "applyDefaultFeeForSelectedMarket" in source
    assert "Runtime defaults:" in source
    assert "top_k_persisted_default" in source


def test_backtest_ui_asset_formats_preflight_memory_and_toggles_risk_visibility() -> None:
    """
    Verify UI shows human-readable memory and hides/shows SL/TP controls by enabled/mode.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        PR2 keeps visibility and memory formatting helpers in browser asset.
    Raises:
        AssertionError: If memory/risk visibility literals disappear from the script.
    Side Effects:
        None.
    """
    source = _read_backtest_ui_asset()

    assert "formatEstimatedMemory" in source
    assert "GiB (base-2)" in source
    assert "MiB (base-2)" in source
    assert "updateRiskUiVisibility" in source
    assert "toggleNodesVisibility" in source
