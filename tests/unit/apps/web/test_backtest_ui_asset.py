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


def test_backtest_ui_asset_uses_runtime_defaults_for_timeframes_ranking_and_top_n_mapping() -> None:
    """
    Verify UI reads runtime defaults contracts for request_timeframes, ranking_metrics, and top_n.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R9-01 launch form is driven by `/api/backtests/runtime-defaults` contract fields.
    Raises:
        AssertionError: If runtime-defaults-driven literals disappear from the UI asset.
    Side Effects:
        None.
    """
    source = _read_backtest_ui_asset()

    assert "request_timeframes" in source
    assert "ranking_metrics" in source
    assert "top_n_default" in source
    assert "top_n_max" in source
    assert "requestPayload.top_k = advanced.topN" in source
    assert "const cappedTopN = Math.min(parsedTopN, topNMax);" in source


def test_backtest_ui_asset_supports_multi_source_selection_and_explicit_axis_payload() -> None:
    """
    Verify source axis supports multiple values and serializes explicit ordered source values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R9-01 allows selecting several `inputs.source` values per indicator grid.
    Raises:
        AssertionError: If source UI stops being multi-select or payload loses
            explicit values array.
    Side Effects:
        None.
    """
    source = _read_backtest_ui_asset()

    assert "source values" in source
    assert "sourceSelect.multiple = true;" in source
    assert "Use Cmd/Ctrl to select multiple inputs.source values." in source
    assert (
        "rawSelectedValues: Array.from(sourceSelect.selectedOptions).map((item) => item.value)"
        in source
    )
    assert "values: selectedSourceValues" in source


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
    assert "request_timeframes=" in source
    assert "ranking_metrics=" in source


def test_backtest_ui_asset_surfaces_background_auto_202_and_toggles_risk_visibility() -> None:
    """
    Verify UI shows explicit `202 Accepted` background_auto fallback and toggles SL/TP controls.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R9-01 keeps explicit launch-outcome messaging and risk visibility helpers in browser asset.
    Raises:
        AssertionError: If background_auto or risk visibility literals disappear from the script.
    Side Effects:
        None.
    """
    source = _read_backtest_ui_asset()

    assert "202 Accepted." in source
    assert "background_auto" in source
    assert "execution_mode=sync_inline." in source
    assert "updateRiskUiVisibility" in source
    assert "toggleNodesVisibility" in source
