from __future__ import annotations

from pathlib import Path


def _read_backtest_runs_ui_asset() -> str:
    """
    Read runs browser UI asset from repository for static behavior assertions.

    Args:
        None.
    Returns:
        str: Full JavaScript source code used by history and run summary pages.
    Assumptions:
        Asset is committed at `apps/web/dist/backtest_runs_ui.js`.
    Raises:
        OSError: If file cannot be read from workspace.
    Side Effects:
        Reads source file from local filesystem.
    """
    return Path("apps/web/dist/backtest_runs_ui.js").read_text(encoding="utf-8")


def test_backtest_runs_ui_asset_uses_public_runs_history_contract() -> None:
    """
    Verify runs UI asset loads history through public `/api/backtests/runs` contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R9-02 history page must use runs vocabulary rather than legacy jobs vocabulary.
    Raises:
        AssertionError: If required runs-history literals disappear from the asset.
    Side Effects:
        None.
    """
    source = _read_backtest_runs_ui_asset()

    assert "data-backtest-runs-page" in source
    assert "apiRunsPath" in source
    assert "apiRunsPathPrefix" in source
    assert "next_cursor" in source
    assert "base64url(json)" not in source
    assert "cursor" in source
    assert "renderPathTemplate" in source
    assert "run_id" in source
    assert "execution_mode" in source
    assert "execution_profile_mode" in source
    assert "requested_top_n" in source
    assert "progress_percent" in source
    assert "eta_seconds" in source


def test_backtest_runs_ui_asset_reads_runtime_defaults_sortable_columns() -> None:
    """
    Verify run summary asset reads runtime defaults sortable-columns contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Local resort options are limited to approved `contracts.summary.sortable_columns`.
    Raises:
        AssertionError: If runtime-defaults or sortable-columns wiring disappears.
    Side Effects:
        None.
    """
    source = _read_backtest_runs_ui_asset()

    assert "apiRuntimeDefaultsPath" in source
    assert "sortable_columns" in source
    assert "buildSortableColumnsFromRuntime" in source
    assert "server_order" in source
    assert "formatEtaSeconds" in source
    assert "readProgressPercent" in source


def test_backtest_runs_ui_asset_adds_summary_actions_for_detail_and_strategy_prefill() -> None:
    """
    Verify run summary asset exposes row actions for dedicated detail page and strategy prefill.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R9-03 moves heavy detail to a separate page while save flow reuses `sessionStorage`.
    Raises:
        AssertionError: If detail/save route literals disappear from the shared runs asset.
    Side Effects:
        None.
    """
    source = _read_backtest_runs_ui_asset()

    assert "Open detail" in source
    assert "Save as Strategy" in source
    assert "detailPathTemplate" in source
    assert "{variant_key}" in source
    assert "prefill" in source
    assert "prefillStorage" in source
    assert "strategyBuilderPath" in source
    assert "buildStrategyPrefillPayload" in source
    assert "persistStrategyPrefillAndNavigate" in source
    assert "Detail, chart, and trades stay on the dedicated variant page only." in source


def test_backtest_runs_ui_asset_applies_local_sort_without_top_refetch() -> None:
    """
    Verify run summary asset sorts loaded rows locally with deterministic variant-key fallback.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Local resort must not trigger `/top` recompute and tie-break stays `variant_key ASC`.
    Raises:
        AssertionError: If local-sort implementation literals drift from the contract.
    Side Effects:
        None.
    """
    source = _read_backtest_runs_ui_asset()

    assert "LOCAL_SORT_NOTE" in source
    assert "Local sort reorders loaded summary rows only." in source
    assert "state.topRowsOriginal.slice().sort(compareTopRows)" in source
    assert "variant_key ASC" in source
    assert "summary_metrics_json" in source


def test_backtest_runs_ui_asset_variant_detail_is_summary_only_without_report_endpoint() -> None:
    """
    Verify shared runs asset renders variant detail from persisted summary rows only.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Variant detail page no longer calls report recompute endpoints.
    Raises:
        AssertionError: If removed report-endpoint literals are present in the asset.
    Side Effects:
        None.
    """
    source = _read_backtest_runs_ui_asset()

    assert "variant-summary-metrics" in source
    assert "variant-payload-json" in source
    assert "summary_metrics_json" in source
    assert "/api/backtests/variant-report" not in source
    assert "/api/backtests/runs/{run_id}/variant-report" not in source
    assert "include_trades" not in source
