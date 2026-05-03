from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKTEST_HISTORY_TEMPLATE = (
    REPO_ROOT / "apps" / "web" / "templates" / "pages" / "backtests_history.html"
)
BACKTEST_RUN_TEMPLATE = REPO_ROOT / "apps" / "web" / "templates" / "pages" / "backtests_run.html"
BACKTEST_RESULT_TEMPLATE = (
    REPO_ROOT / "apps" / "web" / "templates" / "pages" / "backtests_result.html"
)
BACKTEST_HISTORY_ASSET = (
    REPO_ROOT / "apps" / "web" / "dist" / "js" / "pages" / "backtests_history.js"
)
BACKTEST_RUN_ASSET = REPO_ROOT / "apps" / "web" / "dist" / "js" / "pages" / "backtests_run.js"
BACKTEST_RESULT_ASSET = (
    REPO_ROOT / "apps" / "web" / "dist" / "js" / "pages" / "backtests_result.js"
)


def test_backtest_template_exposes_required_public_api_hooks() -> None:
    """
    Verify the split SSR pages expose stable hooks for Stage 8 browser modules.
    """
    history_template = BACKTEST_HISTORY_TEMPLATE.read_text(encoding="utf-8")
    run_template = BACKTEST_RUN_TEMPLATE.read_text(encoding="utf-8")

    history_literals = [
        "/api/backtests/jobs",
        "/api/ui/backtests/counters",
        "data-backtests-history-page",
        "/assets/js/pages/backtests_history.js",
    ]
    run_literals = [
        "/backtests",
        "/api/backtests/runtime-defaults",
        "/api/backtests/preflight",
        "/api/backtests/jobs",
        "/api/market-data/markets",
        "/api/market-data/instruments",
        "/api/indicators",
        "data-backtests-run-page",
        "/assets/js/pages/backtests_run.js",
        "backtest_presets",
    ]
    for literal in history_literals:
        assert literal in history_template
    for literal in run_literals:
        assert literal in run_template
    assert "/assets/backtest_ui.js" not in history_template
    assert "/assets/backtest_ui.js" not in run_template
    assert "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades" not in run_template


def test_backtest_history_and_run_assets_keep_stage_8_boundaries() -> None:
    """
    Verify Stage 8 modules use cursor history and idempotent create without result payloads.
    """
    history_asset = BACKTEST_HISTORY_ASSET.read_text(encoding="utf-8")
    run_asset = BACKTEST_RUN_ASSET.read_text(encoding="utf-8")

    assert "cursor" in history_asset
    assert "createPoller" in history_asset
    assert "documentRef.hidden" not in history_asset
    assert "${paths.jobs}/${encodeURIComponent(jobId)}/cancel" in history_asset
    assert "/top" not in history_asset
    assert "/trades" not in history_asset

    assert "\"Idempotency-Key\"" in run_asset
    assert "paths.preflight" in run_asset
    assert "paths.jobs" in run_asset
    assert "globalThis.crypto.randomUUID" in run_asset
    assert "/top" not in run_asset
    assert "/trades" not in run_asset
    assert "backtest_chart_overlay_v1" not in run_asset
    assert "hit_times/1m" not in run_asset


def test_backtest_result_template_exposes_stage_9_bounded_endpoint_hooks() -> None:
    template = BACKTEST_RESULT_TEMPLATE.read_text(encoding="utf-8")

    required_literals = [
        "/api/backtests/jobs/{{ job_id }}/summary",
        "/api/backtests/jobs/{{ job_id }}/variants/{variant_key}/equity?points=1200",
        "/api/backtests/jobs/{{ job_id }}/variants/{variant_key}/drawdown?points=1200",
        "/api/backtests/jobs/{{ job_id }}/variants/{variant_key}/monthly-stats",
        "/api/backtests/jobs/{{ job_id }}/variants/{variant_key}/symbol-stats",
        "/api/backtests/jobs/{{ job_id }}/variants/{variant_key}/trades",
        "/api/backtests/jobs/{{ job_id }}/variants/{variant_key}/trades.csv",
        "data-backtest-result-page",
        "/assets/js/pages/backtests_result.js",
        "/assets/css/pages/backtests.css",
    ]
    for literal in required_literals:
        assert literal in template
    assert '"trades":' not in template


def test_backtest_result_asset_uses_server_pagination_and_public_variant_key() -> None:
    asset = BACKTEST_RESULT_ASSET.read_text(encoding="utf-8")

    assert "apiRequest(paths.summary" in asset
    assert "variant_key: variantKey" in asset
    assert "variant_hash: variantHash" not in asset
    assert "page_size=${TRADES_PAGE_SIZE}" in asset
    assert ".slice(startIndex" not in asset
    assert "paths.csvTemplate" in asset
    assert "drawTimeSeries" in asset
