from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKTEST_TEMPLATE = REPO_ROOT / "apps" / "web" / "templates" / "backtests.html"
BACKTEST_ASSET = REPO_ROOT / "apps" / "web" / "dist" / "backtest_ui.js"


def test_backtest_template_exposes_required_public_api_hooks() -> None:
    """
    Verify the SSR page exposes stable hooks for the browser-side Backtest UI module.
    """
    template = BACKTEST_TEMPLATE.read_text(encoding="utf-8")

    required_literals = [
        "/backtests",
        "/api/backtests/runtime-defaults",
        "/api/backtests/preflight",
        "/api/backtests/jobs",
        "/api/backtests/jobs/{job_id}",
        "/api/backtests/jobs/{job_id}/top",
        "/api/backtests/jobs/{job_id}/variants/{variant_key}",
        "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades",
        "/api/backtests/jobs/{job_id}/cancel",
        "data-backtest-page",
        "/assets/backtest_ui.js",
    ]
    for literal in required_literals:
        assert literal in template


def test_backtest_ui_asset_uses_lazy_trades_variant_key_contract() -> None:
    """
    Verify the JS module keeps public `variant_key` as lazy-trades route identity.
    """
    asset = BACKTEST_ASSET.read_text(encoding="utf-8")

    assert "backtest_chart_overlay_v1" in asset
    assert "show trades" in asset
    assert "variant_key" in asset
    assert "variant_hash" in asset
    assert "renderBacktestPath(paths.tradesTemplate" in asset
    assert "variant_key: variantKey" in asset
    assert "variant_hash: variantHash" not in asset
    assert "TRADES_PAGE_SIZE = 25" in asset
    assert ".slice(startIndex, endIndex)" in asset
    assert "credentials: \"include\"" in asset
    assert "Idempotency-Key" not in asset
    assert "POST /backtests" not in asset
    assert "hit_times/1m" not in asset
