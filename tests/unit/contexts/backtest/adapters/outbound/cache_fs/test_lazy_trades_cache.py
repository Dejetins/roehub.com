from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from trading.contexts.backtest.adapters.outbound.cache_fs import (
    LocalFileBacktestLazyTradesCache,
)
from trading.contexts.backtest.application.ports import BacktestLazyTradesCacheKey


def test_lazy_trades_cache_views_report_hit_after_materialized_miss(
    tmp_path: Path,
) -> None:
    cache = LocalFileBacktestLazyTradesCache(root=tmp_path)
    cache_key = BacktestLazyTradesCacheKey(
        job_id="job-1",
        variant_key="variant-public",
        variant_hash="variant-hash",
        request_hash="r" * 64,
        engine_params_hash="e" * 64,
        artifact_manifest_hash="a" * 64,
        funding_manifest_hash="f" * 64,
    )
    now = datetime(2026, 5, 23, tzinfo=UTC)
    payload = {
        "job_id": "job-1",
        "variant_key": "variant-public",
        "variant_hash": "variant-hash",
        "request_hash": "r" * 64,
        "engine_params_hash": "e" * 64,
        "artifact_manifest_hash": "a" * 64,
        "funding_manifest_hash": "f" * 64,
        "summary_metrics": {"trade_count": 1},
        "canonical_variant_params": {},
        "readable_params": {},
        "trades": (
            {
                "trade_index": 1,
                "exit_timestamp": "2026-05-23T00:00:00Z",
                "equity_after": 101.0,
                "net_pnl_quote": 1.0,
                "return_pct": 1.0,
            },
        ),
        "chart_overlay": {
            "schema": "backtest_chart_overlay_v1",
            "funding_manifest_hash": "f" * 64,
            "funding_events": [
                {
                    "kind": "funding_event",
                    "trade_index": 1,
                    "funding_rate": 0.001,
                    "timestamp": "2026-05-23T00:00:00Z",
                }
            ],
            "funding_events_count": 1,
            "funding_events_truncated": False,
        },
        "funding": {"funding_manifest_hash": "f" * 64, "included": True},
        "cache": {"status": "miss", "ttl_seconds": 1_209_600},
        "timing": {},
    }

    cache.write(cache_key=cache_key, payload=payload, now=now, ttl_seconds=1_209_600)

    detail = cache.read(cache_key=cache_key, now=now, ttl_seconds=1_209_600)
    series = cache.read_series(
        cache_key=cache_key,
        now=now,
        ttl_seconds=1_209_600,
        kind="equity",
        points=10,
    )
    page = cache.read_page(
        cache_key=cache_key,
        now=now,
        ttl_seconds=1_209_600,
        page=1,
        page_size=10,
    )
    monthly = cache.read_monthly_stats(
        cache_key=cache_key,
        now=now,
        ttl_seconds=1_209_600,
    )

    assert series.payload is not None
    assert detail.payload is not None
    assert page.payload is not None
    assert monthly.payload is not None
    assert detail.payload["funding"]["funding_manifest_hash"] == "f" * 64
    assert detail.payload["chart_overlay"]["funding_events"][0]["kind"] == "funding_event"
    assert series.payload["cache"]["status"] == "hit"
    assert page.payload["cache"]["status"] == "hit"
    assert monthly.payload["cache"]["status"] == "hit"


def test_lazy_trades_cache_key_accounts_for_funding_manifest_hash() -> None:
    base = BacktestLazyTradesCacheKey(
        job_id="job-1",
        variant_key="variant-public",
        variant_hash="variant-hash",
        request_hash="r" * 64,
        engine_params_hash="e" * 64,
        artifact_manifest_hash="a" * 64,
        funding_manifest_hash="f" * 64,
    )
    changed = BacktestLazyTradesCacheKey(
        job_id="job-1",
        variant_key="variant-public",
        variant_hash="variant-hash",
        request_hash="r" * 64,
        engine_params_hash="e" * 64,
        artifact_manifest_hash="a" * 64,
        funding_manifest_hash="0" * 64,
    )

    assert base.as_mapping()["funding_manifest_hash"] == "f" * 64
    assert base.digest != changed.digest
