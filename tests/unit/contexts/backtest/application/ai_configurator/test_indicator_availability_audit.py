from __future__ import annotations

from typing import Any, cast

from tests.unit.contexts.backtest.application.ai_configurator.test_context_snapshot import (
    _builder,
    _summary_hash,
    _summary_payload,
)


def test_indicator_availability_audit_classifies_all_prod_indicators() -> None:
    builder = _builder(summary=_summary_payload())

    snapshot = builder.build(user_message="Собери конфиг для BTCUSDT")

    assert snapshot.indicator_audit["total_indicators"] == 40
    assert snapshot.indicator_audit["available_count"] == 40
    assert snapshot.indicator_audit["excluded_count"] == 0
    assert snapshot.indicator_audit["excluded_indicators"] == []
    assert len(snapshot.indicator_audit["available_indicator_ids"]) == 40


def test_indicator_availability_audit_excludes_missing_summary_coverage() -> None:
    summary = cast(dict[str, Any], _summary_payload())
    instruments = cast(dict[str, Any], summary["instruments"])
    instrument = cast(dict[str, Any], instruments["binance/spot/BTCUSDT"])
    timeframes = cast(dict[str, Any], instrument["timeframes"])
    timeframe_1h = cast(dict[str, Any], timeframes["1h"])
    timeframe_1h["indicator_ids"] = [
        indicator_id
        for indicator_id in timeframe_1h["indicator_ids"]
        if indicator_id != "structure.percent_rank"
    ]
    summary["summary_hash"] = _summary_hash(summary)
    builder = _builder(summary=summary)

    snapshot = builder.build(user_message="percent rank для BTCUSDT")

    excluded = snapshot.indicator_audit["excluded_indicators"]
    assert {
        "indicator_id": "structure.percent_rank",
        "reason": "missing_summary_coverage",
    } in excluded
