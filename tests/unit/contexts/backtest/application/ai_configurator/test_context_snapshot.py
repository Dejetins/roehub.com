from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest.application.ai_configurator.context_snapshot import (
    BacktestAiContextSnapshotBuilder,
    BacktestAiContextSnapshotUnavailable,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
)


def test_context_snapshot_uses_summary_for_symbol_timeframes_and_periods() -> None:
    builder = _builder(summary=_summary_payload(timeframes=("2h",)))

    snapshot = builder.build(user_message="Собери RSI для BTCUSDT")
    prompt_context = snapshot.model_prompt_context()

    assert prompt_context["allowed_values"]["symbol"] == ["BTCUSDT"]
    assert prompt_context["allowed_values"]["timeframe"] == ["2h"]
    assert prompt_context["period"] == {
        "start_date": "2020-01-10",
        "end_date": "2026-05-02",
    }
    assert prompt_context["timeframe_periods"]["2h"] == {
        "start_date": "2020-01-10",
        "end_date": "2026-05-02",
        "bars": 100,
    }
    assert "ETHUSDT" not in prompt_context["allowed_values"]["symbol"]


def test_context_snapshot_records_multi_symbol_request_as_first_symbol_and_ignored() -> None:
    builder = _builder(summary=_summary_payload(symbols=("BTCUSDT", "ETHUSDT")))

    snapshot = builder.build(user_message="Сделай RSI для BTCUSDT и ETHUSDT")

    assert snapshot.resolved_symbol == "BTCUSDT"
    assert snapshot.ignored_symbols == ("ETHUSDT",)
    assert snapshot.model_prompt_context()["ignored_symbols"] == ["ETHUSDT"]
    assert snapshot.warnings == (
        "multiple_symbol_request: using first symbol and recording ignored_symbols",
    )


def test_context_snapshot_preserves_explicit_percent_rank_axis_values() -> None:
    builder = _builder(summary=_summary_payload())

    snapshot = builder.build(user_message="percent rank для BTCUSDT")
    percent_rank = _indicator(snapshot.model_prompt_context(), "structure.percent_rank")

    assert percent_rank["window_axis"] == {
        "mode": "explicit",
        "values": [10, 14, 20, 28, 42, 56, 84, 126],
    }
    assert "start" not in percent_rank["window_axis"]


def test_context_snapshot_represents_no_window_axis() -> None:
    builder = _builder(summary=_summary_payload())

    snapshot = builder.build(user_message="OBV для BTCUSDT")
    obv = _indicator(snapshot.model_prompt_context(), "volume.obv")

    assert obv["window_axis"] == {"mode": "none"}
    assert obv["available"] is True


def test_context_snapshot_missing_summary_fails_closed() -> None:
    builder = _builder(summary=None)

    with pytest.raises(FileNotFoundError, match="availability_summary.yaml"):
        builder.build(user_message="RSI для BTCUSDT")


def test_context_snapshot_corrupt_summary_hash_fails_closed() -> None:
    summary = dict(_summary_payload())
    summary["summary_hash"] = "0" * 64
    builder = _builder(summary=summary)

    with pytest.raises(BacktestAiContextSnapshotUnavailable, match="summary_hash mismatch"):
        builder.build(user_message="RSI для BTCUSDT")


def _indicator(prompt_context: Mapping[str, Any], indicator_id: str) -> Mapping[str, Any]:
    for item in prompt_context["indicators"]:
        if item["indicator_id"] == indicator_id:
            return item
    raise AssertionError(f"indicator not found: {indicator_id}")


def _builder(
    *,
    summary: Mapping[str, Any] | None,
) -> BacktestAiContextSnapshotBuilder:
    defaults_provider = YamlBacktestGridDefaultsProvider.from_yaml(
        config_path="configs/prod/indicators.yaml"
    )
    return BacktestAiContextSnapshotBuilder(
        availability_summary_repository=_SummaryRepository(summary=summary),
        defaults_provider=defaults_provider,
        runtime_defaults_service=BacktestRuntimeDefaultsService(
            defaults_provider=defaults_provider,
            runtime_config=BacktestRuntimeConfig(
                hit_times_tp_levels_pct=(1.0,),
                hit_times_sl_levels_pct=(1.0,),
                artifact_config_hash="a" * 64,
            ),
        ),
    )


def _summary_payload(
    *,
    symbols: tuple[str, ...] = ("BTCUSDT",),
    timeframes: tuple[str, ...] = ("1h",),
) -> Mapping[str, Any]:
    defaults_provider = YamlBacktestGridDefaultsProvider.from_yaml(
        config_path="configs/prod/indicators.yaml"
    )
    indicator_ids = list(defaults_provider.supported_indicator_ids())
    instruments: dict[str, Any] = {}
    for symbol in symbols:
        instruments[f"binance/spot/{symbol}"] = _instrument(
            symbol=symbol,
            timeframes=timeframes,
            indicator_ids=indicator_ids,
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": "2026-05-17T21:15:44Z",
        "artifact_root": "/tmp/not-exposed-to-model",
        "artifact_root_schema_version": 2,
        "summary_hash": "",
        "source": "artifact_publisher_active_slot_scan",
        "instruments": instruments,
    }
    payload["summary_hash"] = _summary_hash(payload)
    return payload


def _instrument(
    *,
    symbol: str,
    timeframes: tuple[str, ...],
    indicator_ids: list[str],
) -> Mapping[str, Any]:
    return {
        "exchange": "binance",
        "market": "spot",
        "symbol": symbol,
        "active_slot": "slot_a",
        "slot_generation": 7,
        "asof_date": "2026-05-02",
        "published_at_utc": "2026-05-02T01:36:16Z",
        "manifest_sha256": "b" * 64,
        "start_date": "2020-01-10",
        "end_date": "2026-05-02",
        "backtest_timeframes": list(timeframes),
        "timeframes": {
            timeframe: {
                "start_date": "2020-01-10",
                "end_date": "2026-05-02",
                "bars": 100,
                "price_available": True,
                "signals_available": True,
                "mappings_available": True,
                "indicator_ids": indicator_ids,
            }
            for timeframe in timeframes
        },
        "hit_times": {"timeframe": "15m", "available": True},
    }


def _summary_hash(payload: Mapping[str, Any]) -> str:
    hash_payload = dict(payload)
    hash_payload.pop("summary_hash", None)
    hash_payload.pop("generated_at_utc", None)
    serialized = json.dumps(
        hash_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _SummaryRepository:
    summary: Mapping[str, Any] | None

    def load_availability_summary(self) -> Mapping[str, Any]:
        if self.summary is None:
            raise FileNotFoundError("availability_summary.yaml not found")
        return self.summary
