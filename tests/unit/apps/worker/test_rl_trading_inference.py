from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from apps.worker.rl_trading_inference.main import main as inference_main
from apps.worker.rl_trading_inference.wiring.modules import (
    RedisRlClosedCandleStream,
    RedisRlFeatureWindowReader,
    RlTradingInferenceMetrics,
    RlTradingInferenceRedisStreamsConfig,
    load_rl_trading_inference_runtime_config,
)
from scripts.rl_trading import stage17_multi_ticker_runtime_load as stage17_load
from trading.contexts.rl_trading.domain import monitor_only_inference as mi

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_runtime_configs_keep_non_prod_disabled_and_enable_bounded_prod_monitor() -> None:
    for profile in ("dev", "test", "prod"):
        config = load_rl_trading_inference_runtime_config(
            REPO_ROOT / "configs" / profile / "rl_trading_ml_runtime.yaml"
        )

        assert config.profile == profile
        assert config.enabled is (profile == "prod")
        assert config.mode == "monitor_only"
        assert config.rollout_phase == ("five_ticker_24h" if profile == "prod" else "disabled")
        assert config.source_events.enabled is (profile == "prod")
        assert config.source_events.source_type == "ml_agent_decision"
        assert config.source_events.outcome == "no_intent"
        assert config.redis_streams.enabled is True
        assert config.redis_streams.stream_prefix == "md.candles.1m"
        assert config.redis_streams.window_size == 90
        assert len(config.instruments) == 5
        assert config.monitor_policy.direction_policy == "long_only"
        assert config.monitor_policy.virtual_hold_minutes == 1
        assert config.latency_budget.feature_to_decision_p95_ms == 100
        expected_reasons = (
            [] if profile == "prod" else ["inference_disabled", "source_events_disabled"]
        )
        assert config.readiness_payload()["degraded_reasons"] == expected_reasons


def test_redis_window_reader_reads_latest_stream_without_ack() -> None:
    fake_redis = _FakeRedis(
        rows=[
            (
                _stream_id("2026-07-03T12:01:00Z"),
                _redis_payload(ts_open="2026-07-03T12:01:00Z", close="103.0"),
            ),
            (
                _stream_id("2026-07-03T12:00:00Z"),
                _redis_payload(ts_open="2026-07-03T12:00:00Z", close="102.0"),
            ),
        ]
    )
    reader = RedisRlFeatureWindowReader(
        redis_client=fake_redis,
        config=RlTradingInferenceRedisStreamsConfig(
            enabled=True,
            host="127.0.0.1",
            port=6379,
            db=0,
            auth_env=None,
            socket_timeout_s=2.0,
            connect_timeout_s=2.0,
            stream_prefix="md.candles.1m",
            window_size=2,
            consumer_group="rl.inference.monitor.v1",
            consumer_name="test",
            read_count=20,
            block_ms=0,
            pending_claim_min_idle_ms=60_000,
        ),
    )

    window = reader.read_latest_window(
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
    )

    assert fake_redis.calls == [("md.candles.1m.binance:futures:BTCUSDT", 2)]
    assert len(window.candles) == 2
    assert window.candles[0].close == 102.0
    assert window.candles[1].close == 103.0


def test_redis_window_reader_rejects_gap_and_wrong_message_boundary() -> None:
    gap_rows = [
        (
            _stream_id("2026-07-03T12:02:00Z"),
            _redis_payload(ts_open="2026-07-03T12:02:00Z", close="103.0"),
        ),
        (
            _stream_id("2026-07-03T12:00:00Z"),
            _redis_payload(ts_open="2026-07-03T12:00:00Z", close="102.0"),
        ),
    ]
    reader = RedisRlFeatureWindowReader(
        redis_client=_FakeRedis(rows=gap_rows),
        config=_redis_config(window_size=2),
    )

    with pytest.raises(ValueError, match="redis_window_not_contiguous"):
        reader.read_window_at_message(
            exchange="binance",
            market_type="futures",
            symbol="BTCUSDT",
            instrument_key="binance:futures:BTCUSDT",
            message_id=_stream_id("2026-07-03T12:02:00Z"),
        )
    with pytest.raises(ValueError, match="does not end at the consumed message"):
        reader.read_window_at_message(
            exchange="binance",
            market_type="futures",
            symbol="BTCUSDT",
            instrument_key="binance:futures:BTCUSDT",
            message_id=_stream_id("2026-07-03T12:03:00Z"),
        )


def test_runtime_config_rejects_invalid_operator_uuid(tmp_path: Path) -> None:
    config_text = (REPO_ROOT / "configs/test/rl_trading_ml_runtime.yaml").read_text(
        encoding="utf-8"
    )
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        config_text.replace(
            "ab094ba2-61d7-4fbf-be8f-cbad9f351572",
            "not-a-uuid",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="owner_user_id must be a UUID"):
        load_rl_trading_inference_runtime_config(config_path)


def test_redis_closed_candle_stream_uses_consumer_group_and_ack() -> None:
    fake_redis = _FakeConsumerRedis()
    config = RlTradingInferenceRedisStreamsConfig(
        enabled=True,
        host="127.0.0.1",
        port=6379,
        db=0,
        auth_env=None,
        socket_timeout_s=2.0,
        connect_timeout_s=2.0,
        stream_prefix="md.candles.1m",
        window_size=90,
        consumer_group="rl.inference.monitor.v1",
        consumer_name="test-consumer",
        read_count=20,
        block_ms=0,
        pending_claim_min_idle_ms=60_000,
    )
    stream = RedisRlClosedCandleStream(
        redis_client=fake_redis,
        config=config,
        instrument_keys=("binance:futures:BTCUSDT",),
    )

    messages = stream.read()
    stream.ack(message=messages[0])

    assert messages[0].instrument_key == "binance:futures:BTCUSDT"
    assert messages[0].message_id == "3-0"
    assert fake_redis.group_creates == 1
    assert fake_redis.acks == [("md.candles.1m.binance:futures:BTCUSDT", "3-0")]


def test_metrics_render_bounded_labels_without_user_or_strategy_ids() -> None:
    metrics = RlTradingInferenceMetrics(registry=CollectorRegistry())
    metrics.set_readiness(ready=False, degraded_reasons=["inference_disabled"])
    metrics.observe_decision(outcome="no_intent", reason="monitor_only_no_intent")
    metrics.observe_segment_latency(segment="feature_to_decision", seconds=0.01)
    metrics.observe_feature_parity(result="accepted")

    rendered = metrics.render_latest().decode("utf-8")

    assert 'mode="monitor_only",outcome="no_intent",reason="monitor_only_no_intent"' in rendered
    assert "00000000-0000-0000-0000-000000013001" not in rendered
    assert "rl_trading_inference_feature_parity_total" in rendered
    assert "rl_trading_inference_degraded_state" in rendered


def test_status_and_parity_cli_return_machine_readable_json(tmp_path: Path, capsys) -> None:
    live_path = tmp_path / "live.json"
    offline_path = tmp_path / "offline.json"
    live_path.write_text(
        json.dumps(
            {
                "exchange": "binance",
                "instrument_key": "binance:futures:BTCUSDT",
                "market_type": "futures",
                "payloads": [
                    _redis_payload(ts_open="2026-07-03T12:00:00Z", close="102.0"),
                    _redis_payload(ts_open="2026-07-03T12:01:00Z", close="103.0"),
                ],
                "symbol": "BTCUSDT",
            }
        ),
        encoding="utf-8",
    )
    offline_path.write_text(
        json.dumps(
            {
                "candles": [
                    {
                        "close": 102.0,
                        "high": 103.0,
                        "low": 99.0,
                        "open": 100.0,
                        "trades_count": 42,
                        "volume_base": 10.0,
                        "volume_quote": 1010.0,
                    },
                    {
                        "close": 103.0,
                        "high": 104.0,
                        "low": 101.0,
                        "open": 102.0,
                        "trades_count": 37,
                        "volume_base": 8.0,
                        "volume_quote": 824.0,
                    }
                ],
                "exchange": "binance",
                "instrument_key": "binance:futures:BTCUSDT",
                "market_type": "futures",
                "symbol": "BTCUSDT",
                "ts_close": "2026-07-03T12:02:00Z",
                "ts_open": "2026-07-03T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    assert inference_main.main(
        [
            "status",
            "--config",
            str(REPO_ROOT / "configs" / "dev" / "rl_trading_ml_runtime.yaml"),
        ]
    ) == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["mode"] == "monitor_only"
    assert status_payload["ready"] is False

    assert inference_main.main(
        [
            "parity",
            "--live-window-json",
            str(live_path),
            "--offline-window-json",
            str(offline_path),
        ]
    ) == 0
    parity_payload = json.loads(capsys.readouterr().out)
    assert parity_payload["status"] == "accepted"
    assert parity_payload["max_abs_diff"] == 0.0


def test_canary_cli_records_source_event_without_intent(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    feature_path = tmp_path / "feature-window.json"
    manifest_path = tmp_path / "candidate-manifest.json"
    feature_path.write_text(
        json.dumps(
            {
                "exchange": "binance",
                "instrument_key": "binance:futures:BTCUSDT",
                "market_type": "futures",
                "payloads": [
                    _redis_payload(ts_open="2026-07-03T12:00:00Z", close="102.0"),
                    _redis_payload(ts_open="2026-07-03T12:01:00Z", close="103.0"),
                ],
                "symbol": "BTCUSDT",
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(_candidate_manifest(feature_count=12), sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mi,
        "STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    assert inference_main.main(
        [
            "canary-once",
            "--candidate-manifest",
            str(manifest_path),
            "--feature-window-json",
            str(feature_path),
            "--owner-user-id",
            "00000000-0000-0000-0000-000000013001",
            "--strategy-id",
            "00000000-0000-0000-0000-000000013101",
            "--strategy-run-id",
            "00000000-0000-0000-0000-000000013201",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_type"] == "ml_agent_decision"
    assert payload["outcome"] == "no_intent"
    assert payload["outcome_reason"] == "monitor_only_no_intent"
    assert payload["source_events_created"] == 1
    assert payload["intents_created"] == 0


def test_paper_cli_records_intent_order_and_parity(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    feature_path = tmp_path / "feature-window.json"
    manifest_path = tmp_path / "candidate-manifest.json"
    feature_path.write_text(
        json.dumps(
            {
                "exchange": "binance",
                "instrument_key": "binance:futures:BTCUSDT",
                "market_type": "futures",
                "payloads": [
                    _redis_payload(ts_open="2026-07-03T12:00:00Z", close="102.0"),
                    _redis_payload(ts_open="2026-07-03T12:01:00Z", close="103.0"),
                ],
                "symbol": "BTCUSDT",
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            _candidate_manifest(feature_count=12, preferred_action="open_long"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mi,
        "STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    assert inference_main.main(
        [
            "paper-once",
            "--candidate-manifest",
            str(manifest_path),
            "--feature-window-json",
            str(feature_path),
            "--owner-user-id",
            "00000000-0000-0000-0000-000000013001",
            "--strategy-id",
            "00000000-0000-0000-0000-000000013101",
            "--strategy-run-id",
            "00000000-0000-0000-0000-000000013201",
            "--quote-notional",
            "50",
            "--reference-price",
            "10000",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_type"] == "ml_agent_decision"
    assert payload["action"] == "open_long"
    assert payload["outcome"] == "risk_rejected"
    assert payload["outcome_reason"] == "paper_no_exchange_submit"
    assert payload["risk_reason"] == "paper_no_exchange_submit"
    assert payload["source_events_created"] == 1
    assert payload["intents_created"] == 1
    assert payload["paper_orders_created"] == 1
    assert payload["paper_fills_created"] == 1
    assert payload["paper_accounting_created"] == 1
    assert payload["duplicate_replay"] is True
    assert payload["simulator_parity"] == {
        "abs_diff": {
            "equity": "0E-8",
            "fee_total": "0E-8",
            "position_quantity": "0E-8",
        },
        "max_abs_diff": "0E-8",
        "status": "accepted",
        "tolerance": "0",
    }


def test_testnet_cli_dispatches_intent_and_duplicate_dispatch(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    feature_path = tmp_path / "feature-window.json"
    manifest_path = tmp_path / "candidate-manifest.json"
    feature_path.write_text(
        json.dumps(
            {
                "exchange": "binance",
                "instrument_key": "binance:futures:BTCUSDT",
                "market_type": "futures",
                "payloads": [
                    _redis_payload(ts_open="2026-07-03T12:00:00Z", close="102.0"),
                    _redis_payload(ts_open="2026-07-03T12:01:00Z", close="103.0"),
                ],
                "symbol": "BTCUSDT",
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            _candidate_manifest(feature_count=12, preferred_action="open_long"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mi,
        "STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    assert inference_main.main(
        [
            "testnet-once",
            "--candidate-manifest",
            str(manifest_path),
            "--feature-window-json",
            str(feature_path),
            "--owner-user-id",
            "00000000-0000-0000-0000-000000013001",
            "--strategy-id",
            "00000000-0000-0000-0000-000000013101",
            "--strategy-run-id",
            "00000000-0000-0000-0000-000000013201",
            "--exchange-connection-id",
            "00000000-0000-0000-0000-000000014001",
            "--quote-notional",
            "50",
            "--quantity",
            "0.001",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_type"] == "ml_agent_decision"
    assert payload["action"] == "open_long"
    assert payload["outcome"] == "intent_created"
    assert payload["risk_status"] == "accepted"
    assert payload["risk_reason"] == "risk_gate_accepted"
    assert payload["intent_status"] == "dispatched"
    assert payload["dispatch"]["result"] == "dispatched"
    assert payload["duplicate_dispatch"]["result"] == "duplicate"
    assert payload["duplicate_replay"] is True
    assert payload["memory_counts"] == {
        "dispatch_messages": 1,
        "intents": 1,
        "source_events": 1,
    }


def test_testnet_cli_blocks_spot_short_without_dispatch(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    feature_path = tmp_path / "feature-window.json"
    manifest_path = tmp_path / "candidate-manifest.json"
    feature_path.write_text(
        json.dumps(
            {
                "candles": [
                    _candle(close=102.0),
                    _candle(close=103.0),
                ],
                "exchange": "bybit",
                "instrument_key": "bybit:spot:BTCUSDT",
                "market_type": "spot",
                "symbol": "BTCUSDT",
                "ts_close": "2026-07-03T12:02:00Z",
                "ts_open": "2026-07-03T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            _candidate_manifest(feature_count=12, preferred_action="open_short"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mi,
        "STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    assert inference_main.main(
        [
            "testnet-once",
            "--candidate-manifest",
            str(manifest_path),
            "--feature-window-json",
            str(feature_path),
            "--owner-user-id",
            "00000000-0000-0000-0000-000000013001",
            "--strategy-id",
            "00000000-0000-0000-0000-000000013101",
            "--strategy-run-id",
            "00000000-0000-0000-0000-000000013201",
            "--exchange-connection-id",
            "00000000-0000-0000-0000-000000014002",
            "--quote-notional",
            "50",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_type"] == "ml_agent_decision"
    assert payload["action"] == "open_short"
    assert payload["outcome"] == "no_intent"
    assert payload["outcome_reason"] == "testnet_spot_short_not_supported"
    assert payload["intent_id"] is None
    assert payload["dispatch"] is None
    assert payload["memory_counts"] == {
        "dispatch_messages": 0,
        "intents": 0,
        "source_events": 1,
    }


def test_stage17_load_harness_exercises_quota_counts_without_dispatch_growth(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "candidate-manifest.json"
    manifest_path.write_text(
        json.dumps(_candidate_manifest(feature_count=12), sort_keys=True),
        encoding="utf-8",
    )
    fake_redis = _Stage17FakeRedis(
        streams={
            f"md.candles.1m.binance:futures:TEST{index:02d}USDT": _stage17_rows(
                symbol=f"TEST{index:02d}USDT"
            )
            for index in range(20)
        },
        lengths={
            "execution.requests.dlq.v1": 2,
            "execution.requests.retry.v1": 1,
            "execution.requests.v1": 49,
        },
    )
    monkeypatch.setattr(stage17_load, "_build_redis_client", lambda **_kwargs: fake_redis)

    assert stage17_load.main(
        [
            "--config",
            str(REPO_ROOT / "configs" / "test" / "rl_trading_ml_runtime.yaml"),
            "--candidate-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path),
            "--generated-at-utc",
            "2026-07-05T18:30:00Z",
            "--max-feed-lag-seconds",
            "999999999",
            "--allow-fixture-manifest-hash",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))

    assert payload["status"] == "accepted"
    assert payload["observations"] == 26
    assert summary["quota_scenarios"] == [
        {
            "label": "free",
            "observation_count": 1,
            "observed_tickers": 1,
            "paid_level": "free",
            "product_label": "Free",
            "quota_bypass_observed": False,
            "requested_live_tickers": 1,
        },
        {
            "label": "pro",
            "observation_count": 5,
            "observed_tickers": 5,
            "paid_level": "pro",
            "product_label": "Pro",
            "quota_bypass_observed": False,
            "requested_live_tickers": 5,
        },
        {
            "label": "premium",
            "observation_count": 20,
            "observed_tickers": 20,
            "paid_level": "ultra",
            "product_label": "Premium",
            "quota_bypass_observed": False,
            "requested_live_tickers": 20,
        },
    ]
    assert summary["redis_execution_streams"]["delta"] == {
        "execution.requests.dlq.v1": 0,
        "execution.requests.retry.v1": 0,
        "execution.requests.v1": 0,
    }
    assert {item["outcome"] for item in summary["observations"]} == {"no_intent"}


class _FakeRedis:
    def __init__(self, *, rows: list[tuple[str, dict[str, str]]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, int]] = []

    def xrevrange(
        self,
        name: str,
        *,
        count: int,
        max: str | None = None,
    ) -> list[tuple[str, dict[str, str]]]:
        del max
        self.calls.append((name, count))
        return self.rows[:count]


class _FakeConsumerRedis:
    def __init__(self) -> None:
        self.group_creates = 0
        self.acks: list[tuple[str, str]] = []

    def xgroup_create(self, **kwargs: object) -> None:
        del kwargs
        self.group_creates += 1

    def xautoclaim(self, *args: object, **kwargs: object) -> tuple[str, list[object], list[object]]:
        del args, kwargs
        return ("0-0", [], [])

    def xreadgroup(self, **kwargs: object) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        streams = kwargs["streams"]
        assert isinstance(streams, dict)
        stream_name = next(iter(streams))
        payload = _redis_payload(ts_open="2026-07-03T12:01:00Z", close="103.0")
        return [(str(stream_name), [("3-0", payload)])]

    def xack(self, stream_name: str, group: str, message_id: str) -> None:
        assert group == "rl.inference.monitor.v1"
        self.acks.append((stream_name, message_id))


class _Stage17FakeRedis:
    def __init__(
        self,
        *,
        streams: dict[str, list[tuple[str, dict[str, str]]]],
        lengths: dict[str, int],
    ) -> None:
        self.streams = streams
        self.lengths = lengths

    def scan_iter(self, *, match: str, count: int) -> list[str]:
        del count
        prefix = match.removesuffix("*")
        return sorted(name for name in self.streams if name.startswith(prefix))

    def xrevrange(self, name: str, *, count: int) -> list[tuple[str, dict[str, str]]]:
        return self.streams[name][:count]

    def xlen(self, name: str) -> int:
        if name in self.lengths:
            return self.lengths[name]
        return len(self.streams[name])


def _redis_payload(*, ts_open: str, close: str) -> dict[str, str]:
    ts_open_value = datetime.fromisoformat(ts_open.replace("Z", "+00:00"))
    return {
        "schema_version": "1",
        "instrument_key": "binance:futures:BTCUSDT",
        "ts_open": ts_open,
        "ts_close": (ts_open_value + timedelta(minutes=1))
        .isoformat()
        .replace("+00:00", "Z"),
        "open": "100.0" if close == "102.0" else "102.0",
        "high": "103.0" if close == "102.0" else "104.0",
        "low": "99.0" if close == "102.0" else "101.0",
        "close": close,
        "volume_base": "10.0" if close == "102.0" else "8.0",
        "volume_quote": "1010.0" if close == "102.0" else "824.0",
        "trades_count": "42" if close == "102.0" else "37",
    }


def _redis_config(*, window_size: int) -> RlTradingInferenceRedisStreamsConfig:
    return RlTradingInferenceRedisStreamsConfig(
        enabled=True,
        host="127.0.0.1",
        port=6379,
        db=0,
        auth_env=None,
        socket_timeout_s=2.0,
        connect_timeout_s=2.0,
        stream_prefix="md.candles.1m",
        window_size=window_size,
        consumer_group="rl.inference.monitor.v1",
        consumer_name="test",
        read_count=20,
        block_ms=0,
        pending_claim_min_idle_ms=60_000,
    )


def _stream_id(ts_open: str) -> str:
    opened = datetime.fromisoformat(ts_open.replace("Z", "+00:00"))
    return f"{int(opened.timestamp() * 1_000)}-0"


def _stage17_rows(*, symbol: str) -> list[tuple[str, dict[str, str]]]:
    instrument_key = f"binance:futures:{symbol}"
    return [
        (
            "2-0",
            {
                **_redis_payload(ts_open="2026-07-05T18:29:00Z", close="103.0"),
                "instrument_key": instrument_key,
                "symbol": symbol,
                "ts_close": "2026-07-05T18:30:00Z",
            },
        ),
        (
            "1-0",
            {
                **_redis_payload(ts_open="2026-07-05T18:28:00Z", close="102.0"),
                "instrument_key": instrument_key,
                "symbol": symbol,
                "ts_close": "2026-07-05T18:29:00Z",
            },
        ),
    ]


def _candle(*, close: float) -> dict[str, object]:
    return {
        "close": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "open": close - 0.5,
        "trades_count": 42,
        "volume_base": 10.0,
        "volume_quote": close * 10.0,
    }


def _candidate_manifest(
    *, feature_count: int, preferred_action: str = "hold"
) -> dict[str, object]:
    weights = [[0.0, -0.1, -0.2] for _index in range(feature_count)]
    if preferred_action == "open_long":
        weights = [[0.0, 1.0, 0.0]] + [[0.0, 0.0, 0.0] for _index in range(feature_count)]
    elif preferred_action == "open_short":
        weights = [[0.0, 0.0, 1.0]] + [[0.0, 0.0, 0.0] for _index in range(feature_count)]
    return {
        "candidate_id": mi.STAGE09_ACCEPTED_CANDIDATE_ID_V1,
        "model_state_hash": "a" * 64,
        "model_state": {
            "feature_count": feature_count,
            "label_order": {"0": "hold", "1": "open_long", "2": "open_short"},
            "scaler_mean": [0.0] * feature_count,
            "scaler_std": [1.0] * feature_count,
            "weights": weights,
        },
        "policy_name": mi.STAGE09_ACCEPTED_CANDIDATE_POLICY_V1,
        "stage": "08M",
        "stage09_allowed": True,
    }
