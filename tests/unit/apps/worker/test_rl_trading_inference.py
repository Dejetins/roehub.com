from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prometheus_client import CollectorRegistry

from apps.worker.rl_trading_inference.main import main as inference_main
from apps.worker.rl_trading_inference.wiring.modules import (
    RedisRlFeatureWindowReader,
    RlTradingInferenceMetrics,
    RlTradingInferenceRedisStreamsConfig,
    load_rl_trading_inference_runtime_config,
)
from trading.contexts.rl_trading.domain import monitor_only_inference as mi

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_runtime_configs_are_monitor_only_and_disabled_by_default() -> None:
    for profile in ("dev", "test", "prod"):
        config = load_rl_trading_inference_runtime_config(
            REPO_ROOT / "configs" / profile / "rl_trading_ml_runtime.yaml"
        )

        assert config.profile == profile
        assert config.enabled is False
        assert config.mode == "monitor_only"
        assert config.source_events.enabled is False
        assert config.source_events.source_type == "ml_agent_decision"
        assert config.source_events.outcome == "no_intent"
        assert config.redis_streams.enabled is True
        assert config.redis_streams.stream_prefix == "md.candles.1m"
        assert config.latency_budget.feature_to_decision_p95_ms == 100
        assert config.readiness_payload()["degraded_reasons"] == ["inference_disabled"]


def test_redis_window_reader_reads_latest_stream_without_ack() -> None:
    fake_redis = _FakeRedis(
        rows=[
            ("2-0", _redis_payload(ts_open="2026-07-03T12:01:00Z", close="103.0")),
            ("1-0", _redis_payload(ts_open="2026-07-03T12:00:00Z", close="102.0")),
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


class _FakeRedis:
    def __init__(self, *, rows: list[tuple[str, dict[str, str]]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, int]] = []

    def xrevrange(self, name: str, *, count: int) -> list[tuple[str, dict[str, str]]]:
        self.calls.append((name, count))
        return self.rows[:count]


def _redis_payload(*, ts_open: str, close: str) -> dict[str, str]:
    return {
        "schema_version": "1",
        "instrument_key": "binance:futures:BTCUSDT",
        "ts_open": ts_open,
        "ts_close": "2026-07-03T12:01:00Z"
        if ts_open.endswith("12:00:00Z")
        else "2026-07-03T12:02:00Z",
        "open": "100.0" if close == "102.0" else "102.0",
        "high": "103.0" if close == "102.0" else "104.0",
        "low": "99.0" if close == "102.0" else "101.0",
        "close": close,
        "volume_base": "10.0" if close == "102.0" else "8.0",
        "volume_quote": "1010.0" if close == "102.0" else "824.0",
        "trades_count": "42" if close == "102.0" else "37",
    }


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
