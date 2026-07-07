from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.rl_trading import stage08o_high_volatility_regime_report as high_vol


def test_quantile_matches_stage08l_tertile_reference() -> None:
    values = [0.05, 0.06, 0.07, 0.08]

    assert high_vol._quantile(values, 0.3333333333) == pytest.approx(  # noqa: SLF001
        0.059999999999
    )
    assert high_vol._quantile(values, 0.6666666667) == pytest.approx(  # noqa: SLF001
        0.070000000001
    )


def test_high_volatility_report_groups_tickers_and_reconciles_bucket(tmp_path: Path) -> None:
    metadata_a = tmp_path / "A" / "metadata.json"
    metadata_b = tmp_path / "B" / "metadata.json"
    metadata_a.parent.mkdir()
    metadata_b.parent.mkdir()
    _write_json(
        metadata_a,
        {
            "sessions": [
                _session("AAAUSDT", "2025-03-01T00:00:00Z", 0.05),
                _session("AAAUSDT", "2025-03-02T00:00:00Z", 0.08),
            ]
        },
    )
    _write_json(
        metadata_b,
        {
            "sessions": [
                _session("BBBUSDT", "2025-03-03T00:00:00Z", 0.06),
                _session("BBBUSDT", "2025-03-04T00:00:00Z", 0.09),
            ]
        },
    )
    stage08j_manifest = tmp_path / "stage08j_manifest.json"
    final_manifest = tmp_path / "final_manifest.json"
    scorecards = tmp_path / "scorecards.json"
    balance_curve = tmp_path / "balance_curve.json"
    _write_json(
        stage08j_manifest,
        {
            "split_artifacts": [
                _artifact("AAAUSDT", metadata_a, 2),
                _artifact("BBBUSDT", metadata_b, 2),
            ]
        },
    )
    _write_json(
        scorecards,
        {
            "scorecards": [
                {
                    "acceptance_backtest": True,
                    "metrics_by_volatility_bucket": [
                        {
                            "bucket": "high",
                            "closed_trades": 1,
                            "net_pnl_after_costs_quote": 20.0,
                            "profitable_trades": 1,
                            "session_count": 2,
                            "win_rate": 1.0,
                        }
                    ],
                }
            ]
        },
    )
    _write_json(
        balance_curve,
        {
            "balance_curve": [
                _row(0, "AAAUSDT", "2025-03-01T00:00:00Z", 100.0, 0),
                _row(1, "AAAUSDT", "2025-03-02T00:00:00Z", 100.0, 1),
                _row(1, "AAAUSDT", "2025-03-02T00:00:00Z", 100.0, 3),
                _row(3, "BBBUSDT", "2025-03-04T00:00:00Z", 100.0, 1),
                _row(3, "BBBUSDT", "2025-03-04T00:00:00Z", 120.0, 3),
            ]
        },
    )
    _write_json(
        final_manifest,
        {
            "artifact_hashes": {
                "scorecards": {"path": str(scorecards)},
                "balance_curve": {"path": str(balance_curve)},
            }
        },
    )

    report = high_vol.build_report(
        Namespace(
            bucket="high",
            dataset_version="hf_period_rebuild_current_trading",
            final_manifest=final_manifest,
            generated_at_utc="2026-07-07T00:00:00Z",
            output_dir=tmp_path / "out",
            split="backtest",
            stage08j_manifest=stage08j_manifest,
        )
    )

    assert report["recomputed_from_balance_curve"]["net_pnl_after_costs_quote"] == 20.0
    assert report["bucket_method"]["bucket_session_counts"] == {
        "high": 1,
        "low": 1,
        "medium": 1,
    }
    ticker_rows = {row["symbol"]: row for row in report["ticker_rows"]}
    assert "AAAUSDT" not in ticker_rows
    assert ticker_rows["BBBUSDT"]["net_pnl_after_costs_quote"] == 20.0
    assert ticker_rows["BBBUSDT"]["traded_in_high_regime"] is True


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifact(symbol: str, metadata_path: Path, count: int) -> dict[str, object]:
    return {
        "candidate_count": count,
        "dataset_version": "hf_period_rebuild_current_trading",
        "files": {"metadata": {"path": str(metadata_path)}},
        "split": "backtest",
        "symbol": symbol,
    }


def _session(symbol: str, signal_time: str, volatility_score: float) -> dict[str, object]:
    return {
        "signal_ts_open": signal_time,
        "session_end_utc": signal_time,
        "session_start_utc": signal_time,
        "symbol": symbol,
        "volatility_score": volatility_score,
    }


def _row(
    source_session_index: int,
    symbol: str,
    signal_time: str,
    shared_balance: float,
    effective_action_id: int,
) -> dict[str, object]:
    return {
        "effective_action_id": effective_action_id,
        "shared_balance_quote": shared_balance,
        "signal_time_utc": signal_time,
        "source_session_index": source_session_index,
        "symbol": symbol,
    }
