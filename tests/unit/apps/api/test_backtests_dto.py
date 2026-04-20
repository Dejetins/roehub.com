from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from apps.api.dto import (
    BacktestsPostRequest,
    BacktestsVariantReportPostRequest,
    build_backtest_run_request,
    build_backtest_variant_report_payload,
    build_backtest_variant_report_run_request,
    build_backtests_post_response,
    build_sha256_from_payload,
    decode_backtest_request_payload,
)
from trading.contexts.backtest.application.dto import (
    BacktestVariantPayloadV1,
    BacktestVariantPreview,
    RunBacktestResponse,
)
from trading.contexts.backtest.application.ports import BacktestStrategySnapshot
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import (
    ExplicitValuesSpec,
    GridSpec,
    RangeValuesSpec,
)
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    UserId,
)


def test_build_backtest_run_request_preserves_int_range_axes() -> None:
    """Ensure range axis values do not get coerced to float.

    This protects integer indicator params (e.g. MA `window`) from failing grid validation
    with: `axis 'window' expects integer values`.

    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/indicators/application/services/grid_builder.py
      - configs/prod/indicators.yaml
    """

    request = BacktestsPostRequest.model_validate(
        {
            "time_range": {
                "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
            },
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1m",
                "indicator_grids": [
                    {
                        "indicator_id": "ma.sma",
                        "source": {"mode": "explicit", "values": ["close"]},
                        "params": {
                            "window": {
                                "mode": "range",
                                "start": 5,
                                "stop_incl": 100,
                                "step": 1,
                            }
                        },
                    }
                ],
            },
        }
    )

    built = build_backtest_run_request(request=request)
    assert built.template is not None
    assert len(built.template.indicator_grids) == 1

    window_spec = built.template.indicator_grids[0].params["window"]
    assert isinstance(window_spec, RangeValuesSpec)
    assert isinstance(window_spec.start, int)
    assert isinstance(window_spec.stop_inclusive, int)
    assert isinstance(window_spec.step, int)

    materialized = window_spec.materialize()
    assert len(materialized) > 0
    assert all(isinstance(item, int) for item in materialized)


def test_build_backtest_run_request_normalizes_ranking_metrics() -> None:
    """
    Verify ranking request block is accepted and normalized into application DTO.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Ranking metric identifiers are case-insensitive and normalized to lowercase literals.
    Raises:
        AssertionError: If ranking block is not converted or normalized deterministically.
    Side Effects:
        None.
    """
    request = BacktestsPostRequest.model_validate(
        {
            "time_range": {
                "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
            },
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1m",
                "indicator_grids": [
                    {
                        "indicator_id": "ma.sma",
                        "source": {"mode": "explicit", "values": ["close"]},
                        "params": {
                            "window": {
                                "mode": "range",
                                "start": 5,
                                "stop_incl": 20,
                                "step": 5,
                            }
                        },
                    }
                ],
            },
            "ranking": {
                "primary_metric": "SHARPE_TRADES",
            },
        }
    )

    built = build_backtest_run_request(request=request)
    assert built.ranking is not None
    assert built.ranking.primary_metric == "sharpe_trades"
    assert built.ranking.secondary_metric is None


def test_decode_backtest_request_payload_reuses_strict_post_backtests_contract() -> None:
    """
    Verify persisted request decoder rebuilds application request via strict API DTO contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Persisted `request_json` keeps the same shape as canonical `POST /backtests` payload.
    Raises:
        AssertionError: If decoded request drifts from strict mapper behavior.
    Side Effects:
        None.
    """
    built = decode_backtest_request_payload(
        payload={
            "time_range": {
                "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
            },
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1m",
                "indicator_grids": [
                    {
                        "indicator_id": "ma.sma",
                        "params": {"window": {"mode": "explicit", "values": [20]}},
                    }
                ],
            },
            "warmup_bars": 144,
        }
    )

    assert built.mode == "template"
    assert built.warmup_bars is None
    assert built.template is not None
    assert built.template.instrument_id == InstrumentId(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
    )


def test_backtests_post_request_rejects_public_warmup_bars_field() -> None:
    """
    Verify strict public request validation rejects removed `warmup_bars` input.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Warmup is now derived internally instead of accepted from the public launch request.
    Raises:
        AssertionError: If removed field is still accepted by the public DTO.
    Side Effects:
        None.
    """
    with pytest.raises(ValidationError, match="warmup_bars"):
        BacktestsPostRequest.model_validate(
            {
                "time_range": {
                    "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
                },
                "template": {
                    "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                    "timeframe": "1m",
                    "indicator_grids": [
                        {
                            "indicator_id": "ma.sma",
                            "params": {"window": {"mode": "explicit", "values": [20]}},
                        }
                    ],
                },
                "warmup_bars": 144,
            }
        )


def test_backtests_post_request_rejects_removed_top_trades_n_field() -> None:
    """
    Verify strict public request validation rejects removed `top_trades_n` input.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public launch transport keeps `top_k` and no longer accepts report-depth knobs.
    Raises:
        AssertionError: If removed field is still accepted by the public DTO.
    Side Effects:
        None.
    """
    with pytest.raises(ValidationError, match="top_trades_n"):
        BacktestsPostRequest.model_validate(
            {
                "time_range": {
                    "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
                },
                "template": {
                    "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                    "timeframe": "1m",
                    "indicator_grids": [
                        {
                            "indicator_id": "ma.sma",
                            "params": {"window": {"mode": "explicit", "values": [20]}},
                        }
                    ],
                },
                "top_trades_n": 3,
            }
        )


def test_decode_backtest_request_payload_strips_legacy_top_trades_n_for_reads() -> None:
    """
    Verify persisted read compatibility strips legacy `top_trades_n` before strict decode.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Historical `request_json` rows may still contain removed report-depth knobs.
    Raises:
        AssertionError: If persisted decode depends on removed field or rejects legacy rows.
    Side Effects:
        None.
    """
    built = decode_backtest_request_payload(
        payload={
            "time_range": {
                "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
            },
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1m",
                "indicator_grids": [
                    {
                        "indicator_id": "ma.sma",
                        "params": {"window": {"mode": "explicit", "values": [20]}},
                    }
                ],
            },
            "top_k": 10,
            "top_trades_n": 3,
        }
    )

    assert built.mode == "template"
    assert built.top_k == 10
    assert built.template is not None


def test_decode_backtest_request_payload_strips_legacy_secondary_metric_for_reads() -> None:
    """
    Verify persisted read compatibility strips legacy `ranking.secondary_metric` before decode.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Historical `request_json` rows may still carry the removed ranking field.
    Raises:
        AssertionError: If persisted decode keeps depending on the removed public field.
    Side Effects:
        None.
    """
    built = decode_backtest_request_payload(
        payload={
            "time_range": {
                "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
            },
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1m",
                "indicator_grids": [
                    {
                        "indicator_id": "ma.sma",
                        "params": {"window": {"mode": "explicit", "values": [20]}},
                    }
                ],
            },
            "ranking": {
                "primary_metric": "SHARPE_TRADES",
                "secondary_metric": "WIN_RATE_PCT",
            },
        }
    )

    assert built.ranking is not None
    assert built.ranking.primary_metric == "sharpe_trades"
    assert built.ranking.secondary_metric is None


def test_backtests_post_request_rejects_unknown_ranking_metric() -> None:
    """
    Verify strict request validation rejects unsupported ranking metric literal.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Allowed ranking metrics are fixed by v1 contract.
    Raises:
        AssertionError: If unsupported metric value is accepted.
    Side Effects:
        None.
    """
    with pytest.raises(ValidationError, match="must be one of"):
        BacktestsPostRequest.model_validate(
            {
                "time_range": {
                    "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
                },
                "template": {
                    "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                    "timeframe": "1m",
                    "indicator_grids": [
                        {
                            "indicator_id": "ma.sma",
                            "params": {
                                "window": {"mode": "explicit", "values": [20]},
                            },
                        }
                    ],
                },
                "ranking": {
                    "primary_metric": "total_return",
                },
            }
        )


def test_backtests_post_request_rejects_removed_secondary_metric_field() -> None:
    """
    Verify strict request validation rejects the removed `ranking.secondary_metric` field.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        New launch requests must use `primary_metric` only.
    Raises:
        AssertionError: If the removed field is still accepted in public requests.
    Side Effects:
        None.
    """
    with pytest.raises(ValidationError, match="secondary_metric"):
        BacktestsPostRequest.model_validate(
            {
                "time_range": {
                    "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
                },
                "template": {
                    "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                    "timeframe": "1m",
                    "indicator_grids": [
                        {
                            "indicator_id": "ma.sma",
                            "params": {
                                "window": {"mode": "explicit", "values": [20]},
                            },
                        }
                    ],
                },
                "ranking": {
                    "primary_metric": "total_return_pct",
                    "secondary_metric": "win_rate_pct",
                },
            }
        )


def test_build_backtest_variant_report_run_request_reuses_mode_validation() -> None:
    """
    Verify variant-report run-context mapper reuses `strategy_id xor template` validation.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Variant-report endpoint shares sync mode contract with `POST /backtests`.
    Raises:
        AssertionError: If mode conflict is not rejected deterministically.
    Side Effects:
        None.
    """
    request = BacktestsVariantReportPostRequest.model_validate(
        {
            "time_range": {
                "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
            },
            "strategy_id": "00000000-0000-0000-0000-000000000123",
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1m",
                "indicator_grids": [
                    {
                        "indicator_id": "ma.sma",
                        "params": {
                            "window": {"mode": "explicit", "values": [20]},
                        },
                    }
                ],
            },
            "variant": _variant_payload_request(),
        }
    )

    with pytest.raises(
        BacktestValidationError,
        match="requires exactly one mode",
    ):
        build_backtest_variant_report_run_request(request=request)


def test_build_backtest_variant_report_payload_normalizes_payload_deterministically() -> None:
    """
    Verify variant payload mapper sorts keys and lowercases signal identifiers deterministically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Request payload can arrive in arbitrary key order from browser-side cache.
    Raises:
        AssertionError: If normalized payload ordering differs from deterministic contract.
    Side Effects:
        None.
    """
    payload = build_backtest_variant_report_payload(
        request=BacktestsVariantReportPostRequest.model_validate(
            {
                "time_range": {
                    "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
                },
                "template": {
                    "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                    "timeframe": "1m",
                    "indicator_grids": [
                        {
                            "indicator_id": "ma.sma",
                            "params": {
                                "window": {"mode": "explicit", "values": [20]},
                            },
                        }
                    ],
                },
                "variant": {
                    "indicator_selections": [
                        {
                            "indicator_id": "ma.sma",
                            "inputs": {"source": "close"},
                            "params": {"window": 20},
                        }
                    ],
                    "signal_params": {"MA.SMA": {"Cross_Up": 0.5}},
                    "risk_params": {"tp_enabled": True, "sl_enabled": True, "sl_pct": 2.0},
                    "execution_params": {
                        "slippage_pct": 0.01,
                        "fee_pct": 0.075,
                        "init_cash_quote": 10000.0,
                    },
                    "direction_mode": "long-short",
                    "sizing_mode": "all_in",
                },
            }
        ).variant
    )

    assert payload.signal_params is not None
    assert payload.risk_params is not None
    assert payload.execution_params is not None
    assert tuple(payload.signal_params.keys()) == ("ma.sma",)
    assert tuple(payload.signal_params["ma.sma"].keys()) == ("cross_up",)
    assert tuple(payload.risk_params.keys()) == ("sl_enabled", "sl_pct", "tp_enabled")
    assert tuple(payload.execution_params.keys()) == (
        "fee_pct",
        "init_cash_quote",
        "slippage_pct",
    )


def test_build_backtest_variant_report_payload_rejects_boolean_indicator_values() -> None:
    """
    Verify variant payload mapper rejects booleans in explicit indicator selection scalars.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Indicator selection scalars follow indicators variant key contract (`int|float|str`).
    Raises:
        AssertionError: If boolean scalar is accepted in selection mapping.
    Side Effects:
        None.
    """
    with pytest.raises(ValidationError, match="must be int, float, or string"):
        BacktestsVariantReportPostRequest.model_validate(
            {
                "time_range": {
                    "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
                },
                "template": {
                    "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                    "timeframe": "1m",
                    "indicator_grids": [
                        {
                            "indicator_id": "ma.sma",
                            "params": {
                                "window": {"mode": "explicit", "values": [20]},
                            },
                        }
                    ],
                },
                "variant": {
                    "indicator_selections": [
                        {
                            "indicator_id": "ma.sma",
                            "inputs": {"source": "close"},
                            "params": {"window": True},
                        }
                    ],
                    "signal_params": {"ma.sma": {"cross_up": 0.5}},
                    "risk_params": {"sl_enabled": True},
                    "execution_params": {"fee_pct": 0.075},
                    "direction_mode": "long-short",
                    "sizing_mode": "all_in",
                },
            }
        )


def test_build_backtests_post_response_maps_persisted_sync_inline_metadata() -> None:
    """
    Verify sync response mapper exposes persisted run identity metadata additively.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R7-02 makes persisted sync-inline metadata mandatory for successful `/backtests`.
    Raises:
        AssertionError: If persisted metadata or stable hashes are mapped incorrectly.
    Side Effects:
        None.
    """
    request = BacktestsPostRequest.model_validate(
        {
            "time_range": {
                "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
            },
            "strategy_id": "00000000-0000-0000-0000-000000000123",
        }
    )
    snapshot_payload = {
        "schema_version": 1,
        "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
        "timeframe": "1m",
    }
    response = RunBacktestResponse(
        mode="saved",
        strategy_id=UUID("00000000-0000-0000-0000-000000000123"),
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        top_k=1,
        preselect=100,
        variants=(
            BacktestVariantPreview(
                variant_index=0,
                variant_key="a" * 64,
                indicator_variant_key="b" * 64,
                total_return_pct=12.5,
                payload=BacktestVariantPayloadV1(
                    indicator_selections=(
                        IndicatorVariantSelection(
                            indicator_id="ma.sma",
                            inputs={"source": "close"},
                            params={"window": 20},
                        ),
                    ),
                    signal_params={"ma.sma": {"cross_up": 0.5}},
                    risk_params={"sl_enabled": True, "sl_pct": 2.0},
                    execution_params={"fee_pct": 0.075},
                    direction_mode="long-short",
                    sizing_mode="all_in",
                ),
            ),
        ),
        total_indicator_compute_calls=1,
        run_id=UUID("00000000-0000-0000-0000-000000000910"),
        state="succeeded",
        execution_mode="sync_inline",
        engine_version="signal_tf + 1m_risk",
        artifact_slot="slot_b",
        artifact_slot_generation=11,
        artifact_asof_date="2026-03-28",
        artifact_manifest_hash="c" * 64,
        spec_hash=build_sha256_from_payload(payload=snapshot_payload),
        engine_params_hash="d" * 64,
    )
    strategy_snapshot = BacktestStrategySnapshot(
        strategy_id=UUID("00000000-0000-0000-0000-000000000123"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000777"),
        is_deleted=False,
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={"window": ExplicitValuesSpec(name="window", values=(20,))},
            ),
        ),
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ma.sma",
                inputs={"source": "close"},
                params={"window": 20},
            ),
        ),
        spec_payload=snapshot_payload,
    )

    built = build_backtests_post_response(
        request=request,
        response=response,
        strategy_snapshot=strategy_snapshot,
        include_reports=False,
    )

    assert built.run_id == UUID("00000000-0000-0000-0000-000000000910")
    assert built.state == "succeeded"
    assert built.execution_mode == "sync_inline"
    assert built.engine_version == "signal_tf + 1m_risk"
    assert built.artifact_slot == "slot_b"
    assert built.artifact_slot_generation == 11
    assert built.artifact_asof_date == "2026-03-28"
    assert built.artifact_manifest_hash == "c" * 64
    assert built.spec_hash == build_sha256_from_payload(payload=snapshot_payload)
    assert built.engine_params_hash == "d" * 64
    assert "warmup_bars" not in built.model_dump(mode="json")
    assert "top_trades_n" not in built.model_dump(mode="json")


def test_build_backtests_post_response_maps_background_auto_launch_metadata() -> None:
    """
    Verify queued `background_auto` launch maps to the additive `/backtests` response contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Background auto launch is explicit and does not include inline-ranked variants.
    Raises:
        AssertionError: If additive queued launch metadata or empty variants policy drift.
    Side Effects:
        None.
    """
    request = BacktestsPostRequest.model_validate(
        {
            "time_range": {
                "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
            },
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1m",
                "indicator_grids": [
                    {
                        "indicator_id": "ma.sma",
                        "params": {"window": {"mode": "explicit", "values": [20]}},
                    }
                ],
            },
        }
    )
    response = RunBacktestResponse(
        mode="template",
        strategy_id=None,
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        top_k=1,
        preselect=100,
        variants=tuple(),
        total_indicator_compute_calls=0,
        run_id=UUID("00000000-0000-0000-0000-000000000911"),
        state="queued",
        execution_mode="background_auto",
        engine_version="signal_tf + 1m_risk",
        artifact_slot="slot_b",
        artifact_slot_generation=11,
        artifact_asof_date="2026-03-28",
        artifact_manifest_hash="c" * 64,
        engine_params_hash="d" * 64,
    )

    built = build_backtests_post_response(
        request=request,
        response=response,
        strategy_snapshot=None,
        include_reports=False,
    )

    assert built.run_id == UUID("00000000-0000-0000-0000-000000000911")
    assert built.state == "queued"
    assert built.execution_mode == "background_auto"
    assert built.engine_version == "signal_tf + 1m_risk"
    assert built.grid_request_hash is not None
    assert built.engine_params_hash == "d" * 64
    assert built.variants == []
    assert "warmup_bars" not in built.model_dump(mode="json")
    assert "top_trades_n" not in built.model_dump(mode="json")


def test_build_backtests_post_response_rejects_missing_persisted_sync_metadata() -> None:
    """
    Verify sync response mapper fails deterministically when persisted run metadata is absent.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Legacy non-persisted sync response contract is invalid after R7-02 cutover.
    Raises:
        AssertionError: If mapper silently accepts missing persisted metadata.
    Side Effects:
        None.
    """
    request = BacktestsPostRequest.model_validate(
        {
            "time_range": {
                "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
            },
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1m",
                "indicator_grids": [
                    {
                        "indicator_id": "ma.sma",
                        "params": {"window": {"mode": "explicit", "values": [20]}},
                    }
                ],
            },
        }
    )
    response = RunBacktestResponse(
        mode="template",
        strategy_id=None,
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        top_k=1,
        preselect=100,
        variants=(
            BacktestVariantPreview(
                variant_index=0,
                variant_key="a" * 64,
                indicator_variant_key="b" * 64,
                total_return_pct=12.5,
                payload=BacktestVariantPayloadV1(
                    indicator_selections=(
                        IndicatorVariantSelection(
                            indicator_id="ma.sma",
                            inputs={"source": "close"},
                            params={"window": 20},
                        ),
                    ),
                    signal_params={},
                    risk_params={},
                    execution_params={"fee_pct": 0.075},
                    direction_mode="long-short",
                    sizing_mode="all_in",
                ),
            ),
        ),
        total_indicator_compute_calls=1,
    )

    with pytest.raises(
        BacktestValidationError,
        match="requires persisted run metadata",
    ):
        build_backtests_post_response(
            request=request,
            response=response,
            strategy_snapshot=None,
            include_reports=False,
        )


def _variant_payload_request() -> dict[str, object]:
    """
    Build minimal explicit variant payload used by DTO variant-report mapping tests.

    Args:
        None.
    Returns:
        dict[str, object]: Explicit variant payload JSON object.
    Assumptions:
        One indicator selection is sufficient for mode-validation tests.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "indicator_selections": [
            {
                "indicator_id": "ma.sma",
                "inputs": {"source": "close"},
                "params": {"window": 20},
            }
        ],
        "signal_params": {"ma.sma": {"cross_up": 0.5}},
        "risk_params": {"sl_enabled": True},
        "execution_params": {"fee_pct": 0.075},
        "direction_mode": "long-short",
        "sizing_mode": "all_in",
    }
