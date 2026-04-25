from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest.application.services import (
    IndicatorSignalEvaluationInputV1,
    evaluate_indicator_signal_encoded_v1,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    SIGNAL_CODE_LONG_V2,
    SIGNAL_CODE_NEUTRAL_V2,
    SIGNAL_CODE_SHORT_V2,
    SIGNAL_CODE_VALUE_SET_V2,
    SignalRuleEvaluationRequestV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.signal_rules_engine_v2 import (
    BacktestSignalRulesEngineV2,
    list_signal_rule_registry_v2,
    supported_indicator_ids_for_signal_rules_v2,
)
from trading.contexts.indicators.application.dto import CandleArrays
from trading.contexts.indicators.domain.specifications import GridParamSpec, GridSpec
from trading.shared_kernel.primitives import MarketId, Symbol, Timeframe, TimeRange, UtcTimestamp


def _build_candles(
    *,
    open_values: tuple[float, ...],
    high_values: tuple[float, ...],
    low_values: tuple[float, ...],
    close_values: tuple[float, ...],
    volume_values: tuple[float, ...],
) -> CandleArrays:
    """
    Build deterministic synthetic candles fixture for v2 signal-rules tests.

    Args:
        open_values: Open series values.
        high_values: High series values.
        low_values: Low series values.
        close_values: Close series values.
        volume_values: Volume series values.
    Returns:
        CandleArrays: Synthetic aligned candle arrays fixture.
    Assumptions:
        All input tuples have equal length and represent one bar-aligned timeline.
    Raises:
        ValueError: If CandleArrays construction invariants are violated.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
    """
    bars = len(close_values)
    ts_open = np.asarray([int(index * 60_000) for index in range(bars)], dtype=np.int64)
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 3, 26, 0, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 3, 26, 0, bars, tzinfo=timezone.utc)),
        ),
        timeframe=Timeframe("1m"),
        ts_open=ts_open,
        open=np.asarray(open_values, dtype=np.float32),
        high=np.asarray(high_values, dtype=np.float32),
        low=np.asarray(low_values, dtype=np.float32),
        close=np.asarray(close_values, dtype=np.float32),
        volume=np.asarray(volume_values, dtype=np.float32),
    )


@pytest.fixture(scope="module")
def prod_defaults_provider() -> YamlBacktestGridDefaultsProvider:
    """
    Load the prod indicator defaults provider used as authoritative v2 signal catalog.

    Args:
        None.
    Returns:
        YamlBacktestGridDefaultsProvider: Loaded defaults provider.
    Assumptions:
        Repository-local `configs/prod/indicators.yaml` is available during unit tests.
    Raises:
        FileNotFoundError: If the prod defaults config is missing.
        ValueError: If the config cannot be parsed deterministically.
    Side Effects:
        Reads repository-local YAML config.
    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/adapters/outbound/defaults/
        indicators_yaml_defaults_provider.py
    """
    return YamlBacktestGridDefaultsProvider.from_yaml(
        config_path=Path("configs/prod/indicators.yaml")
    )


@pytest.fixture(scope="module")
def signal_rules_engine_v2(
    prod_defaults_provider: YamlBacktestGridDefaultsProvider,
) -> BacktestSignalRulesEngineV2:
    """
    Build the production-like v2 signal-rules engine for unit tests.

    Args:
        prod_defaults_provider: Authoritative prod defaults provider fixture.
    Returns:
        BacktestSignalRulesEngineV2: Startup-validated signal-rules engine.
    Assumptions:
        Engine startup validation should succeed against the prod catalog fixture.
    Raises:
        ValueError: If startup contract drift is detected.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - configs/prod/indicators.yaml
    """
    return BacktestSignalRulesEngineV2(defaults_provider=prod_defaults_provider)


def test_signal_rules_engine_v2_keeps_zero_axis_signal_targets_in_supported_catalog() -> None:
    """
    Verify the approved zero-axis signal targets remain present in the explicit v2 rules catalog.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Precompute compatibility must not remove these indicators from the v1/v2 signal catalog.
    Raises:
        AssertionError: If one target disappears or its rule family drifts.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
    """
    registry = dict(list_signal_rule_registry_v2())

    assert registry["structure.candle_stats"] == "candle_body_direction"
    assert registry["volatility.tr"] == "delta_sign"
    assert registry["volume.ad_line"] == "delta_sign"
    assert registry["volume.obv"] == "delta_sign"
    assert "structure.candle_stats" in supported_indicator_ids_for_signal_rules_v2()
    assert "volatility.tr" in supported_indicator_ids_for_signal_rules_v2()
    assert "volume.ad_line" in supported_indicator_ids_for_signal_rules_v2()
    assert "volume.obv" in supported_indicator_ids_for_signal_rules_v2()


@dataclass(frozen=True, slots=True)
class _MissingCatalogIndicatorProvider:
    """
    Provider wrapper used to simulate indicator-catalog drift for startup fail-fast tests.

    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/adapters/outbound/defaults/
        indicators_yaml_defaults_provider.py
      - tests/unit/contexts/backtest/application/services/v2/test_signal_rules_engine_v2.py
    """

    delegate: YamlBacktestGridDefaultsProvider

    def compute_defaults(self, *, indicator_id: str) -> GridSpec | None:
        """
        Proxy compute defaults to the wrapped provider.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            GridSpec | None: Wrapped provider result.
        Assumptions:
            Only `supported_indicator_ids` is intentionally modified in this wrapper.
        Raises:
            ValueError: Propagated from the wrapped provider.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/defaults/
            indicators_yaml_defaults_provider.py
        """
        return self.delegate.compute_defaults(indicator_id=indicator_id)

    def signal_param_defaults(self, *, indicator_id: str) -> Mapping[str, GridParamSpec]:
        """
        Proxy signal defaults to the wrapped provider.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            Mapping[str, GridParamSpec]: Wrapped provider result.
        Assumptions:
            Signal defaults themselves are not modified by this wrapper.
        Raises:
            ValueError: Propagated from the wrapped provider.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/defaults/
            indicators_yaml_defaults_provider.py
        """
        return self.delegate.signal_param_defaults(indicator_id=indicator_id)

    def supported_indicator_ids(self) -> tuple[str, ...]:
        """
        Return a deliberately drifted supported-indicator catalog.

        Args:
            None.
        Returns:
            tuple[str, ...]: Catalog missing one supported indicator id.
        Assumptions:
            Removing one item is enough to exercise startup fail-fast behavior.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
        """
        return tuple(
            indicator_id
            for indicator_id in self.delegate.supported_indicator_ids()
            if indicator_id != "trend.vortex"
        )

    def allowed_source_values(self, *, indicator_id: str) -> tuple[str, ...]:
        """
        Proxy source catalog lookups to the wrapped provider.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            tuple[str, ...]: Wrapped provider result.
        Assumptions:
            Source catalogs are not modified by this wrapper.
        Raises:
            ValueError: Propagated from the wrapped provider.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/defaults/
            indicators_yaml_defaults_provider.py
        """
        return self.delegate.allowed_source_values(indicator_id=indicator_id)


@dataclass(frozen=True, slots=True)
class _MissingSignalDefaultsProvider:
    """
    Provider wrapper used to simulate missing default-only signal params at startup.

    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/use_cases/request_runtime_contract_v1.py
      - tests/unit/contexts/backtest/application/services/v2/test_signal_rules_engine_v2.py
    """

    delegate: YamlBacktestGridDefaultsProvider

    def compute_defaults(self, *, indicator_id: str) -> GridSpec | None:
        """
        Proxy compute defaults to the wrapped provider.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            GridSpec | None: Wrapped provider result.
        Assumptions:
            Only signal defaults are intentionally modified in this wrapper.
        Raises:
            ValueError: Propagated from the wrapped provider.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/defaults/
            indicators_yaml_defaults_provider.py
        """
        return self.delegate.compute_defaults(indicator_id=indicator_id)

    def signal_param_defaults(self, *, indicator_id: str) -> Mapping[str, GridParamSpec]:
        """
        Return missing defaults for `trend.adx` to exercise startup fail-fast behavior.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            Mapping[str, GridParamSpec]: Wrapped provider defaults or empty mapping for `trend.adx`.
        Assumptions:
            `trend.adx` requires `long_delta_periods` and `short_delta_periods`.
        Raises:
            ValueError: Propagated from the wrapped provider.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
        """
        if indicator_id == "trend.adx":
            return {}
        return self.delegate.signal_param_defaults(indicator_id=indicator_id)

    def supported_indicator_ids(self) -> tuple[str, ...]:
        """
        Proxy the full supported-indicator catalog to the wrapped provider.

        Args:
            None.
        Returns:
            tuple[str, ...]: Wrapped provider catalog.
        Assumptions:
            Only signal defaults drift, not the indicator-id catalog.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/defaults/
            indicators_yaml_defaults_provider.py
        """
        return self.delegate.supported_indicator_ids()

    def allowed_source_values(self, *, indicator_id: str) -> tuple[str, ...]:
        """
        Proxy source catalog lookups to the wrapped provider.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            tuple[str, ...]: Wrapped provider result.
        Assumptions:
            Source catalogs are not modified by this wrapper.
        Raises:
            ValueError: Propagated from the wrapped provider.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/defaults/
            indicators_yaml_defaults_provider.py
        """
        return self.delegate.allowed_source_values(indicator_id=indicator_id)


def test_registry_matches_defaults_catalog_across_envs() -> None:
    """
    Verify the explicit v2 registry matches supported indicator catalogs in all target envs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `dev`, `test`, and `prod` configs should expose the same supported indicator ids for R4-01.
    Raises:
        AssertionError: If the explicit v2 registry drifts from any env defaults catalog.
    Side Effects:
        Reads repository-local YAML config files.
    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - configs/dev/indicators.yaml
      - configs/test/indicators.yaml
      - configs/prod/indicators.yaml
    """
    expected_indicator_ids = supported_indicator_ids_for_signal_rules_v2()
    registry_pairs = list_signal_rule_registry_v2()
    assert registry_pairs == tuple(sorted(registry_pairs))
    for env_name in ("dev", "test", "prod"):
        provider = YamlBacktestGridDefaultsProvider.from_yaml(
            config_path=Path(f"configs/{env_name}/indicators.yaml")
        )
        assert provider.supported_indicator_ids() == expected_indicator_ids


def test_engine_startup_fails_fast_on_indicator_catalog_drift(
    prod_defaults_provider: YamlBacktestGridDefaultsProvider,
) -> None:
    """
    Verify engine construction fails fast when defaults-provider catalog drifts from the registry.

    Args:
        prod_defaults_provider: Authoritative prod defaults provider fixture.
    Returns:
        None.
    Assumptions:
        Startup validation must reject missing explicit rule coverage before runtime/precompute use.
    Raises:
        AssertionError: If catalog drift does not raise `ValueError`.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_signal_rules_engine_v2.py
    """
    with pytest.raises(ValueError, match="catalog drift"):
        BacktestSignalRulesEngineV2(
            defaults_provider=_MissingCatalogIndicatorProvider(prod_defaults_provider)
        )


def test_engine_startup_fails_fast_on_missing_default_only_signal_params(
    prod_defaults_provider: YamlBacktestGridDefaultsProvider,
) -> None:
    """
    Verify engine construction fails fast when required `signals.v1.params` defaults are missing.

    Args:
        prod_defaults_provider: Authoritative prod defaults provider fixture.
    Returns:
        None.
    Assumptions:
        Required default-only params must be present before the engine is usable.
    Raises:
        AssertionError: If missing defaults do not raise `ValueError`.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_signal_rules_engine_v2.py
    """
    with pytest.raises(ValueError, match="missing default signals.v1.params"):
        BacktestSignalRulesEngineV2(
            defaults_provider=_MissingSignalDefaultsProvider(prod_defaults_provider)
        )


@pytest.mark.parametrize(
    ("indicator_id", "evaluation_request", "expected_codes"),
    (
        (
            "ma.ema",
            SignalRuleEvaluationRequestV2(
                indicator_id="ma.ema",
                candles=_build_candles(
                    open_values=(9.0, 9.0, 9.0, 9.0),
                    high_values=(20.0, 20.0, 20.0, 20.0),
                    low_values=(5.0, 5.0, 5.0, 5.0),
                    close_values=(8.0, 8.0, 8.0, 8.0),
                    volume_values=(100.0, 100.0, 100.0, 100.0),
                ),
                primary_output=np.asarray((7.0, 25.0, np.nan, 5.0), dtype=np.float32),
                inputs_source="high",
            ),
            np.asarray((1, -1, 0, 1), dtype=np.int8),
        ),
        (
            "momentum.rsi",
            SignalRuleEvaluationRequestV2(
                indicator_id="momentum.rsi",
                candles=_build_candles(
                    open_values=(1.0, 1.0, 1.0, 1.0),
                    high_values=(1.0, 1.0, 1.0, 1.0),
                    low_values=(1.0, 1.0, 1.0, 1.0),
                    close_values=(1.0, 1.0, 1.0, 1.0),
                    volume_values=(1.0, 1.0, 1.0, 1.0),
                ),
                primary_output=np.asarray((20.0, 50.0, 80.0, np.nan), dtype=np.float32),
            ),
            np.asarray((1, 0, -1, 0), dtype=np.int8),
        ),
        (
            "trend.linreg_slope",
            SignalRuleEvaluationRequestV2(
                indicator_id="trend.linreg_slope",
                candles=_build_candles(
                    open_values=(1.0, 1.0, 1.0, 1.0),
                    high_values=(1.0, 1.0, 1.0, 1.0),
                    low_values=(1.0, 1.0, 1.0, 1.0),
                    close_values=(1.0, 1.0, 1.0, 1.0),
                    volume_values=(1.0, 1.0, 1.0, 1.0),
                ),
                primary_output=np.asarray((1.0, 0.0, -1.0, np.nan), dtype=np.float32),
            ),
            np.asarray((1, 0, -1, 0), dtype=np.int8),
        ),
        (
            "trend.adx",
            SignalRuleEvaluationRequestV2(
                indicator_id="trend.adx",
                candles=_build_candles(
                    open_values=(1.0,) * 11,
                    high_values=(1.0,) * 11,
                    low_values=(1.0,) * 11,
                    close_values=(1.0,) * 11,
                    volume_values=(1.0,) * 11,
                ),
                primary_output=np.asarray(
                    (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, -5.0),
                    dtype=np.float32,
                ),
            ),
            np.asarray((0, 0, 0, 0, 0, 1, 1, 1, 0, 0, -1), dtype=np.int8),
        ),
        (
            "volume.volume_sma",
            SignalRuleEvaluationRequestV2(
                indicator_id="volume.volume_sma",
                candles=_build_candles(
                    open_values=(1.0, 1.0, 1.0, 1.0),
                    high_values=(1.0, 1.0, 1.0, 1.0),
                    low_values=(1.0, 1.0, 1.0, 1.0),
                    close_values=(1.0, 1.0, 1.0, 1.0),
                    volume_values=(100.0, 90.0, np.nan, 80.0),
                ),
                primary_output=np.asarray((90.0, 100.0, 100.0, 80.0), dtype=np.float32),
            ),
            np.asarray((1, -1, 0, 0), dtype=np.int8),
        ),
        (
            "structure.candle_stats",
            SignalRuleEvaluationRequestV2(
                indicator_id="structure.candle_stats",
                candles=_build_candles(
                    open_values=(1.0, 5.0, 4.0, 4.0),
                    high_values=(2.0, 5.0, 4.0, 4.0),
                    low_values=(1.0, 4.0, 4.0, 3.0),
                    close_values=(2.0, 4.0, 4.0, 3.0),
                    volume_values=(1.0, 1.0, 1.0, 1.0),
                ),
                primary_output=np.asarray((0.6, 0.7, 0.4, np.nan), dtype=np.float32),
            ),
            np.asarray((1, -1, 0, 0), dtype=np.int8),
        ),
        (
            "structure.pivots",
            SignalRuleEvaluationRequestV2(
                indicator_id="structure.pivots",
                candles=_build_candles(
                    open_values=(1.0, 1.0, 1.0, 1.0),
                    high_values=(1.0, 1.0, 1.0, 1.0),
                    low_values=(1.0, 1.0, 1.0, 1.0),
                    close_values=(1.0, 1.0, 1.0, 1.0),
                    volume_values=(1.0, 1.0, 1.0, 1.0),
                ),
                primary_output=np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32),
                dependency_outputs={
                    "structure.pivot_low": np.asarray(
                        (np.nan, 1.0, np.nan, 1.0),
                        dtype=np.float32,
                    ),
                    "structure.pivot_high": np.asarray(
                        (np.nan, np.nan, 1.0, 1.0),
                        dtype=np.float32,
                    ),
                },
            ),
            np.asarray((0, 1, -1, 0), dtype=np.int8),
        ),
        (
            "trend.vortex",
            SignalRuleEvaluationRequestV2(
                indicator_id="trend.vortex",
                candles=_build_candles(
                    open_values=(1.0, 1.0, 1.0, 1.0),
                    high_values=(1.0, 1.0, 1.0, 1.0),
                    low_values=(1.0, 1.0, 1.0, 1.0),
                    close_values=(1.0, 1.0, 1.0, 1.0),
                    volume_values=(1.0, 1.0, 1.0, 1.0),
                ),
                primary_output=np.asarray((1.2, 1.0, 0.85, np.nan), dtype=np.float32),
            ),
            np.asarray((1, 0, -1, 0), dtype=np.int8),
        ),
    ),
)
def test_rule_family_outputs_match_expected_semantics_and_v1_parity(
    signal_rules_engine_v2: BacktestSignalRulesEngineV2,
    indicator_id: str,
    evaluation_request: SignalRuleEvaluationRequestV2,
    expected_codes: np.ndarray,
) -> None:
    """
    Verify v2 engine semantics for every supported rule family and keep parity with v1 kernels.

    Args:
        signal_rules_engine_v2: Startup-validated v2 signal-rules engine fixture.
        indicator_id: Indicator identifier under test.
        evaluation_request: Deterministic evaluation request.
        expected_codes: Explicit expected compact signal codes.
    Returns:
        None.
    Assumptions:
        The explicit expected arrays represent the approved rule-family semantics for R4-01.
    Raises:
        AssertionError: If v2 output diverges from expected semantics or from v1 parity.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
    """
    result = signal_rules_engine_v2.evaluate(request=evaluation_request)

    assert result.indicator_id == indicator_id
    assert result.signal_codes.dtype == np.int8
    assert set(result.signal_codes.tolist()) <= set(SIGNAL_CODE_VALUE_SET_V2)
    assert result.signal_codes.tolist() == expected_codes.tolist()

    is_compare_price_rule = result.rule_family == "compare_price_to_output"
    v1_result = evaluate_indicator_signal_encoded_v1(
        candles=evaluation_request.candles,
        indicator_input=IndicatorSignalEvaluationInputV1(
            indicator_id=indicator_id,
            primary_output=evaluation_request.primary_output,
            indicator_inputs=(
                {"source": result.inputs_source}
                if result.inputs_source is not None and is_compare_price_rule
                else {}
            ),
            signal_params=result.signal_params,
            dependency_outputs=evaluation_request.dependency_outputs,
        ),
    )
    assert v1_result.tolist() == result.signal_codes.tolist()


def test_compare_price_rule_respects_explicit_inputs_source_axis(
    signal_rules_engine_v2: BacktestSignalRulesEngineV2,
) -> None:
    """
    Verify compare-price rules use explicit `inputs.source` and default to `close` when omitted.

    Args:
        signal_rules_engine_v2: Startup-validated v2 signal-rules engine fixture.
    Returns:
        None.
    Assumptions:
        `ma.ema` supports explicit source-axis values from the prod defaults catalog.
    Raises:
        AssertionError: If `inputs.source` resolution is not explicit/deterministic.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - configs/prod/indicators.yaml
    """
    candles = _build_candles(
        open_values=(9.0, 9.0, 9.0, 9.0),
        high_values=(20.0, 20.0, 20.0, 20.0),
        low_values=(5.0, 5.0, 5.0, 5.0),
        close_values=(8.0, 8.0, 8.0, 8.0),
        volume_values=(100.0, 100.0, 100.0, 100.0),
    )
    primary_output = np.asarray((15.0, 25.0, 15.0, 5.0), dtype=np.float32)

    default_source_result = signal_rules_engine_v2.evaluate(
        request=SignalRuleEvaluationRequestV2(
            indicator_id="ma.ema",
            candles=candles,
            primary_output=primary_output,
        )
    )
    high_source_result = signal_rules_engine_v2.evaluate(
        request=SignalRuleEvaluationRequestV2(
            indicator_id="ma.ema",
            candles=candles,
            primary_output=primary_output,
            inputs_source="high",
        )
    )

    assert default_source_result.inputs_source == "close"
    assert high_source_result.inputs_source == "high"
    assert default_source_result.signal_codes.tolist() == [
        SIGNAL_CODE_SHORT_V2,
        SIGNAL_CODE_SHORT_V2,
        SIGNAL_CODE_SHORT_V2,
        SIGNAL_CODE_LONG_V2,
    ]
    assert high_source_result.signal_codes.tolist() == [
        SIGNAL_CODE_LONG_V2,
        SIGNAL_CODE_SHORT_V2,
        SIGNAL_CODE_LONG_V2,
        SIGNAL_CODE_LONG_V2,
    ]


def test_default_only_signal_params_fill_defaults_and_allow_matching_subset(
    signal_rules_engine_v2: BacktestSignalRulesEngineV2,
) -> None:
    """
    Verify omitted or matching-subset `signals.v1.params` resolve to authoritative defaults.

    Args:
        signal_rules_engine_v2: Startup-validated v2 signal-rules engine fixture.
    Returns:
        None.
    Assumptions:
        `momentum.rsi` defaults are `long_threshold=30`, `short_threshold=70` in prod config.
    Raises:
        AssertionError: If defaults are not filled deterministically.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/use_cases/request_runtime_contract_v1.py
    """
    candles = _build_candles(
        open_values=(1.0, 1.0, 1.0),
        high_values=(1.0, 1.0, 1.0),
        low_values=(1.0, 1.0, 1.0),
        close_values=(1.0, 1.0, 1.0),
        volume_values=(1.0, 1.0, 1.0),
    )
    values = np.asarray((20.0, 50.0, 80.0), dtype=np.float32)

    without_overrides = signal_rules_engine_v2.evaluate(
        request=SignalRuleEvaluationRequestV2(
            indicator_id="momentum.rsi",
            candles=candles,
            primary_output=values,
        )
    )
    with_matching_subset = signal_rules_engine_v2.evaluate(
        request=SignalRuleEvaluationRequestV2(
            indicator_id="momentum.rsi",
            candles=candles,
            primary_output=values,
            signal_params={"long_threshold": 30},
        )
    )

    assert dict(without_overrides.signal_params) == {
        "long_threshold": 30,
        "short_threshold": 70,
    }
    assert dict(with_matching_subset.signal_params) == {
        "long_threshold": 30,
        "short_threshold": 70,
    }
    assert without_overrides.signal_codes.tolist() == [
        SIGNAL_CODE_LONG_V2,
        SIGNAL_CODE_NEUTRAL_V2,
        SIGNAL_CODE_SHORT_V2,
    ]
    assert with_matching_subset.signal_codes.tolist() == without_overrides.signal_codes.tolist()


def test_non_default_signal_params_are_rejected_deterministically(
    signal_rules_engine_v2: BacktestSignalRulesEngineV2,
) -> None:
    """
    Verify non-default `signals.v1.params` overrides are rejected by the v2 engine.

    Args:
        signal_rules_engine_v2: Startup-validated v2 signal-rules engine fixture.
    Returns:
        None.
    Assumptions:
        R4-01 keeps signal params strictly `default-only`.
    Raises:
        AssertionError: If non-default overrides do not raise `ValueError`.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/use_cases/request_runtime_contract_v1.py
    """
    with pytest.raises(ValueError, match="signals.v1.params is default-only"):
        signal_rules_engine_v2.evaluate(
            request=SignalRuleEvaluationRequestV2(
                indicator_id="momentum.rsi",
                candles=_build_candles(
                    open_values=(1.0, 1.0, 1.0),
                    high_values=(1.0, 1.0, 1.0),
                    low_values=(1.0, 1.0, 1.0),
                    close_values=(1.0, 1.0, 1.0),
                    volume_values=(1.0, 1.0, 1.0),
                ),
                primary_output=np.asarray((20.0, 50.0, 80.0), dtype=np.float32),
                signal_params={"long_threshold": 25},
            )
        )


def test_invalid_inputs_source_literal_is_rejected_against_defaults_catalog(
    signal_rules_engine_v2: BacktestSignalRulesEngineV2,
) -> None:
    """
    Verify invalid `inputs.source` literals are rejected deterministically.

    Args:
        signal_rules_engine_v2: Startup-validated v2 signal-rules engine fixture.
    Returns:
        None.
    Assumptions:
        `ma.ema` allows explicit source values from prod defaults but not `hl2`.
    Raises:
        AssertionError: If unsupported source values do not raise `ValueError`.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - configs/prod/indicators.yaml
    """
    with pytest.raises(ValueError, match="inputs.source 'hl2' is not allowed"):
        signal_rules_engine_v2.evaluate(
            request=SignalRuleEvaluationRequestV2(
                indicator_id="ma.ema",
                candles=_build_candles(
                    open_values=(1.0, 1.0, 1.0),
                    high_values=(1.0, 1.0, 1.0),
                    low_values=(1.0, 1.0, 1.0),
                    close_values=(1.0, 1.0, 1.0),
                    volume_values=(1.0, 1.0, 1.0),
                ),
                primary_output=np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
                inputs_source="hl2",
            )
        )
