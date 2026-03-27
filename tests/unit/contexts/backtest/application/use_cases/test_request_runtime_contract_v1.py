from __future__ import annotations

from typing import Mapping

import pytest

from trading.contexts.backtest.application.dto import RunBacktestTemplate
from trading.contexts.backtest.application.use_cases.request_runtime_contract_v1 import (
    validate_signal_overrides_default_only,
    validate_template_runtime_contract,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, Timeframe


class _RuntimeContractDefaultsProvider:
    """
    Minimal defaults-provider fake for shared R1 runtime-contract validation tests.
    """

    def compute_defaults(self, *, indicator_id: str) -> GridSpec | None:
        """
        Return compute defaults for supported indicators.

        Args:
            indicator_id: Requested indicator identifier.
        Returns:
            GridSpec | None: Defaults grid for supported indicators or `None`.
        Assumptions:
            Validation tests depend mainly on support catalog and signal defaults.
        Raises:
            None.
        Side Effects:
            None.
        """
        normalized_id = indicator_id.strip().lower()
        if normalized_id == "ma.sma":
            return GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                source=ExplicitValuesSpec(name="source", values=("close",)),
                params={"window": ExplicitValuesSpec(name="window", values=(20,))},
            )
        if normalized_id == "volume.obv":
            return GridSpec(
                indicator_id=IndicatorId("volume.obv"),
                params={},
            )
        return None

    def signal_param_defaults(self, *, indicator_id: str) -> Mapping[str, ExplicitValuesSpec]:
        """
        Return deterministic signal defaults mapping for supported indicators.

        Args:
            indicator_id: Requested indicator identifier.
        Returns:
            Mapping[str, ExplicitValuesSpec]: Signal defaults mapping or empty mapping.
        Assumptions:
            `ma.sma.cross_up=0.5` is the authoritative default-only contract for tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        if indicator_id.strip().lower() != "ma.sma":
            return {}
        return {"cross_up": ExplicitValuesSpec(name="cross_up", values=(0.5,))}

    def supported_indicator_ids(self) -> tuple[str, ...]:
        """
        Return deterministic supported indicator id catalog.

        Args:
            None.
        Returns:
            tuple[str, ...]: Supported indicator ids.
        Assumptions:
            Removed ids must be absent from this catalog.
        Raises:
            None.
        Side Effects:
            None.
        """
        return ("ma.sma", "volume.obv")

    def allowed_source_values(self, *, indicator_id: str) -> tuple[str, ...]:
        """
        Return deterministic allowed source values for one indicator id.

        Args:
            indicator_id: Requested indicator identifier.
        Returns:
            tuple[str, ...]: Allowed source values or empty tuple.
        Assumptions:
            Source catalog is not directly asserted in these tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        normalized_id = indicator_id.strip().lower()
        if normalized_id == "ma.sma":
            return ("close",)
        return ()


def test_validate_template_runtime_contract_rejects_unsupported_source_value() -> None:
    """
    Verify shared runtime-contract validator rejects explicit unsupported source values early.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `ma.sma` allows only the canonical `close` source in this fake defaults provider.
    Raises:
        AssertionError: If invalid source literals are not rejected with stable details.
    Side Effects:
        None.
    """
    with pytest.raises(BacktestValidationError) as error_info:
        validate_template_runtime_contract(
            template=_template(
                timeframe="15m",
                indicator_grids=(
                    GridSpec(
                        indicator_id=IndicatorId("ma.sma"),
                        source=ExplicitValuesSpec(name="source", values=("hl2",)),
                        params={"window": ExplicitValuesSpec(name="window", values=(20,))},
                    ),
                ),
            ),
            defaults_provider=_RuntimeContractDefaultsProvider(),
            allowed_request_timeframes=("15m",),
            forbidden_request_timeframes=("1m", "5m"),
            root_path="body.template",
        )

    assert error_info.value.errors == (
        {
            "path": "body.template.indicator_grids[0].source",
            "code": "unsupported_value",
            "message": "inputs.source must be one of: close",
        },
    )


def test_validate_template_runtime_contract_rejects_source_for_indicator_without_source_axis(
) -> None:
    """
    Verify shared runtime-contract validator rejects explicit source for no-source indicators.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `volume.obv` is supported by the fake catalog but does not expose configurable source.
    Raises:
        AssertionError: If unsupported source-axis usage is not rejected deterministically.
    Side Effects:
        None.
    """
    with pytest.raises(BacktestValidationError) as error_info:
        validate_template_runtime_contract(
            template=_template(
                timeframe="15m",
                indicator_grids=(
                    GridSpec(
                        indicator_id=IndicatorId("volume.obv"),
                        source=ExplicitValuesSpec(name="source", values=("close",)),
                        params={},
                    ),
                ),
            ),
            defaults_provider=_RuntimeContractDefaultsProvider(),
            allowed_request_timeframes=("15m",),
            forbidden_request_timeframes=("1m", "5m"),
            root_path="body.template",
        )

    assert error_info.value.errors == (
        {
            "path": "body.template.indicator_grids[0].source",
            "code": "unsupported_value",
            "message": "indicator_id 'volume.obv' does not support inputs.source",
        },
    )


def test_validate_template_runtime_contract_rejects_removed_indicator_id() -> None:
    """
    Verify shared R1 validator rejects removed indicator ids deterministically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Defaults-provider supported catalog is the source of truth for valid indicator ids.
    Raises:
        AssertionError: If removed ids are not rejected with canonical details.
    Side Effects:
        None.
    """
    with pytest.raises(BacktestValidationError) as error_info:
        validate_template_runtime_contract(
            template=_template(
                timeframe="15m",
                indicator_grids=(
                    GridSpec(
                        indicator_id=IndicatorId("momentum.macd"),
                        params={
                            "fast_window": ExplicitValuesSpec(name="fast_window", values=(12,))
                        },
                    ),
                ),
            ),
            defaults_provider=_RuntimeContractDefaultsProvider(),
            allowed_request_timeframes=(
                "15m",
                "30m",
                "1h",
                "2h",
                "4h",
                "6h",
                "8h",
                "1d",
                "2d",
                "3d",
            ),
            forbidden_request_timeframes=("1m", "5m"),
            root_path="body.template",
        )

    assert error_info.value.errors == (
        {
            "path": "body.template.indicator_grids[0].indicator_id",
            "code": "unsupported_value",
            "message": "indicator_id 'momentum.macd' is not supported",
        },
    )


def test_validate_template_runtime_contract_rejects_forbidden_request_timeframe() -> None:
    """
    Verify shared R1 validator rejects forbidden request timeframes deterministically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Allowed timeframe contract is frozen and excludes `1m`/`5m`.
    Raises:
        AssertionError: If forbidden timeframes are not rejected with canonical details.
    Side Effects:
        None.
    """
    with pytest.raises(BacktestValidationError) as error_info:
        validate_template_runtime_contract(
            template=_template(timeframe="1m"),
            defaults_provider=_RuntimeContractDefaultsProvider(),
            allowed_request_timeframes=(
                "15m",
                "30m",
                "1h",
                "2h",
                "4h",
                "6h",
                "8h",
                "1d",
                "2d",
                "3d",
            ),
            forbidden_request_timeframes=("1m", "5m"),
            root_path="body.template",
        )

    assert error_info.value.errors == (
        {
            "path": "body.template.timeframe",
            "code": "unsupported_value",
            "message": "timeframe must be one of: 15m, 30m, 1h, 2h, 4h, 6h, 8h, 1d, 2d, 3d",
        },
    )


def test_validate_template_runtime_contract_rejects_non_default_signal_params() -> None:
    """
    Verify shared R1 validator rejects non-default signal params in template mode.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Request-level signal grids must exactly match server defaults when present.
    Raises:
        AssertionError: If non-default signal params are not rejected with canonical details.
    Side Effects:
        None.
    """
    with pytest.raises(BacktestValidationError) as error_info:
        validate_template_runtime_contract(
            template=_template(
                timeframe="15m",
                signal_grids={
                    "ma.sma": {
                        "cross_up": ExplicitValuesSpec(name="cross_up", values=(0.6,))
                    }
                },
            ),
            defaults_provider=_RuntimeContractDefaultsProvider(),
            allowed_request_timeframes=("15m",),
            forbidden_request_timeframes=("1m", "5m"),
            root_path="body.template",
        )

    assert error_info.value.errors == (
        {
            "path": "body.template.signal_grids.ma.sma.cross_up",
            "code": "forbidden_override",
            "message": "signals.v1.params is default-only",
        },
    )


def test_validate_signal_overrides_default_only_rejects_saved_mode_override() -> None:
    """
    Verify saved-mode signal overrides are rejected when they differ from server defaults.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Saved strategies may omit overrides, but provided overrides must equal defaults.
    Raises:
        AssertionError: If saved-mode overrides are not rejected with canonical details.
    Side Effects:
        None.
    """
    with pytest.raises(BacktestValidationError) as error_info:
        validate_signal_overrides_default_only(
            signal_grids={
                "ma.sma": {
                    "cross_up": ExplicitValuesSpec(name="cross_up", values=(0.4, 0.6))
                }
            },
            defaults_provider=_RuntimeContractDefaultsProvider(),
            root_path="body.overrides.signal_grids",
        )

    assert error_info.value.errors == (
        {
            "path": "body.overrides.signal_grids.ma.sma.cross_up",
            "code": "forbidden_override",
            "message": "signals.v1.params is default-only",
        },
    )


def _template(
    *,
    timeframe: str,
    indicator_grids: tuple[GridSpec, ...] | None = None,
    signal_grids: Mapping[str, Mapping[str, ExplicitValuesSpec]] | None = None,
) -> RunBacktestTemplate:
    """
    Build deterministic template fixture for runtime-contract validation tests.

    Args:
        timeframe: Template timeframe literal.
        indicator_grids: Optional indicator grids tuple.
        signal_grids: Optional signal-grid mapping.
    Returns:
        RunBacktestTemplate: Minimal valid template fixture.
    Assumptions:
        `ma.sma` is the only supported indicator in the fake defaults provider.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe(timeframe),
        indicator_grids=indicator_grids
        or (
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={"window": ExplicitValuesSpec(name="window", values=(20,))},
            ),
        ),
        signal_grids=signal_grids,
    )
