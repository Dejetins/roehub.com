from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.strategy.domain.entities import (
    StrategySignalAction,
    StrategySignalOutcome,
    StrategySignalSide,
    StrategySpecV1,
)

EVALUATOR_VERSION_V1 = "ma_cross_close_v1"


@dataclass(frozen=True, slots=True)
class SignalEvaluatorDecision:
    action: StrategySignalAction
    side: StrategySignalSide | None
    outcome: StrategySignalOutcome
    reason_code: str
    confidence: Decimal | None
    metadata_json: dict[str, Any]
    evaluator_version: str = EVALUATOR_VERSION_V1


def evaluate_strategy_signal(
    *,
    spec: StrategySpecV1,
    metadata_json: Mapping[str, Any],
    candle: CandleWithMeta,
    warmup_satisfied: bool,
) -> SignalEvaluatorDecision:
    """
    Evaluate the supported Stage 05 live signal subset.

    Supported v1 subset:
    - one MA indicator with `params.fast` and `params.slow`;
    - signal template equivalent to the persisted `MA(fast,slow)` strategy spec.
    """
    evaluator_state = _read_evaluator_state(metadata_json=metadata_json)
    support = _parse_supported_ma_cross(spec=spec)
    if support is None:
        return SignalEvaluatorDecision(
            action="none",
            side=None,
            outcome="blocked",
            reason_code="unsupported_live_evaluator",
            confidence=None,
            metadata_json=_with_evaluator_state(
                metadata_json=metadata_json,
                evaluator_state=evaluator_state,
            ),
        )

    close_value = Decimal(str(candle.candle.close))
    closes = [*evaluator_state.closes, close_value]
    closes = closes[-support.slow :]
    next_state = _EvaluatorState(closes=tuple(closes), last_relation=evaluator_state.last_relation)

    if not warmup_satisfied or len(closes) < support.slow:
        return SignalEvaluatorDecision(
            action="none",
            side=None,
            outcome="warmup",
            reason_code="warmup_not_satisfied",
            confidence=None,
            metadata_json=_with_evaluator_state(
                metadata_json=metadata_json,
                evaluator_state=next_state,
            ),
        )

    fast_ma = _mean(closes[-support.fast :])
    slow_ma = _mean(closes[-support.slow :])
    relation = _relation(fast_ma=fast_ma, slow_ma=slow_ma)
    next_state = _EvaluatorState(closes=tuple(closes), last_relation=relation)

    if evaluator_state.last_relation is None:
        return SignalEvaluatorDecision(
            action="none",
            side=None,
            outcome="no_signal",
            reason_code="ma_cross_baseline_ready",
            confidence=None,
            metadata_json=_with_evaluator_state(
                metadata_json=metadata_json,
                evaluator_state=next_state,
            ),
        )

    if evaluator_state.last_relation != "above" and relation == "above":
        return SignalEvaluatorDecision(
            action="open",
            side="buy",
            outcome="signal",
            reason_code="ma_fast_crossed_above_slow",
            confidence=Decimal("1"),
            metadata_json=_with_evaluator_state(
                metadata_json=metadata_json,
                evaluator_state=next_state,
            ),
        )
    if evaluator_state.last_relation != "below" and relation == "below":
        return SignalEvaluatorDecision(
            action="close",
            side="sell",
            outcome="signal",
            reason_code="ma_fast_crossed_below_slow",
            confidence=Decimal("1"),
            metadata_json=_with_evaluator_state(
                metadata_json=metadata_json,
                evaluator_state=next_state,
            ),
        )

    return SignalEvaluatorDecision(
        action="none",
        side=None,
        outcome="no_signal",
        reason_code="ma_cross_no_change",
        confidence=None,
        metadata_json=_with_evaluator_state(
            metadata_json=metadata_json,
            evaluator_state=next_state,
        ),
    )


@dataclass(frozen=True, slots=True)
class _SupportedMaCross:
    fast: int
    slow: int


@dataclass(frozen=True, slots=True)
class _EvaluatorState:
    closes: tuple[Decimal, ...]
    last_relation: str | None


def _parse_supported_ma_cross(*, spec: StrategySpecV1) -> _SupportedMaCross | None:
    if len(spec.indicators) != 1:
        return None
    indicator = spec.indicators[0]
    name = str(indicator.get("name") or indicator.get("kind") or indicator.get("id") or "")
    if name.strip().upper() != "MA":
        return None
    params = indicator.get("params")
    if not isinstance(params, Mapping):
        return None
    try:
        fast = int(params["fast"])
        slow = int(params["slow"])
    except (KeyError, TypeError, ValueError):
        return None
    if fast <= 0 or slow <= 0 or fast >= slow:
        return None
    expected_template = f"MA({fast},{slow})"
    if spec.signal_template.strip().upper().replace(" ", "") != expected_template:
        return None
    return _SupportedMaCross(fast=fast, slow=slow)


def _read_evaluator_state(*, metadata_json: Mapping[str, Any]) -> _EvaluatorState:
    raw = metadata_json.get("signal_evaluator")
    if not isinstance(raw, Mapping):
        return _EvaluatorState(closes=(), last_relation=None)
    raw_closes = raw.get("closes")
    closes: list[Decimal] = []
    if isinstance(raw_closes, list):
        for value in raw_closes:
            try:
                closes.append(Decimal(str(value)))
            except Exception:  # noqa: BLE001
                return _EvaluatorState(closes=(), last_relation=None)
    raw_relation = raw.get("last_relation")
    relation = str(raw_relation) if raw_relation in {"above", "below", "equal"} else None
    return _EvaluatorState(closes=tuple(closes), last_relation=relation)


def _with_evaluator_state(
    *,
    metadata_json: Mapping[str, Any],
    evaluator_state: _EvaluatorState,
) -> dict[str, Any]:
    next_metadata = dict(metadata_json)
    next_metadata["signal_evaluator"] = {
        "algorithm": EVALUATOR_VERSION_V1,
        "closes": [str(value) for value in evaluator_state.closes],
        "last_relation": evaluator_state.last_relation or "",
    }
    return next_metadata


def _mean(values: list[Decimal]) -> Decimal:
    return sum(values, Decimal("0")) / Decimal(len(values))


def _relation(*, fast_ma: Decimal, slow_ma: Decimal) -> str:
    if fast_ma > slow_ma:
        return "above"
    if fast_ma < slow_ma:
        return "below"
    return "equal"
