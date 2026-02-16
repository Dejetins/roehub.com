from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trading.contexts.strategy.domain.errors import StrategySpecValidationError
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, Timeframe

STRATEGY_SPEC_KIND_V1 = "roehub.strategy.v1"
_ALLOWED_MARKET_TYPES = {"spot", "futures"}


@dataclass(frozen=True, slots=True)
class StrategySpecV1:
    """
    StrategySpecV1 — immutable schema-versioned specification payload for Strategy v1.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/domain/services/strategy_name.py
      - src/trading/contexts/strategy/domain/entities/strategy.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """

    instrument_id: InstrumentId
    instrument_key: str
    market_type: str
    timeframe: Timeframe
    indicators: tuple[Mapping[str, Any], ...]
    signal_template: str
    schema_version: int = 1
    spec_kind: str = STRATEGY_SPEC_KIND_V1

    def __post_init__(self) -> None:
        """
        Validate immutable StrategySpecV1 invariants and canonical tags.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `timeframe` is validated by shared-kernel primitive and includes `1m`.
        Raises:
            StrategySpecValidationError: If schema version, tags, or payload shape are invalid.
        Side Effects:
            None.
        """
        if self.schema_version != 1:
            raise StrategySpecValidationError(
                f"StrategySpecV1.schema_version must be 1, got {self.schema_version!r}"
            )
        if self.spec_kind != STRATEGY_SPEC_KIND_V1:
            raise StrategySpecValidationError(
                f"StrategySpecV1.spec_kind must be {STRATEGY_SPEC_KIND_V1!r}"
            )

        normalized_key = self.instrument_key.strip()
        if not normalized_key:
            raise StrategySpecValidationError("StrategySpecV1.instrument_key must be non-empty")
        key_parts = normalized_key.split(":")
        if len(key_parts) != 3:
            raise StrategySpecValidationError(
                "StrategySpecV1.instrument_key must follow '{exchange}:{market_type}:{symbol}'"
            )
        symbol_from_key = key_parts[2].strip().upper()
        if symbol_from_key != str(self.instrument_id.symbol):
            raise StrategySpecValidationError(
                "StrategySpecV1.instrument_key symbol must match instrument_id.symbol"
            )

        if self.market_type not in _ALLOWED_MARKET_TYPES:
            raise StrategySpecValidationError(
                f"StrategySpecV1.market_type must be one of {_ALLOWED_MARKET_TYPES}"
            )
        key_market_type = key_parts[1].strip().lower()
        if key_market_type != self.market_type:
            raise StrategySpecValidationError(
                "StrategySpecV1.market_type must match market_type part of instrument_key"
            )

        if not self.signal_template.strip():
            raise StrategySpecValidationError("StrategySpecV1.signal_template must be non-empty")

        for indicator in self.indicators:
            _validate_indicator_payload(indicator=indicator)

    @classmethod
    def from_json(cls, *, payload: Mapping[str, Any]) -> StrategySpecV1:
        """
        Build immutable StrategySpecV1 from persisted JSON payload.

        Args:
            payload: Persisted strategy spec mapping.
        Returns:
            StrategySpecV1: Parsed immutable domain object.
        Assumptions:
            Input follows `spec_json` contract from Strategy v1 architecture doc.
        Raises:
            StrategySpecValidationError: If required fields are absent or malformed.
        Side Effects:
            None.
        """
        instrument_id = _parse_instrument_id(payload=payload)
        timeframe_raw = payload.get("timeframe")
        if not isinstance(timeframe_raw, str):
            raise StrategySpecValidationError("StrategySpecV1 payload.timeframe must be string")

        market_type_raw = payload.get("market_type")
        if not isinstance(market_type_raw, str):
            raise StrategySpecValidationError("StrategySpecV1 payload.market_type must be string")
        market_type = market_type_raw.strip().lower()

        instrument_key_raw = payload.get("instrument_key")
        if not isinstance(instrument_key_raw, str):
            raise StrategySpecValidationError(
                "StrategySpecV1 payload.instrument_key must be string"
            )
        instrument_key = instrument_key_raw.strip()

        indicators_payload = payload.get("indicators", ())
        indicators = _parse_indicators(indicators_payload=indicators_payload)

        signal_template_raw = payload.get("signal_template")
        if isinstance(signal_template_raw, str) and signal_template_raw.strip():
            signal_template = signal_template_raw.strip()
        else:
            signal_template = _build_default_signal_template(indicators=indicators)

        schema_version_raw = payload.get("schema_version", 1)
        if not isinstance(schema_version_raw, int):
            raise StrategySpecValidationError(
                "StrategySpecV1 payload.schema_version must be integer"
            )

        spec_kind_raw = payload.get("spec_kind", STRATEGY_SPEC_KIND_V1)
        if not isinstance(spec_kind_raw, str):
            raise StrategySpecValidationError("StrategySpecV1 payload.spec_kind must be string")

        return cls(
            instrument_id=instrument_id,
            instrument_key=instrument_key,
            market_type=market_type,
            timeframe=Timeframe(timeframe_raw),
            indicators=indicators,
            signal_template=signal_template,
            schema_version=schema_version_raw,
            spec_kind=spec_kind_raw.strip(),
        )

    def to_json(self) -> dict[str, Any]:
        """
        Serialize spec into deterministic JSON mapping for persistence and hashing.

        Args:
            None.
        Returns:
            dict[str, Any]: Canonical strategy spec JSON payload.
        Assumptions:
            JSON must contain `schema_version` and `spec_kind` literals for v1 contracts.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "spec_kind": self.spec_kind,
            "schema_version": self.schema_version,
            "instrument_id": self.instrument_id.as_dict(),
            "instrument_key": self.instrument_key,
            "market_type": self.market_type,
            "symbol": str(self.instrument_id.symbol),
            "timeframe": self.timeframe.code,
            "signal_template": self.signal_template,
            "indicators": [
                _clone_indicator_payload(indicator=indicator) for indicator in self.indicators
            ],
            "tags": {
                "symbol": str(self.instrument_id.symbol),
                "market_type": self.market_type,
                "timeframe": self.timeframe.code,
            },
        }

    def canonical_json(self) -> str:
        """
        Return canonical JSON string used by deterministic name-hash generation.

        Args:
            None.
        Returns:
            str: Stable sorted JSON string.
        Assumptions:
            Canonical serialization must not depend on insertion order of dictionaries.
        Raises:
            TypeError: If payload cannot be serialized to JSON.
        Side Effects:
            None.
        """
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)



def _parse_instrument_id(*, payload: Mapping[str, Any]) -> InstrumentId:
    """
    Parse `instrument_id` mapping from StrategySpecV1 JSON payload.

    Args:
        payload: Source spec mapping.
    Returns:
        InstrumentId: Parsed domain instrument identifier.
    Assumptions:
        `instrument_id` contains integer `market_id` and non-empty `symbol`.
    Raises:
        StrategySpecValidationError: If payload cannot be converted.
    Side Effects:
        None.
    """
    raw_instrument_id = payload.get("instrument_id")
    if not isinstance(raw_instrument_id, Mapping):
        raise StrategySpecValidationError("StrategySpecV1 payload.instrument_id must be object")

    market_id_raw = raw_instrument_id.get("market_id")
    symbol_raw = raw_instrument_id.get("symbol")
    if not isinstance(market_id_raw, int):
        raise StrategySpecValidationError("StrategySpecV1 instrument_id.market_id must be integer")
    if not isinstance(symbol_raw, str):
        raise StrategySpecValidationError("StrategySpecV1 instrument_id.symbol must be string")

    try:
        return InstrumentId(
            market_id=MarketId(market_id_raw),
            symbol=Symbol(symbol_raw),
        )
    except ValueError as error:
        raise StrategySpecValidationError("StrategySpecV1 instrument_id is invalid") from error



def _parse_indicators(*, indicators_payload: Any) -> tuple[Mapping[str, Any], ...]:
    """
    Parse indicators payload into immutable tuple of mapping snapshots.

    Args:
        indicators_payload: Raw `indicators` value from JSON payload.
    Returns:
        tuple[Mapping[str, Any], ...]: Deterministic tuple of indicator mappings.
    Assumptions:
        Each indicator is mapping with stable JSON-serializable values.
    Raises:
        StrategySpecValidationError: If indicator payload shape is invalid.
    Side Effects:
        None.
    """
    if not isinstance(indicators_payload, Sequence) or isinstance(indicators_payload, (str, bytes)):
        raise StrategySpecValidationError("StrategySpecV1 payload.indicators must be array")

    indicators: list[Mapping[str, Any]] = []
    for indicator_payload in indicators_payload:
        if not isinstance(indicator_payload, Mapping):
            raise StrategySpecValidationError("StrategySpecV1 indicator entry must be object")
        _validate_indicator_payload(indicator=indicator_payload)
        indicators.append(_clone_indicator_payload(indicator=indicator_payload))
    return tuple(indicators)



def _validate_indicator_payload(*, indicator: Mapping[str, Any]) -> None:
    """
    Validate indicator mapping shape for StrategySpecV1 signal template generation.

    Args:
        indicator: One indicator payload mapping.
    Returns:
        None.
    Assumptions:
        Indicator has at least one identifier field among `name`, `kind`, or `id`.
    Raises:
        StrategySpecValidationError: If identifier or params fields are malformed.
    Side Effects:
        None.
    """
    identifier = indicator.get("name") or indicator.get("kind") or indicator.get("id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise StrategySpecValidationError(
            "StrategySpecV1 indicator must define non-empty `name`, `kind`, or `id`"
        )
    params = indicator.get("params", {})
    if not isinstance(params, Mapping):
        raise StrategySpecValidationError("StrategySpecV1 indicator.params must be object")



def _clone_indicator_payload(*, indicator: Mapping[str, Any]) -> dict[str, Any]:
    """
    Clone indicator mapping into deterministic plain dict for storage and hashing.

    Args:
        indicator: Indicator mapping to clone.
    Returns:
        dict[str, Any]: JSON-serializable copy.
    Assumptions:
        Nested JSON values are serializable by `json.dumps`.
    Raises:
        StrategySpecValidationError: If indicator cannot be serialized.
    Side Effects:
        None.
    """
    try:
        raw_json = json.dumps(indicator, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError as error:
        raise StrategySpecValidationError(
            "StrategySpecV1 indicator payload must be JSON serializable"
        ) from error
    cloned = json.loads(raw_json)
    if not isinstance(cloned, dict):
        raise StrategySpecValidationError(
            "StrategySpecV1 indicator payload must serialize into JSON object"
        )
    return cloned



def _build_default_signal_template(*, indicators: Sequence[Mapping[str, Any]]) -> str:
    """
    Build fallback signal template string when payload does not provide one explicitly.

    Args:
        indicators: Normalized indicator list from StrategySpecV1 payload.
    Returns:
        str: Deterministic template, for example `MA(20,50)`.
    Assumptions:
        At least one indicator exists when fallback is required.
    Raises:
        StrategySpecValidationError: If indicators list is empty.
    Side Effects:
        None.
    """
    if not indicators:
        raise StrategySpecValidationError(
            "StrategySpecV1 payload.signal_template is required when indicators are empty"
        )

    first = indicators[0]
    name_raw = first.get("name") or first.get("kind") or first.get("id")
    if not isinstance(name_raw, str):
        raise StrategySpecValidationError(
            "StrategySpecV1 first indicator identifier must be string"
        )
    params = first.get("params", {})
    if not isinstance(params, Mapping):
        raise StrategySpecValidationError("StrategySpecV1 indicator.params must be object")

    param_values = ",".join(str(params[key]) for key in sorted(params))
    if not param_values:
        return name_raw.strip().upper()
    return f"{name_raw.strip().upper()}({param_values})"
