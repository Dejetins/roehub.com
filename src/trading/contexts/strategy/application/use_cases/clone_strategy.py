from __future__ import annotations

from typing import Any, Mapping
from uuid import UUID

from trading.contexts.strategy.application.ports.clock import StrategyClock
from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.repositories import (
    StrategyEventRepository,
    StrategyRepository,
)
from trading.contexts.strategy.application.use_cases._shared import (
    append_strategy_event,
    ensure_utc_datetime,
)
from trading.contexts.strategy.application.use_cases.errors import (
    map_strategy_exception,
    strategy_forbidden,
    strategy_not_found,
    validation_error,
)
from trading.contexts.strategy.domain.entities import Strategy, StrategySpecV1
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, Timeframe

_ALLOWED_OVERRIDE_KEYS: frozenset[str] = frozenset({"instrument_id", "timeframe"})


class CloneStrategyUseCase:
    """
    CloneStrategyUseCase — immutable strategy clone from template or existing strategy.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy.py
      - src/trading/contexts/strategy/domain/entities/strategy_spec_v1.py
      - apps/api/routes/strategies.py
    """

    def __init__(
        self,
        *,
        repository: StrategyRepository,
        clock: StrategyClock,
        event_repository: StrategyEventRepository | None = None,
    ) -> None:
        """
        Initialize immutable strategy clone use-case dependencies.

        Args:
            repository: Strategy repository port.
            clock: Clock port for deterministic UTC timestamps.
            event_repository: Optional append-only event repository port.
        Returns:
            None.
        Assumptions:
            Clone operation creates a new strategy snapshot and never mutates source strategy.
        Raises:
            ValueError: If required dependencies are missing.
        Side Effects:
            None.
        """
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("CloneStrategyUseCase requires repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("CloneStrategyUseCase requires clock")
        self._repository = repository
        self._clock = clock
        self._event_repository = event_repository

    def execute(
        self,
        *,
        current_user: CurrentUser,
        source_strategy_id: UUID | None,
        template_spec_payload: Mapping[str, Any] | None,
        overrides: Mapping[str, Any] | None,
    ) -> Strategy:
        """
        Clone strategy from owned source strategy or template payload
        with explicit whitelisted overrides.

        Args:
            current_user: Authenticated current user context.
            source_strategy_id: Optional source strategy identifier.
            template_spec_payload:
                Optional template spec payload when source strategy id is absent.
            overrides: Optional clone overrides mapping.
        Returns:
            Strategy: Persisted cloned strategy snapshot owned by current user.
        Assumptions:
            Exactly one clone source must be provided:
            `source_strategy_id` xor `template_spec_payload`.
        Raises:
            RoehubError: If source/template is invalid, forbidden, missing, or repository fails.
        Side Effects:
            Persists cloned strategy and appends strategy-cloned event when
            repository is configured.
        """
        _validate_clone_source(
            source_strategy_id=source_strategy_id,
            template_spec_payload=template_spec_payload,
        )

        try:
            base_spec = self._resolve_base_spec(
                current_user=current_user,
                source_strategy_id=source_strategy_id,
                template_spec_payload=template_spec_payload,
            )
            normalized_overrides = _normalize_clone_overrides(overrides=overrides)
            cloned_spec = _clone_spec_with_overrides(
                base_spec=base_spec,
                overrides=normalized_overrides,
            )
            created_at = ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
            cloned_strategy = Strategy.create(
                user_id=current_user.user_id,
                spec=cloned_spec,
                created_at=created_at,
            )
            persisted = self._repository.create(strategy=cloned_strategy)
            append_strategy_event(
                repository=self._event_repository,
                strategy_id=persisted.strategy_id,
                current_user=current_user,
                event_type="strategy_cloned",
                ts=created_at,
                payload_json={
                    "strategy_id": str(persisted.strategy_id),
                    "source_strategy_id": (
                        str(source_strategy_id) if source_strategy_id is not None else None
                    ),
                    "override_keys": sorted(normalized_overrides),
                },
            )
            return persisted
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_strategy_exception(error=error) from error

    def _resolve_base_spec(
        self,
        *,
        current_user: CurrentUser,
        source_strategy_id: UUID | None,
        template_spec_payload: Mapping[str, Any] | None,
    ) -> StrategySpecV1:
        """
        Resolve source specification for clone operation from existing strategy or template payload.

        Args:
            current_user: Authenticated current user context.
            source_strategy_id: Optional source strategy identifier.
            template_spec_payload: Optional template spec mapping.
        Returns:
            StrategySpecV1: Base immutable spec used for cloning.
        Assumptions:
            Caller pre-validates that exactly one source mode is provided.
        Raises:
            RoehubError: If source strategy is missing/forbidden or template payload is invalid.
        Side Effects:
            Reads source strategy from repository when strategy-id mode is used.
        """
        if source_strategy_id is not None:
            source_strategy = self._repository.find_any_by_strategy_id(
                strategy_id=source_strategy_id,
            )
            if source_strategy is None or source_strategy.is_deleted:
                raise strategy_not_found(strategy_id=source_strategy_id)
            if source_strategy.user_id != current_user.user_id:
                raise strategy_forbidden(strategy_id=source_strategy_id)
            return source_strategy.spec

        if template_spec_payload is None:
            raise validation_error(
                message="Clone template payload is required",
                errors=(
                    {
                        "path": "body.template",
                        "code": "required",
                        "message": "Clone template payload is required",
                    },
                ),
            )

        if not isinstance(template_spec_payload, Mapping):
            raise validation_error(
                message="Clone template payload must be object",
                errors=(
                    {
                        "path": "body.template",
                        "code": "type_error",
                        "message": "Clone template payload must be object",
                    },
                ),
            )

        return StrategySpecV1.from_json(payload=dict(template_spec_payload))



def _validate_clone_source(
    *,
    source_strategy_id: UUID | None,
    template_spec_payload: Mapping[str, Any] | None,
) -> None:
    """
    Validate that clone request selects exactly one source mode.

    Args:
        source_strategy_id: Optional source strategy identifier.
        template_spec_payload: Optional template spec payload.
    Returns:
        None.
    Assumptions:
        Clone command contract requires xor relation between source id and template payload.
    Raises:
        RoehubError: If both or neither source modes are provided.
    Side Effects:
        None.
    """
    has_source_id = source_strategy_id is not None
    has_template = template_spec_payload is not None
    if has_source_id == has_template:
        raise validation_error(
            message="Clone request must provide exactly one source",
            errors=(
                {
                    "path": "body.source_strategy_id",
                    "code": "invalid_source",
                    "message": "Provide either source_strategy_id or template, but not both",
                },
                {
                    "path": "body.template",
                    "code": "invalid_source",
                    "message": "Provide either source_strategy_id or template, but not both",
                },
            ),
        )



def _normalize_clone_overrides(*, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Normalize clone override payload with explicit whitelist validation.

    Args:
        overrides: Raw clone overrides mapping.
    Returns:
        dict[str, Any]: Normalized override mapping with parsed value objects.
    Assumptions:
        Only whitelisted fields can be overridden to preserve immutable strategy semantics.
    Raises:
        RoehubError: If override keys or values are invalid.
    Side Effects:
        None.
    """
    if overrides is None:
        return {}

    if not isinstance(overrides, Mapping):
        raise validation_error(
            message="Clone overrides must be object",
            errors=(
                {
                    "path": "body.overrides",
                    "code": "type_error",
                    "message": "Clone overrides must be object",
                },
            ),
        )

    override_keys = {str(key) for key in overrides.keys()}
    unknown_keys = sorted(override_keys - _ALLOWED_OVERRIDE_KEYS)
    if unknown_keys:
        raise validation_error(
            message="Clone overrides contain unsupported fields",
            errors=tuple(
                {
                    "path": f"body.overrides.{key}",
                    "code": "unsupported_override",
                    "message": "Override key is not allowed",
                }
                for key in unknown_keys
            ),
        )

    normalized: dict[str, Any] = {}

    if "instrument_id" in overrides:
        normalized["instrument_id"] = _parse_override_instrument_id(
            value=overrides["instrument_id"]
        )

    if "timeframe" in overrides:
        normalized["timeframe"] = _parse_override_timeframe(value=overrides["timeframe"])

    return normalized



def _parse_override_instrument_id(*, value: Any) -> InstrumentId:
    """
    Parse clone override `instrument_id` payload into shared-kernel InstrumentId value object.

    Args:
        value: Raw override value expected as mapping with `market_id` and `symbol`.
    Returns:
        InstrumentId: Parsed instrument identifier value object.
    Assumptions:
        `market_id` must be integer and `symbol` must be non-empty string.
    Raises:
        RoehubError: If override payload cannot be parsed.
    Side Effects:
        None.
    """
    if not isinstance(value, Mapping):
        raise validation_error(
            message="overrides.instrument_id must be object",
            errors=(
                {
                    "path": "body.overrides.instrument_id",
                    "code": "type_error",
                    "message": "overrides.instrument_id must be object",
                },
            ),
        )

    market_id_raw = value.get("market_id")
    symbol_raw = value.get("symbol")
    if not isinstance(market_id_raw, int) or not isinstance(symbol_raw, str):
        raise validation_error(
            message="overrides.instrument_id requires integer market_id and string symbol",
            errors=(
                {
                    "path": "body.overrides.instrument_id.market_id",
                    "code": "type_error",
                    "message": "market_id must be integer",
                },
                {
                    "path": "body.overrides.instrument_id.symbol",
                    "code": "type_error",
                    "message": "symbol must be string",
                },
            ),
        )

    try:
        return InstrumentId(
            market_id=MarketId(market_id_raw),
            symbol=Symbol(symbol_raw),
        )
    except ValueError as error:
        raise validation_error(
            message="overrides.instrument_id is invalid",
            errors=(
                {
                    "path": "body.overrides.instrument_id",
                    "code": "invalid_value",
                    "message": str(error),
                },
            ),
        ) from error



def _parse_override_timeframe(*, value: Any) -> Timeframe:
    """
    Parse clone override `timeframe` payload into shared-kernel Timeframe value object.

    Args:
        value: Raw timeframe override value.
    Returns:
        Timeframe: Parsed timeframe value object.
    Assumptions:
        Timeframe code follows shared-kernel allowed set.
    Raises:
        RoehubError: If timeframe payload is missing or invalid.
    Side Effects:
        None.
    """
    if not isinstance(value, str):
        raise validation_error(
            message="overrides.timeframe must be string",
            errors=(
                {
                    "path": "body.overrides.timeframe",
                    "code": "type_error",
                    "message": "overrides.timeframe must be string",
                },
            ),
        )

    try:
        return Timeframe(value)
    except ValueError as error:
        raise validation_error(
            message="overrides.timeframe is invalid",
            errors=(
                {
                    "path": "body.overrides.timeframe",
                    "code": "invalid_value",
                    "message": str(error),
                },
            ),
        ) from error



def _clone_spec_with_overrides(
    *,
    base_spec: StrategySpecV1,
    overrides: Mapping[str, Any],
) -> StrategySpecV1:
    """
    Build cloned StrategySpecV1 with explicit whitelist overrides applied deterministically.

    Args:
        base_spec: Source immutable spec.
        overrides: Normalized overrides mapping.
    Returns:
        StrategySpecV1: Cloned immutable spec with applied overrides.
    Assumptions:
        Clone overrides are limited to instrument_id and timeframe in v1.
    Raises:
        RoehubError: If override payload contains unsupported normalized types.
    Side Effects:
        None.
    """
    instrument_id_override = overrides.get("instrument_id")
    timeframe_override = overrides.get("timeframe")

    if instrument_id_override is not None and not isinstance(instrument_id_override, InstrumentId):
        raise RoehubError(
            code="unexpected_error",
            message="Normalized clone override instrument_id has invalid type",
            details={"type": type(instrument_id_override).__name__},
        )
    if timeframe_override is not None and not isinstance(timeframe_override, Timeframe):
        raise RoehubError(
            code="unexpected_error",
            message="Normalized clone override timeframe has invalid type",
            details={"type": type(timeframe_override).__name__},
        )

    effective_instrument_id = (
        instrument_id_override
        if isinstance(instrument_id_override, InstrumentId)
        else base_spec.instrument_id
    )
    effective_timeframe = (
        timeframe_override if isinstance(timeframe_override, Timeframe) else base_spec.timeframe
    )
    effective_instrument_key = _rebuild_instrument_key(
        base_key=base_spec.instrument_key,
        market_type=base_spec.market_type,
        instrument_id=effective_instrument_id,
    )

    return StrategySpecV1(
        instrument_id=effective_instrument_id,
        instrument_key=effective_instrument_key,
        market_type=base_spec.market_type,
        timeframe=effective_timeframe,
        indicators=base_spec.indicators,
        signal_template=base_spec.signal_template,
        schema_version=base_spec.schema_version,
        spec_kind=base_spec.spec_kind,
    )



def _rebuild_instrument_key(
    *,
    base_key: str,
    market_type: str,
    instrument_id: InstrumentId,
) -> str:
    """
    Rebuild instrument key deterministically after instrument_id override.

    Args:
        base_key: Original instrument key string.
        market_type: Effective market type.
        instrument_id: Effective instrument identifier.
    Returns:
        str: Rebuilt instrument key preserving exchange and effective market/symbol tags.
    Assumptions:
        Instrument key follows `{exchange}:{market_type}:{symbol}` shape in StrategySpecV1.
    Raises:
        RoehubError: If base key cannot be parsed.
    Side Effects:
        None.
    """
    parts = base_key.split(":")
    if len(parts) != 3:
        raise validation_error(
            message="strategy.spec.instrument_key must follow '{exchange}:{market_type}:{symbol}'",
            errors=(
                {
                    "path": "body.instrument_key",
                    "code": "invalid_value",
                    "message": "instrument_key must have three colon-separated parts",
                },
            ),
        )

    exchange = parts[0].strip()
    if not exchange:
        raise validation_error(
            message="strategy.spec.instrument_key exchange tag must be non-empty",
            errors=(
                {
                    "path": "body.instrument_key",
                    "code": "invalid_value",
                    "message": "instrument_key exchange tag must be non-empty",
                },
            ),
        )

    return f"{exchange}:{market_type}:{instrument_id.symbol}"
