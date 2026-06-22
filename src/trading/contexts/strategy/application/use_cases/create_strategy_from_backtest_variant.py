from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.strategy.application.ports.backtest_variant_launch_reader import (
    BacktestVariantLaunchReader,
    BacktestVariantLaunchSnapshot,
)
from trading.contexts.strategy.application.ports.clock import StrategyClock
from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.repositories import (
    StrategyBacktestVariantProvenanceRepository,
    StrategyEventRepository,
    StrategyRepository,
)
from trading.contexts.strategy.application.use_cases._shared import (
    append_strategy_event,
    ensure_utc_datetime,
)
from trading.contexts.strategy.application.use_cases.errors import map_strategy_exception
from trading.contexts.strategy.domain.entities import (
    Strategy,
    StrategyBacktestVariantProvenance,
    StrategySpecV1,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.direction_policy import (
    SHORT_DIRECTION_REQUIRES_FUTURES_MARKET,
    short_direction_requires_futures_market,
)


@dataclass(frozen=True, slots=True)
class CreateStrategyFromBacktestVariantResult:
    strategy: Strategy
    provenance: StrategyBacktestVariantProvenance
    duplicate: bool
    duplicate_reason: str | None = None


class CreateStrategyFromBacktestVariantUseCase:
    """
    Create an immutable Strategy from an owner-scoped launchable backtest variant.
    """

    def __init__(
        self,
        *,
        variant_reader: BacktestVariantLaunchReader,
        strategy_repository: StrategyRepository,
        provenance_repository: StrategyBacktestVariantProvenanceRepository,
        clock: StrategyClock,
        event_repository: StrategyEventRepository | None = None,
    ) -> None:
        if variant_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateStrategyFromBacktestVariantUseCase requires variant_reader")
        if strategy_repository is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "CreateStrategyFromBacktestVariantUseCase requires strategy_repository"
            )
        if provenance_repository is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "CreateStrategyFromBacktestVariantUseCase requires provenance_repository"
            )
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateStrategyFromBacktestVariantUseCase requires clock")
        self._variant_reader = variant_reader
        self._strategy_repository = strategy_repository
        self._provenance_repository = provenance_repository
        self._clock = clock
        self._event_repository = event_repository

    def execute(
        self,
        *,
        current_user: CurrentUser,
        job_id: UUID,
        variant_key: str,
        idempotency_key: str | None,
        launch_config: Mapping[str, Any] | None = None,
    ) -> CreateStrategyFromBacktestVariantResult:
        key_hash = _require_idempotency_key_hash(idempotency_key=idempotency_key)
        try:
            snapshot = self._variant_reader.get(
                user_id=current_user.user_id,
                job_id=job_id,
                variant_key=variant_key,
            )
            _validate_snapshot_short_policy(snapshot=snapshot)
            _validate_launch_config_matches_snapshot(
                launch_config=launch_config,
                snapshot=snapshot,
            )
            spec = strategy_spec_from_backtest_variant_snapshot(snapshot=snapshot)
            strategy_spec_hash = _sha256_json(spec.to_json())
            launch_request_hash = _sha256_json(
                {
                    "schema": "strategy_backtest_variant_launch_request_v1",
                    "user_id": str(current_user.user_id),
                    "source_job_id": str(snapshot.job_id),
                    "source_variant_key": snapshot.variant_key,
                    "source_variant_hash": snapshot.variant_hash,
                    "strategy_spec_hash": strategy_spec_hash,
                    "launch_config": _normalized_launch_config(launch_config),
                }
            )
            idempotent = self._provenance_repository.find_by_idempotency_key(
                user_id=current_user.user_id,
                idempotency_key_hash=key_hash,
            )
            if idempotent is not None:
                if idempotent.launch_request_hash != launch_request_hash:
                    raise _variant_launch_error(
                        code="strategy_variant_launch.idempotency_key_conflict",
                        message=(
                            "Idempotency-Key was already used with a different "
                            "variant launch request"
                        ),
                        reason="idempotency_key_conflict",
                        details={"idempotency_key_hash": key_hash},
                    )
                return self._duplicate_result(
                    provenance=idempotent,
                    duplicate_reason="idempotent_replay",
                )

            existing = self._provenance_repository.find_by_source_variant(
                user_id=current_user.user_id,
                source_job_id=snapshot.job_id,
                source_variant_key=snapshot.variant_key,
                strategy_spec_hash=strategy_spec_hash,
                launch_request_hash=launch_request_hash,
            )
            if existing is not None:
                return self._duplicate_result(
                    provenance=existing,
                    duplicate_reason="source_variant_exists",
                )

            created_at = ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
            strategy = Strategy.create(
                user_id=current_user.user_id,
                spec=spec,
                created_at=created_at,
            )
            provenance = StrategyBacktestVariantProvenance(
                strategy_id=strategy.strategy_id,
                user_id=current_user.user_id,
                source_job_id=snapshot.job_id,
                source_variant_key=snapshot.variant_key,
                source_variant_hash=snapshot.variant_hash,
                source_indicator_variant_hash=snapshot.indicator_variant_hash,
                backtest_request_hash=snapshot.request_hash,
                backtest_result_config_hash=snapshot.result_config_hash,
                strategy_spec_hash=strategy_spec_hash,
                launch_request_hash=launch_request_hash,
                idempotency_key_hash=key_hash,
                created_at=created_at,
                metadata_json={
                    "schema": "strategy_backtest_variant_provenance_v1",
                    "launchability_status": "launchable",
                    "rank": snapshot.rank,
                    "summary_metrics": dict(snapshot.summary_metrics),
                    "readable_params": dict(snapshot.readable_params),
                    "launch_config": _normalized_launch_config(launch_config),
                },
            )
            persisted_provenance = self._provenance_repository.create_with_strategy(
                strategy=strategy,
                provenance=provenance,
            )
            append_strategy_event(
                repository=self._event_repository,
                strategy_id=strategy.strategy_id,
                current_user=current_user,
                event_type="strategy_created_from_backtest_variant",
                ts=created_at,
                payload_json={
                    "strategy_id": str(strategy.strategy_id),
                    "source_job_id": str(snapshot.job_id),
                    "source_variant_key": snapshot.variant_key,
                    "source_variant_hash": snapshot.variant_hash,
                    "strategy_spec_hash": strategy_spec_hash,
                    "idempotency_key_hash": key_hash,
                },
            )
            return CreateStrategyFromBacktestVariantResult(
                strategy=strategy,
                provenance=persisted_provenance,
                duplicate=False,
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_strategy_exception(error=error) from error

    def _duplicate_result(
        self,
        *,
        provenance: StrategyBacktestVariantProvenance,
        duplicate_reason: str,
    ) -> CreateStrategyFromBacktestVariantResult:
        strategy = self._strategy_repository.find_any_by_strategy_id(
            strategy_id=provenance.strategy_id,
        )
        if strategy is None:
            raise _variant_launch_error(
                code="strategy_variant_launch.not_launchable",
                message="Backtest variant launch provenance points to a missing strategy",
                reason="strategy_missing_for_provenance",
                details={"strategy_id": str(provenance.strategy_id)},
            )
        return CreateStrategyFromBacktestVariantResult(
            strategy=strategy,
            provenance=provenance,
            duplicate=True,
            duplicate_reason=duplicate_reason,
        )


def strategy_spec_from_backtest_variant_snapshot(
    *, snapshot: BacktestVariantLaunchSnapshot
) -> StrategySpecV1:
    if snapshot.job_state != "succeeded":
        raise _variant_launch_error(
            code="strategy_variant_launch.not_launchable",
            message="Backtest variant is not launchable",
            reason="not_launchable",
            details={"job_state": snapshot.job_state},
        )
    canonical = dict(snapshot.canonical_variant_params)
    indicators = canonical.get("indicators")
    if not isinstance(indicators, list) or not indicators:
        raise _variant_launch_error(
            code="strategy_variant_launch.not_launchable",
            message="Backtest variant has no launchable indicator parameters",
            reason="not_launchable",
            details={"field": "canonical_variant_params.indicators"},
        )

    strategy_indicators = [_strategy_indicator(item=item) for item in indicators]
    exchange = snapshot.exchange.strip().lower()
    market_type = snapshot.market_type.strip().lower()
    symbol = snapshot.symbol.strip().upper()
    spec_payload: dict[str, Any] = {
        "instrument_id": {
            "market_id": int(snapshot.market_id),
            "symbol": symbol,
        },
        "instrument_key": f"{exchange}:{market_type}:{symbol}",
        "market_type": market_type,
        "timeframe": snapshot.timeframe,
        "indicators": strategy_indicators,
    }
    signal_template = canonical.get("signal_template")
    if isinstance(signal_template, str) and signal_template.strip():
        spec_payload["signal_template"] = signal_template.strip()
    return StrategySpecV1.from_json(payload=spec_payload)


def _strategy_indicator(*, item: Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise _variant_launch_error(
            code="strategy_variant_launch.not_launchable",
            message="Backtest variant indicator payload is not launchable",
            reason="not_launchable",
            details={"field": "canonical_variant_params.indicators"},
        )
    indicator_id = item.get("indicator_id")
    if not isinstance(indicator_id, str) or not indicator_id.strip():
        raise _variant_launch_error(
            code="strategy_variant_launch.not_launchable",
            message="Backtest variant indicator id is missing",
            reason="not_launchable",
            details={"field": "canonical_variant_params.indicators.indicator_id"},
        )
    params = {
        key: value
        for key, value in {
            "source": item.get("source"),
            "window": item.get("window"),
            "row_id": item.get("row_id"),
            "fast": item.get("fast"),
            "slow": item.get("slow"),
        }.items()
        if value is not None
    }
    return {"id": indicator_id.strip(), "params": params}


def _require_idempotency_key_hash(*, idempotency_key: str | None) -> str:
    normalized = (idempotency_key or "").strip()
    if not normalized:
        raise _variant_launch_error(
            code="strategy_variant_launch.idempotency_key_required",
            message="Idempotency-Key header is required",
            reason="idempotency_key_required",
            details={},
        )
    if len(normalized) > 256:
        raise _variant_launch_error(
            code="strategy_variant_launch.idempotency_key_required",
            message="Idempotency-Key header is too long",
            reason="idempotency_key_too_long",
            details={"max_length": 256},
        )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _normalized_launch_config(
    launch_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if launch_config is None:
        return {}
    return json.loads(
        json.dumps(
            dict(launch_config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
    )


def _validate_launch_config_matches_snapshot(
    *,
    launch_config: Mapping[str, Any] | None,
    snapshot: BacktestVariantLaunchSnapshot,
) -> None:
    normalized = _normalized_launch_config(launch_config)
    if not normalized:
        return
    expected = {
        "market_type": snapshot.market_type.strip().lower(),
        "symbol": snapshot.symbol.strip().upper(),
        "direction": _launch_direction_from_snapshot(snapshot=snapshot),
    }
    actual = {
        "market_type": str(normalized.get("market_type", "")).strip().lower(),
        "symbol": str(normalized.get("symbol", "")).strip().upper(),
        "direction": str(normalized.get("direction", "")).strip().lower(),
    }
    for field, expected_value in expected.items():
        actual_value = actual[field]
        if actual_value and actual_value != expected_value:
            raise _variant_launch_error(
                code="strategy_launch.invalid_config",
                message="Backtest variant launch config does not match the source variant",
                reason=f"{field}_mismatch",
                details={
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                },
            )


def _validate_snapshot_short_policy(*, snapshot: BacktestVariantLaunchSnapshot) -> None:
    market_type = snapshot.market_type.strip().lower()
    direction = _launch_direction_from_snapshot(snapshot=snapshot)
    if not short_direction_requires_futures_market(
        market_type=market_type,
        direction=direction,
    ):
        return
    raise _variant_launch_error(
        code="strategy_launch.invalid_config",
        message="Short-like backtest variants require market_type=futures before launch",
        reason=SHORT_DIRECTION_REQUIRES_FUTURES_MARKET,
        details={
            "field": "market_type",
            "required_market_type": "futures",
            "actual_market_type": market_type,
            "direction": direction,
        },
    )


def _launch_direction_from_snapshot(*, snapshot: BacktestVariantLaunchSnapshot) -> str:
    canonical = _mapping(snapshot.canonical_variant_params)
    execution = _mapping(canonical.get("execution"))
    direction_mode = str(
        execution.get("direction_mode", "long_short_reversal")
    ).strip().lower()
    if direction_mode in {"long_only", "long"}:
        return "long"
    if direction_mode == "short":
        return "short"
    return "long_short_reversal"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _variant_launch_error(
    *,
    code: str,
    message: str,
    reason: str,
    details: Mapping[str, Any],
) -> RoehubError:
    return RoehubError(code=code, message=message, details={"reason": reason, **dict(details)})
