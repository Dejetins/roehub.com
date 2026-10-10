"""Frozen catalog selection and exact compatible demand union for native publication.

This service never dispatches jobs or acquires a queue claim. Publication still uses
0028 physical reservations; the coordinator supplies authorized request snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from trading.contexts.backtest_artifacts.application.services.automatic_base import (
        AutomaticBasePublisher,
    )

from trading.contexts.backtest_artifacts.application.services.management import (
    Coordinate,
    Indicator,
    ManagementError,
    Profile,
    SourceTarget,
    digest,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_precompute_runner import (
    BacktestArtifactPrecomputeRunnerV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactSignalValidationSpecV2,
    ArtifactSlotValidationSpecV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.signal_rules_engine_v2 import (
    BacktestSignalRulesEngineV2,
)
from trading.contexts.indicators.application.services.grid_builder import GridBuilder
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec


@dataclass(frozen=True)
class FrozenProfileDefaults:
    """Immutable serialized snapshot, without reads of YAML/defaults during execution."""

    profile_json: str

    def _entries(self) -> dict[str, Indicator]:
        profile = Profile.model_validate_json(self.profile_json)
        return {i.indicator_id: i for i in profile.catalog_indicators}

    def supported_indicator_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries()))

    def compute_defaults(self, *, indicator_id: str) -> GridSpec:
        item = self._entries()[indicator_id]
        return GridSpec(
            indicator_id=IndicatorId(indicator_id),
            params={
                k: ExplicitValuesSpec(name=k, values=v.expand()) for k, v in item.params.items()
            },
            source=ExplicitValuesSpec(name="source", values=item.sources) if item.sources else None,
        )

    def allowed_source_values(self, *, indicator_id: str) -> tuple[str, ...]:
        return self._entries()[indicator_id].sources

    def signal_param_defaults(self, *, indicator_id: str) -> Mapping[str, ExplicitValuesSpec]:
        return {
            k: ExplicitValuesSpec(name=k, values=v.expand())
            for k, v in self._entries()[indicator_id].signal_params.items()
        }


@dataclass(frozen=True)
class DerivativeDemand:
    """Internal frozen request identity. Never expose request IDs across organizations."""

    request_id: str
    coordinate_json: str
    source_json: str
    source_sha256: str
    root_id: str
    root_revision: int
    profile_json: str

    @classmethod
    def freeze(
        cls,
        *,
        request_id: str,
        coordinate: Coordinate,
        source: SourceTarget,
        source_sha256: str,
        root_id: str,
        root_revision: int,
        profile: Profile,
    ):
        if not profile.catalog_indicators:
            raise ManagementError("catalog_snapshot_required")
        if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
            raise ManagementError("invalid_source_digest")
        if not request_id or not root_id or type(root_revision) is not int or root_revision < 1:
            raise ManagementError("invalid_demand_identity")
        # Deep detach nested mappings: frozen pydantic models alone do not freeze dict values.
        return cls(
            request_id,
            coordinate.model_dump_json(),
            source.model_dump_json(),
            source_sha256,
            root_id,
            root_revision,
            profile.model_dump_json(),
        )

    @property
    def compatibility_key(self) -> str:
        p = Profile.model_validate_json(self.profile_json)
        return digest(
            {
                "coordinate": Coordinate.model_validate_json(self.coordinate_json).model_dump(),
                "source": SourceTarget.model_validate_json(self.source_json).model_dump(
                    mode="json"
                ),
                "payload": self.source_sha256,
                "root": (self.root_id, self.root_revision),
                "catalog": p.catalog_revision,
                "defaults": [i.model_dump(mode="json") for i in p.catalog_indicators],
                "versions": (2, "signals/v1", "numba/v1", "f32-f64/v1"),
            }
        )


@dataclass(frozen=True)
class DerivativePlan:
    demands: tuple[DerivativeDemand, ...]
    selected_rows: tuple[tuple[str, str, tuple[int, ...]], ...]
    tp: tuple[float, ...]
    sl: tuple[float, ...]

    @property
    def active(self) -> bool:
        return bool(self.selected_rows or self.tp or self.sl)

    def validate_source(self, *, coordinate: Coordinate, source_sha256: str, start, end) -> None:
        demand = self.demands[0]
        source = SourceTarget.model_validate_json(demand.source_json)
        if (
            Coordinate.model_validate_json(demand.coordinate_json) != coordinate
            or demand.source_sha256 != source_sha256
            or source.history.start_utc != start
            or source.history.end_utc != end
        ):
            raise ManagementError("derivative_source_changed")

    def configure(self, runner: BacktestArtifactPrecomputeRunnerV2, mappings: tuple[str, ...]):
        targets = tuple(
            ArtifactSignalValidationSpecV2(tf, name) for tf, name, _ in self.selected_rows
        )
        if not {t.timeframe for t in targets} <= set(mappings) or (
            (self.tp or self.sl) and "15m" not in mappings
        ):
            raise ManagementError("derivative_insufficient_history")
        defaults = FrozenProfileDefaults(self.demands[0].profile_json)
        settings = replace(
            runner.runtime_settings,
            price_timeframes=("1m", *mappings),
            mapping_timeframes=mappings,
            partial_publication=True,
            signal_artifacts=targets,
            selected_signal_rows=self.selected_rows,
            precompute_hit_times=bool(self.tp or self.sl),
            hit_times_tp_levels_pct=self.tp,
            hit_times_sl_levels_pct=self.sl,
            config_sha256=digest(
                {
                    "base": runner.runtime_settings.config_sha256,
                    "demand": self.demands[0].compatibility_key,
                    "rows": self.selected_rows,
                    "tp": self.tp,
                    "sl": self.sl,
                }
            ),
            execution_policy=replace(
                runner.runtime_settings.execution_policy, signal_worker_processes=1
            ),
        )
        return replace(
            runner,
            runtime_settings=settings,
            defaults_provider=defaults,
            signal_rules_engine=BacktestSignalRulesEngineV2(
                defaults_provider=defaults, indicator_scope=defaults.supported_indicator_ids()
            ),
        ), (
            ArtifactSlotValidationSpecV2(
                price_timeframes=settings.price_timeframes,
                mapping_timeframes=mappings,
                signal_artifacts=targets,
                require_hit_times_manifest=bool(self.tp or self.sl),
            )
        )


def plan_derivatives(
    demands: tuple[DerivativeDemand, ...], *, grid_builder: GridBuilder, max_rows: int = 5_000_000
) -> DerivativePlan:
    """Union actual row tuples, never independent axes (which would invent combinations).

    History requests are contained in the identical frozen full source. Native output
    retains that complete timeline for warmup and global risk indexes, as the recipe contract
    requires. Budgets reject the original union without shrinking user selections.
    """
    if not demands or len(demands) > 50 or len({d.request_id for d in demands}) != len(demands):
        raise ManagementError("invalid_demand_batch")
    if len({d.compatibility_key for d in demands}) != 1:
        raise ManagementError("incompatible_demand")
    if type(max_rows) is not int or max_rows < 1:
        raise ManagementError("invalid_row_budget")
    union: dict[tuple[str, str], set[int]] = {}
    tp: set[float] = set()
    sl: set[float] = set()
    for demand in demands:
        profile = Profile.model_validate_json(demand.profile_json)
        source = SourceTarget.model_validate_json(demand.source_json)
        if profile.history.kind == "range":
            a, b = source.history.start_utc, source.history.end_utc
            c, d = profile.history.start_utc, profile.history.end_utc
            if a is None or b is None or c is None or d is None or not a <= c < d <= b:
                raise ManagementError("profile_history_outside_source")
        if profile.mode == "on_demand":
            continue
        defaults = FrozenProfileDefaults(demand.profile_json)
        # Validates removed/unsupported rules and default-only signal params explicitly.
        rules = BacktestSignalRulesEngineV2(
                defaults_provider=defaults, indicator_scope=defaults.supported_indicator_ids()
            )
        entries = defaults._entries()
        for item in profile.indicators:
            full = entries.get(item.indicator_id)
            if full is None:
                raise ManagementError("indicator_unavailable")
            if item.signal_params != full.signal_params:
                raise ManagementError("incompatible_signal_defaults")
            grid = grid_builder.materialize_indicator(
                grid=defaults.compute_defaults(indicator_id=item.indicator_id)
            )
            if set(item.params) != set(full.params) or not set(item.sources) <= set(full.sources):
                raise ManagementError("invalid_indicator_axes")
            selected = {k: v.expand() for k, v in item.params.items()}
            if full.sources:
                selected["source"] = item.sources
            if any(not v for v in selected.values()):
                raise ManagementError("invalid_indicator_axes")
            indexes = []
            strides = []
            stride = 1
            for axis in reversed(grid.axes):
                values = selected.get(axis.name, ())
                if not set(values) <= set(axis.values) or not values:
                    raise ManagementError("parameter_outside_catalog")
                indexes.append(tuple(axis.values.index(v) for v in values))
                strides.append(stride)
                stride *= len(axis.values)
            # Compute selected cardinality before allocating row IDs.
            count = 1
            for values in indexes:
                count *= len(values)
            if count * len(profile.derivative_timeframes) > max_rows:
                raise ManagementError("resource_conflict")
            rows = {sum(i * s for i, s in zip(ix, strides)) for ix in product(*indexes)}
            rules.resolved_defaults(indicator_id=item.indicator_id)
            for tf in profile.derivative_timeframes:
                union.setdefault((tf, item.indicator_id), set()).update(rows)
            if sum(len(rows) for rows in union.values()) > max_rows:
                raise ManagementError("resource_conflict")
        tp.update(profile.tp_levels_pct)
        sl.update(profile.sl_levels_pct)
    return DerivativePlan(
        tuple(sorted(demands, key=lambda d: d.request_id)),
        tuple((tf, name, tuple(sorted(rows))) for (tf, name), rows in sorted(union.items())),
        tuple(sorted(tp)),
        tuple(sorted(sl)),
    )


@dataclass(frozen=True)
class ManagedDerivativePublisher:
    """Mode-off/empty demand is a no-op; mandatory base work has its independent lane."""

    base_publisher: AutomaticBasePublisher
    root_id: str
    root_revision: int
    root_path: Path

    def publish(self, *, plan: DerivativePlan, **kwargs):
        if not plan.active:
            return None
        demand = plan.demands[0]
        source = SourceTarget.model_validate_json(demand.source_json)
        coordinates = kwargs["coordinates"]
        actual = self.base_publisher.publisher.artifact_loader.resolve_current_pointer_path(
            coordinates
        ).resolve()
        expected = (
            self.root_path
            / coordinates.exchange
            / coordinates.market_type
            / coordinates.symbol
            / "current.yaml"
        ).resolve()
        if (demand.root_id, demand.root_revision) != (self.root_id, self.root_revision) or (
            actual != expected
        ):
            raise ManagementError("derivative_root_changed")
        if source.compute_identity != "signals/v1|numba/v1|f32-f64/v1":
            raise ManagementError("unsupported_compute_identity")
        funding = "not_applicable" if coordinates.market_type == "spot" else "canonical_funding/v1"
        if source.funding_identity != funding:
            raise ManagementError("unsupported_funding_identity")
        return self.base_publisher.publish(derivatives=plan, **kwargs)
