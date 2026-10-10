"""Management catalog projected from the existing runtime defaults and hard definitions."""

from __future__ import annotations

from uuid import uuid4

from trading.contexts.backtest_artifacts.adapters.outbound.config.backtest_artifacts_runtime_config import (  # noqa: E501
    BacktestArtifactsRuntimeConfig,
)
from trading.contexts.backtest_artifacts.adapters.outbound.defaults import (
    YamlBacktestGridDefaultsProvider,
)
from trading.contexts.backtest_artifacts.application.services.management import (
    Grid,
    Indicator,
    ManagementError,
    Profile,
    digest,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ARTIFACT_MAPPING_TIMEFRAMES_V2,
)
from trading.contexts.indicators.adapters.outbound.registry import YamlIndicatorRegistry
from trading.contexts.indicators.application.services.grid_builder import GridBuilder
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec


class ManagementCatalog:
    """Frozen finite grids; availability is supplied by the server's catalog composition."""

    def __init__(
        self,
        *,
        defaults: YamlBacktestGridDefaultsProvider,
        registry: YamlIndicatorRegistry,
        config: BacktestArtifactsRuntimeConfig,
        unavailable: frozenset[str] = frozenset(),
    ) -> None:
        self._builder = GridBuilder(registry=registry)
        self._entries: dict[str, Indicator] = {}
        self._all_entries: dict[str, Indicator] = {}
        self.unavailable = unavailable
        for name in defaults.supported_indicator_ids():
            grid = defaults.compute_defaults(indicator_id=name)
            sources = defaults.allowed_source_values(indicator_id=name)
            params = {
                key: Grid(mode="explicit", values=spec.materialize())
                for key, spec in (grid.params.items() if grid else ())
            }
            entry = Indicator(
                indicator_id=name,
                sources=sources,
                params=params,
                signal_params={
                    key: Grid(mode="explicit", values=spec.materialize())
                    for key, spec in defaults.signal_param_defaults(indicator_id=name).items()
                },
            )
            self._native(entry)
            self._all_entries[name] = entry
            if name not in unavailable:
                self._entries[name] = entry
        self.tp = tuple(config.hit_times_grid.tp_levels_pct)
        self.sl = tuple(config.hit_times_grid.sl_levels_pct)
        self.revision = digest(
            {
                "indicators": [i.model_dump(mode="json") for i in self._all_entries.values()],
                "unavailable": sorted(unavailable),
                "tp": self.tp,
                "sl": self.sl,
                "catalog_version": 1,
            }
        )

    def _native(self, entry: Indicator) -> None:
        self._builder.materialize_indicator(
            grid=GridSpec(
                indicator_id=IndicatorId(entry.indicator_id),
                params={
                    key: ExplicitValuesSpec(name=key, values=grid.expand())
                    for key, grid in entry.params.items()
                },
                source=ExplicitValuesSpec(name="source", values=entry.sources)
                if entry.sources
                else None,
            )
        )

    def default_profile(self) -> Profile:
        return Profile(
            profile_id=uuid4(),
            revision=1,
            catalog_revision=self.revision,
            derivative_timeframes=ARTIFACT_MAPPING_TIMEFRAMES_V2,
            indicators=tuple(i.model_copy(deep=True) for i in self._entries.values()),
            catalog_indicators=tuple(i.model_copy(deep=True) for i in self._all_entries.values()),
            tp_levels_pct=self.tp,
            sl_levels_pct=self.sl,
        )

    def validate(self, profile: Profile) -> Profile:
        if profile.catalog_revision != self.revision:
            raise ManagementError("catalog_changed")
        frozen = tuple(self._all_entries.values())
        if profile.catalog_indicators and profile.catalog_indicators != frozen:
            raise ManagementError("catalog_snapshot_mismatch")
        normalized = []
        for entry in profile.indicators:
            allowed = self._entries.get(entry.indicator_id)
            if allowed is None:
                raise ManagementError("indicator_unavailable")
            if (
                len(set(entry.sources)) != len(entry.sources)
                or not set(entry.sources) <= set(allowed.sources)
                or bool(entry.sources) != bool(allowed.sources)
                or set(entry.params) != set(allowed.params)
                or set(entry.signal_params) != set(allowed.signal_params)
            ):
                raise ManagementError("invalid_indicator_axes")
            grids = []
            for selected, supported in (
                (entry.params, allowed.params),
                (entry.signal_params, allowed.signal_params),
            ):
                params = {}
                for key, grid in selected.items():
                    values = grid.expand()
                    if not set(values) <= set(supported[key].expand()):
                        raise ManagementError("parameter_outside_catalog")
                    params[key] = Grid(mode="explicit", values=tuple(sorted(values)))
                grids.append(params)
            item = Indicator(
                indicator_id=entry.indicator_id,
                sources=tuple(sorted(entry.sources)),
                params=grids[0],
                signal_params=grids[1],
            )
            self._native(item)
            normalized.append(item)
        if not set(profile.tp_levels_pct) <= set(self.tp) or not set(profile.sl_levels_pct) <= set(
            self.sl
        ):
            raise ManagementError("risk_outside_catalog")
        return profile.model_copy(
            update={
                "indicators": tuple(sorted(normalized, key=lambda i: i.indicator_id)),
                "catalog_indicators": tuple(i.model_copy(deep=True) for i in frozen),
                "derivative_timeframes": tuple(sorted(profile.derivative_timeframes)),
                "tp_levels_pct": tuple(sorted(profile.tp_levels_pct)),
                "sl_levels_pct": tuple(sorted(profile.sl_levels_pct)),
            }
        )

    def items(self) -> tuple[dict[str, object], ...]:
        """Catalog DTO source for S06/S07; unavailable choices remain visible."""
        return tuple(
            {
                "indicator_id": name,
                "available": name not in self.unavailable,
                "reason": "indicator_unavailable" if name in self.unavailable else None,
                "defaults": item.model_dump(mode="json"),
            }
            for name, item in self._all_entries.items()
        )

    def rebase_profile(self, profile: Profile) -> Profile:
        """Explicit edit draft: preserve exclusions, select new IDs, never mutate admitted jobs."""
        if not profile.catalog_indicators:
            raise ManagementError("catalog_snapshot_required")
        old_ids = {i.indicator_id for i in profile.catalog_indicators}
        if any(i.indicator_id not in self._entries for i in profile.indicators):
            raise ManagementError("indicator_unavailable")
        additions = tuple(i for name, i in self._entries.items() if name not in old_ids)
        draft = profile.model_copy(
            update={
                "revision": profile.revision + 1,
                "catalog_revision": self.revision,
                "catalog_indicators": (),
                "indicators": (*profile.indicators, *additions),
            }
        )
        return self.validate(draft)
