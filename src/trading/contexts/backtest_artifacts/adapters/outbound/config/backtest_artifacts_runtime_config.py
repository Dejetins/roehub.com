from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from trading.contexts.backtest_artifacts.application.services.v2 import (
    ALLOWED_ARTIFACT_SLOTS_V2,
    ARTIFACT_MAPPING_TIMEFRAMES_V2,
    ARTIFACT_PRICE_TIMEFRAMES_V2,
    ARTIFACT_SIGNAL_TIMEFRAMES_V2,
    ArtifactPrecomputeExecutionPolicyV2,
    ArtifactPrecomputeRuntimeSettingsV2,
    ArtifactSignalValidationSpecV2,
    ArtifactSlotValidationSpecV2,
    ordered_artifact_slots_v2,
    supported_indicator_ids_for_signals_v1,
    validate_artifact_slot_v2,
    validate_indicator_id_v2,
    validate_mapping_timeframe_v2,
    validate_price_timeframe_v2,
    validate_signal_timeframe_v2,
)

_ENV_NAME_KEY = "ROEHUB_ENV"
_ALLOWED_ENVS = ("dev", "prod", "test")
_BACKTEST_ARTIFACTS_CONFIG_PATH_KEY = "ROEHUB_BACKTEST_ARTIFACTS_CONFIG"
_ARTIFACTS_CONFIG_VERSION = 1
_TOP_LEVEL_REQUIRED_KEYS = ("version", "backtest_artifacts")
_ARTIFACTS_REQUIRED_KEYS = (
    "artifact_root",
    "validation_plan",
    "hit_times_grid",
    "slot_policy",
    "publish_schedule",
    "lookback_policy",
    "validation_budgets",
    "execution_policy",
)
_VALIDATION_PLAN_REQUIRED_KEYS = (
    "price_timeframes",
    "mapping_timeframes",
    "signal_artifacts",
    "require_hit_times_manifest",
)
_SIGNAL_ARTIFACT_REQUIRED_KEYS = ("timeframe", "indicator_id")
_SIGNAL_ARTIFACTS_ALL_SUPPORTED_LITERAL = "all_supported_v1"
_HIT_TIMES_GRID_REQUIRED_KEYS = ("tp_levels_pct", "sl_levels_pct")
_SLOT_POLICY_REQUIRED_KEYS = ("slots",)
_PUBLISH_SCHEDULE_REQUIRED_KEYS = ("full_rebuild_hour_utc", "full_rebuild_minute_utc")
_LOOKBACK_POLICY_REQUIRED_KEYS = (
    "price_tail_bars_1m",
    "mapping_tail_bars_1m",
    "signal_tail_bars_1m",
    "hit_times_tail_bars_1m",
)
_VALIDATION_BUDGETS_REQUIRED_KEYS = (
    "max_price_bars_per_timeframe",
    "max_mapping_rows_per_timeframe",
    "max_signal_rows_per_artifact",
    "max_hit_times_cells",
    "max_hit_times_cells_full_rebuild",
)
_EXECUTION_POLICY_REQUIRED_KEYS = (
    "max_open_timeframe_sessions",
    "signal_worker_processes",
    "signal_worker_memory_budget_bytes",
    "signal_chunk_rows_min",
    "signal_chunk_rows_max",
)


def resolve_backtest_env_name(*, environ: Mapping[str, str]) -> str:
    """
    Resolve normalized Backtest environment name from process environment mapping.

    Args:
        environ: Process environment mapping (`os.environ`-compatible).
    Returns:
        str: Normalized environment name in `{\"dev\", \"test\", \"prod\"}`.
    Assumptions:
        Empty/unknown values must fail fast because config path resolution is env-scoped.
    Raises:
        ValueError: If environment name is missing or unsupported.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest_artifacts/adapters/outbound/config/
        backtest_artifacts_runtime_config.py
    """
    raw_value = environ.get(_ENV_NAME_KEY, "").strip().lower()
    if raw_value not in _ALLOWED_ENVS:
        raise ValueError(
            f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}; got {raw_value!r}"
        )
    return raw_value


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """
    YAML safe loader variant that rejects duplicate mapping keys fail-fast.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """
    Build one YAML mapping while rejecting duplicate keys with a stable error.

    Args:
        loader: Active duplicate-key-safe YAML loader instance.
        node: YAML mapping node being constructed.
        deep: Whether nested objects should be constructed deeply.
    Returns:
        dict[Any, Any]: Constructed mapping payload.
    Assumptions:
        Artifact config contracts use mapping keys as source-of-truth field identifiers.
    Raises:
        ValueError: If the mapping contains the same key more than once.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"backtest_artifacts contains duplicate YAML key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class BacktestArtifactSignalRuntimeConfig:
    """
    Frozen signal validation target loaded from `backtest_artifacts.validation_plan`.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - configs/dev/backtest_artifacts.yaml
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """

    timeframe: str
    indicator_id: str

    def __post_init__(self) -> None:
        """
        Validate one signal artifact target with deterministic literal contracts.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Signal artifact identities stay explicit and are later translated into
            `ArtifactSignalValidationSpecV2`.
        Raises:
            ValueError: If timeframe or indicator id violates the artifact path contract.
        Side Effects:
            Normalizes literals to validated canonical values.
        """
        object.__setattr__(self, "timeframe", validate_signal_timeframe_v2(self.timeframe))
        object.__setattr__(self, "indicator_id", validate_indicator_id_v2(self.indicator_id))

    def to_validation_spec(self) -> ArtifactSignalValidationSpecV2:
        """
        Translate runtime config item into the publish-layer signal validation spec.

        Args:
            None.
        Returns:
            ArtifactSignalValidationSpecV2: Explicit one-signal validation target.
        Assumptions:
            Translation is lossless because both contracts use the same identity fields.
        Raises:
            ValueError: If stored literals became invalid before translation.
        Side Effects:
            None.
        """
        return ArtifactSignalValidationSpecV2(
            timeframe=self.timeframe,
            indicator_id=self.indicator_id,
        )


@dataclass(frozen=True, slots=True)
class BacktestArtifactValidationPlanRuntimeConfig:
    """
    Frozen publish validation plan loaded from `backtest_artifacts.validation_plan`.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - configs/dev/backtest_artifacts.yaml
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """

    price_timeframes: tuple[str, ...]
    mapping_timeframes: tuple[str, ...]
    signal_artifacts: tuple[BacktestArtifactSignalRuntimeConfig, ...]
    require_hit_times_manifest: bool

    def __post_init__(self) -> None:
        """
        Validate and canonically order explicit publish validation targets.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Canonical ordering keeps publish diagnostics and config hashing deterministic.
        Raises:
            ValueError: If one timeframe, signal target, or boolean field is invalid.
        Side Effects:
            Replaces stored sequences with canonical unique tuples.
        """
        object.__setattr__(
            self,
            "price_timeframes",
            _normalize_timeframe_sequence(
                values=self.price_timeframes,
                field_path="backtest_artifacts.validation_plan.price_timeframes",
                allowed_literals=ARTIFACT_PRICE_TIMEFRAMES_V2,
                validator=validate_price_timeframe_v2,
            ),
        )
        object.__setattr__(
            self,
            "mapping_timeframes",
            _normalize_timeframe_sequence(
                values=self.mapping_timeframes,
                field_path="backtest_artifacts.validation_plan.mapping_timeframes",
                allowed_literals=ARTIFACT_MAPPING_TIMEFRAMES_V2,
                validator=validate_mapping_timeframe_v2,
            ),
        )
        object.__setattr__(
            self,
            "signal_artifacts",
            _normalize_signal_artifacts(
                values=self.signal_artifacts,
                field_path="backtest_artifacts.validation_plan.signal_artifacts",
            ),
        )
        object.__setattr__(
            self,
            "require_hit_times_manifest",
            _require_bool(
                value=self.require_hit_times_manifest,
                field_path="backtest_artifacts.validation_plan.require_hit_times_manifest",
            ),
        )

    def to_validation_spec(self) -> ArtifactSlotValidationSpecV2:
        """
        Translate runtime config plan into the publish-layer validation spec contract.

        Args:
            None.
        Returns:
            ArtifactSlotValidationSpecV2: Explicit deterministic slot validation plan.
        Assumptions:
            Publish validation still consumes `ArtifactSlotValidationSpecV2` in R2.
        Raises:
            ValueError: If a stored signal target fails translation.
        Side Effects:
            None.
        """
        return ArtifactSlotValidationSpecV2(
            price_timeframes=self.price_timeframes,
            mapping_timeframes=self.mapping_timeframes,
            signal_artifacts=tuple(item.to_validation_spec() for item in self.signal_artifacts),
            require_hit_times_manifest=self.require_hit_times_manifest,
        )

    def to_prices_mappings_publish_validation_spec(self) -> ArtifactSlotValidationSpecV2:
        """
        Derive the explicit R3-04 publish spec for the `prices + mappings` stage.

        Args:
            None.
        Returns:
            ArtifactSlotValidationSpecV2: Config-driven validation scope for
                `build inactive slot -> validate whole slot -> atomically switch current.yaml`
                without `signals` or real `hit_times`.
        Assumptions:
            R3-04 keeps price and mapping timeframes from `validation_plan`, while
            `signal_artifacts=()` and `require_hit_times_manifest=false` remain explicit stage
            boundaries until later epics materialize those families.
        Raises:
            ValueError: If stored timeframe literals became invalid before translation.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
          - docs/runbooks/backtest-artifacts-rebuild.md
        """
        return ArtifactSlotValidationSpecV2(
            price_timeframes=self.price_timeframes,
            mapping_timeframes=self.mapping_timeframes,
            signal_artifacts=(),
            require_hit_times_manifest=False,
        )


@dataclass(frozen=True, slots=True)
class BacktestArtifactHitTimesGridRuntimeConfig:
    """
    Frozen hit-times grid contract loaded from `backtest_artifacts.hit_times_grid`.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - configs/dev/backtest_artifacts.yaml
      - docs/runbooks/backtest-artifacts-rebuild.md
    """

    tp_levels_pct: tuple[float, ...]
    sl_levels_pct: tuple[float, ...]

    def __post_init__(self) -> None:
        """
        Validate and canonically order TP/SL percentage grids.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Grid levels are stored as positive human-percent values and must stay deterministic.
        Raises:
            ValueError: If one grid is empty, non-positive, or contains duplicates.
        Side Effects:
            Replaces stored sequences with ascending unique tuples.
        """
        object.__setattr__(
            self,
            "tp_levels_pct",
            _normalize_positive_float_sequence(
                values=self.tp_levels_pct,
                field_path="backtest_artifacts.hit_times_grid.tp_levels_pct",
            ),
        )
        object.__setattr__(
            self,
            "sl_levels_pct",
            _normalize_positive_float_sequence(
                values=self.sl_levels_pct,
                field_path="backtest_artifacts.hit_times_grid.sl_levels_pct",
            ),
        )


@dataclass(frozen=True, slots=True)
class BacktestArtifactSlotPolicyRuntimeConfig:
    """
    Frozen slot policy loaded from `backtest_artifacts.slot_policy`.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - configs/dev/backtest_artifacts.yaml
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """

    slots: tuple[str, ...]

    def __post_init__(self) -> None:
        """
        Validate the fixed two-slot publish contract for milestone R2.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            R2 supports exactly `slot_a` and `slot_b`, regardless of author ordering.
        Raises:
            ValueError: If slot list is duplicated, incomplete, or contains unsupported values.
        Side Effects:
            Replaces stored slot tuple with canonical ordered slot literals.
        """
        object.__setattr__(
            self,
            "slots",
            _normalize_slot_sequence(
                values=self.slots,
                field_path="backtest_artifacts.slot_policy.slots",
            ),
        )


@dataclass(frozen=True, slots=True)
class BacktestArtifactPublishScheduleRuntimeConfig:
    """
    Frozen publish schedule contract loaded from `backtest_artifacts.publish_schedule`.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - configs/dev/backtest_artifacts.yaml
      - docs/architecture/backtest/README.md
    """

    full_rebuild_hour_utc: int
    full_rebuild_minute_utc: int

    def __post_init__(self) -> None:
        """
        Validate fixed UTC schedule fields for artifact rebuild planning.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            R2 stores schedule metadata as one daily UTC hour/minute pair.
        Raises:
            ValueError: If hour or minute is not an integer within UTC clock bounds.
        Side Effects:
            None.
        """
        _require_int_range(
            value=self.full_rebuild_hour_utc,
            field_path="backtest_artifacts.publish_schedule.full_rebuild_hour_utc",
            minimum=0,
            maximum=23,
        )
        _require_int_range(
            value=self.full_rebuild_minute_utc,
            field_path="backtest_artifacts.publish_schedule.full_rebuild_minute_utc",
            minimum=0,
            maximum=59,
        )


@dataclass(frozen=True, slots=True)
class BacktestArtifactLookbackPolicyRuntimeConfig:
    """
    Frozen lookback bounds loaded from `backtest_artifacts.lookback_policy`.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - configs/dev/backtest_artifacts.yaml
      - docs/runbooks/backtest-artifacts-rebuild.md
    """

    price_tail_bars_1m: int
    mapping_tail_bars_1m: int
    signal_tail_bars_1m: int
    hit_times_tail_bars_1m: int

    def __post_init__(self) -> None:
        """
        Validate positive 1m lookback budgets for artifact pipeline stages.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Each stage-specific tail window is configured in canonical `1m` bars.
        Raises:
            ValueError: If one lookback value is not a strict-positive integer.
        Side Effects:
            None.
        """
        _require_positive_int(
            value=self.price_tail_bars_1m,
            field_path="backtest_artifacts.lookback_policy.price_tail_bars_1m",
        )
        _require_positive_int(
            value=self.mapping_tail_bars_1m,
            field_path="backtest_artifacts.lookback_policy.mapping_tail_bars_1m",
        )
        _require_positive_int(
            value=self.signal_tail_bars_1m,
            field_path="backtest_artifacts.lookback_policy.signal_tail_bars_1m",
        )
        _require_positive_int(
            value=self.hit_times_tail_bars_1m,
            field_path="backtest_artifacts.lookback_policy.hit_times_tail_bars_1m",
        )


@dataclass(frozen=True, slots=True)
class BacktestArtifactValidationBudgetsRuntimeConfig:
    """
    Frozen validation budget bounds loaded from `backtest_artifacts.validation_budgets`.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - configs/dev/backtest_artifacts.yaml
      - docs/runbooks/backtest-artifacts-rebuild.md
    """

    max_price_bars_per_timeframe: int
    max_mapping_rows_per_timeframe: int
    max_signal_rows_per_artifact: int
    max_hit_times_cells: int
    max_hit_times_cells_full_rebuild: int

    def __post_init__(self) -> None:
        """
        Validate positive validation-budget bounds for whole-slot checks.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Budgets are guard rails for artifact validation workloads and remain additive in R2.
        Raises:
            ValueError: If one budget is not a strict-positive integer.
        Side Effects:
            None.
        """
        _require_positive_int(
            value=self.max_price_bars_per_timeframe,
            field_path="backtest_artifacts.validation_budgets.max_price_bars_per_timeframe",
        )
        _require_positive_int(
            value=self.max_mapping_rows_per_timeframe,
            field_path="backtest_artifacts.validation_budgets.max_mapping_rows_per_timeframe",
        )
        _require_positive_int(
            value=self.max_signal_rows_per_artifact,
            field_path="backtest_artifacts.validation_budgets.max_signal_rows_per_artifact",
        )
        _require_positive_int(
            value=self.max_hit_times_cells,
            field_path="backtest_artifacts.validation_budgets.max_hit_times_cells",
        )
        _require_positive_int(
            value=self.max_hit_times_cells_full_rebuild,
            field_path="backtest_artifacts.validation_budgets.max_hit_times_cells_full_rebuild",
        )
        if self.max_hit_times_cells_full_rebuild < self.max_hit_times_cells:
            raise ValueError(
                "backtest_artifacts.validation_budgets.max_hit_times_cells_full_rebuild "
                "must be >= max_hit_times_cells"
            )


@dataclass(frozen=True, slots=True)
class BacktestArtifactExecutionPolicyRuntimeConfig:
    """
    Strict execution-policy contract loaded from `backtest_artifacts.execution_policy`.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - configs/dev/backtest_artifacts.yaml
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """

    max_open_timeframe_sessions: int
    signal_worker_processes: int
    signal_worker_memory_budget_bytes: int
    signal_chunk_rows_min: int
    signal_chunk_rows_max: int

    def __post_init__(self) -> None:
        """
        Validate strict R12 execution-policy scalars with fail-fast semantics.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Offline precompute orchestration must not guess session or worker limits.
        Raises:
            ValueError: If one scalar is non-positive or the chunk-row bounds are inverted.
        Side Effects:
            None.
        """
        _require_positive_int(
            value=self.max_open_timeframe_sessions,
            field_path="backtest_artifacts.execution_policy.max_open_timeframe_sessions",
        )
        _require_positive_int(
            value=self.signal_worker_processes,
            field_path="backtest_artifacts.execution_policy.signal_worker_processes",
        )
        _require_positive_int(
            value=self.signal_worker_memory_budget_bytes,
            field_path=(
                "backtest_artifacts.execution_policy.signal_worker_memory_budget_bytes"
            ),
        )
        _require_positive_int(
            value=self.signal_chunk_rows_min,
            field_path="backtest_artifacts.execution_policy.signal_chunk_rows_min",
        )
        _require_positive_int(
            value=self.signal_chunk_rows_max,
            field_path="backtest_artifacts.execution_policy.signal_chunk_rows_max",
        )
        if self.signal_chunk_rows_min > self.signal_chunk_rows_max:
            raise ValueError(
                "backtest_artifacts.execution_policy.signal_chunk_rows_min must be <= "
                "signal_chunk_rows_max"
            )

    def to_execution_policy(self) -> ArtifactPrecomputeExecutionPolicyV2:
        """
        Translate runtime-config scalars into the typed service-layer execution policy.

        Args:
            None.
        Returns:
            ArtifactPrecomputeExecutionPolicyV2: Immutable execution-policy DTO for the runner.
        Assumptions:
            Translation is lossless because both contracts expose the same five strict fields.
        Raises:
            ValueError: If a stored scalar violates the service-layer contract.
        Side Effects:
            None.
        """
        return ArtifactPrecomputeExecutionPolicyV2(
            max_open_timeframe_sessions=self.max_open_timeframe_sessions,
            signal_worker_processes=self.signal_worker_processes,
            signal_worker_memory_budget_bytes=self.signal_worker_memory_budget_bytes,
            signal_chunk_rows_min=self.signal_chunk_rows_min,
            signal_chunk_rows_max=self.signal_chunk_rows_max,
        )


@dataclass(frozen=True, slots=True)
class BacktestArtifactsRuntimeConfig:
    """
    Strict artifact pipeline runtime config loaded from `configs/<env>/backtest_artifacts.yaml`.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - configs/dev/backtest_artifacts.yaml
      - apps/api/wiring/modules/backtest.py
    """

    version: int
    artifact_root: str
    validation_plan: BacktestArtifactValidationPlanRuntimeConfig
    hit_times_grid: BacktestArtifactHitTimesGridRuntimeConfig
    slot_policy: BacktestArtifactSlotPolicyRuntimeConfig
    publish_schedule: BacktestArtifactPublishScheduleRuntimeConfig
    lookback_policy: BacktestArtifactLookbackPolicyRuntimeConfig
    validation_budgets: BacktestArtifactValidationBudgetsRuntimeConfig
    execution_policy: BacktestArtifactExecutionPolicyRuntimeConfig

    def __post_init__(self) -> None:
        """
        Validate strict top-level artifact config invariants with fail-fast semantics.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            R2-04 publishes exactly one config schema version and a fixed slot contract.
        Raises:
            ValueError: If version, artifact root, or one required section is invalid.
        Side Effects:
            Normalizes stored artifact root literal.
        """
        if self.version != _ARTIFACTS_CONFIG_VERSION:
            raise ValueError(
                "backtest_artifacts.version must be "
                f"{_ARTIFACTS_CONFIG_VERSION}, got {self.version!r}"
            )
        object.__setattr__(
            self,
            "artifact_root",
            _validate_artifact_root_literal(
                value=self.artifact_root,
                field_path="backtest_artifacts.artifact_root",
            ),
        )
        if self.validation_plan is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest_artifacts.validation_plan section must be configured")
        if self.hit_times_grid is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest_artifacts.hit_times_grid section must be configured")
        if self.slot_policy is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest_artifacts.slot_policy section must be configured")
        if self.publish_schedule is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest_artifacts.publish_schedule section must be configured")
        if self.lookback_policy is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest_artifacts.lookback_policy section must be configured")
        if self.validation_budgets is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest_artifacts.validation_budgets section must be configured")
        if self.execution_policy is None:  # type: ignore[truthy-bool]
            raise ValueError("backtest_artifacts.execution_policy section must be configured")

    def artifact_root_path(self) -> Path:
        """
        Return the configured artifact store root as a `Path` object.

        Args:
            None.
        Returns:
            Path: Filesystem path literal from `artifact_root`.
        Assumptions:
            Relative paths are interpreted the same way as existing runtime config paths.
        Raises:
            None.
        Side Effects:
            None.
        """
        return Path(self.artifact_root)

    def to_validation_spec(self) -> ArtifactSlotValidationSpecV2:
        """
        Translate frozen artifact config into the publish-layer validation plan.

        Args:
            None.
        Returns:
            ArtifactSlotValidationSpecV2: Explicit deterministic whole-slot validation plan.
        Assumptions:
            Publish flow still consumes `ArtifactSlotValidationSpecV2` directly in R2.
        Raises:
            ValueError: If nested validation plan translation fails.
        Side Effects:
            None.
        """
        return self.validation_plan.to_validation_spec()

    def to_prices_mappings_publish_validation_spec(self) -> ArtifactSlotValidationSpecV2:
        """
        Derive the explicit R3-04 publish spec for the `prices + mappings` stage.

        Args:
            None.
        Returns:
            ArtifactSlotValidationSpecV2: Config-driven prices+mappings publish validation scope.
        Assumptions:
            Adapter wiring should derive this stage spec from the same source-of-truth config that
            also drives the full later-stage validation plan.
        Raises:
            ValueError: If nested validation-plan translation fails.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
          - docs/runbooks/backtest-artifacts-rebuild.md
        """
        return self.validation_plan.to_prices_mappings_publish_validation_spec()

    def to_precompute_runtime_settings(
        self,
        *,
        config_sha256: str,
    ) -> ArtifactPrecomputeRuntimeSettingsV2:
        """
        Translate strict artifact config into service-layer precompute runtime settings.

        Args:
            config_sha256: Deterministic hash of this normalized config payload.
        Returns:
            ArtifactPrecomputeRuntimeSettingsV2: Minimal immutable settings for runner wiring.
        Assumptions:
            Signal artifacts remain explicit config-driven targets and R4-03 tail lookback
            settings are forwarded without hidden defaults.
        Raises:
            ValueError: If nested lookback, signal-target, or budget contracts are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        return ArtifactPrecomputeRuntimeSettingsV2(
            price_tail_bars_1m=self.lookback_policy.price_tail_bars_1m,
            mapping_tail_bars_1m=self.lookback_policy.mapping_tail_bars_1m,
            signal_tail_bars_1m=self.lookback_policy.signal_tail_bars_1m,
            hit_times_tail_bars_1m=self.lookback_policy.hit_times_tail_bars_1m,
            hit_times_tp_levels_pct=self.hit_times_grid.tp_levels_pct,
            hit_times_sl_levels_pct=self.hit_times_grid.sl_levels_pct,
            price_timeframes=self.validation_plan.price_timeframes,
            mapping_timeframes=self.validation_plan.mapping_timeframes,
            config_sha256=config_sha256,
            execution_policy=self.execution_policy.to_execution_policy(),
            signal_artifacts=tuple(
                item.to_validation_spec() for item in self.validation_plan.signal_artifacts
            ),
            max_signal_rows_per_artifact=(
                self.validation_budgets.max_signal_rows_per_artifact
            ),
            max_hit_times_cells=self.validation_budgets.max_hit_times_cells,
            max_hit_times_cells_full_rebuild=(
                self.validation_budgets.max_hit_times_cells_full_rebuild
            ),
        )


def resolve_backtest_artifacts_config_path(
    *,
    environ: Mapping[str, str],
) -> Path:
    """
    Resolve artifact runtime config path using explicit override-first precedence.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
    Related:
      - configs/dev/backtest_artifacts.yaml
      - apps/api/wiring/modules/backtest.py

    Args:
        environ: Runtime environment mapping.
    Returns:
        Path: Resolved `backtest_artifacts.yaml` path.
    Assumptions:
        Precedence is `ROEHUB_BACKTEST_ARTIFACTS_CONFIG` >
        `configs/<ROEHUB_ENV>/backtest_artifacts.yaml`.
    Raises:
        ValueError: If `ROEHUB_ENV` value is unsupported.
    Side Effects:
        None.
    """
    override_path = environ.get(_BACKTEST_ARTIFACTS_CONFIG_PATH_KEY, "").strip()
    if override_path:
        return Path(override_path)

    env_name = resolve_backtest_env_name(environ=environ)
    return Path("configs") / env_name / "backtest_artifacts.yaml"


def load_backtest_artifacts_runtime_config(path: str | Path) -> BacktestArtifactsRuntimeConfig:
    """
    Load and strictly validate the artifact pipeline runtime YAML configuration.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - configs/dev/backtest_artifacts.yaml
      - apps/api/wiring/modules/backtest.py

    Args:
        path: Path to `backtest_artifacts.yaml`.
    Returns:
        BacktestArtifactsRuntimeConfig: Parsed validated artifact config object.
    Assumptions:
        All sections are strict-required and reject missing/extra keys fail-fast.
    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If YAML shape, keys, or values are invalid.
    Side Effects:
        Reads one UTF-8 YAML file from filesystem.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"backtest artifacts config not found: {config_path}")

    payload = _load_yaml_mapping(path=config_path)
    _require_exact_keys(
        data=payload,
        expected_keys=_TOP_LEVEL_REQUIRED_KEYS,
        field_path="backtest_artifacts",
    )
    version = _require_int(
        value=payload.get("version"),
        field_path="backtest_artifacts.version",
    )
    artifacts_map = _require_mapping(
        value=payload.get("backtest_artifacts"),
        field_path="backtest_artifacts",
    )
    _require_exact_keys(
        data=artifacts_map,
        expected_keys=_ARTIFACTS_REQUIRED_KEYS,
        field_path="backtest_artifacts",
    )

    validation_plan_map = _require_mapping(
        value=artifacts_map.get("validation_plan"),
        field_path="backtest_artifacts.validation_plan",
    )
    _require_exact_keys(
        data=validation_plan_map,
        expected_keys=_VALIDATION_PLAN_REQUIRED_KEYS,
        field_path="backtest_artifacts.validation_plan",
    )

    hit_times_grid_map = _require_mapping(
        value=artifacts_map.get("hit_times_grid"),
        field_path="backtest_artifacts.hit_times_grid",
    )
    _require_exact_keys(
        data=hit_times_grid_map,
        expected_keys=_HIT_TIMES_GRID_REQUIRED_KEYS,
        field_path="backtest_artifacts.hit_times_grid",
    )

    slot_policy_map = _require_mapping(
        value=artifacts_map.get("slot_policy"),
        field_path="backtest_artifacts.slot_policy",
    )
    _require_exact_keys(
        data=slot_policy_map,
        expected_keys=_SLOT_POLICY_REQUIRED_KEYS,
        field_path="backtest_artifacts.slot_policy",
    )

    publish_schedule_map = _require_mapping(
        value=artifacts_map.get("publish_schedule"),
        field_path="backtest_artifacts.publish_schedule",
    )
    _require_exact_keys(
        data=publish_schedule_map,
        expected_keys=_PUBLISH_SCHEDULE_REQUIRED_KEYS,
        field_path="backtest_artifacts.publish_schedule",
    )

    lookback_policy_map = _require_mapping(
        value=artifacts_map.get("lookback_policy"),
        field_path="backtest_artifacts.lookback_policy",
    )
    _require_exact_keys(
        data=lookback_policy_map,
        expected_keys=_LOOKBACK_POLICY_REQUIRED_KEYS,
        field_path="backtest_artifacts.lookback_policy",
    )

    validation_budgets_map = _require_mapping(
        value=artifacts_map.get("validation_budgets"),
        field_path="backtest_artifacts.validation_budgets",
    )
    _require_exact_keys(
        data=validation_budgets_map,
        expected_keys=_VALIDATION_BUDGETS_REQUIRED_KEYS,
        field_path="backtest_artifacts.validation_budgets",
    )
    execution_policy_map = _require_mapping(
        value=artifacts_map.get("execution_policy"),
        field_path="backtest_artifacts.execution_policy",
    )
    _require_exact_keys(
        data=execution_policy_map,
        expected_keys=_EXECUTION_POLICY_REQUIRED_KEYS,
        field_path="backtest_artifacts.execution_policy",
    )

    signal_artifacts = _load_signal_artifacts(
        value=validation_plan_map.get("signal_artifacts"),
        field_path="backtest_artifacts.validation_plan.signal_artifacts",
    )

    return BacktestArtifactsRuntimeConfig(
        version=version,
        artifact_root=_require_str(
            value=artifacts_map.get("artifact_root"),
            field_path="backtest_artifacts.artifact_root",
        ),
        validation_plan=BacktestArtifactValidationPlanRuntimeConfig(
            price_timeframes=_require_str_sequence(
                value=validation_plan_map.get("price_timeframes"),
                field_path="backtest_artifacts.validation_plan.price_timeframes",
            ),
            mapping_timeframes=_require_str_sequence(
                value=validation_plan_map.get("mapping_timeframes"),
                field_path="backtest_artifacts.validation_plan.mapping_timeframes",
            ),
            signal_artifacts=signal_artifacts,
            require_hit_times_manifest=_require_bool(
                value=validation_plan_map.get("require_hit_times_manifest"),
                field_path="backtest_artifacts.validation_plan.require_hit_times_manifest",
            ),
        ),
        hit_times_grid=BacktestArtifactHitTimesGridRuntimeConfig(
            tp_levels_pct=_require_numeric_sequence(
                value=hit_times_grid_map.get("tp_levels_pct"),
                field_path="backtest_artifacts.hit_times_grid.tp_levels_pct",
            ),
            sl_levels_pct=_require_numeric_sequence(
                value=hit_times_grid_map.get("sl_levels_pct"),
                field_path="backtest_artifacts.hit_times_grid.sl_levels_pct",
            ),
        ),
        slot_policy=BacktestArtifactSlotPolicyRuntimeConfig(
            slots=_require_str_sequence(
                value=slot_policy_map.get("slots"),
                field_path="backtest_artifacts.slot_policy.slots",
            )
        ),
        publish_schedule=BacktestArtifactPublishScheduleRuntimeConfig(
            full_rebuild_hour_utc=_require_int(
                value=publish_schedule_map.get("full_rebuild_hour_utc"),
                field_path="backtest_artifacts.publish_schedule.full_rebuild_hour_utc",
            ),
            full_rebuild_minute_utc=_require_int(
                value=publish_schedule_map.get("full_rebuild_minute_utc"),
                field_path="backtest_artifacts.publish_schedule.full_rebuild_minute_utc",
            ),
        ),
        lookback_policy=BacktestArtifactLookbackPolicyRuntimeConfig(
            price_tail_bars_1m=_require_int(
                value=lookback_policy_map.get("price_tail_bars_1m"),
                field_path="backtest_artifacts.lookback_policy.price_tail_bars_1m",
            ),
            mapping_tail_bars_1m=_require_int(
                value=lookback_policy_map.get("mapping_tail_bars_1m"),
                field_path="backtest_artifacts.lookback_policy.mapping_tail_bars_1m",
            ),
            signal_tail_bars_1m=_require_int(
                value=lookback_policy_map.get("signal_tail_bars_1m"),
                field_path="backtest_artifacts.lookback_policy.signal_tail_bars_1m",
            ),
            hit_times_tail_bars_1m=_require_int(
                value=lookback_policy_map.get("hit_times_tail_bars_1m"),
                field_path="backtest_artifacts.lookback_policy.hit_times_tail_bars_1m",
            ),
        ),
        validation_budgets=BacktestArtifactValidationBudgetsRuntimeConfig(
            max_price_bars_per_timeframe=_require_int(
                value=validation_budgets_map.get("max_price_bars_per_timeframe"),
                field_path=("backtest_artifacts.validation_budgets.max_price_bars_per_timeframe"),
            ),
            max_mapping_rows_per_timeframe=_require_int(
                value=validation_budgets_map.get("max_mapping_rows_per_timeframe"),
                field_path=("backtest_artifacts.validation_budgets.max_mapping_rows_per_timeframe"),
            ),
            max_signal_rows_per_artifact=_require_int(
                value=validation_budgets_map.get("max_signal_rows_per_artifact"),
                field_path=("backtest_artifacts.validation_budgets.max_signal_rows_per_artifact"),
            ),
            max_hit_times_cells=_require_int(
                value=validation_budgets_map.get("max_hit_times_cells"),
                field_path="backtest_artifacts.validation_budgets.max_hit_times_cells",
            ),
            max_hit_times_cells_full_rebuild=_require_int(
                value=validation_budgets_map.get("max_hit_times_cells_full_rebuild"),
                field_path=(
                    "backtest_artifacts.validation_budgets.max_hit_times_cells_full_rebuild"
                ),
            ),
        ),
        execution_policy=BacktestArtifactExecutionPolicyRuntimeConfig(
            max_open_timeframe_sessions=_require_int(
                value=execution_policy_map.get("max_open_timeframe_sessions"),
                field_path=(
                    "backtest_artifacts.execution_policy.max_open_timeframe_sessions"
                ),
            ),
            signal_worker_processes=_require_int(
                value=execution_policy_map.get("signal_worker_processes"),
                field_path="backtest_artifacts.execution_policy.signal_worker_processes",
            ),
            signal_worker_memory_budget_bytes=_require_int(
                value=execution_policy_map.get("signal_worker_memory_budget_bytes"),
                field_path=(
                    "backtest_artifacts.execution_policy.signal_worker_memory_budget_bytes"
                ),
            ),
            signal_chunk_rows_min=_require_int(
                value=execution_policy_map.get("signal_chunk_rows_min"),
                field_path="backtest_artifacts.execution_policy.signal_chunk_rows_min",
            ),
            signal_chunk_rows_max=_require_int(
                value=execution_policy_map.get("signal_chunk_rows_max"),
                field_path="backtest_artifacts.execution_policy.signal_chunk_rows_max",
            ),
        ),
    )


def build_backtest_artifacts_runtime_config_hash(
    *,
    config: BacktestArtifactsRuntimeConfig,
) -> str:
    """
    Build deterministic SHA-256 hash for the canonical artifact config payload.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
      - configs/dev/backtest_artifacts.yaml

    Args:
        config: Parsed artifact runtime config object.
    Returns:
        str: Canonical SHA-256 hash string.
    Assumptions:
        Hash includes all strict artifact pipeline sections to support future provenance use.
    Raises:
        TypeError: If payload normalization encounters an unsupported object type.
    Side Effects:
        None.
    """
    payload = {
        "backtest_artifacts": {
            "artifact_root": config.artifact_root,
            "validation_plan": {
                "price_timeframes": config.validation_plan.price_timeframes,
                "mapping_timeframes": config.validation_plan.mapping_timeframes,
                "signal_artifacts": tuple(
                    {
                        "timeframe": item.timeframe,
                        "indicator_id": item.indicator_id,
                    }
                    for item in config.validation_plan.signal_artifacts
                ),
                "require_hit_times_manifest": config.validation_plan.require_hit_times_manifest,
            },
            "hit_times_grid": {
                "tp_levels_pct": config.hit_times_grid.tp_levels_pct,
                "sl_levels_pct": config.hit_times_grid.sl_levels_pct,
            },
            "slot_policy": {
                "slots": config.slot_policy.slots,
            },
            "publish_schedule": {
                "full_rebuild_hour_utc": config.publish_schedule.full_rebuild_hour_utc,
                "full_rebuild_minute_utc": config.publish_schedule.full_rebuild_minute_utc,
            },
            "lookback_policy": {
                "price_tail_bars_1m": config.lookback_policy.price_tail_bars_1m,
                "mapping_tail_bars_1m": config.lookback_policy.mapping_tail_bars_1m,
                "signal_tail_bars_1m": config.lookback_policy.signal_tail_bars_1m,
                "hit_times_tail_bars_1m": config.lookback_policy.hit_times_tail_bars_1m,
            },
            "validation_budgets": {
                "max_price_bars_per_timeframe": (
                    config.validation_budgets.max_price_bars_per_timeframe
                ),
                "max_mapping_rows_per_timeframe": (
                    config.validation_budgets.max_mapping_rows_per_timeframe
                ),
                "max_signal_rows_per_artifact": (
                    config.validation_budgets.max_signal_rows_per_artifact
                ),
                "max_hit_times_cells": config.validation_budgets.max_hit_times_cells,
                "max_hit_times_cells_full_rebuild": (
                    config.validation_budgets.max_hit_times_cells_full_rebuild
                ),
            },
            "execution_policy": {
                "max_open_timeframe_sessions": (
                    config.execution_policy.max_open_timeframe_sessions
                ),
                "signal_worker_processes": config.execution_policy.signal_worker_processes,
                "signal_worker_memory_budget_bytes": (
                    config.execution_policy.signal_worker_memory_budget_bytes
                ),
                "signal_chunk_rows_min": config.execution_policy.signal_chunk_rows_min,
                "signal_chunk_rows_max": config.execution_policy.signal_chunk_rows_max,
            },
        }
    }
    canonical_json = json.dumps(
        _normalize_json_value(value=payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _load_yaml_mapping(*, path: Path) -> Mapping[str, Any]:
    """
    Read one UTF-8 YAML file into a mapping while rejecting duplicate keys.

    Args:
        path: Artifact config YAML path.
    Returns:
        Mapping[str, Any]: Parsed top-level mapping payload.
    Assumptions:
        Artifact config is authored as one mapping document.
    Raises:
        ValueError: If YAML parsing fails or the document root is not a mapping.
    Side Effects:
        Reads one UTF-8 file from disk.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"backtest_artifacts YAML parse error in {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("backtest_artifacts config must be mapping at top-level")
    return payload


def _require_exact_keys(
    *,
    data: Mapping[str, Any],
    expected_keys: tuple[str, ...],
    field_path: str,
) -> None:
    """
    Enforce exact required keys for one strict artifact config mapping.

    Args:
        data: Mapping payload to validate.
        expected_keys: Canonical key set allowed for the mapping.
        field_path: Stable field path used in fail-fast diagnostics.
    Returns:
        None.
    Assumptions:
        Missing and extra keys must both be rejected deterministically.
    Raises:
        ValueError: If required keys are missing or unsupported keys are present.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    actual_keys = set(data.keys())
    expected_key_set = set(expected_keys)
    missing_keys = tuple(sorted(expected_key_set - actual_keys))
    extra_keys = tuple(sorted(actual_keys - expected_key_set))
    if missing_keys:
        raise ValueError(f"{field_path} missing required keys {missing_keys!r}")
    if extra_keys:
        raise ValueError(f"{field_path} contains unsupported keys {extra_keys!r}")


def _load_signal_artifacts(
    *,
    value: Any,
    field_path: str,
) -> tuple[BacktestArtifactSignalRuntimeConfig, ...]:
    """
    Parse strict `signal_artifacts` items from YAML into frozen config dataclasses.

    Args:
        value: Raw YAML value for `signal_artifacts`.
        field_path: Stable field path used in diagnostics.
    Returns:
        tuple[BacktestArtifactSignalRuntimeConfig, ...]: Parsed signal artifact targets.
    Assumptions:
        YAML may either enumerate explicit `{timeframe, indicator_id}` items or use the special
        literal `all_supported_v1` to expand every registry-backed signal indicator across all
        allowed artifact signal timeframes.
    Raises:
        ValueError: If the sequence or one item violates the strict shape contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - configs/dev/backtest_artifacts.yaml
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    if isinstance(value, str):
        normalized_literal = value.strip().lower()
        if normalized_literal != _SIGNAL_ARTIFACTS_ALL_SUPPORTED_LITERAL:
            raise ValueError(
                f"{field_path} string literal must be "
                f"{_SIGNAL_ARTIFACTS_ALL_SUPPORTED_LITERAL!r}"
            )
        return _expand_all_supported_signal_artifacts()
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{field_path} must be sequence")

    parsed_items: list[BacktestArtifactSignalRuntimeConfig] = []
    for index, item in enumerate(value):
        item_field_path = f"{field_path}[{index}]"
        item_map = _require_mapping(value=item, field_path=item_field_path)
        _require_exact_keys(
            data=item_map,
            expected_keys=_SIGNAL_ARTIFACT_REQUIRED_KEYS,
            field_path=item_field_path,
        )
        parsed_items.append(
            BacktestArtifactSignalRuntimeConfig(
                timeframe=_require_str(
                    value=item_map.get("timeframe"),
                    field_path=f"{item_field_path}.timeframe",
                ),
                indicator_id=_require_str(
                    value=item_map.get("indicator_id"),
                    field_path=f"{item_field_path}.indicator_id",
                ),
            )
        )
    return tuple(parsed_items)


def _expand_all_supported_signal_artifacts() -> tuple[BacktestArtifactSignalRuntimeConfig, ...]:
    """
    Expand the machine-readable `all_supported_v1` signal artifact literal.

    Args:
        None.
    Returns:
        tuple[BacktestArtifactSignalRuntimeConfig, ...]: Full deterministic signal artifact matrix.
    Assumptions:
        Every indicator returned by `supported_indicator_ids_for_signals_v1()` is publishable on
        every artifact signal timeframe.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
    """
    return tuple(
        BacktestArtifactSignalRuntimeConfig(
            timeframe=timeframe,
            indicator_id=indicator_id,
        )
        for timeframe in ARTIFACT_SIGNAL_TIMEFRAMES_V2
        for indicator_id in supported_indicator_ids_for_signals_v1()
    )


def _require_mapping(*, value: Any, field_path: str) -> Mapping[str, Any]:
    """
    Require that one raw YAML value is a mapping with string keys.

    Args:
        value: Raw YAML value.
        field_path: Stable field path used in diagnostics.
    Returns:
        Mapping[str, Any]: Mapping payload when valid.
    Assumptions:
        Artifact config uses string keys only.
    Raises:
        ValueError: If the value is not a mapping.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_path} must be mapping")
    return value


def _require_str(*, value: Any, field_path: str) -> str:
    """
    Require that one raw YAML value is a string.

    Args:
        value: Raw YAML value.
        field_path: Stable field path used in diagnostics.
    Returns:
        str: Original string value.
    Assumptions:
        Scalar artifact config literals are not coerced from other types.
    Raises:
        ValueError: If the value is not a string.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_path} must be str")
    return value


def _require_int(*, value: Any, field_path: str) -> int:
    """
    Require that one raw YAML value is an integer and not boolean.

    Args:
        value: Raw YAML value.
        field_path: Stable field path used in diagnostics.
    Returns:
        int: Parsed integer value.
    Assumptions:
        Bool values must not be accepted as integers in strict config fields.
    Raises:
        ValueError: If the value is not an integer.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_path} must be int")
    return value


def _require_bool(*, value: Any, field_path: str) -> bool:
    """
    Require that one raw YAML value is a boolean.

    Args:
        value: Raw YAML value.
        field_path: Stable field path used in diagnostics.
    Returns:
        bool: Parsed boolean value.
    Assumptions:
        Strict config does not coerce strings like `true` after YAML parsing.
    Raises:
        ValueError: If the value is not boolean.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    if not isinstance(value, bool):
        raise ValueError(f"{field_path} must be bool")
    return value


def _require_str_sequence(*, value: Any, field_path: str) -> tuple[str, ...]:
    """
    Require that one raw YAML value is a sequence of strings.

    Args:
        value: Raw YAML value.
        field_path: Stable field path used in diagnostics.
    Returns:
        tuple[str, ...]: Parsed string tuple preserving author order.
    Assumptions:
        Canonical ordering and duplicate checks are applied later by dataclass validation.
    Raises:
        ValueError: If the value is not a sequence of strings.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_path} must be sequence")
    normalized_values: list[str] = []
    for index, item in enumerate(value):
        normalized_values.append(_require_str(value=item, field_path=f"{field_path}[{index}]"))
    return tuple(normalized_values)


def _require_numeric_sequence(*, value: Any, field_path: str) -> tuple[float, ...]:
    """
    Require that one raw YAML value is a sequence of numeric scalars.

    Args:
        value: Raw YAML value.
        field_path: Stable field path used in diagnostics.
    Returns:
        tuple[float, ...]: Parsed float tuple preserving author order.
    Assumptions:
        Integer and float YAML scalars are both accepted before positivity checks.
    Raises:
        ValueError: If the value is not a sequence of numeric scalars.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_path} must be sequence")
    normalized_values: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field_path}[{index}] must be numeric")
        normalized_values.append(float(item))
    return tuple(normalized_values)


def _normalize_timeframe_sequence(
    *,
    values: Sequence[str],
    field_path: str,
    allowed_literals: tuple[str, ...],
    validator,
) -> tuple[str, ...]:
    """
    Validate, deduplicate, and canonically order one timeframe sequence.

    Args:
        values: Candidate timeframe literals.
        field_path: Stable field path used in diagnostics.
        allowed_literals: Canonical ordering contract for the timeframe family.
        validator: Literal validator callable for the timeframe family.
    Returns:
        tuple[str, ...]: Canonically ordered unique timeframe literals.
    Assumptions:
        Empty timeframe lists are invalid because publish validation targets must be explicit.
    Raises:
        ValueError: If the sequence is empty, contains duplicates, or has invalid literals.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    seen: set[str] = set()
    validated_values: list[str] = []
    for index, raw_value in enumerate(values):
        validated_value = validator(
            _require_str(value=raw_value, field_path=f"{field_path}[{index}]")
        )
        if validated_value in seen:
            raise ValueError(f"{field_path} contains duplicate {validated_value!r}")
        seen.add(validated_value)
        validated_values.append(validated_value)
    if not validated_values:
        raise ValueError(f"{field_path} must be non-empty")
    allowed_order = {literal: index for index, literal in enumerate(allowed_literals)}
    return tuple(sorted(validated_values, key=lambda item: allowed_order[item]))


def _normalize_signal_artifacts(
    *,
    values: Sequence[BacktestArtifactSignalRuntimeConfig],
    field_path: str,
) -> tuple[BacktestArtifactSignalRuntimeConfig, ...]:
    """
    Validate, deduplicate, and canonically order explicit signal artifact targets.

    Args:
        values: Candidate signal artifact config items.
        field_path: Stable field path used in diagnostics.
    Returns:
        tuple[BacktestArtifactSignalRuntimeConfig, ...]: Canonically ordered unique targets.
    Assumptions:
        Ordering follows signal timeframe contract first and `indicator_id` second.
    Raises:
        ValueError: If one `(timeframe, indicator_id)` pair is duplicated.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    timeframe_order = {
        literal: index for index, literal in enumerate(ARTIFACT_SIGNAL_TIMEFRAMES_V2)
    }
    seen: set[tuple[str, str]] = set()
    validated_values: list[BacktestArtifactSignalRuntimeConfig] = []
    for item in values:
        validated_item = BacktestArtifactSignalRuntimeConfig(
            timeframe=item.timeframe,
            indicator_id=item.indicator_id,
        )
        identity = (validated_item.timeframe, validated_item.indicator_id)
        if identity in seen:
            raise ValueError(f"{field_path} contains duplicate {identity!r}")
        seen.add(identity)
        validated_values.append(validated_item)
    return tuple(
        sorted(
            validated_values,
            key=lambda item: (timeframe_order[item.timeframe], item.indicator_id),
        )
    )


def _normalize_positive_float_sequence(
    *,
    values: Sequence[float],
    field_path: str,
) -> tuple[float, ...]:
    """
    Validate, deduplicate, and canonically order one positive numeric sequence.

    Args:
        values: Candidate numeric values.
        field_path: Stable field path used in diagnostics.
    Returns:
        tuple[float, ...]: Ascending unique positive float values.
    Assumptions:
        Hit-times grid levels are represented as human-percent scalars.
    Raises:
        ValueError: If the sequence is empty, non-positive, or contains duplicates.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - configs/dev/backtest_artifacts.yaml
    """
    normalized_values: list[float] = []
    seen: set[float] = set()
    for index, raw_value in enumerate(values):
        numeric_value = float(raw_value)
        if numeric_value <= 0.0:
            raise ValueError(f"{field_path}[{index}] must be > 0")
        if numeric_value in seen:
            raise ValueError(f"{field_path} contains duplicate {numeric_value!r}")
        seen.add(numeric_value)
        normalized_values.append(numeric_value)
    if not normalized_values:
        raise ValueError(f"{field_path} must be non-empty")
    return tuple(sorted(normalized_values))


def _normalize_slot_sequence(
    *,
    values: Sequence[str],
    field_path: str,
) -> tuple[str, ...]:
    """
    Validate and canonically order the fixed two-slot contract for R2.

    Args:
        values: Candidate slot literals from config.
        field_path: Stable field path used in diagnostics.
    Returns:
        tuple[str, ...]: Canonical ordered slot tuple `(\"slot_a\", \"slot_b\")`.
    Assumptions:
        Author order may differ, but the allowed set must match the fixed slot contract exactly.
    Raises:
        ValueError: If values are duplicated, incomplete, or contain unsupported slots.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    seen: set[str] = set()
    validated_values: list[str] = []
    for index, raw_value in enumerate(values):
        validated_value = validate_artifact_slot_v2(
            _require_str(value=raw_value, field_path=f"{field_path}[{index}]")
        )
        if validated_value in seen:
            raise ValueError(f"{field_path} contains duplicate {validated_value!r}")
        seen.add(validated_value)
        validated_values.append(validated_value)
    if tuple(sorted(validated_values)) != tuple(sorted(ALLOWED_ARTIFACT_SLOTS_V2)):
        raise ValueError(f"{field_path} must contain exactly {ordered_artifact_slots_v2()!r}")
    return ordered_artifact_slots_v2()


def _validate_artifact_root_literal(*, value: str, field_path: str) -> str:
    """
    Validate one artifact root path literal without hidden normalization.

    Args:
        value: Candidate artifact root literal.
        field_path: Stable field path used in diagnostics.
    Returns:
        str: Original artifact root literal when valid.
    Assumptions:
        Root path may be relative or absolute, but must be a non-empty explicit string literal.
    Raises:
        ValueError: If the path is empty, contains leading/trailing whitespace, or includes NUL.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - apps/api/wiring/modules/backtest.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    if not value.strip():
        raise ValueError(f"{field_path} must be non-empty path literal")
    if value != value.strip():
        raise ValueError(f"{field_path} must not have leading or trailing whitespace")
    if "\x00" in value:
        raise ValueError(f"{field_path} must not contain NUL")
    return value


def _require_positive_int(*, value: int, field_path: str) -> None:
    """
    Require that one integer field is strict-positive.

    Args:
        value: Integer value to validate.
        field_path: Stable field path used in diagnostics.
    Returns:
        None.
    Assumptions:
        The caller already ensured the value is an integer, not boolean.
    Raises:
        ValueError: If the integer is zero or negative.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    if value <= 0:
        raise ValueError(f"{field_path} must be > 0")


def _require_int_range(
    *,
    value: int,
    field_path: str,
    minimum: int,
    maximum: int,
) -> None:
    """
    Require that one integer field falls inside an inclusive range.

    Args:
        value: Integer value to validate.
        field_path: Stable field path used in diagnostics.
        minimum: Inclusive lower bound.
        maximum: Inclusive upper bound.
    Returns:
        None.
    Assumptions:
        The caller already ensured the value is an integer, not boolean.
    Raises:
        ValueError: If the integer falls outside the inclusive range.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    if value < minimum or value > maximum:
        raise ValueError(f"{field_path} must be in [{minimum}, {maximum}]")


def _normalize_json_value(*, value: Any) -> Any:
    """
    Recursively normalize Python values into canonical JSON-serializable structures.

    Args:
        value: Arbitrary nested payload node.
    Returns:
        Any: Canonical JSON-serializable node.
    Assumptions:
        Dict keys are strings and tuples should serialize as arrays.
    Raises:
        TypeError: If one node type is unsupported by canonical hashing.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_value(value=item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple):
        return [_normalize_json_value(value=item) for item in value]
    if isinstance(value, list):
        return [_normalize_json_value(value=item) for item in value]
    raise TypeError(f"unsupported artifact config hash payload type: {type(value)!r}")
