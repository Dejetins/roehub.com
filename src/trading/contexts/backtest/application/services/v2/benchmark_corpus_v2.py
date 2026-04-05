"""Typed benchmark-corpus helpers for backtest runtime acceleration rollout evidence.

Docs:
  - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
  - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
  - docs/architecture/backtest/backtest-v2-benchmarks.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .execution_profile_v2 import (
    ExecutionProfileModeLiteralV2,
    validate_execution_profile_mode_v2,
)

BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_SCHEMA_VERSION_V2 = 1
BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_KIND_V2 = (
    "backtest_runtime_acceleration_benchmark_corpus_v1"
)
BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_MILESTONE_ID_V2 = "D"
BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_EPIC_ID_V2 = "D2+D3"

type BenchmarkCorpusSliceIdLiteralV2 = Literal[
    "exact_baseline",
    "low_activity",
    "high_correlation",
    "small_grid_overhead",
    "memory_footprint",
]
type BenchmarkCorpusRolloutScopeLiteralV2 = Literal[
    "exact_only",
    "hybrid_rollout",
    "plugin_rollout",
]
type BenchmarkCorpusStageLiteralV2 = Literal["stage_a", "stage_b", "finalizing"]
type BenchmarkCorpusRolloutGateIdLiteralV2 = Literal[
    "top_1_recall",
    "top_10_overlap",
    "low_activity",
    "high_correlation",
    "small_grid_overhead",
    "memory_footprint",
]

_ALLOWED_BENCHMARK_CORPUS_SLICE_IDS_V2: tuple[BenchmarkCorpusSliceIdLiteralV2, ...] = (
    "exact_baseline",
    "low_activity",
    "high_correlation",
    "small_grid_overhead",
    "memory_footprint",
)
_ALLOWED_BENCHMARK_CORPUS_ROLLOUT_SCOPES_V2: tuple[
    BenchmarkCorpusRolloutScopeLiteralV2, ...
] = (
    "exact_only",
    "hybrid_rollout",
    "plugin_rollout",
)
_ALLOWED_BENCHMARK_CORPUS_STAGE_IDS_V2: tuple[BenchmarkCorpusStageLiteralV2, ...] = (
    "stage_a",
    "stage_b",
    "finalizing",
)
_ALLOWED_BENCHMARK_CORPUS_ROLLOUT_GATE_IDS_V2: tuple[
    BenchmarkCorpusRolloutGateIdLiteralV2, ...
] = (
    "top_1_recall",
    "top_10_overlap",
    "low_activity",
    "high_correlation",
    "small_grid_overhead",
    "memory_footprint",
)


@dataclass(frozen=True, slots=True)
class BacktestRuntimeBenchmarkSourceFixturesV2:
    """
    Canonical fixture paths reused by the runtime-acceleration benchmark corpus.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json
      - tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json
      - tests/unit/contexts/backtest/application/services/v2/fixtures/
        stage_b_golden_fixtures_v2.json
    """

    r0_benchmark_scenarios: str
    r5_stage_b_manifest: str
    stage_b_golden_fixture: str

    def __post_init__(self) -> None:
        """
        Validate benchmark-corpus fixture path literals.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Paths are repository-relative JSON fixtures committed under version control.
        Raises:
            ValueError: If one fixture path is blank or does not end with `.json`.
        Side Effects:
            None.
        """
        for field_name, field_value in (
            ("r0_benchmark_scenarios", self.r0_benchmark_scenarios),
            ("r5_stage_b_manifest", self.r5_stage_b_manifest),
            ("stage_b_golden_fixture", self.stage_b_golden_fixture),
        ):
            if not field_value.strip():
                raise ValueError(
                    f"BacktestRuntimeBenchmarkSourceFixturesV2.{field_name} must be non-empty"
                )
            if not field_value.endswith(".json"):
                raise ValueError(
                    f"BacktestRuntimeBenchmarkSourceFixturesV2.{field_name} must point to JSON"
                )


@dataclass(frozen=True, slots=True)
class BacktestRuntimeBenchmarkRolloutGateV2:
    """
    One explicit rollout threshold used to compare hybrid evidence against exact baseline.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/benchmark_corpus_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    """

    metric: str
    slice_id: BenchmarkCorpusSliceIdLiteralV2 | None
    min_ratio: float | None = None
    max_ratio: float | None = None
    min_distinct_count: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        """
        Validate one explicit rollout-gate threshold payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Gates stay additive and conservative; every gate defines at least one threshold or
            count-based requirement that later perf-smoke evidence can assert deterministically.
        Raises:
            ValueError: If metric/notes are blank or no threshold/count is configured.
        Side Effects:
            None.
        """
        if not self.metric.strip():
            raise ValueError("BacktestRuntimeBenchmarkRolloutGateV2.metric must be non-empty")
        if not self.notes.strip():
            raise ValueError("BacktestRuntimeBenchmarkRolloutGateV2.notes must be non-empty")
        if (
            self.min_ratio is None
            and self.max_ratio is None
            and self.min_distinct_count is None
        ):
            raise ValueError(
                "BacktestRuntimeBenchmarkRolloutGateV2 must declare one threshold or count"
            )
        if self.min_ratio is not None and self.min_ratio < 0.0:
            raise ValueError(
                "BacktestRuntimeBenchmarkRolloutGateV2.min_ratio must be >= 0 when provided"
            )
        if self.max_ratio is not None and self.max_ratio < 0.0:
            raise ValueError(
                "BacktestRuntimeBenchmarkRolloutGateV2.max_ratio must be >= 0 when provided"
            )
        if self.min_distinct_count is not None and self.min_distinct_count <= 0:
            raise ValueError(
                "BacktestRuntimeBenchmarkRolloutGateV2.min_distinct_count must be > 0 when "
                "provided"
            )


@dataclass(frozen=True, slots=True)
class BacktestRuntimeBenchmarkRolloutGatesV2:
    """
    Explicit rollout-gate bundle for conservative hybrid shortlist evidence.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/benchmark_corpus_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    """

    top_1_recall: BacktestRuntimeBenchmarkRolloutGateV2
    top_10_overlap: BacktestRuntimeBenchmarkRolloutGateV2
    low_activity: BacktestRuntimeBenchmarkRolloutGateV2
    high_correlation: BacktestRuntimeBenchmarkRolloutGateV2
    small_grid_overhead: BacktestRuntimeBenchmarkRolloutGateV2
    memory_footprint: BacktestRuntimeBenchmarkRolloutGateV2


@dataclass(frozen=True, slots=True)
class BacktestRuntimeBenchmarkSyntheticRunSpecV2:
    """
    Deterministic synthetic run shape used by lightweight benchmark/perf-smoke harnesses.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_staged_runner_perf_smoke.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """

    timeframe: str
    target_bars: int
    warmup_bars: int
    total_candles_bars: int
    indicator_windows: tuple[int, ...]
    tp_values: tuple[float, ...]
    sl_values: tuple[float, ...]
    top_k: int
    preselect: int
    top_trades_n: int
    expected_stage_a_variants_total: int
    expected_stage_b_variants_total: int

    def __post_init__(self) -> None:
        """
        Validate one synthetic benchmark run spec.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Synthetic run specs remain lightweight and deterministic so contributors can execute
            them in normal quality-gate workflows.
        Raises:
            ValueError: If one numeric field is non-positive or the total-candle count drifts.
        Side Effects:
            None.
        """
        if not self.timeframe.strip():
            raise ValueError(
                "BacktestRuntimeBenchmarkSyntheticRunSpecV2.timeframe must be non-empty"
            )
        if self.target_bars <= 0:
            raise ValueError(
                "BacktestRuntimeBenchmarkSyntheticRunSpecV2.target_bars must be > 0"
            )
        if self.warmup_bars <= 0:
            raise ValueError(
                "BacktestRuntimeBenchmarkSyntheticRunSpecV2.warmup_bars must be > 0"
            )
        if self.total_candles_bars != self.target_bars + self.warmup_bars:
            raise ValueError(
                "BacktestRuntimeBenchmarkSyntheticRunSpecV2.total_candles_bars must equal "
                "target_bars + warmup_bars"
            )
        if len(self.indicator_windows) == 0:
            raise ValueError(
                "BacktestRuntimeBenchmarkSyntheticRunSpecV2.indicator_windows must be non-empty"
            )
        if len(self.tp_values) == 0:
            raise ValueError(
                "BacktestRuntimeBenchmarkSyntheticRunSpecV2.tp_values must be non-empty"
            )
        if len(self.sl_values) == 0:
            raise ValueError(
                "BacktestRuntimeBenchmarkSyntheticRunSpecV2.sl_values must be non-empty"
            )
        if self.top_k <= 0:
            raise ValueError(
                "BacktestRuntimeBenchmarkSyntheticRunSpecV2.top_k must be > 0"
            )
        if self.preselect <= 0:
            raise ValueError(
                "BacktestRuntimeBenchmarkSyntheticRunSpecV2.preselect must be > 0"
            )
        if self.top_trades_n <= 0:
            raise ValueError(
                "BacktestRuntimeBenchmarkSyntheticRunSpecV2.top_trades_n must be > 0"
            )
        if self.expected_stage_a_variants_total <= 0:
            raise ValueError(
                "BacktestRuntimeBenchmarkSyntheticRunSpecV2.expected_stage_a_variants_total "
                "must be > 0"
            )
        if self.expected_stage_b_variants_total <= 0:
            raise ValueError(
                "BacktestRuntimeBenchmarkSyntheticRunSpecV2.expected_stage_b_variants_total "
                "must be > 0"
            )


@dataclass(frozen=True, slots=True)
class BacktestRuntimeBenchmarkSliceV2:
    """
    One deterministic benchmark-corpus slice for exact, hybrid, or plugin rollout evaluation.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - tests/perf_smoke/contexts/backtest/test_backtest_staged_runner_perf_smoke.py
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    slice_id: BenchmarkCorpusSliceIdLiteralV2
    title: str
    execution_profile_mode: ExecutionProfileModeLiteralV2
    candidate_execution_profile_mode: ExecutionProfileModeLiteralV2 | None
    rollout_scope: BenchmarkCorpusRolloutScopeLiteralV2
    stage_focus: tuple[BenchmarkCorpusStageLiteralV2, ...]
    evaluation_focus: tuple[str, ...]
    r0_scenario_ids: tuple[str, ...]
    r5_stage_b_case_ids: tuple[str, ...]
    synthetic_run_spec: BacktestRuntimeBenchmarkSyntheticRunSpecV2 | None
    notes: str

    def __post_init__(self) -> None:
        """
        Validate one benchmark-corpus slice contract.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Each slice encodes one stable evidence surface for later rollout work and therefore
            keeps explicit rollout scope, stage focus, and evidence references.
        Raises:
            ValueError: If the slice shape or rollout semantics are inconsistent.
        Side Effects:
            None.
        """
        if not self.title.strip():
            raise ValueError("BacktestRuntimeBenchmarkSliceV2.title must be non-empty")
        if len(self.stage_focus) == 0:
            raise ValueError("BacktestRuntimeBenchmarkSliceV2.stage_focus must be non-empty")
        if len(self.stage_focus) != len(set(self.stage_focus)):
            raise ValueError(
                "BacktestRuntimeBenchmarkSliceV2.stage_focus must not contain duplicates"
            )
        if len(self.evaluation_focus) == 0:
            raise ValueError(
                "BacktestRuntimeBenchmarkSliceV2.evaluation_focus must be non-empty"
            )
        if len(self.evaluation_focus) != len(set(self.evaluation_focus)):
            raise ValueError(
                "BacktestRuntimeBenchmarkSliceV2.evaluation_focus must not contain duplicates"
            )
        if len(self.r0_scenario_ids) != len(set(self.r0_scenario_ids)):
            raise ValueError(
                "BacktestRuntimeBenchmarkSliceV2.r0_scenario_ids must not contain duplicates"
            )
        if len(self.r5_stage_b_case_ids) != len(set(self.r5_stage_b_case_ids)):
            raise ValueError(
                "BacktestRuntimeBenchmarkSliceV2.r5_stage_b_case_ids must not contain duplicates"
            )
        if (
            len(self.r0_scenario_ids) == 0
            and len(self.r5_stage_b_case_ids) == 0
            and self.synthetic_run_spec is None
        ):
            raise ValueError(
                "BacktestRuntimeBenchmarkSliceV2 must reference at least one evidence source"
            )
        if not self.notes.strip():
            raise ValueError("BacktestRuntimeBenchmarkSliceV2.notes must be non-empty")
        if self.rollout_scope == "exact_only":
            if self.candidate_execution_profile_mode is not None:
                raise ValueError(
                    "BacktestRuntimeBenchmarkSliceV2.exact_only slices must not declare "
                    "candidate_execution_profile_mode"
                )
            return
        if self.candidate_execution_profile_mode is None:
            raise ValueError(
                "BacktestRuntimeBenchmarkSliceV2 non-exact slices must declare "
                "candidate_execution_profile_mode"
            )
        if (
            self.rollout_scope == "plugin_rollout"
            and self.candidate_execution_profile_mode != "hybrid_family"
        ):
            raise ValueError(
                "BacktestRuntimeBenchmarkSliceV2.plugin_rollout slices must target "
                "'hybrid_family'"
            )


@dataclass(frozen=True, slots=True)
class BacktestRuntimeAccelerationBenchmarkCorpusV2:
    """
    Versioned benchmark corpus for runtime acceleration exact/hybrid/plugin follow-up work.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    schema_version: int
    fixture_contract: str
    milestone_id: str
    epic_id: str
    status: str
    reference_docs: tuple[str, ...]
    rollout_gates: BacktestRuntimeBenchmarkRolloutGatesV2
    source_fixtures: BacktestRuntimeBenchmarkSourceFixturesV2
    slice_order: tuple[BenchmarkCorpusSliceIdLiteralV2, ...]
    slices: tuple[BacktestRuntimeBenchmarkSliceV2, ...]

    def __post_init__(self) -> None:
        """
        Validate the top-level benchmark-corpus contract.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Top-level ordering is part of the committed benchmark protocol and therefore remains
            byte-stable and reviewable.
        Raises:
            ValueError: If metadata literals drift or slice ordering becomes inconsistent.
        Side Effects:
            None.
        """
        if self.schema_version != BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_SCHEMA_VERSION_V2:
            raise ValueError(
                "BacktestRuntimeAccelerationBenchmarkCorpusV2.schema_version must be "
                f"{BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_SCHEMA_VERSION_V2}"
            )
        if (
            self.fixture_contract
            != BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_KIND_V2
        ):
            raise ValueError(
                "BacktestRuntimeAccelerationBenchmarkCorpusV2.fixture_contract must be "
                f"{BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_KIND_V2!r}"
            )
        if self.milestone_id != BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_MILESTONE_ID_V2:
            raise ValueError(
                "BacktestRuntimeAccelerationBenchmarkCorpusV2.milestone_id must be "
                f"{BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_MILESTONE_ID_V2!r}"
            )
        if self.epic_id != BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_EPIC_ID_V2:
            raise ValueError(
                "BacktestRuntimeAccelerationBenchmarkCorpusV2.epic_id must be "
                f"{BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_EPIC_ID_V2!r}"
            )
        if not self.status.strip():
            raise ValueError(
                "BacktestRuntimeAccelerationBenchmarkCorpusV2.status must be non-empty"
            )
        if len(self.reference_docs) == 0:
            raise ValueError(
                "BacktestRuntimeAccelerationBenchmarkCorpusV2.reference_docs must be non-empty"
            )
        if self.rollout_gates is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "BacktestRuntimeAccelerationBenchmarkCorpusV2.rollout_gates is required"
            )
        if len(self.slice_order) == 0:
            raise ValueError(
                "BacktestRuntimeAccelerationBenchmarkCorpusV2.slice_order must be non-empty"
            )
        if len(self.slice_order) != len(set(self.slice_order)):
            raise ValueError(
                "BacktestRuntimeAccelerationBenchmarkCorpusV2.slice_order must not contain "
                "duplicates"
            )
        authored_order = tuple(slice_item.slice_id for slice_item in self.slices)
        if authored_order != self.slice_order:
            raise ValueError(
                "BacktestRuntimeAccelerationBenchmarkCorpusV2.slices must follow slice_order"
            )

    def slice_for_id(
        self,
        *,
        slice_id: BenchmarkCorpusSliceIdLiteralV2,
    ) -> BacktestRuntimeBenchmarkSliceV2:
        """
        Return one benchmark slice by its stable identifier.

        Args:
            slice_id: Stable benchmark-corpus slice identifier.
        Returns:
            BacktestRuntimeBenchmarkSliceV2: Matching committed slice object.
        Assumptions:
            The caller uses one of the committed `slice_order` identifiers.
        Raises:
            KeyError: If the slice id is not present in this committed corpus.
        Side Effects:
            None.
        """
        for slice_item in self.slices:
            if slice_item.slice_id == slice_id:
                return slice_item
        raise KeyError(f"benchmark slice not found: {slice_id!r}")


def read_backtest_runtime_acceleration_benchmark_corpus_payload_v2(
    *,
    path: Path,
) -> dict[str, object]:
    """
    Read one benchmark-corpus JSON payload from disk.

    Args:
        path: Absolute or repository-relative fixture path.
    Returns:
        dict[str, object]: Raw JSON object preserving authored key ordering.
    Assumptions:
        The committed corpus is a UTF-8 JSON object checked into version control.
    Raises:
        ValueError: If the JSON root is not an object.
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the payload is invalid JSON.
    Side Effects:
        Reads one repository file from disk.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark corpus payload root must be an object")
    return payload


def serialize_backtest_runtime_acceleration_benchmark_corpus_payload_v2(
    *,
    payload: dict[str, object],
) -> bytes:
    """
    Serialize one raw benchmark-corpus payload with canonical repository formatting.

    Args:
        payload: Raw JSON-compatible object root produced by `json.loads`.
    Returns:
        bytes: Canonical UTF-8 bytes with stable indentation and trailing newline.
    Assumptions:
        Authored key ordering is intentional and must stay reviewable.
    Raises:
        TypeError: If the payload contains non-JSON-serializable values.
    Side Effects:
        None.
    """
    return (json.dumps(payload, ensure_ascii=True, indent=2) + "\n").encode("utf-8")


def validate_backtest_runtime_acceleration_benchmark_corpus_payload_v2(
    *,
    payload: dict[str, object],
) -> None:
    """
    Validate one raw benchmark-corpus payload against the typed committed contract.

    Args:
        payload: Raw JSON payload loaded from the committed corpus fixture.
    Returns:
        None.
    Assumptions:
        Successful parsing is sufficient validation because all invariants live in typed objects.
    Raises:
        ValueError: If the payload violates typed benchmark-corpus invariants.
    Side Effects:
        None.
    """
    _parse_backtest_runtime_acceleration_benchmark_corpus_payload_v2(payload=payload)


def load_backtest_runtime_acceleration_benchmark_corpus_v2(
    *,
    path: Path,
) -> BacktestRuntimeAccelerationBenchmarkCorpusV2:
    """
    Load the committed benchmark corpus into typed immutable contracts.

    Args:
        path: Absolute or repository-relative fixture path.
    Returns:
        BacktestRuntimeAccelerationBenchmarkCorpusV2: Parsed benchmark corpus.
    Assumptions:
        The committed corpus stays lightweight and is safe to read in unit/perf-smoke tests.
    Raises:
        ValueError: If the payload violates the typed benchmark-corpus contract.
        OSError: If the fixture file cannot be read.
        json.JSONDecodeError: If the fixture file is invalid JSON.
    Side Effects:
        Reads one repository file from disk.
    """
    payload = read_backtest_runtime_acceleration_benchmark_corpus_payload_v2(path=path)
    return _parse_backtest_runtime_acceleration_benchmark_corpus_payload_v2(payload=payload)


def _parse_backtest_runtime_acceleration_benchmark_corpus_payload_v2(
    *,
    payload: dict[str, object],
) -> BacktestRuntimeAccelerationBenchmarkCorpusV2:
    """
    Parse the raw benchmark-corpus payload into typed immutable contracts.

    Args:
        payload: Raw JSON payload loaded from disk.
    Returns:
        BacktestRuntimeAccelerationBenchmarkCorpusV2: Parsed top-level benchmark corpus object.
    Assumptions:
        Raw payload objects preserve authored ordering from the committed JSON fixture.
    Raises:
        ValueError: If required keys are missing or carry unsupported values.
    Side Effects:
        None.
    """
    reference_docs = _require_string_tuple(payload=payload, key="reference_docs")
    rollout_gates_map = _require_mapping(payload=payload, key="rollout_gates")
    source_fixtures_map = _require_mapping(payload=payload, key="source_fixtures")
    slices_payload = _require_mapping_sequence(payload=payload, key="slices")
    slice_order = cast(
        tuple[BenchmarkCorpusSliceIdLiteralV2, ...],
        tuple(
            _parse_benchmark_slice_id_v2(value=value)
            for value in _require_string_tuple(payload=payload, key="slice_order")
        ),
    )
    slices = tuple(
        _parse_backtest_runtime_benchmark_slice_v2(raw_slice=raw_slice)
        for raw_slice in slices_payload
    )
    return BacktestRuntimeAccelerationBenchmarkCorpusV2(
        schema_version=_require_int(payload=payload, key="schema_version"),
        fixture_contract=_require_str(payload=payload, key="fixture_contract"),
        milestone_id=_require_str(payload=payload, key="milestone_id"),
        epic_id=_require_str(payload=payload, key="epic_id"),
        status=_require_str(payload=payload, key="status"),
        reference_docs=reference_docs,
        rollout_gates=_parse_backtest_runtime_benchmark_rollout_gates_v2(
            raw_rollout_gates=rollout_gates_map
        ),
        source_fixtures=BacktestRuntimeBenchmarkSourceFixturesV2(
            r0_benchmark_scenarios=_require_str(
                payload=source_fixtures_map,
                key="r0_benchmark_scenarios",
            ),
            r5_stage_b_manifest=_require_str(
                payload=source_fixtures_map,
                key="r5_stage_b_manifest",
            ),
            stage_b_golden_fixture=_require_str(
                payload=source_fixtures_map,
                key="stage_b_golden_fixture",
            ),
        ),
        slice_order=slice_order,
        slices=slices,
    )


def _parse_backtest_runtime_benchmark_slice_v2(
    *,
    raw_slice: dict[str, object],
) -> BacktestRuntimeBenchmarkSliceV2:
    """
    Parse one raw benchmark slice object into a typed immutable contract.

    Args:
        raw_slice: Raw JSON object for one slice entry.
    Returns:
        BacktestRuntimeBenchmarkSliceV2: Parsed typed slice.
    Assumptions:
        The caller already validated that `raw_slice` is a JSON object mapping.
    Raises:
        ValueError: If the slice payload violates typed benchmark-corpus invariants.
    Side Effects:
        None.
    """
    candidate_execution_profile_mode = _require_optional_str(
        payload=raw_slice,
        key="candidate_execution_profile_mode",
    )
    synthetic_run_spec_map = _require_optional_mapping(
        payload=raw_slice,
        key="synthetic_run_spec",
    )
    return BacktestRuntimeBenchmarkSliceV2(
        slice_id=_parse_benchmark_slice_id_v2(
            value=_require_str(payload=raw_slice, key="slice_id")
        ),
        title=_require_str(payload=raw_slice, key="title"),
        execution_profile_mode=validate_execution_profile_mode_v2(
            value=_require_str(payload=raw_slice, key="execution_profile_mode")
        ),
        candidate_execution_profile_mode=(
            None
            if candidate_execution_profile_mode is None
            else validate_execution_profile_mode_v2(
                value=candidate_execution_profile_mode
            )
        ),
        rollout_scope=_parse_benchmark_rollout_scope_v2(
            value=_require_str(payload=raw_slice, key="rollout_scope")
        ),
        stage_focus=tuple(
            _parse_benchmark_stage_id_v2(value=value)
            for value in _require_string_tuple(payload=raw_slice, key="stage_focus")
        ),
        evaluation_focus=_require_string_tuple(payload=raw_slice, key="evaluation_focus"),
        r0_scenario_ids=_require_string_tuple(payload=raw_slice, key="r0_scenario_ids"),
        r5_stage_b_case_ids=_require_string_tuple(
            payload=raw_slice,
            key="r5_stage_b_case_ids",
        ),
        synthetic_run_spec=(
            None
            if synthetic_run_spec_map is None
            else _parse_backtest_runtime_benchmark_synthetic_run_spec_v2(
                raw_spec=synthetic_run_spec_map
            )
        ),
        notes=_require_str(payload=raw_slice, key="notes"),
    )


def _parse_backtest_runtime_benchmark_rollout_gates_v2(
    *,
    raw_rollout_gates: dict[str, object],
) -> BacktestRuntimeBenchmarkRolloutGatesV2:
    """
    Parse the top-level rollout-gate bundle from raw JSON payload.

    Args:
        raw_rollout_gates: Raw JSON mapping carrying named rollout gates.
    Returns:
        BacktestRuntimeBenchmarkRolloutGatesV2: Parsed conservative rollout-gate bundle.
    Assumptions:
        Gate keys are fixed literals that mirror the roadmap milestone D evidence surface.
    Raises:
        ValueError: If one required gate is missing or malformed.
    Side Effects:
        None.
    """
    return BacktestRuntimeBenchmarkRolloutGatesV2(
        top_1_recall=_parse_backtest_runtime_benchmark_rollout_gate_v2(
            gate_id="top_1_recall",
            raw_gate=_require_mapping(payload=raw_rollout_gates, key="top_1_recall"),
        ),
        top_10_overlap=_parse_backtest_runtime_benchmark_rollout_gate_v2(
            gate_id="top_10_overlap",
            raw_gate=_require_mapping(payload=raw_rollout_gates, key="top_10_overlap"),
        ),
        low_activity=_parse_backtest_runtime_benchmark_rollout_gate_v2(
            gate_id="low_activity",
            raw_gate=_require_mapping(payload=raw_rollout_gates, key="low_activity"),
        ),
        high_correlation=_parse_backtest_runtime_benchmark_rollout_gate_v2(
            gate_id="high_correlation",
            raw_gate=_require_mapping(payload=raw_rollout_gates, key="high_correlation"),
        ),
        small_grid_overhead=_parse_backtest_runtime_benchmark_rollout_gate_v2(
            gate_id="small_grid_overhead",
            raw_gate=_require_mapping(payload=raw_rollout_gates, key="small_grid_overhead"),
        ),
        memory_footprint=_parse_backtest_runtime_benchmark_rollout_gate_v2(
            gate_id="memory_footprint",
            raw_gate=_require_mapping(payload=raw_rollout_gates, key="memory_footprint"),
        ),
    )


def _parse_backtest_runtime_benchmark_rollout_gate_v2(
    *,
    gate_id: BenchmarkCorpusRolloutGateIdLiteralV2,
    raw_gate: dict[str, object],
) -> BacktestRuntimeBenchmarkRolloutGateV2:
    """
    Parse one raw rollout gate into a typed conservative threshold contract.

    Args:
        gate_id: Stable rollout-gate identifier from the committed corpus.
        raw_gate: Raw JSON object for one rollout gate.
    Returns:
        BacktestRuntimeBenchmarkRolloutGateV2: Parsed typed rollout-gate contract.
    Assumptions:
        Gates may reference one benchmark slice or represent aggregate corpus-wide evidence.
    Raises:
        ValueError: If the gate payload is malformed.
    Side Effects:
        None.
    """
    _parse_benchmark_rollout_gate_id_v2(value=gate_id)
    raw_slice_id = _require_optional_str(payload=raw_gate, key="slice_id")
    return BacktestRuntimeBenchmarkRolloutGateV2(
        metric=_require_str(payload=raw_gate, key="metric"),
        slice_id=(
            None
            if raw_slice_id is None
            else _parse_benchmark_slice_id_v2(value=raw_slice_id)
        ),
        min_ratio=_require_optional_float(payload=raw_gate, key="min_ratio"),
        max_ratio=_require_optional_float(payload=raw_gate, key="max_ratio"),
        min_distinct_count=_require_optional_int(
            payload=raw_gate,
            key="min_distinct_count",
        ),
        notes=_require_str(payload=raw_gate, key="notes"),
    )


def _parse_backtest_runtime_benchmark_synthetic_run_spec_v2(
    *,
    raw_spec: dict[str, object],
) -> BacktestRuntimeBenchmarkSyntheticRunSpecV2:
    """
    Parse one raw synthetic benchmark run spec into a typed immutable contract.

    Args:
        raw_spec: Raw JSON object for one synthetic run spec.
    Returns:
        BacktestRuntimeBenchmarkSyntheticRunSpecV2: Parsed typed synthetic run spec.
    Assumptions:
        Synthetic run specs use only scalar and sequence values committed in JSON.
    Raises:
        ValueError: If one spec field is missing or invalid.
    Side Effects:
        None.
    """
    return BacktestRuntimeBenchmarkSyntheticRunSpecV2(
        timeframe=_require_str(payload=raw_spec, key="timeframe"),
        target_bars=_require_int(payload=raw_spec, key="target_bars"),
        warmup_bars=_require_int(payload=raw_spec, key="warmup_bars"),
        total_candles_bars=_require_int(payload=raw_spec, key="total_candles_bars"),
        indicator_windows=tuple(
            _coerce_int(value=value, field_name="indicator_windows")
            for value in _require_sequence(payload=raw_spec, key="indicator_windows")
        ),
        tp_values=tuple(
            _coerce_float(value=value, field_name="tp_values")
            for value in _require_sequence(payload=raw_spec, key="tp_values")
        ),
        sl_values=tuple(
            _coerce_float(value=value, field_name="sl_values")
            for value in _require_sequence(payload=raw_spec, key="sl_values")
        ),
        top_k=_require_int(payload=raw_spec, key="top_k"),
        preselect=_require_int(payload=raw_spec, key="preselect"),
        top_trades_n=_require_int(payload=raw_spec, key="top_trades_n"),
        expected_stage_a_variants_total=_require_int(
            payload=raw_spec,
            key="expected_stage_a_variants_total",
        ),
        expected_stage_b_variants_total=_require_int(
            payload=raw_spec,
            key="expected_stage_b_variants_total",
        ),
    )


def _parse_benchmark_slice_id_v2(
    *,
    value: str,
) -> BenchmarkCorpusSliceIdLiteralV2:
    """
    Validate one benchmark slice identifier against the committed corpus contract.

    Args:
        value: Raw benchmark slice identifier.
    Returns:
        BenchmarkCorpusSliceIdLiteralV2: Validated benchmark slice identifier.
    Assumptions:
        Slice ids are lowercase snake_case literals from the committed A3 benchmark corpus.
    Raises:
        ValueError: If the literal is blank or unsupported.
    Side Effects:
        None.
    """
    normalized_value = value.strip().lower()
    if normalized_value not in _ALLOWED_BENCHMARK_CORPUS_SLICE_IDS_V2:
        raise ValueError(
            "benchmark slice_id must be one of "
            f"{_ALLOWED_BENCHMARK_CORPUS_SLICE_IDS_V2}, got {value!r}"
        )
    return cast(BenchmarkCorpusSliceIdLiteralV2, normalized_value)


def _parse_benchmark_rollout_scope_v2(
    *,
    value: str,
) -> BenchmarkCorpusRolloutScopeLiteralV2:
    """
    Validate one benchmark rollout scope literal against the committed corpus contract.

    Args:
        value: Raw benchmark rollout scope literal.
    Returns:
        BenchmarkCorpusRolloutScopeLiteralV2: Validated rollout scope literal.
    Assumptions:
        Rollout scopes are lowercase snake_case literals from the committed A3 benchmark corpus.
    Raises:
        ValueError: If the literal is blank or unsupported.
    Side Effects:
        None.
    """
    normalized_value = value.strip().lower()
    if normalized_value not in _ALLOWED_BENCHMARK_CORPUS_ROLLOUT_SCOPES_V2:
        raise ValueError(
            "benchmark rollout_scope must be one of "
            f"{_ALLOWED_BENCHMARK_CORPUS_ROLLOUT_SCOPES_V2}, got {value!r}"
        )
    return cast(BenchmarkCorpusRolloutScopeLiteralV2, normalized_value)


def _parse_benchmark_stage_id_v2(
    *,
    value: str,
) -> BenchmarkCorpusStageLiteralV2:
    """
    Validate one benchmark stage identifier against the committed corpus contract.

    Args:
        value: Raw benchmark stage identifier.
    Returns:
        BenchmarkCorpusStageLiteralV2: Validated stage identifier.
    Assumptions:
        Stage ids mirror the A2 persisted-run vocabulary (`stage_a`, `stage_b`, `finalizing`).
    Raises:
        ValueError: If the literal is blank or unsupported.
    Side Effects:
        None.
    """
    normalized_value = value.strip().lower()
    if normalized_value not in _ALLOWED_BENCHMARK_CORPUS_STAGE_IDS_V2:
        raise ValueError(
            "benchmark stage id must be one of "
            f"{_ALLOWED_BENCHMARK_CORPUS_STAGE_IDS_V2}, got {value!r}"
        )
    return cast(BenchmarkCorpusStageLiteralV2, normalized_value)


def _parse_benchmark_rollout_gate_id_v2(
    *,
    value: str,
) -> BenchmarkCorpusRolloutGateIdLiteralV2:
    """
    Validate one rollout-gate identifier against the committed corpus contract.

    Args:
        value: Raw rollout-gate identifier.
    Returns:
        BenchmarkCorpusRolloutGateIdLiteralV2: Validated rollout-gate identifier.
    Assumptions:
        Rollout-gate ids are lowercase snake_case literals from the committed D2+D3 corpus.
    Raises:
        ValueError: If the literal is blank or unsupported.
    Side Effects:
        None.
    """
    normalized_value = value.strip().lower()
    if normalized_value not in _ALLOWED_BENCHMARK_CORPUS_ROLLOUT_GATE_IDS_V2:
        raise ValueError(
            "benchmark rollout gate id must be one of "
            f"{_ALLOWED_BENCHMARK_CORPUS_ROLLOUT_GATE_IDS_V2}, got {value!r}"
        )
    return cast(BenchmarkCorpusRolloutGateIdLiteralV2, normalized_value)


def _require_mapping(
    *,
    payload: dict[str, object],
    key: str,
) -> dict[str, object]:
    """
    Require one mapping child from a raw JSON object payload.

    Args:
        payload: Raw JSON object payload.
        key: Required mapping key.
    Returns:
        dict[str, object]: Child mapping value.
    Assumptions:
        JSON objects are deserialized as plain Python dictionaries.
    Raises:
        ValueError: If the key is missing or the value is not an object mapping.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object mapping")
    return cast(dict[str, object], value)


def _require_optional_mapping(
    *,
    payload: dict[str, object],
    key: str,
) -> dict[str, object] | None:
    """
    Read one optional mapping child from a raw JSON object payload.

    Args:
        payload: Raw JSON object payload.
        key: Optional mapping key.
    Returns:
        dict[str, object] | None: Child mapping value or `None` when omitted/null.
    Assumptions:
        Optional nested mappings are encoded either as JSON object or explicit `null`.
    Raises:
        ValueError: If the key exists with a non-object non-null value.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object mapping or null")
    return cast(dict[str, object], value)


def _require_mapping_sequence(
    *,
    payload: dict[str, object],
    key: str,
) -> tuple[dict[str, object], ...]:
    """
    Require one sequence of mapping objects from a raw JSON payload.

    Args:
        payload: Raw JSON object payload.
        key: Required sequence key.
    Returns:
        tuple[dict[str, object], ...]: Ordered child object sequence.
    Assumptions:
        Benchmark fixture arrays preserve authored ordering and contain object items only.
    Raises:
        ValueError: If the key is missing, not a JSON array, or contains non-object items.
    Side Effects:
        None.
    """
    values = _require_sequence(payload=payload, key=key)
    normalized: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"{key} items must be object mappings")
        normalized.append(cast(dict[str, object], value))
    return tuple(normalized)


def _require_sequence(
    *,
    payload: dict[str, object],
    key: str,
) -> tuple[object, ...]:
    """
    Require one array value from a raw JSON object payload.

    Args:
        payload: Raw JSON object payload.
        key: Required sequence key.
    Returns:
        tuple[object, ...]: Ordered JSON array items.
    Assumptions:
        Scalars like strings are not accepted as substitute sequence values.
    Raises:
        ValueError: If the key is missing or the value is not a JSON array.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return tuple(value)


def _require_string_tuple(
    *,
    payload: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    """
    Require one array of non-empty strings from a raw JSON object payload.

    Args:
        payload: Raw JSON object payload.
        key: Required array key.
    Returns:
        tuple[str, ...]: Ordered non-empty string literals.
    Assumptions:
        String ordering is part of the committed benchmark contract.
    Raises:
        ValueError: If one item is blank or not a string.
    Side Effects:
        None.
    """
    values = _require_sequence(payload=payload, key=key)
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} items must be non-empty strings")
        normalized.append(value)
    return tuple(normalized)


def _require_str(
    *,
    payload: dict[str, object],
    key: str,
) -> str:
    """
    Require one non-empty string value from a raw JSON object payload.

    Args:
        payload: Raw JSON object payload.
        key: Required scalar key.
    Returns:
        str: Non-empty string value.
    Assumptions:
        String literals are authored explicitly in the committed JSON fixture.
    Raises:
        ValueError: If the key is missing or the value is blank/non-string.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_optional_str(
    *,
    payload: dict[str, object],
    key: str,
) -> str | None:
    """
    Read one optional non-empty string value from a raw JSON object payload.

    Args:
        payload: Raw JSON object payload.
        key: Optional scalar key.
    Returns:
        str | None: Non-empty string value or `None` when omitted/null.
    Assumptions:
        Optional literals are encoded either as JSON string or explicit `null`.
    Raises:
        ValueError: If the key exists with a blank or non-string non-null value.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string or null")
    return value


def _require_int(
    *,
    payload: dict[str, object],
    key: str,
) -> int:
    """
    Require one integer scalar value from a raw JSON object payload.

    Args:
        payload: Raw JSON object payload.
        key: Required scalar key.
    Returns:
        int: Integer scalar value.
    Assumptions:
        Boolean values are rejected even though `bool` subclasses `int`.
    Raises:
        ValueError: If the key is missing or the value is not an integer.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _require_optional_int(
    *,
    payload: dict[str, object],
    key: str,
) -> int | None:
    """
    Read one optional integer scalar value from a raw JSON object payload.

    Args:
        payload: Raw JSON object payload.
        key: Optional scalar key.
    Returns:
        int | None: Integer scalar value or `None` when omitted/null.
    Assumptions:
        Boolean values are rejected even though `bool` subclasses `int`.
    Raises:
        ValueError: If the key exists with a non-integer non-null value.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer or null")
    return int(value)


def _require_optional_float(
    *,
    payload: dict[str, object],
    key: str,
) -> float | None:
    """
    Read one optional numeric scalar value from a raw JSON object payload.

    Args:
        payload: Raw JSON object payload.
        key: Optional scalar key.
    Returns:
        float | None: Numeric scalar value or `None` when omitted/null.
    Assumptions:
        JSON numbers may be authored as integer or float literals.
    Raises:
        ValueError: If the key exists with a non-numeric non-null value.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number or null")
    return float(value)


def _coerce_int(
    *,
    value: object,
    field_name: str,
) -> int:
    """
    Coerce one JSON scalar into an integer while rejecting booleans and non-integers.

    Args:
        value: Raw JSON scalar value.
        field_name: Human-readable field name for error messages.
    Returns:
        int: Coerced integer value.
    Assumptions:
        Sequence items were already extracted from JSON arrays and only need type validation.
    Raises:
        ValueError: If the value is not an integer scalar.
    Side Effects:
        None.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} items must be integers")
    return int(value)


def _coerce_float(
    *,
    value: object,
    field_name: str,
) -> float:
    """
    Coerce one JSON numeric scalar into a float while rejecting booleans.

    Args:
        value: Raw JSON scalar value.
        field_name: Human-readable field name for error messages.
    Returns:
        float: Coerced floating-point value.
    Assumptions:
        JSON numeric arrays may mix ints and floats but remain deterministic.
    Raises:
        ValueError: If the value is not a numeric scalar.
    Side Effects:
        None.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} items must be numeric")
    return float(value)


__all__ = [
    "BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_EPIC_ID_V2",
    "BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_KIND_V2",
    "BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_MILESTONE_ID_V2",
    "BACKTEST_RUNTIME_ACCELERATION_BENCHMARK_CORPUS_SCHEMA_VERSION_V2",
    "BenchmarkCorpusRolloutGateIdLiteralV2",
    "BenchmarkCorpusRolloutScopeLiteralV2",
    "BenchmarkCorpusSliceIdLiteralV2",
    "BenchmarkCorpusStageLiteralV2",
    "BacktestRuntimeAccelerationBenchmarkCorpusV2",
    "BacktestRuntimeBenchmarkRolloutGateV2",
    "BacktestRuntimeBenchmarkRolloutGatesV2",
    "BacktestRuntimeBenchmarkSliceV2",
    "BacktestRuntimeBenchmarkSourceFixturesV2",
    "BacktestRuntimeBenchmarkSyntheticRunSpecV2",
    "load_backtest_runtime_acceleration_benchmark_corpus_v2",
    "read_backtest_runtime_acceleration_benchmark_corpus_payload_v2",
    "serialize_backtest_runtime_acceleration_benchmark_corpus_payload_v2",
    "validate_backtest_runtime_acceleration_benchmark_corpus_payload_v2",
]
