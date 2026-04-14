"""Typed benchmark-corpus helpers for the notebook-parity benchmark authority.

Docs:
  - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
  - docs/architecture/backtest/backtest-v2-benchmarks.md
  - docs/architecture/backtest/backtest-engine-vnext.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

BACKTEST_NOTEBOOK_PARITY_BENCHMARK_CORPUS_SCHEMA_VERSION_V2 = 1
BACKTEST_NOTEBOOK_PARITY_BENCHMARK_CORPUS_KIND_V2 = (
    "backtest_notebook_parity_benchmark_corpus_v1"
)
BACKTEST_NOTEBOOK_PARITY_BENCHMARK_CORPUS_MILESTONE_ID_V2 = "A1"

type NotebookParityBenchmarkClassLiteralV2 = Literal["NR2", "RG-TTR", "RG-ALT"]
type NotebookParityComparisonModeLiteralV2 = Literal[
    "notebook_parity",
    "functional_baseline",
]
type NotebookParityMeasurementFieldLiteralV2 = Literal[
    "wall_clock_seconds",
    "cpu_time_seconds",
    "peak_rss_bytes",
    "numba_threads_used",
    "max_python_processes_seen",
    "stage_b_execution_mode",
    "stage_b_process_fallback_threshold",
    "exact_replay_count",
]
type NotebookParityMeasurementSourceLiteralV2 = Literal["backend", "notebook"]
type NotebookParityAuthorityKindLiteralV2 = Literal[
    "synthetic_contract_validation",
    "live_host_measurement",
]
type NotebookParityLiveHostCaptureStatusLiteralV2 = Literal["missing", "captured"]
type NotebookParityReferenceSourceKindLiteralV2 = Literal["backend", "notebook", "gate"]
type NotebookParityRuntimeSurfaceLiteralV2 = Literal["sync", "worker", "notebook"]
type NotebookParityStageBExecutionModeLiteralV2 = Literal[
    "bypassed_no_risk",
    "in_process",
    "process_pool",
]
type NotebookParityStageBProcessFallbackThresholdLiteralV2 = Literal[
    "none",
    "stage_b_variants_total",
]

_ALLOWED_NOTEBOOK_PARITY_BENCHMARK_CLASSES_V2: tuple[
    NotebookParityBenchmarkClassLiteralV2, ...
] = ("NR2", "RG-TTR", "RG-ALT")
_ALLOWED_NOTEBOOK_PARITY_COMPARISON_MODES_V2: tuple[
    NotebookParityComparisonModeLiteralV2, ...
] = (
    "notebook_parity",
    "functional_baseline",
)
_ALLOWED_NOTEBOOK_PARITY_MEASUREMENT_FIELDS_V2: tuple[
    NotebookParityMeasurementFieldLiteralV2, ...
] = (
    "wall_clock_seconds",
    "cpu_time_seconds",
    "peak_rss_bytes",
    "numba_threads_used",
    "max_python_processes_seen",
    "stage_b_execution_mode",
    "stage_b_process_fallback_threshold",
    "exact_replay_count",
)
_ALLOWED_NOTEBOOK_PARITY_MEASUREMENT_SOURCES_V2: tuple[
    NotebookParityMeasurementSourceLiteralV2, ...
] = ("backend", "notebook")
_ALLOWED_NOTEBOOK_PARITY_AUTHORITY_KINDS_V2: tuple[
    NotebookParityAuthorityKindLiteralV2, ...
] = (
    "synthetic_contract_validation",
    "live_host_measurement",
)
_ALLOWED_NOTEBOOK_PARITY_LIVE_HOST_CAPTURE_STATUSES_V2: tuple[
    NotebookParityLiveHostCaptureStatusLiteralV2, ...
] = ("missing", "captured")
_ALLOWED_NOTEBOOK_PARITY_REFERENCE_SOURCE_KINDS_V2: tuple[
    NotebookParityReferenceSourceKindLiteralV2, ...
] = ("backend", "notebook", "gate")
_ALLOWED_NOTEBOOK_PARITY_RUNTIME_SURFACES_V2: tuple[
    NotebookParityRuntimeSurfaceLiteralV2, ...
] = ("sync", "worker", "notebook")
_ALLOWED_NOTEBOOK_PARITY_STAGE_B_EXECUTION_MODES_V2: tuple[
    NotebookParityStageBExecutionModeLiteralV2, ...
] = (
    "bypassed_no_risk",
    "in_process",
    "process_pool",
)
_ALLOWED_NOTEBOOK_PARITY_STAGE_B_PROCESS_FALLBACK_THRESHOLDS_V2: tuple[
    NotebookParityStageBProcessFallbackThresholdLiteralV2, ...
] = (
    "none",
    "stage_b_variants_total",
)


@dataclass(frozen=True, slots=True)
class BacktestNotebookParityMeasurementContractV2:
    """
    Canonical measurement-field contract for notebook-parity perf-smoke evidence.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - src/trading/contexts/backtest/application/services/v2/
        notebook_parity_benchmark_corpus_v2.py
    """

    required_fields: tuple[NotebookParityMeasurementFieldLiteralV2, ...]
    system_scan_fields: tuple[NotebookParityMeasurementFieldLiteralV2, ...]
    notes: str

    def __post_init__(self) -> None:
        """
        Validate the additive notebook-parity measurement-field contract.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Required measurement fields remain reviewable literals in the committed corpus.
        Raises:
            ValueError: If required/system-scan fields are empty, duplicated, or drift from the
                committed notebook-parity measurement contract.
        Side Effects:
            None.
        """
        if len(self.required_fields) == 0:
            raise ValueError(
                "BacktestNotebookParityMeasurementContractV2.required_fields must be non-empty"
            )
        if len(self.required_fields) != len(set(self.required_fields)):
            raise ValueError(
                "BacktestNotebookParityMeasurementContractV2.required_fields must not contain "
                "duplicates"
            )
        if len(self.system_scan_fields) == 0:
            raise ValueError(
                "BacktestNotebookParityMeasurementContractV2.system_scan_fields must be "
                "non-empty"
            )
        if len(self.system_scan_fields) != len(set(self.system_scan_fields)):
            raise ValueError(
                "BacktestNotebookParityMeasurementContractV2.system_scan_fields must not "
                "contain duplicates"
            )
        if not set(self.system_scan_fields).issubset(set(self.required_fields)):
            raise ValueError(
                "BacktestNotebookParityMeasurementContractV2.system_scan_fields must be a "
                "subset of required_fields"
            )
        if self.required_fields != _ALLOWED_NOTEBOOK_PARITY_MEASUREMENT_FIELDS_V2:
            raise ValueError(
                "BacktestNotebookParityMeasurementContractV2.required_fields must match the "
                "committed notebook-parity measurement authority"
            )
        if not self.notes.strip():
            raise ValueError(
                "BacktestNotebookParityMeasurementContractV2.notes must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class BacktestNotebookParityAuthorityLayerV2:
    """
    Explicit benchmark-authority layer separating synthetic validation from live closure evidence.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_notebook_parity_benchmark_corpus_v1.json
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
    """

    authority_kind: NotebookParityAuthorityKindLiteralV2
    scenario_ids: tuple[str, ...]
    grants_final_closure: bool
    notes: str

    def __post_init__(self) -> None:
        """
        Validate one explicit benchmark-authority layer.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Synthetic perf-smoke can validate the committed contract, but only explicit live host
            measurements can grant final closure for canonical corrective scenarios.
        Raises:
            ValueError: If the layer is empty, duplicates scenarios, or blurs synthetic-vs-live
                closure semantics.
        Side Effects:
            None.
        """
        if len(self.scenario_ids) == 0:
            raise ValueError(
                "BacktestNotebookParityAuthorityLayerV2.scenario_ids must be non-empty"
            )
        if len(self.scenario_ids) != len(set(self.scenario_ids)):
            raise ValueError(
                "BacktestNotebookParityAuthorityLayerV2.scenario_ids must not contain "
                "duplicates"
            )
        if self.authority_kind == "synthetic_contract_validation" and self.grants_final_closure:
            raise ValueError(
                "BacktestNotebookParityAuthorityLayerV2.synthetic_contract_validation must "
                "not grant final closure"
            )
        if self.authority_kind == "live_host_measurement" and not self.grants_final_closure:
            raise ValueError(
                "BacktestNotebookParityAuthorityLayerV2.live_host_measurement must grant "
                "final closure"
            )
        if not self.notes.strip():
            raise ValueError(
                "BacktestNotebookParityAuthorityLayerV2.notes must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class BacktestNotebookParitySourceFixturesV2:
    """
    Canonical fixture and notebook-anchor paths for the notebook-parity benchmark program.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-engine-vnext.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_notebook_parity_benchmark_corpus_v1.json
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
    """

    perf_smoke_harness: str
    nr2_notebook_anchor: str
    rg_ttr_notebook_anchor: str

    def __post_init__(self) -> None:
        """
        Validate the committed notebook-parity fixture paths.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            The harness path stays under `tests/perf_smoke`, while anchors stay on committed
            `.ipynb` files under `tests/notebook_tests/new_engine`.
        Raises:
            ValueError: If one required path is blank or has an unexpected extension.
        Side Effects:
            None.
        """
        path_requirements = (
            ("perf_smoke_harness", self.perf_smoke_harness, ".py"),
            ("nr2_notebook_anchor", self.nr2_notebook_anchor, ".ipynb"),
            ("rg_ttr_notebook_anchor", self.rg_ttr_notebook_anchor, ".ipynb"),
        )
        for field_name, field_value, suffix in path_requirements:
            if not field_value.strip():
                raise ValueError(
                    f"BacktestNotebookParitySourceFixturesV2.{field_name} must be non-empty"
                )
            if not field_value.endswith(suffix):
                raise ValueError(
                    f"BacktestNotebookParitySourceFixturesV2.{field_name} must end with "
                    f"{suffix!r}"
                )


@dataclass(frozen=True, slots=True)
class BacktestNotebookParityEqualThreadBudgetRuleV2:
    """
    Canonical equal-thread-budget normalization rule for backend-vs-notebook comparisons.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - src/trading/contexts/backtest/application/services/v2/
        notebook_parity_benchmark_corpus_v2.py
    """

    rule_id: str
    literal: str
    comparison_field: str
    same_host_required: bool
    same_artifact_slot_required: bool
    comparable_runtime_surfaces: tuple[NotebookParityRuntimeSurfaceLiteralV2, ...]
    invalid_examples: tuple[str, ...]
    notes: str

    def __post_init__(self) -> None:
        """
        Validate the equal-thread-budget normalization rule.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Notebook-parity acceptance depends on identical host, slot, and `numba_threads_used`
            values rather than inferred CPU availability.
        Raises:
            ValueError: If one required rule field is blank, duplicated, or inconsistent with the
                committed equal-thread-budget contract.
        Side Effects:
            None.
        """
        if not self.rule_id.strip():
            raise ValueError(
                "BacktestNotebookParityEqualThreadBudgetRuleV2.rule_id must be non-empty"
            )
        if self.literal != "equal thread budget":
            raise ValueError(
                "BacktestNotebookParityEqualThreadBudgetRuleV2.literal must be "
                "'equal thread budget'"
            )
        if self.comparison_field != "numba_threads_used":
            raise ValueError(
                "BacktestNotebookParityEqualThreadBudgetRuleV2.comparison_field must be "
                "'numba_threads_used'"
            )
        if len(self.comparable_runtime_surfaces) == 0:
            raise ValueError(
                "BacktestNotebookParityEqualThreadBudgetRuleV2.comparable_runtime_surfaces "
                "must be non-empty"
            )
        if len(self.comparable_runtime_surfaces) != len(
            set(self.comparable_runtime_surfaces)
        ):
            raise ValueError(
                "BacktestNotebookParityEqualThreadBudgetRuleV2.comparable_runtime_surfaces "
                "must not contain duplicates"
            )
        if len(self.invalid_examples) == 0:
            raise ValueError(
                "BacktestNotebookParityEqualThreadBudgetRuleV2.invalid_examples must be "
                "non-empty"
            )
        if len(self.invalid_examples) != len(set(self.invalid_examples)):
            raise ValueError(
                "BacktestNotebookParityEqualThreadBudgetRuleV2.invalid_examples must not "
                "contain duplicates"
            )
        if not self.notes.strip():
            raise ValueError(
                "BacktestNotebookParityEqualThreadBudgetRuleV2.notes must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class BacktestNotebookParityAcceptanceGateV2:
    """
    One additive notebook-parity acceptance gate attached to a committed benchmark class.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - src/trading/contexts/backtest/application/services/v2/
        notebook_parity_benchmark_corpus_v2.py
    """

    gate_id: str
    metric: str
    max_ratio: float | None = None
    max_value: float | None = None
    expected_value: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        """
        Validate one additive notebook-parity acceptance gate.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Each gate defines exactly one or more explicit deterministic checks that future
            runtime measurements can evaluate without hidden host-specific logic.
        Raises:
            ValueError: If the gate does not define any threshold or expected-value check.
        Side Effects:
            None.
        """
        if not self.gate_id.strip():
            raise ValueError(
                "BacktestNotebookParityAcceptanceGateV2.gate_id must be non-empty"
            )
        if not self.metric.strip():
            raise ValueError(
                "BacktestNotebookParityAcceptanceGateV2.metric must be non-empty"
            )
        if (
            self.max_ratio is None
            and self.max_value is None
            and self.expected_value is None
        ):
            raise ValueError(
                "BacktestNotebookParityAcceptanceGateV2 must define one threshold or "
                "expected value"
            )
        if self.max_ratio is not None and self.max_ratio <= 0.0:
            raise ValueError(
                "BacktestNotebookParityAcceptanceGateV2.max_ratio must be > 0 when provided"
            )
        if self.max_value is not None and self.max_value < 0.0:
            raise ValueError(
                "BacktestNotebookParityAcceptanceGateV2.max_value must be >= 0 when provided"
            )
        if self.expected_value is not None and not self.expected_value.strip():
            raise ValueError(
                "BacktestNotebookParityAcceptanceGateV2.expected_value must be non-empty "
                "when provided"
            )
        if not self.notes.strip():
            raise ValueError(
                "BacktestNotebookParityAcceptanceGateV2.notes must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class BacktestNotebookParityBaselineReferencePointV2:
    """
    One explicit comparison point committed into the notebook-parity benchmark surface.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_notebook_parity_benchmark_corpus_v1.json
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
    """

    reference_id: str
    source_kind: NotebookParityReferenceSourceKindLiteralV2
    metric_label: str
    host_label: str
    runtime_surface: NotebookParityRuntimeSurfaceLiteralV2
    numba_threads_used: int | None = None
    wall_clock_seconds: float | None = None
    peak_rss_bytes: int | None = None
    max_python_processes_seen: int | None = None
    stage_b_execution_mode: NotebookParityStageBExecutionModeLiteralV2 | None = None
    stage_b_process_fallback_threshold: (
        NotebookParityStageBProcessFallbackThresholdLiteralV2 | None
    ) = None
    runtime_regression_ratio_limit: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        """
        Validate one explicit baseline reference point.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Reference points may be partial because A1 records only already-known benchmark
            authority, while later prompts tighten the missing numeric surfaces.
        Raises:
            ValueError: If the reference point is blank or carries no explicit comparison field.
        Side Effects:
            None.
        """
        if not self.reference_id.strip():
            raise ValueError(
                "BacktestNotebookParityBaselineReferencePointV2.reference_id must be "
                "non-empty"
            )
        if not self.metric_label.strip():
            raise ValueError(
                "BacktestNotebookParityBaselineReferencePointV2.metric_label must be "
                "non-empty"
            )
        if not self.host_label.strip():
            raise ValueError(
                "BacktestNotebookParityBaselineReferencePointV2.host_label must be non-empty"
            )
        if self.numba_threads_used is not None and self.numba_threads_used <= 0:
            raise ValueError(
                "BacktestNotebookParityBaselineReferencePointV2.numba_threads_used must be "
                "> 0 when provided"
            )
        if self.wall_clock_seconds is not None and self.wall_clock_seconds <= 0.0:
            raise ValueError(
                "BacktestNotebookParityBaselineReferencePointV2.wall_clock_seconds must be "
                "> 0 when provided"
            )
        if self.peak_rss_bytes is not None and self.peak_rss_bytes <= 0:
            raise ValueError(
                "BacktestNotebookParityBaselineReferencePointV2.peak_rss_bytes must be > 0 "
                "when provided"
            )
        if (
            self.max_python_processes_seen is not None
            and self.max_python_processes_seen <= 0
        ):
            raise ValueError(
                "BacktestNotebookParityBaselineReferencePointV2.max_python_processes_seen "
                "must be > 0 when provided"
            )
        if (
            self.runtime_regression_ratio_limit is not None
            and self.runtime_regression_ratio_limit <= 0.0
        ):
            raise ValueError(
                "BacktestNotebookParityBaselineReferencePointV2.runtime_regression_ratio_"
                "limit must be > 0 when provided"
            )
        if (
            self.wall_clock_seconds is None
            and self.peak_rss_bytes is None
            and self.max_python_processes_seen is None
            and self.stage_b_execution_mode is None
            and self.stage_b_process_fallback_threshold is None
            and self.runtime_regression_ratio_limit is None
        ):
            raise ValueError(
                "BacktestNotebookParityBaselineReferencePointV2 must define one explicit "
                "comparison value"
            )
        if not self.notes.strip():
            raise ValueError(
                "BacktestNotebookParityBaselineReferencePointV2.notes must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class BacktestNotebookParityScenarioV2:
    """
    One canonical notebook-parity benchmark class contract.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-engine-vnext.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_notebook_parity_benchmark_corpus_v1.json
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
    """

    scenario_id: str
    benchmark_class: NotebookParityBenchmarkClassLiteralV2
    comparison_mode: NotebookParityComparisonModeLiteralV2
    primary_metric: str
    supported_primary_metrics: tuple[str, ...]
    canonical_backend_surface: NotebookParityRuntimeSurfaceLiteralV2
    anchor_notebook: str
    equal_thread_budget_rule_id: str
    baseline_reference_points: tuple[BacktestNotebookParityBaselineReferencePointV2, ...]
    acceptance_gates: tuple[BacktestNotebookParityAcceptanceGateV2, ...]
    notes: str

    def __post_init__(self) -> None:
        """
        Validate one canonical notebook-parity scenario contract.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Scenario ordering, anchor notebooks, and acceptance gates are part of the committed
            benchmark authority and therefore stay byte-stable and reviewable.
        Raises:
            ValueError: If scenario literals, ranking metrics, or gate/reference ordering drift.
        Side Effects:
            None.
        """
        if not self.scenario_id.strip():
            raise ValueError(
                "BacktestNotebookParityScenarioV2.scenario_id must be non-empty"
            )
        if not self.primary_metric.strip():
            raise ValueError(
                "BacktestNotebookParityScenarioV2.primary_metric must be non-empty"
            )
        if len(self.supported_primary_metrics) == 0:
            raise ValueError(
                "BacktestNotebookParityScenarioV2.supported_primary_metrics must be "
                "non-empty"
            )
        if len(self.supported_primary_metrics) != len(set(self.supported_primary_metrics)):
            raise ValueError(
                "BacktestNotebookParityScenarioV2.supported_primary_metrics must not "
                "contain duplicates"
            )
        if self.primary_metric not in self.supported_primary_metrics:
            raise ValueError(
                "BacktestNotebookParityScenarioV2.primary_metric must belong to "
                "supported_primary_metrics"
            )
        if not self.anchor_notebook.endswith(".ipynb"):
            raise ValueError(
                "BacktestNotebookParityScenarioV2.anchor_notebook must point to .ipynb"
            )
        if self.equal_thread_budget_rule_id != "equal_thread_budget":
            raise ValueError(
                "BacktestNotebookParityScenarioV2.equal_thread_budget_rule_id must be "
                "'equal_thread_budget'"
            )
        if len(self.baseline_reference_points) == 0:
            raise ValueError(
                "BacktestNotebookParityScenarioV2.baseline_reference_points must be "
                "non-empty"
            )
        reference_ids = tuple(
            point.reference_id for point in self.baseline_reference_points
        )
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError(
                "BacktestNotebookParityScenarioV2.baseline_reference_points must not contain "
                "duplicate reference ids"
            )
        if len(self.acceptance_gates) == 0:
            raise ValueError(
                "BacktestNotebookParityScenarioV2.acceptance_gates must be non-empty"
            )
        gate_ids = tuple(gate.gate_id for gate in self.acceptance_gates)
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError(
                "BacktestNotebookParityScenarioV2.acceptance_gates must not contain "
                "duplicate gate ids"
            )
        if self.benchmark_class == "RG-ALT":
            if self.comparison_mode != "functional_baseline":
                raise ValueError(
                    "BacktestNotebookParityScenarioV2.RG-ALT must use "
                    "'functional_baseline' comparison mode"
                )
            if len(self.supported_primary_metrics) < 2:
                raise ValueError(
                    "BacktestNotebookParityScenarioV2.RG-ALT must list alternative metrics"
                )
        else:
            if self.comparison_mode != "notebook_parity":
                raise ValueError(
                    "BacktestNotebookParityScenarioV2.NR2 and RG-TTR must use "
                    "'notebook_parity' comparison mode"
                )
            if self.primary_metric != "total_return_pct":
                raise ValueError(
                    "BacktestNotebookParityScenarioV2.NR2 and RG-TTR must keep "
                    "'total_return_pct' as the primary_metric"
                )
        if not self.notes.strip():
            raise ValueError(
                "BacktestNotebookParityScenarioV2.notes must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class BacktestNotebookParityLiveHostCaptureV2:
    """
    Blocking live-host benchmark capture requirement for one canonical notebook-parity scenario.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_notebook_parity_benchmark_corpus_v1.json
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
    """

    capture_id: str
    scenario_id: str
    benchmark_class: NotebookParityBenchmarkClassLiteralV2
    runtime_surface: NotebookParityRuntimeSurfaceLiteralV2
    capture_status: NotebookParityLiveHostCaptureStatusLiteralV2
    blocking_closure: bool
    captured_measurement: BacktestNotebookParityMeasurementV2 | None
    notes: str

    def __post_init__(self) -> None:
        """
        Validate one canonical live-host capture requirement or recorded measurement.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Live host captures are backend-only benchmark artifacts and must stay separate from
            synthetic perf-smoke validation until an explicit canonical measurement exists.
        Raises:
            ValueError: If capture metadata is blank, the runtime surface is unsupported, or the
                optional measurement conflicts with the declared scenario metadata.
        Side Effects:
            None.
        """
        if not self.capture_id.strip():
            raise ValueError(
                "BacktestNotebookParityLiveHostCaptureV2.capture_id must be non-empty"
            )
        if not self.scenario_id.strip():
            raise ValueError(
                "BacktestNotebookParityLiveHostCaptureV2.scenario_id must be non-empty"
            )
        if self.runtime_surface == "notebook":
            raise ValueError(
                "BacktestNotebookParityLiveHostCaptureV2.runtime_surface must not be "
                "'notebook'"
            )
        if self.capture_status == "captured" and self.captured_measurement is None:
            raise ValueError(
                "BacktestNotebookParityLiveHostCaptureV2.captured_measurement must be "
                "provided when capture_status is 'captured'"
            )
        if self.capture_status == "missing" and self.captured_measurement is not None:
            raise ValueError(
                "BacktestNotebookParityLiveHostCaptureV2.captured_measurement must be "
                "absent when capture_status is 'missing'"
            )
        if self.captured_measurement is not None:
            if self.captured_measurement.measurement_source != "backend":
                raise ValueError(
                    "BacktestNotebookParityLiveHostCaptureV2.captured_measurement must use "
                    "'backend' as measurement_source"
                )
            if self.captured_measurement.scenario_id != self.scenario_id:
                raise ValueError(
                    "BacktestNotebookParityLiveHostCaptureV2.captured_measurement.scenario_id "
                    "must match scenario_id"
                )
            if self.captured_measurement.benchmark_class != self.benchmark_class:
                raise ValueError(
                    "BacktestNotebookParityLiveHostCaptureV2.captured_measurement."
                    "benchmark_class must match benchmark_class"
                )
            if self.captured_measurement.runtime_surface != self.runtime_surface:
                raise ValueError(
                    "BacktestNotebookParityLiveHostCaptureV2.captured_measurement."
                    "runtime_surface must match runtime_surface"
                )
        if not self.notes.strip():
            raise ValueError(
                "BacktestNotebookParityLiveHostCaptureV2.notes must be non-empty"
            )

    def has_required_capture_evidence(self) -> bool:
        """
        Report whether this live-host capture currently satisfies its capture-evidence contract.

        Args:
            None.
        Returns:
            bool: `True` when the capture is either non-blocking or already recorded explicitly.
        Assumptions:
            Final closure still depends on scenario gate evaluation; this helper answers only
            whether explicit live-host evidence has been recorded where required.
        Raises:
            None.
        Side Effects:
            None.
        """
        if not self.blocking_closure:
            return True
        return (
            self.capture_status == "captured"
            and self.captured_measurement is not None
        )


@dataclass(frozen=True, slots=True)
class BacktestNotebookParityBenchmarkCorpusV2:
    """
    Versioned notebook-parity benchmark corpus for A1 benchmark authority and perf smoke.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_notebook_parity_benchmark_corpus_v1.json
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
    """

    schema_version: int
    fixture_contract: str
    milestone_id: str
    status: str
    reference_docs: tuple[str, ...]
    measurement_contract: BacktestNotebookParityMeasurementContractV2
    authority_layers: tuple[BacktestNotebookParityAuthorityLayerV2, ...]
    live_host_captures: tuple[BacktestNotebookParityLiveHostCaptureV2, ...]
    source_fixtures: BacktestNotebookParitySourceFixturesV2
    equal_thread_budget_rule: BacktestNotebookParityEqualThreadBudgetRuleV2
    scenario_order: tuple[str, ...]
    scenarios: tuple[BacktestNotebookParityScenarioV2, ...]

    def __post_init__(self) -> None:
        """
        Validate the top-level notebook-parity benchmark corpus.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Top-level ordering and milestone identifiers remain part of the committed benchmark
            protocol and therefore stay stable across reviews.
        Raises:
            ValueError: If metadata literals or scenario ordering drift.
        Side Effects:
            None.
        """
        if self.schema_version != BACKTEST_NOTEBOOK_PARITY_BENCHMARK_CORPUS_SCHEMA_VERSION_V2:
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.schema_version must be "
                f"{BACKTEST_NOTEBOOK_PARITY_BENCHMARK_CORPUS_SCHEMA_VERSION_V2}"
            )
        if self.fixture_contract != BACKTEST_NOTEBOOK_PARITY_BENCHMARK_CORPUS_KIND_V2:
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.fixture_contract must be "
                f"{BACKTEST_NOTEBOOK_PARITY_BENCHMARK_CORPUS_KIND_V2!r}"
            )
        if self.milestone_id != BACKTEST_NOTEBOOK_PARITY_BENCHMARK_CORPUS_MILESTONE_ID_V2:
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.milestone_id must be "
                f"{BACKTEST_NOTEBOOK_PARITY_BENCHMARK_CORPUS_MILESTONE_ID_V2!r}"
            )
        if not self.status.strip():
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.status must be non-empty"
            )
        if len(self.reference_docs) == 0:
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.reference_docs must be non-empty"
            )
        if len(self.authority_layers) == 0:
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.authority_layers must be non-empty"
            )
        authority_kinds = tuple(
            layer.authority_kind for layer in self.authority_layers
        )
        if len(authority_kinds) != len(set(authority_kinds)):
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.authority_layers must not contain "
                "duplicate authority kinds"
            )
        if tuple(sorted(authority_kinds)) != tuple(
            sorted(_ALLOWED_NOTEBOOK_PARITY_AUTHORITY_KINDS_V2)
        ):
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.authority_layers must define "
                "synthetic_contract_validation and live_host_measurement"
            )
        if len(self.scenario_order) == 0:
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.scenario_order must be non-empty"
            )
        if len(self.scenario_order) != len(set(self.scenario_order)):
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.scenario_order must not contain "
                "duplicates"
            )
        authored_order = tuple(scenario.scenario_id for scenario in self.scenarios)
        if authored_order != self.scenario_order:
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.scenarios must follow scenario_order"
            )
        valid_scenario_ids = set(self.scenario_order)
        for layer in self.authority_layers:
            if not set(layer.scenario_ids).issubset(valid_scenario_ids):
                raise ValueError(
                    "BacktestNotebookParityBenchmarkCorpusV2.authority_layers must reference "
                    "committed scenarios only"
                )
        synthetic_layer = self.authority_layer_for_kind(
            authority_kind="synthetic_contract_validation"
        )
        if synthetic_layer.scenario_ids != self.scenario_order:
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.synthetic_contract_validation must "
                "cover every committed scenario"
            )
        capture_ids = tuple(capture.capture_id for capture in self.live_host_captures)
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.live_host_captures must not contain "
                "duplicate capture ids"
            )
        capture_scenario_ids = tuple(
            capture.scenario_id for capture in self.live_host_captures
        )
        if len(capture_scenario_ids) != len(set(capture_scenario_ids)):
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.live_host_captures must not contain "
                "duplicate scenario ids"
            )
        live_layer = self.authority_layer_for_kind(
            authority_kind="live_host_measurement"
        )
        if live_layer.scenario_ids != capture_scenario_ids:
            raise ValueError(
                "BacktestNotebookParityBenchmarkCorpusV2.live_host_measurement scenarios "
                "must match live_host_captures ordering"
            )
        for capture in self.live_host_captures:
            scenario = self.scenario_for_id(scenario_id=capture.scenario_id)
            if scenario.benchmark_class != capture.benchmark_class:
                raise ValueError(
                    "BacktestNotebookParityBenchmarkCorpusV2.live_host_captures must keep "
                    "scenario benchmark_class alignment"
                )

    def scenario_for_id(self, *, scenario_id: str) -> BacktestNotebookParityScenarioV2:
        """
        Return one notebook-parity scenario by its stable identifier.

        Args:
            scenario_id: Stable notebook-parity scenario identifier.
        Returns:
            BacktestNotebookParityScenarioV2: Matching committed scenario object.
        Assumptions:
            The caller uses one of the committed `scenario_order` identifiers.
        Raises:
            KeyError: If the scenario id is not present in this committed corpus.
        Side Effects:
            None.
        """
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(f"notebook-parity scenario not found: {scenario_id!r}")

    def authority_layer_for_kind(
        self,
        *,
        authority_kind: NotebookParityAuthorityKindLiteralV2,
    ) -> BacktestNotebookParityAuthorityLayerV2:
        """
        Return one explicit benchmark-authority layer by its stable kind.

        Args:
            authority_kind: Stable authority-layer literal.
        Returns:
            BacktestNotebookParityAuthorityLayerV2: Matching committed authority layer.
        Assumptions:
            The corpus always defines both synthetic and live authority layers explicitly.
        Raises:
            KeyError: If the requested authority layer is not present in the committed corpus.
        Side Effects:
            None.
        """
        for layer in self.authority_layers:
            if layer.authority_kind == authority_kind:
                return layer
        raise KeyError(f"notebook-parity authority layer not found: {authority_kind!r}")

    def live_host_capture_for_scenario(
        self,
        *,
        scenario_id: str,
    ) -> BacktestNotebookParityLiveHostCaptureV2 | None:
        """
        Return the explicit live-host capture requirement for one scenario when present.

        Args:
            scenario_id: Stable notebook-parity scenario identifier.
        Returns:
            BacktestNotebookParityLiveHostCaptureV2 | None: Matching live-host capture entry, or
                `None` when the scenario has no blocking live requirement.
        Assumptions:
            Only canonical corrective scenarios require explicit live host measurements.
        Raises:
            None.
        Side Effects:
            None.
        """
        for capture in self.live_host_captures:
            if capture.scenario_id == scenario_id:
                return capture
        return None

    def has_required_live_capture_evidence_for_scenario(
        self,
        *,
        scenario_id: str,
    ) -> bool:
        """
        Report whether explicit required live-host evidence exists for one scenario.

        Args:
            scenario_id: Stable notebook-parity scenario identifier.
        Returns:
            bool: `True` when the scenario does not require live host evidence or the blocking
                live capture has been recorded explicitly.
        Assumptions:
            This helper does not evaluate benchmark gates; it answers only whether the required
            live capture artifact exists for later closure review.
        Raises:
            None.
        Side Effects:
            None.
        """
        live_host_capture = self.live_host_capture_for_scenario(scenario_id=scenario_id)
        if live_host_capture is None:
            return True
        return live_host_capture.has_required_capture_evidence()


@dataclass(frozen=True, slots=True)
class BacktestNotebookParityMeasurementV2:
    """
    Canonical runtime-shape measurement payload emitted by the notebook-parity harness.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-engine-vnext.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - src/trading/contexts/backtest/application/services/v2/
        notebook_parity_benchmark_corpus_v2.py
    """

    scenario_id: str
    benchmark_class: NotebookParityBenchmarkClassLiteralV2
    measurement_source: NotebookParityMeasurementSourceLiteralV2
    runtime_surface: NotebookParityRuntimeSurfaceLiteralV2
    host_label: str
    artifact_slot: str
    wall_clock_seconds: float
    cpu_time_seconds: float
    peak_rss_bytes: int
    numba_threads_used: int
    max_python_processes_seen: int
    stage_b_execution_mode: NotebookParityStageBExecutionModeLiteralV2
    stage_b_process_fallback_threshold: NotebookParityStageBProcessFallbackThresholdLiteralV2
    exact_replay_count: int

    def __post_init__(self) -> None:
        """
        Validate one canonical runtime-shape measurement payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Measurement payloads are internal-only benchmark artifacts and must include the full
            runtime-shape contract, not just wall time.
        Raises:
            ValueError: If one required runtime-shape field is blank or non-positive.
        Side Effects:
            None.
        """
        if not self.scenario_id.strip():
            raise ValueError(
                "BacktestNotebookParityMeasurementV2.scenario_id must be non-empty"
            )
        if not self.host_label.strip():
            raise ValueError(
                "BacktestNotebookParityMeasurementV2.host_label must be non-empty"
            )
        if not self.artifact_slot.strip():
            raise ValueError(
                "BacktestNotebookParityMeasurementV2.artifact_slot must be non-empty"
            )
        if self.wall_clock_seconds <= 0.0:
            raise ValueError(
                "BacktestNotebookParityMeasurementV2.wall_clock_seconds must be > 0"
            )
        if self.cpu_time_seconds < 0.0:
            raise ValueError(
                "BacktestNotebookParityMeasurementV2.cpu_time_seconds must be >= 0"
            )
        if self.peak_rss_bytes <= 0:
            raise ValueError(
                "BacktestNotebookParityMeasurementV2.peak_rss_bytes must be > 0"
            )
        if self.numba_threads_used <= 0:
            raise ValueError(
                "BacktestNotebookParityMeasurementV2.numba_threads_used must be > 0"
            )
        if self.max_python_processes_seen <= 0:
            raise ValueError(
                "BacktestNotebookParityMeasurementV2.max_python_processes_seen must be > 0"
            )
        if self.exact_replay_count < 0:
            raise ValueError(
                "BacktestNotebookParityMeasurementV2.exact_replay_count must be >= 0"
            )


@dataclass(frozen=True, slots=True)
class BacktestNotebookParityComparisonV2:
    """
    Deterministic evaluation payload for one backend-vs-reference notebook-parity comparison.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - src/trading/contexts/backtest/application/services/v2/
        notebook_parity_benchmark_corpus_v2.py
    """

    scenario_id: str
    benchmark_class: NotebookParityBenchmarkClassLiteralV2
    thread_budget_aligned: bool
    host_aligned: bool
    artifact_slot_aligned: bool
    wall_clock_ratio: float
    peak_rss_ratio: float
    runtime_regression_ratio: float
    failing_gate_ids: tuple[str, ...]
    rule_violations: tuple[str, ...]
    passed: bool


def read_backtest_notebook_parity_benchmark_corpus_payload_v2(
    *,
    path: Path,
) -> dict[str, object]:
    """
    Read one notebook-parity benchmark-corpus JSON payload from disk.

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
        raise ValueError("notebook-parity benchmark corpus payload root must be an object")
    return cast(dict[str, object], payload)


def serialize_backtest_notebook_parity_benchmark_corpus_payload_v2(
    *,
    payload: dict[str, object],
) -> bytes:
    """
    Serialize one raw notebook-parity benchmark-corpus payload canonically.

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


def validate_backtest_notebook_parity_benchmark_corpus_payload_v2(
    *,
    payload: dict[str, object],
) -> None:
    """
    Validate one raw notebook-parity benchmark-corpus payload against typed contracts.

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
    _parse_backtest_notebook_parity_benchmark_corpus_payload_v2(payload=payload)


def load_backtest_notebook_parity_benchmark_corpus_v2(
    *,
    path: Path,
) -> BacktestNotebookParityBenchmarkCorpusV2:
    """
    Load the committed notebook-parity benchmark corpus into typed immutable contracts.

    Args:
        path: Absolute or repository-relative fixture path.
    Returns:
        BacktestNotebookParityBenchmarkCorpusV2: Parsed benchmark corpus.
    Assumptions:
        The committed corpus stays lightweight and is safe to read in deterministic perf-smoke
        tests.
    Raises:
        ValueError: If the payload violates the typed benchmark-corpus contract.
        OSError: If the fixture file cannot be read.
        json.JSONDecodeError: If the fixture file is invalid JSON.
    Side Effects:
        Reads one repository file from disk.
    """
    payload = read_backtest_notebook_parity_benchmark_corpus_payload_v2(path=path)
    return _parse_backtest_notebook_parity_benchmark_corpus_payload_v2(payload=payload)


def serialize_backtest_notebook_parity_measurements_v2(
    *,
    measurements: tuple[BacktestNotebookParityMeasurementV2, ...],
) -> bytes:
    """
    Serialize canonical notebook-parity measurement payloads deterministically.

    Args:
        measurements: Ordered immutable measurement payloads to serialize.
    Returns:
        bytes: Canonical UTF-8 JSON bytes with stable key ordering and trailing newline.
    Assumptions:
        Measurement ordering is already deterministic in the caller.
    Raises:
        TypeError: If one measurement field becomes non-JSON-serializable.
    Side Effects:
        None.
    """
    payload: dict[str, object] = {
        "measurements": [*_measurement_payloads_v2(measurements=measurements)],
    }
    return serialize_backtest_notebook_parity_benchmark_corpus_payload_v2(payload=payload)


def serialize_backtest_notebook_parity_live_host_captures_v2(
    *,
    captures: tuple[BacktestNotebookParityLiveHostCaptureV2, ...],
) -> bytes:
    """
    Serialize canonical live-host capture payloads deterministically for reviewable benchmark logs.

    Args:
        captures: Ordered immutable live-host capture payloads to serialize.
    Returns:
        bytes: Canonical UTF-8 JSON bytes with stable key ordering and trailing newline.
    Assumptions:
        Benchmark-host runs may record missing or captured live authority in a narrow internal
        payload without exposing these fields through public runtime contracts.
    Raises:
        TypeError: If one nested capture field becomes non-JSON-serializable.
    Side Effects:
        None.
    """
    payload: dict[str, object] = {
        "live_host_captures": [*_live_host_capture_payloads_v2(captures=captures)],
    }
    return serialize_backtest_notebook_parity_benchmark_corpus_payload_v2(payload=payload)


def evaluate_backtest_notebook_parity_scenario_v2(
    *,
    scenario: BacktestNotebookParityScenarioV2,
    equal_thread_budget_rule: BacktestNotebookParityEqualThreadBudgetRuleV2,
    candidate: BacktestNotebookParityMeasurementV2,
    reference: BacktestNotebookParityMeasurementV2,
) -> BacktestNotebookParityComparisonV2:
    """
    Evaluate one backend-vs-reference measurement pair against a notebook-parity scenario.

    Args:
        scenario: Canonical notebook-parity scenario contract.
        equal_thread_budget_rule: Top-level normalization rule shared by the corpus.
        candidate: Candidate backend measurement under evaluation.
        reference: Reference notebook or backend baseline measurement.
    Returns:
        BacktestNotebookParityComparisonV2: Deterministic comparison payload with ratios and
            failing gate identifiers.
    Assumptions:
        The caller provides one backend candidate and one reference measurement for the same
        scenario id and benchmark class.
    Raises:
        ValueError: If scenario ids or benchmark classes do not align with the committed scenario.
    Side Effects:
        None.
    """
    if candidate.scenario_id != scenario.scenario_id:
        raise ValueError("candidate.scenario_id must match scenario.scenario_id")
    if reference.scenario_id != scenario.scenario_id:
        raise ValueError("reference.scenario_id must match scenario.scenario_id")
    if candidate.benchmark_class != scenario.benchmark_class:
        raise ValueError("candidate.benchmark_class must match scenario.benchmark_class")
    if reference.benchmark_class != scenario.benchmark_class:
        raise ValueError("reference.benchmark_class must match scenario.benchmark_class")
    if candidate.measurement_source != "backend":
        raise ValueError("candidate.measurement_source must be 'backend'")

    thread_budget_aligned = (
        candidate.numba_threads_used == reference.numba_threads_used
    )
    host_aligned = (
        (not equal_thread_budget_rule.same_host_required)
        or candidate.host_label == reference.host_label
    )
    artifact_slot_aligned = (
        (not equal_thread_budget_rule.same_artifact_slot_required)
        or candidate.artifact_slot == reference.artifact_slot
    )
    wall_clock_ratio = candidate.wall_clock_seconds / reference.wall_clock_seconds
    peak_rss_ratio = candidate.peak_rss_bytes / reference.peak_rss_bytes
    runtime_regression_ratio = wall_clock_ratio

    rule_violations: list[str] = []
    if not thread_budget_aligned:
        rule_violations.append(equal_thread_budget_rule.rule_id)
    if not host_aligned:
        rule_violations.append("same_host")
    if not artifact_slot_aligned:
        rule_violations.append("same_artifact_slot")

    failing_gate_ids: list[str] = []
    for gate in scenario.acceptance_gates:
        if gate.metric in {"wall_clock_ratio", "runtime_regression_ratio"}:
            actual_ratio = (
                wall_clock_ratio
                if gate.metric == "wall_clock_ratio"
                else runtime_regression_ratio
            )
            if gate.max_ratio is not None and actual_ratio > gate.max_ratio:
                failing_gate_ids.append(gate.gate_id)
            continue
        if gate.metric == "peak_rss_ratio":
            if gate.max_ratio is not None and peak_rss_ratio > gate.max_ratio:
                failing_gate_ids.append(gate.gate_id)
            continue
        if gate.metric == "max_python_processes_seen":
            if (
                gate.max_value is not None
                and float(candidate.max_python_processes_seen) > gate.max_value
            ):
                failing_gate_ids.append(gate.gate_id)
            continue
        if gate.metric == "stage_b_execution_mode":
            if (
                gate.expected_value is not None
                and candidate.stage_b_execution_mode != gate.expected_value
            ):
                failing_gate_ids.append(gate.gate_id)
            continue
        if gate.metric == "stage_b_process_fallback_threshold":
            if (
                gate.expected_value is not None
                and candidate.stage_b_process_fallback_threshold != gate.expected_value
            ):
                failing_gate_ids.append(gate.gate_id)
            continue
        if gate.metric == "exact_replay_count":
            if (
                gate.max_value is not None
                and float(candidate.exact_replay_count) > gate.max_value
            ):
                failing_gate_ids.append(gate.gate_id)
            continue

    return BacktestNotebookParityComparisonV2(
        scenario_id=scenario.scenario_id,
        benchmark_class=scenario.benchmark_class,
        thread_budget_aligned=thread_budget_aligned,
        host_aligned=host_aligned,
        artifact_slot_aligned=artifact_slot_aligned,
        wall_clock_ratio=wall_clock_ratio,
        peak_rss_ratio=peak_rss_ratio,
        runtime_regression_ratio=runtime_regression_ratio,
        failing_gate_ids=tuple(failing_gate_ids),
        rule_violations=tuple(rule_violations),
        passed=len(failing_gate_ids) == 0 and len(rule_violations) == 0,
    )


def _parse_backtest_notebook_parity_benchmark_corpus_payload_v2(
    *,
    payload: dict[str, object],
) -> BacktestNotebookParityBenchmarkCorpusV2:
    """
    Parse the raw notebook-parity benchmark-corpus payload into typed contracts.

    Args:
        payload: Raw JSON payload loaded from disk.
    Returns:
        BacktestNotebookParityBenchmarkCorpusV2: Parsed top-level benchmark corpus object.
    Assumptions:
        Raw payload objects preserve authored ordering from the committed JSON fixture.
    Raises:
        ValueError: If required keys are missing or carry unsupported values.
    Side Effects:
        None.
    """
    reference_docs = _require_string_tuple(payload=payload, key="reference_docs")
    measurement_contract_map = _require_mapping(payload=payload, key="measurement_contract")
    authority_layer_payloads = _require_mapping_sequence(
        payload=payload,
        key="authority_layers",
    )
    live_host_capture_payloads = _require_mapping_sequence(
        payload=payload,
        key="live_host_captures",
    )
    source_fixtures_map = _require_mapping(payload=payload, key="source_fixtures")
    equal_thread_budget_rule_map = _require_mapping(
        payload=payload,
        key="equal_thread_budget_rule",
    )
    scenarios_payload = _require_mapping_sequence(payload=payload, key="scenarios")
    scenario_order = _require_string_tuple(payload=payload, key="scenario_order")

    return BacktestNotebookParityBenchmarkCorpusV2(
        schema_version=_require_int(payload=payload, key="schema_version"),
        fixture_contract=_require_str(payload=payload, key="fixture_contract"),
        milestone_id=_require_str(payload=payload, key="milestone_id"),
        status=_require_str(payload=payload, key="status"),
        reference_docs=reference_docs,
        measurement_contract=_parse_notebook_parity_measurement_contract_v2(
            raw_contract=measurement_contract_map
        ),
        authority_layers=tuple(
            _parse_notebook_parity_authority_layer_v2(raw_layer=raw_layer)
            for raw_layer in authority_layer_payloads
        ),
        live_host_captures=tuple(
            _parse_backtest_notebook_parity_live_host_capture_v2(
                raw_capture=raw_capture
            )
            for raw_capture in live_host_capture_payloads
        ),
        source_fixtures=BacktestNotebookParitySourceFixturesV2(
            perf_smoke_harness=_require_str(
                payload=source_fixtures_map,
                key="perf_smoke_harness",
            ),
            nr2_notebook_anchor=_require_str(
                payload=source_fixtures_map,
                key="nr2_notebook_anchor",
            ),
            rg_ttr_notebook_anchor=_require_str(
                payload=source_fixtures_map,
                key="rg_ttr_notebook_anchor",
            ),
        ),
        equal_thread_budget_rule=_parse_notebook_parity_equal_thread_budget_rule_v2(
            raw_rule=equal_thread_budget_rule_map
        ),
        scenario_order=scenario_order,
        scenarios=tuple(
            _parse_backtest_notebook_parity_scenario_v2(raw_scenario=raw_scenario)
            for raw_scenario in scenarios_payload
        ),
    )


def _parse_notebook_parity_measurement_contract_v2(
    *,
    raw_contract: dict[str, object],
) -> BacktestNotebookParityMeasurementContractV2:
    """
    Parse one raw notebook-parity measurement contract object.

    Args:
        raw_contract: Raw JSON object describing the measurement contract.
    Returns:
        BacktestNotebookParityMeasurementContractV2: Parsed immutable contract.
    Assumptions:
        Required measurement fields remain explicit literals in the committed JSON fixture.
    Raises:
        ValueError: If field arrays contain unsupported literals.
    Side Effects:
        None.
    """
    return BacktestNotebookParityMeasurementContractV2(
        required_fields=cast(
            tuple[NotebookParityMeasurementFieldLiteralV2, ...],
            tuple(
                _parse_notebook_parity_measurement_field_v2(value=value)
                for value in _require_string_tuple(
                    payload=raw_contract,
                    key="required_fields",
                )
            ),
        ),
        system_scan_fields=cast(
            tuple[NotebookParityMeasurementFieldLiteralV2, ...],
            tuple(
                _parse_notebook_parity_measurement_field_v2(value=value)
                for value in _require_string_tuple(
                    payload=raw_contract,
                    key="system_scan_fields",
                )
            ),
        ),
        notes=_require_str(payload=raw_contract, key="notes"),
    )


def _parse_notebook_parity_authority_layer_v2(
    *,
    raw_layer: dict[str, object],
) -> BacktestNotebookParityAuthorityLayerV2:
    """
    Parse one raw benchmark-authority layer object.

    Args:
        raw_layer: Raw JSON object describing one authority layer.
    Returns:
        BacktestNotebookParityAuthorityLayerV2: Parsed immutable authority layer.
    Assumptions:
        Synthetic contract validation and live host measurement stay explicit instead of inferred
        from prose or scenario naming.
    Raises:
        ValueError: If the layer carries unsupported literals or inconsistent closure semantics.
    Side Effects:
        None.
    """
    return BacktestNotebookParityAuthorityLayerV2(
        authority_kind=_parse_notebook_parity_authority_kind_v2(
            value=_require_str(payload=raw_layer, key="authority_kind")
        ),
        scenario_ids=_require_string_tuple(payload=raw_layer, key="scenario_ids"),
        grants_final_closure=_require_bool(
            payload=raw_layer,
            key="grants_final_closure",
        ),
        notes=_require_str(payload=raw_layer, key="notes"),
    )


def _parse_notebook_parity_equal_thread_budget_rule_v2(
    *,
    raw_rule: dict[str, object],
) -> BacktestNotebookParityEqualThreadBudgetRuleV2:
    """
    Parse one raw equal-thread-budget rule object.

    Args:
        raw_rule: Raw JSON object describing the normalization rule.
    Returns:
        BacktestNotebookParityEqualThreadBudgetRuleV2: Parsed immutable rule.
    Assumptions:
        Runtime-surface comparability stays explicitly listed instead of inferred.
    Raises:
        ValueError: If one rule field is unsupported.
    Side Effects:
        None.
    """
    return BacktestNotebookParityEqualThreadBudgetRuleV2(
        rule_id=_require_str(payload=raw_rule, key="rule_id"),
        literal=_require_str(payload=raw_rule, key="literal"),
        comparison_field=_require_str(payload=raw_rule, key="comparison_field"),
        same_host_required=_require_bool(payload=raw_rule, key="same_host_required"),
        same_artifact_slot_required=_require_bool(
            payload=raw_rule,
            key="same_artifact_slot_required",
        ),
        comparable_runtime_surfaces=cast(
            tuple[NotebookParityRuntimeSurfaceLiteralV2, ...],
            tuple(
                _parse_notebook_parity_runtime_surface_v2(value=value)
                for value in _require_string_tuple(
                    payload=raw_rule,
                    key="comparable_runtime_surfaces",
                )
            ),
        ),
        invalid_examples=_require_string_tuple(payload=raw_rule, key="invalid_examples"),
        notes=_require_str(payload=raw_rule, key="notes"),
    )


def _parse_backtest_notebook_parity_scenario_v2(
    *,
    raw_scenario: dict[str, object],
) -> BacktestNotebookParityScenarioV2:
    """
    Parse one raw notebook-parity scenario object into a typed immutable contract.

    Args:
        raw_scenario: Raw JSON object for one scenario entry.
    Returns:
        BacktestNotebookParityScenarioV2: Parsed typed scenario contract.
    Assumptions:
        Scenario arrays preserve authored ordering and contain explicit benchmark metadata.
    Raises:
        ValueError: If the scenario payload violates typed benchmark-corpus invariants.
    Side Effects:
        None.
    """
    baseline_reference_points = tuple(
        _parse_notebook_parity_baseline_reference_point_v2(raw_point=raw_point)
        for raw_point in _require_mapping_sequence(
            payload=raw_scenario,
            key="baseline_reference_points",
        )
    )
    acceptance_gates = tuple(
        _parse_notebook_parity_acceptance_gate_v2(raw_gate=raw_gate)
        for raw_gate in _require_mapping_sequence(
            payload=raw_scenario,
            key="acceptance_gates",
        )
    )
    return BacktestNotebookParityScenarioV2(
        scenario_id=_require_str(payload=raw_scenario, key="scenario_id"),
        benchmark_class=_parse_notebook_parity_benchmark_class_v2(
            value=_require_str(payload=raw_scenario, key="benchmark_class")
        ),
        comparison_mode=_parse_notebook_parity_comparison_mode_v2(
            value=_require_str(payload=raw_scenario, key="comparison_mode")
        ),
        primary_metric=_require_str(payload=raw_scenario, key="primary_metric"),
        supported_primary_metrics=_require_string_tuple(
            payload=raw_scenario,
            key="supported_primary_metrics",
        ),
        canonical_backend_surface=_parse_notebook_parity_runtime_surface_v2(
            value=_require_str(payload=raw_scenario, key="canonical_backend_surface")
        ),
        anchor_notebook=_require_str(payload=raw_scenario, key="anchor_notebook"),
        equal_thread_budget_rule_id=_require_str(
            payload=raw_scenario,
            key="equal_thread_budget_rule_id",
        ),
        baseline_reference_points=baseline_reference_points,
        acceptance_gates=acceptance_gates,
        notes=_require_str(payload=raw_scenario, key="notes"),
    )


def _parse_backtest_notebook_parity_live_host_capture_v2(
    *,
    raw_capture: dict[str, object],
) -> BacktestNotebookParityLiveHostCaptureV2:
    """
    Parse one raw live-host capture object into a typed immutable contract.

    Args:
        raw_capture: Raw JSON object for one live-host capture entry.
    Returns:
        BacktestNotebookParityLiveHostCaptureV2: Parsed typed live-host capture contract.
    Assumptions:
        The committed corpus may record either a missing closure blocker or a captured backend
        measurement, but must always state the status explicitly.
    Raises:
        ValueError: If one live-host capture field is unsupported or inconsistent.
    Side Effects:
        None.
    """
    captured_measurement_map = _require_optional_mapping(
        payload=raw_capture,
        key="captured_measurement",
    )
    return BacktestNotebookParityLiveHostCaptureV2(
        capture_id=_require_str(payload=raw_capture, key="capture_id"),
        scenario_id=_require_str(payload=raw_capture, key="scenario_id"),
        benchmark_class=_parse_notebook_parity_benchmark_class_v2(
            value=_require_str(payload=raw_capture, key="benchmark_class")
        ),
        runtime_surface=_parse_notebook_parity_runtime_surface_v2(
            value=_require_str(payload=raw_capture, key="runtime_surface")
        ),
        capture_status=_parse_notebook_parity_live_host_capture_status_v2(
            value=_require_str(payload=raw_capture, key="capture_status")
        ),
        blocking_closure=_require_bool(payload=raw_capture, key="blocking_closure"),
        captured_measurement=(
            _parse_backtest_notebook_parity_measurement_v2(
                raw_measurement=captured_measurement_map
            )
            if captured_measurement_map is not None
            else None
        ),
        notes=_require_str(payload=raw_capture, key="notes"),
    )


def _parse_notebook_parity_baseline_reference_point_v2(
    *,
    raw_point: dict[str, object],
) -> BacktestNotebookParityBaselineReferencePointV2:
    """
    Parse one raw baseline reference point object into a typed immutable contract.

    Args:
        raw_point: Raw JSON object for one reference-point entry.
    Returns:
        BacktestNotebookParityBaselineReferencePointV2: Parsed typed reference point.
    Assumptions:
        A1 reference points may be partial but must still expose one explicit comparison value.
    Raises:
        ValueError: If one reference-point field is unsupported or inconsistent.
    Side Effects:
        None.
    """
    stage_b_execution_mode = _require_optional_str(
        payload=raw_point,
        key="stage_b_execution_mode",
    )
    stage_b_process_fallback_threshold = _require_optional_str(
        payload=raw_point,
        key="stage_b_process_fallback_threshold",
    )
    return BacktestNotebookParityBaselineReferencePointV2(
        reference_id=_require_str(payload=raw_point, key="reference_id"),
        source_kind=_parse_notebook_parity_reference_source_kind_v2(
            value=_require_str(payload=raw_point, key="source_kind")
        ),
        metric_label=_require_str(payload=raw_point, key="metric_label"),
        host_label=_require_str(payload=raw_point, key="host_label"),
        runtime_surface=_parse_notebook_parity_runtime_surface_v2(
            value=_require_str(payload=raw_point, key="runtime_surface")
        ),
        numba_threads_used=_require_optional_int(
            payload=raw_point,
            key="numba_threads_used",
        ),
        wall_clock_seconds=_require_optional_float(
            payload=raw_point,
            key="wall_clock_seconds",
        ),
        peak_rss_bytes=_require_optional_int(payload=raw_point, key="peak_rss_bytes"),
        max_python_processes_seen=_require_optional_int(
            payload=raw_point,
            key="max_python_processes_seen",
        ),
        stage_b_execution_mode=(
            _parse_notebook_parity_stage_b_execution_mode_v2(value=stage_b_execution_mode)
            if stage_b_execution_mode is not None
            else None
        ),
        stage_b_process_fallback_threshold=(
            _parse_notebook_parity_stage_b_process_fallback_threshold_v2(
                value=stage_b_process_fallback_threshold
            )
            if stage_b_process_fallback_threshold is not None
            else None
        ),
        runtime_regression_ratio_limit=_require_optional_float(
            payload=raw_point,
            key="runtime_regression_ratio_limit",
        ),
        notes=_require_str(payload=raw_point, key="notes"),
    )


def _parse_notebook_parity_acceptance_gate_v2(
    *,
    raw_gate: dict[str, object],
) -> BacktestNotebookParityAcceptanceGateV2:
    """
    Parse one raw notebook-parity acceptance gate object.

    Args:
        raw_gate: Raw JSON object for one acceptance gate entry.
    Returns:
        BacktestNotebookParityAcceptanceGateV2: Parsed immutable gate.
    Assumptions:
        Gate identifiers and thresholds stay explicit in the committed corpus.
    Raises:
        ValueError: If the gate payload violates typed invariants.
    Side Effects:
        None.
    """
    return BacktestNotebookParityAcceptanceGateV2(
        gate_id=_require_str(payload=raw_gate, key="gate_id"),
        metric=_require_str(payload=raw_gate, key="metric"),
        max_ratio=_require_optional_float(payload=raw_gate, key="max_ratio"),
        max_value=_require_optional_float(payload=raw_gate, key="max_value"),
        expected_value=_require_optional_str(payload=raw_gate, key="expected_value"),
        notes=_require_str(payload=raw_gate, key="notes"),
    )


def _parse_backtest_notebook_parity_measurement_v2(
    *,
    raw_measurement: dict[str, object],
) -> BacktestNotebookParityMeasurementV2:
    """
    Parse one raw runtime-shape measurement object into a typed immutable benchmark payload.

    Args:
        raw_measurement: Raw JSON object for one measurement entry.
    Returns:
        BacktestNotebookParityMeasurementV2: Parsed typed runtime-shape measurement.
    Assumptions:
        Live host capture payloads reuse the same internal measurement contract as perf-smoke
        samples to avoid parallel benchmark schemas.
    Raises:
        ValueError: If one measurement field is unsupported or violates typed invariants.
    Side Effects:
        None.
    """
    return BacktestNotebookParityMeasurementV2(
        scenario_id=_require_str(payload=raw_measurement, key="scenario_id"),
        benchmark_class=_parse_notebook_parity_benchmark_class_v2(
            value=_require_str(payload=raw_measurement, key="benchmark_class")
        ),
        measurement_source=_parse_notebook_parity_measurement_source_v2(
            value=_require_str(payload=raw_measurement, key="measurement_source")
        ),
        runtime_surface=_parse_notebook_parity_runtime_surface_v2(
            value=_require_str(payload=raw_measurement, key="runtime_surface")
        ),
        host_label=_require_str(payload=raw_measurement, key="host_label"),
        artifact_slot=_require_str(payload=raw_measurement, key="artifact_slot"),
        wall_clock_seconds=_require_float(payload=raw_measurement, key="wall_clock_seconds"),
        cpu_time_seconds=_require_float(payload=raw_measurement, key="cpu_time_seconds"),
        peak_rss_bytes=_require_int(payload=raw_measurement, key="peak_rss_bytes"),
        numba_threads_used=_require_int(payload=raw_measurement, key="numba_threads_used"),
        max_python_processes_seen=_require_int(
            payload=raw_measurement,
            key="max_python_processes_seen",
        ),
        stage_b_execution_mode=_parse_notebook_parity_stage_b_execution_mode_v2(
            value=_require_str(payload=raw_measurement, key="stage_b_execution_mode")
        ),
        stage_b_process_fallback_threshold=(
            _parse_notebook_parity_stage_b_process_fallback_threshold_v2(
                value=_require_str(
                    payload=raw_measurement,
                    key="stage_b_process_fallback_threshold",
                )
            )
        ),
        exact_replay_count=_require_int(payload=raw_measurement, key="exact_replay_count"),
    )


def _measurement_payloads_v2(
    *,
    measurements: tuple[BacktestNotebookParityMeasurementV2, ...],
) -> tuple[dict[str, object], ...]:
    """
    Convert typed notebook-parity measurements into canonical JSON payload dictionaries.

    Args:
        measurements: Ordered immutable measurement payloads.
    Returns:
        tuple[dict[str, object], ...]: Ordered JSON-compatible payload mappings.
    Assumptions:
        Dataclass field ordering is deliberate and mirrors the committed measurement contract.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple(
        _measurement_payload_v2(measurement=measurement)
        for measurement in measurements
    )


def _measurement_payload_v2(
    *,
    measurement: BacktestNotebookParityMeasurementV2,
) -> dict[str, object]:
    """
    Convert one typed runtime-shape measurement into its canonical JSON payload dictionary.

    Args:
        measurement: Immutable measurement payload to convert.
    Returns:
        dict[str, object]: Ordered JSON-compatible measurement mapping.
    Assumptions:
        Field ordering mirrors the committed internal benchmark contract exactly.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "scenario_id": measurement.scenario_id,
        "benchmark_class": measurement.benchmark_class,
        "measurement_source": measurement.measurement_source,
        "runtime_surface": measurement.runtime_surface,
        "host_label": measurement.host_label,
        "artifact_slot": measurement.artifact_slot,
        "wall_clock_seconds": measurement.wall_clock_seconds,
        "cpu_time_seconds": measurement.cpu_time_seconds,
        "peak_rss_bytes": measurement.peak_rss_bytes,
        "numba_threads_used": measurement.numba_threads_used,
        "max_python_processes_seen": measurement.max_python_processes_seen,
        "stage_b_execution_mode": measurement.stage_b_execution_mode,
        "stage_b_process_fallback_threshold": (
            measurement.stage_b_process_fallback_threshold
        ),
        "exact_replay_count": measurement.exact_replay_count,
    }


def _live_host_capture_payloads_v2(
    *,
    captures: tuple[BacktestNotebookParityLiveHostCaptureV2, ...],
) -> tuple[dict[str, object], ...]:
    """
    Convert typed live-host captures into canonical JSON payload dictionaries.

    Args:
        captures: Ordered immutable live-host capture payloads.
    Returns:
        tuple[dict[str, object], ...]: Ordered JSON-compatible payload mappings.
    Assumptions:
        Missing-vs-captured status must stay reviewable in deterministic serialization output.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple(
        {
            "capture_id": capture.capture_id,
            "scenario_id": capture.scenario_id,
            "benchmark_class": capture.benchmark_class,
            "runtime_surface": capture.runtime_surface,
            "capture_status": capture.capture_status,
            "blocking_closure": capture.blocking_closure,
            "captured_measurement": (
                _measurement_payload_v2(measurement=capture.captured_measurement)
                if capture.captured_measurement is not None
                else None
            ),
            "notes": capture.notes,
        }
        for capture in captures
    )


def _parse_notebook_parity_benchmark_class_v2(
    *,
    value: str,
) -> NotebookParityBenchmarkClassLiteralV2:
    """
    Parse one notebook-parity benchmark-class literal.

    Args:
        value: Raw benchmark-class literal from JSON.
    Returns:
        NotebookParityBenchmarkClassLiteralV2: Supported benchmark class literal.
    Assumptions:
        Benchmark classes are case-sensitive and explicitly authored in the committed corpus.
    Raises:
        ValueError: If the literal is unsupported.
    Side Effects:
        None.
    """
    if value not in _ALLOWED_NOTEBOOK_PARITY_BENCHMARK_CLASSES_V2:
        raise ValueError(f"unsupported notebook-parity benchmark_class: {value!r}")
    return cast(NotebookParityBenchmarkClassLiteralV2, value)


def _parse_notebook_parity_comparison_mode_v2(
    *,
    value: str,
) -> NotebookParityComparisonModeLiteralV2:
    """
    Parse one notebook-parity comparison-mode literal.

    Args:
        value: Raw comparison-mode literal from JSON.
    Returns:
        NotebookParityComparisonModeLiteralV2: Supported comparison-mode literal.
    Assumptions:
        Comparison modes stay additive and explicit in the committed corpus.
    Raises:
        ValueError: If the literal is unsupported.
    Side Effects:
        None.
    """
    if value not in _ALLOWED_NOTEBOOK_PARITY_COMPARISON_MODES_V2:
        raise ValueError(f"unsupported notebook-parity comparison_mode: {value!r}")
    return cast(NotebookParityComparisonModeLiteralV2, value)


def _parse_notebook_parity_measurement_field_v2(
    *,
    value: str,
) -> NotebookParityMeasurementFieldLiteralV2:
    """
    Parse one notebook-parity measurement-field literal.

    Args:
        value: Raw measurement-field literal from JSON.
    Returns:
        NotebookParityMeasurementFieldLiteralV2: Supported measurement-field literal.
    Assumptions:
        Measurement-field names are part of the benchmark contract and therefore case-sensitive.
    Raises:
        ValueError: If the literal is unsupported.
    Side Effects:
        None.
    """
    if value not in _ALLOWED_NOTEBOOK_PARITY_MEASUREMENT_FIELDS_V2:
        raise ValueError(f"unsupported notebook-parity measurement field: {value!r}")
    return cast(NotebookParityMeasurementFieldLiteralV2, value)


def _parse_notebook_parity_measurement_source_v2(
    *,
    value: str,
) -> NotebookParityMeasurementSourceLiteralV2:
    """
    Parse one notebook-parity measurement-source literal.

    Args:
        value: Raw measurement-source literal from JSON.
    Returns:
        NotebookParityMeasurementSourceLiteralV2: Supported measurement-source literal.
    Assumptions:
        Benchmark measurements remain attributable either to backend or notebook sources.
    Raises:
        ValueError: If the literal is unsupported.
    Side Effects:
        None.
    """
    if value not in _ALLOWED_NOTEBOOK_PARITY_MEASUREMENT_SOURCES_V2:
        raise ValueError(f"unsupported notebook-parity measurement_source: {value!r}")
    return cast(NotebookParityMeasurementSourceLiteralV2, value)


def _parse_notebook_parity_authority_kind_v2(
    *,
    value: str,
) -> NotebookParityAuthorityKindLiteralV2:
    """
    Parse one notebook-parity authority-kind literal.

    Args:
        value: Raw authority-kind literal from JSON.
    Returns:
        NotebookParityAuthorityKindLiteralV2: Supported authority-kind literal.
    Assumptions:
        Synthetic validation and live host measurement stay explicit top-level authorities.
    Raises:
        ValueError: If the literal is unsupported.
    Side Effects:
        None.
    """
    if value not in _ALLOWED_NOTEBOOK_PARITY_AUTHORITY_KINDS_V2:
        raise ValueError(f"unsupported notebook-parity authority_kind: {value!r}")
    return cast(NotebookParityAuthorityKindLiteralV2, value)


def _parse_notebook_parity_live_host_capture_status_v2(
    *,
    value: str,
) -> NotebookParityLiveHostCaptureStatusLiteralV2:
    """
    Parse one notebook-parity live-host capture-status literal.

    Args:
        value: Raw capture-status literal from JSON.
    Returns:
        NotebookParityLiveHostCaptureStatusLiteralV2: Supported capture-status literal.
    Assumptions:
        The benchmark corpus must distinguish missing live authority from captured host evidence.
    Raises:
        ValueError: If the literal is unsupported.
    Side Effects:
        None.
    """
    if value not in _ALLOWED_NOTEBOOK_PARITY_LIVE_HOST_CAPTURE_STATUSES_V2:
        raise ValueError(f"unsupported notebook-parity live_host_capture status: {value!r}")
    return cast(NotebookParityLiveHostCaptureStatusLiteralV2, value)


def _parse_notebook_parity_reference_source_kind_v2(
    *,
    value: str,
) -> NotebookParityReferenceSourceKindLiteralV2:
    """
    Parse one notebook-parity reference-source literal.

    Args:
        value: Raw reference-source literal from JSON.
    Returns:
        NotebookParityReferenceSourceKindLiteralV2: Supported reference-source literal.
    Assumptions:
        Reference-source kinds remain explicit to keep the committed comparison points reviewable.
    Raises:
        ValueError: If the literal is unsupported.
    Side Effects:
        None.
    """
    if value not in _ALLOWED_NOTEBOOK_PARITY_REFERENCE_SOURCE_KINDS_V2:
        raise ValueError(f"unsupported notebook-parity source_kind: {value!r}")
    return cast(NotebookParityReferenceSourceKindLiteralV2, value)


def _parse_notebook_parity_runtime_surface_v2(
    *,
    value: str,
) -> NotebookParityRuntimeSurfaceLiteralV2:
    """
    Parse one notebook-parity runtime-surface literal.

    Args:
        value: Raw runtime-surface literal from JSON.
    Returns:
        NotebookParityRuntimeSurfaceLiteralV2: Supported runtime-surface literal.
    Assumptions:
        Sync, worker, and notebook surfaces stay explicitly distinguishable in metrics.
    Raises:
        ValueError: If the literal is unsupported.
    Side Effects:
        None.
    """
    if value not in _ALLOWED_NOTEBOOK_PARITY_RUNTIME_SURFACES_V2:
        raise ValueError(f"unsupported notebook-parity runtime_surface: {value!r}")
    return cast(NotebookParityRuntimeSurfaceLiteralV2, value)


def _parse_notebook_parity_stage_b_execution_mode_v2(
    *,
    value: str,
) -> NotebookParityStageBExecutionModeLiteralV2:
    """
    Parse one notebook-parity Stage B execution-mode literal.

    Args:
        value: Raw execution-mode literal from JSON.
    Returns:
        NotebookParityStageBExecutionModeLiteralV2: Supported Stage B execution-mode literal.
    Assumptions:
        No-risk bypass, in-process, and process-pool modes are the only committed A1 values.
    Raises:
        ValueError: If the literal is unsupported.
    Side Effects:
        None.
    """
    if value not in _ALLOWED_NOTEBOOK_PARITY_STAGE_B_EXECUTION_MODES_V2:
        raise ValueError(f"unsupported stage_b_execution_mode: {value!r}")
    return cast(NotebookParityStageBExecutionModeLiteralV2, value)


def _parse_notebook_parity_stage_b_process_fallback_threshold_v2(
    *,
    value: str,
) -> NotebookParityStageBProcessFallbackThresholdLiteralV2:
    """
    Parse one notebook-parity Stage B process-fallback threshold literal.

    Args:
        value: Raw threshold literal from JSON.
    Returns:
        NotebookParityStageBProcessFallbackThresholdLiteralV2:
            Supported benchmark-visible threshold literal.
    Assumptions:
        The benchmark surface must expose whether the fallback path stayed inactive or crossed the
        explicit `stage_b_variants_total` workload threshold.
    Raises:
        ValueError: If the literal is unsupported.
    Side Effects:
        None.
    """
    if value not in _ALLOWED_NOTEBOOK_PARITY_STAGE_B_PROCESS_FALLBACK_THRESHOLDS_V2:
        raise ValueError(
            f"unsupported stage_b_process_fallback_threshold: {value!r}"
        )
    return cast(NotebookParityStageBProcessFallbackThresholdLiteralV2, value)


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
    Require one optional mapping child from a raw JSON object payload.

    Args:
        payload: Raw JSON object payload.
        key: Optional mapping key.
    Returns:
        dict[str, object] | None: Child mapping value when present, otherwise `None`.
    Assumptions:
        Optional child objects use explicit `null` when no mapping payload exists yet.
    Raises:
        ValueError: If the key is present but the value is not an object mapping.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object mapping when provided")
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


def _require_float(
    *,
    payload: dict[str, object],
    key: str,
) -> float:
    """
    Require one float-or-int child from a raw JSON object payload.

    Args:
        payload: Raw JSON object payload.
        key: Required float key.
    Returns:
        float: Floating-point value, converting JSON integers losslessly.
    Assumptions:
        JSON numeric runtime-shape fields may be authored as ints or floats.
    Raises:
        ValueError: If the key is missing or the value is not numeric.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{key} must be a float")
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"{key} must be a float")


def _require_bool(
    *,
    payload: dict[str, object],
    key: str,
) -> bool:
    """
    Require one boolean scalar value from a raw JSON object payload.

    Args:
        payload: Raw JSON object payload.
        key: Required scalar key.
    Returns:
        bool: Boolean scalar value.
    Assumptions:
        JSON booleans are used only for explicit rule toggles.
    Raises:
        ValueError: If the key is missing or the value is not a boolean.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


__all__ = [
    "BACKTEST_NOTEBOOK_PARITY_BENCHMARK_CORPUS_KIND_V2",
    "BACKTEST_NOTEBOOK_PARITY_BENCHMARK_CORPUS_MILESTONE_ID_V2",
    "BACKTEST_NOTEBOOK_PARITY_BENCHMARK_CORPUS_SCHEMA_VERSION_V2",
    "BacktestNotebookParityAcceptanceGateV2",
    "BacktestNotebookParityAuthorityLayerV2",
    "BacktestNotebookParityBaselineReferencePointV2",
    "BacktestNotebookParityBenchmarkCorpusV2",
    "BacktestNotebookParityComparisonV2",
    "BacktestNotebookParityEqualThreadBudgetRuleV2",
    "BacktestNotebookParityLiveHostCaptureV2",
    "BacktestNotebookParityMeasurementContractV2",
    "BacktestNotebookParityMeasurementV2",
    "BacktestNotebookParityScenarioV2",
    "BacktestNotebookParitySourceFixturesV2",
    "NotebookParityAuthorityKindLiteralV2",
    "NotebookParityBenchmarkClassLiteralV2",
    "NotebookParityComparisonModeLiteralV2",
    "NotebookParityMeasurementFieldLiteralV2",
    "NotebookParityMeasurementSourceLiteralV2",
    "NotebookParityLiveHostCaptureStatusLiteralV2",
    "NotebookParityReferenceSourceKindLiteralV2",
    "NotebookParityRuntimeSurfaceLiteralV2",
    "NotebookParityStageBExecutionModeLiteralV2",
    "NotebookParityStageBProcessFallbackThresholdLiteralV2",
    "evaluate_backtest_notebook_parity_scenario_v2",
    "load_backtest_notebook_parity_benchmark_corpus_v2",
    "read_backtest_notebook_parity_benchmark_corpus_payload_v2",
    "serialize_backtest_notebook_parity_benchmark_corpus_payload_v2",
    "serialize_backtest_notebook_parity_live_host_captures_v2",
    "serialize_backtest_notebook_parity_measurements_v2",
    "validate_backtest_notebook_parity_benchmark_corpus_payload_v2",
]
