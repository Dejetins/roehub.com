from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.unit.contexts.backtest.application.services.v2 import (
    test_stage_a_shortlist_builder_v2 as stage_a_shortlist_builder_testkit,
)
from trading.contexts.backtest.application.services import (
    BacktestNotebookParityMeasurementV2,
    evaluate_backtest_notebook_parity_scenario_v2,
    load_backtest_notebook_parity_benchmark_corpus_v2,
    read_backtest_notebook_parity_benchmark_corpus_payload_v2,
    serialize_backtest_notebook_parity_benchmark_corpus_payload_v2,
    serialize_backtest_notebook_parity_measurements_v2,
)
from trading.contexts.backtest.application.services import (
    numba_runtime_v1 as numba_runtime_module,
)
from trading.contexts.backtest.application.services.v2 import (
    stage_a_shortlist_builder_v2 as stage_a_shortlist_builder_module,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_CORPUS_FIXTURE_PATH = _FIXTURES_DIR / "backtest_notebook_parity_benchmark_corpus_v1.json"


def test_notebook_parity_benchmark_corpus_manifest_is_complete() -> None:
    """
    Verify the A1 notebook-parity corpus publishes the canonical benchmark classes and rules.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-engine-vnext.md
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_notebook_parity_benchmark_corpus_v1.json
      - src/trading/contexts/backtest/application/services/v2/
        notebook_parity_benchmark_corpus_v2.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        A1 establishes the measurement authority and committed comparison points without claiming
        that the current runtime already satisfies the future parity gates.
    Raises:
        AssertionError: If one canonical scenario, measurement field, or equal-thread-budget rule
            is missing.
    Side Effects:
        Reads the committed notebook-parity benchmark corpus fixture from the repository.
    """
    corpus = _load_notebook_parity_benchmark_corpus()

    assert corpus.milestone_id == "A1"
    assert corpus.scenario_order == ("nr2", "rg_ttr", "rg_alt")
    assert corpus.measurement_contract.required_fields == (
        "wall_clock_seconds",
        "cpu_time_seconds",
        "peak_rss_bytes",
        "numba_threads_used",
        "max_python_processes_seen",
        "stage_b_execution_mode",
        "stage_b_process_fallback_threshold",
        "exact_replay_count",
    )
    assert corpus.measurement_contract.system_scan_fields == (
        "peak_rss_bytes",
        "numba_threads_used",
        "max_python_processes_seen",
        "stage_b_execution_mode",
        "stage_b_process_fallback_threshold",
        "exact_replay_count",
    )
    assert corpus.source_fixtures.perf_smoke_harness == (
        "tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py"
    )
    assert corpus.source_fixtures.nr2_notebook_anchor.endswith(
        "02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb"
    )
    assert corpus.source_fixtures.rg_ttr_notebook_anchor.endswith(
        "01_run_322_btcusdt_1h_artifact_probe.ipynb"
    )
    assert corpus.equal_thread_budget_rule.literal == "equal thread budget"
    assert corpus.equal_thread_budget_rule.comparison_field == "numba_threads_used"
    assert corpus.equal_thread_budget_rule.same_host_required is True
    assert corpus.equal_thread_budget_rule.same_artifact_slot_required is True
    assert "notebook on 12 threads vs backend on 4 threads is invalid" in (
        corpus.equal_thread_budget_rule.invalid_examples
    )

    nr2 = corpus.scenario_for_id(scenario_id="nr2")
    assert nr2.benchmark_class == "NR2"
    assert nr2.comparison_mode == "notebook_parity"
    assert nr2.primary_metric == "total_return_pct"
    assert nr2.anchor_notebook == corpus.source_fixtures.nr2_notebook_anchor
    assert [point.reference_id for point in nr2.baseline_reference_points] == [
        "nr2_backend_macstudio_4_threads",
        "nr2_notebook_macstudio_4_threads",
        "nr2_notebook_macstudio_12_threads",
    ]
    assert nr2.baseline_reference_points[0].wall_clock_seconds == 181.3
    assert nr2.baseline_reference_points[0].max_python_processes_seen == 5
    assert nr2.baseline_reference_points[0].stage_b_execution_mode == "process_pool"
    assert nr2.baseline_reference_points[1].wall_clock_seconds == 7.54
    assert nr2.baseline_reference_points[2].wall_clock_seconds == 5.63
    assert [gate.gate_id for gate in nr2.acceptance_gates] == [
        "nr2_wall_clock_ratio",
        "nr2_peak_rss_ratio",
        "nr2_max_python_processes_seen",
        "nr2_stage_b_execution_mode",
        "nr2_stage_b_process_fallback_threshold",
    ]
    assert nr2.acceptance_gates[0].max_ratio == 1.18
    assert nr2.acceptance_gates[1].max_ratio == 1.35
    assert nr2.acceptance_gates[2].max_value == 1.0
    assert nr2.acceptance_gates[3].expected_value == "bypassed_no_risk"
    assert nr2.acceptance_gates[4].expected_value == "none"

    rg_ttr = corpus.scenario_for_id(scenario_id="rg_ttr")
    assert rg_ttr.benchmark_class == "RG-TTR"
    assert rg_ttr.comparison_mode == "notebook_parity"
    assert rg_ttr.anchor_notebook == corpus.source_fixtures.rg_ttr_notebook_anchor
    assert rg_ttr.baseline_reference_points[0].reference_id == (
        "rg_ttr_backend_default_single_process"
    )
    assert rg_ttr.baseline_reference_points[0].max_python_processes_seen == 1
    assert rg_ttr.baseline_reference_points[0].stage_b_execution_mode == "in_process"
    assert rg_ttr.baseline_reference_points[0].stage_b_process_fallback_threshold == "none"
    assert rg_ttr.baseline_reference_points[1].max_python_processes_seen == 1
    assert rg_ttr.baseline_reference_points[1].stage_b_execution_mode == "in_process"
    assert rg_ttr.baseline_reference_points[1].stage_b_process_fallback_threshold == "none"
    assert rg_ttr.acceptance_gates[0].max_ratio == 1.18
    assert rg_ttr.acceptance_gates[1].max_value == 1.0
    assert rg_ttr.acceptance_gates[2].expected_value == "in_process"
    assert rg_ttr.acceptance_gates[3].expected_value == "none"
    assert rg_ttr.acceptance_gates[4].max_value == 64.0

    rg_alt = corpus.scenario_for_id(scenario_id="rg_alt")
    assert rg_alt.benchmark_class == "RG-ALT"
    assert rg_alt.comparison_mode == "functional_baseline"
    assert rg_alt.supported_primary_metrics == (
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "sharpe_trades",
        "win_rate_pct",
    )
    assert rg_alt.baseline_reference_points[0].runtime_regression_ratio_limit == 1.1
    assert rg_alt.acceptance_gates[0].metric == "runtime_regression_ratio"
    assert rg_alt.acceptance_gates[0].max_ratio == 1.1


def test_notebook_parity_benchmark_corpus_serialization_is_byte_stable() -> None:
    """
    Verify the committed A1 notebook-parity corpus keeps canonical byte-stable JSON formatting.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_notebook_parity_benchmark_corpus_v1.json
      - src/trading/contexts/backtest/application/services/v2/
        notebook_parity_benchmark_corpus_v2.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Canonical formatting is part of the reviewable benchmark-corpus contract.
    Raises:
        AssertionError: If canonical serialization drifts from the committed fixture bytes.
    Side Effects:
        Reads one committed JSON fixture from the repository.
    """
    raw_payload = read_backtest_notebook_parity_benchmark_corpus_payload_v2(
        path=_CORPUS_FIXTURE_PATH
    )
    canonical_bytes = serialize_backtest_notebook_parity_benchmark_corpus_payload_v2(
        payload=raw_payload
    )

    assert canonical_bytes == _CORPUS_FIXTURE_PATH.read_bytes()


def test_notebook_parity_measurement_serialization_is_deterministic() -> None:
    """
    Verify runtime-shape measurement payloads serialize deterministically and include all fields.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-engine-vnext.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - src/trading/contexts/backtest/application/services/v2/
        notebook_parity_benchmark_corpus_v2.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Measurement serialization is part of the additive benchmark authority and should remain
        stable across local runs.
    Raises:
        AssertionError: If serialized measurement bytes drift or omit required system-scan
            fields.
    Side Effects:
        None.
    """
    measurements = (
        _build_measurement(
            scenario_id="nr2",
            benchmark_class="NR2",
            measurement_source="notebook",
            runtime_surface="notebook",
            wall_clock_seconds=7.54,
            cpu_time_seconds=6.91,
            peak_rss_bytes=1_024,
            numba_threads_used=4,
            max_python_processes_seen=1,
            stage_b_execution_mode="bypassed_no_risk",
            stage_b_process_fallback_threshold="none",
            exact_replay_count=48,
        ),
        _build_measurement(
            scenario_id="nr2",
            benchmark_class="NR2",
            measurement_source="backend",
            runtime_surface="sync",
            wall_clock_seconds=8.11,
            cpu_time_seconds=7.33,
            peak_rss_bytes=1_200,
            numba_threads_used=4,
            max_python_processes_seen=1,
            stage_b_execution_mode="bypassed_no_risk",
            stage_b_process_fallback_threshold="none",
            exact_replay_count=48,
        ),
    )

    serialized = serialize_backtest_notebook_parity_measurements_v2(
        measurements=measurements
    )

    assert serialized == (
        b'{\n'
        b'  "measurements": [\n'
        b"    {\n"
        b'      "scenario_id": "nr2",\n'
        b'      "benchmark_class": "NR2",\n'
        b'      "measurement_source": "notebook",\n'
        b'      "runtime_surface": "notebook",\n'
        b'      "host_label": "macstudio-class",\n'
        b'      "artifact_slot": "slot_a",\n'
        b'      "wall_clock_seconds": 7.54,\n'
        b'      "cpu_time_seconds": 6.91,\n'
        b'      "peak_rss_bytes": 1024,\n'
        b'      "numba_threads_used": 4,\n'
        b'      "max_python_processes_seen": 1,\n'
        b'      "stage_b_execution_mode": "bypassed_no_risk",\n'
        b'      "stage_b_process_fallback_threshold": "none",\n'
        b'      "exact_replay_count": 48\n'
        b"    },\n"
        b"    {\n"
        b'      "scenario_id": "nr2",\n'
        b'      "benchmark_class": "NR2",\n'
        b'      "measurement_source": "backend",\n'
        b'      "runtime_surface": "sync",\n'
        b'      "host_label": "macstudio-class",\n'
        b'      "artifact_slot": "slot_a",\n'
        b'      "wall_clock_seconds": 8.11,\n'
        b'      "cpu_time_seconds": 7.33,\n'
        b'      "peak_rss_bytes": 1200,\n'
        b'      "numba_threads_used": 4,\n'
        b'      "max_python_processes_seen": 1,\n'
        b'      "stage_b_execution_mode": "bypassed_no_risk",\n'
        b'      "stage_b_process_fallback_threshold": "none",\n'
        b'      "exact_replay_count": 48\n'
        b"    }\n"
        b"  ]\n"
        b"}\n"
    )


def test_stage_a_retained_frontier_memory_shape_is_observable_for_benchmarks() -> None:
    """
    Verify perf-smoke can observe the retained frontier `memory shape` improvement additively.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-engine-vnext.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_stage_a_shortlist_builder_v2.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        The retained frontier now stores only deterministic per-indicator row addresses, while the
        removed legacy contract would have retained one full `final_signal_row` value per bar.
    Raises:
        AssertionError: If the additive benchmark evidence no longer reflects the cutover.
    Side Effects:
        None.
    """
    retained_exact_candidates = (
        stage_a_shortlist_builder_module._RetainedExactCandidateV2(
            base_variant=cast(
                Any,
                SimpleNamespace(stage_a_index=0, base_variant_key="0" * 64),
            ),
            proxy_score=3.0,
            retained_address=stage_a_shortlist_builder_module._RetainedExactCandidateAddressV2(
                indicator_row_indexes=(1, 4, 7)
            ),
        ),
        stage_a_shortlist_builder_module._RetainedExactCandidateV2(
            base_variant=cast(
                Any,
                SimpleNamespace(stage_a_index=1, base_variant_key="1" * 64),
            ),
            proxy_score=2.0,
            retained_address=stage_a_shortlist_builder_module._RetainedExactCandidateAddressV2(
                indicator_row_indexes=(1, 5, 6)
            ),
        ),
    )

    memory_shape = (
        stage_a_shortlist_builder_module.describe_stage_a_retained_frontier_memory_shape_v2(
            retained_exact_candidates=retained_exact_candidates,
            signal_bar_count=4_096,
        )
    )

    assert memory_shape.candidate_count == 2
    assert memory_shape.indicator_count_per_candidate == 3
    assert memory_shape.retained_address_value_count == 6
    assert memory_shape.signal_bar_count == 4_096
    assert memory_shape.legacy_final_signal_value_count == 8_192
    assert round(memory_shape.legacy_to_address_value_ratio or 0.0, 2) == 1365.33
    assert memory_shape.stores_full_final_signal_rows is False


def test_stage_a_streaming_exact_runtime_shape_is_observable_for_benchmarks() -> None:
    """
    Verify perf-smoke can distinguish Stage A streaming exact scoring from deferred replay.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_stage_a_shortlist_builder_v2.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stage A exact work now runs trade-list-first over each retained chunk immediately, so the
        additive runtime shape should report streaming exact scoring with no deferred replay.
    Raises:
        AssertionError: If the additive runtime-shape contract drifts from the streaming cutover.
    Side Effects:
        None.
    """
    runtime_shape = (
        stage_a_shortlist_builder_module.describe_stage_a_streaming_exact_runtime_shape_v2(
            retained_chunk_sizes=(4, 2),
        )
    )

    assert runtime_shape.exact_scoring_mode == "streaming exact scoring"
    assert runtime_shape.retained_chunk_count == 2
    assert runtime_shape.retained_candidate_count == 6
    assert runtime_shape.max_retained_chunk_size == 4
    assert runtime_shape.deferred_replay_count == 0
    assert runtime_shape.execution_shape == "single-process parallel Stage A"
    assert runtime_shape.frontier_compute_mode == "kernel-driven"
    assert runtime_shape.stage_a_workers is None
    assert runtime_shape.numba_threads_used is None


def test_stage_a_streaming_exact_runtime_shape_tracks_live_stage_a_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify perf-smoke derives streaming exact shape from the live Stage A path.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_stage_a_shortlist_builder_v2.py
    Args:
        monkeypatch: pytest fixture used to record retained chunk merges on the live Stage A path.
    Returns:
        None.
    Assumptions:
        Perf-smoke should fail if the active Stage A path stops merging retained chunks directly
        into the shortlist heap and returns to deferred replay.
    Raises:
        AssertionError: If the live Stage A path no longer exposes the expected streaming shape.
    Side Effects:
        Monkeypatches the retained chunk heap-merge helper during one in-memory Stage A run.
    """
    retained_chunk_sizes: list[int] = []
    observed_numba_threads: list[int] = []
    original_method = (
        stage_a_shortlist_builder_module.BacktestStageAShortlistBuilderV2._merge_retained_exact_payload_chunk_into_heap
    )
    original_aggregate = stage_a_shortlist_builder_module.aggregate_ordered_final_signal_rows_v2

    def _recording_method(self: Any, **kwargs: Any) -> None:
        """
        Record one retained chunk size before delegating to the live exact merge helper.

        Args:
            self: Stage A shortlist builder under test.
            **kwargs: Exact merge keyword arguments including `chunk_variants`.
        Returns:
            None.
        Assumptions:
            Each merge call corresponds to one retained chunk selected by combo proxy prefilter.
        Raises:
            None.
        Side Effects:
            Appends one retained chunk size to the in-memory log.
        """
        retained_chunk_sizes.append(len(kwargs["chunk_variants"]))
        original_method(self, **kwargs)

    def _recording_aggregate(**kwargs: Any) -> Any:
        """
        Record the live Stage A `numba_threads_used` value before aggregation.

        Args:
            **kwargs: Ordered aggregation keyword arguments forwarded to the live helper.
        Returns:
            Any: Live aggregated `final_signal` matrix.
        Assumptions:
            The kernel-driven Stage A frontier should aggregate inside the active in-process Numba
            thread scope, making the live thread budget observable to perf-smoke.
        Raises:
            None.
        Side Effects:
            Appends one observed thread count to the in-memory log.
        """
        observed_numba_threads.append(
            numba_runtime_module.current_backtest_numba_threads_v1()
        )
        return original_aggregate(**kwargs)

    monkeypatch.setattr(
        stage_a_shortlist_builder_module.BacktestStageAShortlistBuilderV2,
        "_merge_retained_exact_payload_chunk_into_heap",
        _recording_method,
    )
    monkeypatch.setattr(
        stage_a_shortlist_builder_module,
        "aggregate_ordered_final_signal_rows_v2",
        _recording_aggregate,
    )

    parallelism = numba_runtime_module.BacktestStageAParallelismConfigV1(
        stage_a_workers=2,
        numba_threads=2,
    )

    shortlist = stage_a_shortlist_builder_module.BacktestStageAShortlistBuilderV2(
        price_arrays_loader=stage_a_shortlist_builder_testkit._ComboProxyPriceLoader(),
        signal_matrix_loader=stage_a_shortlist_builder_testkit._combo_proxy_signal_loader(),
    ).build_shortlist(
        grid_context=cast(Any, stage_a_shortlist_builder_testkit._combo_proxy_grid_context()),
        artifact_context=cast(
            Any,
            stage_a_shortlist_builder_testkit._combo_proxy_artifact_context(),
        ),
        target_time_range=stage_a_shortlist_builder_testkit._combo_proxy_target_time_range(),
        shortlist_limit=2,
        batch_size=6,
        parallelism=parallelism,
    )

    assert observed_numba_threads
    runtime_shape = (
        stage_a_shortlist_builder_module.describe_stage_a_streaming_exact_runtime_shape_v2(
            retained_chunk_sizes=tuple(retained_chunk_sizes),
            stage_a_workers=parallelism.stage_a_workers,
            numba_threads_used=max(observed_numba_threads),
        )
    )

    assert retained_chunk_sizes == [4, 2]
    assert tuple(row.base_variant.stage_a_index for row in shortlist) == (0, 1)
    assert tuple(
        row.retained_exact_payload.memory_shape_bucket
        for row in shortlist
        if row.retained_exact_payload is not None
    ) == ("compact_trade_arrays", "compact_trade_arrays")
    assert all(
        row.retained_exact_payload is not None
        and row.retained_exact_payload.trade_count > 0
        and not hasattr(row.retained_exact_payload, "final_signal_row")
        for row in shortlist
    )
    assert runtime_shape.exact_scoring_mode == "streaming exact scoring"
    assert runtime_shape.retained_chunk_count == 2
    assert runtime_shape.retained_candidate_count == 6
    assert runtime_shape.max_retained_chunk_size == 4
    assert runtime_shape.deferred_replay_count == 0
    assert runtime_shape.execution_shape == "single-process parallel Stage A"
    assert runtime_shape.frontier_compute_mode == "kernel-driven"
    assert runtime_shape.stage_a_workers == 2
    assert runtime_shape.numba_threads_used == 2


def test_notebook_parity_comparison_helper_enforces_equal_thread_budget_and_shape_gates() -> None:
    """
    Verify the comparison helper accepts good parity samples and rejects budget/shape regressions.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - src/trading/contexts/backtest/application/services/v2/
        notebook_parity_benchmark_corpus_v2.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        A1 perf smoke verifies that the measurement authority can both pass and fail determinis-
        tically before later prompts wire it into the live runtime.
    Raises:
        AssertionError: If equal-thread-budget checks or runtime-shape gates stop working.
    Side Effects:
        None.
    """
    corpus = _load_notebook_parity_benchmark_corpus()
    rule = corpus.equal_thread_budget_rule

    nr2 = corpus.scenario_for_id(scenario_id="nr2")
    notebook_reference = _build_measurement(
        scenario_id="nr2",
        benchmark_class="NR2",
        measurement_source="notebook",
        runtime_surface="notebook",
        wall_clock_seconds=10.0,
        cpu_time_seconds=8.2,
        peak_rss_bytes=1_000,
        numba_threads_used=4,
        max_python_processes_seen=1,
        stage_b_execution_mode="bypassed_no_risk",
        stage_b_process_fallback_threshold="none",
        exact_replay_count=24,
    )
    backend_candidate = _build_measurement(
        scenario_id="nr2",
        benchmark_class="NR2",
        measurement_source="backend",
        runtime_surface="sync",
        wall_clock_seconds=11.7,
        cpu_time_seconds=9.1,
        peak_rss_bytes=1_340,
        numba_threads_used=4,
        max_python_processes_seen=1,
        stage_b_execution_mode="bypassed_no_risk",
        stage_b_process_fallback_threshold="none",
        exact_replay_count=24,
    )
    passing_comparison = evaluate_backtest_notebook_parity_scenario_v2(
        scenario=nr2,
        equal_thread_budget_rule=rule,
        candidate=backend_candidate,
        reference=notebook_reference,
    )

    assert passing_comparison.thread_budget_aligned is True
    assert round(passing_comparison.wall_clock_ratio, 2) == 1.17
    assert round(passing_comparison.peak_rss_ratio, 2) == 1.34
    assert passing_comparison.failing_gate_ids == ()
    assert passing_comparison.rule_violations == ()
    assert passing_comparison.passed is True

    backend_regressed = _build_measurement(
        scenario_id="nr2",
        benchmark_class="NR2",
        measurement_source="backend",
        runtime_surface="sync",
        wall_clock_seconds=12.5,
        cpu_time_seconds=9.9,
        peak_rss_bytes=1_500,
        numba_threads_used=12,
        max_python_processes_seen=4,
        stage_b_execution_mode="process_pool",
        stage_b_process_fallback_threshold="stage_b_variants_total",
        exact_replay_count=200,
    )
    failing_comparison = evaluate_backtest_notebook_parity_scenario_v2(
        scenario=nr2,
        equal_thread_budget_rule=rule,
        candidate=backend_regressed,
        reference=notebook_reference,
    )

    assert failing_comparison.thread_budget_aligned is False
    assert "equal_thread_budget" in failing_comparison.rule_violations
    assert failing_comparison.failing_gate_ids == (
        "nr2_wall_clock_ratio",
        "nr2_peak_rss_ratio",
        "nr2_max_python_processes_seen",
        "nr2_stage_b_execution_mode",
        "nr2_stage_b_process_fallback_threshold",
    )
    assert failing_comparison.passed is False

    rg_ttr = corpus.scenario_for_id(scenario_id="rg_ttr")
    rg_ttr_reference = _build_measurement(
        scenario_id="rg_ttr",
        benchmark_class="RG-TTR",
        measurement_source="notebook",
        runtime_surface="notebook",
        wall_clock_seconds=20.0,
        cpu_time_seconds=18.0,
        peak_rss_bytes=2_000,
        numba_threads_used=4,
        max_python_processes_seen=1,
        stage_b_execution_mode="in_process",
        stage_b_process_fallback_threshold="none",
        exact_replay_count=30,
    )
    rg_ttr_fast_path_candidate = _build_measurement(
        scenario_id="rg_ttr",
        benchmark_class="RG-TTR",
        measurement_source="backend",
        runtime_surface="sync",
        wall_clock_seconds=21.0,
        cpu_time_seconds=19.0,
        peak_rss_bytes=2_050,
        numba_threads_used=4,
        max_python_processes_seen=1,
        stage_b_execution_mode="in_process",
        stage_b_process_fallback_threshold="none",
        exact_replay_count=64,
    )
    rg_ttr_fast_path_comparison = evaluate_backtest_notebook_parity_scenario_v2(
        scenario=rg_ttr,
        equal_thread_budget_rule=rule,
        candidate=rg_ttr_fast_path_candidate,
        reference=rg_ttr_reference,
    )

    assert rg_ttr_fast_path_comparison.failing_gate_ids == ()
    assert rg_ttr_fast_path_comparison.passed is True
    assert rg_ttr_fast_path_candidate.stage_b_process_fallback_threshold == "none"

    rg_ttr_regressed = _build_measurement(
        scenario_id="rg_ttr",
        benchmark_class="RG-TTR",
        measurement_source="backend",
        runtime_surface="sync",
        wall_clock_seconds=21.0,
        cpu_time_seconds=19.0,
        peak_rss_bytes=2_100,
        numba_threads_used=4,
        max_python_processes_seen=5,
        stage_b_execution_mode="process_pool",
        stage_b_process_fallback_threshold="stage_b_variants_total",
        exact_replay_count=500,
    )
    rg_ttr_comparison = evaluate_backtest_notebook_parity_scenario_v2(
        scenario=rg_ttr,
        equal_thread_budget_rule=rule,
        candidate=rg_ttr_regressed,
        reference=rg_ttr_reference,
    )

    assert rg_ttr_comparison.failing_gate_ids == (
        "rg_ttr_max_python_processes_seen",
        "rg_ttr_stage_b_execution_mode",
        "rg_ttr_stage_b_process_fallback_threshold",
        "rg_ttr_exact_replay_count",
    )
    assert rg_ttr_comparison.passed is False
    assert rg_ttr_regressed.stage_b_process_fallback_threshold == "stage_b_variants_total"


def test_rg_ttr_exact_replay_bound_is_isolated_and_benchmark_visible() -> None:
    """
    Verify the RG-TTR benchmark contract can fail on `exact_replay_count` alone.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Finalist-only exact replay should stay within the committed RG-TTR upper bound, and the
        benchmark layer must expose that count directly without relying on process-mode failures.
    Raises:
        AssertionError: If the RG-TTR gate stops isolating replay-count regressions.
    Side Effects:
        None.
    """
    corpus = _load_notebook_parity_benchmark_corpus()
    rg_ttr = corpus.scenario_for_id(scenario_id="rg_ttr")
    rule = corpus.equal_thread_budget_rule
    notebook_reference = _build_measurement(
        scenario_id="rg_ttr",
        benchmark_class="RG-TTR",
        measurement_source="notebook",
        runtime_surface="notebook",
        wall_clock_seconds=20.0,
        cpu_time_seconds=18.0,
        peak_rss_bytes=2_000,
        numba_threads_used=4,
        max_python_processes_seen=1,
        stage_b_execution_mode="in_process",
        stage_b_process_fallback_threshold="none",
        exact_replay_count=30,
    )
    finalist_only_candidate = _build_measurement(
        scenario_id="rg_ttr",
        benchmark_class="RG-TTR",
        measurement_source="backend",
        runtime_surface="sync",
        wall_clock_seconds=21.0,
        cpu_time_seconds=19.0,
        peak_rss_bytes=2_020,
        numba_threads_used=4,
        max_python_processes_seen=1,
        stage_b_execution_mode="in_process",
        stage_b_process_fallback_threshold="none",
        exact_replay_count=64,
    )
    finalist_only_comparison = evaluate_backtest_notebook_parity_scenario_v2(
        scenario=rg_ttr,
        equal_thread_budget_rule=rule,
        candidate=finalist_only_candidate,
        reference=notebook_reference,
    )

    assert finalist_only_comparison.failing_gate_ids == ()
    assert finalist_only_comparison.passed is True

    shortlist_breadth_regressed = _build_measurement(
        scenario_id="rg_ttr",
        benchmark_class="RG-TTR",
        measurement_source="backend",
        runtime_surface="sync",
        wall_clock_seconds=21.0,
        cpu_time_seconds=19.0,
        peak_rss_bytes=2_020,
        numba_threads_used=4,
        max_python_processes_seen=1,
        stage_b_execution_mode="in_process",
        stage_b_process_fallback_threshold="none",
        exact_replay_count=65,
    )
    shortlist_breadth_comparison = evaluate_backtest_notebook_parity_scenario_v2(
        scenario=rg_ttr,
        equal_thread_budget_rule=rule,
        candidate=shortlist_breadth_regressed,
        reference=notebook_reference,
    )

    assert shortlist_breadth_comparison.failing_gate_ids == ("rg_ttr_exact_replay_count",)
    assert shortlist_breadth_comparison.passed is False


def test_rg_alt_functional_baseline_guardrail_is_evaluable() -> None:
    """
    Verify the RG-ALT functional guardrail can compare backend runs against backend baselines.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - src/trading/contexts/backtest/application/services/v2/
        notebook_parity_benchmark_corpus_v2.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        RG-ALT keeps functional-first coverage in the first wave, so the benchmark authority
        compares backend candidates against backend baselines rather than notebook runtime.
    Raises:
        AssertionError: If the functional regression guard stops catching >10% regressions.
    Side Effects:
        None.
    """
    corpus = _load_notebook_parity_benchmark_corpus()
    rg_alt = corpus.scenario_for_id(scenario_id="rg_alt")
    rule = corpus.equal_thread_budget_rule
    backend_baseline = _build_measurement(
        scenario_id="rg_alt",
        benchmark_class="RG-ALT",
        measurement_source="backend",
        runtime_surface="sync",
        wall_clock_seconds=30.0,
        cpu_time_seconds=24.0,
        peak_rss_bytes=3_000,
        numba_threads_used=4,
        max_python_processes_seen=2,
        stage_b_execution_mode="process_pool",
        stage_b_process_fallback_threshold="stage_b_variants_total",
        exact_replay_count=64,
    )
    backend_candidate_ok = _build_measurement(
        scenario_id="rg_alt",
        benchmark_class="RG-ALT",
        measurement_source="backend",
        runtime_surface="sync",
        wall_clock_seconds=32.7,
        cpu_time_seconds=24.5,
        peak_rss_bytes=3_050,
        numba_threads_used=4,
        max_python_processes_seen=2,
        stage_b_execution_mode="process_pool",
        stage_b_process_fallback_threshold="stage_b_variants_total",
        exact_replay_count=64,
    )
    ok_comparison = evaluate_backtest_notebook_parity_scenario_v2(
        scenario=rg_alt,
        equal_thread_budget_rule=rule,
        candidate=backend_candidate_ok,
        reference=backend_baseline,
    )

    assert round(ok_comparison.runtime_regression_ratio, 2) == 1.09
    assert ok_comparison.failing_gate_ids == ()
    assert ok_comparison.passed is True

    backend_candidate_bad = _build_measurement(
        scenario_id="rg_alt",
        benchmark_class="RG-ALT",
        measurement_source="backend",
        runtime_surface="sync",
        wall_clock_seconds=33.6,
        cpu_time_seconds=25.1,
        peak_rss_bytes=3_100,
        numba_threads_used=4,
        max_python_processes_seen=2,
        stage_b_execution_mode="process_pool",
        stage_b_process_fallback_threshold="stage_b_variants_total",
        exact_replay_count=64,
    )
    bad_comparison = evaluate_backtest_notebook_parity_scenario_v2(
        scenario=rg_alt,
        equal_thread_budget_rule=rule,
        candidate=backend_candidate_bad,
        reference=backend_baseline,
    )

    assert round(bad_comparison.runtime_regression_ratio, 2) == 1.12
    assert bad_comparison.failing_gate_ids == ("rg_alt_runtime_regression_ratio",)
    assert bad_comparison.passed is False


@pytest.mark.parametrize(
    "primary_metric",
    (
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "sharpe_trades",
        "win_rate_pct",
    ),
)
def test_no_risk_alt_metric_runtime_guardrail_caps_regression(
    primary_metric: str,
) -> None:
    """
    Verify no-risk alternative metrics keep the Stage-A bypass and stay within a 10% runtime cap.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - data_load/promts/backtest_emgine_vnext/
        28_codex_backtest_engine_vnext_parity_c1_no_risk_terminal_path_prompt.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
    Args:
        primary_metric: Supported no-risk alternative ranking metric under guard.
    Returns:
        None.
    Assumptions:
        Alternative no-risk metrics should remain on the direct Stage-A terminal path, so backend
        regression checks compare backend-vs-backend measurements with identical thread budgets.
    Raises:
        AssertionError: If the no-risk bypass drifts from `bypassed_no_risk` or runtime grows by
            more than 10% versus the backend baseline.
    Side Effects:
        None.
    """
    baseline = _build_measurement(
        scenario_id="nr2",
        benchmark_class="NR2",
        measurement_source="backend",
        runtime_surface="sync",
        wall_clock_seconds=9.0,
        cpu_time_seconds=7.6,
        peak_rss_bytes=1_750,
        numba_threads_used=4,
        max_python_processes_seen=1,
        stage_b_execution_mode="bypassed_no_risk",
        stage_b_process_fallback_threshold="none",
        exact_replay_count=48,
    )
    candidate_ok = _build_measurement(
        scenario_id="nr2",
        benchmark_class="NR2",
        measurement_source="backend",
        runtime_surface="sync",
        wall_clock_seconds=9.8,
        cpu_time_seconds=7.9,
        peak_rss_bytes=1_790,
        numba_threads_used=4,
        max_python_processes_seen=1,
        stage_b_execution_mode="bypassed_no_risk",
        stage_b_process_fallback_threshold="none",
        exact_replay_count=48,
    )
    candidate_bad = _build_measurement(
        scenario_id="nr2",
        benchmark_class="NR2",
        measurement_source="backend",
        runtime_surface="sync",
        wall_clock_seconds=10.2,
        cpu_time_seconds=8.2,
        peak_rss_bytes=1_820,
        numba_threads_used=4,
        max_python_processes_seen=1,
        stage_b_execution_mode="bypassed_no_risk",
        stage_b_process_fallback_threshold="none",
        exact_replay_count=48,
    )
    ok_ratio = candidate_ok.wall_clock_seconds / baseline.wall_clock_seconds
    bad_ratio = candidate_bad.wall_clock_seconds / baseline.wall_clock_seconds

    assert primary_metric in (
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "sharpe_trades",
        "win_rate_pct",
    )
    assert candidate_ok.numba_threads_used == baseline.numba_threads_used
    assert candidate_ok.stage_b_execution_mode == "bypassed_no_risk"
    assert round(ok_ratio, 2) == 1.09
    assert ok_ratio <= 1.10
    assert candidate_bad.stage_b_execution_mode == "bypassed_no_risk"
    assert round(bad_ratio, 2) == 1.13
    assert bad_ratio > 1.10


def _load_notebook_parity_benchmark_corpus():
    """
    Load the committed notebook-parity benchmark corpus used by A1 perf-smoke coverage.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_notebook_parity_benchmark_corpus_v1.json
    Args:
        None.
    Returns:
        BacktestNotebookParityBenchmarkCorpusV2: Parsed benchmark corpus contract.
    Assumptions:
        The corpus is lightweight enough to load in every deterministic perf-smoke test that
        needs benchmark authority metadata.
    Raises:
        ValueError: If the committed benchmark corpus violates its typed contract.
    Side Effects:
        Reads one committed JSON fixture from the repository.
    """
    return load_backtest_notebook_parity_benchmark_corpus_v2(path=_CORPUS_FIXTURE_PATH)


def _build_measurement(
    *,
    scenario_id: str,
    benchmark_class: str,
    measurement_source: str,
    runtime_surface: str,
    wall_clock_seconds: float,
    cpu_time_seconds: float,
    peak_rss_bytes: int,
    numba_threads_used: int,
    max_python_processes_seen: int,
    stage_b_execution_mode: str,
    exact_replay_count: int,
    stage_b_process_fallback_threshold: str = "none",
) -> BacktestNotebookParityMeasurementV2:
    """
    Build one deterministic notebook-parity runtime-shape measurement for perf-smoke tests.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-engine-vnext.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - src/trading/contexts/backtest/application/services/v2/
        notebook_parity_benchmark_corpus_v2.py
    Args:
        scenario_id: Stable benchmark scenario identifier.
        benchmark_class: Canonical benchmark class literal.
        measurement_source: Measurement source literal.
        runtime_surface: Runtime surface literal.
        wall_clock_seconds: Elapsed wall time for the measurement sample.
        cpu_time_seconds: CPU time for the measurement sample.
        peak_rss_bytes: Peak resident-set-size value in bytes.
        numba_threads_used: Numba thread budget used by the sample.
        max_python_processes_seen: Maximum Python process count observed for the sample.
        stage_b_execution_mode: Stage B execution mode observed for the sample.
        stage_b_process_fallback_threshold:
            Explicit workload threshold that activated the non-default Stage B fallback path, or
            `none` when the run stayed in-process.
        exact_replay_count: Exact replay count observed for the sample.
    Returns:
        BacktestNotebookParityMeasurementV2: Immutable measurement payload.
    Assumptions:
        The perf-smoke harness uses `macstudio-class` and `slot_a` as deterministic placeholders
        for the canonical equal-thread-budget comparisons.
    Raises:
        ValueError: If the typed measurement payload rejects one provided field.
    Side Effects:
        None.
    """
    return BacktestNotebookParityMeasurementV2(
        scenario_id=scenario_id,
        benchmark_class=benchmark_class,  # type: ignore[arg-type]
        measurement_source=measurement_source,  # type: ignore[arg-type]
        runtime_surface=runtime_surface,  # type: ignore[arg-type]
        host_label="macstudio-class",
        artifact_slot="slot_a",
        wall_clock_seconds=wall_clock_seconds,
        cpu_time_seconds=cpu_time_seconds,
        peak_rss_bytes=peak_rss_bytes,
        numba_threads_used=numba_threads_used,
        max_python_processes_seen=max_python_processes_seen,
        stage_b_execution_mode=stage_b_execution_mode,  # type: ignore[arg-type]
        stage_b_process_fallback_threshold=(
            stage_b_process_fallback_threshold  # type: ignore[arg-type]
        ),
        exact_replay_count=exact_replay_count,
    )
