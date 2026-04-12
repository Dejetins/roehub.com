from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trading.contexts.backtest.application.services.v2 import (
    BacktestArtifactBackedStageBScorerV2,
    StageBBestCellReplayCaseV2,
    StageBEntryMappingCaseV2,
    StageBTradeExitCaseV2,
    StageBTradeListCaseV2,
    artifact_backed_stage_b_scorer_v2 as stage_b_scorer_module,
    execute_stage_b_golden_case_v2,
    load_backtest_runtime_acceleration_benchmark_corpus_v2,
    load_stage_b_golden_fixture_catalog_v2,
    map_bar_close_1m_idx_to_entry_exec_v2,
    read_stage_b_golden_fixture_payload_v2,
    serialize_stage_b_golden_fixture_payload_v2,
    validate_stage_b_golden_fixture_payload_v2,
)
from trading.contexts.backtest.application.services.v2.stage_b_golden_fixtures_v2 import (
    load_stage_b_best_cell_replay_reference_case_v2,
)
from trading.contexts.backtest.application.services.v2.trade_compactor_kernel import (
    StageACompactExactPayloadV2,
)
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "stage_b_golden_fixtures_v2.json"
_PERF_MANIFEST_PATH = (
    Path(__file__).resolve().parents[6]
    / "perf_smoke"
    / "contexts"
    / "backtest"
    / "fixtures"
    / "r5_stage_b_golden_cases.json"
)
_BENCHMARK_CORPUS_PATH = (
    Path(__file__).resolve().parents[6]
    / "perf_smoke"
    / "contexts"
    / "backtest"
    / "fixtures"
    / "backtest_runtime_acceleration_benchmark_corpus_v1.json"
)


def test_stage_b_fast_path_stays_enabled_with_retained_exact_payload_for_total_return_pct() -> None:
    """
    Verify retained_exact_payload warming does not disable the fast Stage B path for breadth work.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        RG-TTR breadth ranking with `primary_metric=total_return_pct` must keep using the fast
        Stage B path even after Stage A primes `retained_exact_payload` for finalist authority.
    Raises:
        AssertionError: If retained payload re-disables fast-path breadth scoring.
    Side Effects:
        Populates one in-memory scorer cache with an empty retained payload seed.
    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
    """
    scorer = object.__new__(BacktestArtifactBackedStageBScorerV2)
    scorer._stage_a_payload_cache_by_base_variant_key = {}
    scorer._local_exec_open = np.asarray((100.0,), dtype=np.float64)
    scorer._local_exec_close = np.asarray((100.0,), dtype=np.float64)
    scorer._sentinel_index = 1
    scorer._execution_params = ExecutionParamsV1(
        direction_mode="long-only",
        sizing_mode="fixed_quote",
        init_cash_quote=1_000.0,
        fixed_quote=100.0,
        safe_profit_percent=0.0,
        fee_pct=0.0,
        slippage_pct=0.0,
    )
    scorer._close_on_end = True
    scorer._ranking_primary_by_stage = {}
    scorer._base_variant_key_v2 = (
        lambda *, indicator_variant_key, signal_params: "base-variant-key"
    )

    scorer.prime_retained_exact_payload(
        indicator_variant_key="indicator-variant-key",
        signal_params={},
        retained_exact_payload=StageACompactExactPayloadV2(
            entry_signal_idx=np.asarray((), dtype=np.int64),
            entry_exec_idx=np.asarray((), dtype=np.int64),
            direction=np.asarray((), dtype=np.int8),
            sig_exit_signal_idx=np.asarray((), dtype=np.int64),
            sig_exit_exec_idx=np.asarray((), dtype=np.int64),
        ),
    )
    scorer.configure_stage_ranking_context(
        stage="stage_b",
        primary_metric="total_return_pct",
    )
    scorer._resolve_risk_level_indexes_v2 = lambda *, risk_params: (0, 0)
    scorer._fast_stage_b_search_for_base_variant_v2 = (
        lambda *, indicator_selections, signal_params, base_variant_key: SimpleNamespace(
            total_return_pct=np.asarray(((17.25,),), dtype=np.float64),
            base_variant_key=base_variant_key,
            retained_exact_payload="present",
        )
    )
    scorer._exact_stage_b_cell_cache_v2 = (
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "retained_exact_payload must not disable the fast Stage B path for breadth scoring"
            )
        )
    )

    metrics = scorer.score_variant_metric(
        stage="stage_b",
        candles=SimpleNamespace(),
        indicator_selections=(),
        signal_params={},
        risk_params={
            "tp_enabled": True,
            "tp_pct": 1.0,
            "sl_enabled": True,
            "sl_pct": 1.0,
        },
        indicator_variant_key="indicator-variant-key",
        variant_key="stage-b-variant-key",
    )

    assert scorer._stage_a_payload_cache_by_base_variant_key["base-variant-key"].compact_trades == ()
    assert metrics["total_return_pct"] == 17.25
    assert metrics["Total Return [%]"] == 17.25


def test_stage_b_details_path_keeps_exact_authority_with_retained_exact_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify finalist details still use exact replay even when breadth ranking takes the fast path.

    Args:
        monkeypatch: Pytest fixture used to replace exact-detail collaborators with local stubs.
    Returns:
        None.
    Assumptions:
        This prompt only restores the fast Stage B path for RG-TTR breadth ranking; finalist
        detail authority must remain exact.
    Raises:
        AssertionError: If the details path stops calling the exact replay cache.
    Side Effects:
        Replaces two scorer-module helpers for the duration of the test.
    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
    """
    scorer = object.__new__(BacktestArtifactBackedStageBScorerV2)
    scorer._ranking_primary_by_stage = {}
    scorer._report_target_slice = slice(0, 1)
    scorer._execution_params = ExecutionParamsV1(
        direction_mode="long-only",
        sizing_mode="fixed_quote",
        init_cash_quote=1_000.0,
        fixed_quote=100.0,
        safe_profit_percent=0.0,
        fee_pct=0.0,
        slippage_pct=0.0,
    )
    scorer._local_exec_open = np.asarray((100.0,), dtype=np.float64)
    scorer._local_exec_close = np.asarray((101.0,), dtype=np.float64)
    scorer._local_hit_times = SimpleNamespace(
        tp_values=np.asarray((0.01,), dtype=np.float32),
        sl_values=np.asarray((0.01,), dtype=np.float32),
    )
    scorer._base_variant_key_v2 = (
        lambda *, indicator_variant_key, signal_params: "base-variant-key"
    )
    exact_calls: list[str] = []
    scorer._exact_stage_b_cell_cache_v2 = (
        lambda **kwargs: exact_calls.append(kwargs["variant_key"])
        or SimpleNamespace(replay="exact-replay", metrics="exact-metrics")
    )
    scorer.configure_stage_ranking_context(
        stage="stage_b",
        primary_metric="total_return_pct",
    )
    monkeypatch.setattr(
        stage_b_scorer_module,
        "stage_b_metrics_to_ranking_payload_v2",
        lambda *, metrics: {
            "total_return_pct": 9.5,
            "trade_count": 3.0,
        },
    )
    monkeypatch.setattr(
        stage_b_scorer_module,
        "build_execution_outcome_from_replay_v2",
        lambda **kwargs: SimpleNamespace(authority="exact"),
    )

    details = scorer.score_variant_with_details(
        stage="stage_b",
        candles=SimpleNamespace(),
        indicator_selections=(),
        signal_params={},
        risk_params={
            "tp_enabled": True,
            "tp_pct": 1.0,
            "sl_enabled": True,
            "sl_pct": 1.0,
        },
        indicator_variant_key="indicator-variant-key",
        variant_key="finalist-variant-key",
    )

    assert exact_calls == ["finalist-variant-key"]
    assert details.metrics["total_return_pct"] == 9.5
    assert details.metrics["trade_count"] == 3.0
    assert details.execution_outcome.authority == "exact"


def test_stage_b_golden_fixture_catalog_executes_all_cases_deterministically() -> None:
    """
    Verify the committed Stage B golden catalog executes every case with exact expected outputs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R5-03 fixtures are the canonical notebook-independent validation baseline for Stage B
        `signal_tf + 1m_risk` semantics.
    Raises:
        AssertionError: If one committed case drifts from the executable local oracle.
    Side Effects:
        Reads one JSON fixture catalog from repository.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
    """
    catalog = load_stage_b_golden_fixture_catalog_v2(path=_FIXTURE_PATH)

    assert catalog.case_order == (
        "entry_mapping_request_tf_to_1m",
        "earliest_signal_exit_mapping",
        "tp_sl_earliest_hit",
        "signal_exit_wins_equal_bar_over_tp_sl_tie",
        "sl_wins_tp_tie",
        "exact_best_cell_replay_metrics",
    )

    for case in catalog.cases:
        _assert_case_result_matches_expected(case=case)


def test_stage_b_golden_fixture_catalog_exposes_best_cell_reference_case_for_self_check() -> None:
    """
    Verify the golden catalog keeps one best-cell reference case usable for a bounded subset.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The repository uses the committed best-cell replay fixture as the canonical anchor for the
        reference-vs-fast self-check, so the case must keep enough trades and grid levels to slice
        a smaller deterministic subset.
    Raises:
        AssertionError: If the catalog no longer exposes a usable best-cell reference case.
    Side Effects:
        Reads one JSON fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
    Related:
      - tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
    """
    best_cell_case = load_stage_b_best_cell_replay_reference_case_v2(path=_FIXTURE_PATH)

    assert isinstance(best_cell_case, StageBBestCellReplayCaseV2)
    assert best_cell_case.case_id == "exact_best_cell_replay_metrics"
    assert len(best_cell_case.compact_trades) >= 2
    assert len(best_cell_case.level_factors.long_tp) >= 2
    assert len(best_cell_case.level_factors.long_sl) >= 2
    assert "exact best-cell replay" in best_cell_case.coverage
    assert "metrics over compact trades" in best_cell_case.coverage


def test_stage_b_golden_fixture_catalog_serialization_is_byte_stable() -> None:
    """
    Verify the committed Stage B golden catalog keeps byte-stable canonical JSON formatting.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Catalog ordering and formatting are part of the reviewable fixture contract.
    Raises:
        AssertionError: If canonical serialization or the published SHA drift unexpectedly.
    Side Effects:
        Reads the unit fixture catalog and perf-smoke manifest from repository.
    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json
      - tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json
    """
    raw_payload = read_stage_b_golden_fixture_payload_v2(path=_FIXTURE_PATH)
    canonical_bytes = serialize_stage_b_golden_fixture_payload_v2(payload=raw_payload)
    perf_manifest = read_stage_b_golden_fixture_payload_v2(path=_PERF_MANIFEST_PATH)

    assert canonical_bytes == _FIXTURE_PATH.read_bytes()
    assert sha256(canonical_bytes).hexdigest() == perf_manifest["contract_fixture_sha256"]


def test_stage_b_golden_fixture_catalog_is_referenced_by_runtime_acceleration_corpus() -> None:
    """
    Verify the A3 benchmark corpus points at the canonical Stage B fixture order and manifest.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        A3 exact-baseline coverage must stay aligned with the shipped R5-03 Stage B golden
        fixture catalog instead of duplicating a second Stage B reference surface.
    Raises:
        AssertionError: If the benchmark corpus drifts from the canonical Stage B fixture order.
    Side Effects:
        Reads the committed benchmark corpus and Stage B fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json
      - tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json
      - src/trading/contexts/backtest/application/services/v2/benchmark_corpus_v2.py
    """
    corpus = load_backtest_runtime_acceleration_benchmark_corpus_v2(
        path=_BENCHMARK_CORPUS_PATH
    )
    catalog = load_stage_b_golden_fixture_catalog_v2(path=_FIXTURE_PATH)
    exact_baseline = corpus.slice_for_id(slice_id="exact_baseline")

    assert corpus.source_fixtures.r5_stage_b_manifest == (
        "tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json"
    )
    assert corpus.source_fixtures.stage_b_golden_fixture == (
        "tests/unit/contexts/backtest/application/services/v2/fixtures/stage_b_golden_fixtures_v2.json"
    )
    assert exact_baseline.r5_stage_b_case_ids == catalog.case_order


def test_validate_stage_b_golden_fixture_payload_v2_rejects_unknown_schema_version() -> None:
    """
    Verify Stage B golden fixture validation fails fast on an unsupported schema version.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Schema evolution must stay explicit because R5-03 is a versioned contract artifact.
    Raises:
        AssertionError: If unsupported schema versions are accepted.
    Side Effects:
        Reads one JSON fixture catalog from repository.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
    """
    payload = read_stage_b_golden_fixture_payload_v2(path=_FIXTURE_PATH)
    mutated = deepcopy(payload)
    mutated["schema_version"] = 2

    try:
        validate_stage_b_golden_fixture_payload_v2(payload=mutated)
    except ValueError as exc:
        assert "schema_version must be 1" in str(exc)
    else:  # pragma: no cover - defensive fail-fast path
        raise AssertionError("unsupported schema_version must be rejected")


def test_validate_stage_b_golden_fixture_payload_v2_requires_precedence_assertion() -> None:
    """
    Verify equal-bar precedence cases fail fast when the explicit precedence assertion is omitted.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R5-03 fixtures must encode tie-break intent explicitly, not by implication.
    Raises:
        AssertionError: If equal-bar tie cases load without `precedence_assertion`.
    Side Effects:
        Reads one JSON fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
    """
    payload = read_stage_b_golden_fixture_payload_v2(path=_FIXTURE_PATH)
    mutated = deepcopy(payload)
    tie_case = _case_by_id(payload=mutated, case_id="signal_exit_wins_equal_bar_over_tp_sl_tie")
    expected = tie_case["expected"]
    assert isinstance(expected, dict)
    del expected["precedence_assertion"]

    try:
        validate_stage_b_golden_fixture_payload_v2(payload=mutated)
    except ValueError as exc:
        assert "precedence_assertion" in str(exc)
    else:  # pragma: no cover - defensive fail-fast path
        raise AssertionError("equal-bar precedence fixtures must require precedence_assertion")


def test_validate_stage_b_golden_fixture_payload_v2_rejects_missing_best_cell_metrics() -> None:
    """
    Verify best-cell replay fixtures fail fast when explicit expected metrics are missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R5-03 forbids fuzzy replay checks and therefore requires committed expected metrics.
    Raises:
        AssertionError: If a best-cell replay case loads without expected metrics.
    Side Effects:
        Reads one JSON fixture catalog from repository.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
    """
    payload = read_stage_b_golden_fixture_payload_v2(path=_FIXTURE_PATH)
    mutated = deepcopy(payload)
    replay_case = _case_by_id(payload=mutated, case_id="exact_best_cell_replay_metrics")
    expected = replay_case["expected"]
    assert isinstance(expected, dict)
    del expected["metrics"]

    try:
        validate_stage_b_golden_fixture_payload_v2(payload=mutated)
    except ValueError as exc:
        assert "metrics" in str(exc)
    else:  # pragma: no cover - defensive fail-fast path
        raise AssertionError("best-cell replay fixtures must require expected metrics")


def _assert_case_result_matches_expected(
    *,
    case: StageBEntryMappingCaseV2
    | StageBTradeListCaseV2
    | StageBTradeExitCaseV2
    | StageBBestCellReplayCaseV2,
) -> None:
    """
    Execute one typed Stage B fixture case and compare it to the committed expected payload.

    Args:
        case: Parsed typed Stage B fixture case.
    Returns:
        None.
    Assumptions:
        Case objects were already validated by the catalog loader.
    Raises:
        AssertionError: If the executable oracle diverges from the committed expected output.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
    """
    result = execute_stage_b_golden_case_v2(case=case)

    if isinstance(case, StageBEntryMappingCaseV2):
        artifact_mapping = map_bar_close_1m_idx_to_entry_exec_v2(
            bar_close_1m_idx=case.bar_close_1m_idx,
            sentinel_index=case.sentinel_index,
        )
        assert result == case.expected_entry_exec
        assert artifact_mapping == case.expected_entry_exec
        return

    if isinstance(case, StageBTradeListCaseV2):
        assert result == case.expected_compact_trades
        return

    if isinstance(case, StageBTradeExitCaseV2):
        assert result == case.expected_exit
        if case.precedence_assertion is not None:
            assert case.precedence_assertion in case.coverage
        return

    if isinstance(case, StageBBestCellReplayCaseV2):
        assert result == case.expected_result
        return

    raise AssertionError(f"unsupported case type: {case!r}")


def _case_by_id(*, payload: dict[str, object], case_id: str) -> dict[str, object]:
    """
    Return one raw JSON case object by `case_id` from a mutable fixture payload copy.

    Args:
        payload: Mutable raw JSON payload.
        case_id: Stable case identifier to find.
    Returns:
        dict[str, object]: Mutable raw case object for targeted negative-path mutation.
    Assumptions:
        The caller passes a payload already deep-copied from the committed fixture catalog.
    Raises:
        AssertionError: If the target case id cannot be found.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
    """
    raw_cases = payload["cases"]
    assert isinstance(raw_cases, list)
    for raw_case in raw_cases:
        if isinstance(raw_case, dict) and raw_case.get("case_id") == case_id:
            return raw_case
    raise AssertionError(f"case_id not found: {case_id!r}")
