from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from trading.contexts.backtest.application.services.v2 import (
    StageBBestCellReplayCaseV2,
    StageBEntryMappingCaseV2,
    StageBTradeExitCaseV2,
    StageBTradeListCaseV2,
    execute_stage_b_golden_case_v2,
    load_stage_b_golden_fixture_catalog_v2,
    map_bar_close_1m_idx_to_entry_exec_v2,
    read_stage_b_golden_fixture_payload_v2,
    serialize_stage_b_golden_fixture_payload_v2,
    validate_stage_b_golden_fixture_payload_v2,
)

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "stage_b_golden_fixtures_v2.json"
_PERF_MANIFEST_PATH = (
    Path(__file__).resolve().parents[6]
    / "perf_smoke"
    / "contexts"
    / "backtest"
    / "fixtures"
    / "r5_stage_b_golden_cases.json"
)


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
