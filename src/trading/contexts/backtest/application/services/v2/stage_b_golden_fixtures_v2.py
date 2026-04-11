"""Deterministic golden fixtures and executable oracle helpers for Stage B `signal_tf + 1m_risk`."""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Literal, cast

STAGE_B_GOLDEN_FIXTURE_SCHEMA_VERSION_V2 = 1
STAGE_B_GOLDEN_FIXTURE_KIND_V2 = "stage_b_golden_fixtures_v2"
STAGE_B_GOLDEN_FIXTURE_MILESTONE_ID_V2 = "R5"
STAGE_B_GOLDEN_FIXTURE_EPIC_ID_V2 = "R5-03"
STAGE_B_GOLDEN_FIXTURE_SEMANTICS_V2 = "signal_tf + 1m_risk"

StageBGoldenCaseKindLiteralV2 = Literal[
    "entry_mapping",
    "trade_list",
    "trade_exit",
    "best_cell_replay",
]
StageBTradeExitReasonLiteralV2 = Literal[
    "signal_exit",
    "tp",
    "sl",
    "close_on_end",
    "unclosed",
]

_DECIMAL_ZERO = Decimal("0")
_DECIMAL_ONE = Decimal("1")
_DECIMAL_TWO = Decimal("2")


@dataclass(frozen=True, slots=True)
class StageBCompactTradeV2:
    """
    Compact Stage B trade entry used by deterministic golden fixtures.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    entry_exec: int
    direction: int
    sig_exit_exec: int


@dataclass(frozen=True, slots=True)
class StageBExecutionPricesV2:
    """
    Execution timeline prices used by the Stage B golden fixture oracle.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    exec_open: tuple[Decimal, ...]
    exec_close: tuple[Decimal, ...]


@dataclass(frozen=True, slots=True)
class StageBHitTimesFixtureV2:
    """
    Minimal strict `1m` hit-times tables embedded inside a Stage B golden fixture.

    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
      - tests/notebook_tests/05_hit_time_grid.ipynb
    """

    long_tp: tuple[tuple[int, ...], ...]
    long_sl: tuple[tuple[int, ...], ...]
    short_tp: tuple[tuple[int, ...], ...]
    short_sl: tuple[tuple[int, ...], ...]
    sentinel_index: int


@dataclass(frozen=True, slots=True)
class StageBLevelFactorsV2:
    """
    Exact TP/SL gross factors used by deterministic Stage B fixture replay.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    long_tp: tuple[Decimal, ...]
    long_sl: tuple[Decimal, ...]
    short_tp: tuple[Decimal, ...]
    short_sl: tuple[Decimal, ...]


@dataclass(frozen=True, slots=True)
class StageBTradeExitResultV2:
    """
    Canonical resolved exit for one Stage B compact trade.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    gross_factor: Decimal
    exit_exec: int
    exit_reason: StageBTradeExitReasonLiteralV2
    closed: bool


@dataclass(frozen=True, slots=True)
class StageBReplayMetricsV2:
    """
    Exact replay metrics over compact trades for one best TP/SL cell.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    total_return: Decimal
    max_drawdown: Decimal
    sharpe: Decimal
    winrate: Decimal
    avg_trade_return: Decimal
    avg_trade_bars: Decimal
    exposure: Decimal


@dataclass(frozen=True, slots=True)
class StageBBestCellReplayResultV2:
    """
    Deterministic best-cell search result for one Stage B replay fixture.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    best_tp_index: int
    best_sl_index: int
    trade_count: int
    metrics: StageBReplayMetricsV2


@dataclass(frozen=True, slots=True)
class StageBEntryMappingCaseV2:
    """
    Fixture case for `entry mapping request TF -> 1m`.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    case_id: str
    title: str
    coverage: tuple[str, ...]
    signal_close_time_ms: tuple[int, ...]
    exec_open_time_ms: tuple[int, ...]
    bar_close_1m_idx: tuple[int, ...]
    sentinel_index: int
    expected_entry_exec: tuple[int, ...]
    case_kind: Literal["entry_mapping"] = "entry_mapping"


@dataclass(frozen=True, slots=True)
class StageBTradeListCaseV2:
    """
    Fixture case for compact trade construction and earliest signal-exit mapping.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    case_id: str
    title: str
    coverage: tuple[str, ...]
    final_signal: tuple[int, ...]
    bar_close_1m_idx: tuple[int, ...]
    sentinel_index: int
    expected_compact_trades: tuple[StageBCompactTradeV2, ...]
    case_kind: Literal["trade_list"] = "trade_list"


@dataclass(frozen=True, slots=True)
class StageBTradeExitCaseV2:
    """
    Fixture case for exact one-trade Stage B exit precedence.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    case_id: str
    title: str
    coverage: tuple[str, ...]
    direction: int
    entry_exec: int
    sig_exit_exec: int
    tp_index: int
    sl_index: int
    prices: StageBExecutionPricesV2
    hit_times: StageBHitTimesFixtureV2
    level_factors: StageBLevelFactorsV2
    close_on_end: bool
    expected_exit: StageBTradeExitResultV2
    precedence_assertion: str | None
    case_kind: Literal["trade_exit"] = "trade_exit"


@dataclass(frozen=True, slots=True)
class StageBBestCellReplayCaseV2:
    """
    Fixture case for exact best-cell replay and deterministic compact-trade metrics.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    case_id: str
    title: str
    coverage: tuple[str, ...]
    compact_trades: tuple[StageBCompactTradeV2, ...]
    prices: StageBExecutionPricesV2
    hit_times: StageBHitTimesFixtureV2
    level_factors: StageBLevelFactorsV2
    fee_rate: Decimal
    close_on_end: bool
    bars_per_year_exec: Decimal
    expected_result: StageBBestCellReplayResultV2
    case_kind: Literal["best_cell_replay"] = "best_cell_replay"


StageBGoldenFixtureCaseV2 = (
    StageBEntryMappingCaseV2
    | StageBTradeListCaseV2
    | StageBTradeExitCaseV2
    | StageBBestCellReplayCaseV2
)


@dataclass(frozen=True, slots=True)
class StageBGoldenFixtureCatalogV2:
    """
    Versioned Stage B golden fixture catalog for `signal_tf + 1m_risk` semantics.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """

    schema_version: int
    fixture_contract: str
    milestone_id: str
    epic_id: str
    semantics: str
    required_literals: tuple[str, ...]
    case_order: tuple[str, ...]
    cases: tuple[StageBGoldenFixtureCaseV2, ...]


def read_stage_b_golden_fixture_payload_v2(*, path: Path) -> dict[str, object]:
    """
    Read one Stage B golden fixture catalog JSON payload from disk.

    Args:
        path: Absolute or repository-relative fixture path.
    Returns:
        dict[str, object]: Raw JSON object preserving authored key ordering.
    Assumptions:
        Fixture files are committed JSON documents with object root and UTF-8/ASCII content.
    Raises:
        ValueError: If the JSON root is not an object.
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the payload is not valid JSON.
    Side Effects:
        Reads one repository file from disk.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stage_b golden fixture payload root must be an object")
    return payload


def serialize_stage_b_golden_fixture_payload_v2(*, payload: dict[str, object]) -> bytes:
    """
    Serialize one raw Stage B golden fixture payload with canonical repository formatting.

    Args:
        payload: Raw JSON-compatible object root produced by `json.loads`.
    Returns:
        bytes: Canonical UTF-8 bytes with stable indentation and trailing newline.
    Assumptions:
        Object key order in `payload` is already intentional and must stay deterministic.
    Raises:
        TypeError: If the payload contains non-JSON-serializable values.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    return (json.dumps(payload, ensure_ascii=True, indent=2) + "\n").encode("utf-8")


def load_stage_b_golden_fixture_catalog_v2(*, path: Path) -> StageBGoldenFixtureCatalogV2:
    """
    Read and validate one typed Stage B golden fixture catalog from disk.

    Args:
        path: Absolute or repository-relative fixture path.
    Returns:
        StageBGoldenFixtureCatalogV2: Typed deterministic fixture catalog.
    Assumptions:
        R5-03 fixture catalogs are self-contained and executable without notebooks.
    Raises:
        ValueError: If the payload violates the Stage B golden fixture contract.
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the payload is not valid JSON.
    Side Effects:
        Reads one repository file from disk.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    return validate_stage_b_golden_fixture_payload_v2(
        payload=read_stage_b_golden_fixture_payload_v2(path=path)
    )


def load_stage_b_best_cell_replay_reference_case_v2(
    *,
    path: Path,
) -> StageBBestCellReplayCaseV2:
    """
    Load the canonical Stage B best-cell replay reference case for bounded self-checks.

    Args:
        path: Repository path to the committed Stage B golden fixture catalog.
    Returns:
        StageBBestCellReplayCaseV2: The single committed best-cell replay reference case.
    Assumptions:
        The golden catalog keeps exactly one canonical `best_cell_replay` case that anchors the
        `reference-vs-fast self-check` surface on a bounded subset.
    Raises:
        ValueError: If the catalog contains zero or multiple best-cell replay cases.
    Side Effects:
        Reads and validates the committed Stage B golden fixture catalog from repository.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    catalog = load_stage_b_golden_fixture_catalog_v2(path=path)
    best_cell_cases = tuple(
        case for case in catalog.cases if isinstance(case, StageBBestCellReplayCaseV2)
    )
    if len(best_cell_cases) != 1:
        raise ValueError(
            "Stage B golden fixture catalog must contain exactly one best_cell_replay case"
        )
    return best_cell_cases[0]


def validate_stage_b_golden_fixture_payload_v2(
    *,
    payload: dict[str, object],
) -> StageBGoldenFixtureCatalogV2:
    """
    Validate and parse one Stage B golden fixture payload into typed case objects.

    Args:
        payload: Raw JSON object already loaded in memory.
    Returns:
        StageBGoldenFixtureCatalogV2: Parsed catalog with deterministic case ordering.
    Assumptions:
        Fixture contract hardens only Stage B validation semantics and does not cut over runtime.
    Raises:
        ValueError: If top-level metadata, case ordering, case coverage, or one case schema is
            invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    schema_version = _required_int_v2(
        payload=payload,
        key="schema_version",
        context="stage_b golden fixture catalog",
    )
    if schema_version != STAGE_B_GOLDEN_FIXTURE_SCHEMA_VERSION_V2:
        raise ValueError(
            "stage_b golden fixture catalog schema_version must be "
            f"{STAGE_B_GOLDEN_FIXTURE_SCHEMA_VERSION_V2}; got {schema_version!r}"
        )
    fixture_contract = _required_str_v2(
        payload=payload,
        key="fixture_contract",
        context="stage_b golden fixture catalog",
    )
    if fixture_contract != STAGE_B_GOLDEN_FIXTURE_KIND_V2:
        raise ValueError(
            "stage_b golden fixture catalog fixture_contract must be "
            f"{STAGE_B_GOLDEN_FIXTURE_KIND_V2!r}; got {fixture_contract!r}"
        )
    milestone_id = _required_str_v2(
        payload=payload,
        key="milestone_id",
        context="stage_b golden fixture catalog",
    )
    if milestone_id != STAGE_B_GOLDEN_FIXTURE_MILESTONE_ID_V2:
        raise ValueError(
            "stage_b golden fixture catalog milestone_id must be "
            f"{STAGE_B_GOLDEN_FIXTURE_MILESTONE_ID_V2!r}; got {milestone_id!r}"
        )
    epic_id = _required_str_v2(
        payload=payload,
        key="epic_id",
        context="stage_b golden fixture catalog",
    )
    if epic_id != STAGE_B_GOLDEN_FIXTURE_EPIC_ID_V2:
        raise ValueError(
            "stage_b golden fixture catalog epic_id must be "
            f"{STAGE_B_GOLDEN_FIXTURE_EPIC_ID_V2!r}; got {epic_id!r}"
        )
    semantics = _required_str_v2(
        payload=payload,
        key="semantics",
        context="stage_b golden fixture catalog",
    )
    if semantics != STAGE_B_GOLDEN_FIXTURE_SEMANTICS_V2:
        raise ValueError(
            "stage_b golden fixture catalog semantics must be "
            f"{STAGE_B_GOLDEN_FIXTURE_SEMANTICS_V2!r}; got {semantics!r}"
        )
    required_literals = _required_string_tuple_v2(
        payload=payload,
        key="required_literals",
        context="stage_b golden fixture catalog",
    )
    case_order = _required_string_tuple_v2(
        payload=payload,
        key="case_order",
        context="stage_b golden fixture catalog",
    )
    raw_cases = _required_list_v2(
        payload=payload,
        key="cases",
        context="stage_b golden fixture catalog",
    )
    parsed_cases: list[StageBGoldenFixtureCaseV2] = []
    parsed_case_ids: list[str] = []
    coverage_union: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("stage_b golden fixture case entries must be objects")
        parsed_case = _parse_stage_b_case_v2(raw_case=raw_case)
        parsed_cases.append(parsed_case)
        if parsed_case.case_id in parsed_case_ids:
            raise ValueError(
                f"stage_b golden fixture case_id values must be unique; got {parsed_case.case_id!r}"
            )
        parsed_case_ids.append(parsed_case.case_id)
        coverage_union.update(parsed_case.coverage)
    if tuple(parsed_case_ids) != case_order:
        raise ValueError(
            "stage_b golden fixture case_order must exactly match authored case ids; "
            f"expected {case_order!r}, got {tuple(parsed_case_ids)!r}"
        )
    missing_literals = tuple(
        literal for literal in required_literals if literal not in coverage_union
    )
    if missing_literals:
        raise ValueError(
            "stage_b golden fixture coverage must include every required literal; "
            f"missing {missing_literals!r}"
        )
    return StageBGoldenFixtureCatalogV2(
        schema_version=schema_version,
        fixture_contract=fixture_contract,
        milestone_id=milestone_id,
        epic_id=epic_id,
        semantics=semantics,
        required_literals=required_literals,
        case_order=case_order,
        cases=tuple(parsed_cases),
    )


def map_signal_bars_to_entry_exec_v2(
    *,
    signal_close_time_ms: tuple[int, ...],
    exec_open_time_ms: tuple[int, ...],
    sentinel_index: int,
) -> tuple[int, ...]:
    """
    Map request-timeframe signal closes onto the first `1m` execution bar after close.

    Args:
        signal_close_time_ms: Monotone signal-bar close timestamps in milliseconds.
        exec_open_time_ms: Monotone execution-bar open timestamps in milliseconds.
        sentinel_index: Execution timeline length used as the sentinel fallback.
    Returns:
        tuple[int, ...]: Deterministic execution entry indices for every signal bar.
    Assumptions:
        Artifact-backed runtime uses the same semantics as notebook `searchsorted(close+1)`.
    Raises:
        ValueError: If the execution timeline length does not match `sentinel_index`.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    if len(exec_open_time_ms) != sentinel_index:
        raise ValueError(
            "exec_open_time_ms length must match sentinel_index; "
            f"got len={len(exec_open_time_ms)}, sentinel_index={sentinel_index}"
        )
    mapped_entries: list[int] = []
    for close_ms in signal_close_time_ms:
        mapped_index = bisect_left(exec_open_time_ms, close_ms + 1)
        mapped_entries.append(mapped_index if mapped_index < sentinel_index else sentinel_index)
    return tuple(mapped_entries)


def map_bar_close_1m_idx_to_entry_exec_v2(
    *,
    bar_close_1m_idx: tuple[int, ...],
    sentinel_index: int,
) -> tuple[int, ...]:
    """
    Convert artifact-backed `bar_close_1m_idx` values into Stage B execution entries.

    Args:
        bar_close_1m_idx: Request-timeframe bar-to-`1m` close mapping.
        sentinel_index: Execution timeline length used as the sentinel fallback.
    Returns:
        tuple[int, ...]: Execution entries computed as `entry_exec + 1` with sentinel cap.
    Assumptions:
        Runtime contract uses `entry_exec = bar_close_1m_idx + 1` on shipped mappings.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    return tuple(
        min(mapped_close + 1, sentinel_index) for mapped_close in bar_close_1m_idx
    )


def build_compact_trade_list_from_final_signal_v2(
    *,
    final_signal: tuple[int, ...],
    bar_close_1m_idx: tuple[int, ...],
    sentinel_index: int,
) -> tuple[StageBCompactTradeV2, ...]:
    """
    Build compact Stage B trades from request-TF `final_signal` and `bar_close_1m_idx`.

    Args:
        final_signal: Deterministic request-timeframe signal values from `{-1, 0, 1}`.
        bar_close_1m_idx: Request-timeframe close mapping into the `1m` execution timeline.
        sentinel_index: Execution timeline length used as the sentinel fallback.
    Returns:
        tuple[StageBCompactTradeV2, ...]: Ordered compact trades with mapped signal exits.
    Assumptions:
        Repeated same-direction confirmations are ignored until an opposite confirmation arrives.
    Raises:
        ValueError: If input sequence lengths differ.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    if len(final_signal) != len(bar_close_1m_idx):
        raise ValueError(
            "final_signal length must match bar_close_1m_idx length; "
            f"got {len(final_signal)} vs {len(bar_close_1m_idx)}"
        )
    entry_exec_idx = map_bar_close_1m_idx_to_entry_exec_v2(
        bar_close_1m_idx=bar_close_1m_idx,
        sentinel_index=sentinel_index,
    )
    compact_trades: list[StageBCompactTradeV2] = []
    current_direction = 0
    current_entry = 0
    for direction, entry_exec in zip(final_signal, entry_exec_idx):
        if direction == 0:
            continue
        if entry_exec >= sentinel_index:
            break
        if current_direction == 0:
            current_direction = direction
            current_entry = entry_exec
            continue
        if direction == current_direction:
            continue
        compact_trades.append(
            StageBCompactTradeV2(
                entry_exec=current_entry,
                direction=current_direction,
                sig_exit_exec=entry_exec,
            )
        )
        current_direction = direction
        current_entry = entry_exec
    if current_direction != 0:
        compact_trades.append(
            StageBCompactTradeV2(
                entry_exec=current_entry,
                direction=current_direction,
                sig_exit_exec=sentinel_index,
            )
        )
    return tuple(compact_trades)


def evaluate_stage_b_trade_exit_v2(
    *,
    direction: int,
    entry_exec: int,
    sig_exit_exec: int,
    tp_index: int,
    sl_index: int,
    prices: StageBExecutionPricesV2,
    hit_times: StageBHitTimesFixtureV2,
    level_factors: StageBLevelFactorsV2,
    close_on_end: bool,
) -> StageBTradeExitResultV2:
    """
    Resolve one Stage B trade using deterministic `signal_tf + 1m_risk` precedence rules.

    Args:
        direction: Trade direction (`+1` long, `-1` short).
        entry_exec: Execution-bar entry index.
        sig_exit_exec: Opposite-signal execution exit index or sentinel.
        tp_index: Selected TP level index.
        sl_index: Selected SL level index.
        prices: Execution open/close prices.
        hit_times: Strict `1m` hit-times tables with `sentinel_index`.
        level_factors: Exact TP/SL gross factors per level.
        close_on_end: Whether the final open trade closes at end-of-series.
    Returns:
        StageBTradeExitResultV2: Exact exit reason, execution index, and gross factor.
    Assumptions:
        TP/SL lookup starts at `entry_exec + 1`, `signal exit wins on equal bar`, and
        `SL wins TP tie`.
    Raises:
        ValueError: If the input direction is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    if direction not in (-1, 1):
        raise ValueError(f"trade direction must be -1 or 1; got {direction!r}")
    entry_open = prices.exec_open[entry_exec]
    if entry_open <= _DECIMAL_ZERO:
        return StageBTradeExitResultV2(
            gross_factor=_DECIMAL_ONE,
            exit_exec=entry_exec,
            exit_reason="unclosed",
            closed=False,
        )
    sentinel_index = hit_times.sentinel_index
    lookup_exec = entry_exec + 1
    tp_exec = sentinel_index
    sl_exec = sentinel_index
    tp_factor = _DECIMAL_ONE
    sl_factor = _DECIMAL_ONE
    if lookup_exec < sentinel_index:
        if direction == 1:
            tp_exec = hit_times.long_tp[tp_index][lookup_exec]
            sl_exec = hit_times.long_sl[sl_index][lookup_exec]
            tp_factor = level_factors.long_tp[tp_index]
            sl_factor = level_factors.long_sl[sl_index]
        else:
            tp_exec = hit_times.short_tp[tp_index][lookup_exec]
            sl_exec = hit_times.short_sl[sl_index][lookup_exec]
            tp_factor = level_factors.short_tp[tp_index]
            sl_factor = level_factors.short_sl[sl_index]
    tp_sl_exec = sl_exec if sl_exec <= tp_exec else tp_exec
    tp_sl_reason: StageBTradeExitReasonLiteralV2 = "sl" if sl_exec <= tp_exec else "tp"
    tp_sl_factor = sl_factor if sl_exec <= tp_exec else tp_factor
    if sig_exit_exec < sentinel_index and sig_exit_exec <= tp_sl_exec:
        exit_open = prices.exec_open[sig_exit_exec]
        gross_factor = (
            _signal_or_end_factor_v2(
                direction=direction,
                entry_open=entry_open,
                exit_price=exit_open,
            )
            if exit_open > _DECIMAL_ZERO
            else _DECIMAL_ONE
        )
        return StageBTradeExitResultV2(
            gross_factor=gross_factor,
            exit_exec=sig_exit_exec,
            exit_reason="signal_exit",
            closed=True,
        )
    if tp_sl_exec < sentinel_index:
        return StageBTradeExitResultV2(
            gross_factor=tp_sl_factor,
            exit_exec=tp_sl_exec,
            exit_reason=tp_sl_reason,
            closed=True,
        )
    if close_on_end and sentinel_index > 0:
        last_close = prices.exec_close[sentinel_index - 1]
        gross_factor = (
            _signal_or_end_factor_v2(
                direction=direction,
                entry_open=entry_open,
                exit_price=last_close,
            )
            if last_close > _DECIMAL_ZERO
            else _DECIMAL_ONE
        )
        return StageBTradeExitResultV2(
            gross_factor=gross_factor,
            exit_exec=sentinel_index - 1,
            exit_reason="close_on_end",
            closed=True,
        )
    return StageBTradeExitResultV2(
        gross_factor=_DECIMAL_ONE,
        exit_exec=entry_exec,
        exit_reason="unclosed",
        closed=False,
    )


def replay_stage_b_best_cell_v2(
    *,
    compact_trades: tuple[StageBCompactTradeV2, ...],
    prices: StageBExecutionPricesV2,
    hit_times: StageBHitTimesFixtureV2,
    level_factors: StageBLevelFactorsV2,
    fee_rate: Decimal,
    close_on_end: bool,
    bars_per_year_exec: Decimal,
) -> StageBBestCellReplayResultV2:
    """
    Search the best TP/SL cell and replay exact metrics over compact trades.

    Args:
        compact_trades: Ordered compact trades produced by Stage A transfer semantics.
        prices: Execution open/close prices.
        hit_times: Strict `1m` hit-times tables with `sentinel_index`.
        level_factors: Exact TP/SL gross factors per level.
        fee_rate: Per-side fee rate expressed as a decimal fraction.
        close_on_end: Whether the final open trade closes at end-of-series.
        bars_per_year_exec: Annualization denominator in execution bars.
    Returns:
        StageBBestCellReplayResultV2: Exact best cell and deterministic replay metrics.
    Assumptions:
        Ranking is driven by total return and ties are broken by the smallest `(tp_i, sl_i)`.
    Raises:
        ValueError: If no TP or SL levels are available.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    if len(level_factors.long_tp) == 0 or len(level_factors.long_sl) == 0:
        raise ValueError("best-cell replay requires at least one TP level and one SL level")
    best_tp_index = 0
    best_sl_index = 0
    best_result = _replay_tp_sl_cell_v2(
        compact_trades=compact_trades,
        prices=prices,
        hit_times=hit_times,
        level_factors=level_factors,
        tp_index=0,
        sl_index=0,
        fee_rate=fee_rate,
        close_on_end=close_on_end,
        bars_per_year_exec=bars_per_year_exec,
    )
    best_total_return = best_result.metrics.total_return
    for tp_index in range(len(level_factors.long_tp)):
        for sl_index in range(len(level_factors.long_sl)):
            candidate = _replay_tp_sl_cell_v2(
                compact_trades=compact_trades,
                prices=prices,
                hit_times=hit_times,
                level_factors=level_factors,
                tp_index=tp_index,
                sl_index=sl_index,
                fee_rate=fee_rate,
                close_on_end=close_on_end,
                bars_per_year_exec=bars_per_year_exec,
            )
            candidate_total_return = candidate.metrics.total_return
            if candidate_total_return > best_total_return or (
                candidate_total_return == best_total_return
                and (tp_index, sl_index) < (best_tp_index, best_sl_index)
            ):
                best_tp_index = tp_index
                best_sl_index = sl_index
                best_result = candidate
                best_total_return = candidate_total_return
    return StageBBestCellReplayResultV2(
        best_tp_index=best_tp_index,
        best_sl_index=best_sl_index,
        trade_count=best_result.trade_count,
        metrics=best_result.metrics,
    )


def execute_stage_b_golden_case_v2(
    *,
    case: StageBGoldenFixtureCaseV2,
) -> (
    tuple[int, ...]
    | tuple[StageBCompactTradeV2, ...]
    | StageBTradeExitResultV2
    | StageBBestCellReplayResultV2
):
    """
    Execute one typed Stage B golden fixture case with the local deterministic oracle.

    Args:
        case: Parsed typed Stage B fixture case.
    Returns:
        tuple[int, ...] | tuple[StageBCompactTradeV2, ...] | StageBTradeExitResultV2 |
        StageBBestCellReplayResultV2: Exact oracle result for the requested case.
    Assumptions:
        The case was already validated by `load_stage_b_golden_fixture_catalog_v2`.
    Raises:
        AssertionError: If an unsupported case kind reaches this dispatcher.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    if case.case_kind == "entry_mapping":
        return map_signal_bars_to_entry_exec_v2(
            signal_close_time_ms=case.signal_close_time_ms,
            exec_open_time_ms=case.exec_open_time_ms,
            sentinel_index=case.sentinel_index,
        )
    if case.case_kind == "trade_list":
        return build_compact_trade_list_from_final_signal_v2(
            final_signal=case.final_signal,
            bar_close_1m_idx=case.bar_close_1m_idx,
            sentinel_index=case.sentinel_index,
        )
    if case.case_kind == "trade_exit":
        return evaluate_stage_b_trade_exit_v2(
            direction=case.direction,
            entry_exec=case.entry_exec,
            sig_exit_exec=case.sig_exit_exec,
            tp_index=case.tp_index,
            sl_index=case.sl_index,
            prices=case.prices,
            hit_times=case.hit_times,
            level_factors=case.level_factors,
            close_on_end=case.close_on_end,
        )
    if case.case_kind == "best_cell_replay":
        return replay_stage_b_best_cell_v2(
            compact_trades=case.compact_trades,
            prices=case.prices,
            hit_times=case.hit_times,
            level_factors=case.level_factors,
            fee_rate=case.fee_rate,
            close_on_end=case.close_on_end,
            bars_per_year_exec=case.bars_per_year_exec,
        )
    raise AssertionError(f"unsupported stage_b golden case kind: {case!r}")


def _replay_tp_sl_cell_v2(
    *,
    compact_trades: tuple[StageBCompactTradeV2, ...],
    prices: StageBExecutionPricesV2,
    hit_times: StageBHitTimesFixtureV2,
    level_factors: StageBLevelFactorsV2,
    tp_index: int,
    sl_index: int,
    fee_rate: Decimal,
    close_on_end: bool,
    bars_per_year_exec: Decimal,
) -> StageBBestCellReplayResultV2:
    """
    Replay one explicit TP/SL cell across compact trades and compute exact metrics.

    Args:
        compact_trades: Ordered compact trades produced by Stage A transfer semantics.
        prices: Execution open/close prices.
        hit_times: Strict `1m` hit-times tables with `sentinel_index`.
        level_factors: Exact TP/SL gross factors per level.
        tp_index: Selected TP level index.
        sl_index: Selected SL level index.
        fee_rate: Per-side fee rate expressed as a decimal fraction.
        close_on_end: Whether the final open trade closes at end-of-series.
        bars_per_year_exec: Annualization denominator in execution bars.
    Returns:
        StageBBestCellReplayResultV2: Replay metrics for one explicit TP/SL cell.
    Assumptions:
        Fee handling follows notebook `fee_two_sides = (1-fee)^2`.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    fee_two_sides = (_DECIMAL_ONE - fee_rate) * (_DECIMAL_ONE - fee_rate)
    equity = _DECIMAL_ONE
    peak = _DECIMAL_ONE
    max_drawdown = _DECIMAL_ZERO
    trade_count = 0
    win_count = 0
    sum_trade_return = _DECIMAL_ZERO
    sum_trade_return_squared = _DECIMAL_ZERO
    sum_trade_bars = _DECIMAL_ZERO
    exposure_bars = _DECIMAL_ZERO
    for trade in compact_trades:
        exit_result = evaluate_stage_b_trade_exit_v2(
            direction=trade.direction,
            entry_exec=trade.entry_exec,
            sig_exit_exec=trade.sig_exit_exec,
            tp_index=tp_index,
            sl_index=sl_index,
            prices=prices,
            hit_times=hit_times,
            level_factors=level_factors,
            close_on_end=close_on_end,
        )
        if not exit_result.closed:
            continue
        gross_after_fees = fee_two_sides * exit_result.gross_factor
        equity *= gross_after_fees
        trade_return = gross_after_fees - _DECIMAL_ONE
        trade_count += 1
        if trade_return > _DECIMAL_ZERO:
            win_count += 1
        sum_trade_return += trade_return
        sum_trade_return_squared += trade_return * trade_return
        bars_held = Decimal(max(exit_result.exit_exec - trade.entry_exec, 0))
        sum_trade_bars += bars_held
        exposure_bars += bars_held
        if equity > peak:
            peak = equity
        drawdown = (equity / peak) - _DECIMAL_ONE
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    total_return = equity - _DECIMAL_ONE
    if trade_count > 0:
        winrate = Decimal(win_count) / Decimal(trade_count)
        avg_trade_return = sum_trade_return / Decimal(trade_count)
        avg_trade_bars = sum_trade_bars / Decimal(trade_count)
    else:
        winrate = _DECIMAL_ZERO
        avg_trade_return = _DECIMAL_ZERO
        avg_trade_bars = _DECIMAL_ZERO
    total_exec_bars = Decimal(hit_times.sentinel_index)
    exposure = exposure_bars / total_exec_bars if total_exec_bars > _DECIMAL_ZERO else _DECIMAL_ZERO
    sharpe = _decimal_zero_or_sharpe_v2(
        trade_count=trade_count,
        sum_trade_return=sum_trade_return,
        sum_trade_return_squared=sum_trade_return_squared,
        bars_per_year_exec=bars_per_year_exec,
        total_exec_bars=total_exec_bars,
    )
    return StageBBestCellReplayResultV2(
        best_tp_index=tp_index,
        best_sl_index=sl_index,
        trade_count=trade_count,
        metrics=StageBReplayMetricsV2(
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe=sharpe,
            winrate=winrate,
            avg_trade_return=avg_trade_return,
            avg_trade_bars=avg_trade_bars,
            exposure=exposure,
        ),
    )


def _decimal_zero_or_sharpe_v2(
    *,
    trade_count: int,
    sum_trade_return: Decimal,
    sum_trade_return_squared: Decimal,
    bars_per_year_exec: Decimal,
    total_exec_bars: Decimal,
) -> Decimal:
    """
    Compute notebook-style Sharpe over compact trades with deterministic decimal math.

    Args:
        trade_count: Number of closed trades in the replay.
        sum_trade_return: Sum of per-trade returns after fees.
        sum_trade_return_squared: Sum of squared per-trade returns after fees.
        bars_per_year_exec: Annualization denominator in execution bars.
        total_exec_bars: Total execution bars in the replay timeline.
    Returns:
        Decimal: Deterministic Sharpe ratio or zero when variance is non-positive.
    Assumptions:
        This helper is used only by small golden fixtures and therefore favors exactness over speed.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    if trade_count <= 1:
        return _DECIMAL_ZERO
    mean_trade_return = sum_trade_return / Decimal(trade_count)
    variance = (sum_trade_return_squared / Decimal(trade_count)) - (
        mean_trade_return * mean_trade_return
    )
    if variance <= _DECIMAL_ZERO:
        return _DECIMAL_ZERO
    years = (
        total_exec_bars / bars_per_year_exec
        if bars_per_year_exec > _DECIMAL_ZERO
        else _DECIMAL_ONE
    )
    if years <= _DECIMAL_ZERO:
        years = _DECIMAL_ONE
    trades_per_year = Decimal(trade_count) / years
    with localcontext() as context:
        context.prec = 28
        return (mean_trade_return / variance.sqrt()) * trades_per_year.sqrt()


def _signal_or_end_factor_v2(
    *,
    direction: int,
    entry_open: Decimal,
    exit_price: Decimal,
) -> Decimal:
    """
    Compute gross factor for a signal-exit or end-of-series close under notebook semantics.

    Args:
        direction: Trade direction (`+1` long, `-1` short).
        entry_open: Entry execution-bar open price.
        exit_price: Exit execution-bar open price or final close price.
    Returns:
        Decimal: Gross factor before fees and slippage.
    Assumptions:
        Short trades use the notebook x1 USDT ROI model `max(0, 2 - exit/entry)`.
    Raises:
        ValueError: If the trade direction is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    if direction == 1:
        return exit_price / entry_open
    if direction == -1:
        factor = _DECIMAL_TWO - (exit_price / entry_open)
        return factor if factor > _DECIMAL_ZERO else _DECIMAL_ZERO
    raise ValueError(f"trade direction must be -1 or 1; got {direction!r}")


def _parse_stage_b_case_v2(*, raw_case: dict[str, object]) -> StageBGoldenFixtureCaseV2:
    """
    Parse and validate one raw Stage B fixture case object.

    Args:
        raw_case: Raw case payload from the JSON catalog.
    Returns:
        StageBGoldenFixtureCaseV2: One typed case object.
    Assumptions:
        Case structure follows `case_kind -> inputs -> expected`.
    Raises:
        ValueError: If the case kind or one case-local invariant is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    case_kind = _required_str_v2(
        payload=raw_case,
        key="case_kind",
        context="stage_b golden fixture case",
    )
    if case_kind == "entry_mapping":
        return _parse_entry_mapping_case_v2(raw_case=raw_case)
    if case_kind == "trade_list":
        return _parse_trade_list_case_v2(raw_case=raw_case)
    if case_kind == "trade_exit":
        return _parse_trade_exit_case_v2(raw_case=raw_case)
    if case_kind == "best_cell_replay":
        return _parse_best_cell_replay_case_v2(raw_case=raw_case)
    raise ValueError(f"unsupported stage_b golden fixture case_kind: {case_kind!r}")


def _parse_entry_mapping_case_v2(*, raw_case: dict[str, object]) -> StageBEntryMappingCaseV2:
    """
    Parse one typed `entry mapping request TF -> 1m` fixture case.

    Args:
        raw_case: Raw case payload from the JSON catalog.
    Returns:
        StageBEntryMappingCaseV2: Parsed entry-mapping case.
    Assumptions:
        Both notebook `searchsorted(close+1)` and artifact `bar_close_1m_idx + 1` forms must
        agree for the same case inputs.
    Raises:
        ValueError: If timestamps, mappings, or expected outputs are inconsistent.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    case_id, title, coverage, inputs, expected = _common_case_fields_v2(raw_case=raw_case)
    sentinel_index = _required_int_v2(
        payload=inputs,
        key="sentinel_index",
        context=f"entry_mapping case {case_id!r} inputs",
    )
    signal_close_time_ms = _required_int_tuple_v2(
        payload=inputs,
        key="signal_close_time_ms",
        context=f"entry_mapping case {case_id!r} inputs",
    )
    exec_open_time_ms = _required_int_tuple_v2(
        payload=inputs,
        key="exec_open_time_ms",
        context=f"entry_mapping case {case_id!r} inputs",
    )
    bar_close_1m_idx = _required_int_tuple_v2(
        payload=inputs,
        key="bar_close_1m_idx",
        context=f"entry_mapping case {case_id!r} inputs",
    )
    expected_entry_exec = _required_int_tuple_v2(
        payload=expected,
        key="entry_exec",
        context=f"entry_mapping case {case_id!r} expected",
    )
    if len(signal_close_time_ms) != len(bar_close_1m_idx):
        raise ValueError(
            f"entry_mapping case {case_id!r} signal_close_time_ms length must match "
            f"bar_close_1m_idx length"
        )
    if len(expected_entry_exec) != len(signal_close_time_ms):
        raise ValueError(
            f"entry_mapping case {case_id!r} expected entry_exec length must match signals"
        )
    _validate_monotone_int_sequence_v2(
        values=signal_close_time_ms,
        field_name=f"entry_mapping case {case_id!r} signal_close_time_ms",
    )
    _validate_strictly_monotone_int_sequence_v2(
        values=exec_open_time_ms,
        field_name=f"entry_mapping case {case_id!r} exec_open_time_ms",
    )
    if len(exec_open_time_ms) != sentinel_index:
        raise ValueError(
            f"entry_mapping case {case_id!r} exec_open_time_ms length must equal sentinel_index"
        )
    _validate_bar_close_mapping_v2(
        bar_close_1m_idx=bar_close_1m_idx,
        sentinel_index=sentinel_index,
        context=f"entry_mapping case {case_id!r}",
    )
    time_mapping = map_signal_bars_to_entry_exec_v2(
        signal_close_time_ms=signal_close_time_ms,
        exec_open_time_ms=exec_open_time_ms,
        sentinel_index=sentinel_index,
    )
    artifact_mapping = map_bar_close_1m_idx_to_entry_exec_v2(
        bar_close_1m_idx=bar_close_1m_idx,
        sentinel_index=sentinel_index,
    )
    if time_mapping != artifact_mapping:
        raise ValueError(
            f"entry_mapping case {case_id!r} must keep time and artifact mappings aligned; "
            f"got time={time_mapping!r}, artifact={artifact_mapping!r}"
        )
    return StageBEntryMappingCaseV2(
        case_id=case_id,
        title=title,
        coverage=coverage,
        signal_close_time_ms=signal_close_time_ms,
        exec_open_time_ms=exec_open_time_ms,
        bar_close_1m_idx=bar_close_1m_idx,
        sentinel_index=sentinel_index,
        expected_entry_exec=expected_entry_exec,
    )


def _parse_trade_list_case_v2(*, raw_case: dict[str, object]) -> StageBTradeListCaseV2:
    """
    Parse one typed compact-trade fixture case with earliest signal-exit mapping.

    Args:
        raw_case: Raw case payload from the JSON catalog.
    Returns:
        StageBTradeListCaseV2: Parsed compact-trade case.
    Assumptions:
        `final_signal` uses only `{-1, 0, 1}` and `sig_exit_exec` points to the next opposite
        mapped entry.
    Raises:
        ValueError: If signal values, mappings, or expected compact trades are invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    case_id, title, coverage, inputs, expected = _common_case_fields_v2(raw_case=raw_case)
    sentinel_index = _required_int_v2(
        payload=inputs,
        key="sentinel_index",
        context=f"trade_list case {case_id!r} inputs",
    )
    final_signal = _required_int_tuple_v2(
        payload=inputs,
        key="final_signal",
        context=f"trade_list case {case_id!r} inputs",
    )
    bar_close_1m_idx = _required_int_tuple_v2(
        payload=inputs,
        key="bar_close_1m_idx",
        context=f"trade_list case {case_id!r} inputs",
    )
    if len(final_signal) != len(bar_close_1m_idx):
        raise ValueError(
            f"trade_list case {case_id!r} final_signal length must match bar_close_1m_idx length"
        )
    for signal_value in final_signal:
        if signal_value not in (-1, 0, 1):
            raise ValueError(
                f"trade_list case {case_id!r} final_signal values must stay in {{-1,0,1}}"
            )
    _validate_bar_close_mapping_v2(
        bar_close_1m_idx=bar_close_1m_idx,
        sentinel_index=sentinel_index,
        context=f"trade_list case {case_id!r}",
    )
    expected_compact_trades = _parse_compact_trades_v2(
        payload=expected,
        key="compact_trades",
        context=f"trade_list case {case_id!r} expected",
        sentinel_index=sentinel_index,
    )
    return StageBTradeListCaseV2(
        case_id=case_id,
        title=title,
        coverage=coverage,
        final_signal=final_signal,
        bar_close_1m_idx=bar_close_1m_idx,
        sentinel_index=sentinel_index,
        expected_compact_trades=expected_compact_trades,
    )


def _parse_trade_exit_case_v2(*, raw_case: dict[str, object]) -> StageBTradeExitCaseV2:
    """
    Parse one typed single-trade exit fixture case with explicit precedence assertions.

    Args:
        raw_case: Raw case payload from the JSON catalog.
    Returns:
        StageBTradeExitCaseV2: Parsed trade-exit case.
    Assumptions:
        Any equal-bar TP/SL or signal/TP/SL tie must be documented with an explicit rule string.
    Raises:
        ValueError: If prices, hit-times, factors, or precedence assertions are invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    case_id, title, coverage, inputs, expected = _common_case_fields_v2(raw_case=raw_case)
    direction = _required_int_v2(
        payload=inputs,
        key="direction",
        context=f"trade_exit case {case_id!r} inputs",
    )
    entry_exec = _required_int_v2(
        payload=inputs,
        key="entry_exec",
        context=f"trade_exit case {case_id!r} inputs",
    )
    sig_exit_exec = _required_int_v2(
        payload=inputs,
        key="sig_exit_exec",
        context=f"trade_exit case {case_id!r} inputs",
    )
    tp_index = _required_int_v2(
        payload=inputs,
        key="tp_index",
        context=f"trade_exit case {case_id!r} inputs",
    )
    sl_index = _required_int_v2(
        payload=inputs,
        key="sl_index",
        context=f"trade_exit case {case_id!r} inputs",
    )
    close_on_end = _required_bool_v2(
        payload=inputs,
        key="close_on_end",
        context=f"trade_exit case {case_id!r} inputs",
    )
    prices = _parse_prices_v2(
        payload=inputs,
        key="prices",
        context=f"trade_exit case {case_id!r} inputs",
    )
    hit_times = _parse_hit_times_v2(
        payload=inputs,
        key="hit_times",
        context=f"trade_exit case {case_id!r} inputs",
    )
    level_factors = _parse_level_factors_v2(
        payload=inputs,
        key="level_factors",
        context=f"trade_exit case {case_id!r} inputs",
    )
    _validate_trade_indexes_v2(
        entry_exec=entry_exec,
        sig_exit_exec=sig_exit_exec,
        sentinel_index=hit_times.sentinel_index,
        context=f"trade_exit case {case_id!r}",
    )
    _validate_level_index_v2(
        index=tp_index,
        length=len(level_factors.long_tp),
        field_name=f"trade_exit case {case_id!r} tp_index",
    )
    _validate_level_index_v2(
        index=sl_index,
        length=len(level_factors.long_sl),
        field_name=f"trade_exit case {case_id!r} sl_index",
    )
    expected_exit = _parse_trade_exit_result_v2(
        payload=expected,
        context=f"trade_exit case {case_id!r} expected",
    )
    precedence_assertion = _optional_precedence_assertion_v2(
        expected=expected,
        direction=direction,
        entry_exec=entry_exec,
        sig_exit_exec=sig_exit_exec,
        tp_index=tp_index,
        sl_index=sl_index,
        hit_times=hit_times,
    )
    return StageBTradeExitCaseV2(
        case_id=case_id,
        title=title,
        coverage=coverage,
        direction=direction,
        entry_exec=entry_exec,
        sig_exit_exec=sig_exit_exec,
        tp_index=tp_index,
        sl_index=sl_index,
        prices=prices,
        hit_times=hit_times,
        level_factors=level_factors,
        close_on_end=close_on_end,
        expected_exit=expected_exit,
        precedence_assertion=precedence_assertion,
    )


def _parse_best_cell_replay_case_v2(
    *,
    raw_case: dict[str, object],
) -> StageBBestCellReplayCaseV2:
    """
    Parse one typed exact best-cell replay fixture case with expected metrics.

    Args:
        raw_case: Raw case payload from the JSON catalog.
    Returns:
        StageBBestCellReplayCaseV2: Parsed best-cell replay case.
    Assumptions:
        Metrics are authored over compact trades only and remain notebook-independent.
    Raises:
        ValueError: If compact trades, metrics, or one replay input is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    case_id, title, coverage, inputs, expected = _common_case_fields_v2(raw_case=raw_case)
    prices = _parse_prices_v2(
        payload=inputs,
        key="prices",
        context=f"best_cell_replay case {case_id!r} inputs",
    )
    hit_times = _parse_hit_times_v2(
        payload=inputs,
        key="hit_times",
        context=f"best_cell_replay case {case_id!r} inputs",
    )
    level_factors = _parse_level_factors_v2(
        payload=inputs,
        key="level_factors",
        context=f"best_cell_replay case {case_id!r} inputs",
    )
    compact_trades = _parse_compact_trades_v2(
        payload=inputs,
        key="compact_trades",
        context=f"best_cell_replay case {case_id!r} inputs",
        sentinel_index=hit_times.sentinel_index,
    )
    fee_rate = _required_decimal_v2(
        payload=inputs,
        key="fee_rate",
        context=f"best_cell_replay case {case_id!r} inputs",
    )
    close_on_end = _required_bool_v2(
        payload=inputs,
        key="close_on_end",
        context=f"best_cell_replay case {case_id!r} inputs",
    )
    bars_per_year_exec = _required_decimal_v2(
        payload=inputs,
        key="bars_per_year_exec",
        context=f"best_cell_replay case {case_id!r} inputs",
    )
    if bars_per_year_exec <= _DECIMAL_ZERO:
        raise ValueError(
            f"best_cell_replay case {case_id!r} bars_per_year_exec must be > 0"
        )
    expected_result = _parse_best_cell_replay_result_v2(
        payload=expected,
        context=f"best_cell_replay case {case_id!r} expected",
    )
    return StageBBestCellReplayCaseV2(
        case_id=case_id,
        title=title,
        coverage=coverage,
        compact_trades=compact_trades,
        prices=prices,
        hit_times=hit_times,
        level_factors=level_factors,
        fee_rate=fee_rate,
        close_on_end=close_on_end,
        bars_per_year_exec=bars_per_year_exec,
        expected_result=expected_result,
    )


def _common_case_fields_v2(
    *,
    raw_case: dict[str, object],
) -> tuple[str, str, tuple[str, ...], dict[str, object], dict[str, object]]:
    """
    Extract common metadata and nested mappings shared by every Stage B fixture case.

    Args:
        raw_case: Raw case payload from the JSON catalog.
    Returns:
        tuple[str, str, tuple[str, ...], dict[str, object], dict[str, object]]: Case id, title,
        coverage list, inputs mapping, and expected mapping.
    Assumptions:
        Every case is authored as `case_id/title/coverage/inputs/expected`.
    Raises:
        ValueError: If one required field is missing or malformed.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    case_id = _required_str_v2(
        payload=raw_case,
        key="case_id",
        context="stage_b golden fixture case",
    )
    title = _required_str_v2(
        payload=raw_case,
        key="title",
        context=f"stage_b golden fixture case {case_id!r}",
    )
    coverage = _required_string_tuple_v2(
        payload=raw_case,
        key="coverage",
        context=f"stage_b golden fixture case {case_id!r}",
    )
    if len(set(coverage)) != len(coverage):
        raise ValueError(f"stage_b golden fixture case {case_id!r} coverage must be unique")
    inputs = _required_mapping_v2(
        payload=raw_case,
        key="inputs",
        context=f"stage_b golden fixture case {case_id!r}",
    )
    expected = _required_mapping_v2(
        payload=raw_case,
        key="expected",
        context=f"stage_b golden fixture case {case_id!r}",
    )
    return case_id, title, coverage, inputs, expected


def _parse_prices_v2(
    *,
    payload: dict[str, object],
    key: str,
    context: str,
) -> StageBExecutionPricesV2:
    """
    Parse one execution-price block embedded in a Stage B golden fixture.

    Args:
        payload: Parent JSON mapping.
        key: Field name holding the nested price block.
        context: Human-readable validation context.
    Returns:
        StageBExecutionPricesV2: Typed execution-price structure.
    Assumptions:
        Golden fixtures keep execution opens/closes minimal but shape-complete.
    Raises:
        ValueError: If price arrays are empty, differently sized, or non-positive.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    raw_prices = _required_mapping_v2(payload=payload, key=key, context=context)
    exec_open = _required_decimal_tuple_v2(
        payload=raw_prices,
        key="exec_open",
        context=f"{context} prices",
    )
    exec_close = _required_decimal_tuple_v2(
        payload=raw_prices,
        key="exec_close",
        context=f"{context} prices",
    )
    if len(exec_open) == 0:
        raise ValueError(f"{context} prices exec_open must be non-empty")
    if len(exec_open) != len(exec_close):
        raise ValueError(f"{context} prices exec_open and exec_close lengths must match")
    for price_value in (*exec_open, *exec_close):
        if price_value <= _DECIMAL_ZERO:
            raise ValueError(f"{context} prices must stay > 0")
    return StageBExecutionPricesV2(exec_open=exec_open, exec_close=exec_close)


def _parse_hit_times_v2(
    *,
    payload: dict[str, object],
    key: str,
    context: str,
) -> StageBHitTimesFixtureV2:
    """
    Parse one strict hit-times block embedded in a Stage B golden fixture.

    Args:
        payload: Parent JSON mapping.
        key: Field name holding the nested hit-times block.
        context: Human-readable validation context.
    Returns:
        StageBHitTimesFixtureV2: Typed strict hit-times structure.
    Assumptions:
        Tables already use shipped R5-01 semantics and therefore remain bounded by sentinel.
    Raises:
        ValueError: If the sentinel or one table shape/value range is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_hit_times_compute_v2.py
    """
    raw_hit_times = _required_mapping_v2(payload=payload, key=key, context=context)
    sentinel_index = _required_int_v2(
        payload=raw_hit_times,
        key="sentinel_index",
        context=f"{context} hit_times",
    )
    if sentinel_index <= 0:
        raise ValueError(f"{context} hit_times sentinel_index must be > 0")
    long_tp = _required_table_v2(
        payload=raw_hit_times,
        key="long_tp",
        context=f"{context} hit_times",
        sentinel_index=sentinel_index,
    )
    long_sl = _required_table_v2(
        payload=raw_hit_times,
        key="long_sl",
        context=f"{context} hit_times",
        sentinel_index=sentinel_index,
    )
    short_tp = _required_table_v2(
        payload=raw_hit_times,
        key="short_tp",
        context=f"{context} hit_times",
        sentinel_index=sentinel_index,
    )
    short_sl = _required_table_v2(
        payload=raw_hit_times,
        key="short_sl",
        context=f"{context} hit_times",
        sentinel_index=sentinel_index,
    )
    return StageBHitTimesFixtureV2(
        long_tp=long_tp,
        long_sl=long_sl,
        short_tp=short_tp,
        short_sl=short_sl,
        sentinel_index=sentinel_index,
    )


def _parse_level_factors_v2(
    *,
    payload: dict[str, object],
    key: str,
    context: str,
) -> StageBLevelFactorsV2:
    """
    Parse one exact TP/SL factor block embedded in a Stage B golden fixture.

    Args:
        payload: Parent JSON mapping.
        key: Field name holding the nested factor block.
        context: Human-readable validation context.
    Returns:
        StageBLevelFactorsV2: Typed exact factor structure.
    Assumptions:
        Factors are gross multipliers before fees and therefore must stay positive.
    Raises:
        ValueError: If a factor array is empty, mismatched, or non-positive.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    raw_factors = _required_mapping_v2(payload=payload, key=key, context=context)
    long_tp = _required_decimal_tuple_v2(
        payload=raw_factors,
        key="long_tp",
        context=f"{context} level_factors",
    )
    long_sl = _required_decimal_tuple_v2(
        payload=raw_factors,
        key="long_sl",
        context=f"{context} level_factors",
    )
    short_tp = _required_decimal_tuple_v2(
        payload=raw_factors,
        key="short_tp",
        context=f"{context} level_factors",
    )
    short_sl = _required_decimal_tuple_v2(
        payload=raw_factors,
        key="short_sl",
        context=f"{context} level_factors",
    )
    if len(long_tp) == 0 or len(long_sl) == 0 or len(short_tp) == 0 or len(short_sl) == 0:
        raise ValueError(f"{context} level_factors arrays must be non-empty")
    for factor in (*long_tp, *long_sl, *short_tp, *short_sl):
        if factor <= _DECIMAL_ZERO:
            raise ValueError(f"{context} level_factors values must stay > 0")
    if len(long_tp) != len(short_tp):
        raise ValueError(f"{context} long_tp and short_tp lengths must match")
    if len(long_sl) != len(short_sl):
        raise ValueError(f"{context} long_sl and short_sl lengths must match")
    return StageBLevelFactorsV2(
        long_tp=long_tp,
        long_sl=long_sl,
        short_tp=short_tp,
        short_sl=short_sl,
    )


def _parse_compact_trades_v2(
    *,
    payload: dict[str, object],
    key: str,
    context: str,
    sentinel_index: int,
) -> tuple[StageBCompactTradeV2, ...]:
    """
    Parse one compact-trade array embedded in a Stage B golden fixture.

    Args:
        payload: Parent JSON mapping.
        key: Field name holding the compact-trade array.
        context: Human-readable validation context.
        sentinel_index: Execution timeline length used as the sentinel fallback.
    Returns:
        tuple[StageBCompactTradeV2, ...]: Typed ordered compact trades.
    Assumptions:
        Compact trades must be strictly ordered by entry and preserve deterministic Stage A output.
    Raises:
        ValueError: If one compact trade violates direction/order/sentinel invariants.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    raw_trades = _required_list_v2(payload=payload, key=key, context=context)
    parsed_trades: list[StageBCompactTradeV2] = []
    previous_entry = -1
    for raw_trade in raw_trades:
        if not isinstance(raw_trade, dict):
            raise ValueError(f"{context} compact trade entries must be objects")
        entry_exec = _required_int_v2(
            payload=raw_trade,
            key="entry_exec",
            context=f"{context} compact trade",
        )
        direction = _required_int_v2(
            payload=raw_trade,
            key="direction",
            context=f"{context} compact trade",
        )
        sig_exit_exec = _required_int_v2(
            payload=raw_trade,
            key="sig_exit_exec",
            context=f"{context} compact trade",
        )
        if direction not in (-1, 1):
            raise ValueError(f"{context} compact trade direction must be -1 or 1")
        if entry_exec < 0 or entry_exec >= sentinel_index:
            raise ValueError(
                f"{context} compact trade entry_exec must stay within [0, {sentinel_index})"
            )
        if sig_exit_exec < entry_exec or sig_exit_exec > sentinel_index:
            raise ValueError(
                f"{context} compact trade sig_exit_exec must stay within "
                f"[entry_exec, {sentinel_index}]"
            )
        if entry_exec <= previous_entry:
            raise ValueError(f"{context} compact trades must stay strictly ordered by entry_exec")
        previous_entry = entry_exec
        parsed_trades.append(
            StageBCompactTradeV2(
                entry_exec=entry_exec,
                direction=direction,
                sig_exit_exec=sig_exit_exec,
            )
        )
    return tuple(parsed_trades)


def _parse_trade_exit_result_v2(
    *,
    payload: dict[str, object],
    context: str,
) -> StageBTradeExitResultV2:
    """
    Parse one expected single-trade exit result from a Stage B fixture case.

    Args:
        payload: Expected-result mapping.
        context: Human-readable validation context.
    Returns:
        StageBTradeExitResultV2: Typed expected exit result.
    Assumptions:
        Fixtures always record explicit exit reason, execution index, factor, and closed flag.
    Raises:
        ValueError: If one expected exit field is missing or unsupported.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    exit_reason = _required_str_v2(payload=payload, key="exit_reason", context=context)
    if exit_reason not in {"signal_exit", "tp", "sl", "close_on_end", "unclosed"}:
        raise ValueError(f"{context} exit_reason is unsupported: {exit_reason!r}")
    typed_exit_reason = cast(StageBTradeExitReasonLiteralV2, exit_reason)
    return StageBTradeExitResultV2(
        gross_factor=_required_decimal_v2(payload=payload, key="gross_factor", context=context),
        exit_exec=_required_int_v2(payload=payload, key="exit_exec", context=context),
        exit_reason=typed_exit_reason,
        closed=_required_bool_v2(payload=payload, key="closed", context=context),
    )


def _parse_best_cell_replay_result_v2(
    *,
    payload: dict[str, object],
    context: str,
) -> StageBBestCellReplayResultV2:
    """
    Parse one expected best-cell replay result from a Stage B fixture case.

    Args:
        payload: Expected-result mapping.
        context: Human-readable validation context.
    Returns:
        StageBBestCellReplayResultV2: Typed expected best-cell replay result.
    Assumptions:
        Metrics remain exact and explicit because R5-03 forbids fuzzy notebook comparisons.
    Raises:
        ValueError: If one expected replay field is missing.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    raw_metrics = _required_mapping_v2(payload=payload, key="metrics", context=context)
    return StageBBestCellReplayResultV2(
        best_tp_index=_required_int_v2(payload=payload, key="best_tp_index", context=context),
        best_sl_index=_required_int_v2(payload=payload, key="best_sl_index", context=context),
        trade_count=_required_int_v2(payload=payload, key="trade_count", context=context),
        metrics=StageBReplayMetricsV2(
            total_return=_required_decimal_v2(
                payload=raw_metrics,
                key="total_return",
                context=f"{context} metrics",
            ),
            max_drawdown=_required_decimal_v2(
                payload=raw_metrics,
                key="max_drawdown",
                context=f"{context} metrics",
            ),
            sharpe=_required_decimal_v2(
                payload=raw_metrics,
                key="sharpe",
                context=f"{context} metrics",
            ),
            winrate=_required_decimal_v2(
                payload=raw_metrics,
                key="winrate",
                context=f"{context} metrics",
            ),
            avg_trade_return=_required_decimal_v2(
                payload=raw_metrics,
                key="avg_trade_return",
                context=f"{context} metrics",
            ),
            avg_trade_bars=_required_decimal_v2(
                payload=raw_metrics,
                key="avg_trade_bars",
                context=f"{context} metrics",
            ),
            exposure=_required_decimal_v2(
                payload=raw_metrics,
                key="exposure",
                context=f"{context} metrics",
            ),
        ),
    )


def _optional_precedence_assertion_v2(
    *,
    expected: dict[str, object],
    direction: int,
    entry_exec: int,
    sig_exit_exec: int,
    tp_index: int,
    sl_index: int,
    hit_times: StageBHitTimesFixtureV2,
) -> str | None:
    """
    Validate explicit precedence assertion requirements for one single-trade fixture case.

    Args:
        expected: Expected-result mapping from the JSON catalog.
        direction: Trade direction (`+1` long, `-1` short).
        entry_exec: Execution-bar entry index.
        sig_exit_exec: Opposite-signal execution exit index or sentinel.
        tp_index: Selected TP level index.
        sl_index: Selected SL level index.
        hit_times: Strict `1m` hit-times tables with `sentinel_index`.
    Returns:
        str | None: Explicit precedence assertion string when one is authored.
    Assumptions:
        Equal-bar ties must never be implicit inside the fixture contract.
    Raises:
        ValueError: If a tie exists but no explicit precedence assertion is provided, or if the
            authored assertion does not match the winning rule.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    precedence_assertion = None
    if "precedence_assertion" in expected:
        precedence_assertion = _required_str_v2(
            payload=expected,
            key="precedence_assertion",
            context="trade_exit expected precedence_assertion",
        )
    sentinel_index = hit_times.sentinel_index
    lookup_exec = entry_exec + 1
    tp_exec = sentinel_index
    sl_exec = sentinel_index
    if lookup_exec < sentinel_index:
        if direction == 1:
            tp_exec = hit_times.long_tp[tp_index][lookup_exec]
            sl_exec = hit_times.long_sl[sl_index][lookup_exec]
        else:
            tp_exec = hit_times.short_tp[tp_index][lookup_exec]
            sl_exec = hit_times.short_sl[sl_index][lookup_exec]
    expected_assertion = None
    if sig_exit_exec < sentinel_index and sig_exit_exec <= min(tp_exec, sl_exec):
        if sig_exit_exec == tp_exec or sig_exit_exec == sl_exec:
            expected_assertion = "signal exit wins on equal bar"
    elif tp_exec < sentinel_index and tp_exec == sl_exec:
        expected_assertion = "SL wins TP tie"
    if expected_assertion is not None and precedence_assertion is None:
        raise ValueError(
            "trade_exit expected precedence_assertion is required for equal-bar precedence cases"
        )
    if precedence_assertion is not None and precedence_assertion != expected_assertion:
        raise ValueError(
            "trade_exit expected precedence_assertion does not match fixture inputs; "
            f"expected {expected_assertion!r}, got {precedence_assertion!r}"
        )
    return precedence_assertion


def _validate_bar_close_mapping_v2(
    *,
    bar_close_1m_idx: tuple[int, ...],
    sentinel_index: int,
    context: str,
) -> None:
    """
    Validate one artifact-backed `bar_close_1m_idx` sequence for Stage B golden fixtures.

    Args:
        bar_close_1m_idx: Request-timeframe close mapping into the `1m` timeline.
        sentinel_index: Execution timeline length used as the sentinel fallback.
        context: Human-readable validation context.
    Returns:
        None.
    Assumptions:
        Mapping indices stay monotone and point only to existing execution bars.
    Raises:
        ValueError: If one mapping index is negative, out of bounds, or non-monotone.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    if sentinel_index <= 0:
        raise ValueError(f"{context} sentinel_index must be > 0")
    _validate_monotone_int_sequence_v2(
        values=bar_close_1m_idx,
        field_name=f"{context} bar_close_1m_idx",
    )
    for mapped_close in bar_close_1m_idx:
        if mapped_close < 0 or mapped_close >= sentinel_index:
            raise ValueError(
                f"{context} bar_close_1m_idx values must stay within [0, {sentinel_index})"
            )


def _validate_trade_indexes_v2(
    *,
    entry_exec: int,
    sig_exit_exec: int,
    sentinel_index: int,
    context: str,
) -> None:
    """
    Validate one Stage B trade entry/signal-exit pair against the execution sentinel.

    Args:
        entry_exec: Execution-bar entry index.
        sig_exit_exec: Opposite-signal execution exit index or sentinel.
        sentinel_index: Execution timeline length used as the sentinel fallback.
        context: Human-readable validation context.
    Returns:
        None.
    Assumptions:
        Signal exits must not precede entries and cannot exceed the sentinel.
    Raises:
        ValueError: If trade indexes fall outside the documented Stage B bounds.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    if entry_exec < 0 or entry_exec >= sentinel_index:
        raise ValueError(f"{context} entry_exec must stay within [0, {sentinel_index})")
    if sig_exit_exec < entry_exec or sig_exit_exec > sentinel_index:
        raise ValueError(f"{context} sig_exit_exec must stay within [entry_exec, sentinel_index]")


def _validate_level_index_v2(*, index: int, length: int, field_name: str) -> None:
    """
    Validate one TP/SL level index against a deterministic fixture level count.

    Args:
        index: Candidate TP or SL index.
        length: Available level count for the selected level family.
        field_name: Human-readable validation context.
    Returns:
        None.
    Assumptions:
        Level indexes are zero-based and point to already-validated factor/table rows.
    Raises:
        ValueError: If the index is negative or outside the available level range.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    if index < 0 or index >= length:
        raise ValueError(f"{field_name} must stay within [0, {length}); got {index!r}")


def _validate_monotone_int_sequence_v2(*, values: tuple[int, ...], field_name: str) -> None:
    """
    Validate that one integer sequence is monotonically non-decreasing.

    Args:
        values: Candidate integer sequence.
        field_name: Human-readable validation context.
    Returns:
        None.
    Assumptions:
        Some fixture arrays may contain repeated values but must never decrease.
    Raises:
        ValueError: If one adjacent pair decreases.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    if any(left > right for left, right in zip(values, values[1:])):
        raise ValueError(f"{field_name} must be monotone non-decreasing")


def _validate_strictly_monotone_int_sequence_v2(
    *,
    values: tuple[int, ...],
    field_name: str,
) -> None:
    """
    Validate that one integer sequence is strictly increasing.

    Args:
        values: Candidate integer sequence.
        field_name: Human-readable validation context.
    Returns:
        None.
    Assumptions:
        Execution-bar opens must never share the same timestamp in these fixtures.
    Raises:
        ValueError: If one adjacent pair is not strictly increasing.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/notebook_tests/06_backtest_compute.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{field_name} must be strictly increasing")


def _required_mapping_v2(
    *,
    payload: dict[str, object],
    key: str,
    context: str,
) -> dict[str, object]:
    """
    Read one required object field from a JSON mapping with explicit diagnostics.

    Args:
        payload: Parent JSON object.
        key: Required field name.
        context: Human-readable validation context.
    Returns:
        dict[str, object]: Nested object value.
    Assumptions:
        The caller expects one JSON object and not an array/scalar/null.
    Raises:
        ValueError: If the field is missing or not an object.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context} field {key!r} must be an object")
    return value


def _required_list_v2(
    *,
    payload: dict[str, object],
    key: str,
    context: str,
) -> list[object]:
    """
    Read one required array field from a JSON mapping with explicit diagnostics.

    Args:
        payload: Parent JSON object.
        key: Required field name.
        context: Human-readable validation context.
    Returns:
        list[object]: Nested array value.
    Assumptions:
        The caller expects one JSON array and not an object/scalar/null.
    Raises:
        ValueError: If the field is missing or not an array.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{context} field {key!r} must be an array")
    return value


def _required_str_v2(*, payload: dict[str, object], key: str, context: str) -> str:
    """
    Read one required non-empty string field from a JSON mapping.

    Args:
        payload: Parent JSON object.
        key: Required field name.
        context: Human-readable validation context.
    Returns:
        str: Non-empty string value.
    Assumptions:
        String-valued metadata is authored without surrounding semantic whitespace.
    Raises:
        ValueError: If the field is missing, not a string, or empty after trimming.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} field {key!r} must be a non-empty string")
    return value


def _required_int_v2(*, payload: dict[str, object], key: str, context: str) -> int:
    """
    Read one required integer field from a JSON mapping.

    Args:
        payload: Parent JSON object.
        key: Required field name.
        context: Human-readable validation context.
    Returns:
        int: Integer value.
    Assumptions:
        Boolean values are rejected even though they are subclasses of `int`.
    Raises:
        ValueError: If the field is missing or not a plain integer.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} field {key!r} must be an int")
    return value


def _required_bool_v2(*, payload: dict[str, object], key: str, context: str) -> bool:
    """
    Read one required boolean field from a JSON mapping.

    Args:
        payload: Parent JSON object.
        key: Required field name.
        context: Human-readable validation context.
    Returns:
        bool: Boolean value.
    Assumptions:
        JSON booleans are authored explicitly and not encoded as integers.
    Raises:
        ValueError: If the field is missing or not a boolean.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{context} field {key!r} must be a bool")
    return value


def _required_decimal_v2(*, payload: dict[str, object], key: str, context: str) -> Decimal:
    """
    Read one required decimal-compatible numeric field from a JSON mapping.

    Args:
        payload: Parent JSON object.
        key: Required field name.
        context: Human-readable validation context.
    Returns:
        Decimal: Exact decimal value built from the authored JSON literal.
    Assumptions:
        Numeric fixture values are small and deterministic enough to parse through `Decimal(str())`.
    Raises:
        ValueError: If the field is missing or not numeric/string-numeric.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    value = payload.get(key)
    if isinstance(value, bool) or value is None or not isinstance(value, (int, float, str)):
        raise ValueError(f"{context} field {key!r} must be numeric")
    try:
        return Decimal(str(value))
    except Exception as exc:  # pragma: no cover - exercised through explicit ValueError path
        raise ValueError(f"{context} field {key!r} must be decimal-compatible") from exc


def _required_int_tuple_v2(
    *,
    payload: dict[str, object],
    key: str,
    context: str,
) -> tuple[int, ...]:
    """
    Read one required integer array field from a JSON mapping.

    Args:
        payload: Parent JSON object.
        key: Required field name.
        context: Human-readable validation context.
    Returns:
        tuple[int, ...]: Immutable integer tuple preserving authored ordering.
    Assumptions:
        Fixture ordering is semantic and must stay deterministic.
    Raises:
        ValueError: If one array element is not an integer.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    values = _required_list_v2(payload=payload, key=key, context=context)
    return tuple(
        _coerce_int_v2(value=value, context=f"{context} field {key!r}") for value in values
    )


def _required_decimal_tuple_v2(
    *,
    payload: dict[str, object],
    key: str,
    context: str,
) -> tuple[Decimal, ...]:
    """
    Read one required decimal-compatible numeric array field from a JSON mapping.

    Args:
        payload: Parent JSON object.
        key: Required field name.
        context: Human-readable validation context.
    Returns:
        tuple[Decimal, ...]: Immutable exact-decimal tuple preserving authored ordering.
    Assumptions:
        Fixture values are intentionally small and need stable cross-platform arithmetic.
    Raises:
        ValueError: If one array element is not numeric.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    values = _required_list_v2(payload=payload, key=key, context=context)
    return tuple(
        _coerce_decimal_v2(value=value, context=f"{context} field {key!r}") for value in values
    )


def _required_string_tuple_v2(
    *,
    payload: dict[str, object],
    key: str,
    context: str,
) -> tuple[str, ...]:
    """
    Read one required non-empty string array field from a JSON mapping.

    Args:
        payload: Parent JSON object.
        key: Required field name.
        context: Human-readable validation context.
    Returns:
        tuple[str, ...]: Immutable string tuple preserving authored ordering.
    Assumptions:
        Fixture ordering is semantic and strings are authored without empty placeholders.
    Raises:
        ValueError: If one array element is not a non-empty string.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    values = _required_list_v2(payload=payload, key=key, context=context)
    parsed_values: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context} field {key!r} must contain non-empty strings")
        parsed_values.append(value)
    return tuple(parsed_values)


def _required_table_v2(
    *,
    payload: dict[str, object],
    key: str,
    context: str,
    sentinel_index: int,
) -> tuple[tuple[int, ...], ...]:
    """
    Read one strict hit-times table field from a JSON mapping.

    Args:
        payload: Parent JSON object.
        key: Required field name.
        context: Human-readable validation context.
        sentinel_index: Execution timeline length used as the sentinel fallback.
    Returns:
        tuple[tuple[int, ...], ...]: Immutable strict hit-times table.
    Assumptions:
        Every row spans the entire execution timeline and stays bounded by sentinel.
    Raises:
        ValueError: If the table shape or any cell value is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
    Related:
      - tests/notebook_tests/05_hit_time_grid.ipynb
      - tests/unit/contexts/backtest/application/services/v2/test_hit_times_compute_v2.py
    """
    raw_rows = _required_list_v2(payload=payload, key=key, context=context)
    parsed_rows: list[tuple[int, ...]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, list):
            raise ValueError(f"{context} field {key!r} rows must be arrays")
        parsed_row = tuple(
            _coerce_int_v2(value=value, context=f"{context} field {key!r}") for value in raw_row
        )
        if len(parsed_row) != sentinel_index:
            raise ValueError(
                f"{context} field {key!r} rows must have length {sentinel_index}; "
                f"got {len(parsed_row)}"
            )
        if any(cell < 0 or cell > sentinel_index for cell in parsed_row):
            raise ValueError(
                f"{context} field {key!r} values must stay within [0, {sentinel_index}]"
            )
        parsed_rows.append(parsed_row)
    if len(parsed_rows) == 0:
        raise ValueError(f"{context} field {key!r} must contain at least one row")
    return tuple(parsed_rows)


def _coerce_int_v2(*, value: object, context: str) -> int:
    """
    Convert one JSON scalar into a strict integer with explicit diagnostics.

    Args:
        value: Candidate JSON scalar.
        context: Human-readable validation context.
    Returns:
        int: Strict integer value.
    Assumptions:
        Boolean values are rejected even though they subclass `int`.
    Raises:
        ValueError: If the scalar is not a plain integer.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} entries must be ints")
    return value


def _coerce_decimal_v2(*, value: object, context: str) -> Decimal:
    """
    Convert one JSON scalar into an exact `Decimal` with explicit diagnostics.

    Args:
        value: Candidate JSON scalar.
        context: Human-readable validation context.
    Returns:
        Decimal: Exact decimal value built from the authored JSON literal.
    Assumptions:
        Scalar values are numeric or decimal-compatible strings.
    Raises:
        ValueError: If the scalar is not decimal-compatible.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    if isinstance(value, bool) or value is None or not isinstance(value, (int, float, str)):
        raise ValueError(f"{context} entries must be numeric")
    try:
        return Decimal(str(value))
    except Exception as exc:  # pragma: no cover - exercised through explicit ValueError path
        raise ValueError(f"{context} entries must be decimal-compatible") from exc


__all__ = [
    "STAGE_B_GOLDEN_FIXTURE_EPIC_ID_V2",
    "STAGE_B_GOLDEN_FIXTURE_KIND_V2",
    "STAGE_B_GOLDEN_FIXTURE_MILESTONE_ID_V2",
    "STAGE_B_GOLDEN_FIXTURE_SCHEMA_VERSION_V2",
    "STAGE_B_GOLDEN_FIXTURE_SEMANTICS_V2",
    "StageBBestCellReplayCaseV2",
    "StageBBestCellReplayResultV2",
    "StageBCompactTradeV2",
    "StageBEntryMappingCaseV2",
    "StageBExecutionPricesV2",
    "StageBGoldenFixtureCatalogV2",
    "StageBGoldenFixtureCaseV2",
    "StageBHitTimesFixtureV2",
    "StageBLevelFactorsV2",
    "StageBReplayMetricsV2",
    "StageBTradeExitCaseV2",
    "StageBTradeExitReasonLiteralV2",
    "StageBTradeExitResultV2",
    "StageBTradeListCaseV2",
    "build_compact_trade_list_from_final_signal_v2",
    "evaluate_stage_b_trade_exit_v2",
    "execute_stage_b_golden_case_v2",
    "load_stage_b_best_cell_replay_reference_case_v2",
    "load_stage_b_golden_fixture_catalog_v2",
    "map_bar_close_1m_idx_to_entry_exec_v2",
    "map_signal_bars_to_entry_exec_v2",
    "read_stage_b_golden_fixture_payload_v2",
    "replay_stage_b_best_cell_v2",
    "serialize_stage_b_golden_fixture_payload_v2",
    "validate_stage_b_golden_fixture_payload_v2",
]
