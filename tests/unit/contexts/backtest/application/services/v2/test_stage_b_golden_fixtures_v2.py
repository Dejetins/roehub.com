from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import numpy as np
import pytest

from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
    backtest_job_repository as job_repository_module,
)
from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
    backtest_job_results_repository as job_results_repository_module,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestArtifactBackedStageBScorerV2,
    StageBBestCellReplayCaseV2,
    StageBEntryMappingCaseV2,
    StageBTradeExitCaseV2,
    StageBTradeListCaseV2,
    execute_stage_b_golden_case_v2,
    load_backtest_runtime_acceleration_benchmark_corpus_v2,
    load_stage_b_golden_fixture_catalog_v2,
    map_bar_close_1m_idx_to_entry_exec_v2,
    read_stage_b_golden_fixture_payload_v2,
    serialize_stage_b_golden_fixture_payload_v2,
    validate_stage_b_golden_fixture_payload_v2,
)
from trading.contexts.backtest.application.services.v2 import (
    artifact_backed_stage_b_scorer_v2 as stage_b_scorer_module,
)
from trading.contexts.backtest.application.services.v2 import (
    artifact_runtime_core_v2 as runtime_core_module,
)
from trading.contexts.backtest.application.services.v2 import (
    artifact_runtime_plan_v2 as runtime_plan_module,
)
from trading.contexts.backtest.application.services.v2.stage_b_golden_fixtures_v2 import (
    load_stage_b_best_cell_replay_reference_case_v2,
)
from trading.contexts.backtest.application.services.v2.trade_compactor_kernel import (
    StageACompactExactPayloadV2,
)
from trading.contexts.backtest.domain.entities import BacktestJobTopVariant
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


class _FinalistOnlyStageBScorerStubV2:
    """
    Minimal scorer stub proving finalist-only exact replay after Stage B breadth ranking.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    """

    def __init__(
        self,
        *,
        breadth_metrics_by_variant_key: dict[str, float],
        exact_metrics_by_variant_key: dict[str, dict[str, float]],
    ) -> None:
        """
        Store deterministic breadth and exact metrics for one Stage B finalist-only test run.

        Args:
            breadth_metrics_by_variant_key:
                Cheap breadth `total_return_pct` values keyed by `variant_key`.
            exact_metrics_by_variant_key:
                Exact finalist metrics keyed by `variant_key`.
        Returns:
            None.
        Assumptions:
            Breadth scoring is cheap and metric-light, while exact replay adds full summary
            metrics only for finalists retained by the Stage B heap.
        Raises:
            None.
        Side Effects:
            Initializes in-memory traces for exact replay and ranking-context assertions.
        """
        self._breadth_metrics_by_variant_key = breadth_metrics_by_variant_key
        self._exact_metrics_by_variant_key = exact_metrics_by_variant_key
        self.exact_calls: list[str] = []
        self.ranking_context_calls: list[tuple[str, str]] = []

    def configure_stage_ranking_context(
        self,
        *,
        stage: str,
        primary_metric: str,
    ) -> None:
        """
        Record the Stage B ranking context selected by the runtime.

        Args:
            stage: Stage literal passed by the runtime.
            primary_metric: Primary ranking metric selected for the current run.
        Returns:
            None.
        Assumptions:
            The finalist-only replay cutover keeps the explicit Stage B ranking context unchanged.
        Raises:
            None.
        Side Effects:
            Appends one `(stage, primary_metric)` tuple to the in-memory trace.
        """
        self.ranking_context_calls.append((stage, primary_metric))

    def score_variant_metric(
        self,
        *,
        stage: str,
        candles: SimpleNamespace,
        indicator_selections: tuple[object, ...],
        signal_params: dict[str, dict[str, object]],
        risk_params: dict[str, object],
        indicator_variant_key: str,
        variant_key: str,
    ) -> dict[str, float]:
        """
        Return cheap breadth metrics without triggering exact replay.

        Args:
            stage: Stage literal supplied by the runtime.
            candles: Unused candle payload for compatibility with the runtime signature.
            indicator_selections: Unused indicator selections for compatibility.
            signal_params: Unused signal params for compatibility.
            risk_params: Unused risk params for compatibility.
            indicator_variant_key: Unused indicator variant key for compatibility.
            variant_key: Deterministic full Stage B variant key.
        Returns:
            dict[str, float]: Breadth-only metrics carrying total return only.
        Assumptions:
            `RG-TTR` breadth stays on the cheap Stage B path and must not increment exact replay.
        Raises:
            AssertionError: If the runtime calls the breadth hook for a non-Stage-B request.
        Side Effects:
            None.
        """
        _ = candles, indicator_selections, signal_params, risk_params, indicator_variant_key
        if stage != "stage_b":
            raise AssertionError("finalist-only scorer stub expects Stage B breadth scoring only")
        total_return_pct = self._breadth_metrics_by_variant_key[variant_key]
        return {
            "total_return_pct": total_return_pct,
            "Total Return [%]": total_return_pct,
        }

    def score_variant_with_details(
        self,
        *,
        stage: str,
        candles: SimpleNamespace,
        indicator_selections: tuple[object, ...],
        signal_params: dict[str, dict[str, object]],
        risk_params: dict[str, object],
        indicator_variant_key: str,
        variant_key: str,
    ) -> SimpleNamespace:
        """
        Return exact finalist metrics and record one exact replay for the requested variant.

        Args:
            stage: Stage literal supplied by the runtime.
            candles: Unused candle payload for compatibility with the runtime signature.
            indicator_selections: Unused indicator selections for compatibility.
            signal_params: Unused signal params for compatibility.
            risk_params: Unused risk params for compatibility.
            indicator_variant_key: Unused indicator variant key for compatibility.
            variant_key: Deterministic full Stage B variant key.
        Returns:
            SimpleNamespace: Details-like payload exposing exact metrics for the finalist variant.
        Assumptions:
            Finalist-only replay happens after breadth heap selection and only for retained rows.
        Raises:
            AssertionError: If the runtime calls the details hook for a non-Stage-B request.
        Side Effects:
            Appends the replayed `variant_key` to the in-memory `exact_calls` trace.
        """
        _ = candles, indicator_selections, signal_params, risk_params, indicator_variant_key
        if stage != "stage_b":
            raise AssertionError("finalist-only scorer stub expects Stage B details scoring only")
        self.exact_calls.append(variant_key)
        return SimpleNamespace(metrics=self._exact_metrics_by_variant_key[variant_key])


def test_top_row_serializers_drop_non_finite_summary_metrics_before_json_persistence() -> None:
    """
    Verify both top-row persistence adapters sanitize non-finite summary metrics identically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Persisted top rows keep `total_return_pct` separately and may drop non-finite summary-only
        metrics without changing ranking semantics already decided upstream.
    Raises:
        AssertionError: If either repository keeps `Infinity`/`NaN` inside persisted summary JSON.
    Side Effects:
        Serializes one in-memory top-row payload through both repository helper paths.
    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
    """
    job_id = UUID("00000000-0000-0000-0000-0000000000c5")
    row = BacktestJobTopVariant(
        job_id=job_id,
        rank=1,
        variant_key="a" * 64,
        indicator_variant_key="b" * 64,
        variant_index=0,
        total_return_pct=12.5,
        payload_json={
            "direction_mode": "long-only",
            "execution_params": {"fee_pct": 0.0},
            "risk_params": {
                "tp_enabled": True,
                "tp_pct": 1.5,
                "sl_enabled": True,
                "sl_pct": 0.5,
            },
            "signal_params": {},
            "sizing_mode": "all_in",
        },
        updated_at=datetime(2026, 4, 14, tzinfo=timezone.utc),
        summary_metrics_json={
            "profit_factor": float("inf"),
            "return_over_max_drawdown": float("inf"),
            "sharpe_trades": float("nan"),
            "win_rate_pct": 50.0,
        },
        best_tp_pct=1.5,
        best_sl_pct=0.5,
    )

    job_snapshot_rows = job_repository_module._serialize_top_rows(job_id=job_id, rows=(row,))
    result_snapshot_rows = job_results_repository_module._serialize_top_rows(
        job_id=job_id,
        rows=(row,),
    )

    for serialized_rows in (job_snapshot_rows, result_snapshot_rows):
        summary_metrics = serialized_rows[0]["summary_metrics_json"]
        assert summary_metrics["total_return_pct"] == pytest.approx(12.5)
        assert summary_metrics["win_rate_pct"] == pytest.approx(50.0)
        assert "profit_factor" not in summary_metrics
        assert "return_over_max_drawdown" not in summary_metrics
        assert "sharpe_trades" not in summary_metrics
        json.dumps(serialized_rows, allow_nan=False)


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
    cast(Any, scorer)._resolve_risk_level_indexes_v2 = lambda *, risk_params: (0, 0)
    cast(Any, scorer)._fast_stage_b_search_for_base_variant_v2 = (
        lambda *, indicator_selections, signal_params, base_variant_key: SimpleNamespace(
            total_return_pct=np.asarray(((17.25,),), dtype=np.float64),
            base_variant_key=base_variant_key,
            retained_exact_payload="present",
        )
    )
    cast(Any, scorer)._exact_stage_b_cell_cache_v2 = (
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "retained_exact_payload must not disable the fast Stage B path for breadth scoring"
            )
        )
    )

    metrics = scorer.score_variant_metric(
        stage="stage_b",
        candles=cast(Any, SimpleNamespace()),
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

    assert (
        scorer._stage_a_payload_cache_by_base_variant_key[
            "base-variant-key"
        ].compact_trades
        == ()
    )
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
    cast(Any, scorer)._local_hit_times = SimpleNamespace(
        tp_values=np.asarray((0.01,), dtype=np.float32),
        sl_values=np.asarray((0.01,), dtype=np.float32),
    )
    scorer._base_variant_key_v2 = (
        lambda *, indicator_variant_key, signal_params: "base-variant-key"
    )
    exact_calls: list[str] = []
    cast(Any, scorer)._exact_stage_b_cell_cache_v2 = (
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
        candles=cast(Any, SimpleNamespace()),
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
    assert cast(Any, details.execution_outcome).authority == "exact"


def test_stage_b_exact_replay_observability_is_finalist_only() -> None:
    """
    Verify the scorer exposes benchmark-visible finalist-only exact replay metadata directly.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Benchmarks should not need private-cache introspection to observe `exact_replay_count`
        and replay scope after Stage B finalist replay.
    Raises:
        AssertionError: If the scorer stops exposing the additive replay observability hooks.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
    """
    scorer = object.__new__(BacktestArtifactBackedStageBScorerV2)
    scorer._stage_b_exact_cache_by_variant_key = {  # type: ignore[attr-defined]
        "variant-a": SimpleNamespace(),
        "variant-b": SimpleNamespace(),
    }

    assert scorer.stage_b_exact_replay_scope_v2() == "finalist-only"
    assert scorer.stage_b_exact_replay_count_v2() == 2


def test_stage_b_runtime_replays_exact_only_for_finalist_rows() -> None:
    """
    Verify the shared Stage B runtime exact-replays finalists only after cheap breadth ranking.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `RG-TTR` breadth ranking stays on the fast Stage B path, while exact replay upgrades only
        the retained finalist rows to final authority before the runtime returns them.
    Raises:
        AssertionError: If exact replay expands back to shortlist breadth or finalist rows keep
            breadth-only summary metrics.
    Side Effects:
        Exercises the shared Stage B runtime with a finalist-only scorer stub.
    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """
    runner = runtime_core_module.BacktestArtifactRuntimeRunnerV2()
    risk_variants = (
        runtime_plan_module.BacktestRiskVariantV2(
            risk_index=0,
            risk_params={
                "tp_enabled": True,
                "tp_pct": 1.0,
                "sl_enabled": True,
                "sl_pct": 1.0,
            },
        ),
        runtime_plan_module.BacktestRiskVariantV2(
            risk_index=1,
            risk_params={
                "tp_enabled": True,
                "tp_pct": 2.0,
                "sl_enabled": True,
                "sl_pct": 2.0,
            },
        ),
    )
    shortlist = tuple(
        runtime_core_module.BacktestStageAScoredVariantV2(
            base_variant=runtime_plan_module.BacktestStageABaseVariantV2(
                stage_a_index=index,
                indicator_selections=(),
                signal_params={},
                indicator_variant_key=f"{index + 1:064x}",
                base_variant_key=f"{index + 101:064x}",
            ),
            total_return_pct=float(100 - index),
        )
        for index in range(3)
    )
    runtime_plan = SimpleNamespace(
        stage_b_variants_total=len(shortlist) * len(risk_variants),
        risk_variants=risk_variants,
        stage_b_execution_mode=lambda: "in_process",
    )
    template = SimpleNamespace(
        direction_mode="long-only",
        sizing_mode="fixed_quote",
        execution_params={},
    )
    all_tasks = runtime_core_module.iter_stage_b_tasks_v2(
        template=cast(Any, template),
        runtime_plan=cast(Any, runtime_plan),
        shortlist=shortlist,
    )
    breadth_total_return_by_variant_key = {
        all_tasks[0].variant_key: 5.0,
        all_tasks[1].variant_key: 10.0,
        all_tasks[2].variant_key: 7.0,
        all_tasks[3].variant_key: 11.0,
        all_tasks[4].variant_key: 4.0,
        all_tasks[5].variant_key: 3.0,
    }
    exact_metrics_by_variant_key = {
        task.variant_key: {
            "total_return_pct": breadth_total_return_by_variant_key[task.variant_key],
            "Total Return [%]": breadth_total_return_by_variant_key[task.variant_key],
            "trade_count": float(task.variant_index + 10),
        }
        for task in all_tasks
    }
    scorer = _FinalistOnlyStageBScorerStubV2(
        breadth_metrics_by_variant_key=breadth_total_return_by_variant_key,
        exact_metrics_by_variant_key=exact_metrics_by_variant_key,
    )

    rows, tasks_by_variant_key = runner.run_stage_b_or_finalize_no_risk(
        template=cast(Any, template),
        runtime_plan=cast(Any, runtime_plan),
        shortlist=shortlist,
        candles=cast(Any, SimpleNamespace()),
        scorer=cast(Any, scorer),
        top_k_limit=2,
    )

    assert scorer.ranking_context_calls == [("stage_b", "total_return_pct")]
    assert tuple(row.variant_key for row in rows) == (
        all_tasks[3].variant_key,
        all_tasks[1].variant_key,
    )
    assert scorer.exact_calls == [row.variant_key for row in rows]
    assert tuple(sorted(tasks_by_variant_key.keys())) == tuple(
        sorted(row.variant_key for row in rows)
    )
    assert rows[0].summary_metrics_json["trade_count"] == 13.0
    assert rows[1].summary_metrics_json["trade_count"] == 11.0


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
