from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trading.contexts.backtest.adapters.outbound import (
    BacktestRuntimeConfig,
    load_backtest_runtime_config,
)
from trading.contexts.backtest_artifacts.application.services.v2.adaptive_selector_v2 import (
    AdaptiveSelectorPlanningEvidenceV2,
    CostModelAdaptiveExecutionSelectorV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.benchmark_corpus_v2 import (
    BacktestRuntimeAccelerationBenchmarkCorpusV2,
    load_backtest_runtime_acceleration_benchmark_corpus_v2,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_BENCHMARK_CORPUS_FIXTURE_PATH = (
    _FIXTURES_DIR / "backtest_runtime_acceleration_benchmark_corpus_v1.json"
)
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _runtime_config(*, env_name: str) -> BacktestRuntimeConfig:
    """
    Load one committed env-specific Backtest runtime config for rollout perf-smoke checks.

    Args:
        env_name: Runtime environment literal (`dev`, `test`, or `prod`).
    Returns:
        BacktestRuntimeConfig: Parsed fail-fast runtime config for the environment.
    Assumptions:
        Perf-smoke tests read committed YAML only and do not mutate runtime policy state.
    Raises:
        FileNotFoundError: If the committed env config is missing.
        ValueError: If the env config violates the typed runtime contract.
    Side Effects:
        Reads one committed `configs/<env>/backtest.yaml` file from disk.
    """
    return load_backtest_runtime_config(_REPO_ROOT / "configs" / env_name / "backtest.yaml")


def _benchmark_corpus() -> BacktestRuntimeAccelerationBenchmarkCorpusV2:
    """
    Load the committed benchmark corpus used as the F2 rollout evidence anchor.

    Args:
        None.
    Returns:
        BacktestRuntimeAccelerationBenchmarkCorpusV2: Typed committed benchmark corpus.
    Assumptions:
        The corpus remains a test/documentation evidence surface and not a runtime selector input.
    Raises:
        OSError: If the committed benchmark fixture cannot be read.
        ValueError: If the fixture payload violates the typed corpus contract.
    Side Effects:
        Reads one committed JSON fixture from disk.
    """
    return load_backtest_runtime_acceleration_benchmark_corpus_v2(
        path=_BENCHMARK_CORPUS_FIXTURE_PATH
    )


def _large_ma_run_evidence(*, config: BacktestRuntimeConfig) -> AdaptiveSelectorPlanningEvidenceV2:
    """
    Build large background-capable planning evidence at the shipped `hybrid_family` threshold.

    Args:
        config: Parsed env-specific runtime config.
    Returns:
        AdaptiveSelectorPlanningEvidenceV2: Deterministic large-run selector evidence.
    Assumptions:
        Using the committed `hybrid_family` thresholds keeps this perf-smoke aligned with the
        shipped rollout policy while ensuring the more conservative `hybrid_conservative`
        candidate is also eligible.
    Raises:
        ValueError: If the config-derived selector policy contains invalid thresholds.
    Side Effects:
        None.
    """
    family_policy = config.adaptive_selector_policy.hybrid_family
    return AdaptiveSelectorPlanningEvidenceV2(
        grid_cardinality=family_policy.min_grid_cardinality,
        stage_a_variants_total=family_policy.min_stage_a_variants_total,
        stage_b_variants_total=family_policy.min_stage_b_variants_total,
        estimated_memory_bytes=family_policy.min_estimated_memory_bytes,
        runtime_mode="background_capable",
        indicator_ids=("ma.fast", "ma.slow"),
    )


def _small_sync_evidence(
    *,
    config: BacktestRuntimeConfig,
    corpus: BacktestRuntimeAccelerationBenchmarkCorpusV2,
) -> AdaptiveSelectorPlanningEvidenceV2:
    """
    Build small sync-sized planning evidence anchored to the committed `small_grid_overhead` slice.

    Args:
        config: Parsed env-specific runtime config.
        corpus: Typed committed benchmark corpus.
    Returns:
        AdaptiveSelectorPlanningEvidenceV2: Deterministic small-run selector evidence.
    Assumptions:
        The `small_grid_overhead` slice protects exact-first behavior for small sync runs, so its
        scale stays well below the shipped hybrid thresholds.
    Raises:
        KeyError: If the committed corpus no longer exposes `small_grid_overhead`.
        ValueError: If the resulting selector evidence violates its typed contract.
    Side Effects:
        None.
    """
    small_slice = corpus.slice_for_id(slice_id="small_grid_overhead")
    if small_slice.synthetic_run_spec is None:
        raise AssertionError("small_grid_overhead synthetic_run_spec is required")
    exact_small_budget = config.execution_profiles.default_profile().launch_budget
    return AdaptiveSelectorPlanningEvidenceV2(
        grid_cardinality=small_slice.synthetic_run_spec.expected_stage_a_variants_total,
        stage_a_variants_total=small_slice.synthetic_run_spec.expected_stage_a_variants_total,
        stage_b_variants_total=small_slice.synthetic_run_spec.expected_stage_b_variants_total,
        estimated_memory_bytes=max(1, exact_small_budget.max_estimated_memory_bytes // 2),
        runtime_mode="sync_inline",
        indicator_ids=("ma.fast", "ma.slow"),
    )


def test_dev_active_rollout_keeps_hybrid_family_narrower_than_hybrid_conservative() -> None:
    """
    Verify `dev` may apply selective defaulting for large runs while keeping family rollout shadow.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The committed `memory_footprint` slice anchors large-run hybrid evidence, while the
        shipped `dev` config keeps `hybrid_family` narrower than `hybrid_conservative`.
    Raises:
        AssertionError: If `dev` stops applying large-run conservative rollout or widens family
            rollout beyond the committed config.
    Side Effects:
        None.
    """
    corpus = _benchmark_corpus()
    config = _runtime_config(env_name="dev")
    selector = CostModelAdaptiveExecutionSelectorV2()
    memory_slice = corpus.slice_for_id(slice_id="memory_footprint")

    decision = selector.select(
        evidence=_large_ma_run_evidence(config=config),
        execution_profiles=config.execution_profiles,
        policy=config.adaptive_selector_policy,
    )

    assert memory_slice.rollout_scope == "hybrid_rollout"
    assert config.adaptive_selector_policy.mode == "active"
    assert config.adaptive_selector_policy.hybrid_conservative.rollout_mode == "active"
    assert config.adaptive_selector_policy.hybrid_family.rollout_mode == "shadow"
    assert decision.effective_profile.mode == "hybrid_conservative"
    assert decision.recommended_profile.mode == "hybrid_family"
    assert decision.recommendation_applied is False


@pytest.mark.parametrize("env_name", ("test", "prod"))
def test_shadow_envs_keep_large_ma_runs_exact_while_reporting_family_recommendation(
    env_name: str,
) -> None:
    """
    Verify `test` and `prod` stay in shadow mode for large pure `ma.` selector recommendations.

    Args:
        env_name: Environment literal under test.
    Returns:
        None.
    Assumptions:
        The committed family-plugin evidence surface justifies `hybrid_family` recommendation
        visibility, but the env rollout remains `shadow` by default outside `dev`.
    Raises:
        AssertionError: If shadow environments silently switch execution away from exact fallback.
    Side Effects:
        None.
    """
    config = _runtime_config(env_name=env_name)
    selector = CostModelAdaptiveExecutionSelectorV2()

    decision = selector.select(
        evidence=_large_ma_run_evidence(config=config),
        execution_profiles=config.execution_profiles,
        policy=config.adaptive_selector_policy,
    )

    assert config.adaptive_selector_policy.mode == "shadow"
    assert config.adaptive_selector_policy.hybrid_family.rollout_mode == "shadow"
    assert decision.effective_profile.mode == "exact_parallel"
    assert decision.recommended_profile.mode == "hybrid_family"
    assert decision.recommendation_applied is False


def test_dev_active_rollout_keeps_small_sync_runs_exact_first() -> None:
    """
    Verify `dev` active rollout still keeps `exact_small` authoritative for small sync-sized runs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The `small_grid_overhead` corpus slice is the committed evidence anchor for preserving
        exact-first behavior on small sync requests.
    Raises:
        AssertionError: If active rollout promotes the small sync slice away from `exact_small`.
    Side Effects:
        None.
    """
    corpus = _benchmark_corpus()
    config = _runtime_config(env_name="dev")
    selector = CostModelAdaptiveExecutionSelectorV2()
    small_slice = corpus.slice_for_id(slice_id="small_grid_overhead")

    decision = selector.select(
        evidence=_small_sync_evidence(config=config, corpus=corpus),
        execution_profiles=config.execution_profiles,
        policy=config.adaptive_selector_policy,
    )

    assert small_slice.execution_profile_mode == "exact_small"
    assert config.adaptive_selector_policy.mode == "active"
    assert decision.effective_profile.mode == "exact_small"
    assert decision.recommended_profile.mode == "exact_small"
    assert decision.recommendation_applied is False


def test_prod_opt_in_phase_is_explicit_without_becoming_active_by_default() -> None:
    """
    Verify the explicit prod `opt_in` phase stays distinct from both shadow and active rollout.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The roadmap requires an explicit prod opt-in phase before selective defaulting becomes
        live, while the committed prod config may still remain `shadow` by default.
    Raises:
        AssertionError: If the explicit opt-in policy literal is collapsed back into shadow or
            active semantics.
    Side Effects:
        None.
    """
    config = _runtime_config(env_name="prod")
    selector = CostModelAdaptiveExecutionSelectorV2()
    opt_in_policy = replace(config.adaptive_selector_policy, mode="opt_in")

    decision = selector.select(
        evidence=_large_ma_run_evidence(config=config),
        execution_profiles=config.execution_profiles,
        policy=opt_in_policy,
    )

    assert config.adaptive_selector_policy.mode == "shadow"
    assert opt_in_policy.mode == "opt_in"
    assert decision.policy_mode == "opt_in"
    assert decision.effective_profile.mode == "exact_parallel"
    assert decision.recommended_profile.mode == "hybrid_family"
    assert decision.recommendation_applied is False
