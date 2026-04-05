from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.backtest.adapters.outbound.config import (
    build_backtest_runtime_config_hash,
    load_backtest_runtime_config,
    resolve_backtest_config_path,
    resolve_backtest_env_name,
)

_DEFAULT_JOBS_BLOCK = """
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1.0
    lease_seconds: 60
    heartbeat_seconds: 15
    snapshot_seconds: 30
    snapshot_variants_step: 1000
    parallel_workers: 1
""".rstrip()

_DEFAULT_SYNC_BLOCK = """
  sync:
    sync_deadline_seconds: 55.0
""".rstrip()



def _write_backtest_config(tmp_path: Path, *, body: str, filename: str = "backtest.yaml") -> Path:
    """
    Write temporary Backtest runtime YAML used by config-loader tests.

    Args:
        tmp_path: pytest temporary directory fixture.
        body: Full YAML content.
    Returns:
        Path: Written config path.
    Assumptions:
        Input text is valid UTF-8.
    Raises:
        OSError: If write operation fails.
    Side Effects:
        Creates one temp YAML file.
    """
    config_path = tmp_path / filename
    config_path.write_text(body, encoding="utf-8")
    return config_path



def test_load_backtest_runtime_config_reads_yaml_values() -> None:
    """
    Verify loader parses documented Backtest defaults from source-of-truth YAML.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Config schema follows BKT Milestone 5 runtime contract.
    Raises:
        AssertionError: If parsed values differ from YAML payload.
    Side Effects:
        None.
    """
    config = load_backtest_runtime_config(Path("configs/dev/backtest.yaml"))

    assert config.version == 1
    assert config.warmup_bars_default == 200
    assert config.top_k_default == 300
    assert config.preselect_default == 20000
    assert config.ranking.primary_metric_default == "total_return_pct"
    assert config.ranking.secondary_metric_default is None
    assert config.contracts.allowed_request_timeframes == (
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "1d",
        "2d",
        "3d",
    )
    assert config.contracts.forbidden_request_timeframes == ("1m", "5m")
    assert config.contracts.top_n_default == 100
    assert config.contracts.top_n_max == 300
    assert config.contracts.ranking_metrics == (
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "sharpe_trades",
        "win_rate_pct",
    )
    assert config.contracts.sortable_summary_columns == (
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "sharpe_trades",
        "win_rate_pct",
        "trade_count",
        "avg_trade_ret_pct",
        "avg_trade_exec_bars",
        "exposure_pct",
        "best_tp_pct",
        "best_sl_pct",
    )
    assert config.contracts.signals_v1_params_path == "signals.v1.params"
    assert config.contracts.signals_v1_params_policy == "default-only"
    assert config.contracts.risk_model == "signal_tf + 1m_risk"
    assert config.contracts.execution_mode == "auto"
    assert config.contracts.auto_preflight_enabled is True
    assert config.contracts.auto_fallback_to_background_enabled is True
    assert config.execution_profiles.default_mode == "exact_small"
    assert config.adaptive_selector_policy.mode == "active"
    assert config.adaptive_selector_policy.hybrid_conservative.rollout_mode == "active"
    assert config.adaptive_selector_policy.hybrid_family.rollout_mode == "shadow"
    assert config.adaptive_selector_policy.hybrid_conservative.min_grid_cardinality == 6000
    assert config.adaptive_selector_policy.hybrid_family.min_stage_b_variants_total == 80000
    assert tuple(profile.mode for profile in config.execution_profiles.available_profiles) == (
        "exact_small",
        "exact_parallel",
        "hybrid_conservative",
        "hybrid_family",
    )
    assert config.execution_profiles.default_profile().feature_flags.runtime_enabled is True
    assert config.execution_profiles.available_profiles[1].feature_flags.runtime_enabled is True
    assert (
        config.execution_profiles.available_profiles[1].feature_flags.parallel_stage_b_enabled
        is True
    )
    assert (
        config.execution_profiles.default_profile().launch_budget.max_stage_a_variants_total
        == 1500
    )
    assert (
        config.execution_profiles.available_profiles[1].launch_budget.max_stage_b_variants_total
        == 180000
    )
    assert config.execution_profiles.available_profiles[1].progress_weights.stage_a == 35
    assert (
        config.execution_profiles.default_profile().planning_budget_ms
        == 25
    )
    assert config.execution_profiles.default_profile().family_plugin_budget_ms == 10
    assert (
        config.execution_profiles.default_profile().shortlist_config.scoring.activity_ratio_weight
        == 0.4
    )
    assert (
        config.execution_profiles.default_profile().shortlist_config.retention.diversity_buckets
        == ("activity_band", "direction_band")
    )
    assert (
        config.execution_profiles.available_profiles[2].shortlist_config.retention.max_per_bucket
        == 750
    )
    assert config.execution_profiles.available_profiles[2].feature_flags.runtime_enabled is True
    assert (
        config.execution_profiles.available_profiles[2].feature_flags
        .heuristic_shortlist_enabled
        is True
    )
    assert (
        config.execution_profiles.available_profiles[3].shortlist_config.scoring
        .transition_ratio_weight
        == 0.3
    )
    assert config.execution_profiles.available_profiles[3].feature_flags.runtime_enabled is True
    assert (
        config.execution_profiles.available_profiles[3].feature_flags
        .heuristic_shortlist_enabled
        is True
    )
    assert (
        config.execution_profiles.available_profiles[3].feature_flags
        .family_plugin_enabled
        is True
    )
    assert config.guards.max_variants_per_compute == 600000
    assert config.guards.max_compute_bytes_total == 5368709120
    assert config.cpu.max_numba_threads == 4
    assert config.sync.sync_deadline_seconds == 55.0
    assert config.reporting.top_trades_n_default == 3
    assert config.reporting.eager_top_reports_enabled is False
    assert config.execution.init_cash_quote_default == 10000.0
    assert config.execution.fixed_quote_default == 100.0
    assert config.execution.safe_profit_percent_default == 30.0
    assert config.execution.slippage_pct_default == 0.01
    assert dict(config.execution.fee_pct_default_by_market_id) == {
        1: 0.075,
        2: 0.1,
        3: 0.075,
        4: 0.1,
    }
    assert config.jobs.enabled is True
    assert config.jobs.top_k_persisted_default == 300
    assert config.jobs.max_active_jobs_per_user == 3
    assert config.jobs.claim_poll_seconds == 1.0
    assert config.jobs.lease_seconds == 60
    assert config.jobs.heartbeat_seconds == 15
    assert config.jobs.snapshot_seconds == 30
    assert config.jobs.snapshot_variants_step == 1000
    assert config.jobs.parallel_workers == 1


@pytest.mark.parametrize(
    ("config_path", "expected_mode"),
    (
        (Path("configs/test/backtest.yaml"), "shadow"),
        (Path("configs/prod/backtest.yaml"), "shadow"),
    ),
)
def test_load_backtest_runtime_config_reads_env_specific_selector_modes(
    config_path: Path,
    expected_mode: str,
) -> None:
    """
    Verify committed env configs keep selector rollout explicit and conservative by environment.

    Args:
        config_path: Environment-specific runtime config path.
        expected_mode: Expected env-level adaptive-selector mode.
    Returns:
        None.
    Assumptions:
        Candidate rollout caps remain explicit so `hybrid_family` may stay narrower than
        `hybrid_conservative` even when env-level settings later change.
    Raises:
        AssertionError: If committed env configs drift from the F2 rollout contract.
    Side Effects:
        None.
    """
    config = load_backtest_runtime_config(config_path)

    assert config.adaptive_selector_policy.mode == expected_mode
    assert config.adaptive_selector_policy.hybrid_conservative.rollout_mode == "active"
    assert config.adaptive_selector_policy.hybrid_family.rollout_mode == "shadow"



def test_load_backtest_runtime_config_uses_defaults_when_optional_keys_absent(
    tmp_path: Path,
) -> None:
    """
    Verify optional non-jobs/non-sync scalar keys fallback to documented defaults.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `backtest.jobs.*` and `backtest.sync.*` keys are strict-required and provided.
    Raises:
        AssertionError: If fallback defaults are not applied.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body=(
            """
version: 1
backtest:
"""
            + _DEFAULT_SYNC_BLOCK
            + "\n"
            + _DEFAULT_JOBS_BLOCK
        ).strip(),
    )

    config = load_backtest_runtime_config(config_path)

    assert config.warmup_bars_default == 200
    assert config.top_k_default == 300
    assert config.preselect_default == 20000
    assert config.ranking.primary_metric_default == "total_return_pct"
    assert config.ranking.secondary_metric_default is None
    assert config.contracts.allowed_request_timeframes == (
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "1d",
        "2d",
        "3d",
    )
    assert config.contracts.forbidden_request_timeframes == ("1m", "5m")
    assert config.contracts.top_n_default == 100
    assert config.contracts.top_n_max == 300
    assert config.contracts.signals_v1_params_path == "signals.v1.params"
    assert config.contracts.signals_v1_params_policy == "default-only"
    assert config.contracts.risk_model == "signal_tf + 1m_risk"
    assert config.contracts.execution_mode == "auto"
    assert config.contracts.auto_preflight_enabled is True
    assert config.contracts.auto_fallback_to_background_enabled is True
    assert config.execution_profiles.default_mode == "exact_small"
    assert config.adaptive_selector_policy.mode == "disabled"
    assert config.adaptive_selector_policy.hybrid_conservative.rollout_mode == "active"
    assert config.adaptive_selector_policy.hybrid_family.rollout_mode == "active"
    assert tuple(profile.mode for profile in config.execution_profiles.available_profiles) == (
        "exact_small",
        "exact_parallel",
        "hybrid_conservative",
        "hybrid_family",
    )
    assert (
        config.execution_profiles.default_profile().launch_budget.max_stage_a_variants_total
        == 1500
    )
    assert config.execution_profiles.default_profile().progress_weights.stage_b == 70
    assert (
        config.execution_profiles.default_profile().shortlist_config.scoring
        .active_span_ratio_weight
        == 0.1
    )
    assert (
        config.execution_profiles.available_profiles[2].shortlist_config.retention.max_per_bucket
        == 750
    )
    assert config.guards.max_variants_per_compute == 600000
    assert config.guards.max_compute_bytes_total == 5 * 1024**3
    assert config.cpu.max_numba_threads > 0
    assert config.sync.sync_deadline_seconds == 55.0
    assert config.reporting.top_trades_n_default == 3
    assert config.reporting.eager_top_reports_enabled is False
    assert config.execution.init_cash_quote_default == 10000.0
    assert config.execution.fixed_quote_default == 100.0
    assert config.execution.safe_profit_percent_default == 30.0
    assert config.execution.slippage_pct_default == 0.01
    assert dict(config.execution.fee_pct_default_by_market_id) == {
        1: 0.075,
        2: 0.1,
        3: 0.075,
        4: 0.1,
    }



def test_load_backtest_runtime_config_requires_jobs_section(tmp_path: Path) -> None:
    """
    Verify runtime loader fails fast when `backtest.jobs` section is absent.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Milestone 5 contract marks jobs section as strict-required.
    Raises:
        AssertionError: If missing jobs section does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest: {}
""".strip(),
    )

    with pytest.raises(ValueError, match="jobs"):
        load_backtest_runtime_config(config_path)



def test_load_backtest_runtime_config_requires_jobs_required_keys(tmp_path: Path) -> None:
    """
    Verify runtime loader fails fast for missing strict-required jobs key.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Required key list includes `backtest.jobs.top_k_persisted_default`.
    Raises:
        AssertionError: If missing required key does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="top_k_persisted_default"):
        load_backtest_runtime_config(config_path)



def test_load_backtest_runtime_config_requires_sync_section(tmp_path: Path) -> None:
    """
    Verify runtime loader fails fast when `backtest.sync` section is absent.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Sync deadline is a strict-required runtime knob for sync API route.
    Raises:
        AssertionError: If missing sync section does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1.0
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="sync"):
        load_backtest_runtime_config(config_path)


def test_resolve_backtest_config_path_precedence() -> None:
    """
    Verify path resolution precedence is override env first, then env fallback.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Fallback format is `configs/<ROEHUB_ENV>/backtest.yaml`.
    Raises:
        AssertionError: If precedence order differs from runtime contract.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "prod",
        "ROEHUB_BACKTEST_CONFIG": "configs/test/custom-backtest.yaml",
    }

    assert resolve_backtest_config_path(environ=environ) == Path(
        "configs/test/custom-backtest.yaml"
    )

    assert resolve_backtest_config_path(environ={"ROEHUB_ENV": "test"}) == Path(
        "configs/test/backtest.yaml"
    )

    assert resolve_backtest_config_path(environ={}) == Path("configs/dev/backtest.yaml")



def test_resolve_backtest_config_path_rejects_invalid_env_name() -> None:
    """
    Verify unsupported `ROEHUB_ENV` value fails fast with deterministic message.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Allowed environment literals are `dev`, `prod`, and `test`.
    Raises:
        AssertionError: If invalid env value does not raise ValueError.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="ROEHUB_ENV"):
        resolve_backtest_config_path(environ={"ROEHUB_ENV": "stage"})


def test_resolve_backtest_env_name_normalizes_and_defaults() -> None:
    """
    Verify shared env resolver normalizes case/whitespace and defaults to `dev`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Artifact and runtime config path helpers share the same env-name contract.
    Raises:
        AssertionError: If normalized/default env value is incorrect.
    Side Effects:
        None.
    """
    assert resolve_backtest_env_name(environ={}) == "dev"
    assert resolve_backtest_env_name(environ={"ROEHUB_ENV": " PROD "}) == "prod"



def test_load_backtest_runtime_config_reads_execution_overrides(tmp_path: Path) -> None:
    """
    Verify loader parses explicit execution/jobs defaults with fail-fast semantics.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `backtest.execution` and `backtest.jobs` sections follow runtime schema.
    Raises:
        AssertionError: If parsed values mismatch YAML payload.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  warmup_bars_default: 10
  top_k_default: 20
  preselect_default: 30
  ranking:
    primary_metric_default: SHARPE_TRADES
    secondary_metric_default: WIN_RATE_PCT
  contracts:
    request_timeframes:
      allowed: [30m, 1h]
      forbidden: [1m, 5m]
    summary:
      top_n_default: 25
      top_n_max: 40
      ranking_metrics: [profit_factor, total_return_pct]
      sortable_columns: [profit_factor, total_return_pct, best_tp_pct]
    signals:
      params_path: signals.v1.params
      params_policy: default-only
    execution:
      risk_model: signal_tf + 1m_risk
    launch:
      execution_mode: auto
      auto_preflight_enabled: true
      auto_fallback_to_background_enabled: true
  execution_profiles:
    default: exact_small
    adaptive_selector:
      mode: shadow
      hybrid_conservative:
        rollout_mode: active
        min_grid_cardinality: 3100
        min_stage_a_variants_total: 3200
        min_stage_b_variants_total: 3300
        min_estimated_memory_bytes: 3400
        minimum_exceeded_signals: 2
      hybrid_family:
        rollout_mode: shadow
        min_grid_cardinality: 4100
        min_stage_a_variants_total: 4200
        min_stage_b_variants_total: 4300
        min_estimated_memory_bytes: 4400
        minimum_exceeded_signals: 4
    profiles:
      - mode: exact_small
        family_plugin_budget_ms: 7
        planning_budget_ms: 15
        shortlist:
          enabled: false
          scoring:
            activity_ratio_weight: 0.50
            direction_balance_weight: 0.10
            transition_ratio_weight: 0.20
            active_span_ratio_weight: 0.20
          retention:
            diversity_buckets:
              - activity_band
              - transition_band
        launch_budget:
          max_stage_a_variants_total: 120
          max_stage_b_variants_total: 900
          max_estimated_memory_bytes: 104857600
        progress:
          stage_a: 20
          stage_b: 75
          finalizing: 5
        parallelism:
          stage_a_workers: 1
          stage_b_workers: 1
        feature_flags:
          runtime_enabled: true
          heuristic_shortlist_enabled: false
          parallel_stage_b_enabled: false
          family_plugin_enabled: false
      - mode: exact_parallel
        family_plugin_budget_ms: 17
        planning_budget_ms: 35
        shortlist:
          enabled: false
          scoring:
            activity_ratio_weight: 0.45
            direction_balance_weight: 0.15
            transition_ratio_weight: 0.20
            active_span_ratio_weight: 0.20
          retention:
            diversity_buckets:
              - activity_band
              - direction_band
        launch_budget:
          max_stage_a_variants_total: 3200
          max_stage_b_variants_total: 24000
          max_estimated_memory_bytes: 536870912
        progress:
          stage_a: 30
          stage_b: 65
          finalizing: 5
        parallelism:
          stage_a_workers: 1
          stage_b_workers: 3
        feature_flags:
          runtime_enabled: false
          heuristic_shortlist_enabled: false
          parallel_stage_b_enabled: false
          family_plugin_enabled: false
      - mode: hybrid_conservative
        family_plugin_budget_ms: 27
        planning_budget_ms: 55
        shortlist:
          enabled: true
          max_candidates: 1500
          scoring:
            activity_ratio_weight: 0.55
            direction_balance_weight: 0.05
            transition_ratio_weight: 0.20
            active_span_ratio_weight: 0.20
          retention:
            diversity_buckets:
              - activity_band
              - transition_band
            max_per_bucket: 200
        progress:
          stage_a: 45
          stage_b: 50
          finalizing: 5
        parallelism:
          stage_a_workers: 1
          stage_b_workers: 3
        feature_flags:
          runtime_enabled: false
          heuristic_shortlist_enabled: false
          parallel_stage_b_enabled: false
          family_plugin_enabled: false
      - mode: hybrid_family
        family_plugin_budget_ms: 37
        planning_budget_ms: 65
        shortlist:
          enabled: true
          max_candidates: 750
          scoring:
            activity_ratio_weight: 0.30
            direction_balance_weight: 0.20
            transition_ratio_weight: 0.25
            active_span_ratio_weight: 0.25
          retention:
            diversity_buckets:
              - activity_band
              - direction_band
            max_per_bucket: 100
        parallelism:
          stage_a_workers: 1
          stage_b_workers: 2
        feature_flags:
          runtime_enabled: false
          heuristic_shortlist_enabled: false
          parallel_stage_b_enabled: false
          family_plugin_enabled: false
  reporting:
    top_trades_n_default: 5
  guards:
    max_variants_per_compute: 1200
    max_compute_bytes_total: 1234567
  cpu:
    max_numba_threads: 6
  sync:
    sync_deadline_seconds: 42.5
  execution:
    init_cash_quote_default: 5000
    fixed_quote_default: 250
    safe_profit_percent_default: 15
    slippage_pct_default: 0.05
    fee_pct_default_by_market_id:
      1: 0.05
      8: 0.2
  jobs:
    enabled: false
    top_k_persisted_default: 42
    max_active_jobs_per_user: 8
    claim_poll_seconds: 0.5
    lease_seconds: 120
    heartbeat_seconds: 20
    snapshot_seconds: 10
    snapshot_variants_step: 200
    parallel_workers: 4
""".strip(),
    )

    config = load_backtest_runtime_config(config_path)

    assert config.warmup_bars_default == 10
    assert config.top_k_default == 20
    assert config.preselect_default == 30
    assert config.ranking.primary_metric_default == "sharpe_trades"
    assert config.ranking.secondary_metric_default == "win_rate_pct"
    assert config.contracts.allowed_request_timeframes == ("30m", "1h")
    assert config.contracts.forbidden_request_timeframes == ("1m", "5m")
    assert config.contracts.top_n_default == 25
    assert config.contracts.top_n_max == 40
    assert config.contracts.ranking_metrics == ("profit_factor", "total_return_pct")
    assert config.contracts.sortable_summary_columns == (
        "profit_factor",
        "total_return_pct",
        "best_tp_pct",
    )
    assert config.execution_profiles.default_mode == "exact_small"
    assert config.adaptive_selector_policy.mode == "shadow"
    assert config.adaptive_selector_policy.hybrid_conservative.rollout_mode == "active"
    assert config.adaptive_selector_policy.hybrid_conservative.min_grid_cardinality == 3100
    assert (
        config.adaptive_selector_policy.hybrid_conservative.minimum_exceeded_signals
        == 2
    )
    assert config.adaptive_selector_policy.hybrid_family.rollout_mode == "shadow"
    assert config.adaptive_selector_policy.hybrid_family.min_estimated_memory_bytes == 4400
    assert config.adaptive_selector_policy.hybrid_family.minimum_exceeded_signals == 4
    assert tuple(profile.mode for profile in config.execution_profiles.available_profiles) == (
        "exact_small",
        "exact_parallel",
        "hybrid_conservative",
        "hybrid_family",
    )
    assert config.execution_profiles.available_profiles[1].parallelism.stage_b_workers == 3
    assert config.execution_profiles.available_profiles[1].feature_flags.runtime_enabled is False
    assert (
        config.execution_profiles.available_profiles[1].feature_flags.parallel_stage_b_enabled
        is False
    )
    assert (
        config.execution_profiles.available_profiles[0].launch_budget.max_stage_a_variants_total
        == 120
    )
    assert config.execution_profiles.available_profiles[0].progress_weights.stage_b == 75
    assert (
        config.execution_profiles.available_profiles[1].launch_budget.max_stage_b_variants_total
        == 24000
    )
    assert config.execution_profiles.available_profiles[1].progress_weights.stage_a == 30
    assert (
        config.execution_profiles.available_profiles[2].shortlist_config.max_candidates
        == 1500
    )
    assert (
        config.execution_profiles.available_profiles[0].shortlist_config.scoring
        .activity_ratio_weight
        == 0.5
    )
    assert (
        config.execution_profiles.available_profiles[0].shortlist_config.retention
        .diversity_buckets
        == ("activity_band", "transition_band")
    )
    assert config.execution_profiles.available_profiles[2].progress_weights.stage_b == 50
    assert (
        config.execution_profiles.available_profiles[2].shortlist_config.retention
        .max_per_bucket
        == 200
    )
    assert (
        config.execution_profiles.available_profiles[3].shortlist_config.scoring
        .active_span_ratio_weight
        == 0.25
    )
    assert config.execution_profiles.available_profiles[3].family_plugin_budget_ms == 37
    assert config.execution_profiles.available_profiles[3].planning_budget_ms == 65
    assert config.guards.max_variants_per_compute == 1200
    assert config.guards.max_compute_bytes_total == 1234567
    assert config.cpu.max_numba_threads == 6
    assert config.sync.sync_deadline_seconds == 42.5
    assert config.reporting.top_trades_n_default == 5
    assert config.execution.init_cash_quote_default == 5000.0
    assert config.execution.fixed_quote_default == 250.0
    assert config.execution.safe_profit_percent_default == 15.0
    assert config.execution.slippage_pct_default == 0.05
    assert dict(config.execution.fee_pct_default_by_market_id) == {1: 0.05, 8: 0.2}
    assert config.jobs.enabled is False
    assert config.jobs.top_k_persisted_default == 42
    assert config.jobs.max_active_jobs_per_user == 8
    assert config.jobs.claim_poll_seconds == 0.5
    assert config.jobs.lease_seconds == 120
    assert config.jobs.heartbeat_seconds == 20
    assert config.jobs.snapshot_seconds == 10
    assert config.jobs.snapshot_variants_step == 200
    assert config.jobs.parallel_workers == 4



def test_load_backtest_runtime_config_rejects_invalid_jobs_defaults(tmp_path: Path) -> None:
    """
    Verify loader fails fast when jobs defaults violate deterministic schema bounds.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `backtest.jobs.top_k_persisted_default` must be strictly positive.
    Raises:
        AssertionError: If invalid jobs payload does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 0
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="top_k_persisted_default"):
        load_backtest_runtime_config(config_path)


def test_load_backtest_runtime_config_rejects_invalid_contract_top_n_bounds(
    tmp_path: Path,
) -> None:
    """
    Verify loader fails fast when frozen `top_n` bounds violate deterministic ordering.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Frozen R0 contract requires `top_n_default <= top_n_max`.
    Raises:
        AssertionError: If invalid contract payload does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  contracts:
    summary:
      top_n_default: 200
      top_n_max: 100
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="top_n_default"):
        load_backtest_runtime_config(config_path)


def test_load_backtest_runtime_config_rejects_non_exact_default_execution_profile(
    tmp_path: Path,
) -> None:
    """
    Verify loader fails fast when default execution profile drifts to hybrid mode in A1.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Milestone A1 keeps current runtime behavior exact-only even though hybrid literals are
        already published additively.
    Raises:
        AssertionError: If invalid default execution profile does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  execution_profiles:
    default: hybrid_conservative
    profiles:
      - mode: exact_small
        planning_budget_ms: 25
      - mode: exact_parallel
        planning_budget_ms: 50
      - mode: hybrid_conservative
        planning_budget_ms: 75
      - mode: hybrid_family
        planning_budget_ms: 100
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="default_mode"):
        load_backtest_runtime_config(config_path)


def test_load_backtest_runtime_config_rejects_invalid_adaptive_selector_mode(
    tmp_path: Path,
) -> None:
    """
    Verify loader fails fast when adaptive-selector rollout mode is unsupported.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Selector rollout must stay on the explicit `disabled`, `shadow`, and `active` literals.
    Raises:
        AssertionError: If invalid selector mode does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  execution_profiles:
    adaptive_selector:
      mode: automatic
    default: exact_small
    profiles:
      - mode: exact_small
        planning_budget_ms: 25
      - mode: exact_parallel
        planning_budget_ms: 50
      - mode: hybrid_conservative
        planning_budget_ms: 75
      - mode: hybrid_family
        planning_budget_ms: 100
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="Adaptive selector policy mode"):
        load_backtest_runtime_config(config_path)


def test_load_backtest_runtime_config_rejects_invalid_ranking_defaults(
    tmp_path: Path,
) -> None:
    """
    Verify loader fails fast when ranking metric defaults violate supported literals contract.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `backtest.ranking.primary_metric_default` must be one of v1 allowed literals.
    Raises:
        AssertionError: If invalid ranking payload does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  ranking:
    primary_metric_default: total_return
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="primary_metric_default"):
        load_backtest_runtime_config(config_path)


def test_load_backtest_runtime_config_rejects_invalid_shortlist_diversity_bucket(
    tmp_path: Path,
) -> None:
    """
    Verify loader fails fast when shortlist retention references an unsupported bucket axis.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Conservative shortlist retention may only use the fixed exported bucket vocabulary.
    Raises:
        AssertionError: If invalid retention config does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  execution_profiles:
    default: exact_small
    profiles:
      - mode: exact_small
        planning_budget_ms: 25
      - mode: exact_parallel
        planning_budget_ms: 50
      - mode: hybrid_conservative
        planning_budget_ms: 75
        shortlist:
          enabled: true
          max_candidates: 1000
          retention:
            diversity_buckets:
              - correlation_band
      - mode: hybrid_family
        planning_budget_ms: 100
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="diversity bucket"):
        load_backtest_runtime_config(config_path)


def test_load_backtest_runtime_config_rejects_invalid_execution_profile_progress_weights(
    tmp_path: Path,
) -> None:
    """
    Verify loader fails fast when execution-profile progress weights stop summing to `100`.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Progress/ETA semantics now live in the execution-profile contract and must remain valid
        at startup.
    Raises:
        AssertionError: If invalid progress weights do not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  execution_profiles:
    default: exact_small
    profiles:
      - mode: exact_small
        planning_budget_ms: 25
        progress:
          stage_a: 10
          stage_b: 10
          finalizing: 10
      - mode: exact_parallel
        planning_budget_ms: 50
      - mode: hybrid_conservative
        planning_budget_ms: 75
      - mode: hybrid_family
        planning_budget_ms: 100
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="must sum to 100"):
        load_backtest_runtime_config(config_path)


def test_load_backtest_runtime_config_rejects_family_plugin_budget_above_planning_budget(
    tmp_path: Path,
) -> None:
    """
    Verify loader fails fast when family-plugin budget exceeds the shared planning budget.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Family-plugin timeout must stay inside the typed execution-profile planning budget rather
        than introducing a detached timeout surface.
    Raises:
        AssertionError: If invalid family-plugin budget does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  execution_profiles:
    default: exact_small
    profiles:
      - mode: exact_small
        family_plugin_budget_ms: 30
        planning_budget_ms: 25
      - mode: exact_parallel
        family_plugin_budget_ms: 20
        planning_budget_ms: 50
      - mode: hybrid_conservative
        family_plugin_budget_ms: 30
        planning_budget_ms: 75
      - mode: hybrid_family
        family_plugin_budget_ms: 40
        planning_budget_ms: 100
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="family_plugin_budget_ms"):
        load_backtest_runtime_config(config_path)


def test_load_backtest_runtime_config_rejects_duplicate_ranking_defaults(
    tmp_path: Path,
) -> None:
    """
    Verify loader fails fast when ranking secondary metric duplicates primary metric.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Runtime ranking contract forbids duplicate primary/secondary metric identifiers.
    Raises:
        AssertionError: If duplicated ranking defaults do not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  ranking:
    primary_metric_default: total_return_pct
    secondary_metric_default: total_return_pct
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="secondary_metric_default"):
        load_backtest_runtime_config(config_path)


def test_load_backtest_runtime_config_rejects_invalid_sync_defaults(tmp_path: Path) -> None:
    """
    Verify loader fails fast when `backtest.sync.sync_deadline_seconds` is non-positive.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Sync deadline is strict-positive to keep cooperative cancellation deterministic.
    Raises:
        AssertionError: If invalid sync payload does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  sync:
    sync_deadline_seconds: 0
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="sync_deadline_seconds"):
        load_backtest_runtime_config(config_path)


def test_load_backtest_runtime_config_rejects_invalid_cpu_defaults(tmp_path: Path) -> None:
    """
    Verify loader fails fast when `backtest.cpu.max_numba_threads` is non-positive.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        CPU knob is validated at startup to keep runtime configuration fail-fast.
    Raises:
        AssertionError: If invalid CPU payload does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  cpu:
    max_numba_threads: 0
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
    )

    with pytest.raises(ValueError, match="max_numba_threads"):
        load_backtest_runtime_config(config_path)



def test_build_backtest_runtime_config_hash_is_deterministic_for_same_config() -> None:
    """
    Verify runtime hash is deterministic for identical config payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Canonical JSON hashing uses sorted keys and compact separators.
    Raises:
        AssertionError: If hash value differs between identical evaluations.
    Side Effects:
        None.
    """
    config = load_backtest_runtime_config(Path("configs/dev/backtest.yaml"))

    assert build_backtest_runtime_config_hash(config=config) == build_backtest_runtime_config_hash(
        config=config
    )



def test_build_backtest_runtime_config_hash_changes_on_result_affecting_jobs_field(
    tmp_path: Path,
) -> None:
    """
    Verify runtime hash changes when result-affecting jobs field is modified.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `backtest.jobs.top_k_persisted_default` participates in runtime hash payload.
    Raises:
        AssertionError: If hash value does not change.
    Side Effects:
        None.
    """
    config_a = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  top_k_default: 300
  warmup_bars_default: 200
  preselect_default: 20000
  sync:
    sync_deadline_seconds: 55
  reporting:
    top_trades_n_default: 3
  execution:
    init_cash_quote_default: 10000
    fixed_quote_default: 100
    safe_profit_percent_default: 30
    slippage_pct_default: 0.01
    fee_pct_default_by_market_id:
      1: 0.075
      2: 0.1
      3: 0.075
      4: 0.1
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
        filename="backtest_a.yaml",
    )
    config_b = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  top_k_default: 300
  warmup_bars_default: 200
  preselect_default: 20000
  sync:
    sync_deadline_seconds: 40
  reporting:
    top_trades_n_default: 3
  execution:
    init_cash_quote_default: 10000
    fixed_quote_default: 100
    safe_profit_percent_default: 30
    slippage_pct_default: 0.01
    fee_pct_default_by_market_id:
      1: 0.075
      2: 0.1
      3: 0.075
      4: 0.1
  jobs:
    enabled: true
    top_k_persisted_default: 250
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
        filename="backtest_b.yaml",
    )

    hash_a = build_backtest_runtime_config_hash(config=load_backtest_runtime_config(config_a))
    hash_b = build_backtest_runtime_config_hash(config=load_backtest_runtime_config(config_b))

    assert hash_a != hash_b


def test_build_backtest_runtime_config_hash_changes_on_ranking_defaults(
    tmp_path: Path,
) -> None:
    """
    Verify runtime hash changes when result-affecting ranking defaults are modified.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Ranking defaults participate in runtime hash payload.
    Raises:
        AssertionError: If hash value does not change.
    Side Effects:
        None.
    """
    config_a = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  ranking:
    primary_metric_default: total_return_pct
    secondary_metric_default: null
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
        filename="backtest_ranking_a.yaml",
    )
    config_b = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  ranking:
    primary_metric_default: return_over_max_drawdown
    secondary_metric_default: profit_factor
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
        filename="backtest_ranking_b.yaml",
    )

    hash_a = build_backtest_runtime_config_hash(config=load_backtest_runtime_config(config_a))
    hash_b = build_backtest_runtime_config_hash(config=load_backtest_runtime_config(config_b))

    assert hash_a != hash_b


def test_build_backtest_runtime_config_hash_ignores_contract_freeze_fields(
    tmp_path: Path,
) -> None:
    """
    Verify additive R0 contract-freeze fields do not affect current v1 runtime hash.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `build_backtest_runtime_config_hash` includes only current result-affecting sections.
    Raises:
        AssertionError: If hash changes when only additive freeze fields differ.
    Side Effects:
        None.
    """
    config_a = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  contracts:
    summary:
      top_n_default: 100
      top_n_max: 300
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
        filename="backtest_contract_a.yaml",
    )
    config_b = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  contracts:
    request_timeframes:
      allowed: [30m, 1h, 4h]
      forbidden: [1m, 5m]
    summary:
      top_n_default: 25
      top_n_max: 40
      ranking_metrics: [profit_factor, total_return_pct]
      sortable_columns: [profit_factor, total_return_pct]
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    parallel_workers: 1
""".strip(),
        filename="backtest_contract_b.yaml",
    )

    hash_a = build_backtest_runtime_config_hash(config=load_backtest_runtime_config(config_a))
    hash_b = build_backtest_runtime_config_hash(config=load_backtest_runtime_config(config_b))

    assert hash_a == hash_b


def test_build_backtest_runtime_config_hash_ignores_operational_jobs_fields(
    tmp_path: Path,
) -> None:
    """
    Verify runtime hash ignores operational-only jobs fields.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Operational jobs knobs are excluded from result-affecting hash payload.
    Raises:
        AssertionError: If hash value changes for operational-only modifications.
    Side Effects:
        None.
    """
    config_path_a = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  sync:
    sync_deadline_seconds: 30
  jobs:
    enabled: true
    top_k_persisted_default: 300
    max_active_jobs_per_user: 3
    claim_poll_seconds: 1
    lease_seconds: 60
    heartbeat_seconds: 15
    snapshot_seconds: 30
    snapshot_variants_step: 1000
    parallel_workers: 1
""".strip(),
        filename="backtest_operational_a.yaml",
    )
    hash_a = build_backtest_runtime_config_hash(
        config=load_backtest_runtime_config(config_path_a)
    )

    config_path_b = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  sync:
    sync_deadline_seconds: 55
  jobs:
    enabled: false
    top_k_persisted_default: 300
    max_active_jobs_per_user: 99
    claim_poll_seconds: 0.25
    lease_seconds: 300
    heartbeat_seconds: 30
    snapshot_seconds: 5
    snapshot_variants_step: 50
    parallel_workers: 8
""".strip(),
        filename="backtest_operational_b.yaml",
    )
    hash_b = build_backtest_runtime_config_hash(
        config=load_backtest_runtime_config(config_path_b)
    )

    assert hash_a == hash_b
