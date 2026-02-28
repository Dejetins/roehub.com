from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.backtest.adapters.outbound.config import (
    build_backtest_runtime_config_hash,
    load_backtest_runtime_config,
    resolve_backtest_config_path,
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
    primary_metric_default: RETURN_OVER_MAX_DRAWDOWN
    secondary_metric_default: profit_factor
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
    assert config.ranking.primary_metric_default == "return_over_max_drawdown"
    assert config.ranking.secondary_metric_default == "profit_factor"
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
