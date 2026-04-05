from __future__ import annotations

from trading.contexts.backtest.application.services.v2.family_plugins import (
    FamilyPluginCircuitBreakerV2,
)


def test_family_plugin_circuit_breaker_opens_after_repeated_timeout_failures() -> None:
    """
    Verify repeated timeout failures open the breaker and surface explicit fallback warnings.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Timeout handling is budget-aware and repeated failures should block the plugin for the
        rest of the run.
    Raises:
        AssertionError: If timeout failures do not open the breaker or omit warning semantics.
    Side Effects:
        None.
    """
    breaker = FamilyPluginCircuitBreakerV2(failure_threshold=2)

    first_warning = breaker.record_timeout(plugin_id="ma_accel", budget_ms=40)
    second_warning = breaker.record_timeout(plugin_id="ma_accel", budget_ms=40)
    open_warning = breaker.warning_for_open_breaker(plugin_id="ma_accel")

    assert first_warning.reason == "timeout"
    assert first_warning.plugin_id == "ma_accel"
    assert breaker.state_for(plugin_id="ma_accel").consecutive_failures == 2
    assert breaker.is_open(plugin_id="ma_accel") is True
    assert second_warning.reason == "timeout"
    assert "circuit breaker opened" in second_warning.message
    assert open_warning.reason == "open_breaker"
    assert open_warning.fallback_action == "warning + universal fallback"


def test_family_plugin_circuit_breaker_success_resets_failures_before_open() -> None:
    """
    Verify success clears repeated failures while the breaker is still closed.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The breaker tracks repeated failures, so an intervening success should reset the counter
        until the breaker opens.
    Raises:
        AssertionError: If success does not reset the repeated-failure counter.
    Side Effects:
        None.
    """
    breaker = FamilyPluginCircuitBreakerV2(failure_threshold=2)

    breaker.record_error(plugin_id="ma_accel", error=RuntimeError("boom"))
    state_after_failure = breaker.state_for(plugin_id="ma_accel")
    state_after_success = breaker.record_success(plugin_id="ma_accel")
    breaker.record_error(plugin_id="ma_accel", error=RuntimeError("boom again"))
    state_after_reset_failure = breaker.state_for(plugin_id="ma_accel")

    assert state_after_failure.consecutive_failures == 1
    assert state_after_success.consecutive_failures == 0
    assert state_after_success.open_for_run is False
    assert state_after_reset_failure.consecutive_failures == 1
    assert state_after_reset_failure.open_for_run is False


def test_family_plugin_circuit_breaker_stays_open_for_run_after_threshold() -> None:
    """
    Verify an open breaker remains open even if later calls report success within the same run.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Milestone E breaker state lasts for the rest of the run once repeated failures trip it.
    Raises:
        AssertionError: If a later success closes the breaker during the same run.
    Side Effects:
        None.
    """
    breaker = FamilyPluginCircuitBreakerV2(failure_threshold=1)

    breaker.record_error(plugin_id="ma_accel", error=RuntimeError("boom"))
    state_after_open = breaker.state_for(plugin_id="ma_accel")
    state_after_success = breaker.record_success(plugin_id="ma_accel")

    assert state_after_open.open_for_run is True
    assert state_after_success.open_for_run is True
    assert state_after_success.consecutive_failures == 1
