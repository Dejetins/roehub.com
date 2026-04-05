"""Per-run circuit breaker and warning helpers for proposal-only family accelerators."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts_v2 import FamilyPluginWarningV2, normalize_family_plugin_identifier_v2


@dataclass(frozen=True, slots=True)
class FamilyPluginCircuitBreakerStateV2:
    """
    Immutable snapshot of one plugin's circuit-breaker state within a single run.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/circuit_breaker_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - tests/unit/contexts/backtest/application/services/v2/
        test_family_plugin_circuit_breaker_v2.py
    """

    plugin_id: str
    consecutive_failures: int
    failure_threshold: int
    open_for_run: bool

    def __post_init__(self) -> None:
        """
        Validate one immutable per-run circuit-breaker state snapshot.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            circuit_breaker_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/
            test_family_plugin_circuit_breaker_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Circuit-breaker state is local to one run and is never persisted as a cross-run cache.
        Raises:
            ValueError: If one counter or threshold is invalid.
        Side Effects:
            Normalizes `plugin_id` to canonical lower-case form.
        """
        object.__setattr__(
            self,
            "plugin_id",
            normalize_family_plugin_identifier_v2(
                value=self.plugin_id,
                field_name="FamilyPluginCircuitBreakerStateV2.plugin_id",
            ),
        )
        if self.consecutive_failures < 0:
            raise ValueError(
                "FamilyPluginCircuitBreakerStateV2.consecutive_failures must be >= 0"
            )
        if self.failure_threshold <= 0:
            raise ValueError(
                "FamilyPluginCircuitBreakerStateV2.failure_threshold must be > 0"
            )


class FamilyPluginCircuitBreakerV2:
    """
    Mutable per-run circuit breaker tracking repeated plugin failures for the current run only.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/circuit_breaker_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - tests/unit/contexts/backtest/application/services/v2/
        test_family_plugin_circuit_breaker_v2.py
    """

    def __init__(self, *, failure_threshold: int = 2) -> None:
        """
        Initialize one empty per-run circuit breaker with deterministic failure threshold.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            circuit_breaker_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/
            test_family_plugin_circuit_breaker_v2.py

        Args:
            failure_threshold: Number of repeated failures required before the breaker opens.
        Returns:
            None.
        Assumptions:
            Each run owns its own breaker instance; an open breaker remains open for the rest of
            that run.
        Raises:
            ValueError: If the threshold is non-positive.
        Side Effects:
            Initializes empty in-memory counters for the current run.
        """
        if failure_threshold <= 0:
            raise ValueError("FamilyPluginCircuitBreakerV2.failure_threshold must be > 0")
        self._failure_threshold = failure_threshold
        self._consecutive_failures_by_plugin_id: dict[str, int] = {}
        self._open_plugin_ids: set[str] = set()

    def state_for(self, *, plugin_id: str) -> FamilyPluginCircuitBreakerStateV2:
        """
        Return the current immutable state snapshot for one plugin id in this run.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            circuit_breaker_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/
            test_family_plugin_circuit_breaker_v2.py

        Args:
            plugin_id: Stable internal plugin identifier.
        Returns:
            FamilyPluginCircuitBreakerStateV2: Current failure-count/open state snapshot.
        Assumptions:
            State reads are pure and do not mutate breaker counters.
        Raises:
            ValueError: If `plugin_id` is blank.
        Side Effects:
            None.
        """
        normalized_plugin_id = normalize_family_plugin_identifier_v2(
            value=plugin_id,
            field_name="FamilyPluginCircuitBreakerV2.plugin_id",
        )
        return FamilyPluginCircuitBreakerStateV2(
            plugin_id=normalized_plugin_id,
            consecutive_failures=self._consecutive_failures_by_plugin_id.get(
                normalized_plugin_id,
                0,
            ),
            failure_threshold=self._failure_threshold,
            open_for_run=normalized_plugin_id in self._open_plugin_ids,
        )

    def is_open(self, *, plugin_id: str) -> bool:
        """
        Return whether a plugin already sits behind an open circuit breaker for this run.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            circuit_breaker_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/
            test_family_plugin_circuit_breaker_v2.py

        Args:
            plugin_id: Stable internal plugin identifier.
        Returns:
            bool: `True` when the breaker is open for the rest of this run.
        Assumptions:
            Open-breaker checks happen before proposal invocation in future runtime wiring.
        Raises:
            ValueError: If `plugin_id` is blank.
        Side Effects:
            None.
        """
        return self.state_for(plugin_id=plugin_id).open_for_run

    def record_success(self, *, plugin_id: str) -> FamilyPluginCircuitBreakerStateV2:
        """
        Record one successful plugin attempt and reset repeated-failure count when allowed.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            circuit_breaker_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/
            test_family_plugin_circuit_breaker_v2.py

        Args:
            plugin_id: Stable internal plugin identifier.
        Returns:
            FamilyPluginCircuitBreakerStateV2: Updated breaker state after the success.
        Assumptions:
            Success resets repeated failures only while the breaker is still closed; once opened,
            it remains open for the rest of the run.
        Raises:
            ValueError: If `plugin_id` is blank.
        Side Effects:
            Mutates in-memory per-run counters.
        """
        normalized_plugin_id = normalize_family_plugin_identifier_v2(
            value=plugin_id,
            field_name="FamilyPluginCircuitBreakerV2.plugin_id",
        )
        if normalized_plugin_id not in self._open_plugin_ids:
            self._consecutive_failures_by_plugin_id.pop(normalized_plugin_id, None)
        return self.state_for(plugin_id=normalized_plugin_id)

    def record_timeout(self, *, plugin_id: str, budget_ms: int) -> FamilyPluginWarningV2:
        """
        Record one timeout failure and return the explicit timeout warning payload.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            circuit_breaker_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/
            test_family_plugin_circuit_breaker_v2.py

        Args:
            plugin_id: Stable internal plugin identifier.
            budget_ms: Typed family-plugin budget from the resolved execution profile.
        Returns:
            FamilyPluginWarningV2: Timeout warning carrying explicit fallback semantics.
        Assumptions:
            Plugin timeout is budget-aware and must stay tied to the execution-profile planning
            contract instead of relying on a detached hardcoded timeout.
        Raises:
            ValueError: If `plugin_id` is blank or `budget_ms` is non-positive.
        Side Effects:
            Mutates in-memory failure counters and may open the breaker for the rest of the run.
        """
        if budget_ms <= 0:
            raise ValueError("FamilyPluginCircuitBreakerV2.record_timeout budget_ms must be > 0")
        state = self._record_failure(plugin_id=plugin_id)
        return FamilyPluginWarningV2(
            reason="timeout",
            plugin_id=state.plugin_id,
            message=(
                f"Family plugin {state.plugin_id!r} exceeded its {budget_ms}ms budget; "
                f"{_failure_suffix_v2(state=state)}"
            ),
        )

    def record_error(self, *, plugin_id: str, error: Exception) -> FamilyPluginWarningV2:
        """
        Record one plugin exception and return the explicit error warning payload.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            circuit_breaker_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/
            test_family_plugin_circuit_breaker_v2.py

        Args:
            plugin_id: Stable internal plugin identifier.
            error: Raised plugin exception.
        Returns:
            FamilyPluginWarningV2: Error warning carrying explicit fallback semantics.
        Assumptions:
            Runtime callers surface errors as warnings and fall back to the universal path rather
            than swallowing or silently retrying non-idempotent plugin work.
        Raises:
            ValueError: If `plugin_id` is blank.
        Side Effects:
            Mutates in-memory failure counters and may open the breaker for the rest of the run.
        """
        state = self._record_failure(plugin_id=plugin_id)
        return FamilyPluginWarningV2(
            reason="error",
            plugin_id=state.plugin_id,
            message=(
                f"Family plugin {state.plugin_id!r} raised "
                f"{error.__class__.__name__}: {str(error).strip() or 'unknown error'}; "
                f"{_failure_suffix_v2(state=state)}"
            ),
        )

    def warning_for_open_breaker(self, *, plugin_id: str) -> FamilyPluginWarningV2:
        """
        Return the explicit open-breaker warning used when a plugin is skipped for this run.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            circuit_breaker_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/
            test_family_plugin_circuit_breaker_v2.py

        Args:
            plugin_id: Stable internal plugin identifier.
        Returns:
            FamilyPluginWarningV2: Open-breaker warning carrying explicit fallback semantics.
        Assumptions:
            Future runtime wiring checks the breaker before invoking the plugin and degrades
            directly to the universal path when the breaker is already open.
        Raises:
            ValueError: If `plugin_id` is blank.
        Side Effects:
            None.
        """
        state = self.state_for(plugin_id=plugin_id)
        return FamilyPluginWarningV2(
            reason="open_breaker",
            plugin_id=state.plugin_id,
            message=(
                f"Family plugin {state.plugin_id!r} sits behind an open circuit breaker for "
                "the rest of this run; warning + universal fallback applies."
            ),
        )

    def _record_failure(self, *, plugin_id: str) -> FamilyPluginCircuitBreakerStateV2:
        """
        Increment repeated-failure counters and open the breaker when the threshold is reached.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            circuit_breaker_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/
            test_family_plugin_circuit_breaker_v2.py

        Args:
            plugin_id: Stable internal plugin identifier.
        Returns:
            FamilyPluginCircuitBreakerStateV2: Updated breaker state after recording the failure.
        Assumptions:
            Repeated failures are counted per plugin within one run; once open, the breaker stays
            open until the run ends.
        Raises:
            ValueError: If `plugin_id` is blank.
        Side Effects:
            Mutates in-memory failure counters and open-breaker membership for this run.
        """
        normalized_plugin_id = normalize_family_plugin_identifier_v2(
            value=plugin_id,
            field_name="FamilyPluginCircuitBreakerV2.plugin_id",
        )
        if normalized_plugin_id in self._open_plugin_ids:
            return self.state_for(plugin_id=normalized_plugin_id)
        consecutive_failures = (
            self._consecutive_failures_by_plugin_id.get(normalized_plugin_id, 0) + 1
        )
        self._consecutive_failures_by_plugin_id[normalized_plugin_id] = consecutive_failures
        if consecutive_failures >= self._failure_threshold:
            self._open_plugin_ids.add(normalized_plugin_id)
        return self.state_for(plugin_id=normalized_plugin_id)


def _failure_suffix_v2(*, state: FamilyPluginCircuitBreakerStateV2) -> str:
    """
    Build the explicit warning suffix for timeout/error fallback messages.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/circuit_breaker_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - tests/unit/contexts/backtest/application/services/v2/
        test_family_plugin_circuit_breaker_v2.py

    Args:
        state: Updated per-run breaker state after recording the latest failure.
    Returns:
        str: Human-readable suffix describing fallback and optional breaker opening.
    Assumptions:
        Warning text stays concise but explicit about whether the latest failure opened the
        circuit breaker for the rest of the run.
    Raises:
        None.
    Side Effects:
        None.
    """
    if state.open_for_run:
        return (
            "circuit breaker opened for the rest of this run; warning + universal fallback "
            "applies."
        )
    return "warning + universal fallback applies."


__all__ = [
    "FamilyPluginCircuitBreakerStateV2",
    "FamilyPluginCircuitBreakerV2",
]
