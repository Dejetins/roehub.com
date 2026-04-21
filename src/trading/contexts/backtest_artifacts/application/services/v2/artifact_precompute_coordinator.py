"""R12 stage-oriented coordinator and timeframe-session lifecycle for artifact precompute."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Mapping, TypeVar

from .contracts import (
    ARTIFACT_PRECOMPUTE_STAGE_ORDER_V2,
    ArtifactCoordinatesV2,
    ArtifactPrecomputeExecutionPolicyV2,
    ArtifactPrecomputeProgressEventV2,
    ArtifactPrecomputeStageIdV2,
    ArtifactPrecomputeStageInputV2,
    ArtifactPrecomputeStageOutputV2,
    ArtifactPrecomputeStageResultV2,
    ArtifactSlotLiteralV2,
    ArtifactTailRebuildBarsV2,
    freeze_artifact_payload_mapping_v2,
    validate_mapping_timeframe_v2,
)

log = logging.getLogger(__name__)

_COORDINATOR_COMPONENT_LITERAL_V2 = "backtest-artifact-precompute-runner"
_StageResultT = TypeVar("_StageResultT")


@dataclass(slots=True)
class ArtifactTimeframeSessionV2:
    """
    Explicit lifecycle owner for one open target timeframe inside R12 precompute.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_coordinator.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    coordinator: "ArtifactPrecomputeCoordinatorV2"
    timeframe: str
    opened_details: Mapping[str, Any] = field(default_factory=dict)
    _finish_details: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False)
    _started_at: float | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate one explicit timeframe-session lease before it is opened.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Session identity is one target request timeframe and one immutable start payload.
        Raises:
            ValueError: If timeframe or details payload violates the strict session contract.
        Side Effects:
            Freezes `opened_details` into a stable key-sorted read-only mapping.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_coordinator.py
        """
        self.timeframe = validate_mapping_timeframe_v2(self.timeframe)
        self.opened_details = freeze_artifact_payload_mapping_v2(self.opened_details)

    def __enter__(self) -> "ArtifactTimeframeSessionV2":
        """
        Open the timeframe session and emit the typed `timeframe_started` progress event.

        Args:
            None.
        Returns:
            ArtifactTimeframeSessionV2: The opened session handle.
        Assumptions:
            Only the coordinator owns concurrent-session accounting.
        Raises:
            ValueError: If opening this session would violate `max_open_timeframe_sessions`.
        Side Effects:
            Updates coordinator session state and writes one structured log record.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        self._started_at = time.perf_counter()
        self.coordinator._enter_timeframe_session(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: object | None,
    ) -> bool:
        """
        Close the timeframe session and emit the typed `timeframe_finished` progress event.

        Args:
            exc_type: Exception type from the context body, when present.
            exc: Exception instance from the context body, when present.
            exc_tb: Traceback object from the context body, when present.
        Returns:
            bool: Always `False` to propagate exceptions from the context body.
        Assumptions:
            Session release must happen even when the stage body raises.
        Raises:
            None.
        Side Effects:
            Updates coordinator session state and writes one structured log record.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        del exc_type, exc, exc_tb
        self.close()
        return False

    def set_finish_details(self, *, details: Mapping[str, Any]) -> None:
        """
        Store the deterministic session-finish payload emitted on `timeframe_finished`.

        Args:
            details: JSON-friendly session summary payload.
        Returns:
            None.
        Assumptions:
            The session owner computes the final summary after the stage body completes.
        Raises:
            ValueError: If the details payload contains non-string keys.
        Side Effects:
            Replaces the in-memory finish payload for the current session.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        self._finish_details = freeze_artifact_payload_mapping_v2(details)

    def close(self) -> None:
        """
        Close the timeframe session exactly once and emit the finish event.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            The session was opened through the matching coordinator.
        Raises:
            ValueError: If the session was not opened successfully.
        Side Effects:
            Releases one open-session slot and writes one structured log record.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        if self._closed:
            return
        started_at = self._started_at
        if started_at is None:
            raise ValueError("ArtifactTimeframeSessionV2.close requires an opened session")
        self.coordinator._exit_timeframe_session(
            session=self,
            elapsed_seconds=time.perf_counter() - started_at,
        )
        self._closed = True


@dataclass(slots=True)
class ArtifactPrecomputeCoordinatorV2(Generic[_StageResultT]):
    """
    Explicit owner of R12 stage order, structured progress events, and stage summaries.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """

    coordinates: ArtifactCoordinatesV2
    slot: ArtifactSlotLiteralV2
    slot_generation: int
    force_full_rebuild: bool
    execution_policy: ArtifactPrecomputeExecutionPolicyV2
    _stage_results: list[ArtifactPrecomputeStageResultV2] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _open_timeframes: list[str] = field(default_factory=list, init=False, repr=False)
    _last_stage_index: int = field(default=-1, init=False, repr=False)

    def open_timeframe_session(
        self,
        *,
        timeframe: str,
        details: Mapping[str, Any] | None = None,
    ) -> ArtifactTimeframeSessionV2:
        """
        Create one explicit timeframe session governed by `max_open_timeframe_sessions`.

        Args:
            timeframe: Target request timeframe opened by the session.
            details: Optional JSON-friendly payload for the `timeframe_started` event.
        Returns:
            ArtifactTimeframeSessionV2: Explicit session handle that must be closed.
        Assumptions:
            The current R12 runner opens one target timeframe at a time, while the contract
            keeps the upper bound explicit for later chunk-worker work.
        Raises:
            ValueError: If timeframe or details payload is invalid.
        Side Effects:
            None until the returned session is entered.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        return ArtifactTimeframeSessionV2(
            coordinator=self,
            timeframe=timeframe,
            opened_details={} if details is None else details,
        )

    def run_stage(
        self,
        *,
        stage_input: ArtifactPrecomputeStageInputV2,
        execute: Callable[[], _StageResultT],
        build_output: Callable[[_StageResultT], ArtifactPrecomputeStageOutputV2],
    ) -> _StageResultT:
        """
        Execute one stage, emit structured lifecycle events, and record the stage summary.

        Args:
            stage_input: Structured immutable stage-start payload.
            execute: Stage body callback.
            build_output: Builder translating the raw callback result into a typed stage summary.
        Returns:
            _StageResultT: Raw stage body result produced by `execute`.
        Assumptions:
            The coordinator owns stage-order validation and timing; stage bodies own domain work.
        Raises:
            ValueError: If stage order regresses or the output builder returns a drifted stage id.
            Exception: Propagates any stage-body exception without swallowing it.
        Side Effects:
            Writes start/finish structured logs and appends one typed stage result.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        self._advance_stage_order(stage_input.stage)
        started_at = time.perf_counter()
        self._emit_progress_event(
            ArtifactPrecomputeProgressEventV2(
                event="artifact_precompute_stage_started",
                stage=stage_input.stage,
                current_timeframe=stage_input.current_timeframe,
                details=stage_input.details,
            )
        )
        result = execute()
        stage_output = build_output(result)
        stage_result = ArtifactPrecomputeStageResultV2(
            stage_input=stage_input,
            stage_output=stage_output,
            elapsed_seconds=time.perf_counter() - started_at,
        )
        self._stage_results.append(stage_result)
        self._emit_progress_event(
            ArtifactPrecomputeProgressEventV2(
                event="artifact_precompute_stage_finished",
                stage=stage_output.stage,
                current_timeframe=stage_output.current_timeframe,
                elapsed_seconds=stage_result.elapsed_seconds,
                details=stage_output.as_dict(),
            )
        )
        return result

    def stage_results(self) -> tuple[ArtifactPrecomputeStageResultV2, ...]:
        """
        Return the deterministic ordered stage-result summaries accumulated so far.

        Args:
            None.
        Returns:
            tuple[ArtifactPrecomputeStageResultV2, ...]: Immutable stage results in execution
                order.
        Assumptions:
            Callers aggregate these summaries into logs and later metrics without reordering.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        return tuple(self._stage_results)

    def emit_completed(
        self,
        *,
        elapsed_seconds: float,
        tail_rebuild_bars: ArtifactTailRebuildBarsV2,
    ) -> None:
        """
        Emit the final structured completion log with ordered `stage_results`.

        Args:
            elapsed_seconds: Full precompute wall-clock duration.
            tail_rebuild_bars: Existing rewritten-tail summary exposed by public diagnostics.
        Returns:
            None.
        Assumptions:
            The final completion event stays additive and preserves the existing event name.
        Raises:
            None.
        Side Effects:
            Writes one INFO log record.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        log.info(
            "event=artifact_precompute_finished component=%s stage=%s exchange=%s "
            "market_type=%s symbol=%s slot=%s slot_generation=%s force_full_rebuild=%s "
            "elapsed_seconds=%.3f details=%s",
            _COORDINATOR_COMPONENT_LITERAL_V2,
            "root_manifest",
            self.coordinates.exchange,
            self.coordinates.market_type,
            self.coordinates.symbol,
            self.slot,
            self.slot_generation,
            self.force_full_rebuild,
            elapsed_seconds,
            json.dumps(
                {
                    "stage_results": [result.as_dict() for result in self.stage_results()],
                    "tail_rebuild_bars": tail_rebuild_bars.as_dict(),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )

    def _enter_timeframe_session(self, session: ArtifactTimeframeSessionV2) -> None:
        """
        Register one opened timeframe session and emit `timeframe_started`.

        Args:
            session: Session handle being entered by the caller.
        Returns:
            None.
        Assumptions:
            The coordinator is the single owner of open-session accounting.
        Raises:
            ValueError: If the session would exceed `max_open_timeframe_sessions` or duplicate
                the same timeframe lease.
        Side Effects:
            Updates open-session state and writes one structured log record.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        if session.timeframe in self._open_timeframes:
            raise ValueError(
                f"timeframe session {session.timeframe!r} is already open for this coordinator"
            )
        if len(self._open_timeframes) >= self.execution_policy.max_open_timeframe_sessions:
            raise ValueError(
                "max_open_timeframe_sessions exceeded: "
                f"open_sessions={tuple(self._open_timeframes)!r}, "
                f"requested_timeframe={session.timeframe!r}, "
                f"max_open_timeframe_sessions={self.execution_policy.max_open_timeframe_sessions!r}"
            )
        self._open_timeframes.append(session.timeframe)
        self._emit_progress_event(
            ArtifactPrecomputeProgressEventV2(
                event="timeframe_started",
                stage="timeframe_session",
                current_timeframe=session.timeframe,
                details={
                    "current_timeframe": session.timeframe,
                    "open_timeframe_sessions": len(self._open_timeframes),
                    "max_open_timeframe_sessions": (
                        self.execution_policy.max_open_timeframe_sessions
                    ),
                    **dict(session.opened_details),
                },
            )
        )

    def _exit_timeframe_session(
        self,
        *,
        session: ArtifactTimeframeSessionV2,
        elapsed_seconds: float,
    ) -> None:
        """
        Release one opened timeframe session and emit `timeframe_finished`.

        Args:
            session: Session handle being closed by the caller.
            elapsed_seconds: Measured session wall-clock duration.
        Returns:
            None.
        Assumptions:
            Session close order is deterministic and release happens exactly once.
        Raises:
            ValueError: If the timeframe is not currently tracked as open.
        Side Effects:
            Updates open-session state and writes one structured log record.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        if session.timeframe not in self._open_timeframes:
            raise ValueError(
                f"timeframe session {session.timeframe!r} is not currently open"
            )
        self._open_timeframes.remove(session.timeframe)
        self._emit_progress_event(
            ArtifactPrecomputeProgressEventV2(
                event="timeframe_finished",
                stage="timeframe_session",
                current_timeframe=session.timeframe,
                elapsed_seconds=elapsed_seconds,
                details={
                    "current_timeframe": session.timeframe,
                    "open_timeframe_sessions": len(self._open_timeframes),
                    "max_open_timeframe_sessions": (
                        self.execution_policy.max_open_timeframe_sessions
                    ),
                    **dict(session._finish_details),
                },
            )
        )

    def _advance_stage_order(self, stage: ArtifactPrecomputeStageIdV2) -> None:
        """
        Enforce monotonic high-level stage order across repeated stage invocations.

        Args:
            stage: Next stage identifier requested by the caller.
        Returns:
            None.
        Assumptions:
            Only `timeframe_session` may repeat consecutively for multiple target timeframes.
        Raises:
            ValueError: If the requested stage regresses or duplicates a non-repeatable stage.
        Side Effects:
            Updates the in-memory last-stage cursor.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        stage_index = ARTIFACT_PRECOMPUTE_STAGE_ORDER_V2.index(stage)
        if stage_index < self._last_stage_index:
            raise ValueError(
                "artifact precompute stage order regression detected: "
                f"last_stage={ARTIFACT_PRECOMPUTE_STAGE_ORDER_V2[self._last_stage_index]!r}, "
                f"requested_stage={stage!r}"
            )
        if stage_index == self._last_stage_index and stage != "timeframe_session":
            raise ValueError(f"artifact precompute stage {stage!r} may not execute twice")
        self._last_stage_index = stage_index

    def _emit_progress_event(self, event: ArtifactPrecomputeProgressEventV2) -> None:
        """
        Write one structured progress log record for the coordinator lifecycle.

        Args:
            event: Typed structured progress event payload.
        Returns:
            None.
        Assumptions:
            Operators grep these events directly from manual CLI logs and service logs.
        Raises:
            None.
        Side Effects:
            Writes one INFO log record.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        details_payload = json.dumps(
            dict(event.details),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        if event.elapsed_seconds is None:
            log.info(
                "event=%s component=%s stage=%s exchange=%s market_type=%s symbol=%s "
                "slot=%s slot_generation=%s force_full_rebuild=%s current_timeframe=%s "
                "details=%s",
                event.event,
                _COORDINATOR_COMPONENT_LITERAL_V2,
                event.stage,
                self.coordinates.exchange,
                self.coordinates.market_type,
                self.coordinates.symbol,
                self.slot,
                self.slot_generation,
                self.force_full_rebuild,
                event.current_timeframe,
                details_payload,
            )
            return
        log.info(
            "event=%s component=%s stage=%s exchange=%s market_type=%s symbol=%s "
            "slot=%s slot_generation=%s force_full_rebuild=%s current_timeframe=%s "
            "elapsed_seconds=%.3f details=%s",
            event.event,
            _COORDINATOR_COMPONENT_LITERAL_V2,
            event.stage,
            self.coordinates.exchange,
            self.coordinates.market_type,
            self.coordinates.symbol,
            self.slot,
            self.slot_generation,
            self.force_full_rebuild,
            event.current_timeframe,
            event.elapsed_seconds,
            details_payload,
        )
