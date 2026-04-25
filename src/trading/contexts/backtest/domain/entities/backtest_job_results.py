from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Sequence, cast
from uuid import UUID

from trading.contexts.backtest.domain.entities.backtest_job import BacktestJobState
from trading.contexts.backtest.domain.errors import BacktestJobTransitionError


@dataclass(frozen=True, slots=True)
class BacktestJobTopVariant:
    """
    Persisted ranked top-variant snapshot row for one Backtest job.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
      - alembic/versions/20260222_0003_backtest_jobs_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    job_id: UUID
    rank: int
    variant_key: str
    indicator_variant_key: str
    variant_index: int
    total_return_pct: float
    payload_json: Mapping[str, Any]
    updated_at: datetime
    summary_metrics_json: Mapping[str, Any] = field(default_factory=dict)
    best_tp_pct: float | None = None
    best_sl_pct: float | None = None
    report_table_md: str | None = None
    trades_json: tuple[Mapping[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        """
        Validate top-variant row shape and normalize JSON payloads.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
          - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
            backtest_job_results_repository.py
          - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
        Args:
            None.
        Returns:
            None.
        Assumptions:
            Variant keys are canonical lowercase SHA-256 hex literals.
        Raises:
            BacktestJobTransitionError: If one field violates storage contract.
        Side Effects:
            Replaces payload and trades items with immutable normalized structures.
        """
        if self.rank <= 0:
            raise BacktestJobTransitionError("BacktestJobTopVariant.rank must be > 0")
        if self.variant_index < 0:
            raise BacktestJobTransitionError("BacktestJobTopVariant.variant_index must be >= 0")

        normalized_variant_key = self.variant_key.strip().lower()
        normalized_indicator_key = self.indicator_variant_key.strip().lower()
        _ensure_sha256_key(name="variant_key", value=normalized_variant_key)
        _ensure_sha256_key(name="indicator_variant_key", value=normalized_indicator_key)

        if isinstance(self.total_return_pct, bool) or not isinstance(
            self.total_return_pct,
            int | float,
        ):
            raise BacktestJobTransitionError(
                "BacktestJobTopVariant.total_return_pct must be numeric"
            )

        payload = _normalize_json_object(value=self.payload_json)
        summary_metrics = _normalize_json_object(value=self.summary_metrics_json)
        summary_metrics["total_return_pct"] = float(self.total_return_pct)

        _ensure_utc_datetime(name="updated_at", value=self.updated_at)

        if self.report_table_md is not None:
            raise BacktestJobTransitionError(
                "BacktestJobTopVariant.report_table_md must stay null in summary-only contract"
            )
        if self.trades_json is not None:
            raise BacktestJobTransitionError(
                "BacktestJobTopVariant.trades_json must stay null in summary-only contract"
            )

        object.__setattr__(self, "variant_key", normalized_variant_key)
        object.__setattr__(self, "indicator_variant_key", normalized_indicator_key)
        object.__setattr__(self, "total_return_pct", float(self.total_return_pct))
        object.__setattr__(self, "payload_json", MappingProxyType(payload))
        object.__setattr__(self, "summary_metrics_json", MappingProxyType(summary_metrics))
        object.__setattr__(
            self,
            "best_tp_pct",
            _normalize_optional_non_negative_float(name="best_tp_pct", value=self.best_tp_pct),
        )
        object.__setattr__(
            self,
            "best_sl_pct",
            _normalize_optional_non_negative_float(name="best_sl_pct", value=self.best_sl_pct),
        )
        object.__setattr__(self, "trades_json", None)
        object.__setattr__(self, "report_table_md", None)


@dataclass(frozen=True, slots=True)
class BacktestJobParityRetainedRowsCounter:
    """
    Persisted retained-row counter for one indicator inside the compact parity runtime state.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
    """

    indicator_id: str
    retained_rows: int

    def __post_init__(self) -> None:
        """
        Validate one persisted parity retained-row counter.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Parity retained-row counters remain compact additive metadata only and must stay
            deterministic for mixed-version reads.
        Raises:
            BacktestJobTransitionError: If indicator id is blank or retained rows are non-positive.
        Side Effects:
            Normalizes `indicator_id` to stripped builtin `str`.
        """
        normalized_indicator_id = self.indicator_id.strip()
        if not normalized_indicator_id:
            raise BacktestJobTransitionError(
                "BacktestJobParityRetainedRowsCounter.indicator_id must be non-empty"
            )
        if isinstance(self.retained_rows, bool) or not isinstance(self.retained_rows, int):
            raise BacktestJobTransitionError(
                "BacktestJobParityRetainedRowsCounter.retained_rows must be integer"
            )
        if self.retained_rows <= 0:
            raise BacktestJobTransitionError(
                "BacktestJobParityRetainedRowsCounter.retained_rows must be > 0"
            )
        object.__setattr__(self, "indicator_id", normalized_indicator_id)

    def to_json_object(self) -> dict[str, Any]:
        """
        Convert one retained-row counter into deterministic JSON payload.

        Args:
            None.
        Returns:
            dict[str, Any]: JSON-safe counter payload.
        Assumptions:
            Indicator order is preserved by the enclosing parity runtime state tuple.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "indicator_id": self.indicator_id,
            "retained_rows": self.retained_rows,
        }

    @classmethod
    def from_json_object(
        cls,
        *,
        value: Mapping[str, Any],
    ) -> "BacktestJobParityRetainedRowsCounter":
        """
        Build one retained-row counter from persisted JSON object payload.

        Args:
            value: Raw persisted JSON object.
        Returns:
            BacktestJobParityRetainedRowsCounter: Normalized immutable counter payload.
        Assumptions:
            Repository mapping validates the top-level object boundary before calling this helper.
        Raises:
            BacktestJobTransitionError: If one required field is missing or malformed.
        Side Effects:
            None.
        """
        return cls(
            indicator_id=_required_non_empty_str_from_json(
                name="BacktestJobParityRetainedRowsCounter.indicator_id",
                value=value.get("indicator_id"),
            ),
            retained_rows=_required_positive_int_from_json(
                name="BacktestJobParityRetainedRowsCounter.retained_rows",
                value=value.get("retained_rows"),
            ),
        )


@dataclass(frozen=True, slots=True)
class BacktestJobParityClassification:
    """
    Persisted parity-first classification evidence for the canonical no-risk exact runtime class.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
    """

    parity_class: str = "parity_first_no_risk_exact"
    disabled_risk_single_cell: bool = True
    low_indicator_block_cardinality: bool = True
    narrowed_retained_row_evidence: bool = True
    notebook_shaped_cost_units: bool = True
    nr2_classification_reason: str = ""

    def __post_init__(self) -> None:
        """
        Validate persisted parity-classification evidence.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            D5 persists only the compact classification required to prove the parity-first
            no-risk exact class and keep worker resume deterministic.
        Raises:
            BacktestJobTransitionError: If one field violates the persisted parity contract.
        Side Effects:
            Normalizes string fields to stripped builtin `str`.
        """
        normalized_parity_class = self.parity_class.strip().lower()
        if normalized_parity_class != "parity_first_no_risk_exact":
            raise BacktestJobTransitionError(
                "BacktestJobParityClassification.parity_class must be "
                "'parity_first_no_risk_exact'"
            )
        for field_name in (
            "disabled_risk_single_cell",
            "low_indicator_block_cardinality",
            "narrowed_retained_row_evidence",
            "notebook_shaped_cost_units",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise BacktestJobTransitionError(
                    f"BacktestJobParityClassification.{field_name} must be bool"
                )
        normalized_reason = self.nr2_classification_reason.strip()
        if not normalized_reason:
            raise BacktestJobTransitionError(
                "BacktestJobParityClassification.nr2_classification_reason must be non-empty"
            )
        object.__setattr__(self, "parity_class", normalized_parity_class)
        object.__setattr__(self, "nr2_classification_reason", normalized_reason)

    def to_json_object(self) -> dict[str, Any]:
        """
        Convert persisted parity-classification evidence into deterministic JSON payload.

        Args:
            None.
        Returns:
            dict[str, Any]: JSON-safe classification payload.
        Assumptions:
            Boolean evidence fields stay explicit to avoid mixed-version inference.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "disabled_risk_single_cell": self.disabled_risk_single_cell,
            "low_indicator_block_cardinality": self.low_indicator_block_cardinality,
            "narrowed_retained_row_evidence": self.narrowed_retained_row_evidence,
            "notebook_shaped_cost_units": self.notebook_shaped_cost_units,
            "nr2_classification_reason": self.nr2_classification_reason,
            "parity_class": self.parity_class,
        }

    @classmethod
    def from_json_object(
        cls,
        *,
        value: Mapping[str, Any],
    ) -> "BacktestJobParityClassification":
        """
        Build persisted parity-classification evidence from JSON object payload.

        Args:
            value: Raw persisted JSON object.
        Returns:
            BacktestJobParityClassification: Normalized immutable classification payload.
        Assumptions:
            Repository mapping validates the top-level object boundary before calling this helper.
        Raises:
            BacktestJobTransitionError: If one required field is missing or malformed.
        Side Effects:
            None.
        """
        return cls(
            parity_class=_required_non_empty_str_from_json(
                name="BacktestJobParityClassification.parity_class",
                value=value.get("parity_class"),
            ),
            disabled_risk_single_cell=_required_bool_from_json(
                name="BacktestJobParityClassification.disabled_risk_single_cell",
                value=value.get("disabled_risk_single_cell"),
            ),
            low_indicator_block_cardinality=_required_bool_from_json(
                name="BacktestJobParityClassification.low_indicator_block_cardinality",
                value=value.get("low_indicator_block_cardinality"),
            ),
            narrowed_retained_row_evidence=_required_bool_from_json(
                name="BacktestJobParityClassification.narrowed_retained_row_evidence",
                value=value.get("narrowed_retained_row_evidence"),
            ),
            notebook_shaped_cost_units=_required_bool_from_json(
                name="BacktestJobParityClassification.notebook_shaped_cost_units",
                value=value.get("notebook_shaped_cost_units"),
            ),
            nr2_classification_reason=_required_non_empty_str_from_json(
                name="BacktestJobParityClassification.nr2_classification_reason",
                value=value.get("nr2_classification_reason"),
            ),
        )


@dataclass(frozen=True, slots=True)
class BacktestJobParityRuntimeState:
    """
    Compact persisted parity runtime state reused by worker resume for the no-risk parity class.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
    """

    execution_profile_mode: str
    parity_classification: BacktestJobParityClassification
    retained_rows_per_indicator: tuple[BacktestJobParityRetainedRowsCounter, ...]
    retained_rows_total: int
    narrowed_combo_total: int
    narrowed_compute_combo_total: int
    no_risk_finalization_count: int
    exact_replay_count: int = 0
    deterministic_combo_ordering: str = "stage_a_index"
    stage_b_execution_mode: str = "bypassed_no_risk"
    stage_b_process_fallback_threshold: str = "none"

    def __post_init__(self) -> None:
        """
        Validate compact persisted parity runtime state invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            D5 persists enough runtime-shape evidence to let the worker resume the canonical
            no-risk exact class without silently re-entering hybrid-era reduced-plan semantics.
        Raises:
            BacktestJobTransitionError: If one field violates the persisted parity-state contract.
        Side Effects:
            Normalizes literal strings and retained-row counters into immutable tuples.
        """
        normalized_execution_profile_mode = self.execution_profile_mode.strip().lower()
        if normalized_execution_profile_mode != "exact_no_risk_parity":
            raise BacktestJobTransitionError(
                "BacktestJobParityRuntimeState.execution_profile_mode must be "
                "'exact_no_risk_parity'"
            )
        if not isinstance(self.parity_classification, BacktestJobParityClassification):
            raise BacktestJobTransitionError(
                "BacktestJobParityRuntimeState.parity_classification must be "
                "BacktestJobParityClassification"
            )
        if len(self.retained_rows_per_indicator) == 0:
            raise BacktestJobTransitionError(
                "BacktestJobParityRuntimeState.retained_rows_per_indicator must be non-empty"
            )
        normalized_retained_rows: list[BacktestJobParityRetainedRowsCounter] = []
        for raw_counter in self.retained_rows_per_indicator:
            if not isinstance(raw_counter, BacktestJobParityRetainedRowsCounter):
                raise BacktestJobTransitionError(
                    "BacktestJobParityRuntimeState.retained_rows_per_indicator items must be "
                    "BacktestJobParityRetainedRowsCounter"
                )
            normalized_retained_rows.append(raw_counter)
        indicator_ids = tuple(item.indicator_id for item in normalized_retained_rows)
        if len(indicator_ids) != len(set(indicator_ids)):
            raise BacktestJobTransitionError(
                "BacktestJobParityRuntimeState.retained_rows_per_indicator must not duplicate "
                "indicator ids"
            )
        retained_rows_total = _required_positive_int_from_json(
            name="BacktestJobParityRuntimeState.retained_rows_total",
            value=self.retained_rows_total,
        )
        if sum(item.retained_rows for item in normalized_retained_rows) != retained_rows_total:
            raise BacktestJobTransitionError(
                "BacktestJobParityRuntimeState.retained_rows_total must equal the sum of "
                "retained_rows_per_indicator"
            )
        normalized_narrowed_combo_total = _required_positive_int_from_json(
            name="BacktestJobParityRuntimeState.narrowed_combo_total",
            value=self.narrowed_combo_total,
        )
        normalized_narrowed_compute_combo_total = _required_positive_int_from_json(
            name="BacktestJobParityRuntimeState.narrowed_compute_combo_total",
            value=self.narrowed_compute_combo_total,
        )
        normalized_no_risk_finalization_count = _required_positive_int_from_json(
            name="BacktestJobParityRuntimeState.no_risk_finalization_count",
            value=self.no_risk_finalization_count,
        )
        normalized_exact_replay_count = _required_non_negative_int_from_json(
            name="BacktestJobParityRuntimeState.exact_replay_count",
            value=self.exact_replay_count,
        )
        normalized_combo_ordering = self.deterministic_combo_ordering.strip()
        if not normalized_combo_ordering:
            raise BacktestJobTransitionError(
                "BacktestJobParityRuntimeState.deterministic_combo_ordering must be non-empty"
            )
        normalized_stage_b_execution_mode = self.stage_b_execution_mode.strip().lower()
        if normalized_stage_b_execution_mode != "bypassed_no_risk":
            raise BacktestJobTransitionError(
                "BacktestJobParityRuntimeState.stage_b_execution_mode must be "
                "'bypassed_no_risk'"
            )
        normalized_stage_b_process_fallback_threshold = (
            self.stage_b_process_fallback_threshold.strip().lower()
        )
        if normalized_stage_b_process_fallback_threshold != "none":
            raise BacktestJobTransitionError(
                "BacktestJobParityRuntimeState.stage_b_process_fallback_threshold must be 'none'"
            )
        object.__setattr__(self, "execution_profile_mode", normalized_execution_profile_mode)
        object.__setattr__(
            self,
            "retained_rows_per_indicator",
            tuple(normalized_retained_rows),
        )
        object.__setattr__(self, "retained_rows_total", retained_rows_total)
        object.__setattr__(self, "narrowed_combo_total", normalized_narrowed_combo_total)
        object.__setattr__(
            self,
            "narrowed_compute_combo_total",
            normalized_narrowed_compute_combo_total,
        )
        object.__setattr__(
            self,
            "no_risk_finalization_count",
            normalized_no_risk_finalization_count,
        )
        object.__setattr__(self, "exact_replay_count", normalized_exact_replay_count)
        object.__setattr__(
            self,
            "deterministic_combo_ordering",
            normalized_combo_ordering,
        )
        object.__setattr__(
            self,
            "stage_b_execution_mode",
            normalized_stage_b_execution_mode,
        )
        object.__setattr__(
            self,
            "stage_b_process_fallback_threshold",
            normalized_stage_b_process_fallback_threshold,
        )

    def to_json_object(self) -> dict[str, Any]:
        """
        Convert compact parity runtime state into deterministic JSON-safe payload.

        Args:
            None.
        Returns:
            dict[str, Any]: JSON-safe parity runtime-state payload.
        Assumptions:
            Persisted parity runtime state remains internal-only and additive.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "deterministic_combo_ordering": self.deterministic_combo_ordering,
            "exact_replay_count": self.exact_replay_count,
            "execution_profile_mode": self.execution_profile_mode,
            "narrowed_combo_total": self.narrowed_combo_total,
            "narrowed_compute_combo_total": self.narrowed_compute_combo_total,
            "no_risk_finalization_count": self.no_risk_finalization_count,
            "parity_classification": self.parity_classification.to_json_object(),
            "retained_rows_per_indicator": [
                item.to_json_object() for item in self.retained_rows_per_indicator
            ],
            "retained_rows_total": self.retained_rows_total,
            "stage_b_execution_mode": self.stage_b_execution_mode,
            "stage_b_process_fallback_threshold": self.stage_b_process_fallback_threshold,
        }

    @classmethod
    def from_json_object(
        cls,
        *,
        value: Mapping[str, Any],
    ) -> "BacktestJobParityRuntimeState":
        """
        Build compact parity runtime state from persisted JSON object payload.

        Args:
            value: Raw persisted JSON object.
        Returns:
            BacktestJobParityRuntimeState: Normalized immutable parity runtime state.
        Assumptions:
            Repository mapping validates the top-level object boundary before calling this helper.
        Raises:
            BacktestJobTransitionError: If one required field is missing or malformed.
        Side Effects:
            None.
        """
        raw_retained_rows = _required_json_sequence(
            value=value,
            field_name="retained_rows_per_indicator",
        )
        retained_rows: list[BacktestJobParityRetainedRowsCounter] = []
        for raw_item in raw_retained_rows:
            if not isinstance(raw_item, Mapping):
                raise BacktestJobTransitionError(
                    "BacktestJobParityRuntimeState.retained_rows_per_indicator must contain "
                    "JSON objects"
                )
            retained_rows.append(
                BacktestJobParityRetainedRowsCounter.from_json_object(
                    value=dict(raw_item)
                )
            )
        raw_parity_classification = _required_json_object(
            value=value,
            field_name="parity_classification",
            entity_name="BacktestJobParityRuntimeState",
        )
        return cls(
            execution_profile_mode=_required_non_empty_str_from_json(
                name="BacktestJobParityRuntimeState.execution_profile_mode",
                value=value.get("execution_profile_mode"),
            ),
            parity_classification=BacktestJobParityClassification.from_json_object(
                value=raw_parity_classification
            ),
            retained_rows_per_indicator=tuple(retained_rows),
            retained_rows_total=_required_positive_int_from_json(
                name="BacktestJobParityRuntimeState.retained_rows_total",
                value=value.get("retained_rows_total"),
            ),
            narrowed_combo_total=_required_positive_int_from_json(
                name="BacktestJobParityRuntimeState.narrowed_combo_total",
                value=value.get("narrowed_combo_total"),
            ),
            narrowed_compute_combo_total=_required_positive_int_from_json(
                name="BacktestJobParityRuntimeState.narrowed_compute_combo_total",
                value=value.get("narrowed_compute_combo_total"),
            ),
            no_risk_finalization_count=_required_positive_int_from_json(
                name="BacktestJobParityRuntimeState.no_risk_finalization_count",
                value=value.get("no_risk_finalization_count"),
            ),
            exact_replay_count=_required_non_negative_int_from_json(
                name="BacktestJobParityRuntimeState.exact_replay_count",
                value=value.get("exact_replay_count"),
            ),
            deterministic_combo_ordering=_required_non_empty_str_from_json(
                name="BacktestJobParityRuntimeState.deterministic_combo_ordering",
                value=value.get("deterministic_combo_ordering"),
            ),
            stage_b_execution_mode=_required_non_empty_str_from_json(
                name="BacktestJobParityRuntimeState.stage_b_execution_mode",
                value=value.get("stage_b_execution_mode"),
            ),
            stage_b_process_fallback_threshold=_required_non_empty_str_from_json(
                name="BacktestJobParityRuntimeState.stage_b_process_fallback_threshold",
                value=value.get("stage_b_process_fallback_threshold"),
            ),
        )


@dataclass(frozen=True, slots=True)
class BacktestJobStageAShortlist:
    """
    Persisted Stage-A shortlist projection used for deterministic restart/resume in worker.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
      - alembic/versions/20260222_0003_backtest_jobs_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
    """

    job_id: UUID
    stage_a_indexes: tuple[int, ...]
    stage_a_variants_total: int
    risk_total: int
    preselect_used: int
    updated_at: datetime
    no_risk_exact_rows: tuple["BacktestJobStageANoRiskExactRow", ...] | None = None
    parity_runtime_state: BacktestJobParityRuntimeState | None = None

    def __post_init__(self) -> None:
        """
        Validate Stage-A shortlist payload shape and invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `stage_a_indexes` order is deterministic as produced by Stage-A ranking.
        Raises:
            BacktestJobTransitionError: If one invariant is violated.
        Side Effects:
            Normalizes indexes and optional no-risk exact rows into immutable tuples.
        """
        if self.stage_a_variants_total <= 0:
            raise BacktestJobTransitionError(
                "BacktestJobStageAShortlist.stage_a_variants_total must be > 0"
            )
        if self.risk_total <= 0:
            raise BacktestJobTransitionError("BacktestJobStageAShortlist.risk_total must be > 0")
        if self.preselect_used <= 0:
            raise BacktestJobTransitionError(
                "BacktestJobStageAShortlist.preselect_used must be > 0"
            )

        normalized_indexes: list[int] = []
        if len(self.stage_a_indexes) == 0:
            raise BacktestJobTransitionError(
                "BacktestJobStageAShortlist.stage_a_indexes must be non-empty"
            )
        for raw_index in self.stage_a_indexes:
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise BacktestJobTransitionError(
                    "BacktestJobStageAShortlist.stage_a_indexes items must be integers"
                )
            if raw_index < 0:
                raise BacktestJobTransitionError(
                    "BacktestJobStageAShortlist.stage_a_indexes items must be >= 0"
                )
            normalized_indexes.append(raw_index)

        normalized_no_risk_exact_rows: tuple[BacktestJobStageANoRiskExactRow, ...] | None = None
        if self.no_risk_exact_rows is not None:
            if self.risk_total != 1:
                raise BacktestJobTransitionError(
                    "BacktestJobStageAShortlist.no_risk_exact_rows requires risk_total == 1"
                )
            normalized_rows: list[BacktestJobStageANoRiskExactRow] = []
            for raw_row in self.no_risk_exact_rows:
                if not isinstance(raw_row, BacktestJobStageANoRiskExactRow):
                    raise BacktestJobTransitionError(
                        "BacktestJobStageAShortlist.no_risk_exact_rows items must be "
                        "BacktestJobStageANoRiskExactRow"
                    )
                normalized_rows.append(raw_row)
            if len(normalized_rows) != len(normalized_indexes):
                raise BacktestJobTransitionError(
                    "BacktestJobStageAShortlist.no_risk_exact_rows must align with "
                    "stage_a_indexes"
                )
            normalized_no_risk_exact_rows = tuple(normalized_rows)

        normalized_parity_runtime_state: BacktestJobParityRuntimeState | None = None
        if self.parity_runtime_state is not None:
            if not isinstance(self.parity_runtime_state, BacktestJobParityRuntimeState):
                raise BacktestJobTransitionError(
                    "BacktestJobStageAShortlist.parity_runtime_state must be "
                    "BacktestJobParityRuntimeState"
                )
            if self.risk_total != 1:
                raise BacktestJobTransitionError(
                    "BacktestJobStageAShortlist.parity_runtime_state requires risk_total == 1"
                )
            if normalized_no_risk_exact_rows is None:
                raise BacktestJobTransitionError(
                    "BacktestJobStageAShortlist.parity_runtime_state requires "
                    "no_risk_exact_rows"
                )
            if (
                self.parity_runtime_state.no_risk_finalization_count
                != len(normalized_indexes)
            ):
                raise BacktestJobTransitionError(
                    "BacktestJobStageAShortlist.parity_runtime_state."
                    "no_risk_finalization_count must align with stage_a_indexes"
                )
            normalized_parity_runtime_state = self.parity_runtime_state

        _ensure_utc_datetime(name="updated_at", value=self.updated_at)
        object.__setattr__(self, "stage_a_indexes", tuple(normalized_indexes))
        object.__setattr__(self, "no_risk_exact_rows", normalized_no_risk_exact_rows)
        object.__setattr__(self, "parity_runtime_state", normalized_parity_runtime_state)

    def to_json_array(self) -> list[int]:
        """
        Convert shortlist indexes into JSON array payload for SQL adapter writes.

        Args:
            None.
        Returns:
            list[int]: Deterministic ordered indexes list.
        Assumptions:
            Order is preserved from immutable `stage_a_indexes` tuple.
        Raises:
            None.
        Side Effects:
            None.
        """
        return list(self.stage_a_indexes)

    def to_no_risk_exact_rows_json_array(self) -> list[dict[str, Any]] | None:
        """
        Convert optional no-risk exact rows into JSON array payload for SQL adapter writes.

        Args:
            None.
        Returns:
            list[dict[str, Any]] | None: Deterministic row payload aligned to `stage_a_indexes`.
        Assumptions:
            Missing payload means additive compatibility fallback for legacy rows.
        Raises:
            None.
        Side Effects:
            None.
        """
        if self.no_risk_exact_rows is None:
            return None
        return [row.to_json_object() for row in self.no_risk_exact_rows]

    def to_parity_runtime_state_json_object(self) -> dict[str, Any] | None:
        """
        Convert optional parity runtime state into JSON object payload for SQL adapter writes.

        Args:
            None.
        Returns:
            dict[str, Any] | None: Compact parity runtime-state payload, or `None` for legacy rows.
        Assumptions:
            Missing payload means explicit fallback to live Stage A recomputation on resume.
        Raises:
            None.
        Side Effects:
            None.
        """
        if self.parity_runtime_state is None:
            return None
        return self.parity_runtime_state.to_json_object()


@dataclass(frozen=True, slots=True)
class BacktestJobStageANoRiskExactRow:
    """
    Compact exact no-risk Stage A row persisted additively for worker finalization reuse.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
    """

    entry_signal_idx: tuple[int, ...]
    entry_exec_idx: tuple[int, ...]
    direction: tuple[int, ...]
    sig_exit_signal_idx: tuple[int, ...]
    sig_exit_exec_idx: tuple[int, ...]
    total_return_pct: float
    max_drawdown_pct: float
    return_over_max_drawdown: float
    profit_factor: float
    trade_count: int
    sharpe_trades: float
    win_rate_pct: float
    avg_trade_ret_pct: float
    avg_trade_exec_bars: float
    exposure_pct: float
    memory_shape_bucket: str = "compact_trade_arrays"

    def __post_init__(self) -> None:
        """
        Validate one compact no-risk exact row and normalize persisted scalar/array payloads.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Persisted row payload stays compact-trade-array-only and keeps metric values exact,
            including non-finite ranking edge cases encoded outside JSON as deterministic strings.
        Raises:
            BacktestJobTransitionError: If array lengths drift or one scalar violates the contract.
        Side Effects:
            Normalizes compact-trade arrays into immutable builtin tuples and metric scalars into
            builtin `float`/`int`.
        """
        entry_signal_idx = _normalize_integer_tuple(
            name="entry_signal_idx",
            values=self.entry_signal_idx,
            minimum=0,
        )
        entry_exec_idx = _normalize_integer_tuple(
            name="entry_exec_idx",
            values=self.entry_exec_idx,
            minimum=0,
        )
        direction = _normalize_integer_tuple(name="direction", values=self.direction, minimum=-1)
        sig_exit_signal_idx = _normalize_integer_tuple(
            name="sig_exit_signal_idx",
            values=self.sig_exit_signal_idx,
            minimum=-1,
        )
        sig_exit_exec_idx = _normalize_integer_tuple(
            name="sig_exit_exec_idx",
            values=self.sig_exit_exec_idx,
            minimum=0,
        )
        trade_count = len(entry_signal_idx)
        if self.memory_shape_bucket != "compact_trade_arrays":
            raise BacktestJobTransitionError(
                "BacktestJobStageANoRiskExactRow.memory_shape_bucket must be "
                "'compact_trade_arrays'"
            )
        if len(entry_exec_idx) != trade_count:
            raise BacktestJobTransitionError(
                "BacktestJobStageANoRiskExactRow.entry_exec_idx must align with entry_signal_idx"
            )
        if len(direction) != trade_count:
            raise BacktestJobTransitionError(
                "BacktestJobStageANoRiskExactRow.direction must align with entry_signal_idx"
            )
        if len(sig_exit_signal_idx) != trade_count:
            raise BacktestJobTransitionError(
                "BacktestJobStageANoRiskExactRow.sig_exit_signal_idx must align with "
                "entry_signal_idx"
            )
        if len(sig_exit_exec_idx) != trade_count:
            raise BacktestJobTransitionError(
                "BacktestJobStageANoRiskExactRow.sig_exit_exec_idx must align with "
                "entry_signal_idx"
            )
        if any(raw_value not in (-1, 1) for raw_value in direction):
            raise BacktestJobTransitionError(
                "BacktestJobStageANoRiskExactRow.direction must contain only -1 or 1"
            )
        if any(
            exit_exec_idx < entry_exec_idx[item_index]
            for item_index, exit_exec_idx in enumerate(sig_exit_exec_idx)
        ):
            raise BacktestJobTransitionError(
                "BacktestJobStageANoRiskExactRow.sig_exit_exec_idx must stay >= entry_exec_idx"
            )
        if isinstance(self.trade_count, bool) or not isinstance(self.trade_count, int):
            raise BacktestJobTransitionError(
                "BacktestJobStageANoRiskExactRow.trade_count must be integer"
            )
        if self.trade_count < 0:
            raise BacktestJobTransitionError(
                "BacktestJobStageANoRiskExactRow.trade_count must be >= 0"
            )
        if self.trade_count != trade_count:
            raise BacktestJobTransitionError(
                "BacktestJobStageANoRiskExactRow.trade_count must align with compact arrays"
            )

        object.__setattr__(self, "entry_signal_idx", entry_signal_idx)
        object.__setattr__(self, "entry_exec_idx", entry_exec_idx)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "sig_exit_signal_idx", sig_exit_signal_idx)
        object.__setattr__(self, "sig_exit_exec_idx", sig_exit_exec_idx)
        object.__setattr__(
            self,
            "total_return_pct",
            _normalize_metric_float(name="total_return_pct", value=self.total_return_pct),
        )
        object.__setattr__(
            self,
            "max_drawdown_pct",
            _normalize_metric_float(name="max_drawdown_pct", value=self.max_drawdown_pct),
        )
        object.__setattr__(
            self,
            "return_over_max_drawdown",
            _normalize_metric_float(
                name="return_over_max_drawdown",
                value=self.return_over_max_drawdown,
            ),
        )
        object.__setattr__(
            self,
            "profit_factor",
            _normalize_metric_float(name="profit_factor", value=self.profit_factor),
        )
        object.__setattr__(
            self,
            "sharpe_trades",
            _normalize_metric_float(name="sharpe_trades", value=self.sharpe_trades),
        )
        object.__setattr__(
            self,
            "win_rate_pct",
            _normalize_metric_float(name="win_rate_pct", value=self.win_rate_pct),
        )
        object.__setattr__(
            self,
            "avg_trade_ret_pct",
            _normalize_metric_float(name="avg_trade_ret_pct", value=self.avg_trade_ret_pct),
        )
        object.__setattr__(
            self,
            "avg_trade_exec_bars",
            _normalize_metric_float(
                name="avg_trade_exec_bars",
                value=self.avg_trade_exec_bars,
            ),
        )
        object.__setattr__(
            self,
            "exposure_pct",
            _normalize_metric_float(name="exposure_pct", value=self.exposure_pct),
        )

    def to_json_object(self) -> dict[str, Any]:
        """
        Convert one exact no-risk row into deterministic JSON-safe storage payload.

        Args:
            None.
        Returns:
            dict[str, Any]: Compact JSON object aligned with the additive shortlist contract.
        Assumptions:
            Non-finite metric scalars are serialized as deterministic string tokens so JSONB
            storage stays backward-readable before global finite-metric sanitization lands.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "avg_trade_exec_bars": _json_metric_scalar(value=self.avg_trade_exec_bars),
            "avg_trade_ret_pct": _json_metric_scalar(value=self.avg_trade_ret_pct),
            "direction": list(self.direction),
            "entry_exec_idx": list(self.entry_exec_idx),
            "entry_signal_idx": list(self.entry_signal_idx),
            "exposure_pct": _json_metric_scalar(value=self.exposure_pct),
            "max_drawdown_pct": _json_metric_scalar(value=self.max_drawdown_pct),
            "memory_shape_bucket": self.memory_shape_bucket,
            "profit_factor": _json_metric_scalar(value=self.profit_factor),
            "return_over_max_drawdown": _json_metric_scalar(
                value=self.return_over_max_drawdown
            ),
            "sharpe_trades": _json_metric_scalar(value=self.sharpe_trades),
            "sig_exit_exec_idx": list(self.sig_exit_exec_idx),
            "sig_exit_signal_idx": list(self.sig_exit_signal_idx),
            "total_return_pct": _json_metric_scalar(value=self.total_return_pct),
            "trade_count": self.trade_count,
            "win_rate_pct": _json_metric_scalar(value=self.win_rate_pct),
        }

    @classmethod
    def from_json_object(
        cls,
        *,
        value: Mapping[str, Any],
    ) -> BacktestJobStageANoRiskExactRow:
        """
        Build one compact no-risk exact row from persisted JSON object payload.

        Args:
            value: Raw persisted JSON object for one shortlisted Stage A row.
        Returns:
            BacktestJobStageANoRiskExactRow: Normalized immutable row payload.
        Assumptions:
            Storage rows may carry special metric string tokens (`Infinity`, `-Infinity`, `NaN`)
            until C5 introduces global finite-summary normalization.
        Raises:
            BacktestJobTransitionError: If one JSON field is missing or malformed.
        Side Effects:
            None.
        """
        return cls(
            entry_signal_idx=_normalize_integer_tuple(
                name="entry_signal_idx",
                values=_required_json_sequence(value=value, field_name="entry_signal_idx"),
                minimum=0,
            ),
            entry_exec_idx=_normalize_integer_tuple(
                name="entry_exec_idx",
                values=_required_json_sequence(value=value, field_name="entry_exec_idx"),
                minimum=0,
            ),
            direction=_normalize_integer_tuple(
                name="direction",
                values=_required_json_sequence(value=value, field_name="direction"),
                minimum=-1,
            ),
            sig_exit_signal_idx=_normalize_integer_tuple(
                name="sig_exit_signal_idx",
                values=_required_json_sequence(
                    value=value,
                    field_name="sig_exit_signal_idx",
                ),
                minimum=-1,
            ),
            sig_exit_exec_idx=_normalize_integer_tuple(
                name="sig_exit_exec_idx",
                values=_required_json_sequence(value=value, field_name="sig_exit_exec_idx"),
                minimum=0,
            ),
            total_return_pct=_metric_float_from_json(
                name="total_return_pct",
                value=value.get("total_return_pct"),
            ),
            max_drawdown_pct=_metric_float_from_json(
                name="max_drawdown_pct",
                value=value.get("max_drawdown_pct"),
            ),
            return_over_max_drawdown=_metric_float_from_json(
                name="return_over_max_drawdown",
                value=value.get("return_over_max_drawdown"),
            ),
            profit_factor=_metric_float_from_json(
                name="profit_factor",
                value=value.get("profit_factor"),
            ),
            trade_count=_required_non_negative_int_from_json(
                name="trade_count",
                value=value.get("trade_count"),
            ),
            sharpe_trades=_metric_float_from_json(
                name="sharpe_trades",
                value=value.get("sharpe_trades"),
            ),
            win_rate_pct=_metric_float_from_json(
                name="win_rate_pct",
                value=value.get("win_rate_pct"),
            ),
            avg_trade_ret_pct=_metric_float_from_json(
                name="avg_trade_ret_pct",
                value=value.get("avg_trade_ret_pct"),
            ),
            avg_trade_exec_bars=_metric_float_from_json(
                name="avg_trade_exec_bars",
                value=value.get("avg_trade_exec_bars"),
            ),
            exposure_pct=_metric_float_from_json(
                name="exposure_pct",
                value=value.get("exposure_pct"),
            ),
            memory_shape_bucket=str(
                value.get("memory_shape_bucket", "compact_trade_arrays")
            ),
        )



def report_table_md_allowed_for_state(*, state: BacktestJobState) -> bool:
    """
    Check legacy `report_table_md` compatibility policy for one job state.

    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
      - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
      - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
    Args:
        state: Job lifecycle state.
    Returns:
        bool: Always `False` under the summary-only persisted rows contract.
    Assumptions:
        `report_table_md` is deprecated and remains outside the R7-01 persisted results contract.
    Raises:
        None.
    Side Effects:
        None.
    """
    _ = state
    return False


def _normalize_optional_non_negative_float(*, name: str, value: float | None) -> float | None:
    """
    Normalize one optional non-negative percentage field used by summary-only persisted rows.

    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
      - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
    Args:
        name: Field name used in deterministic error messages.
        value: Optional raw percentage value.
    Returns:
        float | None: Normalized float value or `None`.
    Assumptions:
        Persisted best TP/SL percentages are nullable and must be non-negative when present.
    Raises:
        BacktestJobTransitionError: If the value is boolean, non-numeric, or negative.
    Side Effects:
        None.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BacktestJobTransitionError(f"BacktestJobTopVariant.{name} must be numeric")
    normalized = float(value)
    if normalized < 0.0:
        raise BacktestJobTransitionError(f"BacktestJobTopVariant.{name} must be >= 0")
    return normalized



def _ensure_sha256_key(*, name: str, value: str) -> None:
    """
    Validate canonical SHA-256 hex key literal format.

    Args:
        name: Field name used in deterministic error messages.
        value: Candidate key literal.
    Returns:
        None.
    Assumptions:
        Keys are lowercase 64-char SHA-256 hex values.
    Raises:
        BacktestJobTransitionError: If key shape is invalid.
    Side Effects:
        None.
    """
    if len(value) != 64:
        raise BacktestJobTransitionError(
            f"BacktestJobTopVariant.{name} must be 64 lowercase hex chars"
        )
    allowed = set("0123456789abcdef")
    if any(char not in allowed for char in value):
        raise BacktestJobTransitionError(
            f"BacktestJobTopVariant.{name} must be 64 lowercase hex chars"
        )



def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    """
    Validate timezone-aware UTC datetime field.

    Args:
        name: Field name used in deterministic error messages.
        value: Datetime value.
    Returns:
        None.
    Assumptions:
        Persisted result timestamps are UTC-aware.
    Raises:
        BacktestJobTransitionError: If datetime is naive or non-UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise BacktestJobTransitionError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise BacktestJobTransitionError(f"{name} must be UTC datetime")



def _normalize_json_object(*, value: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize mapping payload into deterministic JSON-compatible object.

    Args:
        value: Raw object payload.
    Returns:
        dict[str, Any]: Key-sorted normalized JSON object.
    Assumptions:
        Mapping keys can be represented as strings.
    Raises:
        BacktestJobTransitionError: If payload cannot be normalized to JSON object.
    Side Effects:
        None.
    """
    normalized = _normalize_json_value(value=dict(value))
    if not isinstance(normalized, Mapping):
        raise BacktestJobTransitionError("Expected JSON object payload")
    try:
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError as error:
        raise BacktestJobTransitionError("Payload must be JSON-serializable") from error
    return dict(normalized)


def _normalize_integer_tuple(
    *,
    name: str,
    values: Sequence[Any],
    minimum: int,
) -> tuple[int, ...]:
    """
    Normalize one persisted integer array field into immutable builtin tuple.

    Args:
        name: Field name used in deterministic error messages.
        values: Raw sequence payload.
        minimum: Smallest allowed integer value.
    Returns:
        tuple[int, ...]: Normalized immutable integer tuple.
    Assumptions:
        Compact exact payload arrays stay one-dimensional and ordered.
    Raises:
        BacktestJobTransitionError: If the payload is not a sequence of integers above `minimum`.
    Side Effects:
        None.
    """
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise BacktestJobTransitionError(
            f"BacktestJobStageANoRiskExactRow.{name} must be sequence"
        )
    normalized: list[int] = []
    for raw_value in values:
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise BacktestJobTransitionError(
                f"BacktestJobStageANoRiskExactRow.{name} must contain integers"
            )
        if raw_value < minimum:
            raise BacktestJobTransitionError(
                f"BacktestJobStageANoRiskExactRow.{name} must be >= {minimum}"
            )
        normalized.append(raw_value)
    return tuple(normalized)


def _normalize_metric_float(*, name: str, value: Any) -> float:
    """
    Normalize one persisted metric scalar into builtin float while allowing non-finite values.

    Args:
        name: Field name used in deterministic error messages.
        value: Raw metric scalar.
    Returns:
        float: Normalized metric scalar.
    Assumptions:
        Stage A no-risk metrics may temporarily carry `inf`/`nan` edge cases before C5.
    Raises:
        BacktestJobTransitionError: If value is boolean or non-numeric.
    Side Effects:
        None.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BacktestJobTransitionError(
            f"BacktestJobStageANoRiskExactRow.{name} must be numeric"
        )
    return float(value)


def _required_json_sequence(
    *,
    value: Mapping[str, Any],
    field_name: str,
) -> Sequence[Any]:
    """
    Resolve one required JSON array field from a persisted exact-row object payload.

    Args:
        value: Raw persisted exact-row object.
        field_name: Required array field name.
    Returns:
        Sequence[Any]: Raw array-like payload for later normalization.
    Assumptions:
        Repository mapping validates the top-level object boundary before calling this helper.
    Raises:
        BacktestJobTransitionError: If the field is missing.
    Side Effects:
        None.
    """
    if field_name not in value:
        raise BacktestJobTransitionError(
            f"BacktestJobStageANoRiskExactRow.{field_name} is required"
    )
    return cast(Sequence[Any], value[field_name])


def _required_json_object(
    *,
    value: Mapping[str, Any],
    field_name: str,
    entity_name: str,
) -> dict[str, Any]:
    """
    Resolve one required nested JSON object field from persisted payload.

    Args:
        value: Raw persisted JSON object.
        field_name: Required nested field name.
        entity_name: Entity name used in deterministic error messages.
    Returns:
        dict[str, Any]: Nested JSON object payload.
    Assumptions:
        Repository mapping validates the top-level object boundary before calling this helper.
    Raises:
        BacktestJobTransitionError: If the field is missing or is not a JSON object.
    Side Effects:
        None.
    """
    if field_name not in value:
        raise BacktestJobTransitionError(f"{entity_name}.{field_name} is required")
    nested_value = value[field_name]
    if not isinstance(nested_value, Mapping):
        raise BacktestJobTransitionError(f"{entity_name}.{field_name} must be JSON object")
    return dict(nested_value)


def _required_non_negative_int_from_json(*, name: str, value: Any) -> int:
    """
    Parse one required non-negative integer from persisted JSON payload.

    Args:
        name: Field name used in deterministic error messages.
        value: Raw JSON scalar.
    Returns:
        int: Normalized non-negative integer.
    Assumptions:
        `trade_count` remains explicit in the additive persisted exact-row contract.
    Raises:
        BacktestJobTransitionError: If value is missing, boolean, non-integer, or negative.
    Side Effects:
        None.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise BacktestJobTransitionError(
            f"BacktestJobStageANoRiskExactRow.{name} must be integer"
        )
    if value < 0:
        raise BacktestJobTransitionError(
            f"BacktestJobStageANoRiskExactRow.{name} must be >= 0"
        )
    return int(value)


def _required_positive_int_from_json(*, name: str, value: Any) -> int:
    """
    Parse one required positive integer from persisted JSON payload.

    Args:
        name: Field name used in deterministic error messages.
        value: Raw JSON scalar.
    Returns:
        int: Normalized positive integer.
    Assumptions:
        Compact parity counters use explicit integer fields rather than inferred defaults.
    Raises:
        BacktestJobTransitionError: If value is missing, boolean, non-integer, or non-positive.
    Side Effects:
        None.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise BacktestJobTransitionError(f"{name} must be integer")
    if value <= 0:
        raise BacktestJobTransitionError(f"{name} must be > 0")
    return int(value)


def _required_bool_from_json(*, name: str, value: Any) -> bool:
    """
    Parse one required boolean from persisted JSON payload.

    Args:
        name: Field name used in deterministic error messages.
        value: Raw JSON scalar.
    Returns:
        bool: Normalized boolean value.
    Assumptions:
        Persisted parity evidence keeps boolean flags explicit for backward-readable resume.
    Raises:
        BacktestJobTransitionError: If value is not boolean.
    Side Effects:
        None.
    """
    if not isinstance(value, bool):
        raise BacktestJobTransitionError(f"{name} must be bool")
    return value


def _required_non_empty_str_from_json(*, name: str, value: Any) -> str:
    """
    Parse one required non-empty string from persisted JSON payload.

    Args:
        name: Field name used in deterministic error messages.
        value: Raw JSON scalar.
    Returns:
        str: Normalized stripped string.
    Assumptions:
        Persisted parity literals remain explicit and do not rely on implicit defaults.
    Raises:
        BacktestJobTransitionError: If value is missing, non-string, or blank.
    Side Effects:
        None.
    """
    if not isinstance(value, str):
        raise BacktestJobTransitionError(f"{name} must be string")
    normalized_value = value.strip()
    if not normalized_value:
        raise BacktestJobTransitionError(f"{name} must be non-empty")
    return normalized_value


def _json_metric_scalar(*, value: float) -> float | str:
    """
    Convert one metric scalar into JSON-safe persisted representation.

    Args:
        value: Metric scalar to persist.
    Returns:
        float | str: Float for finite values, or deterministic string token for non-finite ones.
    Assumptions:
        String tokens are temporary additive compatibility encoding until global C5 sanitization.
    Raises:
        None.
    Side Effects:
        None.
    """
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    return float(value)


def _metric_float_from_json(*, name: str, value: Any) -> float:
    """
    Parse one persisted metric scalar from JSON-safe representation back into float.

    Args:
        name: Field name used in deterministic error messages.
        value: Raw JSON scalar or deterministic non-finite string token.
    Returns:
        float: Parsed metric scalar.
    Assumptions:
        Non-finite values are encoded only as `Infinity`, `-Infinity`, or `NaN`.
    Raises:
        BacktestJobTransitionError: If the persisted scalar cannot be interpreted as metric float.
    Side Effects:
        None.
    """
    if isinstance(value, str):
        if value == "Infinity":
            return math.inf
        if value == "-Infinity":
            return -math.inf
        if value == "NaN":
            return math.nan
    return _normalize_metric_float(name=name, value=value)



def _normalize_json_value(*, value: Any) -> Any:
    """
    Normalize arbitrary JSON-like node into deterministic structure.

    Args:
        value: Raw JSON-like node.
    Returns:
        Any: Deterministic mapping/list/scalar value.
    Assumptions:
        Unknown non-JSON objects are stringified for stable persistence.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda item: str(item)):
            normalized_mapping[str(raw_key)] = _normalize_json_value(value=value[raw_key])
        return normalized_mapping

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_value(value=item) for item in value]

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


__all__ = [
    "BacktestJobParityClassification",
    "BacktestJobParityRetainedRowsCounter",
    "BacktestJobParityRuntimeState",
    "BacktestJobStageANoRiskExactRow",
    "BacktestJobStageAShortlist",
    "BacktestJobTopVariant",
    "report_table_md_allowed_for_state",
]
