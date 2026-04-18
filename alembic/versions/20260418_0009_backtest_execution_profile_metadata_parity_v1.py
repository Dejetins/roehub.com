"""Extend `backtest_jobs` execution-profile metadata constraints for parity-first sync runs."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260418_0009"
down_revision = "20260418_0008"
branch_labels = None
depends_on = None

_PARITY_AWARE_EXECUTION_PROFILE_MODES: tuple[str, ...] = (
    "exact_small",
    "exact_parallel",
    "exact_no_risk_parity",
    "hybrid_conservative",
    "hybrid_family",
)
_LEGACY_EXECUTION_PROFILE_MODES: tuple[str, ...] = (
    "exact_small",
    "exact_parallel",
    "hybrid_conservative",
    "hybrid_family",
)


def _render_metadata_check_sql(
    *,
    column_name: str,
    constraint_name: str,
    allowed_modes: tuple[str, ...],
) -> str:
    """
    Build deterministic SQL that recreates one execution-profile metadata check constraint.

    Args:
        column_name: Nullable `backtest_jobs` metadata column being constrained.
        constraint_name: Existing or target Postgres check-constraint name.
        allowed_modes: Canonical lowercase literals accepted for the metadata column.
    Returns:
        str: One `ALTER TABLE ... DROP CONSTRAINT ... ADD CONSTRAINT ...` statement.
    Assumptions:
        Callers pass repository-owned identifiers from this migration module only.
    Raises:
        None.
    Side Effects:
        None.
    """
    allowed_literals = ",\n".join(
        f"                        '{mode}'" for mode in allowed_modes
    )
    return f"""
    ALTER TABLE backtest_jobs
        DROP CONSTRAINT IF EXISTS {constraint_name},
        ADD CONSTRAINT {constraint_name}
        CHECK (
            {column_name} IS NULL
            OR {column_name} IN (
{allowed_literals}
            )
        )
    """


def upgrade() -> None:
    """
    Extend persisted execution-profile metadata constraints for parity-first sync launches.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        D0-D5 runtime already persists `exact_no_risk_parity` into
        `execution_profile_mode_hint/effective_execution_profile_mode` for canonical sync-inline
        parity runs, so schema must accept the same literal without mutating existing rows.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Recreates both `backtest_jobs` execution-profile metadata check constraints.
    """
    op.execute(
        _render_metadata_check_sql(
            column_name="execution_profile_mode_hint",
            constraint_name="backtest_jobs_execution_profile_mode_hint_chk",
            allowed_modes=_PARITY_AWARE_EXECUTION_PROFILE_MODES,
        )
    )
    op.execute(
        _render_metadata_check_sql(
            column_name="effective_execution_profile_mode",
            constraint_name="backtest_jobs_effective_execution_profile_mode_chk",
            allowed_modes=_PARITY_AWARE_EXECUTION_PROFILE_MODES,
        )
    )


def downgrade() -> None:
    """
    Revert execution-profile metadata checks to the legacy pre-parity literal set.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Legacy readers can still recover historical profile metadata from `request_json` fallback
        once parity-only additive columns are nulled before the stricter checks are restored.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Clears parity-only additive metadata values and recreates the legacy check constraints.
    """
    op.execute(
        """
        UPDATE backtest_jobs
        SET execution_profile_mode_hint = NULL
        WHERE execution_profile_mode_hint = 'exact_no_risk_parity'
        """
    )
    op.execute(
        """
        UPDATE backtest_jobs
        SET effective_execution_profile_mode = NULL
        WHERE effective_execution_profile_mode = 'exact_no_risk_parity'
        """
    )
    op.execute(
        _render_metadata_check_sql(
            column_name="execution_profile_mode_hint",
            constraint_name="backtest_jobs_execution_profile_mode_hint_chk",
            allowed_modes=_LEGACY_EXECUTION_PROFILE_MODES,
        )
    )
    op.execute(
        _render_metadata_check_sql(
            column_name="effective_execution_profile_mode",
            constraint_name="backtest_jobs_effective_execution_profile_mode_chk",
            allowed_modes=_LEGACY_EXECUTION_PROFILE_MODES,
        )
    )
