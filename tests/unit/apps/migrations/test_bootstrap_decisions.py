from __future__ import annotations

from apps.migrations.bootstrap import (
    IdentityExchangeKeysLayout,
    decide_identity_exchange_keys_v2_action,
)


def _layout(
    *,
    table_exists: bool = True,
    row_count: int = 0,
    has_api_key: bool = True,
    has_api_key_enc: bool = False,
    has_api_key_hash: bool = False,
    has_api_key_last4: bool = False,
) -> IdentityExchangeKeysLayout:
    """
    Build deterministic layout fixture for identity exchange keys bootstrap decisions.

    Args:
        table_exists: Whether `identity_exchange_keys` exists.
        row_count: Number of rows currently present in the table.
        has_api_key: Presence of v1 column `api_key`.
        has_api_key_enc: Presence of v2 column `api_key_enc`.
        has_api_key_hash: Presence of v2 column `api_key_hash`.
        has_api_key_last4: Presence of v2 column `api_key_last4`.
    Returns:
        IdentityExchangeKeysLayout: In-memory schema snapshot fixture.
    Assumptions:
        Caller provides coherent column combination for desired scenario.
    Raises:
        None.
    Side Effects:
        None.
    """
    return IdentityExchangeKeysLayout(
        table_exists=table_exists,
        row_count=row_count,
        has_api_key=has_api_key,
        has_api_key_enc=has_api_key_enc,
        has_api_key_hash=has_api_key_hash,
        has_api_key_last4=has_api_key_last4,
    )


def test_decision_skips_when_v2_columns_are_present() -> None:
    """
    Verify `0004` is skipped when schema already has v2 exchange-key columns.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Existing v2 columns imply migration already completed on this schema.
    Raises:
        AssertionError: If decision does not return `skip`.
    Side Effects:
        None.
    """
    decision = decide_identity_exchange_keys_v2_action(
        layout=_layout(
            has_api_key=False,
            has_api_key_enc=True,
            has_api_key_hash=True,
            has_api_key_last4=True,
        )
    )

    assert decision.action == "skip"
    assert "already has v2 columns" in decision.reason


def test_decision_applies_when_v1_table_is_empty() -> None:
    """
    Verify `0004` is applied only for empty v1 table layout.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Empty table means no data re-encryption is needed before v2 migration.
    Raises:
        AssertionError: If decision does not return `apply`.
    Side Effects:
        None.
    """
    decision = decide_identity_exchange_keys_v2_action(layout=_layout(row_count=0))

    assert decision.action == "apply"
    assert "safe to apply" in decision.reason


def test_decision_fails_when_v1_table_has_data() -> None:
    """
    Verify bootstrap fails fast for non-empty v1 table to avoid unsafe `0004` execution.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Existing rows require explicit re-encryption migration before v2 schema change.
    Raises:
        AssertionError: If decision does not return `fail`.
    Side Effects:
        None.
    """
    decision = decide_identity_exchange_keys_v2_action(layout=_layout(row_count=3))

    assert decision.action == "fail"
    assert decision.hint is not None
    assert "re-encryption" in decision.hint


def test_decision_fails_when_table_is_missing() -> None:
    """
    Verify bootstrap fails when `identity_exchange_keys` table is absent after baseline stage.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Missing table indicates drift or failed `0003` execution.
    Raises:
        AssertionError: If decision does not return `fail`.
    Side Effects:
        None.
    """
    decision = decide_identity_exchange_keys_v2_action(
        layout=_layout(
            table_exists=False,
            has_api_key=False,
        )
    )

    assert decision.action == "fail"
    assert "table is missing" in decision.reason
