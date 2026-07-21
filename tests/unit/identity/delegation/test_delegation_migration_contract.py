from __future__ import annotations

import runpy
from pathlib import Path


def test_delegation_migration_declares_exact_reversible_alembic_lineage() -> None:
    migration = runpy.run_path(
        str(
            Path(__file__).parents[4]
            / "alembic/versions/20260720_0044_identity_delegated_capabilities_v1.py"
        )
    )

    assert migration["revision"] == "20260720_0044"
    assert migration["down_revision"] == "20260711_0043"
    assert callable(migration["upgrade"])
    assert callable(migration["downgrade"])
