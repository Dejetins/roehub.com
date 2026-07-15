from __future__ import annotations

from tools.backup.generate_schemas import generate


def test_backup_schemas_are_current() -> None:
    generate(check=True)
