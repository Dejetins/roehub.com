"""Read-only bounded storage admission for a trusted configured local artifact root."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from trading.contexts.backtest_artifacts.application.services.management import ManagementError


@dataclass(frozen=True)
class BaseStorageAdmission:
    root: Path
    root_id: str
    root_revision: int
    budget_bytes: int
    max_entries: int = 100_000

    def __call__(
        self,
        snapshot: Mapping[str, Any],
        reservation_bytes: int,
        *,
        budget_bytes: int | None = None,
    ) -> None:
        if (snapshot["root_id"], snapshot["root_revision"]) != (self.root_id, self.root_revision):
            raise ManagementError("root_revision_changed")
        budget = self.budget_bytes if budget_bytes is None else budget_bytes
        if budget <= 0:
            raise ManagementError("storage_unconfigured")
        root = self.root.absolute()
        for path in (root, *root.parents):
            if path.is_symlink():
                raise ManagementError("untrusted_root")
        parent = root
        while not parent.exists():
            parent = parent.parent
        usage, count = 0, 0
        pending = [root] if root.exists() else []
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    count += 1
                    if count > self.max_entries:
                        raise ManagementError("storage_inventory_unavailable")
                    if entry.is_symlink():
                        raise ManagementError("untrusted_root")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        usage += entry.stat(follow_symlinks=False).st_size
                    else:
                        raise ManagementError("untrusted_root")
        needed = reservation_bytes + 1_073_741_824
        if usage + needed > budget or shutil.disk_usage(parent).free < needed:
            raise ManagementError("resource_conflict")
