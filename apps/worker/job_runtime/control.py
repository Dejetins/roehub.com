"""Port used by the OCI policy engine without owning Docker access."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from typing import Protocol


class DockerCommandRunner(Protocol):
    def run(
        self,
        command: Sequence[str],
        *,
        environ: Mapping[str, str],
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]: ...


__all__ = ["DockerCommandRunner"]
