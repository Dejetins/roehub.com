"""Recheck only changed source against immutable original per-regime raw digests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def main():
    output = Path(__file__).parent.resolve()
    root = output.parents[3]
    scratch = Path(tempfile.mkdtemp(prefix="roehub-layer-parity-"))
    reference = json.loads((output / "parity-prefix-result.json").read_text())["cases"][0]
    case = json.loads((output / "prefix-case.json").read_text())[0]
    task = dict(
        source=str(root),
        research="/Users/daniildegtyarev/.codex/worktrees/224b/roehub.com",
        dataset="/Users/daniildegtyarev/Projects/roehub.com/.local_artifacts/"
        "ethusdt-binance-usdm-testset-20260920T001656Z",
        cache=str(scratch / "cache"),
        case=case,
        origin="compile",
    )
    report: dict[str, Any] = {"status": "running", "cap_seconds": 120}
    try:
        task_path = scratch / "task.json"
        task_path.write_text(json.dumps(task))
        process = subprocess.run(
            [sys.executable, "-I", "-B", str(output / "parity_worker.py"), str(task_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert process.returncode == 0, process.stderr[-8000:]
        result = json.loads(process.stdout)
        pairs = []
        for index, answer in enumerate(result["answers"]):
            mode = "compile-warm" if index else "compile"
            expected = next(p for p in reference["comparisons"] if p["mode"] == mode)
            actual = hashlib.sha256(
                json.dumps(answer["behavior"], sort_keys=True).encode()
            ).hexdigest()
            assert actual == expected["original_sha256"], (mode, actual)
            pairs.append(
                dict(mode=mode, original_sha256=actual, production_sha256=actual, mismatches=0)
            )
        report = dict(
            status="passed",
            cap_seconds=120,
            comparisons=pairs,
            imports=result["imports"],
            rationale="Only envelope selection and additive diagnostics changed; "
            "unchanged financial kernels. Reuse four-regime prior proof; "
            "two new same-regime raw comparisons, no new speed claim.",
        )
    finally:
        shutil.rmtree(scratch)
        report["scratch_removed"] = not scratch.exists()
        (output / "layer-parity-result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
