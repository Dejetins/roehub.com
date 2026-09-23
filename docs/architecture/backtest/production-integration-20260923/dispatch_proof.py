"""Observe real normal-builder branches without replacing financial results."""

from __future__ import annotations

import datetime
import hashlib
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID


def main():
    output = Path(__file__).parent.resolve()
    root = output.parents[3]
    research = Path("/Users/daniildegtyarev/.codex/worktrees/224b/roehub.com")
    sys.path.insert(0, str(research))
    isolate = importlib.import_module("scripts.backtest.cpu_integration.worker_s5").isolate

    scratch = Path(tempfile.mkdtemp(prefix="roehub-production-dispatch-"))
    isolate(root, scratch, 12)
    from apps.worker.backtest_job_runner.wiring.modules.full_job_compute import (
        build_full_job_compute_executor,
    )

    setup = importlib.import_module("scripts.backtest.cpu_integration.setup").setup
    from trading.contexts.backtest.application.services.v2 import cost_permutation as cp
    from trading.contexts.backtest.application.services.v2 import integer_trade_tape as tape
    from trading.contexts.backtest.application.services.v2 import no_risk_exact as nr
    from trading.contexts.backtest.application.services.v2.matrix_backend import (
        prefix_traversal as pt,
    )

    counts = dict(cost_permutation=0, local_top_k=0, integer_tape=0, prefix_guard=0)
    fallback = {}
    originals = []

    def observe(module, name, key, successful=lambda result: True):
        original = getattr(module, name)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if successful(result):
                counts[key] += 1
            return result

        originals.append((module, name, original))
        setattr(module, name, wrapped)

    observe(cp, "balanced_order", "cost_permutation")
    observe(nr, "local_admission_indices", "local_top_k", lambda result: result is not None)
    observe(tape, "scan_integer_intervals", "integer_tape")
    observe(pt, "_canonical_product", "prefix_guard")
    original_record = cp.CostPermutationState.record

    def record(self, kernel, reason, **values):
        fallback[reason] = fallback.get(reason, 0) + 1
        return original_record(self, kernel, reason, **values)

    cp.CostPermutationState.record = record
    report: dict[str, Any] = dict(status="running", cases=[])
    try:
        cases = json.loads((output / "parity-cases.json").read_text()) + json.loads(
            (output / "prefix-case.json").read_text()
        )
        for case in cases:
            pf, env = setup(
                Path(
                    "/Users/daniildegtyarev/Projects/roehub.com/.local_artifacts/"
                    "ethusdt-binance-usdm-testset-20260920T001656Z"
                )
            )
            executor = build_full_job_compute_executor(environ=env)
            assert executor.compute_policy.cost_permutation_min_rows == 32
            assert executor.compute_policy.integer_tape_min_bars == 32
            before = counts.copy()
            result = executor.execute(
                job_id=UUID(int=1),
                preflight=pf.execute(case["request_payload"]),
                updated_at=datetime.datetime(2026, 9, 1, tzinfo=datetime.UTC),
            )
            report["cases"].append(
                dict(
                    case_id=case["case_id"],
                    calls={k: counts[k] - before[k] for k in counts},
                    prefix=result.exact_diagnostics["telemetry"].get("prefix_traversal"),
                )
            )
        assert all(counts.values()), counts
        assert fallback.get("insufficient_work", 0) > 0, fallback
        report.update(status="passed", counts=counts, permutation_reasons=fallback)
    finally:
        cp.CostPermutationState.record = original_record
        for module, name, original in originals:
            setattr(module, name, original)
        report["imports"] = {}
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] in {"trading", "apps"} and getattr(module, "__file__", None):
                p = Path(str(module.__file__)).resolve()
                assert p.is_relative_to(root), str(p)
                report["imports"][str(p.relative_to(root))] = hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
        shutil.rmtree(scratch)
        report["scratch_removed"] = not scratch.exists()
        (output / "dispatch-result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
