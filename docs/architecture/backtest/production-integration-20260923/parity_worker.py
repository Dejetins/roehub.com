"""Bounded source-isolated raw parity; full details go only to the controller pipe."""

from __future__ import annotations

import dataclasses
import datetime
import importlib
import json
import sys
from pathlib import Path
from uuid import UUID


def main():
    task = json.loads(Path(sys.argv[1]).read_text())
    source = Path(task["source"])
    research = Path(task["research"])
    sys.path.insert(0, str(research))
    cache_stats = importlib.import_module("scripts.backtest.cpu_integration.worker_s5").cache_stats
    isolate = importlib.import_module("scripts.backtest.cpu_integration.worker_s5").isolate

    isolate(source, Path(task["cache"]), 12)
    import numba as nb

    from apps.worker.backtest_job_runner.wiring.modules.full_job_compute import (
        build_full_job_compute_executor,
    )

    collect_details = importlib.import_module(
        "scripts.backtest.cpu_integration.capture_s2"
    ).collect_details
    score_capture = importlib.import_module(
        "scripts.backtest.cpu_integration.capture_s2"
    ).score_capture
    encode = importlib.import_module("scripts.backtest.cpu_integration.exact").encode
    CaptureAssembly = importlib.import_module(
        "scripts.backtest.cpu_integration.setup"
    ).CaptureAssembly
    setup = importlib.import_module("scripts.backtest.cpu_integration.setup").setup

    answers = []
    for index in range(2):
        pf, env = setup(Path(task["dataset"]))
        pre = pf.execute(task["case"]["request_payload"])
        executor = build_full_job_compute_executor(environ=env)
        capture = CaptureAssembly(executor.top_result_assembly)
        executor = dataclasses.replace(executor, top_result_assembly=capture)
        with score_capture() as audit:
            result = executor.execute(
                job_id=UUID(int=1),
                preflight=pre,
                updated_at=datetime.datetime(2026, 9, 1, tzinfo=datetime.UTC),
            )
        details = collect_details(executor, pre, result)
        stats = cache_stats(nb)
        hits = sum(x["hits"] for x in stats.values())
        misses = sum(x["misses"] for x in stats.values())
        if task["origin"] == "compile":
            assert misses > 0 and hits == 0, (hits, misses)
        else:
            assert hits > 0 and misses == 0, (hits, misses)
        assert nb.get_num_threads() == 12
        behavior = encode(
            dict(
                top_variants=result.top_variants,
                raw_top=capture.top,
                details=details,
                all_rows=audit,
                request_hash=pre.request_hash,
                result_config_hash=pre.result_config_hash,
            )
        )
        answers.append(
            dict(
                behavior=behavior,
                hits=hits,
                misses=misses,
                top_count=len(result.top_variants),
            )
        )
    checked = {}
    import hashlib

    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in {"trading", "apps"}:
            continue
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            assert path.is_relative_to(source), (name, str(path))
            checked[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(dict(answers=answers, imports=checked), separators=(",", ":")))


if __name__ == "__main__":
    main()
