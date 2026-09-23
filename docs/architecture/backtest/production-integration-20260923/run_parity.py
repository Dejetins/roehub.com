"""Run three fixed cases serially; 600-second cap; compact evidence only."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[4]
    research = Path("/Users/daniildegtyarev/.codex/worktrees/224b/roehub.com")
    original = research / ".local_artifacts/cpu-integration-S0/sources/baseline"
    dataset = Path(
        "/Users/daniildegtyarev/Projects/roehub.com/.local_artifacts/"
        "ethusdt-binance-usdm-testset-20260920T001656Z"
    )
    output = Path(__file__).parent
    cases = json.loads(
        (output / (sys.argv[1] if len(sys.argv) > 1 else "parity-cases.json")).read_text()
    )
    result_name = sys.argv[2] if len(sys.argv) > 2 else "parity-result.json"
    scratch = Path(tempfile.mkdtemp(prefix="roehub-production-parity-"))
    began = time.monotonic()
    report: dict[str, Any] = dict(
        status="running", cap_seconds=600, cases=[], failures=[], scratch=str(scratch)
    )
    identity = json.loads((original / "source-identity.json").read_text())["files"]
    try:
        for path, sha in identity.items():
            assert hashlib.sha256((original / path).read_bytes()).hexdigest() == sha, path
        report["original_closure_sha256"] = digest(identity)
        for case in cases:
            reference = {}
            entry = dict(case_id=case["case_id"], comparisons=[], runs=[])
            report["cases"].append(entry)
            for variant, source in [("original", original), ("production", root)]:
                cache = scratch / case["case_id"] / variant
                cache.mkdir(parents=True)
                for origin in ["compile", "load"]:
                    task = dict(
                        source=str(source),
                        research=str(research),
                        dataset=str(dataset),
                        cache=str(cache),
                        case=case,
                        origin=origin,
                    )
                    path = scratch / "task.json"
                    path.write_text(json.dumps(task))
                    remaining = 600 - (time.monotonic() - began)
                    assert remaining > 0, "campaign budget exhausted"
                    proc = subprocess.run(
                        [sys.executable, "-I", "-B", str(output / "parity_worker.py"), str(path)],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        timeout=remaining,
                    )
                    assert proc.returncode == 0, proc.stderr[-12000:]
                    result = json.loads(proc.stdout)
                    entry["runs"].append(
                        dict(
                            variant=variant,
                            origin=origin,
                            imports_sha256=digest(result["imports"]),
                            samples=[
                                {k: v for k, v in a.items() if k != "behavior"}
                                for a in result["answers"]
                            ],
                        )
                    )
                    for index, sample in enumerate(result["answers"]):
                        mode = origin + ("-warm" if index else "")
                        if variant == "original":
                            reference[mode] = sample["behavior"]
                        else:
                            expected, actual = reference[mode], sample["behavior"]
                            assert expected == actual, (
                                case["case_id"],
                                mode,
                                digest(expected),
                                digest(actual),
                            )
                            entry["comparisons"].append(
                                dict(
                                    mode=mode,
                                    mismatches=0,
                                    original_sha256=digest(expected),
                                    production_sha256=digest(actual),
                                )
                            )
                    (output / result_name).write_text(json.dumps(report, indent=2) + "\n")
                    print(case["case_id"], variant, origin, "passed", flush=True)
            del reference
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["failures"].append(str(error))
        raise
    finally:
        shutil.rmtree(scratch)
        report["scratch_removed"] = not scratch.exists()
        report["elapsed_seconds"] = time.monotonic() - began
        (output / result_name).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
