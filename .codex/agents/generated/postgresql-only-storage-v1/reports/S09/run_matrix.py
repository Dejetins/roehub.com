"""Full wall-clock matrix; stop on evidence failure, never shorten phases."""

import json
import subprocess
import sys

from tools.qa.postgres_production_analyze import OUT, analyze

phases = [
    ("normal1", "normal", 300),
    ("stress1", "stress", 300),
    ("normal2", "normal", 300),
    ("stress2", "stress", 300),
    ("normal3", "normal", 300),
    ("stress3", "stress", 300),
    ("soak1", "stress", 3600),
]
for name, mode, duration in phases:
    subprocess.run(
        [
            sys.executable,
            "tools/qa/postgres_production_driver.py",
            name,
            "--mode",
            mode,
            "--duration",
            str(duration),
        ],
        check=True,
    )
    result = analyze(name)
    (OUT / (name + "-analysis.json")).write_text(json.dumps(result, indent=2))
    print(json.dumps({"phase": name, "guardrails": result["guardrails"]}), flush=True)
    assert result["guardrails_pass"], name + " guardrail failure"
    assert result["server_oom_delta"] == 0
    assert all(
        v["completed_in_window"] == v["offered"]
        and v["completed"] == v["offered"]
        and v["errors"] == 0
        for v in result["summary"]["lanes"].values()
    )
    assert result["summary"]["lanes"]["full"]["completed"] >= 30
    assert result["summary"]["lanes"]["year"]["completed"] >= 30
print("FULL_MATRIX_COMPLETE", flush=True)
