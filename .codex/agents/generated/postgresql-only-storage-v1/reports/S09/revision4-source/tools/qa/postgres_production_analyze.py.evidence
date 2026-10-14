"""Evidence-only S09 guardrail and resource analysis; never edits stage state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".codex/agents/generated/postgresql-only-storage-v1/reports/S09"
LIMITS = {
    "latest": ("p95_s", 0.1),
    "hot": ("p95_s", 0.1),
    "catalog": ("p95_s", 0.1),
    "gap": ("p95_s", 0.1),
    "stream": ("p95_s", 0.1),
    "history": ("p95_s", 0.25),
    "year": ("median_s", 2),
    "full": ("p95_s", 10),
}


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(name):
    summary = json.loads((OUT / f"{name}-summary.json").read_text())
    checks = {
        lane: {
            "metric": metric,
            "limit_s": limit,
            "actual_s": summary["lanes"][lane][metric],
            "pass": summary["lanes"][lane][metric] <= limit,
        }
        for lane, (metric, limit) in LIMITS.items()
    }
    monitor = lines(OUT / f"{name}-pg-resources.jsonl")
    server = lines(OUT / f"{name}-server-resources.jsonl")
    samples = lines(OUT / f"{name}-samples.jsonl")
    whole_driver = {
        "pg_first_utc": monitor[0]["utc"],
        "pg_last_utc": monitor[-1]["utc"],
        "server_first_utc": server[0]["utc"],
        "server_last_utc": server[-1]["utc"],
    }
    window_path = OUT / f"{name}-window.json"
    window = json.loads(window_path.read_text()) if window_path.exists() else None
    if window:
        monitor = [r for r in monitor if window["start_utc"] <= r["utc"] <= window["end_utc"]]
        server = [r for r in server if window["start_utc"] <= r["utc"] <= window["end_utc"]]
    beginning, end = monitor[0], monitor[-1]
    pg_delta = {
        group: {
            key: float(end["pg"][group][key]) - float(beginning["pg"][group][key])
            for key in beginning["pg"][group]
            if isinstance(beginning["pg"][group][key], (int, float))
            and isinstance(end["pg"][group][key], (int, float))
        }
        for group in ("checkpoint", "wal", "database")
    }

    def kv(value):
        return {
            line.split()[0]: int(line.split()[1])
            for line in value.splitlines()
            if len(line.split()) == 2
        }

    def memory(row):
        raw = row["memory"]["data"].splitlines()
        return {"current": int(raw[0]), "lifetime_peak": int(raw[1]), **kv("\n".join(raw[2:]))}

    server_mem = [memory(row) for row in server]
    elapsed = server[-1]["utc"] - server[0]["utc"]
    cpu_delta = (
        kv(server[-1]["cpu"]["data"])["usage_usec"] - kv(server[0]["cpu"]["data"])["usage_usec"]
    )
    pool = [r["pool_wait_s"] for r in samples if r.get("pool_wait_s") is not None]
    return {
        "name": name,
        "resource_boundary": "samples strictly inside timed window"
        if window
        else "whole driver including setup and verification; pilot diagnostic only",
        "declared_window": window,
        "whole_driver_bounds": whole_driver,
        "pg_sample_bounds_utc": [beginning["utc"], end["utc"]],
        "server_sample_bounds_utc": [server[0]["utc"], server[-1]["utc"]],
        "summary": summary,
        "guardrails": checks,
        "guardrails_pass": all(c["pass"] for c in checks.values()),
        "pool_wait_p95_s": float(np.quantile(pool, 0.95)),
        "pool_wait_max_s": max(pool),
        "server_cpu_seconds": cpu_delta / 1e6,
        "server_mean_cores": cpu_delta / 1e6 / elapsed,
        "server_memory_max_bytes": max(m["current"] for m in server_mem),
        "server_anon_max_bytes": max(m["anon"] for m in server_mem),
        "server_file_max_bytes": max(m["file"] for m in server_mem),
        "server_oom_delta": server_mem[-1]["oom"] - server_mem[0]["oom"],
        "client_cgroup_memory_max_bytes": max(
            int(r["client_cgroup"]["memory.current"]) for r in monitor
        ),
        "client_process_rss_sum_max_bytes": max(
            r["client_cgroup"]["process_rss_sum_bytes"] for r in monitor
        ),
        "pg_deltas": pg_delta,
        "disk_first": beginning["pg"]["disk"],
        "disk_last": end["pg"]["disk"],
        "wal_disk_first": beginning["pg"]["wal_disk"],
        "wal_disk_last": end["pg"]["wal_disk"],
        "tables_first": beginning["pg"]["tables"],
        "tables_last": end["pg"]["tables"],
        "sample_count": len(samples),
        "resource_sample_count": len(monitor),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    for name in args.names:
        result = analyze(name)
        (OUT / f"{name}-analysis.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    "name": name,
                    "guardrails": result["guardrails"],
                    "oom": result["server_oom_delta"],
                }
            )
        )
