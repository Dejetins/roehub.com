"""Sanitized server/client cgroup and PG samples for S09; no environment or SQL params."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools.qa.postgres_production_soak import DSN, OUT
from trading.platform.postgres_budget import connect


def cgroup():
    root = Path("/sys/fs/cgroup")
    result = {}
    for name in (
        "cpu.stat",
        "memory.current",
        "memory.peak",
        "memory.stat",
        "memory.events",
        "io.stat",
    ):
        p = root / name
        if p.exists():
            result[name] = p.read_text()
    rss = 0
    cpu = 0
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                stat = (p / "stat").read_text().split()
                rss += int(stat[23]) * 4096
                cpu += int(stat[13]) + int(stat[14])
            except (OSError, ValueError, IndexError):
                pass
    result.update(process_rss_sum_bytes=rss, process_cpu_ticks_sum=cpu)
    return result


def sample(c):
    queries = {
        "checkpoint": "SELECT row_to_json(s) FROM pg_stat_bgwriter s",
        "wal": "SELECT row_to_json(s) FROM pg_stat_wal s",
        "database": (
            "SELECT row_to_json(s) FROM pg_stat_database s WHERE datname=current_da" "tabase()"
        ),
        "tables": """SELECT json_agg(s) FROM (
            SELECT schemaname,regexp_replace(relname,'_[0-9]{6}$','') AS family,
            sum(n_live_tup) AS live,sum(n_dead_tup) AS dead,
            sum(n_tup_ins) AS inserted,sum(n_tup_upd) AS updated,sum(n_tup_del) AS deleted,
            sum(autovacuum_count) AS autovacuum_count,max(last_autovacuum) AS last_autovacuum,
            sum(autoanalyze_count) AS autoanalyze_count FROM pg_stat_user_tables
            WHERE schemaname='market_data' OR relname IN ('market_data_work_requests',
            'market_data_stream_recovery','identity_local_auth_rate_limits') GROUP BY 1,2) s""",
        "sessions": (
            "SELECT json_agg(s) FROM (SELECT usename,application_name,state,wait_ev"
            "ent_type,wait_event,count(*) FROM pg_stat_activity GROUP BY 1,2,3,4,5)"
            " s"
        ),
        "disk": """SELECT json_agg(s) FROM (
            SELECT n.nspname,regexp_replace(c.relname,'_[0-9]{6}$','') AS family,
            sum(pg_relation_size(c.oid)) AS heap_bytes,
            sum(pg_indexes_size(c.oid)) AS index_bytes,
            sum(pg_total_relation_size(c.oid)) AS total_bytes
            FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname IN ('public','market_data') AND c.relkind='r' GROUP BY 1,2) s""",
        "wal_disk": "SELECT sum(size) FROM pg_ls_waldir()",
    }
    result = {k: c.execute(q).fetchone()[0] for k, q in queries.items()}
    result["registry"] = {
        kind: sum(
            p.stat().st_size > 0 for p in Path("/run/roehub-pg-budget").glob(kind + "-*.json")
        )
        for kind in ("market", "control", "admin")
    }
    return result


def monitor(name):
    with (
        connect(DSN, budget_class="admin", autocommit=True) as c,
        (OUT / f"{name}-pg-resources.jsonl").open("x") as f,
    ):
        while True:
            row = {"utc": time.time(), "client_cgroup": cgroup(), "pg": sample(c)}
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            if (OUT / f"{name}-monitor-stop").exists():
                break
            time.sleep(5)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name", nargs="?")
    p.add_argument("--cgroup-only", action="store_true")
    a = p.parse_args()
    if a.cgroup_only:
        print(json.dumps(cgroup()))
    else:
        monitor(a.name)
