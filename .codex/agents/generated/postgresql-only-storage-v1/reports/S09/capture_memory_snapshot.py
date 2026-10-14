"""Low-frequency in-soak context/RSS evidence; no backend reset or settings change."""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

out = Path(__file__).parent
label = sys.argv[1]
start = time.time()
code = """import json,time
from tools.qa.postgres_production_soak import DSN
from trading.platform.postgres_budget import connect
with connect(DSN,budget_class="admin",autocommit=True) as c:
 for attempt in range(60):
  rows=c.execute("SELECT a.pid,a.application_name,a.state,a.backend_start,l.objid "
   "FROM pg_stat_activity a LEFT JOIN pg_locks l ON l.pid=a.pid AND "
   "l.locktype='advisory' AND l.classid=563298 WHERE a.datname=current_database() "
   "AND a.backend_type='client backend' AND a.pid<>pg_backend_pid()").fetchall()
  bulk=[r for r in rows if r[4] in (10,11)]
  if len(bulk)==2 and all(r[2]=='idle' for r in bulk):break
  time.sleep(.2)
 for row in rows:c.execute("SELECT pg_log_backend_memory_contexts(%s)",(row[0],))
 print(json.dumps({"utc":time.time(),"backends":rows,"idle_poll_attempts":attempt},default=str))
"""
result = json.loads(
    subprocess.check_output(
        ["docker", "exec", "roehub-s09-client", "/opt/roehub/.venv/bin/python", "-c", code],
        text=True,
    )
)
status = subprocess.check_output(
    [
        "docker",
        "exec",
        "roehub-s09-pg",
        "sh",
        "-c",
        'for p in /proc/[0-9]*; do test -r "$p/status" && '
        'awk "/^(Name|Pid|RssAnon|RssFile|RssShmem):/" "$p/status"; done',
    ],
    text=True,
)
processes = []
current = {}
for line in status.splitlines():
    key, value = line.split(":", 1)
    if key == "Name":
        if current:
            processes.append(current)
        current = {}
    current[key] = value.strip()
processes.append(current)
# Give the asynchronous server log request time to finish; no measured samples are removed.
time.sleep(1)
logs = subprocess.run(
    ["docker", "logs", "--since", str(int(start)), "roehub-s09-pg"],
    capture_output=True,
    text=True,
    check=True,
)
contexts = []
totals = []
for line in (logs.stdout + logs.stderr).splitlines():
    pid = re.search(r"\[(\d+)\]", line)
    context = re.search(
        r"level: (\d+); (.*?): (\d+) total in (\d+) blocks; (\d+) free.*; (\d+) used", line
    )
    grand = re.search(r"Grand total: (\d+) bytes in (\d+) blocks; (\d+) free .*; (\d+) used", line)
    if pid and context:
        contexts.append(
            {
                "pid": int(pid[1]),
                "name": context[2],
                "level": int(context[1]),
                "total": int(context[3]),
                "used": int(context[6]),
            }
        )
    if pid and grand:
        totals.append({"pid": int(pid[1]), "total": int(grand[1]), "used": int(grand[4])})
result.update(
    start_utc=start,
    end_utc=time.time(),
    processes=processes,
    contexts=contexts,
    grand_totals=totals,
    workload_progress=json.loads((out / "r4-soak1-progress.json").read_text()),
)
assert len(totals) >= len(result["backends"]), (len(totals), len(result["backends"]))
(out / f"r4-soak-memory-{label}.json").write_text(json.dumps(result, indent=2) + "\n")
print(label, len(totals), "backend context snapshots saved")
