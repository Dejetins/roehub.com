"""Counted operator on the exact installed direct PG16 endpoint."""

import json
import os
import sys
from pathlib import Path

from tools.installation.postgres_budget_recovery import Engine, enroll, recover, resume

root = Path("/run/roehub-pg-budget")
engine = Engine("/var/run/docker.sock")
dsn = os.environ["ROEHUB_STORAGE_POSTGRES_DSN"]
name = "roehub-local-postgresql-1"
action = sys.argv[1]
if action == "enroll":
    print(json.dumps(enroll(root, engine, name, dsn)))
elif action == "recover":
    print(json.dumps(recover(root, engine, name)))
elif action == "resume":
    print(json.dumps(resume(root, engine, name, dsn)))
else:
    raise ValueError(action)
