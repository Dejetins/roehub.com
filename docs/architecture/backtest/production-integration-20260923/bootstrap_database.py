"""Initialize only an owned disposable PostgreSQL database using canonical migrations."""

import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.meta_path[:] = [x for x in sys.meta_path if "__editable__" not in str(x)]
sys.path[:] = [str(ROOT / "src"), str(ROOT)] + [
    x for x in sys.path if "/Projects/roehub.com" not in x or "/.venv/" in x
]
import psycopg  # noqa: E402

from apps.migrations.bootstrap import run_dev_db_bootstrap  # noqa: E402

DSN = os.environ["S6_DSN"]
assert "@127.0.0.1:" in DSN and DSN.endswith("/s6")
with contextlib.redirect_stdout(io.StringIO()):
    run_dev_db_bootstrap(
        identity_dsn=DSN, postgres_dsn=DSN, migrations_dir=ROOT / "migrations/postgres"
    )
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute(
            (ROOT / "migrations/postgres/0014_research_organization_isolation_v1.sql").read_bytes()
        )
from uuid import UUID  # noqa: E402

with psycopg.connect(DSN) as c:
    c.execute(
        "INSERT INTO identity_users (user_id,paid_level,created_at,last_login_at) VALUES (%s,'pro',now(),now())",  # noqa: E501
        (UUID(int=6001),),
    )
    c.execute(
        "INSERT INTO identity_installations (installation_id,display_name,created_at) VALUES (%s,'S6 disposable',now())",  # noqa: E501
        (UUID(int=6003),),
    )
    c.execute(
        "INSERT INTO identity_organizations (organization_id,installation_id,slug,display_name,created_at) VALUES (%s,%s,'s6-proof','S6 disposable',now())",  # noqa: E501
        (UUID(int=6002), UUID(int=6003)),
    )
    c.execute(
        "INSERT INTO identity_memberships (organization_id,user_id,role,created_at,updated_at) VALUES (%s,%s,'owner',now(),now())",  # noqa: E501
        (UUID(int=6002), UUID(int=6001)),
    )
    version = c.execute("SELECT version()").fetchone()
print(
    json.dumps(
        {
            "status": "pass",
            "schema": "canonical migrations plus research organization isolation",
            "version": version,
        }
    )
)
