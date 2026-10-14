"""Retire verified prepublication failure after byte-exact slot archive and quiesce."""

import json
import os

from trading.platform.postgres_budget import connect

with connect(os.environ["ROEHUB_STORAGE_POSTGRES_DSN"], budget_class="admin") as db:
    db.execute("SELECT pg_advisory_xact_lock(290029)")
    assert db.execute(
        "SELECT count(*) FROM pg_stat_activity WHERE backend_type='client backend' AND pid<>pg_backend_pid()"  # noqa: E501
    ).fetchone() == (0,)
    assert db.execute("SELECT count(*) FROM backtest_artifact_slot_readers").fetchone() == (0,)
    assert db.execute(
        "SELECT count(*) FROM backtest_artifact_slot_ownership WHERE state<>'available'"
    ).fetchone() == (0,)
    builds = db.execute(
        "SELECT build_id::text,state FROM backtest_artifact_builds WHERE state IN ('running','quarantined') FOR UPDATE"  # noqa: E501
    ).fetchall()
    assert builds == [("8ca02437-ffcb-4c8b-bfc3-bf898e9365b9", "quarantined")]
    n = db.execute(
        "UPDATE backtest_artifact_slot_ownership SET generation=NULL,manifest_sha256=NULL,updated_at=now() WHERE state='available'"  # noqa: E501
    ).rowcount
    db.execute(
        "UPDATE backtest_artifact_builds SET state='error',reservation_bytes=0 WHERE build_id='8ca02437-ffcb-4c8b-bfc3-bf898e9365b9'"  # noqa: E501
    )
    db.execute(
        "UPDATE backtest_artifact_attempts SET state='error' WHERE build_id='8ca02437-ffcb-4c8b-bfc3-bf898e9365b9'"  # noqa: E501
    )
    requests = db.execute(
        "UPDATE backtest_artifact_requests SET state='error',reason='retained_manifest_precheck_refused',control_version=control_version+1 WHERE request_id IN (SELECT request_id FROM backtest_artifact_build_requests WHERE build_id='8ca02437-ffcb-4c8b-bfc3-bf898e9365b9') RETURNING request_id"  # noqa: E501
    ).fetchall()
    for (request,) in requests:
        db.execute(
            "INSERT INTO backtest_artifact_events(request_id,state,reason,progress) VALUES (%s,'error','retained_manifest_precheck_refused','{}')",  # noqa: E501
            (request,),
        )
    print(
        json.dumps(
            {
                "retired_prepublication_build": builds,
                "terminal_requests": requests,
                "available_slots_metadata_reset": n,
                "writer_epochs_preserved": True,
                "active_readers": 0,
            },
            default=str,
        )
    )
