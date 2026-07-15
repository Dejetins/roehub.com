from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg


def main() -> int:
    bridge = os.environ.get("ROEHUB_CONTROL_AGENT_BRIDGE", "")
    bridge_server = None
    bridge_thread = None
    bridge_socket = None
    if bridge:
        from tests.fixtures.control_agent.socket_bridge import start_tcp_to_unix_bridge

        host, port_text = bridge.rsplit(":", 1)
        bridge_socket = Path("/tmp/roehub-control-agent-bridge.sock")
        bridge_server, bridge_thread = start_tcp_to_unix_bridge(
            host=host,
            port=int(port_text),
            unix_path=bridge_socket,
        )
        os.environ["ROEHUB_CONTROL_AGENT_SOCKET"] = str(bridge_socket)
    try:
        from apps.api.main.app import app

        cursor = int(app.state.control_agent_audit_cursor)
        dsn = os.environ["ROEHUB_STORAGE_POSTGRES_DSN"]
        with psycopg.connect(dsn, autocommit=True) as connection:
            with connection.cursor() as database_cursor:
                database_cursor.execute(
                    "SELECT COUNT(*)::bigint FROM control_operation_audit_events"
                )
                count_row = database_cursor.fetchone()
                database_cursor.execute(
                    "SELECT sequence FROM control_operation_audit_cursor "
                    "WHERE singleton = TRUE"
                )
                cursor_row = database_cursor.fetchone()
        if count_row is None or cursor_row is None:
            raise RuntimeError("API control audit database evidence is missing")
        event_count = int(count_row[0])
        persisted_cursor = int(cursor_row[0])
        if cursor != persisted_cursor or event_count != persisted_cursor or cursor <= 0:
            raise RuntimeError("API control audit reconciliation is incomplete")
        print(
            json.dumps(
                {
                    "schema": "io.roehub.api-control-audit-proof/v1alpha1",
                    "status": "passed",
                    "events": event_count,
                    "cursor": persisted_cursor,
                    "sink": "postgresql",
                    "api_startup_wiring": "passed",
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        if bridge_server is not None and bridge_thread is not None:
            from tests.fixtures.control_agent.socket_bridge import stop_bridge

            stop_bridge(
                bridge_server,
                bridge_thread,
                unix_path=bridge_socket,
            )


if __name__ == "__main__":
    raise SystemExit(main())
