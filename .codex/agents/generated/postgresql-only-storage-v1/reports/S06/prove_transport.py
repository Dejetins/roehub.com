"""Direct disposable PostgreSQL transport completion proof; no credential output."""
import asyncio
import json
from pathlib import Path
import psycopg
from trading.platform.postgres_budget import connect, async_connect

root=Path('/run/roehub-pg-budget')
def snapshot():
    return {p.name:p.read_bytes() for kind in ('market','control','admin') for p in root.glob(f'{kind}-*.json')}
before=snapshot()
for i in range(60):
    try:
        connect('host=s06-pg dbname=roehub user=s06_nonexistent',budget_class='admin')
    except psycopg.OperationalError:
        pass
    else:
        raise AssertionError('unexpected successful authentication')
    assert snapshot()==before, ('failure',i)
for i in range(60):
    c=connect('host=s06-pg dbname=roehub user=postgres',budget_class='admin')
    c.close()
    assert snapshot()==before, ('success',i)
async def main():
    for i in range(60):
        c=await async_connect('host=s06-pg dbname=roehub user=postgres')
        await c.close()
        assert snapshot()==before, ('async',i)
asyncio.run(main())
print(json.dumps({'completed_startup_failures':60,'completed_successful_closes':60,'async_closes':60,'registry_unchanged':True,'release_evidence':'raw_peer_EOF','remote_server_filesystem_required':False,'unknown_crash_recovery':'not_proven'}))
