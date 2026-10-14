import asyncio,json
from trading.contexts.market_data.adapters.outbound.persistence.postgres.async_stream_recovery_store import AsyncPostgresStreamRecoveryStore
async def main():
 s=AsyncPostgresStreamRecoveryStore('host=roehub-s10r1-test-pg user=postgres dbname=postgres')
 async with s.lease():
  c=s._conn
  row=await(await c.execute("SELECT pg_backend_pid(), current_setting('tcp_keepalives_idle'),current_setting('tcp_keepalives_interval'),current_setting('tcp_keepalives_count'),current_setting('tcp_user_timeout')")).fetchone()
  print(json.dumps(row),flush=True)
  await asyncio.sleep(120)
asyncio.run(main())
