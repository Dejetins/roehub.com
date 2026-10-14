"""Real partition DDL under maintenance role; temporary bounded request projection."""
import json,os,uuid
from datetime import datetime,UTC,timedelta
from trading.platform.postgres_budget import connect
from apps.migrations.market_data import provision_requested_partitions
from apps.migrations.storage import provision_runtime_roles
from apps.monitoring import postgres_observer
D='host=s06-pg dbname=roehub user=postgres'
provision_runtime_roles(D)
class RollbackProof(Exception):pass
try:
 with connect(D,budget_class='admin') as c:
  c.execute('CREATE TEMP TABLE market_data_work_requests(job_id uuid,start_at timestamptz,end_at timestamptz,kind text,state text)')
  c.execute('GRANT SELECT ON market_data_work_requests TO roehub_maintenance')
  now=datetime.now(UTC).replace(day=2,hour=0,minute=0,second=0,microsecond=0)
  for i in range(65):
   start=now if i<64 else datetime(1970,1,2,tzinfo=UTC)
   c.execute("INSERT INTO market_data_work_requests VALUES(%s,%s,%s,'candle_ingestion','queued')",(uuid.UUID(int=i+1),start,start+timedelta(days=1)))
  c.execute('SET LOCAL ROLE roehub_maintenance')
  count,cursor=provision_requested_partitions(c)
  assert count==0,count
  count,cursor=provision_requested_partitions(c,after_job_id=cursor)
  assert count==1,count
  c.execute('RESET ROLE');c.execute('TRUNCATE market_data_work_requests')
  for i in range(13):
   start=datetime(1971+i//12,i%12+1,2,tzinfo=UTC)
   c.execute("INSERT INTO market_data_work_requests VALUES(%s,%s,%s,'candle_ingestion','queued')",(uuid.UUID(int=i+1),start,start+timedelta(days=1)))
  c.execute('SET LOCAL ROLE roehub_maintenance')
  count,cursor=provision_requested_partitions(c)
  assert count==12,count
  count,cursor=provision_requested_partitions(c,after_job_id=cursor)
  assert count==1,count
  partitions=c.execute("SELECT count(*) FROM pg_inherits JOIN pg_class ON oid=inhrelid WHERE relkind='r' AND inhrelid::regclass::text LIKE 'market_data.%_197%' ").fetchone()[0]
  assert partitions==84,partitions
  raise RollbackProof
except RollbackProof:pass
os.environ['ROEHUB_PG_MONITOR_DSN']='host=s06-pg dbname=roehub user=roehub_monitor'
os.environ['ROEHUB_PG_MAINTENANCE_DSN']='host=s06-pg dbname=roehub user=roehub_maintenance'
assert postgres_observer.sample()
metrics=postgres_observer.metrics_text()
for marker in ['reservations_active','connections','lock_waiters','long_transactions','wal_bytes_total','checkpoints_timed_total','dead_tuples','autovacuums_total']:
 assert 'roehub_postgres_'+marker in metrics,marker
for connection in (postgres_observer._connection,postgres_observer._maintenance):connection.close()
print(json.dumps({'maintenance_role_ddl':'passed','requests_past_64_not_starved':True,'max_missing_months_per_tick':12,'six_partitioned_families_verified':True,'fixture_ddl_rolled_back':True,'observer_metrics':'passed','scope':'real PG and maintenance grants; temp bounded work-request projection'}))
