"""Narrow paired warm writer cost; disposable data, never the S09 fixture."""
import hashlib,json,subprocess,time,statistics,sys,types
from pathlib import Path
from uuid import uuid4
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from apps.migrations.storage import apply_postgres_migrations
from tests.real_infra.market_data.test_postgres_readers_frontier import candle
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import IngestionGateway
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import PostgresRawKlineWriter
root=Path(__file__).parent;repo=Path.cwd();baseline=(root/'writer-s09-baseline.py').read_bytes()
assert hashlib.sha256(baseline).hexdigest()=='8b4a97d2e8f439ad44da0e5f44e21038805ae2f43edea4c535749edc539ca328'
mod=types.ModuleType('trading.contexts.market_data.adapters.outbound.persistence.postgres.s09_baseline');mod.__package__=mod.__name__.rsplit('.',1)[0];exec(compile(baseline,'S09-baseline','exec'),mod.__dict__)
x=json.loads(subprocess.check_output(['docker','inspect','roehub-s10r1-test-pg']))[0];port=x['NetworkSettings']['Ports']['5432/tcp'][0]['HostPort'];admin=f'host=127.0.0.1 port={port} user=postgres dbname=postgres';name='s10cost_'+uuid4().hex
with psycopg.connect(admin,autocommit=True) as c:c.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
dsn=make_conninfo(admin,dbname=name);observations=[]
try:
 apply_postgres_migrations(dsn,repo_root=repo,manifest_path=repo/'migrations/postgres/manifest.json')
 with psycopg.connect(dsn) as c:c.execute("SELECT market_data.ensure_partitions('2026-02-01',3)")
 for amount in (5,100,1000):
  for repetition in range(3):
   order=('baseline','candidate') if repetition%2==0 else ('candidate','baseline')
   for label in order:
    gateway=IngestionGateway(dsn);writer=(mod.PostgresRawKlineWriter if label=='baseline' else PostgresRawKlineWriter)(gateway);symbol=f'PERF{amount}R{repetition}{label.upper()}'
    try:
     for batch in range(35):
      rows=[candle(symbol,batch*amount+i) for i in range(amount)]
      start=time.perf_counter();writer.write_1m(rows);elapsed=(time.perf_counter()-start)*1000
      if batch>=5:observations.append({'rows':amount,'repetition':repetition,'variant':label,'ms':elapsed})
    finally:gateway.close()
 summaries=[]
 for amount in (5,100,1000):
  for label in ('baseline','candidate'):
   values=sorted(o['ms'] for o in observations if o['rows']==amount and o['variant']==label)
   summaries.append({'rows':amount,'variant':label,'n':len(values),'median_ms':statistics.median(values),'p95_ms':values[int(.95*(len(values)-1))],'min_ms':min(values),'max_ms':max(values)})
 result={'scope':'isolated paired writer component; not an S09 mixed soak or installed SLA','server':'disposable PG16 2CPU 4GiB','client':'same host venv sequential alternating order','warmup_batches':5,'repetitions':3,'samples_per_repetition':30,'baseline_sha256':hashlib.sha256(baseline).hexdigest(),'candidate_sha256':hashlib.sha256((repo/'src/trading/contexts/market_data/adapters/outbound/persistence/postgres/raw_kline_writer.py').read_bytes()).hexdigest(),'summaries':summaries,'observations':observations}
 (root/'append-cost.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summaries))
finally:
 with psycopg.connect(admin,autocommit=True) as c:c.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))
