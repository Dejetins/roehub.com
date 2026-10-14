"""Reproducible S04 representative PG16 query plans/parity, not an S09 soak."""
import hashlib
import json
import os
import resource
import threading
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import subprocess

def rss_bytes():
    return int(subprocess.check_output(["ps","-o","rss=","-p",str(os.getpid())],text=True).strip())*1024

import psycopg

from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_reader import PostgresCanonicalCandleReader
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_index_reader import PostgresCanonicalCandleIndexReader
from trading.contexts.market_data.adapters.outbound.persistence.postgres.catalog_coverage_reader import PostgresCatalogCoverageReader
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_coverage_reader import PostgresHistoryCoverageReader
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_range_verifier import PostgresHistoryRangeVerifier
from trading.contexts.market_data.adapters.outbound.persistence.postgres.instrument_coverage_reader import PostgresInstrumentCoverageReader
from trading.contexts.market_data.adapters.outbound.persistence.postgres.reference_readers import PostgresEnabledMarketReader,PostgresEnabledInstrumentReader,PostgresEnabledTradableInstrumentSearchReader,PostgresBTCUSDTMarketReadinessReader
from trading.contexts.market_data.adapters.outbound.persistence.postgres.funding_rate_store import PostgresFundingRateStore
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import IngestionGateway
from trading.contexts.market_data.adapters.outbound.persistence.postgres.coverage_maintenance import PostgresCoverageMaintenance
from trading.contexts.market_data.adapters.outbound.persistence.postgres.reference_writers import PostgresMarketRefWriter,PostgresInstrumentRefWriter
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.stored_coordinates import PostgresStoredArtifactCoordinates
from trading.shared_kernel.primitives import InstrumentId,MarketId,Symbol,UtcTimestamp,TimeRange

OUT=Path(__file__).parent
DSN=os.environ['ROEHUB_S04_DISPOSABLE_DSN']
START=datetime(2024,1,1,tzinfo=UTC)
END=datetime(2026,1,1,tzinfo=UTC)
I=InstrumentId(MarketId(1),Symbol('DENSE'))
R=TimeRange(UtcTimestamp(START),UtcTimestamp(END))


def seed():
    with psycopg.connect(DSN) as c:
        assert c.info.host=='127.0.0.1' and 160000<=c.info.server_version<170000
        # Fixture exclusively in the separately labeled S04 container/database.
        for year in (2024,2025):
            c.execute('SELECT market_data.ensure_partitions(%s,12)',(date(year,1,1),))
        c.execute("INSERT INTO market_data.ref_market(market_id,exchange_name,market_type,market_code) VALUES (1,'binance','spot','binance:spot'),(2,'binance','futures','binance:futures'),(3,'bybit','spot','bybit:spot'),(4,'bybit','futures','bybit:futures') ON CONFLICT DO NOTHING")
        c.execute("INSERT INTO market_data.ref_instruments(market_id,symbol,status,is_tradable) SELECT 1,'KEY'||lpad(n::text,5,'0'),'ENABLED',1 FROM generate_series(1,10000)n ON CONFLICT DO NOTHING")
        c.execute("INSERT INTO market_data.ref_instruments(market_id,symbol,status,is_tradable) VALUES (1,'DENSE','ENABLED',1),(1,'BTCUSDT','ENABLED',1) ON CONFLICT DO NOTHING")
        # One dense two-year history, 48 representative monthly siblings, one sparse long history.
        c.execute("""INSERT INTO market_data.canonical_candles_1m
         (market_id,symbol,instrument_key,ts_open,ts_close,open,high,low,close,volume_base,
          volume_quote,trades_count,taker_buy_volume_base,taker_buy_volume_quote,source,ingested_at,ingest_id)
         SELECT 1,symbol,'binance:spot:'||symbol,t,t+interval '1 minute',10,20,1,11,2,
                NULL,NULL,NULL,NULL,'rest',t+interval '2 days',NULL
         FROM (
          SELECT 'DENSE' symbol,t FROM generate_series(%s::timestamptz,%s::timestamptz-interval '1 minute',interval '1 minute')t
          UNION ALL SELECT 'SIB'||n,t FROM generate_series(1,48)n CROSS JOIN
          generate_series('2025-01-01'::timestamptz,'2025-02-01'::timestamptz-interval '1 minute',interval '1 minute')t
          UNION ALL SELECT 'SPARSE',t FROM generate_series('2024-01-01'::timestamptz,'2025-12-31'::timestamptz,interval '13 days')t
         ) q ON CONFLICT DO NOTHING""",(START,END))
        c.execute("INSERT INTO market_data.instrument_generations(market_id,symbol,index_state,content_revision) SELECT DISTINCT market_id,symbol,'ready',1 FROM market_data.canonical_candles_1m ON CONFLICT DO NOTHING")
        c.execute("""INSERT INTO market_data.candle_coverage_days(market_id,symbol,day,generation,actual,policy_version)
        SELECT c.market_id,c.symbol,ts_open::date,g.generation,
        bit_or(set_bit(0::bit(1440),(extract(hour FROM ts_open)*60+extract(minute FROM ts_open))::int,1)),1
        FROM market_data.canonical_candles_1m c JOIN market_data.instrument_generations g USING(market_id,symbol)
        GROUP BY c.market_id,c.symbol,ts_open::date,g.generation ON CONFLICT DO NOTHING""")
        c.execute("""INSERT INTO market_data.candle_coverage_summary(market_id,symbol,generation,actual_count,first_ts,last_ts)
        SELECT c.market_id,c.symbol,g.generation,count(*),min(ts_open),max(ts_open)
        FROM market_data.canonical_candles_1m c JOIN market_data.instrument_generations g USING(market_id,symbol)
        GROUP BY c.market_id,c.symbol,g.generation ON CONFLICT DO NOTHING""")
        c.execute("""INSERT INTO market_data.canonical_funding_rates(market_id,symbol,instrument_key,funding_time,funding_rate,funding_interval_minutes,funding_interval_source,source,ingested_at)
        SELECT 2,'DENSE','binance:futures:DENSE',t,0.01,60,'provider','fixture',t
        FROM generate_series(%s::timestamptz,%s::timestamptz-interval '1 hour',interval '1 hour')t ON CONFLICT DO NOTHING""",(START,END))
        c.execute("""INSERT INTO market_data.funding_instrument_universe(market_id,symbol,instrument_key,exchange,market_type,status,is_tradable,funding_interval_minutes)
        VALUES (2,'DENSE','binance:futures:DENSE','binance','futures','ENABLED',1,60) ON CONFLICT DO NOTHING""")
        c.execute("UPDATE market_data.canonical_candles_1m SET source='rest' WHERE source='fixture'")
        c.execute("UPDATE market_data.instrument_generations SET index_state='ready',content_revision=1 WHERE symbol='SPARSE'")
        c.execute('ANALYZE')
        result=c.execute('SELECT count(*),count(DISTINCT symbol),min(ts_open),max(ts_open) FROM market_data.canonical_candles_1m').fetchone()
        settings=dict(c.execute("SELECT name,setting FROM pg_settings WHERE name IN ('server_version','max_connections','shared_buffers','work_mem','maintenance_work_mem','fsync','synchronous_commit','autovacuum')"))
    (OUT/'representative-fixture.json').write_text(json.dumps(dict(rows=result[0],instruments=result[1],first=str(result[2]),last=str(result[3]),reference_rows=10002,settings=settings,seed_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))


class Cursor:
    def __init__(self,capture,cursor): self.capture,self.cursor=capture,cursor
    def __enter__(self): self.cursor.__enter__();return self
    def __exit__(self,*args): return self.cursor.__exit__(*args)
    def execute(self,q,p=(),**kw):
        self.capture.record(q,p)
        self.cursor.execute(q,p,**kw)
        return self
    def copy(self,q,p=()):
        if q.startswith('COPY ('):
            self.capture.record(q[6:q.rindex(') TO STDOUT')],p)
        return self.cursor.copy(q,p)
    def __getattr__(self,name): return getattr(self.cursor,name)


class Connection:
    def __init__(self,g,c): self.g,self.c=g,c
    def record(self,q,p):
        if hasattr(q,'as_string'): q=q.as_string(self.c)
        if isinstance(q,str) and q.lstrip().upper().startswith(('SELECT','WITH')) and 'invalidate_instrument' not in q:
            self.g.queries.append((self.g.family,q,p))
    def execute(self,q,p=(),**kw):
        self.record(q,p)
        return self.c.execute(q,p,**kw)
    def cursor(self,*args,**kw): return Cursor(self,self.c.cursor(*args,**kw))


class Capture:
    def __init__(self):
        self.gateways={k:IngestionGateway(DSN,workload=k,deadline_s=v) for k,v in [('bulk',30),('interactive',1),('reconcile',10),('reference',5)]}
        self.queries=[]
        self.family=''
    def run(self,fn,**kw):
        k=('bulk' if self.family.startswith(('Q02','Q15-range')) else
           'reconcile' if self.family.startswith(('Q07','Q08','Q09','Q16')) else
           'reference' if self.family.startswith(('Q10','Q11','Q13')) else 'interactive')
        return self.gateways[k].run(lambda c:fn(Connection(self,c)),**kw)


def probe():
    g=Capture()
    results=[]
    def run(family,call,check):
        g.family=family
        rss=[];stop=threading.Event()
        def sample():
            while not stop.wait(.05): rss.append(rss_bytes())
        before=rss_bytes()
        sampler=threading.Thread(target=sample);sampler.start()
        t=time.perf_counter();cpu=time.process_time()
        try:
            result=call();check(result)
            results.append(dict(family=family,elapsed_s=time.perf_counter()-t,cpu_s=time.process_time()-cpu,
                rss_before=before,rss_peak=max(rss or [before]),parity='pass',result_sha256=hashlib.sha256(repr(result).encode()).hexdigest()))
            (OUT/'query-results-progress.json').write_text(json.dumps(results,indent=2))
        finally: stop.set();sampler.join()
        return result
    def equal(expected):
        def check(x): assert x==expected,(x,expected)
        return check
    n=int((END-START).total_seconds()/60)
    reader=PostgresCanonicalCandleReader(g)
    def arrays_check(x):
        assert x.row_count()==n and isinstance(x.ohlcv_f32,np.memmap)
        # Bounded independent expected vectors, every returned value and timestamp checked.
        for a in range(0,n,4096):
            b=min(n,a+4096)
            assert np.array_equal(x.open_time_ms[a:b],int(START.timestamp()*1000)+np.arange(a,b)*60000)
            assert np.array_equal(x.close_time_ms[a:b],x.open_time_ms[a:b]+60000)
            assert np.all(x.ohlcv_f32[a:b]==np.array([10,20,1,11,2],dtype=np.float32))
    run('Q02-arrays-2year',lambda:reader.read_1m_arrays(I,R),arrays_check)
    def dto():
        count=0
        for r in reader.read_1m(I,R):
            assert r.candle.ts_open.value==START+timedelta(minutes=count)
            assert (r.candle.open,r.candle.high,r.candle.low,r.candle.close,r.candle.volume_base,r.candle.volume_quote)==(10,20,1,11,2,None)
            assert r.candle.instrument_id==I and r.candle.ts_close.value==r.candle.ts_open.value+timedelta(minutes=1)
            assert (r.meta.source,r.meta.instrument_key,r.meta.ingest_id,r.meta.trades_count,r.meta.taker_buy_volume_base,r.meta.taker_buy_volume_quote)==('rest','binance:spot:DENSE',None,None,None,None)
            assert r.meta.ingested_at.value==r.candle.ts_open.value+timedelta(days=2)
            count+=1
        return count
    run('Q02-dto-2year',dto,equal(n))
    idx=PostgresCanonicalCandleIndexReader(g)
    run('Q03-bounds',lambda:idx.bounds(I),equal((UtcTimestamp(START),UtcTimestamp(END-timedelta(minutes=1)))))
    run('Q03-before-partial',lambda:idx.bounds_1m(instrument_id=I,before=UtcTimestamp(START+timedelta(days=366,seconds=30))),equal((UtcTimestamp(START),UtcTimestamp(START+timedelta(days=366)))))
    run('Q03-tail',lambda:reader.latest(instrument_id=I,before=UtcTimestamp(END),limit=100),lambda x:equal([END-timedelta(minutes=m) for m in range(100,0,-1)])([v['ts_open'] for v in x]))
    run('Q04-daily',lambda:idx.daily_counts(instrument_id=I,time_range=R),lambda x:equal(n)(sum(v.count for v in x)))
    week=TimeRange(UtcTimestamp(END-timedelta(days=7)),UtcTimestamp(END))
    run('Q04-distinct',lambda:idx.distinct_ts_opens(instrument_id=I,time_range=week),lambda x:equal(10080)(len(x)))
    cat=PostgresCatalogCoverageReader(g)
    run('Q05-catalog',lambda:cat.counts(market_ids=[1],start_at=week.start.value,end_at=END),lambda x:equal(10080)(x[(1,'DENSE')]))
    run('Q06-instrument',lambda:PostgresInstrumentCoverageReader(g).read(instrument_id=I,expected_start_at=START,expected_end_at=END),lambda x:equal(('complete',100.))((x.state,x.percent)))
    cov=PostgresHistoryCoverageReader(g)
    run('Q07-gap-dense-2year',lambda:cov.first_gap(instrument_id=I,time_range=R),equal(None))
    sparse=InstrumentId(MarketId(1),Symbol('SPARSE'))
    run('Q07-gap-sparse-2year',lambda:cov.first_gap(instrument_id=sparse,time_range=R),lambda x:equal(START+timedelta(minutes=1))(x.start.value))
    run('Q07-absence-count',lambda:cov.source_absent_counts(starts={(1,'DENSE'):START},end_at=END),equal({(1,'DENSE'):0}))
    verifier=PostgresHistoryRangeVerifier(g)
    g.family='Q08-lease'
    fence=verifier.acquire(instrument_id=I)
    run('Q08-verify-week',lambda:verifier.confirm(instrument_id=I,time_range=week,observed_minutes=set(),observed_at=END,accept_absence=False,check_lease=lambda:None,fence=fence,origin=START),equal(None))
    verifier.release(instrument_id=I,fence=fence)
    stored=PostgresStoredArtifactCoordinates(g)
    def observation_check(items):
        assert len(items)==50
        cutoff=START+timedelta(days=400,seconds=30)
        with psycopg.connect(DSN) as independent:
            expected=independent.execute("SELECT symbol,min(ts_open),max(ts_open),count(*) FROM market_data.canonical_candles_1m WHERE ts_open<%s GROUP BY symbol ORDER BY symbol",(cutoff,)).fetchall()
        assert [(str(v.instrument.symbol),v.first.value if v.first else None,v.last.value if v.last else None,v.actual_count) for v in items]==expected
        assert all(v.index_state=='ready' and v.generation is not None and v.content_revision==1 for v in items)
    run('Q09-observation-page',lambda:stored.observation_page(after=None,before=UtcTimestamp(START+timedelta(days=400,seconds=30)),limit=50),observation_check)
    run('Q10-market-ids',lambda:PostgresMarketRefWriter(g).existing_market_ids([MarketId(x) for x in (1,2,3,4)]),equal({1,2,3,4}))
    refs=PostgresInstrumentRefWriter(g)
    run('Q11-reference-batch',lambda:refs.existing_latest_enrichment(market_id=MarketId(1),symbols=[Symbol('KEY'+str(x).zfill(5)) for x in range(1,1001)]),lambda x:equal({('KEY'+str(n).zfill(5),'ENABLED',1,None,None,None,None,None) for n in range(1,1001)})({(k,v.status,v.is_tradable,v.base_asset,v.quote_asset,v.price_step,v.qty_step,v.min_notional) for k,v in x.items()}))
    run('Q12-markets',lambda:PostgresEnabledMarketReader(g).list_enabled_markets(),lambda x:equal(4)(len(x)))
    run('Q12-enabled',lambda:PostgresEnabledInstrumentReader(g).list_enabled_tradable(),lambda x:equal(10002)(len(x)))
    run('Q12-search',lambda:PostgresEnabledTradableInstrumentSearchReader(g).search_enabled_tradable_by_market(market_id=MarketId(1),symbol_prefix='KEY09',limit=100),lambda x:equal(['KEY'+str(n).zfill(5) for n in range(9000,9100)])([str(i.symbol) for i in x]))
    run('Q12-readiness',lambda:PostgresBTCUSDTMarketReadinessReader(g).list_btcusdt_reference_rows(),lambda x:equal({(1,'ENABLED',1),(2,None,None),(3,None,None),(4,None,None)})({(v.market_id.value,v.status,v.is_tradable) for v in x}))
    funding=PostgresFundingRateStore(g)
    fi=InstrumentId(MarketId(2),Symbol('DENSE'))
    run('Q13-universe',lambda:funding.list_tradable_funding_instruments(market_ids=[MarketId(2)]),lambda x:equal(60)(x[0].funding_interval_minutes))
    run('Q13-get',lambda:funding.get_funding_instrument(fi),lambda x:equal((fi,60,None,None,None))((x.instrument_id,x.funding_interval_minutes,x.base_asset,x.funding_cap,x.funding_floor)))
    run('Q15-latest',lambda:funding.latest_funding_time(fi),equal(UtcTimestamp(END-timedelta(hours=1))))
    def funding_check(x):
        assert x.status=='ready' and len(x.records)==n//60
        for ix,row in enumerate(x.records):
            assert row.funding_time.value==START+timedelta(hours=ix)
            assert (row.funding_rate,row.funding_interval_minutes,row.mark_price,row.data_quality)==(.01,60,None,1)
    run('Q15-range-2year',lambda:funding.read_funding_coverage(instrument_id=fi,time_range=R),funding_check)
    # Explicit rebuild uses a different sparse key; one step is bounded to seven days.
    g.family='Q16-rebuild'
    prior=stored.observation(instrument=sparse,before=UtcTimestamp(END))
    maintenance=PostgresCoverageMaintenance(g)
    new_generation=maintenance.begin(instrument_id=sparse,expected_generation=prior.generation,reason='representative-rebuild')
    run('Q16-rebuild-step',lambda:maintenance.step(instrument_id=sparse,generation=new_generation),equal(False))
    plans=[]
    with psycopg.connect(DSN) as c:
        for family,q,p in g.queries:
            with c.transaction():
                c.execute("SET LOCAL statement_timeout='30s'")
                plan=c.execute('EXPLAIN (ANALYZE,BUFFERS,FORMAT JSON) '+q,p).fetchone()[0]
                plans.append(dict(family=family,sql=q,parameters=p,plan=plan))
            c.rollback()
    for gateway in g.gateways.values(): gateway.close()
    (OUT/'query-plans.json').write_text(json.dumps(plans,indent=2,default=str))
    (OUT/'query-results.json').write_text(json.dumps(results,indent=2,default=str))
    print(json.dumps(dict(operations=len(results),plans=len(plans),parity='pass')))


if __name__=='__main__':
    seed()
    probe()
