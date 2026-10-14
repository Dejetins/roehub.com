"""Disposable Linux multi-process envelope proof; credentials never written to output."""
import multiprocessing as mp
import os
import time
import json
from trading.platform.postgres_budget import connect, reap, BudgetExhausted

DSN=os.environ.get('S06_DSN','host=s06-pg dbname=roehub user=postgres')
from psycopg.conninfo import make_conninfo
APP_DSN=make_conninfo(DSN,user='roehub_app',passfile='/run/roehub-secrets/pgpass')

def worker(kind, ready, finish):
    dsn=DSN if kind=='admin' else APP_DSN
    c=connect(dsn,budget_class=kind,autocommit=True)
    ready.put((kind,c.info.backend_pid))
    while not os.path.exists(finish):
        time.sleep(.05)
    c.close()

if __name__=='__main__':
    mp.set_start_method('spawn')
    observer=connect(DSN,budget_class='admin',reap_closed=True,autocommit=True)
    for _ in range(20):
        c=connect(DSN,budget_class='admin',reap_closed=True,autocommit=True);c.close()
    for _ in range(20):reap(observer);time.sleep(.05)
    ready=mp.Queue();finish="/tmp/s06-finish";ps=[]
    for kind,count in [('market',16),('control',24),('admin',7)]:
        for i in range(count):
            p=mp.Process(target=worker,args=(kind,ready,finish));p.start();ps.append(p)
    results=[ready.get(timeout=20) for _ in ps]
    actual=observer.execute("SELECT count(*) FROM pg_stat_activity WHERE backend_type='client backend'").fetchone()[0]
    assert actual==48,actual
    for kind in ('market','control','admin'):
        started=time.monotonic()
        try:connect(DSN,budget_class=kind)
        except BudgetExhausted:pass
        else:raise AssertionError('budget bypass')
        assert time.monotonic()-started<2.5
    observer.execute('CREATE TEMP TABLE headroom_proof(n int)')
    observer.execute('INSERT INTO headroom_proof VALUES(1)')
    assert observer.execute('SELECT count(*) FROM headroom_proof').fetchone()[0]==1
    # Kill one owner: loss of transport proof must retain the slot.
    ps[0].kill();ps[0].join()
    for _ in range(50):
        if reap(observer):break
        time.sleep(.05)
    try: connect(APP_DSN,budget_class="market")
    except BudgetExhausted: pass
    else: raise AssertionError("unknown crash reservation reused")
    open(finish,"w").close()
    for p in ps[1:]:p.join(10);assert p.exitcode==0
    for _ in range(10):reap(observer);time.sleep(.02)
    print(json.dumps({'physical_peak':actual,'market':16,'control':24,'admin':8,
        'multi_process':True,'sequential_admin_connections':20,'overflow_bounded':True,
        'admin_dml_at_saturation':True,'killed_owner_quarantined':True,'unknown_crash_recovery':'not_proven'}))
    observer.close()
