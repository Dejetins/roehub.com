"""Crash/recovery proof against disposable direct PG with no shared server volume."""
import json,multiprocessing as mp,os,socket,time
from pathlib import Path
from tools.installation.postgres_budget_recovery import Engine,recover,resume
from trading.platform.postgres_budget import BudgetExhausted,connect,reserve
ROOT=Path('/run/roehub-pg-budget')
DSN='host=s06-pg dbname=roehub user=postgres'

def startup(kind,ready):
    reservation=reserve(kind,deadline=time.monotonic()+2)
    reservation.endpoint(DSN+' port=5432 hostaddr='+socket.gethostbyname('s06-pg'))
    sock=socket.create_connection(('s06-pg',5432),timeout=3)
    ready.put(True)
    time.sleep(45)
    raise AssertionError('owner was not killed')

if __name__=='__main__':
    mp.set_start_method('spawn')
    engine=Engine('/var/run/docker.sock')
    assert all('pg-budget' not in m['Destination'] for m in engine.inspect('s06-pg')['Mounts'])
    rounds=[]
    for round_number in range(2):
        queue=mp.Queue();workers=[]
        try:
            for kind,count in [('market',16),('control',24),('admin',8)]:
                for _ in range(count):
                    process=mp.Process(target=startup,args=(kind,queue));process.start();workers.append(process)
            for _ in workers:assert queue.get(timeout=20)
            network=engine.execute('s06-pg',['cat','/proc/1/net/tcp','/proc/1/net/tcp6'])
            established=sum(len(c:=line.split())>9 and c[3]=='01' and c[1].endswith(':1538') for line in network.splitlines())
            assert established==48,established
            try:recover(ROOT,engine,'s06-pg')
            except BlockingIOError:pass
            else:raise AssertionError('live owners were not protected')
            assert engine.inspect('s06-pg')['State']['Running']
        finally:
            for process in workers:
                if process.is_alive():process.kill()
                process.join(5)
        records=[p for kind in ('market','control','admin') for p in ROOT.glob(f'{kind}-*.json')]
        assert sum(bool(p.stat().st_size) for p in records)==48
        try:connect(DSN,budget_class='admin')
        except BudgetExhausted:pass
        else:raise AssertionError('exhaustion bypass')
        if round_number == 0:
            original_call = engine.call
            def ambiguous_stop(method,path,body=None,**kwargs):
                result=original_call(method,path,body,**kwargs)
                if '/stop?' in path:raise TimeoutError('simulated lost stop response')
                return result
            engine.call=ambiguous_stop
            try:recover(ROOT,engine,'s06-pg')
            except TimeoutError:pass
            else:raise AssertionError('lost response not simulated')
            finally:engine.call=original_call
            assert not engine.inspect('s06-pg')['State']['Running']
            assert sum(bool(p.stat().st_size) for p in records)==48
        stopped=recover(ROOT,engine,'s06-pg')
        assert stopped['cleared_reservations']==48
        assert all(not p.stat().st_size for p in records)
        try:reserve('admin',deadline=time.monotonic()+.1)
        except BudgetExhausted:pass
        else:raise AssertionError('pending recovery admitted a handshake')
        resumed=resume(ROOT,engine,'s06-pg',DSN)
        for _ in range(60):
            connection=connect(DSN,budget_class='admin');connection.close()
        assert all(not p.stat().st_size for p in records)
        rounds.append({'preauth_tcp_connections':established,'crashed_owners':48,'stopped':stopped,'resumed':resumed,'post_recovery_closes':60})
    print(json.dumps({'rounds':rounds,'server_shared_registry_volume':False,'active_owners_protected':True,'total_unknown_crashes':96,'operator_recovery':'passed','ambiguous_stop_retry':'passed'}))
