import json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
client='roehub-s10r1-lease-client';pg='roehub-s10r1-test-pg'
def query():
 return subprocess.check_output(['docker','exec',pg,'psql','-U','postgres','-Atc',"SELECT count(*) FROM pg_locks WHERE locktype='advisory' AND classid=173 AND objid=1919003420 AND granted"],text=True).strip()
x=json.loads(subprocess.check_output(['docker','inspect',client]))[0]
assert x['State']['Running']
assert query()=='1'
r={'started_at':datetime.now(timezone.utc).isoformat(),'client':x['Id'],'scope':'disposable client and PostgreSQL only','healthy_idle_lease_present':True,'client_settings':subprocess.check_output(['docker','logs',client],text=True).strip()}
t=time.monotonic();subprocess.run(['docker','network','disconnect','roehub-local-platform',x['Id']],check=True)
try:
 while query()=='1' and time.monotonic()-t<60:time.sleep(1)
 r['lease_released']=query()=='0';r['elapsed_seconds']=round(time.monotonic()-t,3)
finally:
 subprocess.run(['docker','network','connect','roehub-local-platform',x['Id']],check=True)
Path(__file__).with_name('lease-drop-proof.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r));assert r['lease_released']
