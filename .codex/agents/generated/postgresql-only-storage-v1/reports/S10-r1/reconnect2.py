"""One bounded network interruption of the exact installed WS worker only."""
import json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
name='roehub-local-market-data-ws-1';out=Path(__file__).parent/'reconnect2.json'
x=json.loads(subprocess.check_output(['docker','inspect',name]))[0];networks=x['NetworkSettings']['Networks'];assert set(networks)=={'roehub-local-platform'}
net=networks['roehub-local-platform'];assert x['State']['Running']
record={'container':name,'id':x['Id'],'network':'roehub-local-platform','original_ip':net['IPAddress'],'aliases':net['Aliases'],'duration_requested_seconds':45,'scope':'only WS worker network; no host/tunnel/provider/selection changes'}
def stamp():return datetime.now(timezone.utc).isoformat()
record['disconnected_at']=stamp();out.write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(['docker','network','disconnect','roehub-local-platform',x['Id']],check=True)
try:time.sleep(45)
finally:
 args=['docker','network','connect','--ip',net['IPAddress']]
 for alias in net['Aliases'] or []:args+=['--alias',alias]
 subprocess.run(args+['roehub-local-platform',x['Id']],check=True)
 record['reconnected_at']=stamp();after=json.loads(subprocess.check_output(['docker','inspect',x['Id']]))[0]
 record['same_container']=after['Id']==x['Id'];record['restored_ip']=after['NetworkSettings']['Networks']['roehub-local-platform']['IPAddress'];record['restored_aliases']=after['NetworkSettings']['Networks']['roehub-local-platform']['Aliases'];out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
