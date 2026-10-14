"""Run actual generated services against task-owned embedded/external fixtures."""
import json,os,subprocess,sys,uuid,copy
from pathlib import Path
import yaml
from tools.release.generate_runtime_topology import run as generate
from tools.release.verify_runtime_topology import _inspect_runtime_services,_actual_role_readiness,_actual_storage_readiness
ROOT=Path.cwd()
mode,profile=sys.argv[1:3]
out=ROOT/'.local_artifacts/s06-run'/mode
args=['--write','--output',str(out),'--profile',profile]
if mode=='external':args += ['--config',str(ROOT/'tests/fixtures/installation/roehub-external.yaml')]
assert generate(args)==0
folder=out/profile
compose=yaml.safe_load((folder/'compose.yaml').read_text())
base=yaml.safe_load((ROOT/'configs/installation/generated'/profile/'compose.yaml').read_text())
if mode=='external':
 for name in ('postgresql','redis'):
  compose['services'][name]=copy.deepcopy(base['services'][name])
 for name,value in base['volumes'].items():compose['volumes'].setdefault(name,value)
for name,row in compose['services'].items():
 row.pop('ports',None)
 row['restart']='no'
 row.setdefault('labels',{})['io.roehub.proof']='S06-RUN'
 if row.get('image','').startswith('roehub/runtime'):
  row['image']='roehub/s06-ml-proof:local' if profile=='ml' else 'roehub/s06-proof:local'
  row['entrypoint']=['python','/workspace/infra/docker/runtime-entrypoint.py']
  row['working_dir']='/workspace'
  row.setdefault('environment',{})['PYTHONPATH']='/workspace/src:/workspace'
  row.setdefault('volumes',[]).append(str(ROOT)+':/workspace:ro')
 if name=='openbao':row['image']='sha256:436eaf9778cad75507ff70ea26ace30dcbe15606e619ac3823495663d7f7c115'
 binds=[]
 for volume in row.get('volumes',[]):
  if not isinstance(volume,str):raise AssertionError('unexpected bind')
  source,*rest=volume.split(':')
  if source.startswith('../../../../'):source=str(ROOT/source.removeprefix('../../../../'))
  elif source.startswith('./'):source=str(folder/source)
  binds.append(':'.join([source,*rest]))
 row['volumes']=binds
compose_path=folder/'proof-compose.yaml'
compose_path.write_text(yaml.safe_dump(compose,sort_keys=True))
project='s06-run-'+mode+'-'+profile+'-'+uuid.uuid4().hex[:6]
command=['docker','compose','-p',project,'-f',str(compose_path)]
report=ROOT/'.codex/agents/generated/postgresql-only-storage-v1/reports/S06'
runid=os.environ.get('S06_RUN_ID','1')
log=report/f'run-{mode}-{profile}-{runid}.log'
def run(args,timeout=300):
 result=subprocess.run(command+args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
 with log.open('a') as f:f.write(result.stdout)
 if result.returncode:raise RuntimeError('compose '+str(args)+' failed; see '+str(log))
 return result.stdout
try:
 run(['config','--quiet'])
 run(['up','-d','--wait','--wait-timeout','180'],240)
 topology=json.loads((folder/'runtime-topology.json').read_text())
 inspected=_inspect_runtime_services(project=project,compose=compose_path,override=None,topology=topology)
 storage=_actual_storage_readiness(project=project,compose=compose_path)
 readiness=_actual_role_readiness(project=project,compose=compose_path,topology=topology)
 assert storage['ready'] is True
 result={'mode':mode,'profile':profile,'status':'passed','runtime_services':inspected,'application_readiness':readiness,'storage_ready':True,'scope':'actual source services; disposable stores; upstream same-version OpenBao; no provider delivery or release image claim'}
 (report/f'run-{mode}-{profile}-{runid}.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'mode':mode,'profile':profile,'status':'passed','services':len(inspected)}),flush=True)
finally:
 run(['ps','-a'])
 # Runtime logs may include provider payloads; capture only status and targeted errors separately.
 run(['down','-v','--remove-orphans'],180)
