"""Actual CLI entrypoint, PG16 subprocesses, closed argv and exact pgpass lookup."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
from psycopg.conninfo import make_conninfo
from tools.installation.postgres_native import original_password,native_arguments
from trading.platform.postgres_budget import connect,reservation_counts
D='host=s06-pg port=5432 dbname=roehub user=postgres sslmode=verify-full sslrootcert=/input/root.crt'
for utility in ['pg_dump','pg_restore']:
 for flag in ['--db=host=foreign','-vdhost=foreign','--hos=foreign','--jobs=2','-j2','--create','-C']:
  try:native_arguments(utility,[flag])
  except ValueError:pass
  else:raise AssertionError('override accepted')
with tempfile.TemporaryDirectory() as tmp:
 passfile=Path(tmp)/'pgpass';passfile.write_text('s06-pg:5432:roehub:postgres:fixture\\:only\n');passfile.chmod(0o600)
 assert original_password({'host':'s06-pg','port':'5432','dbname':'roehub','user':'postgres','passfile':str(passfile)})=='fixture:only'
 before=reservation_counts()
 with connect(D,budget_class='admin',autocommit=True) as c:
  c.execute('DROP DATABASE IF EXISTS s06_cli_restore');c.execute('CREATE DATABASE s06_cli_restore')
 env=dict(os.environ,ROEHUB_STORAGE_POSTGRES_DSN=D)
 cli=[sys.executable,'-m','tools.installation.postgres_native']
 archive=Path(tmp)/'dump'
 with archive.open('wb') as f:subprocess.run(cli+['pg_dump','--format=custom','--no-owner','--no-privileges'],env=env,stdout=f,check=True,timeout=60)
 env['ROEHUB_STORAGE_POSTGRES_DSN']=make_conninfo(D,dbname='s06_cli_restore')
 subprocess.run(cli+['pg_restore','--exit-on-error','--no-owner','--no-privileges',str(archive)],env=env,check=True,timeout=60)
 out=subprocess.check_output(cli+['psql','-c','SELECT n FROM s06_native_fixture'],env=env,timeout=20)
 assert b'42' in out
 with connect(D,budget_class='admin',autocommit=True) as c:c.execute('DROP DATABASE s06_cli_restore')
 assert reservation_counts()==before
 print(json.dumps({'actual_pg16_cli_dump_restore_psql':'passed','tls_verify_full':True,'exact_pgpass_lookup':'passed','abbreviation_cluster_parallel_reconnect_rejections':14,'registry_unchanged':True}))
