"""Native dump/restore through one counted upstream; disposable containers/data."""
import base64,hashlib,json,struct,uuid,os
from pathlib import Path
from psycopg.conninfo import make_conninfo
from tools.installation.postgres_native import run_native
from tools.installation.postgres_budget_recovery import Engine
from trading.platform.postgres_budget import connect,reservation_counts
DSN=os.environ.get('S06_NATIVE_DSN','host=s06-pg dbname=roehub user=postgres')
engine=Engine('/var/run/docker.sock')

def native(command,options):
    created=engine.call('POST','/containers/create',{
        'Image':'postgres:16','Cmd':([*command,'--dbname',make_conninfo('',**options)] if command[0]!='sh' else command),'Env':['PGDATABASE='+make_conninfo('',**options)],
        'Labels':{'io.roehub.proof':'S06-native'},
        'HostConfig':{'NetworkMode':'container:s06-native-proof','CapDrop':['ALL'],
                     'Binds':['s06-native-transfer:/input:ro']},
    })['Id']
    try:
        engine.call('POST',f'/containers/{created}/start')
        state=engine.call('POST',f'/containers/{created}/wait')
        raw=engine.call('GET',f'/containers/{created}/logs?stdout=true&stderr=true',raw=True)
        output=bytearray();errors=bytearray()
        while raw:
            n=struct.unpack('>I',raw[4:8])[0]
            if raw[0]==1:output.extend(raw[8:8+n])
            else:errors.extend(raw[8:8+n])
            raw=raw[8+n:]
        assert state['StatusCode']==0,('native_exit',state['StatusCode'],errors.decode())
        return bytes(output)
    finally:engine.call('DELETE',f'/containers/{created}?force=true')

if __name__=='__main__':
    with connect(DSN,budget_class='admin',autocommit=True) as c:
        c.execute('CREATE TABLE IF NOT EXISTS s06_native_fixture(n int)')
        c.execute('TRUNCATE s06_native_fixture');c.execute('INSERT INTO s06_native_fixture VALUES(42)')
        c.execute('DROP DATABASE IF EXISTS s06_native_restore')
        c.execute('CREATE DATABASE s06_native_restore')
    before=reservation_counts()
    dump=base64.b64decode(run_native(DSN,lambda options:native(['sh','-ec','pg_dump --no-password --format=custom --no-owner --no-privileges --dbname "$PGDATABASE" | base64 -w0'],options)),validate=True)
    assert dump.startswith(b'PGDMP')
    Path('/native/dump').write_bytes(dump)
    run_native(make_conninfo(DSN,dbname='s06_native_restore'),lambda options:native(
        ['sh','-ec','exec pg_restore --no-password --exit-on-error --no-owner --no-privileges --dbname "$PGDATABASE" /input/dump'],options))
    with connect(make_conninfo(DSN,dbname='s06_native_restore'),budget_class='admin') as c:
        assert c.execute('SELECT n FROM s06_native_fixture').fetchone()==(42,)
    for _ in range(12):
        output=run_native(DSN,lambda options:native(['psql','--no-password','--no-psqlrc','-Atc','SELECT 42'],options))
        assert output.strip()==b'42'
    assert reservation_counts()==before,(before,reservation_counts())
    with connect(DSN,budget_class='admin',autocommit=True) as c:c.execute('DROP DATABASE s06_native_restore')
    Path('/native/dump').unlink()
    print(json.dumps({'native_dump_restore':'passed','payload_bytes':len(dump),'dump_sha256':hashlib.sha256(dump).hexdigest(),'native_query_sessions':12,'registry_restored':True,'pg_upstreams_per_native_session':1,'scope':'disposable direct PG', 'sslmode':os.environ.get('S06_NATIVE_DSN','').split('sslmode=')[-1].split(' ')[0] if 'sslmode=' in os.environ.get('S06_NATIVE_DSN','') else 'prefer'}))
