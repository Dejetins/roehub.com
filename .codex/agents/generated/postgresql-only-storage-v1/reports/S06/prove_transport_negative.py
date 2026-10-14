"""Linux transport-witness failure/race fixtures; never an external-provider proof."""
import json, os, socket, struct, tempfile, threading
from trading.platform.postgres_budget import Reservation

def pair():
    listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen()
    client=socket.create_connection(listener.getsockname());server,_=listener.accept()
    listener.close();return client,server

for mode in ('fin','consumed_rst','concurrent_fin'):
    client,server=pair()
    with tempfile.TemporaryFile() as record:
        reservation=Reservation(os.dup(record.fileno()),'test')
        reservation.peer=socket.socket(fileno=os.dup(client.fileno()))
        if mode=='consumed_rst':
            server.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0))
        server.close()
        if mode=='consumed_rst':
            try:client.recv(1)
            except ConnectionResetError:pass
            else:raise AssertionError('fixture did not consume reset')
        client.close()
        errors=[]
        def finish():
            try:reservation.finish_transport()
            except Exception as error:errors.append(type(error).__name__)
        if mode=='concurrent_fin':
            threads=[threading.Thread(target=finish) for _ in range(8)]
            for t in threads:t.start()
            for t in threads:t.join()
        else:finish()
        assert not errors,errors
        assert bool(os.fstat(record.fileno()).st_size)==(mode=='consumed_rst'),mode
print(json.dumps({'fin_released':True,'consumed_rst_quarantined':True,'eight_concurrent_finalizers_safe':True,'scope':'Linux loopback transport, not unknown crash recovery'}))
