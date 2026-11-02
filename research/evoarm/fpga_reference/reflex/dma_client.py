import json,socket,struct,time
import numpy as np
from .feedback_vision import pixel_labels

def read_exact(sock,n):
    buf=bytearray()
    while len(buf)<n:
        x=sock.recv(n-len(buf))
        if not x:raise EOFError('Image RPC disconnected')
        buf.extend(x)
    return bytes(buf)

class TaskImageClient:
    def __init__(self,host='127.0.0.1',port=18767):
        self.sock=socket.create_connection((host,port),timeout=15);self.sock.settimeout(15)
        self.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);self.calls=[]
    def __call__(self,rgb,foreground):
        t0=time.perf_counter_ns();h,w=foreground.shape
        if not(0<w<=640 and 0<h<=480):raise ValueError('image size')
        payload=np.empty((h,w,4),np.uint8);payload[:,:,:3]=rgb;payload[:,:,3]=foreground
        raw=payload.tobytes();t1=time.perf_counter_ns()
        self.sock.sendall(struct.pack('!4sHHI',b'RD02',w,h,len(raw))+raw)
        n=struct.unpack('!I',read_exact(self.sock,4))[0]
        if n>4096:raise RuntimeError('invalid metadata length')
        meta=json.loads(read_exact(self.sock,n))
        if not meta.get('ok'):raise RuntimeError(meta)
        labels=np.frombuffer(read_exact(self.sock,w*h),np.uint8).reshape(h,w).copy()
        t2=time.perf_counter_ns();meta.update(pack_ms=(t1-t0)/1e6,rpc_ms=(t2-t1)/1e6,end_to_end_ms=(t2-t0)/1e6)
        self.calls.append(meta);return labels
    def close(self):self.sock.close()
