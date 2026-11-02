"""Bounded image-only physical PL service. No CAN, motor API or external GPIO.
Uses two named, address-verified UIO mappings and a signature-verified image core.
"""
import argparse,ctypes,fcntl,json,mmap,os,socket,struct,time
from pathlib import Path

def recv_exact(sock,n):
    buf=bytearray()
    while len(buf)<n:
        part=sock.recv(n-len(buf))
        if not part:raise EOFError
        buf.extend(part)
    return bytes(buf)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seconds',type=int,default=7200);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    lock=open('/tmp/reflex-vision-pl.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    maps={};fds=[];arrays={}
    expected={'reflex-vision-input':0xa0000000,'reflex-vision-output':0xa0010000}
    for node in Path('/sys/class/uio').glob('uio*'):
        name=(node/'name').read_text().strip()
        if name not in expected:continue
        addr=int((node/'maps/map0/addr').read_text(),16)
        if addr!=expected[name] or name in maps:raise RuntimeError('UIO identity mismatch')
        fd=os.open('/dev/'+node.name,os.O_RDWR|os.O_SYNC);fds.append(fd)
        maps[name]=mmap.mmap(fd,4096,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE)
        arrays[name]=(ctypes.c_uint32*1024).from_buffer(maps[name])
    if set(maps)!=set(expected):raise RuntimeError('Image core UIO mappings missing')
    inp=arrays['reflex-vision-input'];output=arrays['reflex-vision-output'];inp[2]=0
    if output[2]!=0x52565031:raise RuntimeError('Image core signature mismatch')
    lib=ctypes.CDLL(str(Path(__file__).resolve().parents[1]/'build/pl_vision/libplvision.so'))
    u32=ctypes.POINTER(ctypes.c_uint32);u8=ctypes.POINTER(ctypes.c_uint8)
    lib.vision_frame.argtypes=[u32,u32,u8,u8,ctypes.c_int,ctypes.c_int,u32];lib.vision_frame.restype=ctypes.c_int
    server=socket.socket();server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);server.bind(('172.21.16.34',18766));server.listen(2);server.settimeout(2)
    log=(out/'frames.jsonl').open('a',buffering=1);deadline=time.monotonic()+a.seconds;frames=0
    (out/'identity.json').write_text(json.dumps({'pid':os.getpid(),'signature':hex(output[2]),'uio':expected,'motor_commands':0,'external_pins':0,'protocol':'RV01 RGB+foreground pixel labels','timeout_s':a.seconds},indent=2))
    print('PL_VISION_READY',flush=True)
    try:
        while time.monotonic()<deadline:
            try:conn,peer=server.accept()
            except socket.timeout:continue
            if peer[0]!='172.21.16.96':conn.close();continue
            conn.settimeout(30)
            with conn:
                while time.monotonic()<deadline:
                    try:
                        magic,w,h,n=struct.unpack('!4sHHI',recv_exact(conn,12))
                        if magic!=b'RV01' or not(0<w<=640 and 0<h<=480) or n!=4*w*h:raise ValueError('Invalid bounded image request')
                        payload=recv_exact(conn,n);src=(ctypes.c_uint8*n).from_buffer_copy(payload)
                        dst=(ctypes.c_uint8*(w*h))();stats=(ctypes.c_uint32*28)();start=time.perf_counter_ns()
                        rc=lib.vision_frame(inp,output,src,dst,w,h,stats);end=time.perf_counter_ns()
                        if rc:raise RuntimeError('PL processing error '+str(rc))
                        frames+=1;meta={'ok':True,'frame':frames,'width':w,'height':h,'ps_mmio_ms':(end-start)/1e6,'stats':list(stats),'motor_commands':0}
                        log.write(json.dumps(meta)+'\n');encoded=json.dumps(meta).encode()
                        conn.sendall(struct.pack('!I',len(encoded))+encoded+bytes(dst))
                    except (EOFError,socket.timeout,ConnectionError):break
                    except Exception as exc:
                        encoded=json.dumps({'ok':False,'error':str(exc)}).encode();conn.sendall(struct.pack('!I',len(encoded))+encoded)
                        log.write(json.dumps({'error':str(exc)})+'\n');break
    finally:
        log.close();server.close();(out/'closed.json').write_text(json.dumps({'frames':frames,'motor_commands':0}))
        # mmap ownership ends with the process. No motor or unrelated UIO writes.
if __name__=='__main__':main()
