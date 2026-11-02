"""Common bounded RPC for honest DMA / MMIO / ARM CPU comparisons."""
import argparse,ctypes,fcntl,json,mmap,os,socket,struct,time,signal
from pathlib import Path
from pl_vision_service import recv_exact

CAP=2*1024*1024
RUN=0x40045201

class ImageProcessor:
    def __init__(self,backend):
        self.backend=backend;self.refs=[];self.identity={'backend':backend,'motor_commands':0}
        root=Path(__file__).resolve().parents[1]
        if backend=='dma':
            identity=self.uio('reflex-dma-identity',0xa0010000)
            assert identity[0]==0x52444d31,'DMA signature mismatch'
            self.identity['signature']=hex(identity[0]);self.ident=identity
            self.fd=os.open('/dev/reflex-image-dma',os.O_RDWR)
            self.input=mmap.mmap(self.fd,CAP,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE,offset=0)
            self.output=mmap.mmap(self.fd,CAP,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE,offset=4096)
        elif backend=='mmio':
            self.inp=self.uio('reflex-vision-input',0xa0000000);self.out=self.uio('reflex-vision-output',0xa0010000)
            self.inp[2]=0;assert self.out[2]==0x52565031
            self.identity['signature']=hex(self.out[2]);self.lib=ctypes.CDLL(str(root/'build/pl_vision/libplvision.so'))
            self.lib.vision_frame.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint32),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        else:
            self.lib=ctypes.CDLL(str(root/'build/pl_dma/libimagecpu.so'))
            self.lib.image_labels.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
    def uio(self,name,addr):
        found=[]
        for node in Path('/sys/class/uio').glob('uio*'):
            if (node/'name').read_text().strip()==name:found.append(node)
        assert len(found)==1,(name,found)
        node=found[0];assert int((node/'maps/map0/addr').read_text(),16)==addr
        fd=os.open('/dev/'+node.name,os.O_RDWR|os.O_SYNC);mem=mmap.mmap(fd,4096,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE)
        array=(ctypes.c_uint32*1024).from_buffer(mem);self.refs.extend([fd,mem,array]);return array
    def process(self,payload,w,h):
        start=time.perf_counter_ns();n=w*h
        if self.backend=='dma':
            before=self.ident[2];self.input[:len(payload)]=payload;t1=time.perf_counter_ns()
            fcntl.ioctl(self.fd,RUN,struct.pack('I',len(payload)));t2=time.perf_counter_ns()
            result=self.output[:n];t3=time.perf_counter_ns();after=self.ident[2]
            if ((after-before)&0xffffffff)!=1:raise RuntimeError('DMA frame counter mismatch')
            extra={'dma_ioctl_ms':(t2-t1)/1e6,'copy_in_ms':(t1-start)/1e6,'copy_out_ms':(t3-t2)/1e6,'frame_count':after}
        else:
            src=(ctypes.c_uint8*len(payload)).from_buffer_copy(payload);dst=(ctypes.c_uint8*n)()
            if self.backend=='mmio':
                stats=(ctypes.c_uint32*28)();rc=self.lib.vision_frame(self.inp,self.out,src,dst,w,h,stats)
                if rc:raise RuntimeError('MMIO processing failed '+str(rc))
            else:self.lib.image_labels(src,dst,n)
            result=bytes(dst);extra={}
        return result,{'processing_ms':(time.perf_counter_ns()-start)/1e6,**extra}

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['dma','mmio','cpu'],required=True);p.add_argument('--out',required=True);p.add_argument('--seconds',type=int,default=7200);p.add_argument('--port',type=int,default=18767);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGTERM,lambda *_:exit(0))
    lock=open('/tmp/reflex-task-image.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    processor=ImageProcessor(a.backend)
    (out/'identity.json').write_text(json.dumps({**processor.identity,'pid':os.getpid(),'max_pixels':640*480,'protocol':'RD02'},indent=2))
    server=socket.socket();server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);server.bind(('172.21.16.34',a.port));server.listen(2);server.settimeout(2)
    deadline=time.monotonic()+a.seconds;log=(out/'frames.jsonl').open('a',buffering=1);count=0
    print('TASK_IMAGE_READY '+a.backend,flush=True)
    try:
        while time.monotonic()<deadline:
            try:conn,peer=server.accept()
            except socket.timeout:continue
            if peer[0]!='172.21.16.96':conn.close();continue
            conn.settimeout(10);conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            with conn:
                while time.monotonic()<deadline:
                    try:
                        magic,w,h,n=struct.unpack('!4sHHI',recv_exact(conn,12))
                        if magic!=b'RD02' or not(0<w<=640 and 0<h<=480) or n!=4*w*h:raise ValueError('Invalid bounded image request')
                        upload_begin=time.perf_counter_ns();payload=recv_exact(conn,n);upload_end=time.perf_counter_ns();result,timing=processor.process(payload,w,h);timing['receive_payload_ms']=(upload_end-upload_begin)/1e6
                        count+=1;meta={'ok':True,'frame':count,'width':w,'height':h,'backend':a.backend,**timing}
                        encoded=json.dumps(meta).encode();conn.sendall(struct.pack('!I',len(encoded))+encoded+result);log.write(json.dumps(meta)+'\n')
                    except (EOFError,ConnectionError,socket.timeout):break
                    except ValueError as exc:
                        encoded=json.dumps({'ok':False,'error':str(exc)}).encode();conn.sendall(struct.pack('!I',len(encoded))+encoded);log.write(encoded.decode()+'\n');break
                    # DMA/kernel faults exit the whole service; do not reuse an uncertain stream state.
    finally:
        log.close();server.close();(out/'closed.json').write_text(json.dumps({'frames':count}))
if __name__=='__main__':main()
