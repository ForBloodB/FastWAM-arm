"""Bounded, owned CPU workload on KR260. No network, CAN or MMIO access."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time

def worker(stop,ready,queue,cpu):
    os.sched_setaffinity(0,{cpu});value=17;loops=0;start=time.monotonic();cpu0=time.process_time();ready.set()
    while not stop.is_set():
        for _ in range(10000):value=(value*1664525+1013904223)&0xffffffff
        loops+=10000
    queue.put({'pid':os.getpid(),'cpu':cpu,'iterations':loops,
               'wall_s':time.monotonic()-start,'process_cpu_s':time.process_time()-cpu0})

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,default=180)
    a=p.parse_args()
    if not 10<=a.seconds<=600:p.error('10..600 seconds only')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    ctx=mp.get_context('spawn');stop=ctx.Event();queue=ctx.Queue();children=[];rows=[]
    # Never acquire a multiprocessing Event lock from a signal handler.
    requested_stop=[False]
    def request_stop(*_):requested_stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    info={'pid':os.getpid(),'child_pids':[]}
    try:
        cpus=sorted(os.sched_getaffinity(0))[1:]
        for cpu in cpus:
            ready=ctx.Event();child=ctx.Process(target=worker,args=(stop,ready,queue,cpu));child.start();children.append(child)
            if not ready.wait(10):raise RuntimeError('CPU load child not ready')
        info={'pid':os.getpid(),'child_pids':[c.pid for c in children],'cpu_affinity':cpus,'reserved_cpu':0,
              'actuator_access':False,'requested_seconds':a.seconds}
        (out/'ready.json').write_text(json.dumps(info));print('READY',flush=True)
        end=time.monotonic()+a.seconds
        while not requested_stop[0] and time.monotonic()<end:time.sleep(.1)
    finally:
        stop.set()
        for child in children:
            child.join(10)
            if child.is_alive():child.terminate();child.join()
        while not queue.empty():rows.append(queue.get())
        (out/'summary.json').write_text(json.dumps({'workload':info,'workers':rows},indent=2))

if __name__=='__main__':main()
