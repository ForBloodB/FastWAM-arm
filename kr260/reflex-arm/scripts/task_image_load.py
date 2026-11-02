"""Bounded CPU contention for the image service, with explicit stop file."""
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'runs/residual_dma_20261023'
os.sched_setaffinity(0,{3});start=time.monotonic();cpu=time.process_time();x=1
(root/f'load_{a.tag}_started.json').write_text(json.dumps({'pid':os.getpid(),'affinity':[3],'maximum_s':480,'wall_unix':time.time()}))
while time.monotonic()-start<480 and not (root/f'load_{a.tag}_stop').exists():
    for _ in range(20000):x=(x*1664525+1013904223)&0xffffffff
(root/f'load_{a.tag}_finished.json').write_text(json.dumps({'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'pid':os.getpid()}))
