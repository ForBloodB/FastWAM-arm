"""Bounded same-core workload for the image transport benchmark only."""
import json,os,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'runs/feedback_vision_20261023'
os.sched_setaffinity(0,{3})
begin=time.monotonic();cpu=time.process_time();value=1
(root/'load_started.json').write_text(json.dumps({'pid':os.getpid(),'affinity':list(os.sched_getaffinity(0)),'wall_unix':time.time(),'maximum_seconds':180}))
while time.monotonic()-begin<180 and not (root/'load_stop').exists():
    for _ in range(20000):value=(value*1664525+1013904223)&0xffffffff
(root/'load_finished.json').write_text(json.dumps({'elapsed_s':time.monotonic()-begin,'cpu_s':time.process_time()-cpu,'pid':os.getpid()}))
