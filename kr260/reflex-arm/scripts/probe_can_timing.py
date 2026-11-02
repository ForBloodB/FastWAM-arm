"""Read-only feedback timing with bounded scheduler/GC treatment. No enables."""
import argparse,gc,json,os,time
from pathlib import Path
from reflex.canbus import ReadOnlyCAN

p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seconds',type=int,default=30)
p.add_argument('--no-gc',action='store_true');p.add_argument('--rt-priority',type=int,default=0);p.add_argument('--cpu',type=int)
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
c=ReadOnlyCAN();c.capture=[];rows=[];events=[];current={}
def gc_event(phase,info):
    now=time.monotonic_ns()
    if phase=='start':current[info['generation']]=now
    else:events.append({'generation':info['generation'],'start_ns':current.get(info['generation'],now),'end_ns':now})
gc.callbacks.append(gc_event)
try:
    if a.cpu is not None:os.sched_setaffinity(0,{a.cpu})
    if a.rt_priority:os.sched_setscheduler(0,os.SCHED_FIFO,os.sched_param(a.rt_priority))
    if a.no_gc:gc.collect();gc.disable()
    c.poll(.15)
    if any(v['mode_bits_raw'] or v['fault_mask_raw'] for v in c.latest_status.values()):raise RuntimeError('All motors must be disabled and fault-free')
    start=time.monotonic_ns();previous=start
    for i in range(a.seconds*100):
        time.sleep(max(0,(start+i*10_000_000-time.monotonic_ns())/1e9));begin=time.monotonic_ns()
        values=[c.read(1,param,.02) for param in (0x7019,0x701B,0x701C,0x701A)]
        now=time.monotonic_ns()
        rows.append({'ns':now,'feedback_gap_ms':(now-previous)/1e6,'bundle_ms':(now-begin)/1e6,'q_v_volts_current':values})
        previous=now
        if i%10==0:
            for j in range(2,8):c.read(j,0x7019,.02)
    result={'completed':True,'actuator_access':'read_requests_only','gc_events':len(events),
            'configuration':vars(a),
            'max_gap_ms':max(r['feedback_gap_ms'] for r in rows),
            'gaps_over_50ms':sum(r['feedback_gap_ms']>50 for r in rows),
            'long_gc_events':[dict(e,duration_ms=(e['end_ns']-e['start_ns'])/1e6) for e in events if e['end_ns']-e['start_ns']>5_000_000]}
except BaseException as exc:result={'completed':False,'error':repr(exc)}
finally:
    gc.callbacks.remove(gc_event);c.close()
    (out/'samples.json').write_text(json.dumps(rows));(out/'gc_events.json').write_text(json.dumps(events))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
