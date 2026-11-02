"""Bounded read-only stationary joint window. No motor enable/write API."""
import argparse
import json
from pathlib import Path
import statistics
import time
from reflex.canbus import ReadOnlyCAN

p=argparse.ArgumentParser();p.add_argument('--out',required=True)
p.add_argument('--seconds',type=float,default=12.)
p.add_argument('--allow-enabled-hold',action='store_true')
a=p.parse_args()
if not 6<=a.seconds<=20:p.error('Stationary capture window must be 6..20 seconds')
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
c=ReadOnlyCAN(host_id=0xFC);rows=[];result={'success':False,'motor_enables_sent':0,'writes_sent':0,
    'requested_duration_s':a.seconds,'max_joint_range_rad':.001}
result.update(diagnostic_host_id=0xFC,enabled_stationary_hold_allowed=a.allow_enabled_hold)
try:
    c.poll(.15)
    start=time.monotonic();initial_modes=None
    for i in range(round(a.seconds*20)+1):
        time.sleep(max(0,start+i*.05-time.monotonic()))
        q=[c.read(j,0x7019,.03) for j in range(1,8)]
        c.drain();now=time.monotonic_ns()
        status=c.latest_status
        if len(status)!=7 or any(v['fault_mask_raw'] or v['mode_bits_raw'] not in ((0,2) if a.allow_enabled_hold else (0,)) or
            now-v['rx_ns']>120_000_000 for v in status.values()):
            raise RuntimeError('Stationary acquisition requires fresh fault-free permitted motor states')
        modes={j:v['mode_bits_raw'] for j,v in status.items()}
        if initial_modes is None:initial_modes=modes
        elif modes!=initial_modes:raise RuntimeError('Motor state changed during stationary acquisition')
        rows.append({'ns':now,'q_rad':q})
        if i==0:print('READY',flush=True)
    columns=list(zip(*(r['q_rad'] for r in rows)))
    ranges=[max(v)-min(v) for v in columns]
    result.update(success=all(r<.001 for r in ranges),samples=len(rows),
        duration_s=time.monotonic()-start,q_mean_rad=[statistics.mean(v) for v in columns],
        q_range_rad=ranges,last_status=c.latest_status,motor_modes_unchanged=True)
    if not result['success']:result['error']='Joint movement exceeded stationary acquisition tolerance'
except Exception as exc:result['error']=repr(exc)
finally:
    c.close();(out/'samples.json').write_text(json.dumps(rows))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
raise SystemExit(0 if result['success'] else 1)
