"""Read-only state after a guarded stop; never enables or writes a parameter."""
import argparse
import json
from pathlib import Path
import time
from reflex.canbus import ReadOnlyCAN

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
c=ReadOnlyCAN(host_id=0xFC);c.capture=[]
result={'motor_control_writes':0,'motor_enables':0,'diagnostic_host_id':0xFC,'samples':[]}
try:
    c.poll(.2)
    result['initial_status']={k:dict(v) for k,v in c.latest_status.items()}
    result['parameters']={j:{hex(param):c.read(j,param,.04) for param in
        (0x7005,0x700B,0x7017,0x7018,0x701E,0x701F,0x7020,0x7021,0x7028)} for j in range(1,8)}
    start=time.monotonic()
    for i in range(601):
        time.sleep(max(0,start+i*.02-time.monotonic()))
        q=c.read(6,0x7019,.03);q_ns=c.last_reply['rx_ns']
        v=c.read(6,0x701B,.03);iq=c.read(6,0x701A,.03)
        result['samples'].append({'position_rx_ns':q_ns,'q_rad':q,'v_rad_s':v,'iq_A':iq})
    c.poll(.1)
    result['final_status']={k:dict(v) for k,v in c.latest_status.items()}
    positions=[s['q_rad'] for s in result['samples']]
    result['j6_position_range_rad']=max(positions)-min(positions)
    result['j6_remained_disabled']=all(
        v['mode_bits_raw']==0 for v in (result['initial_status'][6],result['final_status'][6]))
    result['all_final_fault_masks_zero']=len(c.latest_status)==7 and all(
        v['fault_mask_raw']==0 for v in c.latest_status.values())
finally:
    (a.out/'state.json').write_text(json.dumps(result,indent=2))
    (a.out/'rx_frames.json').write_text(json.dumps(c.capture))
    c.close()
print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
