"""Read J2/J3 planning registers without entering a motor control mode."""
import argparse
import json
from pathlib import Path
from reflex.canbus import ReadOnlyCAN

p=argparse.ArgumentParser()
p.add_argument('--out',required=True,type=Path)
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
result={'motor_control_writes':0,'motor_enables':0,'mode_changes':0,
        'motor_model_identified':False,'PP_motion_qualified':False,'motors':{}}
for joint in (2,3):
    c=ReadOnlyCAN(host_id=0xFC);c.capture=[]
    entry={}
    try:
        c.poll(.2)
        entry['initial_status']={j:dict(s) for j,s in c.latest_status.items()}
        if len(c.latest_status)!=7 or any(s['fault_mask_raw'] or s['mode_bits_raw']!=0 for s in c.latest_status.values()):
            raise RuntimeError('This readback snapshot requires seven disabled fault-free states')
        entry['registers']={}
        entry['raw_replies']=[]
        for param in (0x7005,0x7024,0x7025,0x7019):
            values=[]
            for _ in range(3):
                values.append(c.read(joint,param,.1))
                entry['raw_replies'].append(dict(c.last_reply))
            entry['registers'][hex(param)]=values
        c.poll(.1)
        entry['final_status']={j:dict(s) for j,s in c.latest_status.items()}
        entry['readback_complete']=True
    except Exception as exc:
        entry['error']=repr(exc);entry['readback_complete']=False
    finally:
        entry['requests']=c.log
        (a.out/f'j{joint}_rx.json').write_text(json.dumps(c.capture))
        result['motors'][joint]=entry
        c.close()
(a.out/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
