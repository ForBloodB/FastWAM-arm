"""Adopt the measured J6 speed gain in RAM, only while J6 stays disabled."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
from reflex.canbus import ReadOnlyCAN
from reflex.qualification import require_experiment_release

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1]
require_experiment_release(json.loads((root/'configs/j6_csp_profile.json').read_text()),'gain_adoption')
proof_path=root/'runs/j6_repair_20261020/gain2p5_1deg_repeat10/summary.json'
proof=json.loads(proof_path.read_text())
if not (proof['passed'] and proof['parameters_restored'] and proof['initial_enable_state_restored']
    and proof['temporary_speed_kp']==2.5 and len(proof['cycle_results'])==10
    and all(r['passed'] for r in proof['cycle_results'])):
    raise RuntimeError('Ten positive cycles are required before gain adoption')
lock=os.fdopen(os.open('/tmp/reflex-can-actuation.lock',os.O_RDWR|os.O_NOFOLLOW),'r+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
c=ReadOnlyCAN();c.capture=[];old=None;written=False
result={'success':False,'joint':6,'enables_sent':0,'motion_references_sent':0,'flash_writes':0,
    'parameter':'0x701F','applied_speed_kp':2.5,'persistence':'RAM; guarded bench reapplies on future use',
    'verified_scope':'Positive unloaded CSP excursion <=1 degree only; reverse requires independent qualification',
    'evidence_sha256':hashlib.sha256(proof_path.read_bytes()).hexdigest()}
def disabled():
    c.poll(.08);s=c.latest_status.get(6)
    if not s or s['mode_bits_raw']!=0 or s['fault_mask_raw'] or s['temperature_C']>=45:
        raise RuntimeError('J6 must remain disabled and fault-free')
def write_gain(value):
    c.sock.send(struct.pack('=IB3x8s',0x80000000|(18<<24)|(0xFD<<8)|6,8,struct.pack('<H2xf',0x701F,value)))
    if not math.isclose(c.read(6,0x701F,.03),value,abs_tol=1e-6):raise RuntimeError('Gain readback mismatch')
try:
    disabled();result['initial_run_mode']=c.read(6,0x7005,.03)
    if result['initial_run_mode']!=0:raise RuntimeError('Expected disabled MIT state')
    q=c.read(6,0x7019,.03);old=c.read(6,0x701F,.03);result['original_speed_kp']=old
    if old not in (5.,2.5):raise RuntimeError('Unknown J6 gain, refusing automatic replacement')
    written=True;write_gain(2.5);disabled()
    result['position_change_deg']=math.degrees(c.read(6,0x7019,.03)-q)
    if abs(result['position_change_deg'])>.05:raise RuntimeError('Unexpected disabled movement')
    result['success']=True
except BaseException as exc:
    result['error']=repr(exc)
    if written and old is not None:
        try:disabled();write_gain(old);result['rolled_back']=True
        except Exception as err:result['rollback_error']=repr(err)
finally:
    result['final_status']={k:dict(v) for k,v in c.latest_status.items()}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))
    (a.out/'rx_frames.json').write_text(json.dumps(c.capture));c.close();lock.close()
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['success'] else 1)
