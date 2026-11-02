"""Verify parameter latching while J6 remains disabled; never sends enable."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import struct
import time
from reflex.canbus import ReadOnlyCAN

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
lock=os.fdopen(os.open('/tmp/reflex-can-actuation.lock',os.O_RDWR|os.O_NOFOLLOW),'r+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
c=ReadOnlyCAN();c.capture=[];writes=[];original={};changed=[]
caps={0x700B:.5,0x7018:.6,0x7017:math.radians(.25),0x7028:2000}
kinds={0x700B:'f',0x7018:'f',0x7017:'f',0x7028:'I',0x7005:'b'}
result={'motor_enable_commands':0,'motor_mode_expected':0,'success':False}
def snapshot():return {hex(p):c.read(6,p,.03) for p in (0x7005,0x700B,0x7018,0x7017,0x7028,0x701E,0x701F,0x7020,0x7021)}
def write(p,value):
    frame=struct.pack('=IB3x8s',0x80000000|(18<<24)|(0xFD<<8)|6,8,
        struct.pack('<H2x',p)+struct.pack('<'+kinds[p],value).ljust(4,b'\0'))
    c.sock.send(frame);writes.append({'ns':time.monotonic_ns(),'parameter':hex(p),'value':value})
    actual=c.read(6,p,.03)
    if not math.isclose(value,actual,rel_tol=1e-5,abs_tol=1e-6):raise RuntimeError('Readback mismatch')
def assert_disabled():
    c.poll(.06);s=c.latest_status.get(6)
    if not s or s['mode_bits_raw']!=0 or s['fault_mask_raw']:raise RuntimeError('J6 must remain disabled')
try:
    assert_disabled();q=c.read(6,0x7019,.03)
    result['before']=snapshot();original={p:c.read(6,p,.03) for p in kinds}
    if original[0x7005]!=0:raise RuntimeError('Expected MIT mode before audit')
    for p,value in caps.items():changed.append(p);write(p,value)
    result['caps_before_mode_switch']=snapshot()
    changed.append(0x7005);write(0x7005,5)
    result['after_mode_switch']=snapshot();assert_disabled()
    result['caps_survived_mode_switch']=all(math.isclose(result['after_mode_switch'][hex(p)],v,rel_tol=1e-5,abs_tol=1e-6) for p,v in caps.items())
    result['position_change_deg']=math.degrees(c.read(6,0x7019,.03)-q)
    if abs(result['position_change_deg'])>.05:raise RuntimeError('Unexpected disabled displacement')
    result['success']=True
except BaseException as exc:result['error']=repr(exc)
finally:
    try:
        assert_disabled()
        for p in (0x7005,0x700B,0x7018,0x7017,0x7028):
            if p in changed:write(p,original[p])
        result['restored']=snapshot();result['parameters_restored']=True;assert_disabled()
    except BaseException as exc:result['restoration_error']=repr(exc)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))
    (a.out/'rx_frames.json').write_text(json.dumps(c.capture))
    (a.out/'writes.json').write_text(json.dumps(writes))
    c.close();lock.close()
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['success'] and result.get('parameters_restored') else 1)
