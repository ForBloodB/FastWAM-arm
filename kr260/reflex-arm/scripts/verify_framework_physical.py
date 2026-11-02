"""Audit completed CAN tests and read the final state, without control writes."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
from reflex.canbus import ReadOnlyCAN,parse_status_frame

root=Path('runs/framework_20261020')
initial=json.loads((root/'parameter_snapshot.json').read_text())
lock=os.fdopen(os.open('/tmp/reflex-can-actuation.lock',os.O_RDWR|os.O_NOFOLLOW),'r+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
c=ReadOnlyCAN();c.capture=[]
audit=[];faults=0
try:
    for path in sorted(root.glob('*/tx_frames.json')):
        summary=json.loads(path.with_name('summary.json').read_text())
        assert summary.get('parameters_restored') and not summary.get('restoration_error'),path
        tx=json.loads(path.read_text());rx=json.loads(path.with_name('rx_frames.json').read_text())
        for row in tx:
            identifier,dlc,payload=struct.unpack('=IB3x8s',bytes.fromhex(row['frame']))
            kind=(identifier>>24)&31
            assert dlc==8 and kind==row['kind'] and kind in (1,3,4,18)
            assert (identifier&255)==summary['joint']==row['joint']
            if kind==18:assert struct.unpack_from('<H',payload)[0] in (0x7005,0x700b,0x7016,0x7017,0x7018,0x701f,0x7028)
        case_faults=0
        for row in rx:
            frame=bytes.fromhex(row['frame_hex'])
            try:case_faults+=bool(parse_status_frame(frame)['fault_mask_raw'])
            except ValueError:pass
        faults+=case_faults
        audit.append({'case':path.parent.name,'joint':summary['joint'],'only_target_written':True,
            'zero_and_flash_commands':0,'control_frames':len(tx),'status_fault_frames':case_faults,
            'parameters_restored':True,'completed_profile':summary['completed_profile'],
            'tracking_passed':summary['passed'],'cycles_completed':len(summary.get('cycle_results',[]))})
    c.poll(.3);motors=[]
    unchanged=(0x7005,0x700b,0x7017,0x7018,0x701e,0x701f,0x7020,0x7021,0x7028,0x7029)
    for j in range(1,8):
        values={hex(p):c.read(j,p,.05) for p in (*unchanged,0x7019,0x701A,0x701C)}
        for p in unchanged:
            assert math.isclose(values[hex(p)],initial['parameters'][str(j)][hex(p)],rel_tol=1e-6,abs_tol=1e-6),(j,hex(p))
        motors.append({'joint':j,'parameters':values})
    c.poll(.2)
    for row in motors:
        st=dict(c.latest_status[row['joint']]);row['status']=st
        assert st['fault_mask_raw']==0
        assert st['mode_bits_raw']==initial['before_status'][str(row['joint'])]['mode_bits_raw']
    result={'evidence':'physical_CAN_post_test_audit','verification_control_writes':0,
        'all_parameters_restored':True,'all_modes_match_initial':True,'all_final_faults_zero':True,
        'status_fault_frames_during_tests':faults,'motors':motors,'cases':audit}
    (root/'final_verification.json').write_text(json.dumps(result,indent=2))
    (root/'final_verification_rx.json').write_text(json.dumps(c.capture))
    print(json.dumps({'all_parameters_restored':True,'cases':audit,
        'modes':{r['joint']:r['status']['mode_bits_raw'] for r in motors}}))
finally:c.close();lock.close()
