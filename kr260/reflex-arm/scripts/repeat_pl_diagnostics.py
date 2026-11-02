"""Repeated tests of verified PL diagnostic registers. No motor/external-pin I/O."""
import argparse
import fcntl
import json
import math
import mmap
import os
from pathlib import Path
import struct
import time

p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--trials',type=int,default=100)
p.add_argument('--label',required=True);a=p.parse_args()
if not 1<=a.trials<=100:p.error('1..100 trials')
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
lock=open('/tmp/reflex-pl-diagnostics.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
expected={0xA0000000:'control',0xA0010000:'stats',0xA0020000:'clock',0xA0030000:'period',0xA0040000:'fault'}
maps={};fds=[];rows=[];periods=[]
result={'label':a.label,'completed':False,'evidence':'real_KR260_PL_internal_diagnostics',
        'actuator_connected':False,'physical_estop_tested':False,'nominal_clock_Hz':100_000_000,
        'fault_response_measurement':'upper bound from pre-MMIO PL counter read to fault latch; includes PS/AXI latency',
        'period_measurement':'sampled last-period register plus total tick count; not a continuous jitter histogram'}

def read(name,offset=0):return struct.unpack_from('<I',maps[name],offset)[0]
def write(value):struct.pack_into('<I',maps['control'],0,value)
def status():
    value=read('stats')
    return {'enabled':bool(value&1),'state':(value>>1)&3,'fault_code':(value>>4)&15}
def counter():
    while True:
        high=read('clock');low=read('stats',8)
        if high==read('clock'):return (high<<32)|low
def recover():
    write(0);time.sleep(.001);write(5);time.sleep(.001);write(0);time.sleep(.001)
    assert status()['state']==0 and not status()['enabled'],'Safe reset did not disarm'
def arm():
    write(3);time.sleep(.001);write(2);time.sleep(.001)
    assert status()['state']==1 and status()['enabled'],'Fresh heartbeat arm failed'
def until_fault():
    limit=time.monotonic()+.2
    while time.monotonic()<limit:
        state=status()
        if state['state']==2:return state
        time.sleep(.0005)
    raise TimeoutError('Fault latch not observed')
def quantiles(values):
    values=sorted(values)
    return {'n':len(values),'min':values[0],'mean':sum(values)/len(values),
            'p99':values[min(len(values)-1,math.ceil(len(values)*.99)-1)],'max':values[-1]}

try:
    for node in Path('/sys/class/uio').glob('uio*'):
        address=int((node/'maps/map0/addr').read_text(),16)
        if address not in expected:continue
        name=(node/'name').read_text().strip()
        if name!='reflex-'+expected[address]:raise RuntimeError('Unexpected UIO name at known address')
        fd=os.open('/dev/'+node.name,os.O_RDWR|os.O_SYNC);fds.append(fd)
        maps[expected[address]]=mmap.mmap(fd,4096,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE)
    if set(maps)!=set(expected.values()):raise RuntimeError('Verified PL diagnostic regions missing')
    if read('stats')>>16!=0x5246:raise RuntimeError('Diagnostic signature mismatch')
    write(0);recover()
    cycle0=counter();ticks0=read('clock',8)
    for _ in range(2000):
        periods.append({'host_ns':time.monotonic_ns(),'pl_cycle':counter(),
                        'tick_count':read('clock',8),'last_period_cycles':read('period')})
        time.sleep(.001)
    cycle1=counter();ticks1=read('clock',8)
    result.update(sampled_period_cycles=quantiles([r['last_period_cycles'] for r in periods]),
                  elapsed_pl_cycles=cycle1-cycle0,elapsed_ticks=(ticks1-ticks0)&0xffffffff)
    assert all(r['last_period_cycles']==100000 for r in periods),'Sampled PL period changed'
    assert abs(result['elapsed_ticks']-(cycle1-cycle0)/100000)<3,'Tick accumulation disagrees with elapsed PL clock'
    for trial in range(a.trials):
        recover();arm();heartbeat=read('period',8)
        observed=until_fault();cycles=(read('fault')-heartbeat)&0xffffffff
        assert observed=={'enabled':False,'state':2,'fault_code':5},'Bad watchdog state'
        assert cycles==10000001,'Hardware watchdog cycle count changed'
        rows.append({'trial':trial,'case':'watchdog','cycles_since_heartbeat':cycles,'status':observed,'passed':True})
        # A reset with arm held high must never re-enable a latched fault.
        write(7);time.sleep(.001);assert status()['state']==2 and not status()['enabled']
        for case,bit,code in [('estop_register',3,1),('limit_register',4,2),('sensor_register',5,3),('command_register',6,4)]:
            recover();arm();before=counter();write(1<<bit);observed=until_fault()
            fault_cycle=read('fault');upper=(fault_cycle-(before&0xffffffff))&0xffffffff
            assert observed=={'enabled':False,'state':2,'fault_code':code},'Incorrect fault state/code'
            assert upper<5_000_000,'Injection response upper bound exceeded 50 ms'
            write(0);time.sleep(.001);assert status()['state']==2,'Fault did not latch'
            write(7);time.sleep(.001);assert status()['state']==2 and not status()['enabled'],'Armed reset bypassed latch'
            rows.append({'trial':trial,'case':case,'prewrite_cycle':before,'fault_cycle_low':fault_cycle,
                         'response_upper_bound_us':upper/100,'status':observed,'passed':True})
        if (trial+1)%20==0:print(json.dumps({'label':a.label,'trials_completed':trial+1}),flush=True)
    recover();time.sleep(.12);write(2);time.sleep(.001)
    result['stale_arm_rejected']=not status()['enabled']
    assert result['stale_arm_rejected']
    recover();result['completed']=True
except BaseException as exc:result['error']=repr(exc)
finally:
    if 'control' in maps:
        write(0);result['final_status']=status() if 'stats' in maps else None
    result['cases_completed']=len(rows)
    result['fault_response_upper_bound_us']={name:quantiles([r['response_upper_bound_us'] for r in rows if r['case']==name])
        for name in ('estop_register','limit_register','sensor_register','command_register') if any(r['case']==name for r in rows)}
    for mapping in maps.values():mapping.close()
    for fd in fds:os.close(fd)
    lock.close()
    (out/'cases.json').write_text(json.dumps(rows));(out/'period_samples.json').write_text(json.dumps(periods))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
raise SystemExit(0 if result['completed'] else 1)
