"""Verified-address UIO diagnostic test. No CAN, motor or physical GPIO access."""
import json
import mmap
import os
from pathlib import Path
import struct
import time

EXPECTED={0xA0000000:'control',0xA0010000:'stats',0xA0020000:'clock',0xA0030000:'period',0xA0040000:'fault'}
maps={};fds=[]
for node in Path('/sys/class/uio').glob('uio*'):
    address=int((node/'maps/map0/addr').read_text(),16)
    if address not in EXPECTED:continue
    name=(node/'name').read_text().strip()
    if not name.startswith('reflex-'):raise RuntimeError(f'Unexpected device name at {hex(address)}')
    fd=os.open('/dev/'+node.name,os.O_RDWR|os.O_SYNC)
    fds.append(fd);maps[EXPECTED[address]]=mmap.mmap(fd,4096,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE)
if set(maps)!=set(EXPECTED.values()):raise RuntimeError('All five verified UIO regions must be present')
def read(name,offset=0):return struct.unpack_from('<I',maps[name],offset)[0]
def write(value):struct.pack_into('<I',maps['control'],0,value)
def status():
    value=read('stats')
    return {'raw':value,'enabled':bool(value&1),'state':(value>>1)&3,'fault_code':(value>>4)&15}
def counter():
    while True:
        hi=read('clock');lo=read('stats',8)
        if read('clock')==hi:return (hi<<32)|lo
def arm():
    write(3);time.sleep(.001);write(2);time.sleep(.001)
def recover():
    write(0);time.sleep(.001);write(5);time.sleep(.001);write(0);time.sleep(.001)

result={'evidence':'real_KR260_PL_diagnostic_only','actuator_connected_to_design':False,
        'fpga_clock_nominal_Hz':100_000_000,'timing_period_cycles':100_000,'tests':[]}
out=Path('runs/pl_board');out.mkdir(parents=True,exist_ok=True)
try:
    if read('stats')>>16!=0x5246:raise RuntimeError('PL signature mismatch; refusing writes')
    write(0)
    start_host=time.monotonic_ns();start_cycle=counter()
    periods=[];ticks0=read('clock',8)
    for _ in range(1000):
        periods.append({'host_ns':time.monotonic_ns(),'pl_cycle':counter(),'tick_count':read('clock',8),'last_period_cycles':read('period')})
        time.sleep(.001)
    end_cycle=counter();end_host=time.monotonic_ns()
    ticks=read('clock',8)-ticks0
    result['observed_clock_Hz']=(end_cycle-start_cycle)/((end_host-start_host)/1e9)
    result['tick_count_delta']=ticks
    result['unique_measured_period_cycles']=sorted(set(p['last_period_cycles'] for p in periods))
    assert result['unique_measured_period_cycles']==[100000]
    recover();time.sleep(.12);write(2);time.sleep(.001)
    result['tests'].append({'case':'stale_arm_rejected','status':status(),'passed':not status()['enabled']})
    write(0);time.sleep(.001);arm()
    assert status()['state']==1 and status()['enabled']
    heartbeat=read('period',8)
    time.sleep(.15)
    observed=status()
    delta=(read('fault')-heartbeat)&0xffffffff
    result['tests'].append({'case':'hardware_watchdog','status':observed,'cycles_since_heartbeat':delta,
                            'passed':observed['state']==2 and observed['fault_code']==5 and not observed['enabled']})
    for name,bit,code in [('estop_injection',3,1),('limit_injection',4,2),('sensor_injection',5,3),('command_injection',6,4)]:
        recover();arm();write(1<<bit);time.sleep(.002)
        observed=status()
        passed=observed['state']==2 and observed['fault_code']==code and not observed['enabled']
        result['tests'].append({'case':name,'status':observed,'passed':passed})
        write(0);time.sleep(.002)
        assert status()['state']==2,'Fault must latch'
    recover()
    result['final_status']=status()
    assert result['final_status']['state']==0 and not result['final_status']['enabled']
    result['all_passed']=all(t['passed'] for t in result['tests'])
    (out/'period_samples.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in periods))
except BaseException as exc:
    result['error']=repr(exc);result['all_passed']=False
finally:
    write(0)
    for m in maps.values():m.close()
    for fd in fds:os.close(fd)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if not result['all_passed']:raise SystemExit(1)
