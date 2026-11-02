"""KR260 internal diagnostics via atomic snapshots; never opens CAN devices."""
import argparse
import fcntl
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    lock=open('/tmp/reflex-pl-diagnostics.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    regions={};fds=[];history=[];tests=[];control=0
    for node in Path('/sys/class/uio').glob('uio*'):
        name=(node/'name').read_text().strip()
        if not name.startswith('reflex-v3-'):continue
        address=int((node/'maps/map0/addr').read_text(),16)
        index=(address-0xa0000000)//0x10000
        if not 0<=index<=10 or address!=0xa0000000+index*0x10000:
            raise RuntimeError('Unexpected v3 UIO region')
        expected='reflex-v3-control' if index==0 else f'reflex-v3-snapshot{index-1}'
        if name!=expected or index in regions:raise RuntimeError('UIO identity mismatch')
        fd=os.open('/dev/'+node.name,os.O_RDWR|os.O_SYNC);fds.append(fd)
        regions[index]=mmap.mmap(fd,4096,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE)
    if set(regions)!=set(range(11)):raise RuntimeError('All eleven v3 regions required')
    def write(value):
        nonlocal control
        if value&~0x3ff:raise ValueError('Only internal diagnostic controls permitted')
        control=value;struct.pack_into('<I',regions[0],0,value)
    def words():
        return [struct.unpack_from('<I',regions[1+i//2],8*(i%2))[0] for i in range(20)]
    def snap():
        write(control&~256);write(control|256)
        time.sleep(.00005)
        w=words();seq=struct.unpack_from('<I',regions[1],8)[0]
        if w[0]!=0x52463301 or seq!=w[1]:raise RuntimeError('Snapshot signature/sequence failure')
        write(control&~256)
        row={'host_ns':time.monotonic_ns(),'sequence':w[1],'control':w[2],
            'status':w[3],'cycle':w[4]+(w[5]<<32),'tick_count':w[6],
            'last_period':w[7],'heartbeat_cycle':w[8],'fault_cycle':w[9],
            'ramp_raw':w[10],'period_count':w[11],'period_min':w[12],'period_max':w[13],
            'period_sum':w[14]+(w[15]<<32),'off_nominal':w[16],
            'late_count':w[17],'stale_events':w[18],'tick_stale':bool(w[19]&1),
            'saturated':bool(w[19]&2)}
        row.update(enabled=bool(w[3]&1),state=(w[3]>>1)&3,fault=(w[3]>>4)&15)
        history.append(row);return row
    def clear_epoch():
        write(control|128);write(control&~128)
    def recover():
        write(0);write(5);write(0)
    def arm():
        write(3);write(2)
    def record(name,passed,**details):
        tests.append({'case':name,'passed':bool(passed),**details})
        print(json.dumps(tests[-1]),flush=True)
        if not passed:raise RuntimeError(name+' failed')
    result={'evidence':'physical_KR260_internal_PL_diagnostics_v3',
        'actuator_connected_to_design':False,'motor_commands':0,'tests':tests}
    stress=None
    try:
        write(0);first=snap()
        frozen=words();time.sleep(.005)
        record('snapshot_immutable_without_trigger',frozen==words())
        measurements=[]
        for phase in ('idle','one_core_cpu_load'):
            if phase!='idle':
                # A single owned, time-bounded CPU process; motors are stationary.
                stress=subprocess.Popen([sys.executable,'-c',
                    'import time; end=time.monotonic()+12\nwhile time.monotonic()<end: pass'])
            clear_epoch();time.sleep(.004);start=snap()
            end_host=time.monotonic()+10
            while time.monotonic()<end_host:
                time.sleep(.02);last=snap()
            delta=last['period_count']-start['period_count']
            total=last['period_sum']-start['period_sum']
            passed=(delta>=9000 and total==delta*100000 and last['period_min']==100000
                and last['period_max']==100000 and last['off_nominal']==0
                and last['late_count']==0 and not last['tick_stale'] and not last['saturated'])
            clock=(last['cycle']-start['cycle'])/((last['host_ns']-start['host_ns'])/1e9)
            measurements.append({'phase':phase,'continuous_intervals':delta,'min_cycles':last['period_min'],
                'max_cycles':last['period_max'],'mean_cycles':total/delta,'p99_cycles':100000 if passed else None,
                'observed_clock_Hz':clock})
            record('continuous_periods_'+phase,passed,**measurements[-1])
            if stress:
                stress.terminate();stress.wait(timeout=2);stress=None
        clear_epoch();time.sleep(.005);before=snap()
        write(512);time.sleep(.008);missing=snap()
        write(0);time.sleep(.004);after=snap()
        record('injected_monitor_tick_loss',missing['tick_stale'] and missing['stale_events']==1
            and after['late_count']>=1 and not after['tick_stale'],
            missing=missing,recovered=after)
        clear_epoch();time.sleep(.005);cleared=snap()
        record('clear_statistics_epoch',cleared['period_count']<100 and cleared['off_nominal']==0
            and cleared['stale_events']==0 and not cleared['tick_stale'])
        recover();arm();armed=snap();record('internal_arm',armed['enabled'] and armed['state']==1)
        time.sleep(.13);fault=snap()
        cycles=(fault['fault_cycle']-fault['heartbeat_cycle'])&0xffffffff
        record('internal_watchdog_latched',fault['state']==2 and fault['fault']==5
            and not fault['enabled'] and 9_999_998<=cycles<=10_000_005,cycles_since_heartbeat=cycles)
        for name,bit,code in (('estop',3,1),('limit',4,2),('sensor',5,3),('command',6,4)):
            recover();arm();write(1<<bit);fault=snap()
            write(0);latched=snap()
            record('internal_'+name+'_injection',fault['fault']==code and fault['state']==2
                and not fault['enabled'] and latched['state']==2)
        recover();final=snap();record('final_internal_disarmed',final['state']==0 and not final['enabled'])
        result.update(all_passed=True,period_measurements=measurements,final=final)
    except BaseException as exc:
        result.update(all_passed=False,error=repr(exc))
    finally:
        if stress:
            stress.terminate();stress.wait(timeout=2)
        write(0)
        for region in regions.values():region.close()
        for fd in fds:os.close(fd)
        lock.close()
        (out/'snapshots.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
        (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if not result['all_passed']:raise SystemExit(1)

if __name__=='__main__':main()
