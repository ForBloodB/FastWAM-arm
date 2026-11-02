"""Bounded J1/J7 diagnostic procedures, explicitly authorized by operator.

No arm pose, zero, or saved configuration is changed. Results distinguish lack
of motion from a successful test. Other joints are never enabled.
"""
import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import signal
import time
from reflex.canbus import ReadOnlyCAN
from reflex.diagnostic_motor import DiagnosticMotor
from reflex.motion import VelocityGuard,can_timeout_raw,j1_test_target,check_j1_sample,repeated_diagnostic_target

p=argparse.ArgumentParser()
p.add_argument('action',choices=['move-j1'])
p.add_argument('--out',required=True)
p.add_argument('--execute-authorized-test',action='store_true',required=True)
p.add_argument('--kp',type=float,default=None)
p.add_argument('--torque-limit',type=float,default=None)
p.add_argument('--repetitions',type=int,default=1)
p.add_argument('--rt-priority',type=int,default=0)
p.add_argument('--cpu',type=int,default=None)
a=p.parse_args()
if not 1<=a.repetitions<=20:p.error('Repetitions must be 1..20')
if not 0<=a.rt_priority<=20:p.error('Diagnostic RT priority must be 0..20')
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
try:lock_fd=os.open('/tmp/reflex-can-actuation.lock',os.O_RDWR|os.O_NOFOLLOW)
except FileNotFoundError:
    lock_fd=os.open('/tmp/reflex-can-actuation.lock',os.O_RDWR|os.O_NOFOLLOW|os.O_CREAT|os.O_EXCL,0o600)
lock=os.fdopen(lock_fd,'r+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
reader=ReadOnlyCAN();reader.capture=[]
mid=1
motor=DiagnosticMotor(reader,mid)
result={'action':a.action,'evidence':'real_KR260_CAN','success':False,'enabled_joints':[],
        'physical_grasp':False,'pl_motor_protection':False}
original={};changed=[];samples=[]
gc_was_enabled=gc.isenabled();scheduler_before=os.sched_getscheduler(0)
priority_before=os.sched_getparam(0);affinity_before=os.sched_getaffinity(0)
def interrupted(*_):raise KeyboardInterrupt('Interrupted')
signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
try:
    if a.cpu is not None:
        if a.cpu not in affinity_before:raise ValueError('Unavailable requested CPU')
        os.sched_setaffinity(0,{a.cpu})
    if a.rt_priority:os.sched_setscheduler(0,os.SCHED_FIFO,os.sched_param(a.rt_priority))
    result['scheduling']={'policy':os.sched_getscheduler(0),'priority':os.sched_getparam(0).sched_priority,
                          'affinity':sorted(os.sched_getaffinity(0))}
    reader.poll(.05)
    baseline=[reader.read(i,0x7019) for i in range(1,8)]
    initial=baseline[mid-1]
    result['baseline_positions_rad']=baseline
    pre_modes={i:reader.read(i,0x7005) for i in range(1,8)}
    pre_zero={i:reader.read(i,0x7029) for i in range(1,8)}
    pre_voltages={i:reader.read(i,0x701C) for i in range(1,8)}
    result['preflight']={'mode_parameters':pre_modes,'zero_states':pre_zero,'voltages_V':pre_voltages,
        'note':'Disabled motors do not emit periodic status; fault and temperature checks run immediately after enable.'}
    if any(mode!=0 for mode in pre_modes.values()):
        raise RuntimeError('All seven motors must use MIT mode before test')
    if any(zero!=1 for zero in pre_zero.values()):
        raise RuntimeError('All seven motors must have valid zero state')
    if any(not 42<voltage<54 for voltage in pre_voltages.values()):
        raise RuntimeError('Supply outside diagnostic envelope')
    if a.action=='clear-j7':
        if pre['fault_mask_raw']!=16:raise RuntimeError('Expected isolated stall overload only')
        # Clearing is itself a type-4 stop, not enable. Exactly one clear attempt.
        motor.clear_stall()
        reader.poll(.05)
        # This firmware retained the status latch after clear=1 alone; the
        # following normal stop completed the observed clear/stop sequence.
        motor.disable()
        reader.poll(3.)
        post=motor.status();result['post_status']=dict(post)
        result['clear_attempts']=1
        result['success']=post['fault_mask_raw']==0 and post['mode_bits_raw']==0
    else:
        result['fault_check']='Required immediately after enable and throughout motion'
        params=(0x700B,0x7028,0x7005,0x7016,0x7017,0x7018) if a.action=='csp-j7' else (0x700B,0x7028)
        original={hex(p):reader.read(mid,p) for p in params}
        result['original']=original
        limit=.5 if mid==1 else .15
        if a.torque_limit is not None:
            if not .1<=a.torque_limit<=.5:raise ValueError('Diagnostic torque limit must be 0.1..0.5 Nm')
            limit=a.torque_limit
        changed.append(0x700B);motor.set_volatile(0x700B,limit)
        result['temporary_torque_limit_Nm']=limit
        if a.action=='watchdog-j1':
            trials=[]
            for raw in (can_timeout_raw(100),can_timeout_raw(200)):
                changed.append(0x7028);motor.set_volatile(0x7028,raw)
                motor.mit(initial,0.,0.)
                reader.drain();motor.enable()
                # Constant 0.05 Nm, below the measured static-friction floor,
                # isolates timeout from a decaying position-controller error.
                # No commands or parameter reads follow during observation.
                motor.mit(initial,0.,0.,tau=.05)
                last_tx=motor.tx[-1]['ns']
                result['enabled_joints']=[1]
                states=[];end=time.monotonic()+.6
                while time.monotonic()<end:
                    reader.poll(.001)
                    st=reader.latest_status.get(mid)
                    if st and st['rx_ns']>=last_tx and (not states or st['rx_ns']!=states[-1]['rx_ns']):
                        states.append(dict(st))
                        if abs(st['position_u16']-pre['position_u16'])*8*math.pi/65535>math.radians(.2):
                            raise RuntimeError('Watchdog diagnostic position moved >0.2 deg')
                run=[s for s in states if s['mode_bits_raw']==2]
                reset=[s for s in states if s['mode_bits_raw']==0 and run and s['rx_ns']>run[0]['rx_ns']]
                trials.append({'raw':raw,'manual_timeout_ms':raw/20,
                    'last_tx_ns':last_tx,'states':states,'run_observed':bool(run),
                    'reset_after_ms':(reset[0]['rx_ns']-last_tx)/1e6 if reset else None})
                effort=[st for st in states if abs(st['torque_u16']-32767)>5]
                first_effort=effort[0]['rx_ns'] if effort else None
                quiet=[st for st in states if first_effort and st['rx_ns']>effort[-1]['rx_ns'] and st['torque_u16']==32767]
                trials[-1]['effort_observed']=bool(effort)
                trials[-1]['effort_zero_after_ms']=(quiet[0]['rx_ns']-last_tx)/1e6 if quiet else None
                motor.disable();reader.poll(.05)
            result['trials']=trials
            result['success']=all(t['effort_observed'] and t['effort_zero_after_ms'] is not None and
                t['manual_timeout_ms']-30<t['effort_zero_after_ms']<t['manual_timeout_ms']+60 for t in trials)
            result['timeout_guarantees_disabled_mode']=False
        else:
            changed.append(0x7028);motor.set_volatile(0x7028,can_timeout_raw(100))
            result['watchdog_raw']=2000
            if mid==1 and not -2.7<initial<2.7:raise RuntimeError('J1 near limit')
            if mid==7 and not -.02<initial<5.9:raise RuntimeError('J7 outside opening test envelope')
            # J7 has a much larger observed stationary speed-estimator noise.
            # Encoder-based 3 deg/s limits remain identical for both motors.
            raw_limit=3. if mid==1 else 12.
            guard=VelocityGuard(raw_limit_deg_s=raw_limit)
            result['encoder_speed_limit_deg_s']=3.
            result['raw_estimator_limit_deg_s']=raw_limit
            # Stationary preflight exercises the same guards before torque enable.
            for i in range(30):
                pos=reader.read(mid,0x7019,.02);vel=reader.read(mid,0x701B,.02)
                guard.update(time.monotonic_ns(),pos,vel);time.sleep(.008)
            kp,kd=(50.,3.) if mid==1 else (5.,.2)
            if a.kp is not None:kp=a.kp
            result['kp']=kp;result['kd']=kd
            if a.action=='csp-j7':
                for param,value in ((0x7018,.45),(0x7017,math.radians(.5)),(0x7016,initial),(0x7005,5)):
                    changed.append(param);motor.set_volatile(param,value)
                result['internal_velocity_limit_deg_s']=.5;result['current_limit_A']=.45
                result['CSP_effort_limited_by_current_not_MIT_torque_parameter']=True
            else:motor.mit(initial,kp,kd)
            # Keep cyclic GC out of a bounded active interval; refcounting remains.
            gc.collect();gc.disable();result['cyclic_gc_disabled_during_motion']=True
            motor.enable();result['enabled_joints']=[mid]
            # Seed the current-position MIT target once after enable. On this
            # firmware the first Run-mode status follows the first MIT command.
            motor.mit(initial,kp,kd)
            enable_deadline=time.monotonic()+1.0
            while True:
                reader.poll(.01)
                enabled_status=reader.latest_status.get(mid)
                if enabled_status and (time.monotonic_ns()-enabled_status['rx_ns'])/1e9<.08:
                    if enabled_status['fault_mask_raw']:
                        raise RuntimeError('Motor reported fault immediately after enable')
                    if enabled_status['temperature_C']>=45:
                        raise RuntimeError('Motor too warm for diagnostic')
                    if enabled_status['mode_bits_raw']==2:
                        result['enabled_status']=dict(enabled_status)
                        break
                if time.monotonic()>enable_deadline:
                    raise RuntimeError('No fresh Run-mode status after enable')
            start=time.monotonic();previous=start
            duration=12. if a.action=='csp-j7' else 6.
            cycle_duration=duration+2.5;result['requested_cycles']=a.repetitions
            reported_cycle=0
            for i in range(int(cycle_duration*a.repetitions*100)+1):
                time.sleep(max(0,start+i*.01-time.monotonic()))
                now=time.monotonic();elapsed=now-start;gap=now-previous;previous=now
                if gap>.05:raise RuntimeError('Control gap >50 ms')
                cycle_index=min(int(elapsed/cycle_duration),a.repetitions-1)
                cycle_elapsed=elapsed-cycle_index*cycle_duration
                # Every cycle uses the same initial reference, avoiding cumulative drift.
                amplitude=.5 if mid==1 else (2. if a.action=='csp-j7' else 5.)
                target=initial+math.radians(amplitude)*.5*(1-math.cos(2*math.pi*min(cycle_elapsed,duration)/duration))
                if a.action in ('move-j1','csp-j7'):
                    target,cycle_index,cycle_elapsed=repeated_diagnostic_target(mid,initial,elapsed,a.repetitions)
                if a.action=='csp-j7':motor.set_volatile(0x7016,target)
                else:motor.mit(target,kp,kd)
                read_start=time.monotonic_ns()
                pos=reader.read(mid,0x7019,.02);vel=reader.read(mid,0x701B,.02)
                volts=reader.read(mid,0x701C,.02);current=reader.read(mid,0x701A,.02)
                ns=time.monotonic_ns()
                row={'ns':ns,'elapsed_s':elapsed,'cycle':cycle_index+1,'cycle_elapsed_s':cycle_elapsed,
                     'command_rad':target,'actual_rad':pos,
                     'velocity_raw_rad_s':vel,'voltage_V':volts,'iq_A':current,'gap_s':gap,
                     'read_bundle_ms':(ns-read_start)/1e6}
                samples.append(row) # Preserve the offending sample before any check.
                observed=guard.update(ns,pos,vel);row['velocity_guard']=observed
                if not 42<=volts<=54:raise RuntimeError('Supply voltage outside envelope')
                deviation=1. if mid==1 else 7.
                if abs(math.degrees(pos-initial))>deviation or abs(math.degrees(pos-target))>deviation:
                    raise RuntimeError('Position outside diagnostic envelope')
                if mid==7 and abs(current)>.5:raise RuntimeError('Gripper current >0.5 A')
                st=motor.status();row['status']=dict(st)
                if st['fault_mask_raw']:raise RuntimeError('Motor reported fault')
                if st['temperature_C']>=45:raise RuntimeError('Motor temperature reached diagnostic limit')
                if elapsed>.08 and st['mode_bits_raw']!=2:raise RuntimeError('Motor left Run mode')
                if i%10==0:
                    for j in range(1,8):
                        if j!=mid and abs(reader.read(j,0x7019,.02)-baseline[j-1])>math.radians(.5):
                            raise RuntimeError('Uncommanded joint moved >0.5 deg')
                    if any(v['fault_mask_raw'] or (j!=mid and v['mode_bits_raw']!=0) for j,v in reader.latest_status.items()):
                        raise RuntimeError('Other joint fault or unexpected enable')
                if cycle_index>reported_cycle:
                    print(json.dumps({'action':a.action,'cycles_completed':cycle_index}),flush=True)
                    reported_cycle=cycle_index
            # Disabled firmware emits one acknowledgement, not periodic status.
            # Validate that acknowledgement against this stop's transmit time;
            # applying the Run-mode freshness limit here creates a false error.
            motor.disable();disable_ns=motor.tx[-1]['ns'];reader.poll(.1)
            stopped=reader.latest_status.get(mid)
            if not stopped or stopped['rx_ns']<disable_ns or stopped['mode_bits_raw']!=0:
                raise RuntimeError('Motor did not acknowledge disabled mode')
            result['disabled_status']=dict(stopped)
            maximum=max(math.degrees(s['actual_rad']-initial) for s in samples)
            final=math.degrees(samples[-1]['actual_rad']-initial)
            cycles=[]
            for cycle in range(1,a.repetitions+1):
                part=[s for s in samples if s['cycle']==cycle]
                peak=max(math.degrees(s['actual_rad']-initial) for s in part)
                error=math.degrees(part[-1]['actual_rad']-initial)
                passed=(.1<peak<.85 and abs(error)<.25) if mid==1 else (1.<peak<6.5 and abs(error)<1.)
                cycles.append({'cycle':cycle,'actual_amplitude_deg':peak,'final_error_deg':error,
                    'tracking_rmse_deg':math.sqrt(sum(math.degrees(s['actual_rad']-s['command_rad'])**2 for s in part)/len(part)),
                    'peak_abs_current_A':max(abs(s['iq_A']) for s in part),'passed':passed,
                    'samples':len(part)})
            result.update(actual_amplitude_deg=maximum,final_error_deg=final,samples=len(samples),cycles=cycles,
                completed_cycles=len(cycles),fixed_reference_position_rad=initial,
                isolated_raw_outliers=sum(s.get('velocity_guard',{}).get('isolated_raw_outlier',False) for s in samples),
                tracking_rmse_deg=math.sqrt(sum(math.degrees(s['actual_rad']-s['command_rad'])**2 for s in samples)/len(samples)),
                success=all(c['passed'] for c in cycles))
except BaseException as exc:result['error']=repr(exc)
finally:
    try:
        motor.disable();disable_ns=motor.tx[-1]['ns'];reader.poll(.1)
        stopped=reader.latest_status.get(mid)
        result['disabled_after_test']=bool(
            stopped and stopped['rx_ns']>=disable_ns and stopped['mode_bits_raw']==0)
        if stopped:result['final_disabled_status']=dict(stopped)
    except BaseException as exc:result['disable_error']=repr(exc)
    if gc_was_enabled:gc.enable()
    try:
        os.sched_setscheduler(0,scheduler_before,priority_before);os.sched_setaffinity(0,affinity_before)
        result['scheduler_restored']=True
    except BaseException as exc:result['scheduler_restore_error']=repr(exc)
    if 0x7016 in changed:
        changed=[p for p in changed if p!=0x7016]
        result['disabled_dynamic_loc_ref']='Firmware reseeds this target from encoder; historical target is intentionally not restored'
    for param in sorted(set(changed),key=lambda p:p==0x7005):
        try:motor.set_volatile(param,int(original[hex(param)]) if param in (0x7028,0x7005) else original[hex(param)])
        except BaseException as exc:result.setdefault('restore_errors',[]).append(repr(exc))
    try:
        reader.poll(.15);result['final_status']=reader.latest_status
    except BaseException as exc:result['final_status_error']=repr(exc)
    result['restored_parameters']={}
    for param in set(changed):
        try:result['restored_parameters'][hex(param)]=reader.read(mid,param)
        except BaseException as exc:result['restored_parameters'][hex(param)]=repr(exc)
    (out/'samples.json').write_text(json.dumps(samples))
    (out/'rx_frames.json').write_text(json.dumps(reader.capture))
    (out/'tx_frames.json').write_text(json.dumps(motor.tx))
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    reader.close();lock.close()
    print(json.dumps(result),flush=True)
raise SystemExit(0 if result['success'] and result.get('disabled_after_test') and not result.get('restore_errors') else 1)
