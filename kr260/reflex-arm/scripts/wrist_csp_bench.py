"""Bounded CSP joint tests, with explicit catch/brake prerequisite for J2-J4.

Never zeros, saves, clears faults, or writes firmware. Other joints retain
their existing enabled/disabled state and receive no writes. A loaded-axis
test first checks a stationary reference before its quarter-degree excursion.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
from statistics import median
import struct
import time

from reflex.canbus import ReadOnlyCAN
from reflex.motion import VelocityGuard
from reflex.wrist_mit import wrist_mit_frame
from reflex.native_pp_metrics import wrist_tracking_metrics

p=argparse.ArgumentParser()
p.add_argument('--joint',type=int,choices=(2,3,4,5,6,7),required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--load-support-verified',action='store_true')
p.add_argument('--j4-speed-kp',type=float,choices=(2.,4.),default=None)
p.add_argument('--loaded-speed-scale',type=float,choices=(.5,),default=None)
p.add_argument('--loaded-duration',type=int,choices=(12,30),default=12)
p.add_argument('--loaded-speed-limit',type=float,choices=(.25,.5),default=.25)
p.add_argument('--loaded-peak-hold',type=int,choices=(0,8),default=0)
p.add_argument('--loaded-return-hold',type=int,choices=(2,8),default=2)
p.add_argument('--wrist-duration',type=int,choices=(8,12,20,40,80,160,200),default=8)
p.add_argument('--wrist-amplitude',type=float,choices=(.25,.5,1.,2.,5.,10.,12.),default=.25)
p.add_argument('--wrist-direction',type=int,choices=(-1,1),default=1)
p.add_argument('--wrist-peak-hold',type=int,choices=(0,20),default=0)
p.add_argument('--wrist-return-hold',type=int,choices=(3,8),default=3)
p.add_argument('--announce-peak',action='store_true')
p.add_argument('--peak-settle-seconds',type=int,choices=(3,8),default=3)
p.add_argument('--j6-speed-kp',type=float,choices=(2.5,),default=None)
p.add_argument('--web-mit-profile',action='store_true')
p.add_argument('--web-mit-range-check',action='store_true')
p.add_argument('--mit-damping-comparison',action='store_true')
pp_group=p.add_mutually_exclusive_group()
pp_group.add_argument('--native-pp-profile',action='store_true')
pp_group.add_argument('--native-pp-calibration',action='store_true')
pp_group.add_argument('--loaded-native-pp-profile',action='store_true')
p.add_argument('--j5-torque-limit',type=float,choices=(.5,.6),default=.5)
p.add_argument('--j5-current',type=float,choices=(.6,.7),default=.6)
p.add_argument('--gripper-amplitude',type=float,choices=(.5,2.),default=2.)
p.add_argument('--gripper-direction',type=int,choices=(-1,1),default=1)
p.add_argument('--gripper-current',type=float,choices=(.45,.5),default=.45)
p.add_argument('--gripper-duration',type=int,choices=(12,30),default=12)
p.add_argument('--gripper-peak-hold',type=int,choices=(0,8),default=0)
p.add_argument('--gripper-return-hold',type=int,choices=(3,8),default=3)
p.add_argument('--repetitions',type=int,choices=(1,3,10),default=1)
p.add_argument('--execute-authorized-test',action='store_true',required=True)
a=p.parse_args();mid=a.joint
wrist_pp=a.native_pp_profile or a.native_pp_calibration
loaded_pp=a.loaded_native_pp_profile
pp=wrist_pp or loaded_pp
loaded=mid in (2,3,4)
if loaded and not a.load_support_verified:p.error('Loaded-axis test requires verified brake/catch support')
if mid!=4 and a.j4_speed_kp is not None:p.error('Temporary J4 tuning is restricted to J4')
if mid not in (2,3) and a.loaded_speed_scale is not None:p.error('Half speed gain is restricted to J2/J3')
if mid not in (2,3) and a.loaded_speed_limit!=.25:p.error('Speed-limit comparison is restricted to J2/J3')
if not loaded and a.loaded_duration!=12:p.error('Loaded duration is restricted to J2-J4')
if not loaded and (a.loaded_peak_hold or a.loaded_return_hold!=2):p.error('Loaded holds are restricted to J2-J4')
if mid not in (5,6) and a.wrist_duration!=8:p.error('Wrist duration is restricted to J5/J6')
if mid not in (5,6) and (a.wrist_amplitude!=.25 or a.wrist_direction!=1 or a.wrist_peak_hold or a.wrist_return_hold!=3 or a.announce_peak):
    p.error('Wrist calibration options are restricted to J5/J6')
if a.wrist_amplitude>.25 and a.wrist_duration<math.pi*a.wrist_amplitude/.2:
    p.error('Larger wrist profiles require reference peak speed <=0.2 deg/s')
if a.announce_peak and not a.wrist_peak_hold:p.error('Peak event requires a bounded hold')
if a.peak_settle_seconds!=3 and (mid not in (5,6) or not a.announce_peak):p.error('Longer peak settling requires wrist capture announcements')
if a.j6_speed_kp is not None and mid!=6:p.error('Reduced J6 speed gain is restricted to J6')
if a.web_mit_profile:
    allowed=(.5,1.,2.,5.,10.) if a.web_mit_range_check else (.25,)
    if mid not in (5,6) or a.wrist_amplitude not in allowed or a.j6_speed_kp is not None or a.j5_current!=.6 or a.j5_torque_limit!=.5:
        p.error('Web MIT baseline requires staged J5/J6 excursions, fixed web gains and original diagnostic rails')
    if a.wrist_amplitude>1 and a.repetitions!=1:p.error('Larger web MIT poses require one bounded cycle')
elif a.web_mit_range_check:p.error('Range characterization requires the explicit web MIT profile')
if a.mit_damping_comparison and (not a.web_mit_profile or mid!=6 or a.wrist_amplitude>2 or a.repetitions!=1):
    p.error('Damping comparison requires a single bounded J6 MIT profile <=2 degrees')
if a.mit_damping_comparison and a.wrist_amplitude>=1 and a.wrist_duration<40*a.wrist_amplitude:
    p.error('Damping comparison reference peak speed must be <=0.079 deg/s for 1..2 degrees')
if a.native_pp_profile and (mid not in (5,6) or a.web_mit_profile or a.mit_damping_comparison or a.j6_speed_kp is not None or
    a.wrist_amplitude not in (.25,.5,1.,2.) or a.repetitions!=1 or a.announce_peak and a.peak_settle_seconds!=3):
    p.error('Native PP diagnostic is wrist only, one <=2 degree profile, no gain tuning or image acquisition')
if a.native_pp_calibration and (mid not in (5,6) or a.web_mit_profile or a.mit_damping_comparison or a.j6_speed_kp is not None or
    a.wrist_amplitude not in (1.,2.,5.,10.) or a.repetitions!=1 or not a.announce_peak or a.peak_settle_seconds!=8):
    p.error('Native PP acquisition requires one staged wrist pose, no gain tuning, and the reviewed stationary capture window')
if loaded_pp and (mid!=3 or not a.load_support_verified or a.loaded_duration!=30 or
    a.loaded_peak_hold!=8 or a.loaded_return_hold!=8 or a.loaded_speed_limit!=.25 or
    a.loaded_speed_scale is not None or a.j4_speed_kp is not None or a.j6_speed_kp is not None or
    a.web_mit_profile or a.web_mit_range_check or a.mit_damping_comparison or a.repetitions!=1 or
    a.wrist_duration!=8 or a.wrist_amplitude!=.25 or a.wrist_direction!=1 or a.wrist_peak_hold or
    a.wrist_return_hold!=3 or a.announce_peak or a.peak_settle_seconds!=3 or
    a.j5_current!=.6 or a.j5_torque_limit!=.5):
    p.error('Supported loaded PP is J3 only: +0.25 degree, one 30/8/8 second cycle, no wrist/capture/gain options')
if wrist_pp and (a.j5_current!=.6 or a.j5_torque_limit!=.5):
    p.error('Native PP requires unchanged diagnostic effort limits: 0.6 A and 0.5 Nm')
if wrist_pp and a.wrist_duration<{.25:20,.5:20,1.:40,2.:80,5.:80,10.:160}[a.wrist_amplitude]:
    p.error('Native PP requires the reviewed minimum planning/settling duration')
if mid==6 and a.j6_speed_kp is None and not (a.web_mit_profile or pp):a.j6_speed_kp=2.5
if a.j5_torque_limit!=.5 and mid!=5:p.error('J5 torque comparison is restricted to J5')
if a.j5_current!=.6 and mid!=5:p.error('J5 current comparison is restricted to J5')
if mid!=7 and (a.gripper_amplitude!=2. or a.gripper_direction!=1 or a.gripper_current!=.45 or a.gripper_duration!=12):
    p.error('Gripper direction/amplitude options are restricted to J7')
if mid!=7 and (a.gripper_peak_hold or a.gripper_return_hold!=3):
    p.error('Gripper holds are restricted to J7')
amplitude=a.gripper_amplitude if mid==7 else (a.wrist_amplitude if mid in (5,6) else .25)
direction=a.gripper_direction if mid==7 else (a.wrist_direction if mid in (5,6) else 1)
a.out.mkdir(parents=True,exist_ok=False)
source_folder=a.out/'source';source_folder.mkdir()
source_hashes={}
for source_path in (Path(__file__),Path(__file__).resolve().parents[1]/'reflex/canbus.py',
                    Path(__file__).resolve().parents[1]/'reflex/motion.py',
                    Path(__file__).resolve().parents[1]/'reflex/wrist_mit.py',
                    Path(__file__).resolve().parents[1]/'reflex/native_pp_metrics.py'):
    payload=source_path.read_bytes();(source_folder/source_path.name).write_bytes(payload)
    source_hashes[source_path.name]=hashlib.sha256(payload).hexdigest()
lock_fd=os.open('/tmp/reflex-can-actuation.lock',os.O_RDWR|os.O_NOFOLLOW)
lock=os.fdopen(lock_fd,'r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
c=ReadOnlyCAN();c.capture=[];tx=[];samples=[];original={};changed=[]
result={'joint':mid,'amplitude_motor_deg':amplitude,'direction':direction,
        'full_gripper_stroke':False,'completed_profile':False,'passed':False,
        'no_other_joint_writes':True,'parameter_persistence_writes':0}
result['source_sha256']=source_hashes
result['control_mode']='MIT (web gains)' if a.web_mit_profile else 'CSP'
if pp:
    result.update(control_mode='PP (bounded native position)',native_pp_velocity_deg_s=.25,
        native_pp_acceleration_deg_s2=.25,gain_parameters_changed=False,
        qualification_scope='Bounded wrist PP; image acquisition is a separate stationary qualification; no grasp release',
        capture_qualification_requested=a.native_pp_calibration)
if loaded_pp:
    result.update(control_mode='PP (J3 supported diagnostic only)',identity_unverified=True,
        initial_all_joints_disabled_required=True,production_motion_release=False,handeye_pose_release=False,
        original_load_holding_control_restored=False,parameters_restored=False,
        qualification_scope='J3 +0.25 degree supported diagnostic; initial seven joints disabled; identity_unverified; no production or image release')
if a.web_mit_profile:
    result.update(mit_kp=30.,mit_kd=2. if a.mit_damping_comparison else 1.,requested_feedforward_torque_Nm=0.,
        web_comparison_scope='Same saved UI MIT gains; bounded reference, cadence and diagnostic torque/watchdog rails differ from manual clicking')
    if a.mit_damping_comparison:
        result['control_mode']='MIT (damping comparison)'
        result['web_comparison_scope']='J6 diagnostic damping comparison: KP=30, KD=2; not the saved web KD=1 baseline'
result.update(requested_repetitions=a.repetitions,fixed_anchor_between_cycles=True,cycle_results=[])
result['load_support_verified']=a.load_support_verified if loaded else None
result['tracking_criteria']={'amplitude_ratio_min':.8,'amplitude_ratio_max':1.2,
    'absolute_return_error_max_motor_deg':.05 if mid!=7 else max(.05,.1*amplitude),
    'tracking_rmse_max_motor_deg':.3*amplitude,
    'scope':'Predeclared bounded-profile engineering criteria; not TCP accuracy certification'}
if a.native_pp_calibration:
    result['tracking_criteria'].update(tracking_rmse_max_motor_deg=.05,
        scope='PP settled peak/return pose accuracy; transient goal-step response error is recorded separately, not a trajectory accuracy certificate')
was_enabled=None;baseline=None;phase='preflight';guard=None
old_gc=gc.isenabled();old_sched=os.sched_getscheduler(0)
old_priority=os.sched_getparam(0);old_affinity=os.sched_getaffinity(0)
if a.announce_peak or loaded_pp:print(json.dumps({'event':'BENCH_STARTED','pid':os.getpid()}),flush=True)


def interrupted(*_):raise KeyboardInterrupt('Operator/host interruption')
signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)


def command(kind,data=b'\0'*8):
    if kind not in (1,3,4,18):raise ValueError('Unapproved command kind')
    # Every command, including restoration, is scoped to this one target.
    frame=struct.pack('=IB3x8s',0x80000000|(kind<<24)|(0xFD<<8)|mid,8,data)
    c.sock.send(frame);tx.append({'ns':time.monotonic_ns(),'kind':kind,'joint':mid,'frame':frame.hex()})


def write(param,value):
    kinds={0x700B:'f',0x7028:'I',0x7018:'f',0x7017:'f',0x7016:'f',0x7005:'b'}
    if pp:kinds.update({0x7024:'f',0x7025:'f'})
    if a.j4_speed_kp is not None or a.loaded_speed_scale is not None or a.j6_speed_kp is not None:kinds[0x701F]='f'
    if param not in kinds:raise ValueError('Parameter not whitelisted')
    dtype=kinds[param]
    command(18,struct.pack('<H',param)+b'\0\0'+struct.pack('<'+dtype,value).ljust(4,b'\0'))
    actual=c.read(mid,param,.025)
    tolerance=.0005 if param==0x7016 else 1e-6
    if not math.isclose(actual,value,rel_tol=1e-5,abs_tol=tolerance):
        raise RuntimeError(f'Parameter {hex(param)} readback mismatch')


def send_mit(position,effort=True):
    if not a.web_mit_profile:raise ValueError('MIT frame outside requested web baseline')
    frame=wrist_mit_frame(mid,position,effort,damping_comparison=a.mit_damping_comparison)
    c.sock.send(frame);tx.append({'ns':time.monotonic_ns(),'kind':1,'joint':mid,'frame':frame.hex()})


def status():
    c.drain();s=c.latest_status.get(mid)
    if not s or time.monotonic_ns()-s['rx_ns']>80_000_000:raise RuntimeError('Target status stale')
    if s['fault_mask_raw'] or s['temperature_C']>=45:raise RuntimeError('Target fault or temperature limit')
    return s


try:
    os.sched_setaffinity(0,{3});os.sched_setscheduler(0,os.SCHED_FIFO,os.sched_param(10))
    c.poll(.2)
    if set(c.latest_status)!=set(range(1,8)):raise RuntimeError('Seven fresh motor states required')
    pre={k:dict(v) for k,v in c.latest_status.items()};result['pre_status']=pre
    if any(v['fault_mask_raw'] or v['mode_bits_raw'] not in (0,2) or v['temperature_C']>=45 for v in pre.values()):
        raise RuntimeError('Unexpected initial mode, temperature or fault')
    if loaded_pp and any(v['mode_bits_raw']!=0 for v in pre.values()):
        raise RuntimeError('Supported J3 PP requires all seven joints initially disabled')
    was_enabled=pre[mid]['mode_bits_raw']==2
    baseline=[c.read(j,0x7019,.03) for j in range(1,8)];anchor=baseline[mid-1]
    result['anchor_motor_rad']=anchor;result['baseline_all_motor_rad']=baseline
    if mid==7 and not -.005<anchor<5.5:raise RuntimeError('J7 outside small-stroke starting envelope')
    if mid!=7 and abs(anchor)>.05:raise RuntimeError('First joint test requires near-zero starting pose')
    if mid in (2,3) and not 0.<=anchor<=3.14-math.radians(amplitude):raise RuntimeError('Loaded joint target outside URDF motor-angle interval')
    for param in (0x700B,0x7028,0x7018,0x7017,0x7005):original[param]=c.read(mid,param,.03)
    if pp:
        for param in (0x7024,0x7025):
            original[param]=c.read(mid,param,.03)
            if original[param]<math.radians(.25):raise RuntimeError('PP test must not increase existing planning limits')
    if loaded_pp:
        for param in (0x701E,0x701F,0x7020,0x7021):original[param]=c.read(mid,param,.03)
        result['retained_torque_limit_Nm']=original[0x700B]
    if a.j4_speed_kp is not None:
        original[0x701F]=c.read(mid,0x701F,.03)
        if not 0<a.j4_speed_kp<=original[0x701F]:raise RuntimeError('J4 test may only reduce existing speed gain')
    if a.loaded_speed_scale is not None:
        original[0x701F]=c.read(mid,0x701F,.03)
        if not 0<original[0x701F]<=20:raise RuntimeError('Unexpected loaded-axis speed gain')
    if a.j6_speed_kp is not None:
        original[0x701F]=c.read(mid,0x701F,.03)
        if not 0<a.j6_speed_kp<=original[0x701F]<=5.:
            raise RuntimeError('J6 diagnostic may only reduce its verified speed gain')
    result['original_parameters']={hex(k):v for k,v in original.items()}
    if original[0x7005]!=0:raise RuntimeError('Expected existing MIT mode; no arbitrary mode takeover')
    guard=VelocityGuard(limit_deg_s=3.,raw_limit_deg_s=15.)
    for _ in range(80):
        q=c.read(mid,0x7019,.02);v=c.read(mid,0x701B,.02)
        guard.update(time.monotonic_ns(),q,v)
        if abs(q-anchor)>math.radians(.1):raise RuntimeError('Wrist drifting before control takeover')
        time.sleep(.008)
    phase='configuration';command(4);c.poll(.05)
    if status()['mode_bits_raw']!=0:raise RuntimeError('Target failed to disable for control preparation')
    # Disabling releases the previous effort. Re-sample the settled encoder,
    # rather than writing the stale enabled-position anchor into loc_ref.
    settled=[]
    for _ in range(12):
        settled.append(c.read(mid,0x7019,.02));time.sleep(.008)
    if max(settled)-min(settled)>math.radians(.06):raise RuntimeError('Target not stationary after disabling')
    new_anchor=median(settled)
    if abs(new_anchor-anchor)>math.radians(.2):raise RuntimeError('Release displacement exceeds 0.2 deg')
    result['enabled_to_disabled_release_motor_deg']=math.degrees(new_anchor-anchor)
    result['disabled_settling_range_motor_deg']=math.degrees(max(settled)-min(settled))
    anchor=new_anchor;result['anchor_motor_rad']=anchor
    if mid==7 and not 0.<=anchor+direction*math.radians(amplitude)<=5.5:
        raise RuntimeError('Proposed J7 excursion crosses conservative motor-position boundary')
    if mid in (5,6):
        limit=1.57 if mid==5 else 3.14
        if not -limit<=anchor+direction*math.radians(amplitude)<=limit:
            raise RuntimeError('Wrist target outside verified URDF joint interval')
    speed=.5 if mid==7 else (a.loaded_speed_limit if mid in (2,3) else .25)
    current=a.gripper_current if mid==7 else {2:5.,3:12.,4:3.,5:a.j5_current,6:.6}[mid]
    current_guard=.6 if mid==7 else (current+.5 if loaded else current+.15)
    if current>original[0x7018]:raise RuntimeError('Test current limit would exceed existing device limit')
    result.update(speed_limit_motor_deg_s=None if a.web_mit_profile else speed,
        current_limit_A=None if a.web_mit_profile else current,software_current_limit_A=current_guard)
    torque_limit=a.j5_torque_limit if mid==5 else .5
    if not loaded and torque_limit>original[0x700B]:raise RuntimeError('Test torque limit exceeds existing device limit')
    parameters=[] if loaded else [(0x700B,torque_limit)]
    if not loaded:result['torque_limit_Nm']=torque_limit
    if a.j4_speed_kp is not None:
        parameters.append((0x701F,a.j4_speed_kp));result['temporary_speed_kp']=a.j4_speed_kp
    if a.loaded_speed_scale is not None:
        reduced=original[0x701F]*a.loaded_speed_scale
        parameters.append((0x701F,reduced));result['temporary_speed_kp']=reduced
    if a.j6_speed_kp is not None:
        parameters.append((0x701F,a.j6_speed_kp));result['temporary_speed_kp']=a.j6_speed_kp
    if pp:
        parameters=[(0x7005,1)]+parameters+[(0x7028,2000),(0x7018,current),
            (0x7024,math.radians(.25)),(0x7025,math.radians(.25)),(0x7016,anchor)]
    elif a.web_mit_profile:parameters=parameters+[(0x7028,2000)]
    else:parameters=[(0x7005,5)]+parameters+[(0x7028,2000),(0x7018,current),(0x7017,math.radians(speed)),(0x7016,anchor)]
    for param,value in parameters:
        changed.append(param);write(param,value)
    if a.web_mit_profile:send_mit(anchor,effort=False)
    gc.collect();gc.disable();command(3)
    if a.web_mit_profile:send_mit(anchor)
    phase='enable_verification'
    # Verify the limits after mode entry and enable, before sending a moving reference.
    expected={param:value for param,value in parameters if param!=0x7016}
    if loaded_pp:
        expected.update({param:original[param] for param in (0x700B,0x701E,0x701F,0x7020,0x7021)})
    if a.web_mit_profile:expected[0x7005]=0
    actual={param:c.read(mid,param,.02) for param in expected}
    result['post_enable_parameters']={hex(k):v for k,v in actual.items()}
    if any(not math.isclose(actual[k],v,rel_tol=1e-5,abs_tol=1e-6) for k,v in expected.items()):
        raise RuntimeError('Safety configuration changed after enabling')
    if abs(math.degrees(c.read(mid,0x7019,.02)-anchor))>.1:
        raise RuntimeError('Stationary enable displacement >0.1 deg')
    phase='profile'
    start=time.monotonic();previous=start
    duration=float(a.gripper_duration) if mid==7 else (float(a.loaded_duration) if loaded else float(a.wrist_duration))
    initial_grace=1. if loaded else .25
    result['profile_duration_s']=duration
    peak_hold=float(a.loaded_peak_hold) if loaded else (float(a.gripper_peak_hold) if mid==7 else float(a.wrist_peak_hold))
    return_hold=float(a.loaded_return_hold) if loaded else (float(a.gripper_return_hold) if mid==7 else (float(a.wrist_return_hold) if a.wrist_return_hold==8 else 3.-initial_grace))
    result.update(peak_hold_s=peak_hold,return_hold_s=return_hold)
    guard=VelocityGuard(limit_deg_s=3.,raw_limit_deg_s=15.)
    # The optional bounded dwell separates response lag from static offset.
    # Neither dwell changes the excursion, gains, current or speed limits.
    total_duration=initial_grace+duration+peak_hold+return_hold
    cycle_steps=round(total_duration*100)
    cycle_samples=[];cycle_number=0;peak_announced=False
    def cycle_metrics(rows,number):
        peak=max(direction*math.degrees(s['actual_rad']-anchor) for s in rows)
        final=math.degrees(rows[-1]['actual_rad']-anchor)
        metrics=wrist_tracking_metrics(rows,stationary_pp=a.native_pp_calibration)
        rmse=metrics['tracking_rmse_motor_deg']
        criteria=result['tracking_criteria']
        passed=(criteria['amplitude_ratio_min']<=peak/amplitude<=criteria['amplitude_ratio_max']
            and abs(final)<=criteria['absolute_return_error_max_motor_deg']
            and rmse<=criteria['tracking_rmse_max_motor_deg'])
        return {'cycle':number,'actual_amplitude_motor_deg':peak,'final_error_motor_deg':final,
                **metrics,'passed':passed,'samples':len(rows)}
    for i in range(cycle_steps*a.repetitions+1):
        time.sleep(max(0,start+i*.01-time.monotonic()))
        now=time.monotonic();elapsed=now-start;gap=now-previous;previous=now
        if gap>.05:raise RuntimeError('Control freshness gap >50 ms')
        # One absolute reference is retained throughout the complete repetition
        # block; no per-cycle encoder re-anchoring hides cumulative return drift.
        move_elapsed=elapsed-cycle_number*total_duration-initial_grace
        if move_elapsed<duration/2:fraction=move_elapsed/duration
        elif move_elapsed<duration/2+peak_hold:fraction=.5
        else:fraction=(move_elapsed-peak_hold)/duration
        fraction=min(max(fraction,0.),1.)
        target=anchor+direction*math.radians(amplitude)*.5*(1-math.cos(2*math.pi*fraction))
        if pp:
            # The motor plans velocity/acceleration for a fixed target, rather
            # than resetting its position command to a new sine sample at 100 Hz.
            target=anchor+direction*math.radians(amplitude) if 0<=move_elapsed<duration/2+peak_hold else anchor
        if a.web_mit_profile:send_mit(target)
        else:write(0x7016,target)
        q=c.read(mid,0x7019,.02);position_reply=getattr(c,'last_reply',None)
        v=c.read(mid,0x701B,.02)
        volts=c.read(mid,0x701C,.02);iq=c.read(mid,0x701A,.02)
        ns=time.monotonic_ns();st=dict(status())
        row={'ns':ns,'elapsed_s':elapsed,'cycle':cycle_number,'gap_s':gap,'command_rad':target,
             'actual_rad':q,'raw_velocity_rad_s':v,'voltage_V':volts,'iq_A':iq,'status':st}
        if a.native_pp_calibration:
            if duration/2+8<=move_elapsed<duration/2+peak_hold:row['pp_settled_phase']='peak'
            elif move_elapsed>=duration+peak_hold:row['pp_settled_phase']='return'
        if position_reply is not None:row['position_reply_ns']=position_reply['rx_ns']
        samples.append(row)
        cycle_samples.append(row)
        if a.web_mit_profile:
            row['estimated_MIT_torque_Nm']=result['mit_kp']*(target-q)-result['mit_kd']*v
            if abs(row['estimated_MIT_torque_Nm'])>.5:
                raise RuntimeError('Web MIT estimated effort exceeds diagnostic 0.5 Nm envelope')
        row['velocity_guard']=guard.update(ns,q,v)
        if not 42<=volts<=54 or abs(iq)>current_guard:raise RuntimeError('Supply or current outside envelope')
        if loaded and elapsed<initial_grace and abs(math.degrees(q-anchor))>.1:
            raise RuntimeError('Loaded-axis stationary hold drift >0.1 deg')
        envelope=amplitude+2. if mid==7 else (max(.65,amplitude+.4) if mid in (5,6) else .65)
        if abs(math.degrees(q-anchor))>envelope or abs(math.degrees(q-target))>envelope:
            raise RuntimeError('Position outside bounded test envelope')
        if elapsed>.1 and st['mode_bits_raw']!=2:raise RuntimeError('Target left Run mode')
        if i%5==0:
            neighbours={j:c.read(j,0x7019,.02) for j in range(1,8) if j!=mid}
            row['neighbours_rad']=neighbours
            if any(abs(qj-baseline[j-1])>math.radians(.3) for j,qj in neighbours.items()):
                raise RuntimeError('Uncommanded joint displacement >0.3 deg')
            for j,s in c.latest_status.items():
                if s['fault_mask_raw'] or (j!=mid and s['mode_bits_raw']!=pre[j]['mode_bits_raw']):
                    raise RuntimeError('Other joint fault or unexpected mode transition')
        if a.native_pp_calibration and i>0 and i%3000==0:
            segment='outbound' if move_elapsed<duration/2 else ('peak_hold' if move_elapsed<duration/2+peak_hold else 'return')
            print(json.dumps({'event':'PROFILE_PROGRESS','joint':mid,'elapsed_s':elapsed,'segment':segment,
                'measured_delta_deg':math.degrees(q-anchor),'goal_delta_deg':math.degrees(target-anchor),
                'iq_A':iq,'fault_mask_raw':st['fault_mask_raw']}),flush=True)
        if a.announce_peak and not peak_announced and move_elapsed>=duration/2+a.peak_settle_seconds:
            if move_elapsed>=duration/2+peak_hold:raise RuntimeError('Missed peak hold capture window')
            event={'event':'PEAK_HOLD_READY','joint':mid,'ns':ns,
                'target_rad':target,'remaining_hold_s':duration/2+peak_hold-move_elapsed}
            result['peak_capture_event']=event
            print(json.dumps(event),flush=True)
            peak_announced=True
        if i>0 and i%cycle_steps==0:
            metric=cycle_metrics(cycle_samples,cycle_number)
            result['cycle_results'].append(metric)
            if a.announce_peak:print(json.dumps({'event':'CYCLE_COMPLETE',**metric}),flush=True)
            if not metric['passed'] or cycle_number+1>=a.repetitions:break
            cycle_samples=[];cycle_number+=1
    command(4);c.poll(.08)
    if status()['mode_bits_raw']!=0:raise RuntimeError('Target failed to disable at end of profile')
    peak=max(direction*math.degrees(s['actual_rad']-anchor) for s in samples)
    final=math.degrees(samples[-1]['actual_rad']-anchor)
    if mid==7:
        passed=(1.<peak<3.0 and abs(final)<.8) if amplitude==2. else (.1<peak<.85 and abs(final)<.25)
    else:passed=.05<peak<.5 and abs(final)<.15
    metrics=wrist_tracking_metrics(samples,stationary_pp=a.native_pp_calibration)
    rmse=metrics['tracking_rmse_motor_deg']
    criteria=result['tracking_criteria']
    tracking_ok=(criteria['amplitude_ratio_min']<=peak/amplitude<=criteria['amplitude_ratio_max']
                 and abs(final)<=criteria['absolute_return_error_max_motor_deg']
                 and rmse<=criteria['tracking_rmse_max_motor_deg'])
    all_cycles_ok=(len(result['cycle_results'])==a.repetitions and all(r['passed'] for r in result['cycle_results']))
    result.update(completed_profile=True,passed=tracking_ok and all_cycles_ok,movement_diagnostic_passed=passed,
                  tracking_criteria_passed=tracking_ok and all_cycles_ok,actual_amplitude_motor_deg=peak,
                  final_error_motor_deg=final,
                  tracking_rmse_motor_deg=rmse,
                  peak_abs_current_A=max(abs(s['iq_A']) for s in samples),
                  maximum_feedback_gap_ms=max(s['gap_s'] for s in samples)*1000)
    result.update(metrics)
except BaseException as exc:
    result['error']=repr(exc);result['failure_phase']=phase
    if guard is not None and guard.trip_evidence is not None:
        result['guard_stop_evidence']=guard.trip_evidence
finally:
    if changed:
        try:
            command(4);c.poll(.08)
            if status()['mode_bits_raw']!=0:raise RuntimeError('Cannot verify disabled target for restoration')
            for param in (0x7018,0x7017,0x701F,0x7024,0x7025,0x700B,0x7028,0x7005):
                if param in changed:
                    write(param,int(original[param]) if param in (0x7028,0x7005) else original[param])
            if loaded_pp:
                retained={param:c.read(mid,param,.025) for param in (0x700B,0x701E,0x701F,0x7020,0x7021)}
                result['restored_retained_parameters']={hex(k):v for k,v in retained.items()}
                result['gain_and_torque_readback_unchanged']=all(
                    math.isclose(retained[k],original[k],rel_tol=1e-5,abs_tol=1e-6) for k in retained)
                if not result['gain_and_torque_readback_unchanged']:
                    raise RuntimeError('Supported J3 PP retained gain/torque readback changed; no automatic gain write allowed')
            result['parameters_restored']=True
            # Firmware reseeds loc_ref while disabled. Never restore an old target.
            # Return to the operator's initial Run/Reset state only after a completed
            # bounded profile, with a zero-effort MIT seed and no new position move.
            if result['completed_profile'] and was_enabled and not (a.web_mit_profile or pp):
                q=c.read(mid,0x7019,.02)
                enc=int((q+4*math.pi)/(8*math.pi)*65535)
                payload=struct.pack('>4H',enc,32767,0,0)
                frame=struct.pack('=IB3x8s',0x80000000|(1<<24)|(32767<<8)|mid,8,payload)
                c.sock.send(frame);tx.append({'ns':time.monotonic_ns(),'kind':1,'joint':mid,'frame':frame.hex(),'restoration':'zero_effort_MIT'})
                command(3);c.poll(.08)
            result['initial_enable_state_restored']=(status()['mode_bits_raw']==(2 if was_enabled else 0))
            if a.web_mit_profile or pp:
                result['target_deliberately_left_disabled']=status()['mode_bits_raw']==0
        except BaseException as exc:
            result['restoration_error']=repr(exc)
            if loaded_pp:result['parameters_restored']=False
            try:command(4)
            except BaseException:pass
    if loaded_pp:
        try:
            result['target_deliberately_left_disabled']=status()['mode_bits_raw']==0
            if not result['target_deliberately_left_disabled']:
                raise RuntimeError('Supported J3 PP did not leave the target disabled')
        except BaseException as exc:
            result['final_disabled_verification_error']=repr(exc);result['passed']=False
    if old_gc:gc.enable()
    os.sched_setscheduler(0,old_sched,old_priority);os.sched_setaffinity(0,old_affinity)
    try:c.poll(.2);result['final_status']={k:dict(v) for k,v in c.latest_status.items()}
    except BaseException as exc:result['final_status_error']=repr(exc)
    result['samples']=len(samples);result['commands_sent_by_test']=len(tx)
    (a.out/'samples.json').write_text(json.dumps(samples))
    (a.out/'tx_frames.json').write_text(json.dumps(tx))
    (a.out/'rx_frames.json').write_text(json.dumps(c.capture))
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))
    c.close();lock.close();print(json.dumps(result),flush=True)
raise SystemExit(0 if result['passed'] and result.get('parameters_restored') and not result.get('restoration_error') else 1)
