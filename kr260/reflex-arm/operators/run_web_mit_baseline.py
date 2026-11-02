"""One authorized wrist trial with two observation recordings; no calibration release."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'.tools'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paramiko
from reflex.native_pp_release import resolve_pp_prerequisite,verified_pp_trial


def main():
    p=argparse.ArgumentParser();p.add_argument('--joint',type=int,choices=(5,6),required=True)
    p.add_argument('--direction',type=int,choices=(-1,1),required=True)
    p.add_argument('--name',required=True);p.add_argument('--repetitions',type=int,choices=(1,3,10),default=1)
    p.add_argument('--amplitude',type=float,choices=(.25,.5,1.,2.,5.,10.),default=.25)
    p.add_argument('--capture-actual-pose',action='store_true')
    p.add_argument('--slow-2deg-diagnostic',action='store_true',
        help='One J6 positive 2-degree trial with reference speed/acceleration halved/quartered; no gain or guard change')
    p.add_argument('--mit-damping-comparison',action='store_true')
    p.add_argument('--native-pp-diagnostic',action='store_true')
    p.add_argument('--native-pp-capture',action='store_true')
    p.add_argument('--site-fixed-target-confirmed',action='store_true')
    p.add_argument('--execute-authorized-test',action='store_true',required=True);a=p.parse_args()
    pp=a.native_pp_diagnostic or a.native_pp_capture
    if not re.fullmatch('[a-z][a-z0-9_]{0,48}',a.name):p.error('Invalid case name')
    if a.capture_actual_pose and (a.amplitude<1 or a.repetitions!=1):p.error('Actual-pose capture requires one qualified 1..10 degree staged profile')
    if a.capture_actual_pose and not a.site_fixed_target_confirmed:p.error('Actual-pose acquisition requires confirmed fixed target and camera mount')
    if a.amplitude>1 and a.repetitions!=1:p.error('Larger measured-pose profiles use one bounded cycle only')
    if a.slow_2deg_diagnostic and (a.joint!=6 or a.direction!=1 or a.amplitude!=2 or a.repetitions!=1):
        p.error('Slow diagnostic is restricted to one positive J6 2-degree profile')
    if a.mit_damping_comparison and (a.joint!=6 or a.amplitude>2 or a.repetitions!=1 or a.capture_actual_pose):
        p.error('Damping comparison is diagnostic only: J6, one cycle, <=2 degrees; no calibration release')
    if a.native_pp_diagnostic and (a.amplitude>2 or a.repetitions!=1 or a.capture_actual_pose or
        a.slow_2deg_diagnostic or a.mit_damping_comparison or a.native_pp_capture):
        p.error('Native PP diagnostic is one wrist cycle <=2 degrees, without calibration or gain changes')
    if a.native_pp_capture and (a.amplitude<1 or a.repetitions!=1 or not a.capture_actual_pose or
        a.slow_2deg_diagnostic or a.mit_damping_comparison):
        p.error('Native PP collection requires a staged, explicitly requested stationary image pair')
    local=Path(__file__).resolve().parents[2]/'artifacts/web_baseline_20261020'/a.name
    local.mkdir(parents=True,exist_ok=False)
    clients=[];robot=None;ai=None;gateway_stopped=False;bench_pid=None;record_started=False
    record_channel=None;bench_channel=None
    bench_path='/home/ubuntu/reflex-arm/runs/web_baseline_20261020/'+a.name
    record_path='/home/aup/reflex-arm/runs/web_baseline_20261020/'+a.name
    result={'success':False,'case':a.name,'joint':a.joint,'direction':a.direction,
        'requested_amplitude_deg':a.amplitude,'repetitions':a.repetitions,'hardware_synchronized':False,
        'client_started':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'cross_device_timestamps_subtracted':False,'camera_extrinsic_calibrated':False}
    result['operator_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (local/'operator_source.py').write_bytes(Path(__file__).read_bytes())
    if pp:
        validator=Path(__file__).resolve().parents[1]/'reflex/native_pp_release.py'
        (local/'release_validator_source.py').write_bytes(validator.read_bytes())
        result['release_validator_sha256']=hashlib.sha256(validator.read_bytes()).hexdigest()
    session_id='eih_pp_20261020' if a.native_pp_capture else 'eih_mit_actual_20261020'
    result['capture_session']=session_id if a.capture_actual_pose else None
    capture_path='/home/aup/reflex-arm/runs/calibration/'+session_id+'/'+a.name
    result['actual_pose_acquisition_passed']=False
    capture_attempted=False
    bench_log=[];record_log=[]
    def connect(host,user,key):
        c=paramiko.SSHClient();c.load_host_keys(str(Path.home()/'.ssh/known_hosts'));c.set_missing_host_key_policy(paramiko.RejectPolicy())
        c.connect(host,username=user,password=os.environ[key],look_for_keys=False,allow_agent=False,timeout=15)
        clients.append(c);return c
    def run_robot(command,background=False):
        stdin,stdout,stderr=robot.exec_command('sudo -S -p "" bash -c '+shlex.quote(command),timeout=650)
        stdin.write(os.environ['REFLEX_ROBOT_PASSWORD']+'\n');stdin.flush();stdin.channel.shutdown_write()
        if background:return stdout.channel
        out=stdout.read().decode();err=stderr.read().decode()
        if stdout.channel.recv_exit_status():raise RuntimeError('Remote action failed: '+err[-300:]+out[-300:])
        return out
    def load(client,path):
        with client.open_sftp() as s, s.open(path) as f:return json.loads(f.read())
    def absent(client,path):
        with client.open_sftp() as s:
            try:s.stat(path)
            except FileNotFoundError:return
        raise RuntimeError('Existing remote evidence: '+path)
    def progress(phase,message):
        path='/home/aup/reflex-arm/runs/calibration_progress.json'
        value=load(ai,path);value.update(phase=phase,message=message)
        with ai.open_sftp() as s:
            with s.open(path+'.baseline-tmp','w') as f:f.write(json.dumps(value,ensure_ascii=True,indent=2))
            s.posix_rename(path+'.baseline-tmp',path)
    def consume(channel,lines,buffer):
        nonlocal bench_pid
        if channel.recv_stderr_ready():lines.append(channel.recv_stderr(65536).decode('utf-8','replace'))
        if channel.recv_ready():buffer+=channel.recv(65536)
        messages=[]
        while b'\n' in buffer:
            line,buffer=buffer.split(b'\n',1);text=line.decode('utf-8','replace');lines.append(text)
            try:message=json.loads(text)
            except ValueError:continue
            messages.append(message)
            if message.get('event')=='BENCH_STARTED':bench_pid=int(message['pid'])
        return buffer,messages
    try:
        robot=connect('172.21.16.34','ubuntu','REFLEX_ROBOT_PASSWORD');ai=connect('172.21.16.96','aup','REFLEX_AI_PASSWORD')
        absent(robot,bench_path);absent(ai,record_path)
        guard_sha=hashlib.sha256((Path(__file__).resolve().parents[1]/'reflex/motion.py').read_bytes()).hexdigest()
        if pp and (a.native_pp_capture or a.joint==5):
            result['strict_j6_pp_prerequisites']=[]
            for amp,suffix in ((.25,'pilot'),(.5,'range_0p5'),(1.,'range_1'),(2.,'range_2')):
                for direction in (1,-1):
                    path=f'/home/ubuntu/reflex-arm/runs/web_baseline_20261020/pp_j6_{"pos" if direction>0 else "neg"}_{suffix}/summary.json'
                    verified_pp_trial(load(robot,path),6,amp,direction,guard_sha)
                    result['strict_j6_pp_prerequisites'].append(path)
        if a.amplitude>.25:
            # This is range characterization at unchanged effort rails, not a
            # claim that a completed but undertracking micro-profile passed.
            proofs=[]
            suffixes=['pilot']
            if a.amplitude>=1:suffixes.append('range_0p5')
            if a.amplitude>=2 or a.capture_actual_pose and not pp:suffixes.append('range_1')
            if a.amplitude>=5:suffixes.append('range_2')
            if a.amplitude>=10:suffixes.append('range_5')
            for suffix in suffixes:
                for direction in ('pos','neg'):
                    if pp:
                        path,proof=resolve_pp_prerequisite(lambda path:load(robot,path),
                            '/home/ubuntu/reflex-arm/runs/web_baseline_20261020',a.joint,
                            {'pilot':.25,'range_0p5':.5,'range_1':1.,'range_2':2.,'range_5':5.}[suffix],
                            1 if direction=='pos' else -1,guard_sha)
                        valid=True
                    else:
                        prefix='damp_j6' if a.mit_damping_comparison else f'mit_j{a.joint}'
                        path=f'/home/ubuntu/reflex-arm/runs/web_baseline_20261020/{prefix}_{direction}_{suffix}/summary.json'
                        proof=load(robot,path)
                        valid=(proof.get('completed_profile') and not proof.get('error') and
                            proof.get('parameters_restored') and proof.get('target_deliberately_left_disabled') and
                            proof.get('control_mode')==('MIT (damping comparison)' if a.mit_damping_comparison else 'MIT (web gains)') and
                            proof.get('mit_kp')==30. and proof.get('mit_kd')==(2. if a.mit_damping_comparison else 1.))
                    if not valid:
                        raise RuntimeError('Smaller completed guard-clear MIT trials required before range characterization')
                    proofs.append({'path':path,'tracking_passed':proof['passed']})
            result['range_characterization_prerequisites']=proofs
            result['qualification_scope']='Amplitude response only; unchanged effort rails; no calibration or grasp release'
        # Pin this run to the locally reviewed bench; the bench also saves its
        # actual runtime sources and hashes before sending any motor command.
        result['reviewed_source_sha256']={}
        for relative in ('scripts/wrist_csp_bench.py','reflex/canbus.py','reflex/motion.py','reflex/wrist_mit.py','reflex/native_pp_metrics.py'):
            expected=(Path(__file__).resolve().parents[1]/relative).read_bytes()
            with robot.open_sftp() as s,s.open('/home/ubuntu/reflex-arm/'+relative) as f:
                if hashlib.sha256(f.read()).digest()!=hashlib.sha256(expected).digest():raise RuntimeError('Remote reviewed source differs: '+relative)
            result['reviewed_source_sha256'][relative]=hashlib.sha256(expected).hexdigest()
        if a.capture_actual_pose:absent(ai,capture_path)
        _,stdout,_=ai.exec_command('cd /home/aup/reflex-arm && .venv/bin/python -u scripts/record_dual_mjpeg.py --out '+shlex.quote(record_path)+' --max-seconds '+('600' if a.repetitions>1 or a.amplitude>=2 else '120'),timeout=650)
        record_channel=stdout.channel;record_started=True;buffer=b'';deadline=time.monotonic()+12;ready=False
        while time.monotonic()<deadline:
            buffer,messages=consume(record_channel,record_log,buffer)
            if any(m.get('event')=='RECORDING_READY' for m in messages):ready=True;break
            if record_channel.exit_status_ready():break
            time.sleep(.02)
        if not ready:raise RuntimeError('Both recording streams must be ready before motion')
        result['recorder_ready_client_monotonic_ns']=time.monotonic_ns()
        label='阻尼对照 KP30 / KD2' if a.mit_damping_comparison else '网页 MIT 增益 KP30 / KD1'
        if pp:label='固件 PP 规划（速度 0.25°/秒，加速度 0.25°/秒²，增益不变）'
        progress('baseline_recording','正在录制 '+a.name+f'；使用{label}进行 {a.amplitude:g}°有保护对照。手眼外参尚未完成。')
        run_robot('systemctl stop reflex-motorbridge-gateway.service');gateway_stopped=True
        duration={.25:12,.5:12,1.:20,2.:40,5.:80,10.:160}[a.amplitude]
        if a.slow_2deg_diagnostic:duration=80
        if a.mit_damping_comparison and a.amplitude>=1:duration=int(40*a.amplitude)
        if pp:duration={.25:20,.5:20,1.:40,2.:80,5.:80,10.:160}[a.amplitude]
        result['reference_duration_s']=duration
        result['reference_peak_speed_deg_s']=3.141592653589793*a.amplitude/duration
        result['control_variant']='slower reference only; same MIT gains and all guard thresholds' if a.slow_2deg_diagnostic else 'staged reference'
        if a.mit_damping_comparison:result['control_variant']='J6 KP30 / KD2 damping comparison; all guard thresholds unchanged'
        if pp:
            result['control_variant']='PP: native speed/acceleration planning; no gain change or guard relaxation'
            result['reference_peak_speed_deg_s']=None
        cmd=('cd /home/ubuntu/reflex-arm && PYTHONPATH=. python3 scripts/wrist_csp_bench.py '
            f'--joint {a.joint} --out {shlex.quote(bench_path)} '+('--native-pp-calibration ' if a.native_pp_capture else '--native-pp-profile ' if a.native_pp_diagnostic else '--web-mit-profile ')+
            f'--wrist-amplitude {a.amplitude} '
            f'--wrist-direction {a.direction} --wrist-duration {duration} --wrist-peak-hold 20 --wrist-return-hold 8 '
            f'--repetitions {a.repetitions} --announce-peak --execute-authorized-test')
        if a.amplitude>.25 and not pp:cmd+=' --web-mit-range-check'
        if a.mit_damping_comparison:cmd+=' --mit-damping-comparison'
        if a.capture_actual_pose:cmd+=' --peak-settle-seconds 8'
        result['bench_launch_client_monotonic_ns']=time.monotonic_ns()
        bench_channel=run_robot(cmd,True);bench_buffer=b'';deadline=time.monotonic()+(duration+29)*a.repetitions+40
        while time.monotonic()<deadline:
            bench_buffer,messages=consume(bench_channel,bench_log,bench_buffer)
            for message in messages:
                if message.get('event') in ('BENCH_STARTED','CYCLE_COMPLETE','PROFILE_PROGRESS'):print(json.dumps(message),flush=True)
                if a.capture_actual_pose and message.get('event')=='PEAK_HOLD_READY' and not capture_attempted:
                    capture_attempted=True
                    try:
                        proc=subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'capture_handeye_pair.py'),
                            '--session',session_id,'--sample',a.name,
                            '--allow-enabled-hold','--monitor-seconds','6'],capture_output=True,text=True,timeout=35)
                        (local/'capture.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
                        result['capture_exit']=proc.returncode
                    except Exception as exc:result['capture_error']=repr(exc)
            if bench_channel.exit_status_ready() and not bench_channel.recv_ready():break
            if record_channel.exit_status_ready():raise RuntimeError('Observation recorder exited during the owned bench')
            time.sleep(.02)
        else:raise TimeoutError('Owned wrist bench deadline exceeded')
        result['bench_exit']=bench_channel.recv_exit_status();bench_pid=None
        if bench_buffer:bench_log.append(bench_buffer.decode('utf-8','replace'))
        result['bench_finished_client_monotonic_ns']=time.monotonic_ns()
        result['bench']=load(robot,bench_path+'/summary.json')
        if a.capture_actual_pose and result.get('capture_exit')==0:
            try:
                with ai.open_sftp() as s:
                    with s.open(capture_path+'/bench_summary.json','w') as f:f.write(json.dumps(result['bench'],indent=2))
                _,stdout,stderr=ai.exec_command('cd /home/aup/reflex-arm && PYTHONPATH=. .venv/bin/python '
                    'scripts/qualify_measured_handeye_pair.py --capture '+shlex.quote(capture_path),timeout=15)
                output=stdout.read().decode();error=stderr.read().decode()
                if stdout.channel.recv_exit_status():raise RuntimeError(error[-500:]+output[-500:])
                result['actual_pose_acquisition_passed']=True
            except Exception as exc:result['capture_qualification_error']=repr(exc)
    except BaseException as exc:result['error']=repr(exc)
    finally:
        if robot is not None:
            if bench_pid is not None:
                try:
                    run_robot('kill -TERM '+str(bench_pid))
                    deadline=time.monotonic()+10
                    while bench_channel is not None and not bench_channel.exit_status_ready() and time.monotonic()<deadline:
                        _,_=consume(bench_channel,bench_log,b'');time.sleep(.05)
                except Exception as exc:result['bench_interrupt_error']=repr(exc)
            if gateway_stopped:
                try:run_robot('systemctl start reflex-motorbridge-gateway.service');result['gateway_restarted']=True
                except Exception as exc:result['gateway_restart_error']=repr(exc)
        if record_started:
            try:
                with ai.open_sftp() as s,s.open(record_path+'/STOP','w') as f:f.write('owner requested stop\n')
                deadline=time.monotonic()+35;buffer=b''
                while time.monotonic()<deadline:
                    buffer,_=consume(record_channel,record_log,buffer)
                    if record_channel.exit_status_ready() and not record_channel.recv_ready():break
                    time.sleep(.03)
                result['recording']=load(ai,record_path+'/summary.json')
                progress('baseline_recorded','已保存观察录像：'+a.name+'。正在核对跟踪与回位数据；手眼外参尚未完成。')
            except Exception as exc:result['recording_error']=repr(exc)
        bench=result.get('bench',{});recording=result.get('recording',{})
        result['success']=bool(bench.get('passed') and bench.get('parameters_restored') and
            bench.get('target_deliberately_left_disabled') and recording.get('success') and not result.get('error'))
        result['completed_profile_without_guard_stop']=bool(bench.get('completed_profile') and not bench.get('error'))
        (local/'result.json').write_text(json.dumps(result,indent=2));(local/'bench.log').write_text('\n'.join(bench_log));(local/'recording.log').write_text('\n'.join(record_log))
        for c in clients:c.close()
        print(json.dumps({'case':a.name,'success':result['success'],'actual_pose_qualified':result['actual_pose_acquisition_passed'],'completed':bench.get('completed_profile'),
            'peak_deg':bench.get('actual_amplitude_motor_deg'),'return_error_deg':bench.get('final_error_motor_deg'),
            'rmse_deg':bench.get('tracking_rmse_motor_deg'),'error':result.get('error') or bench.get('error')}),flush=True)
    if not (result['actual_pose_acquisition_passed'] if a.capture_actual_pose else result['success']):raise SystemExit(1)


if __name__=='__main__':main()
