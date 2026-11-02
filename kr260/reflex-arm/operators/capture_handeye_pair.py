"""Windows/local SSH coordinator: bounded read-only robot/RGB-D capture.

Credentials are transient environment variables REFLEX_AI_PASSWORD and
REFLEX_ROBOT_PASSWORD. No secrets are written or sent in shell command text.
Use a new sample ID for every attempt, including rejected captures.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time

task_deps=Path(__file__).resolve().parents[2]/'.tools'
if task_deps.exists():sys.path.insert(0,str(task_deps))
import paramiko

p=argparse.ArgumentParser()
p.add_argument('--session',default='eih_session_01')
p.add_argument('--sample',default=datetime.now().strftime('sample_%Y%m%d_%H%M%S'))
a=p.parse_args()
if not all(re.fullmatch(r'[A-Za-z0-9_-]{1,64}',x) for x in (a.session,a.sample)):
    p.error('Session/sample names must be short alphanumeric identifiers')
ai_root='/home/aup/reflex-arm';robot_root='/home/ubuntu/reflex-arm'
capture=f'runs/calibration/{a.session}/{a.sample}'
monitor=f'runs/calibration_pose_windows/{a.session}/{a.sample}'
clients=[];result={'pair_accepted':False,'camera_extrinsic_calibrated':False}

def connect(host,user,password_env):
    c=paramiko.SSHClient();c.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect(host,username=user,password=os.environ[password_env],look_for_keys=False,allow_agent=False,timeout=15)
    clients.append(c);return c

def run(c,command):
    _,out,err=c.exec_command(command,timeout=60)
    text=out.read().decode('utf-8','replace');error=err.read().decode('utf-8','replace')
    return out.channel.recv_exit_status(),text,error

try:
    ai=connect('172.21.16.96','aup','REFLEX_AI_PASSWORD')
    robot=connect('172.21.16.34','ubuntu','REFLEX_ROBOT_PASSWORD')
    with ai.open_sftp() as sftp:
        try:sftp.stat(ai_root+'/'+capture)
        except FileNotFoundError:pass
        else:raise RuntimeError('Capture folder exists; choose a new sample ID')
    cmd=f'cd {robot_root} && PYTHONPATH=. python3 scripts/monitor_calibration_pose.py --seconds 12 --out {shlex.quote(monitor)}'
    _,mon,monerr=robot.exec_command(cmd,timeout=25)
    mon.channel.settimeout(5.)
    if mon.readline().strip()!='READY':raise RuntimeError('Robot monitor did not become ready')
    ready=time.monotonic_ns();print('Stationary robot monitor ready; capturing RGB-D.',flush=True)
    launched=time.monotonic_ns()
    code,output,error=run(ai,f'cd {ai_root} && .venv/bin/python scripts/orbbec_probe.py --out {shlex.quote(capture)}')
    completed=time.monotonic_ns();within=not mon.channel.exit_status_ready()
    mon.channel.settimeout(20.)
    monitor_output=mon.read();monitor_error=monerr.read()
    monitor_code=mon.channel.recv_exit_status()
    window={'coordinator_clock':'local_monotonic_only','monitor_ready_ns':ready,
        'camera_launch_ns':launched,'camera_completed_ns':completed,
        'camera_started_after_monitor_ready':launched>=ready,
        'camera_completed_before_monitor_end':within,
        'cross_device_timestamps_subtracted':False,'robot_monitor_exit':monitor_code,
        'camera_capture_exit':code}
    if code:raise RuntimeError('RGB-D capture failed: '+output[-500:]+error[-500:])
    with robot.open_sftp() as source,ai.open_sftp() as target:
        for filename in ('summary.json','samples.json'):
            with source.open(robot_root+'/'+monitor+'/'+filename,'r') as f:payload=f.read()
            with target.open(ai_root+'/'+capture+'/robot_'+filename,'w') as f:f.write(payload)
        with target.open(ai_root+'/'+capture+'/stationary_window.json','w') as f:f.write(json.dumps(window))
    print('RGB-D and robot window saved; checking the printed target.',flush=True)
    run(ai,f'cd {ai_root} && PYTHONPATH=. .venv/bin/python scripts/analyze_calibration_board.py '
        f'--image {shlex.quote(capture+"/color.jpg")} --factory {shlex.quote(capture+"/summary.json")} '
        f'--out {shlex.quote(capture+"/board_detection")}')
    run(ai,f'cd {ai_root} && PYTHONPATH=. .venv/bin/python scripts/record_handeye_pair.py --capture {shlex.quote(capture)}')
    with ai.open_sftp() as sftp:
        with sftp.open(ai_root+'/'+capture+'/pair.json','r') as f:result=json.loads(f.read())
    result['capture_folder']=ai_root+'/'+capture
except Exception as exc:result['error']=repr(exc)
finally:
    for c in clients:c.close()
print(json.dumps(result),flush=True)
raise SystemExit(0 if result.get('pair_accepted') else 1)
