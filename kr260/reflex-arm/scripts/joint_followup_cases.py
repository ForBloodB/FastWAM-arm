"""Run only named, bounded cases using the guarded single-joint entry point."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
evidence=root/'runs/joint_followup_20261020'
cases={
    'j2_original30':(2,['--loaded-duration','30']),
    'j3_original30':(3,['--loaded-duration','30']),
    'j2_dwell30':(2,['--loaded-duration','30','--loaded-peak-hold','8','--loaded-return-hold','8']),
    'j3_dwell30':(3,['--loaded-duration','30','--loaded-peak-hold','8','--loaded-return-hold','8']),
    'j7_positive05':(7,['--gripper-amplitude','0.5','--gripper-current','0.45']),
    'j7_negative05':(7,['--gripper-amplitude','0.5','--gripper-current','0.45','--gripper-direction','-1']),
}
for mid in (4,5,6):
    for n in range(1,4):
        cases[f'j{mid}_repeat{n}']=(mid,['--j4-speed-kp','4'] if mid==4 else [])
groups={f'j{mid}_repeats':[f'j{mid}_repeat{n}' for n in range(1,4)] for mid in (4,5,6)}
for mid in (2,3):
    for n in (2,3):cases[f'j{mid}_original30_repeat{n}']=(mid,['--loaded-duration','30'])
    groups[f'j{mid}_confirm30']=[f'j{mid}_original30_repeat{n}' for n in (2,3)]
for n in range(1,4):cases[f'j6_slow12_repeat{n}']=(6,['--wrist-duration','12'])
groups['j6_slow12_repeats']=[f'j6_slow12_repeat{n}' for n in range(1,4)]
for n in (2,3):cases[f'j2_dwell30_repeat{n}']=(2,['--loaded-duration','30','--loaded-peak-hold','8','--loaded-return-hold','8'])
groups['j2_confirm_dwell']=[f'j2_dwell30_repeat{n}' for n in (2,3)]
p=argparse.ArgumentParser()
p.add_argument('case',choices=sorted([*cases,*groups]))
p.add_argument('--execute-authorized-test',required=True,action='store_true')
a=p.parse_args()
for name in groups.get(a.case,[a.case]):
    mid,extra=cases[name]
    target=evidence/name
    if target.exists():raise SystemExit(f'Refusing to overwrite existing case {name}')
    args=[sys.executable,str(root/'scripts/wrist_csp_bench.py'),'--joint',str(mid),
          '--out',str(target),'--execute-authorized-test']+extra
    if mid in (2,3,4):args+=['--load-support-verified']
    env=dict(os.environ);env['PYTHONPATH']=str(root)
    completed=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True)
    (evidence/(name+'.stdout.json')).write_text(completed.stdout)
    (evidence/(name+'.stderr.txt')).write_text(completed.stderr)
    if not (target/'summary.json').is_file():
        print(completed.stderr,flush=True);raise SystemExit(completed.returncode or 1)
    result=json.loads((target/'summary.json').read_text())
    keys=['joint','completed_profile','passed','temporary_speed_kp','profile_duration_s',
          'actual_amplitude_motor_deg','final_error_motor_deg','tracking_rmse_motor_deg',
          'peak_abs_current_A','maximum_feedback_gap_ms','parameters_restored',
          'initial_enable_state_restored','peak_hold_s','return_hold_s','error','restoration_error']
    print(json.dumps({'case':name,**{k:result[k] for k in keys if k in result}}),flush=True)
    if completed.returncode or not result['passed']:
        # Repetition groups never continue after a failed profile or criterion.
        raise SystemExit(completed.returncode or 1)
