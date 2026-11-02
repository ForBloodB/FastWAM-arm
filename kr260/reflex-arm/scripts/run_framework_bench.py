"""Bounded, sequential cases; stop globally on a safety or restoration failure.

Tracking failures stop repetitions for that target. Other independent targets
can be examined only after a completed, restored profile with zero faults.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
cases={
    'j2_hold_025':(2,['--loaded-duration','30','--loaded-peak-hold','8','--loaded-return-hold','8']),
    'j2_repeat3':(2,['--loaded-duration','30','--loaded-peak-hold','8','--loaded-return-hold','8','--repetitions','3']),
    'j3_hold_025':(3,['--loaded-duration','30','--loaded-peak-hold','8','--loaded-return-hold','8']),
    'j3_speed05_comparison':(3,['--loaded-duration','30','--loaded-peak-hold','8',
        '--loaded-return-hold','8','--loaded-speed-limit','.5']),
    'j7_positive_hold':(7,['--gripper-amplitude','.5','--gripper-duration','30',
        '--gripper-peak-hold','8','--gripper-return-hold','8']),
    'j7_negative_hold':(7,['--gripper-amplitude','.5','--gripper-duration','30',
        '--gripper-peak-hold','8','--gripper-return-hold','8','--gripper-direction','-1']),
    'j2_fixed10':(2,['--loaded-duration','30','--loaded-peak-hold','8','--loaded-return-hold','8','--repetitions','10']),
    'j2_half_gain_fixed10':(2,['--loaded-duration','30','--loaded-peak-hold','8',
        '--loaded-return-hold','8','--loaded-speed-scale','.5','--repetitions','10']),
    'j4_fixed10':(4,['--j4-speed-kp','4','--repetitions','10']),
    'j5_fixed10':(5,['--repetitions','10']),
    'j5_slow12_fixed10':(5,['--wrist-duration','12','--repetitions','10']),
    'j6_fixed10':(6,['--wrist-duration','12','--repetitions','10']),
}
p=argparse.ArgumentParser();p.add_argument('cases',nargs='+',choices=sorted(cases))
p.add_argument('--out-root', type=Path, default=root/'runs/framework_20261020')
p.add_argument('--execute-authorized-test',action='store_true',required=True)
a=p.parse_args()
out=a.out_root
out.mkdir(parents=True,exist_ok=True)
coordinator=out/'coordinator_source.py'
if coordinator.exists():raise SystemExit('Use a new batch directory; coordinator evidence already exists')
coordinator.write_bytes(Path(__file__).read_bytes())
for name in a.cases:
    mid,options=cases[name];target=out/name
    if target.exists():raise SystemExit('Refusing to overwrite '+name)
    command=[sys.executable,str(root/'scripts/wrist_csp_bench.py'),'--joint',str(mid),
        '--out',str(target),'--execute-authorized-test',*options]
    if mid in (2,3,4):command.append('--load-support-verified')
    env=dict(os.environ);env['PYTHONPATH']=str(root)
    process=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True)
    (out/(name+'.stdout')).write_text(process.stdout)
    (out/(name+'.stderr')).write_text(process.stderr)
    if not (target/'summary.json').exists():raise SystemExit('No summary: '+process.stderr)
    result=json.loads((target/'summary.json').read_text())
    keys=('completed_profile','passed','actual_amplitude_motor_deg','final_error_motor_deg',
        'tracking_rmse_motor_deg','peak_abs_current_A','parameters_restored','cycle_results','error','restoration_error')
    print(json.dumps({'case':name,**{k:result[k] for k in keys if k in result}}),flush=True)
    if (not result['completed_profile'] or result.get('error') or not result.get('parameters_restored')
        or result.get('restoration_error') or len(result.get('final_status',{}))!=7
        or any(s['fault_mask_raw'] for s in result.get('final_status',{}).values())):
        raise SystemExit('Safety/restoration failure: subsequent targets blocked')
