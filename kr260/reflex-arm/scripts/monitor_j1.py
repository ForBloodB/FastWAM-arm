"""Post-test stationary telemetry monitoring; read-only, no motor enables."""
import csv
import json
from pathlib import Path
import time
import numpy as np
from reflex.canbus import ReadOnlyCAN

out=Path('runs/j1_monitor');out.mkdir(parents=True,exist_ok=True)
can=ReadOnlyCAN()
rows=[]
start=time.monotonic()
try:
    for i in range(500):
        time.sleep(max(0,start+i*.01-time.monotonic()))
        position=can.read(1,0x7019)
        velocity=can.read(1,0x701B)
        rows.append({'monotonic_ns':time.monotonic_ns(),'position_rad':position,'reported_velocity_rad_s':velocity})
finally:can.close()
with (out/'samples.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
positions=np.array([r['position_rad'] for r in rows])
velocities=np.array([r['reported_velocity_rad_s'] for r in rows])
result={'evidence':'read_only_post_test_telemetry','enabled_by_script':False,'samples':len(rows),
        'position_range_deg':float(np.rad2deg(np.ptp(positions))),
        'max_abs_reported_velocity_deg_s':float(np.rad2deg(max(abs(velocities)))),
        'reported_velocity_gt_3_deg_s_samples':int(np.sum(abs(velocities)>np.deg2rad(3))),
        'does_not_establish_failure_cause':True}
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
