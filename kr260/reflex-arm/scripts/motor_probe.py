"""Reads actual B601 motor parameters without enabling any motor."""
import json
from pathlib import Path
import time
import argparse
from reflex.canbus import ReadOnlyCAN

p=argparse.ArgumentParser()
p.add_argument('--out',default='runs/can/motor_probe.json')
a=p.parse_args()
output=Path(a.out)
if output.exists(): p.error('Output exists; choose a new filename')
output.parent.mkdir(parents=True,exist_ok=True)
can=ReadOnlyCAN()
rows=[]
try:
    for i in range(1,8):
        row={'id':i}
        for param,name in [(0x7019,'position_rad'),(0x701B,'velocity_rad_s'),(0x701C,'voltage_V'),
                           (0x7005,'run_mode'),(0x7028,'timeout_raw'),(0x7029,'zero_state')]:
            try: row[name]=can.read(i,param)
            except (TimeoutError,RuntimeError) as exc: row[name+'_error']=str(exc)
        rows.append(row)
finally: can.close()
summary={'evidence':'real_motor_parameter_reads','monotonic_ns':time.monotonic_ns(),
         'enabled_by_this_script':False,'read_only':True,'motors':rows,'transmissions':can.log}
output.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
