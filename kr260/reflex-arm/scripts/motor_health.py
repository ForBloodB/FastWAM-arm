"""Read MotorBridge-decoded telemetry. Never enables a motor."""
from dataclasses import asdict
import json
from pathlib import Path
import time
from motorbridge import Controller

controller=Controller('can0')
motors=[]
try:
    motors=[controller.add_robstride_motor(i,0xFD,'rs-06' if i<=3 else 'rs-00') for i in range(1,8)]
    rows=[]
    for motor in motors:
        position=motor.robstride_get_param_f32(0x7019,300)
        state=motor.get_state()
        rows.append({'position_read_rad':position,'state':asdict(state) if state else None})
    result={'evidence':'MotorBridge_decoded_real_telemetry','enabled':False,'motors':rows}
    out=Path('runs/can/health.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)); print(json.dumps(result))
finally:
    controller.close_bus()
    for motor in motors: motor.close()
    controller.close()
