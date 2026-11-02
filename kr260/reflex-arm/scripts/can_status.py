"""Read-only native type-24 status capture, with raw status bits preserved."""
import json
from pathlib import Path
import socket
import time
from reflex.canbus import parse_status_frame

s=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
s.bind(('can0',));s.settimeout(.1)
deadline=time.monotonic()+2
latest={};count=0
try:
    while time.monotonic()<deadline:
        try:frame=s.recv(16)
        except socket.timeout:continue
        try:status=parse_status_frame(frame)
        except ValueError:continue
        count+=1;latest[status['motor_id']]={'rx_ns':time.monotonic_ns(),**status}
finally:s.close()
result={'evidence':'real_passive_motor_status','transmitted':0,'frames':count,
        'motors':latest,'fault_bit_meanings_verified_for_firmware':False}
out=Path('runs/can/status_type24.json');out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
