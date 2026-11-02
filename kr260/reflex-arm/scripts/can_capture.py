"""Passive SocketCAN capture only. This script never calls send()."""
import argparse
import json
from pathlib import Path
import socket
import struct
import time

p=argparse.ArgumentParser()
p.add_argument('--channel',default='can0')
p.add_argument('--seconds',type=float,default=5)
p.add_argument('--out',default='runs/can/passive.jsonl')
a=p.parse_args()
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
s=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
s.bind((a.channel,)); s.settimeout(.1)
deadline=time.monotonic()+a.seconds
rows=[]
try:
    with out.open('w') as log:
        while time.monotonic()<deadline:
            try: frame=s.recv(16)
            except socket.timeout: continue
            canid,dlc,data=struct.unpack('=IB3x8s',frame)
            eid=canid&0x1fffffff
            row={'rx_ns':time.monotonic_ns(),'can_id_hex':hex(eid),'extended':bool(canid&0x80000000),
                 'error_frame':bool(canid&0x20000000),'dlc':dlc,'data_hex':data[:dlc].hex(),
                 'robstride_type':(eid>>24)&0x1f,'sender_field':(eid>>8)&0xff,'destination_field':eid&0xff}
            log.write(json.dumps(row)+'\n'); rows.append(row)
finally:
    s.close()
print(json.dumps({'evidence':'passive_CAN_bus_capture','frames':len(rows),'transmitted':0,
                  'unique_ids':sorted(set(r['can_id_hex'] for r in rows)),'out':str(out)}))
print(json.dumps(rows[:12]))
