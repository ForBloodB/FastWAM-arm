"""Actual diagnostic parser tests against KR260; no actuator faults are injected."""
import json
from pathlib import Path
import socket
import time
import struct
import zlib
from reflex.protocol import Packet,HEADER

sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
sock.connect(('172.21.16.34',47000));sock.settimeout(.1)
seq=time.monotonic_ns()
good=Packet(seq,5,{'mode':'diagnostic'},timestamp_ns=seq).encode()
sock.send(good)
reply=Packet.decode(sock.recv(4096))
rows=[{'case':'valid','reply':reply.payload,'passed':reply.kind==2 and reply.sequence==seq}]
def raw_frame(body,sequence):
    data=HEADER.pack(b'RFAX',1,1,0,sequence,5,time.monotonic_ns(),100,len(body))+body
    return data+struct.pack('!I',zlib.crc32(data))
for name,data in [('duplicate',good),('out_of_order',Packet(seq-1,5,{'mode':'diagnostic'}).encode()),
                  ('crc_corruption',good[:-1]+bytes([good[-1]^1])),('short',b'bad'),
                  ('unsupported_command',Packet(seq+1,5,{'mode':'move'}).encode()),
                  ('numeric_overflow',raw_frame(b'{"mode":"diagnostic","x":1e999}',seq+2)),
                  ('deep_nesting',raw_frame(b'{"mode":"diagnostic","x":'+b'['*32+b'0'+b']'*32+b'}',seq+3))]:
    sock.send(data)
    try:
        sock.recv(4096);passed=False
    except socket.timeout:passed=True
    rows.append({'case':name,'passed':passed})
sock.close()
out=Path('runs/network_faults');out.mkdir(parents=True,exist_ok=True)
result={'evidence':'real_KR260_PS_diagnostic_parser_tests','actuator_access':False,'cases':rows}
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
assert all(r['passed'] for r in rows)
