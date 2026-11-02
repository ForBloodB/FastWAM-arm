"""Read-only, raw-frame evidence for J1 velocity and J7 fault diagnosis."""
import argparse
import json
import math
from pathlib import Path
import socket
import struct
import time
from reflex.canbus import parse_status_frame

p=argparse.ArgumentParser()
p.add_argument('--out',required=True)
p.add_argument('--seconds',type=float,default=15.)
p.add_argument('--motor',type=int,choices=(1,7),default=1)
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
s=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
s.bind(('can0',));s.settimeout(.002)
rows=[];latest={};frame_count=0
rawlog=(out/'frames.jsonl').open('w')

def receive():
    global frame_count
    raw=s.recv(16);ns=time.monotonic_ns()
    cid,n,data=struct.unpack('=IB3x8s',raw);eid=cid&0x1fffffff
    frame_count+=1
    rawlog.write(json.dumps({'rx_ns':ns,'can_id':hex(cid),'dlc':n,'data':data.hex()})+'\n')
    try:
        status=parse_status_frame(raw)
        latest[status['motor_id']]={'rx_ns':ns,**status}
    except ValueError: pass
    return ns,eid,n,data

def read(mid,param):
    # Drain old replies before each request. All received bytes remain in evidence.
    s.setblocking(False)
    try:
        while True: receive()
    except BlockingIOError: pass
    s.settimeout(.002)
    tx=time.monotonic_ns()
    data=struct.pack('<H',param)+b'\0'*6
    s.send(struct.pack('=IB3x8s',0x80000000|(17<<24)|(253<<8)|mid,8,data))
    deadline=time.monotonic()+.08
    while time.monotonic()<deadline:
        try:ns,cid,n,data=receive()
        except socket.timeout:continue
        if cid>>24==17 and (cid>>8)&255==mid and cid&255==253 and n==8 and data[:2]==struct.pack('<H',param):
            return {'tx_ns':tx,'rx_ns':ns,'status':(cid>>16)&255,'data_hex':data.hex(),
                    'f32':struct.unpack('<f',data[4:])[0],'u32':struct.unpack('<I',data[4:])[0]}
    return {'error':'timeout','tx_ns':tx}

try:
    probe={str(mid):{hex(param):read(mid,param) for param in
        (0x7005,0x7019,0x3016,0x701B,0x3017,0x701C,0x3022,0x3034,0x701A,0x302C,0x7028)} for mid in range(1,8)}
    (out/'probe.json').write_text(json.dumps(probe,indent=2))
    start=time.monotonic()
    for i in range(int(a.seconds*100)):
        time.sleep(max(0,start+i*.01-time.monotonic()))
        row={'ns':time.monotonic_ns()}
        for param in (0x7019,0x701B,0x3017):row[hex(param)]=read(a.motor,param)
        row['status_j1']=latest.get(1);row['status_j7']=latest.get(7)
        rows.append(row)
    (out/'samples.json').write_text(json.dumps(rows))
    valid=[r for r in rows if all('f32' in r[x] and r[x]['status']==0 for x in ('0x7019','0x701b'))]
    positions=[r['0x7019']['f32'] for r in valid]
    vel=[r['0x701b']['f32'] for r in valid]
    vel2=[r['0x3017']['f32'] for r in valid if r['0x3017'].get('status')==0]
    slopes=[]
    for i in range(10,len(valid)):
        older=valid[i-10];newer=valid[i]
        dt=(newer['0x7019']['rx_ns']-older['0x7019']['rx_ns'])/1e9
        slopes.append((newer['0x7019']['f32']-older['0x7019']['f32'])/dt)
    summary={'evidence':'real_read_only_raw_CAN','motor':a.motor,'enabled_by_script':False,'samples':len(rows),'valid_samples':len(valid),
        'duration_s':time.monotonic()-start,'frames':frame_count,'latest_status':latest,
        'position_range_deg':math.degrees(max(positions)-min(positions)) if positions else None,
        'velocity_701b_max_deg_s':max(map(lambda v:abs(math.degrees(v)),vel),default=None),
        'velocity_3017_max_deg_s':max(map(lambda v:abs(math.degrees(v)),vel2),default=None),
        'position_100ms_velocity_max_deg_s':max(map(lambda v:abs(math.degrees(v)),slopes),default=None),
        'spikes_701b_gt3':sum(abs(math.degrees(v))>3 for v in vel),
        'velocity_3017_valid_samples':len(vel2),
        'spikes_3017_gt3':sum(abs(math.degrees(v))>3 for v in vel2)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
finally:s.close();rawlog.close()
