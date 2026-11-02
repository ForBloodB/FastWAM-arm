"""Measure host timing and actual UDP RTT. Never label PS/host data as PL timing."""
import argparse
import csv
import json
from pathlib import Path
import socket
import time
import numpy as np
from .protocol import Packet


def summary(values):
    a = np.asarray(values,float)
    if a.size == 0:
        return None
    return {'n':len(a),'mean':float(a.mean()),'p50':float(np.quantile(a,.5)),
            'p99':float(np.quantile(a,.99)),'max':float(a.max())}


def rtt(host,port,count,out):
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.settimeout(.2)
    sock.connect((host,port))
    rows=[]
    base=time.monotonic_ns()
    for i in range(count):
        seq=base+i
        start=time.monotonic_ns()
        sock.send(Packet(seq,1,{'mode':'diagnostic'},timestamp_ns=start).encode())
        deadline=time.monotonic()+.2
        ok=False
        while time.monotonic()<deadline:
            try:
                reply=Packet.decode(sock.recv(8192))
                end=time.monotonic_ns()
            except (socket.timeout,ValueError):
                break
            if reply.sequence==seq and reply.kind==2 and reply.payload.get('echo_sender_ns')==start:
                ok=True
                rows.append({'sequence':seq,'send_ns':start,'receive_ns':end,'rtt_us':(end-start)/1e3,
                    'board_rx_ns':reply.payload['board_rx_ns'],'board_tx_ns':reply.payload['board_tx_ns'],'ok':1})
                break
        if not ok:
            rows.append({'sequence':seq,'send_ns':start,'receive_ns':'','rtt_us':'','board_rx_ns':'','board_tx_ns':'','ok':0})
        time.sleep(.002)
    sock.close()
    write_csv(out/'rtt.csv',rows)
    return {'evidence':'ethernet_round_trip_to_KR260_PS' if host!='127.0.0.1' else 'loopback',
            'host':host,'unit':'microseconds','rtt':summary([r['rtt_us'] for r in rows if r['ok']]),
            'timeout_rate':sum(not r['ok'] for r in rows)/len(rows),
            'notes':'One-way latency is not inferred from unsynchronized clocks. This is PS, not PL.'}


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def host_timing(count,period_s,out):
    period_ns=int(period_s*1e9)
    origin=time.monotonic_ns()
    rows=[]
    cpu_start=time.process_time()
    for i in range(count):
        deadline=origin+(i+1)*period_ns
        remaining=(deadline-time.monotonic_ns())/1e9
        if remaining>0:
            time.sleep(remaining)
        actual=time.monotonic_ns()
        rows.append({'index':i,'scheduled_ns':deadline,'actual_ns':actual,
                     'lateness_us':max(0,actual-deadline)/1000,
                     'deadline_miss':int(actual>=deadline+period_ns)})
    elapsed=(time.monotonic_ns()-origin)/1e9
    write_csv(out/'host_timing.csv',rows)
    intervals=np.diff([r['actual_ns'] for r in rows])/1e3
    return {'evidence':'measured_host_scheduler_only','target_period_us':period_ns/1000,
            'actual_period_us':summary(intervals),'absolute_period_jitter_us':summary(abs(intervals-period_ns/1000)),
            'lateness_us':summary([r['lateness_us'] for r in rows]),
            'deadline_miss_definition':'wake >= scheduled wake + one full target period',
            'deadline_miss_rate':sum(r['deadline_miss'] for r in rows)/count,
            'one_core_process_cpu_fraction':(time.process_time()-cpu_start)/elapsed,
            'tracking_rmse':None,'fpga_pl_comparison':None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=47000)
    p.add_argument('--count',type=int,default=1000)
    p.add_argument('--period-ms',type=float,default=5.)
    p.add_argument('--out',default='runs/timing')
    a=p.parse_args()
    if a.count<2 or a.period_ms<=0: p.error('count >= 2 and positive period required')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    result={'udp':rtt(a.host,a.port,a.count,out),'host':host_timing(a.count,a.period_ms/1000,out)}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
