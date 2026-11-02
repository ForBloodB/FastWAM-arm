"""Passive CAN capture for a future operator web comparison. Never transmits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import socket
import time
from reflex.passive_wrist import decode_j6_frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=int,choices=(60,180,300),default=180)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    stop=False
    def interrupted(*unused):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    counts=Counter();commands=[];total=0
    result={'CAN_transmissions':0,'motor_enables':0,'motor_control_writes':0,
        'working_behavior_verified':False,'automatic_motion_release':False,
        'clock':'KR260 local monotonic only','requested_seconds':args.seconds}
    bus=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
    bus.bind(('can0',));bus.settimeout(.2)
    start=time.monotonic()
    print(json.dumps({'event':'PASSIVE_CAPTURE_READY','out':str(args.out),'CAN_transmissions':0}),flush=True)
    try:
        with (args.out/'raw_frames.jsonl').open('x') as raw_log, (args.out/'decoded_j6.jsonl').open('x') as decoded_log:
            while not stop and time.monotonic()-start<args.seconds and not (args.out/'STOP').exists():
                try:frame=bus.recv(16)
                except socket.timeout:continue
                ns=time.monotonic_ns();total+=1
                raw_log.write(json.dumps({'rx_ns':ns,'frame_hex':frame.hex()})+'\n')
                decoded=decode_j6_frame(frame)
                if decoded is not None:
                    decoded.update(rx_ns=ns);counts[decoded['category']]+=1
                    decoded_log.write(json.dumps(decoded)+'\n')
                    if decoded['category']=='control':commands.append(decoded)
    except Exception as exc:
        result['error']=repr(exc)
    finally:
        bus.close()
        result.update(elapsed_s=time.monotonic()-start,total_frames=total,
            j6_categories=dict(counts),observed_control_commands=len(commands),
            note='Observed commands are from other sockets; capture does not enable, move, stop, or qualify the arm.')
        (args.out/'summary.json').write_text(json.dumps(result,indent=2))
        (args.out/'observed_j6_commands.json').write_text(json.dumps(commands,indent=2))
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
