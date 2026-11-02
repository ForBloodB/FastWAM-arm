"""KR260 PS echo/validation daemon: deliberately has NO motor/MMIO driver.

Board-local receipt time controls heartbeat freshness. Remote arming and resetting
are disabled. Binding beyond loopback is explicit and intended only for a trusted
isolated lab network. No incoming field can enable physical outputs.
"""
import argparse
import json
from pathlib import Path
import signal
import socket
import time
from .protocol import Packet, Receiver
from .safety import SafetyFSM


def serve(bind, port, logpath, allowed_peer='127.0.0.1'):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind,port))
    sock.settimeout(.01)
    receiver, fsm = Receiver(), SafetyFSM()
    log = Path(logpath)
    log.parent.mkdir(parents=True, exist_ok=True)
    running = True
    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(json.dumps({'bind':bind,'port':port,'mode':'NO_ACTUATORS','ready':True}),flush=True)
    with log.open('a',encoding='utf-8') as out:
        try:
            while running:
                now = time.monotonic_ns()
                fsm.tick(now)
                try:
                    data, peer = sock.recvfrom(8192)
                except socket.timeout:
                    continue
                if peer[0] != allowed_peer:
                    continue
                now = time.monotonic_ns()
                try:
                    packet = receiver.accept(data, now)
                    if packet.kind != 1 or packet.payload.get('mode') != 'diagnostic':
                        raise ValueError('Only diagnostic heartbeats are implemented')
                    fsm.tick(now,heartbeat=True)
                    response = Packet(packet.sequence,packet.task,
                        {'mode':'NO_ACTUATORS','state':int(fsm.state),'board_rx_ns':now,
                         'board_tx_ns':time.monotonic_ns(),'echo_sender_ns':packet.timestamp_ns},kind=2)
                    sock.sendto(response.encode(),peer)
                    event = {'sequence':packet.sequence,'rx_ns':now,'peer':peer[0],'accepted':True,'mode':'NO_ACTUATORS'}
                except (ValueError, OverflowError) as exc:
                    event = {'rx_ns':now,'peer':peer[0],'accepted':False,'error':str(exc)}
                out.write(json.dumps(event)+'\n')
                out.flush()
        finally:
            sock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bind',default='127.0.0.1')
    p.add_argument('--port',type=int,default=47000)
    p.add_argument('--log',default='runs/bridge/packets.jsonl')
    p.add_argument('--allow-peer',default='127.0.0.1')
    a = p.parse_args()
    serve(a.bind,a.port,a.log,a.allow_peer)

if __name__ == '__main__':
    main()
