"""Explicit CLI-only staged J1 out/back, with all production protections."""
import argparse,json,time
from pathlib import Path
from board_server import State

if __name__=='__main__':
    out=Path('/home/ubuntu/reflex-arm/runs/leader_web_20261031')
    parser=argparse.ArgumentParser();parser.add_argument('--amplitude',type=float,choices=[.5,1.],required=True);args=parser.parse_args()
    state=State(out)
    try:
        time.sleep(1.5)
        state.follow.start(dict(axis=1,site_confirmed=True,gain=.1),pilot=args.amplitude)
        state.follow.thread.join(25.)
        if state.follow.thread.is_alive():state.follow.stop('CLI时间到');state.follow.thread.join(3.)
        print(json.dumps(state.follow.snapshot(),ensure_ascii=False,indent=2),flush=True)
    finally:state.stop.set()
