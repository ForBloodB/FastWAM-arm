"""Assemble an acquired stationary pair from saved files; no device I/O."""
import argparse
import hashlib
import json
from pathlib import Path
from reflex.handeye_capture import assemble_pair
from reflex.kinematics import URDFModel

p=argparse.ArgumentParser();p.add_argument('--capture',required=True);a=p.parse_args()
folder=Path(a.capture);out=folder/'pair.json'
if out.exists():p.error('Pair result already exists')
root=Path(__file__).resolve().parents[1]
result={'pair_accepted':False,'camera_extrinsic_calibrated':False}
try:
    def read(p):return json.loads(p.read_text())
    robot=read(folder/'robot_summary.json');board=read(folder/'board_detection/summary.json')
    timing=read(folder/'stationary_window.json')
    result=assemble_pair(URDFModel(root/'assets/B601_RS.urdf'),robot,board,timing)
    result.update(evidence='actual_stationary_robot_and_RGB_pair',
        urdf_sha256=hashlib.sha256((root/'assets/B601_RS.urdf').read_bytes()).hexdigest(),
        board_sha256=hashlib.sha256((root/'calibration/board.json').read_bytes()).hexdigest(),
        camera_serial=read(folder/'summary.json')['device']['serial'],stationary_window=timing)
except Exception as exc:result['error']=repr(exc)
out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
raise SystemExit(0 if result['pair_accepted'] else 1)
