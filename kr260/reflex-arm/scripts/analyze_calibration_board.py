"""Analyze saved RGB image only; never touches robot control."""
import argparse
import json
from pathlib import Path
import cv2
from reflex.calibration import board_pose

p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--factory',required=True)
p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
result={'success':False,'camera_to_robot_calibrated':False}
try:
    image=cv2.imread(a.image)
    if image is None:raise ValueError('Cannot read input image')
    result.update(board_pose(image,json.loads(Path(a.factory).read_text())))
    result['success']=True
    for x,y in result['image_points_px']:cv2.circle(image,(round(x),round(y)),3,(0,255,0),-1)
    cv2.imwrite(str(out/'detected.jpg'),image)
except Exception as exc:result['error']=repr(exc)
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
raise SystemExit(0 if result['success'] else 1)
