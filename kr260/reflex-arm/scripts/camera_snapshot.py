"""Capture one RGB frame for local engineering inspection; no pose calibration inferred."""
import json
from pathlib import Path
import cv2

out=Path('runs/camera'); out.mkdir(parents=True,exist_ok=True)
rows=[]
for i in [10]:
    capture=cv2.VideoCapture(i,cv2.CAP_V4L2)
    try:
        if not capture.isOpened():
            rows.append({'device':i,'opened':False}); continue
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        for _ in range(10): ok,frame=capture.read()
        if ok:
            path=out/f'video{i}.jpg'
            cv2.imwrite(str(path),frame)
            rows.append({'device':i,'shape':list(frame.shape),'path':str(path)})
    finally: capture.release()
(out/'capture.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
