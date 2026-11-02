import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--frames", type=int, default=10)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args = parser.parse_args()

args.output.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera: {args.device}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
shape = None
stamps = []
statistics = []
try:
    for index in range(args.frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Capture failed at frame {index}")
        stamps.append(time.monotonic())
        shape = list(frame.shape)
        statistics.append([float(frame.mean()), float(frame.std())])
        if index in (0, args.frames - 1):
            if not cv2.imwrite(str(args.output / f"frame_{index:03d}.png"), frame):
                raise RuntimeError("PNG write failed")
finally:
    cap.release()

elapsed = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
print(json.dumps({
    "device": args.device,
    "frames": len(stamps),
    "shape_bgr": shape,
    "elapsed_seconds": elapsed,
    "mean_capture_fps": (len(stamps) - 1) / elapsed if elapsed > 0 else None,
    "frame_mean_std": statistics,
    "output": str(args.output),
}, indent=2))
