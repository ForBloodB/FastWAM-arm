from __future__ import annotations

import time


class Gemini305Camera:
    """Open one explicitly selected V4L2 node and return timestamped RGB frames."""

    def __init__(self, device: str, width: int, height: int, fps: int):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.capture = None

    def connect(self):
        import cv2

        self.capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open Gemini 305 node {self.device}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            self.disconnect()
            raise RuntimeError(f"Gemini 305 node {self.device} returned no frame")
        return frame

    def read(self):
        if self.capture is None:
            raise RuntimeError("camera is not connected")
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("Gemini 305 frame capture failed")
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return {"timestamp_unix_ns": time.time_ns(), "rgb": rgb}

    def disconnect(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
