from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass


WRIST_CAMERA_DEVICE = (
    "/dev/v4l/by-id/usb-Orbbec_R__Orbbec_Gemini_305_CV278640000A-video-index4"
)
EXTERNAL_CAMERA_DEVICE = (
    "/dev/v4l/by-id/usb-Generic_USB_Camera_202001010001-video-index0"
)


@dataclass
class V4L2Camera:
    role: str
    device: str
    width: int = 224
    height: int = 224
    fps: int = 30

    def __post_init__(self) -> None:
        if self.role not in {"wrist", "external"}:
            raise ValueError("camera role must be 'wrist' or 'external'")
        if not self.device or self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("camera device and positive dimensions/fps are required")
        self.capture = None

    def connect(self) -> None:
        import cv2

        self.capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.disconnect()
            raise RuntimeError(f"cannot open {self.role} camera at {self.device}")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.read()

    def read(self) -> dict[str, object]:
        if self.capture is None:
            raise RuntimeError(f"{self.role} camera is not connected")
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError(f"{self.role} camera capture failed")
        timestamp_unix_ns = time.time_ns()
        timestamp_monotonic_ns = time.monotonic_ns()

        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (self.height, self.width):
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return {
            "role": self.role,
            "device": self.device,
            "timestamp_unix_ns": timestamp_unix_ns,
            "timestamp_monotonic_ns": timestamp_monotonic_ns,
            "rgb": rgb,
        }

    def disconnect(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class Gemini305Camera(V4L2Camera):
    def __init__(self, device: str = WRIST_CAMERA_DEVICE, width: int = 224,
                 height: int = 224, fps: int = 30):
        super().__init__("wrist", device, width, height, fps)


class ExternalCamera(V4L2Camera):
    def __init__(self, device: str = EXTERNAL_CAMERA_DEVICE, width: int = 224,
                 height: int = 224, fps: int = 30):
        super().__init__("external", device, width, height, fps)


class DualCameraRig:
    def __init__(self, wrist: V4L2Camera, external: V4L2Camera,
                 max_skew_ms: float = 100.0):
        if wrist.role != "wrist" or external.role != "external":
            raise ValueError("camera roles must map wrist and external explicitly")
        if max_skew_ms <= 0:
            raise ValueError("max_skew_ms must be positive")
        self.cameras = {"wrist": wrist, "external": external}
        self.max_skew_ns = int(max_skew_ms * 1_000_000)
        self._readers: ThreadPoolExecutor | None = None

    def connect(self) -> None:
        try:
            for camera in self.cameras.values():
                camera.connect()
            self._readers = ThreadPoolExecutor(max_workers=len(self.cameras))
        except Exception:
            self.disconnect()
            raise

    def read(self) -> dict[str, dict[str, object]]:
        if self._readers is None:
            raise RuntimeError("camera rig is not connected")
        pending = {
            role: self._readers.submit(camera.read)
            for role, camera in self.cameras.items()
        }
        frames = {role: future.result() for role, future in pending.items()}
        capture_times = [frame["timestamp_monotonic_ns"] for frame in frames.values()]
        if max(capture_times) - min(capture_times) > self.max_skew_ns:
            raise RuntimeError("wrist and external camera samples exceed the skew limit")
        return frames

    def disconnect(self) -> None:
        if self._readers is not None:
            self._readers.shutdown(wait=True, cancel_futures=True)
            self._readers = None
        for camera in self.cameras.values():
            camera.disconnect()
