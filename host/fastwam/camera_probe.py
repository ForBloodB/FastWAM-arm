from __future__ import annotations

import statistics

from camera_source import DualCameraRig, ExternalCamera, Gemini305Camera


def main() -> None:
    cameras = DualCameraRig(Gemini305Camera(), ExternalCamera())
    try:
        cameras.connect()
        skew_ms = []
        frames = None
        for _ in range(10):
            frames = cameras.read()
            wrist_ns = int(frames["wrist"]["timestamp_monotonic_ns"])
            external_ns = int(frames["external"]["timestamp_monotonic_ns"])
            skew_ms.append(abs(wrist_ns - external_ns) / 1_000_000)
        assert frames is not None
        print(
            "PASS",
            {
                "samples": len(skew_ms),
                "skew_ms_max": max(skew_ms),
                "skew_ms_mean": statistics.mean(skew_ms),
                "frames": {
                    role: {
                        "shape": tuple(frame["rgb"].shape),
                        "dtype": str(frame["rgb"].dtype),
                        "range": (
                            int(frame["rgb"].min()),
                            int(frame["rgb"].max()),
                        ),
                    }
                    for role, frame in frames.items()
                },
            },
        )
    finally:
        cameras.disconnect()


if __name__ == "__main__":
    main()
