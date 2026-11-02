from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from camera_source import DualCameraRig, ExternalCamera, Gemini305Camera
from fastwam_b601_adapter import adapt_libero_action_to_b601
from fastwam_inference_smoke import _load_policy


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run non-actuating FastWAM inference on the two live cameras."
    )
    parser.add_argument("--root", type=Path, default=Path("/home/aup/lerobot-fastwam"))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--state", type=float, nargs=8, default=[0.0] * 8,
                        help="LIBERO-format 8D end-effector/gripper state; defaults to a diagnostic placeholder")
    parser.add_argument("--adapt-b601", action="store_true",
                        help="integrate the first LIBERO action into a bounded B601 base_link target")
    parser.add_argument("--max-translation-step", type=float, default=0.04,
                        help="B601 adapter translation bound in meters")
    return parser.parse_args()


def main() -> None:
    args = _args()
    root = args.root.resolve()
    checkpoint = root / "checkpoints/fastwam_release/libero_optional_idm_2cam224.pt"
    stats = root / "checkpoints/fastwam_release/libero_optional_idm_2cam224_dataset_stats.json"
    context_path = root / "checkpoints/fastwam_release/task_context.pt"
    for path in (checkpoint, stats, context_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(root / "checkpoints")
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    cfg, model, processor = _load_policy(root, checkpoint, stats, args.steps)
    context_blob = torch.load(context_path, map_location="cpu", weights_only=True)

    rig = DualCameraRig(Gemini305Camera(), ExternalCamera())
    try:
        rig.connect()
        frames = rig.read()
        rgb = np.concatenate(
            (frames["external"]["rgb"], frames["wrist"]["rgb"]), axis=1
        )
        image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        image = image.to(device="cuda", dtype=torch.bfloat16)
        image = image * (2.0 / 255.0) - 1.0

        state_key = processor.shape_meta["state"][0]["key"]
        raw_state = torch.tensor(args.state, dtype=torch.float32).unsqueeze(0)
        state_batch = processor.action_state_transform({"state": {state_key: raw_state}})
        normalized_state = processor.normalizer.forward(state_batch)["state"][state_key]

        torch.manual_seed(1234)
        with torch.inference_mode():
            result = model.infer_action(
                prompt=None,
                input_image=image,
                action_horizon=4,
                context=context_blob["context"].to("cuda"),
                context_mask=context_blob["context_mask"].to("cuda"),
                proprio=normalized_state.to("cuda"),
                num_inference_steps=args.steps,
                seed=1234,
                rand_device="cpu",
                compile_action_infer=False,
                action_infer_mode="first_frame",
                negative_prompt="",
                text_cfg_scale=1.0,
                tiled=False,
            )

        normalized_action = result["action"].detach().float().cpu()
        action_key = processor.shape_meta["action"][0]["key"]
        action_normalizer = processor.normalizer.normalizers["action"][action_key]
        libero_action = action_normalizer.backward(normalized_action).numpy()
        if libero_action.shape != (4, 7) or not np.isfinite(libero_action).all():
            raise RuntimeError(f"invalid action output: {libero_action.shape}")

        output = {
            "result": "inference_only_no_dispatch",
            "device": torch.cuda.get_device_name(0),
            "checkpoint": str(checkpoint),
            "prompt": context_blob.get("prompt"),
            "steps": args.steps,
            "state": args.state,
            "state_is_placeholder": args.state == [0.0] * 8,
            "camera_shapes": {name: tuple(frame["rgb"].shape) for name, frame in frames.items()},
            "camera_skew_ms": abs(
                int(frames["external"]["timestamp_monotonic_ns"])
                - int(frames["wrist"]["timestamp_monotonic_ns"])
            ) / 1_000_000,
            "normalized_action": normalized_action.tolist(),
            "libero_action_after_checkpoint_stats": libero_action.tolist(),
            "action_semantics": "6D LIBERO end-effector delta plus gripper; not B601 joint targets",
        }
        if args.adapt_b601:
            output["b601_adapter"] = adapt_libero_action_to_b601(
                args.state,
                libero_action[0],
                max_translation_step_m=args.max_translation_step,
            )
        print(json.dumps(output, indent=2))
    finally:
        rig.disconnect()


if __name__ == "__main__":
    main()
