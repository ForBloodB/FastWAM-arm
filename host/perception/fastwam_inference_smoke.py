from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load FastWAM and run deterministic, non-actuating inference.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/aup/lerobot-fastwam"),
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    return parser.parse_args()


def _compose_config(root: Path, checkpoint: Path, stats: Path, steps: int):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    resolvers = {
        "eval": eval,
        "max": lambda value: max(value),
        "split": lambda value, index: value.split("/")[int(index)],
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)

    config_dir = root / "vendor/FastWAM-main/configs"
    overrides = [
        "task=libero_optional_idm_2cam224_1e-4",
        f"ckpt={checkpoint}",
        "model.load_text_encoder=false",
        "model.skip_dit_load_from_pretrain=true",
        "model.action_dit_pretrained_path=null",
        "model.redirect_common_files=false",
        "mixed_precision=bf16",
        "EVALUATION.device=cuda",
        f"EVALUATION.num_inference_steps={steps}",
        "EVALUATION.compile_action_infer=false",
        f"EVALUATION.dataset_stats_path={stats}",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="sim_libero", overrides=overrides)


def _load_policy(root: Path, checkpoint: Path, stats: Path, steps: int):
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    cfg = _compose_config(root, checkpoint, stats, steps)
    dtype = torch.bfloat16
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = instantiate(cfg.model, model_dtype=dtype, device="cuda")
    finally:
        torch.set_default_dtype(previous_dtype)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if "mot" not in payload or "proprio_encoder" not in payload:
        raise RuntimeError("FastWAM checkpoint is missing its MoT or proprio encoder weights")
    missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"FastWAM MoT mismatch: missing={missing}, unexpected={unexpected}")
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    del payload
    model.eval()

    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats)))
    return cfg, model, processor


def _fixed_input(processor, context_path: Path):
    context_blob = torch.load(context_path, map_location="cpu", weights_only=True)
    context = context_blob["context"]
    context_mask = context_blob["context_mask"]
    if context.shape != (1, 128, 4096) or context_mask.shape != (1, 128):
        raise RuntimeError(f"unexpected cached text context shapes: {context.shape}, {context_mask.shape}")

    external = np.full((224, 224, 3), 96, dtype=np.uint8)
    wrist = np.full((224, 224, 3), 160, dtype=np.uint8)
    wrist[::16, ::16] = 220
    rgb = np.concatenate((external, wrist), axis=1)
    image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
    image = image * (2.0 / 255.0) - 1.0

    state_key = processor.shape_meta["state"][0]["key"]
    raw_state = torch.zeros((1, 8), dtype=torch.float32)
    state_batch = {"state": {state_key: raw_state}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    proprio = state_batch["state"][state_key]
    return image, proprio, context.to(device="cuda"), context_mask.to(device="cuda"), context_blob["prompt"]


def main() -> None:
    args = _parse_args()
    if args.repeats < 2:
        raise ValueError("at least two runs are required to verify deterministic inference")
    root = args.root.resolve()
    checkpoint = root / "checkpoints/fastwam_release/libero_optional_idm_2cam224.pt"
    stats = root / "checkpoints/fastwam_release/libero_optional_idm_2cam224_dataset_stats.json"
    context_path = root / "checkpoints/fastwam_release/task_context.pt"
    for path in (checkpoint, stats, context_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(root / "checkpoints")
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    cfg, model, processor = _load_policy(root, checkpoint, stats, args.steps)
    image, proprio, context, context_mask, prompt = _fixed_input(processor, context_path)
    action_horizon = 4
    outputs = []
    durations = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        with torch.inference_mode():
            result = model.infer_action(
                prompt=None,
                input_image=image,
                action_horizon=action_horizon,
                context=context,
                context_mask=context_mask,
                proprio=proprio,
                num_inference_steps=args.steps,
                seed=1234,
                rand_device="cpu",
                compile_action_infer=False,
                action_infer_mode="first_frame",
                negative_prompt="",
                text_cfg_scale=1.0,
                tiled=False,
            )
        action = result["action"].detach().cpu().float()
        if action.shape != (action_horizon, 7) or not torch.isfinite(action).all():
            raise RuntimeError(f"invalid action tensor: {tuple(action.shape)}")
        outputs.append(action)
        durations.append(time.perf_counter() - started)

    max_error = float((outputs[0] - outputs[1]).abs().max())
    if max_error > 1e-4:
        raise RuntimeError(f"fixed-seed outputs are not deterministic: max error {max_error}")
    digest = hashlib.sha256(outputs[0].numpy().tobytes()).hexdigest()
    print(
        "PASS",
        {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "checkpoint": str(checkpoint),
            "conditioning": "cached UMT5 context",
            "prompt": prompt,
            "seed": 1234,
            "steps": args.steps,
            "action_shape": tuple(outputs[0].shape),
            "repeat_max_error": max_error,
            "action_sha256": digest,
            "inference_seconds": durations,
            "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
            "dispatch": False,
        },
    )


if __name__ == "__main__":
    main()
