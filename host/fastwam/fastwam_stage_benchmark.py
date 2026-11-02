from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from fastwam_inference_smoke import _compose_config


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _elapsed(start: float) -> float:
    _sync()
    return time.perf_counter() - start


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time FastWAM stages on the ROCm/HIP runtime.")
    parser.add_argument("--root", type=Path, default=Path("/home/aup/lerobot-fastwam"))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
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
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch did not expose the GPU")
    torch.cuda.reset_peak_memory_stats()

    stages: dict[str, float] = {}
    started = time.perf_counter()
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    stages["import_hydra_fastwam_s"] = _elapsed(started)

    started = time.perf_counter()
    cfg = _compose_config(root, checkpoint, stats, args.steps)
    stages["compose_config_s"] = _elapsed(started)

    started = time.perf_counter()
    dtype = torch.bfloat16
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = instantiate(cfg.model, model_dtype=dtype, device="cuda")
    finally:
        torch.set_default_dtype(previous_dtype)
    stages["instantiate_model_s"] = _elapsed(started)

    started = time.perf_counter()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    stages["load_checkpoint_cpu_s"] = _elapsed(started)

    started = time.perf_counter()
    missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"MoT mismatch: missing={missing}, unexpected={unexpected}")
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    del payload
    model.eval()
    stages["load_checkpoint_into_model_s"] = _elapsed(started)

    started = time.perf_counter()
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats)))
    stages["build_processor_and_stats_s"] = _elapsed(started)

    started = time.perf_counter()
    context_blob = torch.load(context_path, map_location="cpu", weights_only=True)
    context = context_blob["context"].to(device="cuda")
    context_mask = context_blob["context_mask"].to(device="cuda")
    external = np.full((224, 224, 3), 96, dtype=np.uint8)
    wrist = np.full((224, 224, 3), 160, dtype=np.uint8)
    wrist[::16, ::16] = 220
    rgb = np.concatenate((external, wrist), axis=1)
    image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
    image = image * (2.0 / 255.0) - 1.0
    state_key = processor.shape_meta["state"][0]["key"]
    raw_state = torch.zeros((1, 8), dtype=torch.float32)
    state_batch = processor.action_state_transform({"state": {state_key: raw_state}})
    state_batch = processor.normalizer.forward(state_batch)
    proprio = state_batch["state"][state_key].to(device="cuda")
    _sync()
    stages["load_context_prepare_inputs_s"] = time.perf_counter() - started

    durations = []
    outputs = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        with torch.inference_mode():
            result = model.infer_action(
                prompt=None,
                input_image=image,
                action_horizon=4,
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
        _sync()
        action = result["action"].detach().cpu().float()
        if action.shape != (4, 7) or not torch.isfinite(action).all():
            raise RuntimeError(f"invalid action tensor: {tuple(action.shape)}")
        outputs.append(action)
        durations.append(time.perf_counter() - started)

    if len(outputs) >= 2:
        repeat_error = float((outputs[0] - outputs[1]).abs().max())
    else:
        repeat_error = None
    print(json.dumps({
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "archs": torch.cuda.get_arch_list(),
        "rocm_device_api": "torch.cuda backed by HIP",
        "steps": args.steps,
        "repeats": args.repeats,
        "stages_s": stages,
        "infer_action_each_s": durations,
        "repeat_max_error": repeat_error,
        "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
        "compile_action_infer": False,
        "dispatch": False,
    }, indent=2))


if __name__ == "__main__":
    main()
