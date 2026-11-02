from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FastWAM on official LIBERO simulator tasks.")
    parser.add_argument("--root", type=Path, default=Path("/home/aup/lerobot-fastwam"))
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(root / "checkpoints")
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    sys.path.insert(0, str(root / "vendor/LIBERO-master"))
    sys.path.insert(0, str(root / "vendor/FastWAM-main"))
    sys.path.insert(0, str(root / "vendor/FastWAM-main/src"))
    sys.path.insert(0, str(root / "vendor/FastWAM-main/experiments/libero"))
    sys.path.insert(0, str(root / "host"))

    from libero.libero import benchmark
    from experiments.libero import eval_libero_single as evaluator
    from fastwam_inference_smoke import _load_policy

    default_prompt = "A video recorded from a robot's point of view executing the following instruction: {task}"

    # The evaluator only reads these fields for a batch-1 simulator rollout.
    cfg, policy_model, processor = _load_policy(
        root,
        root / "checkpoints/fastwam_release/libero_optional_idm_2cam224.pt",
        root / "checkpoints/fastwam_release/libero_optional_idm_2cam224_dataset_stats.json",
        args.steps,
    )

    context_root = root / "checkpoints/fastwam_release/libero_spatial_contexts"
    context_root.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    summaries = []
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    def context_encoder(prompt):
        if isinstance(prompt, (list, tuple)):
            blobs = [context_encoder(item) for item in prompt]
            return torch.cat([item[0] for item in blobs]), torch.cat([item[1] for item in blobs])
        prompt = str(prompt)
        index = prompts.index(prompt)
        blob = torch.load(context_paths[index], map_location="cuda", weights_only=True)
        return blob["context"].to("cuda"), blob["context_mask"].to("cuda", dtype=torch.bool)

    tasks = [(task_id, suite.get_task(task_id)) for task_id in args.task_ids]
    prompts = [default_prompt.format(task=task.language) for _, task in tasks]
    context_paths = [context_root / f"task_{task_id:02d}.pt" for task_id, _ in tasks]
    missing = [
        (path, prompt)
        for path, prompt in zip(context_paths, prompts)
        if not path.is_file()
    ]

    if missing:
        from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder

        model_root = root / "checkpoints/Wan-AI/Wan2.2-TI2V-5B"
        tokenizer_root = root / "checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
        text_path = model_root / "models_t5_umt5-xxl-enc-bf16.pth"
        encoder = WanTextEncoder().to(device="cuda", dtype=torch.bfloat16)
        state = torch.load(text_path, map_location="cpu", weights_only=True, mmap=True)
        if isinstance(state, dict) and len(state) == 1:
            for key in ("state_dict", "module", "model_state"):
                if key in state:
                    state = state[key]
                    break
        encoder.load_state_dict(state, strict=True)
        del state
        gc.collect()
        encoder.eval()
        tokenizer = HuggingfaceTokenizer(name=str(tokenizer_root), seq_len=128, clean="whitespace")
        with torch.inference_mode():
            for path, prompt in missing:
                token_ids, token_mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
                token_ids = token_ids.to("cuda")
                token_mask = token_mask.to("cuda", dtype=torch.bool)
                context = encoder(token_ids, token_mask)
                lengths = token_mask.sum(dim=1).long()
                for batch_index, length in enumerate(lengths):
                    context[batch_index, int(length) :] = 0
                torch.save(
                    {
                        "context": context.cpu(),
                        "context_mask": torch.ones_like(token_mask, dtype=torch.bool).cpu(),
                        "prompt": prompt,
                    },
                    path,
                )
        del encoder, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Replace text encoding with exact cached UMT5 contexts for each official task instruction.
    policy_model.encode_prompt = context_encoder

    cfg.EVALUATION.task_suite_name = args.suite
    cfg.EVALUATION.num_trials = args.trials
    cfg.EVALUATION.num_steps_wait = 5
    cfg.EVALUATION.replan_steps = 5
    cfg.EVALUATION.binarize_gripper = True
    cfg.EVALUATION.use_action_ensembler = False
    cfg.EVALUATION.visualize_future_video = False
    cfg.EVALUATION.compile_action_infer = False
    cfg.EVALUATION.action_horizon = 4
    cfg.EVALUATION.device = "cuda"
    cfg.EVALUATION.dataset_stats_path = str(root / "checkpoints/fastwam_release/libero_optional_idm_2cam224_dataset_stats.json")
    cfg.seed = args.seed

    video_dir = output_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = output_root / "predicted_videos"
    predicted_video_dir.mkdir(parents=True, exist_ok=True)
    for task_id, task in tasks:
        cfg.EVALUATION.task_id = task_id
        result_path, result = evaluator._run_task_to_file(
            cfg=cfg,
            suite_name=args.suite,
            task_id=task_id,
            model=policy_model,
            processor=processor,
            action_horizon=4,
            # Each of the two LIBERO cameras is resized to 224x224 and then
            # concatenated horizontally, so the model input is 224x448.
            input_w=448,
            input_h=224,
            model_device="cuda",
            output_root=output_root,
            worker_id=None,
        )
        summary = {
            "suite": args.suite,
            "task_id": task_id,
            "task_description": task.language,
            "successes": result["successes"],
            "episodes": result["total_episodes"],
            "success_rate": result["successes"] / result["total_episodes"],
            "duration_seconds": result["duration"],
            "result_file": str(result_path),
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        (output_root / "progress.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    aggregate = {
        "suite": args.suite,
        "checkpoint": str(root / "checkpoints/fastwam_release/libero_optional_idm_2cam224.pt"),
        "checkpoint_training_embodiment": "LIBERO Panda",
        "tasks_requested": len(tasks),
        "trials_per_task": args.trials,
        "total_episodes": sum(item["episodes"] for item in summaries),
        "total_successes": sum(item["successes"] for item in summaries),
        "micro_success_rate": sum(item["successes"] for item in summaries) / max(1, sum(item["episodes"] for item in summaries)),
        "mean_task_success_rate": sum(item["success_rate"] for item in summaries) / max(1, len(summaries)),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "seed": args.seed,
        "evaluation": "official LIBERO simulator initial states and task success predicates; not real-robot testing",
        "tasks": summaries,
    }
    (output_root / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
