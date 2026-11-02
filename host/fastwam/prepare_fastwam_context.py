from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch


MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER_ID = "Wan-AI/Wan2.1-T2V-1.3B"
TEXT_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
TOKENIZER_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache one FastWAM prompt embedding on 395.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--model-base",
        type=Path,
        default=Path("/home/aup/lerobot-fastwam/checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/aup/lerobot-fastwam/checkpoints/fastwam_release/task_context.pt"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_root = args.model_base / MODEL_ID
    tokenizer_root = args.model_base / TOKENIZER_ID
    text_path = model_root / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer_path = tokenizer_root / "google/umt5-xxl"
    required_tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
    missing = [str(path) for path in (text_path,) if not path.is_file()]
    missing.extend(
        str(tokenizer_path / name)
        for name in required_tokenizer_files
        if not (tokenizer_path / name).is_file()
    )
    if missing:
        raise FileNotFoundError("Missing downloaded UMT5 files: " + ", ".join(missing))

    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder

    torch.manual_seed(0)
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        encoder = WanTextEncoder().to(device=args.device, dtype=torch.bfloat16)
    finally:
        torch.set_default_dtype(original_dtype)

    state = torch.load(text_path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, dict) and len(state) == 1:
        for key in ("state_dict", "module", "model_state"):
            if key in state:
                state = state[key]
                break
    missing_keys, unexpected_keys = encoder.load_state_dict(state, strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(f"UMT5 checkpoint mismatch: missing={missing_keys}, unexpected={unexpected_keys}")
    del state
    gc.collect()
    encoder.eval()

    tokenizer = HuggingfaceTokenizer(
        name=str(tokenizer_path),
        seq_len=128,
        clean="whitespace",
    )
    token_ids, token_mask = tokenizer(args.prompt, return_mask=True, add_special_tokens=True)
    with torch.inference_mode():
        context = encoder(token_ids.to(args.device), token_mask.to(args.device, dtype=torch.bool))
        lengths = token_mask.gt(0).sum(dim=1).long()
        for batch_index, length in enumerate(lengths):
            context[batch_index, int(length) :] = 0
        context_mask = torch.ones_like(token_mask, dtype=torch.bool)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "context": context.cpu(),
            "context_mask": context_mask,
            "prompt": args.prompt,
            "text_revision": TEXT_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
        },
        args.output,
    )
    print("Cached prompt context", args.output, tuple(context.shape), context.dtype)

    del encoder
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
