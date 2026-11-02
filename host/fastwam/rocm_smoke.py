from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch did not expose the GPU")
    archs = torch.cuda.get_arch_list()
    if "gfx1151" not in archs:
        raise RuntimeError(f"PyTorch was not built for gfx1151: {archs}")

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    allocated = torch.empty((256, 256), device=device, dtype=torch.bfloat16)
    left = torch.linspace(-1, 1, 256 * 256, device=device, dtype=torch.bfloat16).reshape(256, 256)
    right = left.transpose(0, 1).contiguous()
    product = left @ right
    if not torch.isfinite(product).all():
        raise RuntimeError("BF16 matrix multiplication returned non-finite values")

    query = torch.linspace(-1, 1, 1 * 4 * 16 * 64, device=device, dtype=torch.bfloat16)
    query = query.reshape(1, 4, 16, 64)
    default_attention = F.scaled_dot_product_attention(query, query, query, is_causal=True)
    with sdpa_kernel(SDPBackend.MATH):
        math_attention = F.scaled_dot_product_attention(query, query, query, is_causal=True)
    if not torch.isfinite(default_attention).all() or not torch.isfinite(math_attention).all():
        raise RuntimeError("scaled dot product attention returned non-finite values")

    print(
        "PASS",
        {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "archs": archs,
            "allocation_shape": tuple(allocated.shape),
            "matmul_shape": tuple(product.shape),
            "attention_shape": tuple(default_attention.shape),
            "attention_repeat_error": float((default_attention - F.scaled_dot_product_attention(query, query, query, is_causal=True)).abs().max()),
        },
    )


if __name__ == "__main__":
    main()
