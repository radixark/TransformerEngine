# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP4 weight fake-quantization for quantization-aware training.

Projects weights onto the MXFP4 (E2M1) grid with 1x32 power-of-two block scales,
returning bf16/fp32 values for the host recipe to encode. Subsequent host encoding
may round these values depending on its block layout and scale range. The scale
floor is 2^-126 and the cap is 2^125; non-finite values NaN-poison their 1x32 block.
FP16 is rejected. Numerical reference: tests/pytorch/references/mxfp4_qat_reference.py.
"""
import torch

__all__ = ["mxfp4_fake_quantize"]

_MXFP4_BLOCK = 32


class _MXFP4FakeQuantizeSTE(torch.autograd.Function):
    """Straight-through estimator: identity gradient through the projection."""

    @staticmethod
    def forward(_ctx, weight):  # pylint: disable=arguments-differ
        """Project the weight without modifying its master storage."""
        import transformer_engine_torch as tex

        if not hasattr(tex, "mxfp4_fake_quantize"):
            raise RuntimeError(
                "transformer_engine_torch was built without mxfp4_fake_quantize; "
                "rebuild Transformer Engine."
            )
        w = weight.contiguous()
        if w.data_ptr() % 16 != 0:
            # ``contiguous()`` keeps storage offsets; the kernel's vector loads need 16B alignment.
            w = w.clone()
        return tex.mxfp4_fake_quantize(w)

    @staticmethod
    def backward(_ctx, grad_output):  # pylint: disable=arguments-differ
        """Pass the gradient through to the high-precision master."""
        return grad_output


def mxfp4_fake_quantize(weight: torch.Tensor) -> torch.Tensor:
    """Project ``weight`` onto the MXFP4 grid and return it in the input dtype.

    Per 1x32 block: scale = 2^clamp(ceil(log2(amax / 6)), -126, 125), RTNE onto E2M1,
    saturate at 6, multiply back. Straight-through (identity) gradient.
    """
    if not weight.is_cuda:
        raise ValueError("MXFP4 QAT expects a CUDA weight tensor")
    if weight.dim() != 2:
        raise ValueError(f"MXFP4 QAT expects a 2D weight, got {tuple(weight.shape)}")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"MXFP4 QAT supports bf16/fp32 weights, got {weight.dtype} "
            "(fp16 cannot represent the full MXFP4 scale range)"
        )
    if weight.shape[1] % _MXFP4_BLOCK != 0:
        raise ValueError(
            "MXFP4 QAT needs the weight inner dim divisible by "
            f"{_MXFP4_BLOCK}, got {weight.shape[1]}"
        )
    return _MXFP4FakeQuantizeSTE.apply(weight)
