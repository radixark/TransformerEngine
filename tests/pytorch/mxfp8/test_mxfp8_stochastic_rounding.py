# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 cast with stochastic rounding: scales match round-to-nearest, each element lands on one of
its two neighbouring FP8 values, the rounding is unbiased, and a seed fixes the bits."""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import MXFP8Quantizer

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)

SHAPES = [(128, 256), (256, 1024), (192, 160)]
FP8_DTYPES = {
    te.DType.kFloat8E4M3: torch.float8_e4m3fn,
    te.DType.kFloat8E5M2: torch.float8_e5m2,
}


def _valid_scales(data: torch.Tensor, scale_inv: torch.Tensor, columnwise: bool) -> torch.Tensor:
    """The E8M0 scales of real blocks; the rest of the padded scale buffer is never written."""
    rows, cols = data.shape
    if columnwise:
        return scale_inv[: rows // 32, :cols].view(torch.uint8)
    return scale_inv[:rows, : (cols + 31) // 32].view(torch.uint8)


def _decode(data: torch.Tensor, scale_inv: torch.Tensor, torch_fp8: torch.dtype, columnwise: bool):
    """(FP8 values as fp32, per-element power-of-two scale) of one MXFP8 direction."""
    cols = data.shape[1]
    values = data.view(torch.uint8).view(torch_fp8).float()
    exponents = _valid_scales(data, scale_inv, columnwise)
    if columnwise:
        exponents = exponents.repeat_interleave(32, dim=0)
    else:
        exponents = exponents.repeat_interleave(32, dim=1)[:, :cols]
    return values, torch.exp2(exponents.float() - 127.0)


def _directions(quantized, rowwise: bool, columnwise: bool):
    if rowwise:
        yield quantized._rowwise_data, quantized._rowwise_scale_inv, False
    if columnwise:
        yield quantized._columnwise_data, quantized._columnwise_scale_inv, True


def _grid_neighbours(target: torch.Tensor, torch_fp8: torch.dtype):
    """The FP8 grid values just below and just above |target| (equal when |target| is on the grid)."""
    magnitude = target.abs()
    nearest = magnitude.to(torch_fp8)
    code = nearest.view(torch.uint8).to(torch.int16)
    up = (code + 1).clamp(max=127).to(torch.uint8).view(torch_fp8).float()
    down = (code - 1).clamp(min=0).to(torch.uint8).view(torch_fp8).float()
    nearest = nearest.float()
    low = torch.where(nearest <= magnitude, nearest, down)
    high = torch.where(nearest >= magnitude, nearest, up)
    return low, high


def _quantize(x, fp8_dtype, rowwise, columnwise, stochastic_rounding):
    quantizer = MXFP8Quantizer(
        fp8_dtype=fp8_dtype,
        rowwise=rowwise,
        columnwise=columnwise,
        stochastic_rounding=stochastic_rounding,
    )
    return quantizer(x)


DIRECTIONS = pytest.mark.parametrize(
    "rowwise, columnwise", [(True, False), (False, True), (True, True)], ids=["row", "col", "both"]
)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("fp8_dtype", list(FP8_DTYPES), ids=str)
@DIRECTIONS
def test_stochastic_rounding_picks_a_neighbour_and_keeps_scales(
    shape, x_dtype, fp8_dtype, rowwise, columnwise
):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=x_dtype, device="cuda") * 3.0
    torch_fp8 = FP8_DTYPES[fp8_dtype]
    nearest = _quantize(x, fp8_dtype, rowwise, columnwise, stochastic_rounding=False)
    stochastic = _quantize(x, fp8_dtype, rowwise, columnwise, stochastic_rounding=True)

    for (rn_data, rn_scale, colwise), (sr_data, sr_scale, _) in zip(
        _directions(nearest, rowwise, columnwise), _directions(stochastic, rowwise, columnwise)
    ):
        torch.testing.assert_close(
            _valid_scales(sr_data, sr_scale, colwise),
            _valid_scales(rn_data, rn_scale, colwise),
            rtol=0,
            atol=0,
        )

        rn_values, scale = _decode(rn_data, rn_scale, torch_fp8, colwise)
        target = x.float() / scale
        # Round-to-nearest still equals a PyTorch cast of the same scaled input, bit for bit.
        torch.testing.assert_close(
            rn_data.view(torch.uint8),
            target.to(torch_fp8).view(torch.uint8),
            rtol=0,
            atol=0,
        )

        sr_values, _ = _decode(sr_data, sr_scale, torch_fp8, colwise)
        low, high = _grid_neighbours(target, torch_fp8)
        magnitude = sr_values.abs()
        assert torch.all((magnitude == low) | (magnitude == high))
        assert torch.all((sr_values == 0) | (torch.sign(sr_values) == torch.sign(target)))
        # Some elements off the grid must round away from nearest, or rounding is not stochastic.
        assert torch.any(sr_data.view(torch.uint8) != rn_data.view(torch.uint8))


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("fp8_dtype", list(FP8_DTYPES), ids=str)
@DIRECTIONS
def test_stochastic_rounding_is_unbiased(fp8_dtype, rowwise, columnwise):
    torch.manual_seed(1)
    x = torch.randn((128, 256), dtype=torch.bfloat16, device="cuda")
    torch_fp8 = FP8_DTYPES[fp8_dtype]
    draws = 512

    sums = None
    for _ in range(draws):
        quantized = _quantize(x, fp8_dtype, rowwise, columnwise, stochastic_rounding=True)
        decoded = [_decode(d, s, torch_fp8, c) for d, s, c in _directions(quantized, rowwise, columnwise)]
        values = [v for v, _ in decoded]
        sums = values if sums is None else [a + v for a, v in zip(sums, values)]

    for (values_sum, (_, scale)) in zip(sums, decoded):
        target = x.float() / scale
        low, high = _grid_neighbours(target, torch_fp8)
        off_grid = high > low
        ulp = (high - low)[off_grid]
        bias = ((values_sum / draws).abs() - target.abs())[off_grid] / ulp
        nearest_bias = (target.to(torch_fp8).float().abs() - target.abs())[off_grid] / ulp
        # One draw is a Bernoulli over one ulp, so the mean of 512 has std <= 0.5 / sqrt(512) ~ 0.022.
        assert bias.abs().mean() < 0.03, bias.abs().mean()
        assert bias.abs().max() < 0.15, bias.abs().max()
        assert nearest_bias.abs().mean() > 0.15


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@DIRECTIONS
def test_stochastic_rounding_is_reproducible_from_the_seed(rowwise, columnwise):
    torch.manual_seed(2)
    x = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda")
    fp8_dtype = te.DType.kFloat8E4M3

    def bits(seed):
        torch.cuda.manual_seed(seed)
        quantized = _quantize(x, fp8_dtype, rowwise, columnwise, stochastic_rounding=True)
        return [d.view(torch.uint8).clone() for d, _, _ in _directions(quantized, rowwise, columnwise)]

    first, again, other = bits(7), bits(7), bits(8)
    for a, b, c in zip(first, again, other):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.any(a != c)
