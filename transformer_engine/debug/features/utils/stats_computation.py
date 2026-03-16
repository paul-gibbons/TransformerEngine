# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Mathematical functions used to tensor statistics computation.
"""

import math
from collections import namedtuple
from typing import Optional

import torch
import torch.nn.functional as F
import transformer_engine_torch as tex
from transformer_engine.common.recipe import Format

OSCILLATION_STAT_NAMES = frozenset(
    {
        "oscillation_ratio",
        "oscillation_ratio_reduced",
        "oscillation_l1distqw",
        "oscillation_l1distw",
        "oscillation_l1distw_reduced",
    }
)

# FP4 E2M1: 16 representable values. Index 0 = +0, index 8 = -0.
_FP4_E2M1_NUM_BINS = 16
_FP4_E2M1_NEG_ZERO_INDEX = 8
# Midpoints between adjacent positive FP4 E2M1 magnitudes (0, 0.5, 1, 1.5, 2, 3, 4, 6).
# Used with torch.bucketize for memory-efficient bin assignment.
_FP4_E2M1_POS_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
# FP4_MAX * FP8_E4M3_MAX = 6.0 * 448.0, used in NVFP4 two-level scaling.
_NVFP4_SCALE_FACTOR = 6.0 * 448.0

# Finer master-weight binning (2× FP4 resolution) for oscillation detection.
# Boundaries at quarter-points between FP4 values and FP4 midpoints so that
# FP4 quantization thresholds (0.25, 0.75, …) land at bin CENTERS, not edges.
# Small master-weight oscillations near a quantization boundary stay in one
# master bin while the quantized histogram flips → ratio correctly spikes.
_MASTER_W_NUM_BINS = 30  # 15 positive + 15 negative
_MASTER_W_NEG_ZERO_INDEX = 15  # index of -0
_MASTER_W_POS_BOUNDARIES = [
    0.125, 0.375, 0.625, 0.875,
    1.125, 1.375, 1.625, 1.875,
    2.25, 2.75, 3.25, 3.75, 4.5, 5.5,
]

_MASTER_29_TO_15_COARSE_INDEX = (
    0,
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    7,
    8,
    8,
    9,
    9,
    10,
    10,
    11,
    11,
    12,
    12,
    13,
    13,
    14,
    14,
)

_HISTOGRAM_LATENT_CHUNK_ELEMS = 1 << 20


class BlockwiseDynamicRangeStat(
    namedtuple("BlockwiseDynamicRangeStat", ["block_size", "dims", "max_over_orientations"])
):
    """Named tuple representing a blockwise dynamic range statistic configuration."""

    def __str__(self) -> str:
        """Convert to string representation for stat name. Used for logging."""
        suffix = "_max_over_orientations" if self.max_over_orientations else ""
        return f"max_blockwise_dynamic_range_block_size_{self.block_size}_dims_{self.dims}{suffix}"


@torch.compile
def _compute_dynamic_range_top(tensor):
    """Computes the log2 of the amax of the tensor"""
    tensor_abs = tensor.abs()
    tensor_abs = tensor_abs[tensor_abs != 0]
    if tensor_abs.numel() == 0:
        return torch.inf
    amax = tensor_abs.max().float()
    if not amax.all():
        amax = torch.tensor(1, device=tensor.device).to(torch.float)
    return torch.log2(amax)


@torch.compile
def _compute_dynamic_range_bottom(tensor):
    """Computes the log2 of the amin of the tensor"""
    tensor_abs = tensor.abs()
    tensor_abs = tensor_abs[tensor_abs != 0]
    if tensor_abs.any():
        amin = tensor_abs.min().float()
    else:
        amin = torch.tensor(1, device=tensor.device).to(torch.float)
    return torch.log2(amin)


def compute_max_blockwise_dynamic_range(tensor, stat_config):
    """
    Computes maximum blockwise dynamic range (log2 max/min_nonzero) within blocks.

    Flattens tensor to 2D and computes maximum dynamic range within blocks. If max_over_orientations
    is True, computes for both rowwise and columnwise orientations and returns the maximum,
    capturing the worst-case scenario regardless of how the tensor is used in GEMM operations.
    If False, computes only for rowwise orientation.

    Returns 0 if all blocks are zeros, otherwise computes dynamic range over non-zero blocks.

    Args:
        tensor: Input tensor (will be flattened to 2D)
        stat_config: BlockwiseDynamicRangeStat named tuple with:
            - block_size: Size of blocks (int)
            - dims: 1 for 1D blocks (consecutive elements), 2 for 2D blocks (tiles)
            - max_over_orientations: If True, compute max over rowwise and columnwise orientations
    """
    # Extract parameters from stat_config
    block_size = stat_config.block_size
    dims = stat_config.dims
    max_over_orientations = stat_config.max_over_orientations

    def _compute_for_one_orientation(tensor):
        total_numel = tensor.numel()
        assert dims in [1, 2], f"dims must be 1 or 2, got {dims}"

        # torch.compile friendly code - standard ** power does not work with jit
        total_block_size = block_size * block_size if dims == 2 else block_size
        assert (
            total_numel % total_block_size == 0
        ), f"Tensor numel ({total_numel}) is not divisible by block_size ({block_size})."

        tensor = tensor.abs().float()
        if dims == 1:
            tensor = tensor.reshape(-1, block_size)
            per_block_amax = tensor.amax(dim=1)
            per_block_amin = tensor.masked_fill(tensor == 0, float("inf")).amin(dim=1)
        else:
            # We want to have tensor of shape [nr_blocks, block_size, block_size],
            # where each block is a block_size x block_size tile of the original tensor.
            dim_y = tensor.shape[-1] // block_size
            tensor = (
                tensor.reshape(-1, block_size, dim_y, block_size)
                .permute(0, 2, 1, 3)
                .reshape(-1, block_size, block_size)
            )
            per_block_amax = tensor.amax(dim=(1, 2))
            per_block_amin = tensor.masked_fill(tensor == 0, float("inf")).amin(dim=(1, 2))

        # Identify blocks that contain any non-zero element
        nonzero_blocks = per_block_amax != 0
        dynamic_range_per_block = torch.where(
            nonzero_blocks,
            torch.log2(per_block_amax) - torch.log2(per_block_amin),
            torch.zeros_like(per_block_amax, dtype=torch.float32),
        )
        return dynamic_range_per_block.max()

    # Flatten to 2D
    tensor_2d = tensor.reshape(-1, tensor.shape[-1])
    if max_over_orientations:
        return max(
            _compute_for_one_orientation(tensor_2d),  # Rowwise orientation
            _compute_for_one_orientation(tensor_2d.transpose(-2, -1)),  # Columnwise orientation
        )
    return _compute_for_one_orientation(tensor_2d)


@torch.compile
def compute_variance(variances, numels, sums):
    """Welford algorithm is used for numerically stable distributed variance computation."""
    mean = torch.sum(sums) / torch.sum(numels)
    means = sums / numels
    var = torch.sum(numels * (variances - torch.pow((means - mean), 2))) / torch.sum(numels)
    return var


@torch.compile
def compute_std(variances, numels, sums):
    """Computates standard deviation."""
    return torch.sqrt(compute_variance(variances, numels, sums))


def compute_fp8_delayed_scaling_overflows_num(tensor, quantized_tensor):
    """Computes the overflows of the tensor."""
    scale_inv = quantized_tensor._scale_inv
    dtype = quantized_tensor._fp8_dtype

    # Map each supported FP8 dtype to its corresponding max forward value.
    dtype_to_max = {
        tex.DType.kFloat8E4M3: Format.E4M3.value.max_fwd,
        tex.DType.kFloat8E5M2: Format.E5M2.value.max_fwd,
    }

    if dtype not in dtype_to_max:
        raise ValueError(
            f"Unsupported FP8 dtype {dtype} passed to compute_fp8_delayed_scaling_overflows_num()."
        )

    fp8_max = dtype_to_max[dtype]
    fp8_min = -fp8_max

    overflows = (tensor > fp8_max * scale_inv) | (tensor < fp8_min * scale_inv)
    return overflows.sum()


# buffers is tensor of shape [nr_buffers, nr_stats]
def _get(buffers, stat_name):
    stat_nr = stats_to_num[stat_name]
    return buffers[:, stat_nr]


stats_to_num = {
    "min": 0,
    "max": 1,
    "sum": 2,
    "mean": 3,
    "numel": 4,
    "l1_norm": 5,
    "l2_norm_square": 6,
    "l2_norm": 7,
    "variance": 8,
    "cur_amax": 9,
    "dynamic_range_top": 10,
    "dynamic_range_bottom": 11,
    "std": 12,
    "dynamic_range": 13,
    "fp8_delayed_scaling_overflows_num": 14,
    "fp8_delayed_scaling_overflows%": 15,
    "overflows_num": 16,
    "overflows%": 17,
}

DEPENDENCIES = {
    "min": {"min"},
    "max": {"max"},
    "sum": {"sum"},
    "mean": {"sum", "numel"},
    "numel": {"numel"},
    "l1_norm": {"l1_norm"},
    "l2_norm_square": {"l2_norm_square", "numel"},
    "l2_norm": {"l2_norm_square"},
    "variance": {"variance", "numel", "sum"},
    "cur_amax": {"cur_amax"},
    "dynamic_range_top": {"dynamic_range_top"},
    "dynamic_range_bottom": {"dynamic_range_bottom"},
    "std": {"variance", "numel", "sum"},
    "dynamic_range": {"dynamic_range_top", "dynamic_range_bottom"},
    "fp8_delayed_scaling_overflows_num": {"fp8_delayed_scaling_overflows_num"},
    "fp8_delayed_scaling_overflows%": {"fp8_delayed_scaling_overflows_num", "numel"},
    "overflows_num": {"overflows_num"},
    "overflows%": {"overflows_num", "numel"},
}

STATS = {
    "min": (lambda x, aux_dict: torch.min(x), lambda buffers: min(_get(buffers, "min"))),
    "max": (lambda x, aux_dict: torch.max(x), lambda buffers: max(_get(buffers, "max"))),
    "sum": (lambda x, aux_dict: torch.sum(x), lambda buffers: sum(_get(buffers, "sum"))),
    "mean": (
        lambda x, aux_dict: torch.mean(x),
        lambda buffers: sum(_get(buffers, "sum")) / sum(_get(buffers, "numel")),
    ),
    "numel": (
        lambda x, aux_dict: x.numel() if hasattr(x, "numel") else x.get_data_tensors()[0].numel(),
        lambda buffers: sum(_get(buffers, "numel")),
    ),
    "l1_norm": (
        lambda x, aux_dict: torch.norm(x, p=1),
        lambda buffers: sum(_get(buffers, "l1_norm")),
    ),
    "l2_norm_square": (
        lambda x, aux_dict: torch.sum(x**2),
        lambda buffers: sum(_get(buffers, "l2_norm_square")),
    ),
    "l2_norm": (
        lambda x, aux_dict: torch.norm(x, p=2),
        lambda buffers: math.sqrt(sum(_get(buffers, "l2_norm_square"))),
    ),
    "variance": (
        lambda x, aux_dict: torch.var(x),
        lambda buffers: compute_variance(
            _get(buffers, "variance"), _get(buffers, "numel"), _get(buffers, "sum")
        ),
    ),
    "cur_amax": (lambda x, aux_dict: x.abs().max(), lambda buffers: max(_get(buffers, "cur_amax"))),
    "dynamic_range_top": (
        lambda x, aux_dict: _compute_dynamic_range_top(x),
        lambda buffers: max(_get(buffers, "dynamic_range_top")),
    ),
    "dynamic_range_bottom": (
        lambda x, aux_dict: _compute_dynamic_range_bottom(x),
        lambda buffers: min(_get(buffers, "dynamic_range_bottom")),
    ),
    "std": (
        lambda x, aux_dict: torch.std(x),
        lambda buffers: compute_std(
            _get(buffers, "variance"), _get(buffers, "numel"), _get(buffers, "sum")
        ),
    ),
    "dynamic_range": (
        lambda x, aux_dict: _compute_dynamic_range_top(x) - _compute_dynamic_range_bottom(x),
        lambda buffers: max(_get(buffers, "dynamic_range_top"))
        - min(_get(buffers, "dynamic_range_bottom")),
    ),
    "fp8_delayed_scaling_overflows_num": (
        lambda x, aux_dict: compute_fp8_delayed_scaling_overflows_num(
            x, aux_dict["fp8_delayed_scaling"]
        ),
        lambda buffers: sum(_get(buffers, "fp8_delayed_scaling_overflows_num")),
    ),
    "fp8_delayed_scaling_overflows%": (
        lambda x, aux_dict: compute_fp8_delayed_scaling_overflows_num(
            x, aux_dict["fp8_delayed_scaling"]
        )
        / x.numel()
        * 100,
        lambda buffers: 100
        * sum(_get(buffers, "fp8_delayed_scaling_overflows_num"))
        / sum(_get(buffers, "numel")),
    ),
    "overflows_num": (
        lambda x, aux_dict: compute_fp8_delayed_scaling_overflows_num(x, aux_dict[""]),
        lambda buffers: sum(_get(buffers, "overflows_num")),
    ),
    "overflows%": (
        lambda x, aux_dict: compute_fp8_delayed_scaling_overflows_num(x, aux_dict[""])
        / x.numel()
        * 100,
        lambda buffers: 100 * sum(_get(buffers, "overflows_num")) / sum(_get(buffers, "numel")),
    ),
}

FP8_NEGATIVE_ZERO = 128  # represnts -0.0 in fp8


def count_nonzero_fp8(fp8_data: torch.Tensor) -> torch.Tensor:
    """Count the number of non-zero elements in the fp8 data."""
    fp8_data = fp8_data.view(dtype=torch.uint8)
    zero_vals = torch.tensor([0, FP8_NEGATIVE_ZERO], device=fp8_data.device, dtype=torch.uint8)
    return fp8_data.numel() - torch.isin(fp8_data, zero_vals).sum()


def add_underflows_stats(recipe_name: str, columnwise: bool = False):
    """Register *both* underflow stats (num and %) for the given recipe."""
    columnwise_suffix = "_columnwise" if columnwise else ""

    # Stat names
    stat_num = f"{recipe_name}{'_' if recipe_name != '' else ''}underflows_num{columnwise_suffix}"
    stat_pct = f"{recipe_name}{'_' if recipe_name != '' else ''}underflows%{columnwise_suffix}"

    stats_to_num[stat_num] = len(stats_to_num)
    stats_to_num[stat_pct] = len(stats_to_num)

    STATS[stat_num] = (
        lambda x, aux_dict: x.count_nonzero()
        - count_nonzero_fp8(
            aux_dict[recipe_name].get_data_tensors(
                rowwise_data=not columnwise, columnwise_data=columnwise
            )
        ),
        lambda buffers, _sn=stat_num: sum(_get(buffers, _sn)),
    )
    STATS[stat_pct] = (
        lambda x, aux_dict: (
            x.count_nonzero()
            - count_nonzero_fp8(
                aux_dict[recipe_name].get_data_tensors(
                    rowwise_data=not columnwise, columnwise_data=columnwise
                )
            )
        )
        / aux_dict[recipe_name].numel()
        * 100,
        lambda buffers, _sn_num=stat_num: 100
        * sum(_get(buffers, _sn_num))
        / sum(_get(buffers, "numel")),
    )

    DEPENDENCIES[stat_num] = {stat_num}
    DEPENDENCIES[stat_pct] = {stat_num, "numel"}


def add_scale_inv_stats(recipe_name: str, columnwise: bool = False):
    """Register *both* scale-inv min and max stats for a given recipe.

    This replaces the earlier separate helpers and avoids duplicated boilerplate.
    """
    # Determine which attribute holds the scale-inverse tensor.

    def get_scale_inv(quantized_tensor, columnwise):
        if hasattr(quantized_tensor, "_scale_inv"):
            return getattr(quantized_tensor, "_scale_inv")
        if columnwise:
            return getattr(quantized_tensor, "_columnwise_scale_inv")
        return getattr(quantized_tensor, "_rowwise_scale_inv")

    columnwise_suffix = "_columnwise" if columnwise else ""
    # Prepare stat names.
    stat_name_min = (
        f"{recipe_name}{'_' if recipe_name != '' else ''}scale_inv_min{columnwise_suffix}"
    )
    stat_name_max = (
        f"{recipe_name}{'_' if recipe_name != '' else ''}scale_inv_max{columnwise_suffix}"
    )

    # Assign indices in `stats_to_num` (order matters — keep insertion order deterministic).
    stats_to_num[stat_name_min] = len(stats_to_num)
    stats_to_num[stat_name_max] = len(stats_to_num)

    # Capture the attribute name inside lambdas via default args to avoid late binding.
    STATS[stat_name_min] = (
        lambda x, aux_dict, _col=columnwise: get_scale_inv(aux_dict[recipe_name], _col).min(),
        lambda buffers, _sn=stat_name_min: min(_get(buffers, _sn)),
    )
    STATS[stat_name_max] = (
        lambda x, aux_dict, _col=columnwise: get_scale_inv(aux_dict[recipe_name], _col).max(),
        lambda buffers, _sn=stat_name_max: max(_get(buffers, _sn)),
    )

    DEPENDENCIES[stat_name_min] = {stat_name_min}
    DEPENDENCIES[stat_name_max] = {stat_name_max}


def add_mse_stats(recipe_name: str, columnwise: bool = False):
    """Register mse and total_square_error stats for the recipe."""
    columnwise_suffix = "_columnwise" if columnwise else ""

    stat_mse = f"{recipe_name}{'_' if recipe_name != '' else ''}mse{columnwise_suffix}"
    stat_err = (
        f"{recipe_name}{'_' if recipe_name != '' else ''}total_square_error{columnwise_suffix}"
    )

    stats_to_num[stat_mse] = len(stats_to_num)
    stats_to_num[stat_err] = len(stats_to_num)

    STATS[stat_mse] = (
        lambda x, aux_dict: F.mse_loss(x, aux_dict[recipe_name].dequantize(), reduction="mean"),
        lambda buffers, _sn_err=stat_err: torch.sum(_get(buffers, _sn_err))
        / sum(_get(buffers, "numel")),
    )
    STATS[stat_err] = (
        lambda x, aux_dict: F.mse_loss(x, aux_dict[recipe_name].dequantize(), reduction="sum"),
        lambda buffers, _sn_err=stat_err: torch.sum(_get(buffers, _sn_err)),
    )

    DEPENDENCIES[stat_err] = {stat_err}
    DEPENDENCIES[stat_mse] = {stat_mse, stat_err, "numel"}


def add_max_blockwise_dynamic_range_stats(
    block_size: int, dims: int, max_over_orientations: bool = False
):
    """Register max_blockwise_X_dynamic_range stats for the recipe.

    Args:
        block_size: Size of blocks for computing blockwise dynamic range
        dims: 1 for 1D blocks, 2 for 2D blocks
        max_over_orientations: Whether to compute max over rowwise and columnwise orientations

    Returns:
        BlockwiseDynamicRangeStat named tuple representing this stat (used as the stat key)
    """
    # Use named tuple directly as the stat key - this is cleaner than string keys
    stat_key = BlockwiseDynamicRangeStat(block_size, dims, max_over_orientations)

    if stat_key in stats_to_num:
        return stat_key  # already registered

    assert dims in [1, 2], f"dims must be 1 or 2, got {dims}"
    stats_to_num[stat_key] = len(stats_to_num)
    DEPENDENCIES[stat_key] = {stat_key}

    STATS[stat_key] = (
        lambda x, aux_dict, _stat_key=stat_key: compute_max_blockwise_dynamic_range(x, _stat_key),
        lambda buffers, _stat_key=stat_key: max(_get(buffers, _stat_key)),
    )

    return stat_key


for _columnwise in [True, False]:
    for _recipe_name in [
        "",  # default recipe
        "fp8_delayed_scaling",
        "mxfp8",
        "fp8_current_scaling",
        "fp8_block_scaling",
    ]:
        add_underflows_stats(_recipe_name, _columnwise)
        add_scale_inv_stats(_recipe_name, _columnwise)
        add_mse_stats(_recipe_name, _columnwise)


# NVFP4-specific statistics


def count_nonzero_nvfp4(fp4_data: torch.Tensor) -> torch.Tensor:
    """Count the number of non-zero elements in the FP4 data.

    FP4 data is stored as 2 4-bit values per byte (uint8).
    We need to unpack and count non-zeros.
    """
    # Each byte contains two FP4 values
    # Value 0 in FP4 E2M1 format is represented as 0 (and also 8 for -0.0)
    zero_vals = torch.tensor([0, 8], device=fp4_data.device, dtype=torch.uint8)

    # Extract first and second nibbles
    first_nibble = fp4_data % 16
    second_nibble = fp4_data // 16

    # Count zeros
    first_zeros = torch.isin(first_nibble, zero_vals).sum()
    second_zeros = torch.isin(second_nibble, zero_vals).sum()

    total_elements = fp4_data.numel() * 2
    return total_elements - first_zeros - second_zeros


def add_nvfp4_underflows_stats():
    """Register underflow stats for NVFP4.

    Computes underflows by counting zeros in packed FP4 data vs original tensor.
    """
    stat_num = "nvfp4_underflows_num"
    stat_pct = "nvfp4_underflows%"

    stats_to_num[stat_num] = len(stats_to_num)
    stats_to_num[stat_pct] = len(stats_to_num)

    # Count non-zeros in original vs FP4 packed data
    STATS[stat_num] = (
        lambda x, aux_dict: x.count_nonzero()
        - count_nonzero_nvfp4(aux_dict["nvfp4"]._rowwise_data),
        lambda buffers, _sn=stat_num: sum(_get(buffers, _sn)),
    )
    STATS[stat_pct] = (
        lambda x, aux_dict: (
            x.count_nonzero() - count_nonzero_nvfp4(aux_dict["nvfp4"]._rowwise_data)
        )
        / aux_dict["nvfp4"].numel()
        * 100,
        lambda buffers, _sn_num=stat_num: 100
        * sum(_get(buffers, _sn_num))
        / sum(_get(buffers, "numel")),
    )

    DEPENDENCIES[stat_num] = {stat_num}
    DEPENDENCIES[stat_pct] = {stat_num, "numel"}


# Register NVFP4 stats
add_nvfp4_underflows_stats()
add_mse_stats("nvfp4")  # Reuse existing MSE function


# ---------------------------------------------------------------------------
# Oscillation ratio: FP4 bin histogram EMA tracking
# ---------------------------------------------------------------------------


def _unpack_fp4_packed(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 tensor (2 FP4 values per byte) to individual 4-bit bin indices.

    Uses the same approach as TE's canonical ``unpack_fp4`` in the NVFP4
    test suite (tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py).
    Each byte encodes [lo_nibble, hi_nibble]: even columns get the low
    nibble, odd columns get the high nibble.
    """
    repeated = packed.repeat_interleave(2, dim=-1)
    repeated = repeated.clone()  # avoid modifying input
    repeated[..., 0::2] &= 0x0F
    repeated[..., 1::2] >>= 4
    return repeated


def _fp4_bin_histogram(bin_indices: torch.Tensor) -> torch.Tensor:
    """Compute 15-bin histogram from FP4 bin indices, merging +0 into -0.

    Convention: index 0 = +0, index 8 = -0. Merge +0 count into -0,
    then exclude the +0 bin, yielding 15 bins.
    """
    hist = torch.bincount(
        bin_indices.reshape(-1).long(), minlength=_FP4_E2M1_NUM_BINS
    ).clone()
    hist[_FP4_E2M1_NEG_ZERO_INDEX] += hist[0]
    return hist[1:]  # 15 bins


def _master_w_bin_histogram(bin_indices: torch.Tensor) -> torch.Tensor:
    """Compute 29-bin histogram from finer master-weight bin indices, merging +0 into -0.

    Convention mirrors _fp4_bin_histogram: index 0 = +0, index 15 = -0.
    Merge +0 count into -0, then exclude the +0 bin, yielding 29 bins.
    """
    hist = torch.bincount(
        bin_indices.reshape(-1).long(), minlength=_MASTER_W_NUM_BINS
    ).clone()
    hist[_MASTER_W_NEG_ZERO_INDEX] += hist[0]
    return hist[1:]  # 29 bins


def _reduce_master_histogram_29_to_15(hist_29: torch.Tensor) -> torch.Tensor:
    """Reduce 29-bin master histogram to the same 15-bin space as qw."""
    if hist_29.dim() != 1 or hist_29.shape[0] != 29:
        raise ValueError("hist_29 must be 1D of length 29")
    out = hist_29.new_zeros(15)
    for i, coarse_idx in enumerate(_MASTER_29_TO_15_COARSE_INDEX):
        out[coarse_idx] += hist_29[i]
    return out


def _latent_to_master_bin_index(latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Map latent values to finer master-weight bin indices (0-29).

    Uses 2× FP4 resolution with quarter-point boundaries so that FP4
    quantization thresholds fall at bin centers.  Small oscillations near
    a boundary stay in one master bin.

    Positive bins 0-14, negative bins 15-29.
    """
    boundaries = torch.tensor(
        _MASTER_W_POS_BOUNDARIES, dtype=torch.float32, device=device
    )
    pos_bin = torch.bucketize(latent.abs(), boundaries)  # 0..14
    return torch.where(latent < 0, pos_bin + _MASTER_W_NEG_ZERO_INDEX, pos_bin)


def _latent_to_fp4_bin_index(latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Map latent (encoded) values to FP4 E2M1 bin indices (0-15).

    Uses torch.bucketize on absolute values for memory efficiency (avoids
    materialising an (N, 16) distance matrix).

    Positive bins 0-7 map to FP4 values [0, 0.5, 1, 1.5, 2, 3, 4, 6].
    Negative bins 8-15 mirror the positive bins.
    """
    boundaries = torch.tensor(_FP4_E2M1_POS_BOUNDARIES, dtype=torch.float32, device=device)
    pos_bin = torch.bucketize(latent.abs(), boundaries)  # 0..7
    return torch.where(latent < 0, pos_bin + _FP4_E2M1_NEG_ZERO_INDEX, pos_bin)


def nvfp4_qw_histogram(packed_data: torch.Tensor) -> Optional[torch.Tensor]:
    """Build 15-bin histogram directly from packed NVFP4 quantized data.

    Args:
        packed_data: uint8 tensor, 2 FP4 values per byte.
    Returns:
        Histogram (15,) long tensor, or None if input is invalid/empty.
    """
    if packed_data is None or packed_data.numel() == 0 or packed_data.dtype != torch.uint8:
        return None
    return _fp4_bin_histogram(_unpack_fp4_packed(packed_data))


def nvfp4_w_histogram(
    w: torch.Tensor,
    scale_inv_uint8: torch.Tensor,
    amax: torch.Tensor,
    block_size: int = 16,
) -> Optional[torch.Tensor]:
    """Build 29-bin histogram of master weights using 2× FP4 resolution.

    Uses quarter-point boundaries so FP4 quantization thresholds land at
    bin centers.  This makes the master histogram insensitive to small
    oscillations near FP4 bin boundaries while the quantized histogram
    flips — improving oscillation-ratio detection.

    Encoding: latent = w * encode_scale, where
        encode_scale = _NVFP4_SCALE_FACTOR / (amax * float(per_block_scale_inv)).

    Args:
        w: Master weight, shape (M, K).
        scale_inv_uint8: Per-block scale_inv as uint8 (FP8 E4M3 encoded),
            shape (M_padded, ceil(K/block_size)_padded).
        amax: Per-tensor amax, float32 scalar tensor.
        block_size: Elements per quantisation block (16 for NVFP4).
    Returns:
        Histogram (29,) long tensor, or None if incompatible/empty.
    """
    if (
        w is None
        or w.numel() == 0
        or scale_inv_uint8 is None
        or amax is None
        or w.dim() != 2
    ):
        return None
    amax_val = amax.float().item()
    if amax_val == 0:
        return None
    M, K = w.shape
    if K % block_size != 0:
        return None
    num_k_blocks = K // block_size

    scale_inv_view = scale_inv_uint8[:M, :num_k_blocks]
    boundaries = torch.tensor(_MASTER_W_POS_BOUNDARIES, dtype=torch.float32, device=w.device)
    hist = torch.zeros(_MASTER_W_NUM_BINS, dtype=torch.long, device=w.device)
    any_valid = False

    row_chunk = max(1, _HISTOGRAM_LATENT_CHUNK_ELEMS // K)
    for r_start in range(0, M, row_chunk):
        r_end = min(r_start + row_chunk, M)
        w_chunk = w[r_start:r_end, :].float()
        scale_chunk = (
            scale_inv_view[r_start:r_end, :]
            .repeat_interleave(block_size, dim=1)[:, :K]
            .contiguous()
            .view(torch.float8_e4m3fn)
            .to(torch.float32)
        )
        safe_scale = scale_chunk.clamp(min=1e-30)
        encode_scale = _NVFP4_SCALE_FACTOR / (amax_val * safe_scale)
        latent_chunk = (w_chunk * encode_scale).reshape(-1)
        valid = torch.isfinite(latent_chunk)
        if not valid.any():
            continue
        any_valid = True
        latent_chunk = latent_chunk[valid]
        pos_bin = torch.bucketize(latent_chunk.abs(), boundaries)
        bin_idx = torch.where(latent_chunk < 0, pos_bin + _MASTER_W_NEG_ZERO_INDEX, pos_bin)
        hist += torch.bincount(bin_idx, minlength=_MASTER_W_NUM_BINS)

    if not any_valid:
        return None

    hist[_MASTER_W_NEG_ZERO_INDEX] += hist[0]
    return hist[1:]  # 29 bins


def _compute_oscillation_stat(
    tensor: torch.Tensor, aux_dict: dict, variant: str
) -> float:
    """Compute one oscillation scalar using persistent buffer state.

    All oscillation variants are computed together on the first call per feed()
    and cached in persistent_state for the remaining sibling stats.

    Args:
        tensor: Original (master) weight tensor.
        aux_dict: Must contain ``_persistent_state`` (dict surviving across
            iterations) and ``_nvfp4_quantized_tensor`` (the NVFP4Tensor).
        variant: One of ``"oscillation_ratio"``, ``"oscillation_ratio_reduced"``,
            ``"oscillation_l1distqw"``, ``"oscillation_l1distw"``, or
            ``"oscillation_l1distw_reduced"``.
    """
    persistent_state = aux_dict.get("_persistent_state")
    if persistent_state is None:
        return 0.0

    # Per-feed cache: avoid recomputing histograms for the sibling oscillation stats.
    iteration = aux_dict.get("_iteration")
    cache = persistent_state.get("_oscillation_cache")
    if cache is not None and cache.get("_iter") == iteration:
        return cache.get(variant, 0.0)

    _ZERO_RESULT = {
        "_iter": iteration,
        "oscillation_ratio": 0.0,
        "oscillation_ratio_reduced": 0.0,
        "oscillation_l1distqw": 0.0,
        "oscillation_l1distw": 0.0,
        "oscillation_l1distw_reduced": 0.0,
    }

    nvfp4_tensor = aux_dict.get("_nvfp4_quantized_tensor")
    if nvfp4_tensor is None:
        persistent_state["_oscillation_cache"] = _ZERO_RESULT
        return 0.0

    packed_data = getattr(nvfp4_tensor, "_rowwise_data", None)
    scale_inv = getattr(nvfp4_tensor, "_rowwise_scale_inv", None)
    amax = getattr(nvfp4_tensor, "_amax_rowwise", None)

    hist_qw = nvfp4_qw_histogram(packed_data)
    hist_w = nvfp4_w_histogram(tensor, scale_inv, amax)

    if hist_qw is None or hist_w is None:
        persistent_state["_oscillation_cache"] = _ZERO_RESULT
        return 0.0

    total_qw = hist_qw.sum().item()
    total_w = hist_w.sum().item()
    if total_qw <= 0 or total_w <= 0:
        persistent_state["_oscillation_cache"] = _ZERO_RESULT
        return 0.0

    new_qw_norm = hist_qw.float() / total_qw
    new_w_norm = hist_w.float() / total_w

    ema_decay = aux_dict.get("_ema_decay", 0.0)
    ema_qw = persistent_state.get("_ema_qw")
    ema_w = persistent_state.get("_ema_w")

    # Handle histogram size change (e.g. 15-bin → 29-bin master upgrade).
    if ema_w is not None and ema_w.shape != new_w_norm.shape:
        ema_qw = None
        ema_w = None

    if ema_qw is None or ema_w is None:
        # First iteration: seed EMAs, report zeros.
        persistent_state["_ema_qw"] = new_qw_norm.clone()
        persistent_state["_ema_w"] = new_w_norm.clone()
        persistent_state["_oscillation_cache"] = _ZERO_RESULT
        return 0.0

    # L1 deltas vs previous EMA (before update).
    l1distqw = (new_qw_norm - ema_qw).abs().sum().item()
    l1distw = (new_w_norm - ema_w).abs().sum().item()
    new_w_reduced = _reduce_master_histogram_29_to_15(new_w_norm)
    ema_w_reduced = _reduce_master_histogram_29_to_15(ema_w)
    l1distw_reduced = (new_w_reduced - ema_w_reduced).abs().sum().item()

    # Update EMAs in-place.
    ema_qw.mul_(ema_decay).add_(new_qw_norm, alpha=1.0 - ema_decay)
    ema_w.mul_(ema_decay).add_(new_w_norm, alpha=1.0 - ema_decay)

    eps = 1e-12
    ratio = l1distqw / (l1distw + eps) if l1distw > 0 else 0.0
    ratio_reduced = l1distqw / (l1distw_reduced + eps) if l1distw_reduced > 0 else 0.0

    result = {
        "_iter": iteration,
        "oscillation_ratio": ratio,
        "oscillation_ratio_reduced": ratio_reduced,
        "oscillation_l1distqw": l1distqw,
        "oscillation_l1distw": l1distw,
        "oscillation_l1distw_reduced": l1distw_reduced,
    }
    persistent_state["_oscillation_cache"] = result
    return result[variant]


def add_oscillation_ratio_stats():
    """Register kitchen-aligned oscillation stats for NVFP4 tensors."""
    if OSCILLATION_STAT_NAMES.issubset(stats_to_num):
        return

    for stat_name in OSCILLATION_STAT_NAMES:
        if stat_name not in stats_to_num:
            stats_to_num[stat_name] = len(stats_to_num)
        DEPENDENCIES[stat_name] = {stat_name}
        # Each variant delegates to the shared helper which caches per-feed results.
        STATS[stat_name] = (
            lambda x, aux_dict, variant=stat_name: _compute_oscillation_stat(x, aux_dict, variant),
            lambda buffers, variant=stat_name: torch.mean(_get(buffers, variant)),
        )


add_oscillation_ratio_stats()
