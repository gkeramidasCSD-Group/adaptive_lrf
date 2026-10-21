"""Dimension-generic FLOPs/rank-sizing formulas for the Tucker/Conv1x1-SVD
low-rank layers in modules/lrf.py.

Rank-from-compression-ratio and FLOPs formulas for the SVD/Tucker layers
in modules/lrf.py: the Linear helpers (calculate_fc_ranks, calculate_fc_flops,
calculate_svd_fc_flops) and the conv helpers, which take a single
`spatial_size`/`kernel_size` that's either an int (1 spatial dim, e.g.
Conv1d) or an N-tuple (N spatial dims, e.g. Conv2d's (H, W)).

Only 1 and 2 spatial dims are supported for the conv helpers. Conv3d would
need the quadratic rank-sizing formula re-derived and verified first, so
it raises NotImplementedError rather than silently extrapolating.
"""

import math

import torch


def _spatial_tuple(spatial_size):
    """spatial_size: int (1 spatial dim) or tuple/list (N spatial dims).
    Output: always a tuple."""
    return (spatial_size,) if isinstance(spatial_size, int) else tuple(spatial_size)


def _prod(sizes):
    """Output: the product of all values in `sizes`."""
    n = 1
    for s in sizes:
        n *= s
    return n


def _resolve_ndims(*spatial_sizes):
    """Checks that every spatial size in `spatial_sizes` has the same
    number of dims, and that it's 1 or 2. Output: that shared ndims.
    Raises ValueError on a mismatch, NotImplementedError for ndims > 2."""
    ndims_seen = {len(_spatial_tuple(s)) for s in spatial_sizes}
    if len(ndims_seen) != 1:
        raise ValueError(f"inconsistent number of spatial dimensions among {spatial_sizes!r}")
    ndims = ndims_seen.pop()
    if ndims not in (1, 2):
        raise NotImplementedError(
            f"calculate_conv2D_tucker_rank/calculate_tucker_flops/etc. only support "
            f"1 or 2 spatial dimensions right now (got {ndims}-d spatial size(s) "
            f"{spatial_sizes!r}). Conv3d support would need this formula re-derived "
            f"and verified first."
        )
    return ndims


def calculate_conv2D_tucker_rank(cp, in_spatial_size, out_spatial_size, in_channels, out_channels, kernel_size):
    """Smallest rank whose Tucker FLOPs fit within compression ratio `cp`
    (solves the quadratic cost balance, clamped to [1, min(in, out)]).

    Input: cp (fraction of FLOPs removed), spatial sizes and kernel size
    as int (1D) or tuple (2D), channel counts. Output: an int rank."""
    _resolve_ndims(in_spatial_size, out_spatial_size, kernel_size)
    spatial_in = _prod(_spatial_tuple(in_spatial_size))
    spatial_out = _prod(_spatial_tuple(out_spatial_size))
    kernel_vol = _prod(_spatial_tuple(kernel_size))

    a = spatial_out * kernel_vol
    b = (spatial_in * in_channels) + (spatial_out * out_channels)
    c_partial = (kernel_vol * in_channels + 0.5) * (spatial_out * out_channels)
    c = -1 * (((1 - cp) * c_partial) + (spatial_out * out_channels / 2))
    discriminant = b**2 - (4 * a * c)
    in_out_rank = int((-1 * b + math.sqrt(discriminant)) / (2 * a))
    if in_out_rank <= 0 or in_out_rank > min(in_channels, out_channels):
        temp = int((-1 * b - math.sqrt(discriminant)) / (2 * a))
        if temp > 0:
            in_out_rank = temp

    if in_out_rank < 1:
        in_out_rank = 1

    if in_out_rank > min(in_channels, out_channels):
        in_out_rank = min(in_channels, out_channels)

    return in_out_rank


def calculate_conv2D_tucker_ranks(compressions_list, in_spatial_size, out_spatial_size, in_channels, out_channels, kernel_size):
    """calculate_conv2D_tucker_rank, applied to each ratio in
    `compressions_list`. Output: a list of ranks, same length/order as
    `compressions_list`."""
    return [calculate_conv2D_tucker_rank(cp, in_spatial_size, out_spatial_size, in_channels, out_channels, kernel_size)
            for cp in compressions_list]


def calculate_Conv2D_flops(out_spatial_size, in_channels, out_channels, kernel_size):
    """Generalized version of calculate_Conv2D_flops (uncompressed/full conv FLOPs)."""
    spatial_out = _prod(_spatial_tuple(out_spatial_size))
    kernel_vol = _prod(_spatial_tuple(kernel_size))
    return 2 * spatial_out * out_channels * (kernel_vol * in_channels + 0.5)


def calculate_tucker_flops(ranks, in_spatial_size, out_spatial_size, in_channels, out_channels, kernel_size):
    """FLOPs of a Tucker-decomposed conv layer, for each rank in `ranks`
    (assuming the same rank on both the in- and out-channel modes;
    spatial dims are never decomposed). `rank == math.inf` is the
    original-weight choice, computed via calculate_Conv2D_flops instead.

    Output: a list of FLOPs counts (or a single int if `ranks` has/is
    exactly one value), same length/order as `ranks`."""
    _resolve_ndims(in_spatial_size, out_spatial_size, kernel_size)
    spatial_in = _prod(_spatial_tuple(in_spatial_size))
    spatial_out = _prod(_spatial_tuple(out_spatial_size))
    kernel_vol = _prod(_spatial_tuple(kernel_size))

    if isinstance(ranks, int):
        ranks = [ranks]
    flops_per_rank = []
    for rank in ranks:
        if rank == math.inf:
            flops = calculate_Conv2D_flops(out_spatial_size, in_channels, out_channels, kernel_size)
        else:
            flops = (2 * rank * spatial_in * in_channels) + (2 * rank * rank * spatial_out * kernel_vol) + (2 * rank * spatial_out * out_channels) + (spatial_out * out_channels)
            if isinstance(flops, torch.Tensor):
                flops = flops.item()
        flops_per_rank.append(int(round(flops)))
    return flops_per_rank if len(flops_per_rank) > 1 else flops_per_rank[0]


def calculate_conv1x1_flops(spatial_size, in_channels, out_channels):
    """FLOPs of an uncompressed pointwise conv layer. Output: an int."""
    return 2 * _prod(_spatial_tuple(spatial_size)) * in_channels * out_channels


def calculate_svd_conv1x1_flops(ranks, spatial_size, in_channels, out_channels):
    """FLOPs of an SVD-decomposed pointwise conv layer, for each rank in
    `ranks`. `rank == math.inf` is the original-weight choice, computed
    via calculate_conv1x1_flops instead.

    Output: a list of FLOPs counts (or a single int if `ranks` has/is
    exactly one value), same length/order as `ranks`."""
    spatial = _prod(_spatial_tuple(spatial_size))
    if isinstance(ranks, int):
        ranks = [ranks]
    flops_per_rank = []
    for rank in ranks:
        if rank == min(in_channels, out_channels):
            flops = 2 * spatial * in_channels * out_channels
        elif rank == math.inf:
            # the "keep original weight" choice (see calculate_svd_fc_flops for the same handling)
            flops = calculate_conv1x1_flops(spatial_size, in_channels, out_channels)
        else:
            flops = rank * spatial * ((2 * in_channels) + 1 + (2 * out_channels))
            if isinstance(flops, torch.Tensor):
                flops = flops.item()
        flops_per_rank.append(int(round(flops)))
    return flops_per_rank if len(flops_per_rank) > 1 else flops_per_rank[0]


def calculate_fc_ranks(compressions, in_features, out_features):
    """For each compression ratio, the SVD rank of a Linear layer, floored at 1.
    Output: a list of ints, same order as `compressions`."""
    ranks = []
    for comp in compressions:
        rank = int((1-comp) * ((in_features * out_features)/(in_features + 1 + out_features)))
        if rank < 1:
            rank = 1
        ranks.append(rank)
    return ranks


def calculate_fc_flops(in_features, out_features):
    """FLOPs of an uncompressed Linear layer. Output: an int."""
    return 2 * in_features * out_features


def calculate_svd_fc_flops(ranks, in_features, out_features):
    """FLOPs of a Linear layer at each SVD rank in `ranks`. `rank == math.inf`
    is the original-weight choice (uses calculate_fc_flops).

    Output: a list of ints, one per rank (or a single int if `ranks` has/is
    exactly one value)."""
    if isinstance(ranks, int):
        ranks = [ranks]
    flops_per_rank = []
    for rank in ranks:
        if rank == min(in_features, out_features):
            flops = 2 * in_features * out_features
        elif rank == math.inf:
            flops = calculate_fc_flops(in_features, out_features)
        else:
            flops = 2 * rank * (in_features + out_features)
            if isinstance(flops, torch.Tensor):
                flops = flops.item()
        flops_per_rank.append(int(round(flops)))
    return flops_per_rank if len(flops_per_rank) > 1 else flops_per_rank[0]
