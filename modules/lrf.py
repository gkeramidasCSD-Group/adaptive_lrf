"""Low-rank replacements for nn.Linear/pointwise-conv(1x1)/general conv,
each exposing a set of compression-ratio "choices" that the RNN gate
(arch/gates.py) selects between at every forward pass. Every layer type
comes in two weight-sharing flavors, matching the "plain" / "_unified"
architecture variants in arch/models_adaptive*.py:

  - plain (<Layer>_<Method>_NonUnified): one independent SVD/Tucker
    decomposition per compression choice (built from the corresponding
    <Layer>_<Method>_per_rank helper), plus an optional "keep the
    original weight" choice (<Layer>_<Method>_OriginalWeight), controlled
    by the `include_original_weight` constructor argument (and, at the
    architecture level, the `include_original_weight` config field).
  - <Layer>_<Method>_Unified: a single shared low-rank basis at the
    largest allowed rank; smaller compression choices just use a
    truncated slice of that one basis (fewer total parameters).
    `include_original_weight` doesn't apply here - there's no separate
    per-choice storage to add an extra choice to.

Conv1x1_SVD_*/Conv_Tucker_* are dimension-generic: `spatial_size` (or
in_spatial_size/out_spatial_size/kernel_size for the Tucker family) is an
int for 1 spatial dim (Conv1d) or an N-tuple for N spatial dims (Conv2d's
(H, W)), and the class dispatches to F.conv1d/conv2d internally based on
that - Conv3d isn't wired up yet (see modules/flops_utils.py). Linear has
no spatial dims to generalize and is unaffected.

Every class's forward(x, cr) takes the gate's selected choice: 99 uses 
the min-rank choice (only present if the layer was built with a `min_rank`),
0 uses the original weight (only present if `include_original_weight`), 
and any other float is a compression ratio from the layer's 
`compression_ratio_list`. cr=None means "no choice given": on a Unified 
layer that's the shared basis truncated to its max_rank (a real, near-full-rank
slice); on a NonUnified layer it's the 0 (original weight) choice if
`include_original_weight`, otherwise the least-compressed ratio in
`compression_ratio_list` - still a compressed, per-rank choice, not a
full-rank one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.flops_utils import calculate_conv2D_tucker_rank


def _as_spatial_tuple(spatial_size):
    """spatial_size: int (1 spatial dim, e.g. Conv1d) or a tuple/list of
    ints (N spatial dims, e.g. Conv2d's (H, W))."""
    return (spatial_size,) if isinstance(spatial_size, int) else tuple(spatial_size)


def _spatial_ndims(spatial_size):
    return len(_as_spatial_tuple(spatial_size))


def _resolve_kernel_dims(kernel_size, ndims):
    """kernel_size: an int, broadcast to all `ndims` spatial dims (like
    nn.Conv2d(kernel_size=3) meaning 3x3) - unlike spatial_size, a bare int
    here is a *value* repeated across dims, not "1 spatial dim" - or an
    explicit N-tuple that must already match `ndims`."""
    if isinstance(kernel_size, int):
        return (kernel_size,) * ndims
    kernel_dims = tuple(kernel_size)
    if len(kernel_dims) != ndims:
        raise ValueError(f"kernel_size {kernel_size!r} doesn't match {ndims} spatial dim(s)")
    return kernel_dims


def _conv_nd(ndims):
    """The F.convNd matching however many spatial dims this layer has."""
    return getattr(F, f'conv{ndims}d')


class Linear_SVD_Unified(nn.Module):
    """Linear layer with a single shared low-rank basis at max rank; compression choices slice it (variant used by the "_unified" architectures)."""
    def __init__(self, in_features, out_features, compression_ratio_list, min_rank=[], bias=None):
        super(Linear_SVD_Unified, self).__init__()
        self.register_buffer("in_features", torch.tensor(in_features, dtype=torch.int))
        self.register_buffer("out_features", torch.tensor(out_features, dtype=torch.int))
        if min_rank != []:
            self.register_buffer("min_rank", torch.tensor(min_rank, dtype=torch.int))
        else:
            self.min_rank = None

        min_compression_ratio = min(k for k in compression_ratio_list)
        max_rank = int((1-min_compression_ratio) * ((self.in_features.item() * self.out_features.item())/(self.in_features.item() + 1 + self.out_features.item())))
        self.register_buffer("max_rank", torch.tensor(max_rank, dtype=torch.int))

        self.U_full = nn.Parameter(torch.zeros(out_features, self.max_rank), requires_grad=True)        # [out_features, max_rank]
        self.S_full = nn.Parameter(torch.zeros(self.max_rank), requires_grad=True)                      # [max_ranknk]
        self.Vh_full = nn.Parameter(torch.zeros(self.max_rank, in_features), requires_grad=True)        # [max_rank, in_features]
        if bias is not None:
            self.SVD_components_bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)                       # [out_features]
        else:
            self.SVD_components_bias = None

    def set_svd_weights(self, U, S, Vh, bias):
        self.U_full.data.copy_(U[:, :self.max_rank])
        self.S_full.data.copy_(S[:self.max_rank])
        self.Vh_full.data.copy_(Vh[:self.max_rank, :])
        if self.SVD_components_bias is not None:
            self.SVD_components_bias.data.copy_(bias)

    def forward(self, x, cr = None):
        # cr is the reduction ratio. cr 90 means we keep 10 percent of computations
        if cr is None:# or cr == 0:
            rank = self.max_rank
        else:
            if cr == 99:
                rank = self.min_rank
            else:
                rank = int((1-cr) * ((self.in_features.item() * self.out_features.item())/(self.in_features.item() + 1 + self.out_features.item())))

        U = self.U_full[:, :rank]  # [out_features, rank]
        S = self.S_full[:rank]     # [rank]
        Vh = self.Vh_full[:rank, :]  # [in_features, rank]

        x = F.linear(x, Vh)                                       # [batch, rank]
        x = x * S                                                 # element-wise scale by S [batch, rank]
        out = F.linear(x, U, bias = self.SVD_components_bias)     # [batch, out_features]
        return out
class Linear_SVD_NonUnified(nn.Module):
    """Linear layer with one independent SVD decomposition per compression choice, plus an optional "keep original weight" choice."""
    def __init__(self, in_features, out_features, compression_ratio_list, min_rank=[], bias=None, include_original_weight=True):
        super(Linear_SVD_NonUnified, self).__init__()
        self.register_buffer("in_features", torch.tensor(in_features, dtype=torch.int))
        self.register_buffer("out_features", torch.tensor(out_features, dtype=torch.int))
        if min_rank != []:
            self.register_buffer("min_rank", torch.tensor(min_rank, dtype=torch.int))
        else:
            self.min_rank = None
        self.include_original_weight = include_original_weight
        self.options_object_dictionary = nn.ModuleDict()
        if min_rank != []:
            svd_object = Linear_SVD_per_rank(in_features, out_features, min_rank)
            self.options_object_dictionary[str(99)] = svd_object

        for compression_ratio in compression_ratio_list:
            rank = int((1-compression_ratio) * ((in_features * out_features)/(in_features + 1 + out_features)))
            if rank < 1:
                rank = 1
            svd_object = Linear_SVD_per_rank(in_features, out_features, rank, bias= bias)
            self.options_object_dictionary[str(int(compression_ratio*100))] = svd_object

        if include_original_weight:
            original_object = Linear_SVD_OriginalWeight(in_features, out_features, bias = bias)
            self.options_object_dictionary[str(0)] = original_object

        key = min(float(k) for k in self.options_object_dictionary.keys() if float(k) != 0)
        min_compression_ratio = key/100
        max_rank = int((1-min_compression_ratio) * ((self.in_features.item() * self.out_features.item())/(self.in_features.item() + 1 + self.out_features.item())))
        self.register_buffer("max_rank", torch.tensor(max_rank, dtype=torch.int))

    def set_svd_weights(self, W, U, S, Vh, bias):
        for compression_ratio in self.options_object_dictionary.keys():
            if compression_ratio == str(99):
                self.options_object_dictionary[str(99)].set_svd_weights(U[:, :self.min_rank.item()], S[:self.min_rank.item()], Vh[:self.min_rank.item(), :], bias)
            elif compression_ratio == str(0):
                self.options_object_dictionary[str(0)].set_svd_weights(W, bias)
            else:
                compression_ratio = float(compression_ratio) / 100
                rank = int((1-compression_ratio) * ((self.in_features.item() * self.out_features.item())/(self.in_features.item() + 1 + self.out_features.item())))
                if rank < 1:
                    rank = 1
                self.options_object_dictionary[str(int(compression_ratio*100))].set_svd_weights(U[:, :rank], S[:rank], Vh[:rank, :], bias)

    def forward(self, x, cr = None):
        # cr is the reduction ratio. cr 90 means we keep 10 percent of computations
        if cr == 99:
            return self.options_object_dictionary[str(99)].forward(x)
        if cr is None:
            if self.include_original_weight:
                return self.options_object_dictionary[str(0)].forward(x)
            key = min(float(k) for k in self.options_object_dictionary.keys() if k != str(99))
            return self.options_object_dictionary[str(int(key))].forward(x)
        if cr == 0 and self.include_original_weight:
            return self.options_object_dictionary[str(0)].forward(x)
        return self.options_object_dictionary[str(int(cr*100))].forward(x)

class Linear_SVD_per_rank(nn.Module):
    """A single fixed-rank SVD factorization (U, S, Vh) of one Linear layer."""
    def __init__(self, in_features, out_features, rank, bias=True):
        super(Linear_SVD_per_rank, self).__init__()
        self.register_buffer("in_features", torch.tensor(in_features, dtype=torch.int))
        self.register_buffer("out_features", torch.tensor(out_features, dtype=torch.int))
        self.register_buffer("rank", torch.tensor(rank, dtype=torch.int))


        self.U_full = nn.Parameter(torch.zeros(out_features, rank), requires_grad=True)       # [out_features, rank]
        self.S_full = nn.Parameter(torch.zeros(rank), requires_grad=True)       # [rank]
        self.Vh_full = nn.Parameter(torch.zeros(rank, in_features), requires_grad=True)      # [rank, in_features]
        if bias is not None:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)    # [out_features]
        else:
            self.bias = None
    def set_svd_weights(self, U, S, Vh, bias):
        self.U_full.data.copy_(U)
        self.S_full.data.copy_(S)
        self.Vh_full.data.copy_(Vh)
        if self.bias is not None:
            self.bias.data.copy_(bias)

    def forward(self, x):
        # x: [batch_size, in_features]
        x = F.linear(x, self.Vh_full)  # [batch, rank]
        x = x * self.S_full          # element-wise scale by S [batch, rank]
        out = F.linear(x, self.U_full, bias = self.bias)  # [batch, out_features]
        return out
class Linear_SVD_OriginalWeight(nn.Module):
    """Wraps the uncompressed original Linear weight as one of the gate's choices."""
    def __init__(self, in_features, out_features, min_rank = 1, bias=True):
        super(Linear_SVD_OriginalWeight, self).__init__()
        self.register_buffer("in_features", torch.tensor(in_features, dtype=torch.int))
        self.register_buffer("out_features", torch.tensor(out_features, dtype=torch.int))
        self.Original_weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=True)       # [out_features, in_features]
        if bias is not None:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)    # [out_features]
        else:
            self.bias = None

    def set_svd_weights(self, W, bias):
        self.Original_weight.data.copy_(W)
        if self.bias is not None:
            self.bias.data.copy_(bias)
    def forward(self, x):
        out = F.linear(x, self.Original_weight, bias = self.bias)  # [batch, out_features]
        return out


class Conv1x1_SVD_Unified(nn.Module):
    """Pointwise conv (1x1 for Conv2d, 1 for Conv1d - any number of
    spatial dims) with a single shared low-rank basis at max rank;
    compression choices slice it (variant used by the "_unified"
    architectures). `spatial_size` is an int (1 spatial dim) or an N-tuple
    (N spatial dims) - it only decides how many (always-size-1) spatial
    dims this layer's kernels carry, no value here depends on it."""
    def __init__(self, spatial_size, in_channels, out_channels, compression_ratio_list, min_rank=[], bias=None):
        super(Conv1x1_SVD_Unified, self).__init__()
        self.register_buffer("in_channels", torch.tensor(in_channels, dtype=torch.int))
        self.register_buffer("out_channels", torch.tensor(out_channels, dtype=torch.int))
        self.register_buffer("spatial_size", torch.tensor(_as_spatial_tuple(spatial_size), dtype=torch.int))
        ndims = _spatial_ndims(spatial_size)
        self._conv = _conv_nd(ndims)
        kernel_ones = (1,) * ndims
        if min_rank != []:
            self.register_buffer("min_rank", torch.tensor(min_rank, dtype=torch.int))
        else:
            self.min_rank = None

        min_compression_ratio = min(k for k in compression_ratio_list)
        self.max_rank = int((1-min_compression_ratio) * ((self.in_channels.item() * self.out_channels.item())/(self.in_channels.item() + 1 + self.out_channels.item())))
        self.U_full = nn.Parameter(torch.zeros(out_channels, self.max_rank, *kernel_ones), requires_grad=True)   # [out_c, rank]
        self.S_full = nn.Parameter(torch.zeros(1, self.max_rank, *kernel_ones), requires_grad=True)                 # [rank]
        self.Vh_full = nn.Parameter(torch.zeros(self.max_rank, in_channels, *kernel_ones), requires_grad=True)   # [rank, in_c]

        if bias is not None:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None

    def set_svd_weights(self, U, S, Vh, bias):
        # Copy the max_rank subset of decomposition components
        self.Vh_full.data.copy_(Vh[:self.max_rank, ...])         # [r, in, *spatial]
        self.S_full.data.copy_(S[:, :self.max_rank, ...])           # [1, r, *spatial]
        self.U_full.data.copy_(U[:, :self.max_rank, ...])           # [out, r, *spatial]
        if self.bias is not None:
            self.bias.data.copy_(bias)

    def forward(self, x, cr = None):
        # cr is the reduction ratio. cr 90 means we keep 10 percent of computations
        if cr is None:# or cr == 0:
            rank = self.max_rank
        else:
            if cr == 99:
                rank = self.min_rank
            else:
                rank = int((1-cr) * ((self.in_channels.item() * self.out_channels.item())/(self.in_channels.item() + 1 + self.out_channels.item())))
        U = self.U_full[:, :rank, ...]  # [out_features, rank]
        S = self.S_full[:, :rank, ...]     # [rank]
        Vh = self.Vh_full[:rank, ...]  # [in_features, rank]

        x = self._conv(x, Vh, bias=None)       # [B, rank, *spatial]
        x = x * S                # [B, rank, *spatial]
        out = self._conv(x, U, bias=self.bias)   # [B, out_c, *spatial]
        return out
class Conv1x1_SVD_NonUnified(nn.Module):
    """Pointwise conv (any number of spatial dims) with one independent
    SVD decomposition per compression choice, plus an explicit "keep
    original weight" choice. `spatial_size`: see Conv1x1_SVD_Unified."""
    def __init__(self, spatial_size, in_channels, out_channels, compression_ratio_list, min_rank=[], bias=None, include_original_weight=True):
        super(Conv1x1_SVD_NonUnified, self).__init__()
        self.register_buffer("in_channels", torch.tensor(in_channels, dtype=torch.int))
        self.register_buffer("out_channels", torch.tensor(out_channels, dtype=torch.int))
        self.register_buffer("spatial_size", torch.tensor(_as_spatial_tuple(spatial_size), dtype=torch.int))
        if min_rank != []:
            self.register_buffer("min_rank", torch.tensor(min_rank, dtype=torch.int))
        else:
            self.min_rank = None
        self.include_original_weight = include_original_weight

        self.options_object_dictionary = nn.ModuleDict()
        if min_rank != []:
            svd_object = Conv1x1_SVD_per_rank(spatial_size, in_channels, out_channels, min_rank, bias = bias)
            self.options_object_dictionary[str(99)] = svd_object

        for compression_ratio in compression_ratio_list:
            rank = int((1-compression_ratio) * ((in_channels * out_channels)/(in_channels + 1 + out_channels)))
            if rank < 1:
                rank = 1
            svd_object = Conv1x1_SVD_per_rank(spatial_size, in_channels, out_channels, rank, bias= bias)
            self.options_object_dictionary[str(int(compression_ratio*100))] = svd_object

        if include_original_weight:
            original_object = Conv1x1_SVD_OriginalWeight(spatial_size, in_channels, out_channels, bias = bias)
            self.options_object_dictionary[str(0)] = original_object

        key = min(float(k) for k in self.options_object_dictionary.keys() if float(k) != 0)
        min_compression_ratio = key/100
        max_rank = int((1-min_compression_ratio) * ((self.in_channels.item() * self.out_channels.item())/(self.in_channels.item() + 1 + self.out_channels.item())))
        self.register_buffer("max_rank", torch.tensor(max_rank, dtype=torch.int))

    def set_svd_weights(self, W, U, S, Vh, bias):
        for compression_ratio in self.options_object_dictionary.keys():
            if compression_ratio == str(99):
                self.options_object_dictionary[str(99)].set_svd_weights(U[:, :self.min_rank.item(), ...], S[:, :self.min_rank.item(), ...], Vh[:self.min_rank.item(), ...], bias)
            elif compression_ratio == str(0):
                self.options_object_dictionary[str(0)].set_svd_weights(W, bias)
            else:
                compression_ratio = float(compression_ratio) / 100
                rank = int((1-compression_ratio) * ((self.in_channels.item() * self.out_channels.item())/(self.in_channels.item() + 1 + self.out_channels.item())))
                if rank < 1:
                    rank = 1
                self.options_object_dictionary[str(int(compression_ratio*100))].set_svd_weights(U[:, :rank, ...], S[:,:rank, ...], Vh[:rank, ...], bias)

    def forward(self, x, cr = None):
        # cr is the reduction ratio. cr 90 means we keep 10 percent of computations
        if cr == 99:
            return self.options_object_dictionary[str(99)].forward(x)
        if cr is None:
            if self.include_original_weight:
                return self.options_object_dictionary[str(0)].forward(x)
            key = min(float(k) for k in self.options_object_dictionary.keys() if k != str(99))
            return self.options_object_dictionary[str(int(key))].forward(x)
        if cr == 0 and self.include_original_weight:
            return self.options_object_dictionary[str(0)].forward(x)
        return self.options_object_dictionary[str(int(cr*100))].forward(x)
class Conv1x1_SVD_per_rank(nn.Module):
    """A single fixed-rank SVD factorization of one pointwise conv layer
    (any number of spatial dims; see Conv1x1_SVD_Unified's `spatial_size`)."""
    def __init__(self, spatial_size, in_channels, out_channels, rank, bias=None):
        super(Conv1x1_SVD_per_rank, self).__init__()
        self.register_buffer("in_channels", torch.tensor(in_channels, dtype=torch.int))
        self.register_buffer("out_channels", torch.tensor(out_channels, dtype=torch.int))
        self.register_buffer("rank", torch.tensor(rank, dtype=torch.int))
        ndims = _spatial_ndims(spatial_size)
        self._conv = _conv_nd(ndims)
        kernel_ones = (1,) * ndims

        # Decomposition params
        self.U_full = nn.Parameter(torch.zeros(out_channels, rank, *kernel_ones), requires_grad=True)   # [out_c, rank]
        self.S_full = nn.Parameter(torch.zeros(1, rank, *kernel_ones), requires_grad=True)                 # [rank]
        self.Vh_full = nn.Parameter(torch.zeros(rank, in_channels, *kernel_ones), requires_grad=True)   # [rank, in_c]

        if bias is not None:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None

    def set_svd_weights(self, U, S, Vh, bias=None):
        # Store as conv-style tensors
        self.Vh_full.data.copy_(Vh)         # [r, in, *spatial]
        self.S_full.data.copy_(S)           # [1, r, *spatial]
        self.U_full.data.copy_(U)           # [out, r, *spatial]
        if self.bias is not None:
            self.bias.data.copy_(bias)

    def forward(self, x):
        x = self._conv(x, self.Vh_full, bias=None)       # [B, rank, *spatial]
        x = x * self.S_full                # [B, rank, *spatial]
        out = self._conv(x, self.U_full, bias=self.bias)   # [B, out_c, *spatial]
        return out
class Conv1x1_SVD_OriginalWeight(nn.Module):
    """Wraps the uncompressed original pointwise conv weight as one of the
    gate's choices (any number of spatial dims; see Conv1x1_SVD_Unified's
    `spatial_size`)."""
    def __init__(self, spatial_size, in_channels, out_channels, min_rank = 1, bias=None):
        super(Conv1x1_SVD_OriginalWeight, self).__init__()
        self.register_buffer("in_channels", torch.tensor(in_channels, dtype=torch.int))
        self.register_buffer("out_channels", torch.tensor(out_channels, dtype=torch.int))
        ndims = _spatial_ndims(spatial_size)
        self._conv = _conv_nd(ndims)
        kernel_ones = (1,) * ndims
        self.Original_weight = nn.Parameter(torch.zeros(out_channels, in_channels, *kernel_ones), requires_grad=True)       # [out_channels, in_channels, *spatial]
        if bias is not None:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)    # [out_channels]
        else:
            self.bias = None

    def set_svd_weights(self, W, bias=None):
        if W.shape != self.Original_weight.shape:
            raise ValueError(f"Shape mismatch: W has shape {W.shape}, "
                            f"but self.Original_weight has shape {self.Original_weight.shape}")
        self.Original_weight.data.copy_(W)
        if self.bias is not None:
            self.bias.data.copy_(bias)


    def forward(self, x):
        out = self._conv(x, self.Original_weight, bias=self.bias)


        return out

class Conv_Tucker_NonUnified(nn.Module):
    """General conv layer (any number of spatial dims: 1 for Conv1d, 2 for
    Conv2d) with one independent Tucker decomposition per compression
    choice, plus an explicit "keep original weight" choice.
    in_spatial_size/out_spatial_size/kernel_size are each an int (1
    spatial dim) or an N-tuple (N spatial dims, e.g. Conv2d's (H, W)) -
    all three must agree on how many spatial dims there are."""
    def __init__(self, in_spatial_size, out_spatial_size, in_channels, out_channels, kernel_size,
                 compression_ratio_list, min_rank=[], stride=1, padding=0, bias=None, include_original_weight=True):
        super(Conv_Tucker_NonUnified, self).__init__()

        self.register_buffer("in_channels", torch.tensor(in_channels, dtype=torch.int))
        self.register_buffer("out_channels", torch.tensor(out_channels, dtype=torch.int))
        self.register_buffer("in_spatial_size", torch.tensor(_as_spatial_tuple(in_spatial_size), dtype=torch.int))
        self.register_buffer("out_spatial_size", torch.tensor(_as_spatial_tuple(out_spatial_size), dtype=torch.int))
        ndims = _spatial_ndims(in_spatial_size)
        kernel_dims = _resolve_kernel_dims(kernel_size, ndims)
        self.register_buffer("kernel_size", torch.tensor(kernel_dims, dtype=torch.int))

        if min_rank != []:
            self.register_buffer("min_rank", torch.tensor(min_rank, dtype=torch.int))
        else:
            self.min_rank = None

        self.include_original_weight = include_original_weight
        self.options_object_dictionary = nn.ModuleDict()
        if min_rank != []:
            tucker_object = Conv_Tucker_per_rank(in_spatial_size, out_spatial_size, in_channels, out_channels, kernel_dims,
                 min_rank, stride=stride, padding=padding, bias=bias)
            self.options_object_dictionary[str(99)] = tucker_object

        for compression_ratio in compression_ratio_list:
            rank = calculate_conv2D_tucker_rank(compression_ratio, in_spatial_size, out_spatial_size, in_channels, out_channels, kernel_dims)
            if rank < 1:
                rank = 1
            tucker_object = Conv_Tucker_per_rank(in_spatial_size, out_spatial_size, in_channels, out_channels, kernel_dims,
                 rank, stride=stride, padding=padding, bias=bias)
            self.options_object_dictionary[str(int(compression_ratio*100))] = tucker_object

        if include_original_weight:
            original_object = Conv_Tucker_OriginalWeight(in_channels, out_channels, kernel_dims, stride=stride, padding=padding, bias = bias)
            self.options_object_dictionary[str(0)] = original_object

        key = min(float(k) for k in self.options_object_dictionary.keys() if float(k) != 0)
        min_compression_ratio = key/100
        max_rank = calculate_conv2D_tucker_rank(min_compression_ratio, in_spatial_size, out_spatial_size, in_channels, out_channels, kernel_dims)
        self.register_buffer("max_rank", torch.tensor(max_rank, dtype=torch.int))

    def set_weights(self, W, core, factors, bias=None):
        for compression_ratio in self.options_object_dictionary.keys():
            if compression_ratio == str(99): # Min rank
                self.options_object_dictionary[str(99)].set_weights(core[:self.min_rank.item(), :self.min_rank.item(), ...],
                                                                        factors[0][:, :self.min_rank.item()],
                                                                        factors[1][:, :self.min_rank.item()],
                                                                        bias)
            elif compression_ratio == str(0): # original weights
                self.options_object_dictionary[str(0)].set_weights(W, bias)
            else: # Other ranks
                compression_ratio = float(compression_ratio) / 100
                rank = calculate_conv2D_tucker_rank(compression_ratio, self.in_spatial_size.tolist(), self.out_spatial_size.tolist(),
                                                    self.in_channels.item(), self.out_channels.item(), self.kernel_size.tolist())
                if rank < 1:
                    rank = 1

                self.options_object_dictionary[str(int(compression_ratio*100))].set_weights(core[:rank, :rank, ...],
                                                                        factors[0][:, :rank],
                                                                        factors[1][:, :rank],
                                                                        bias)
    def forward(self, x, cr = None):
        # cr is the reduction ratio. cr 90 means we keep 10 percent of computations
        if cr == 99: # Minimum rank
            return self.options_object_dictionary[str(99)].forward(x)
        if cr is None:
            if self.include_original_weight:
                return self.options_object_dictionary[str(0)].forward(x)
            key = min(float(k) for k in self.options_object_dictionary.keys() if k != str(99))
            return self.options_object_dictionary[str(int(key))].forward(x)
        if cr == 0 and self.include_original_weight:
            return self.options_object_dictionary[str(0)].forward(x)
        return self.options_object_dictionary[str(int(cr*100))].forward(x)
class Conv_Tucker_per_rank(nn.Module):
    """A single fixed-rank Tucker factorization (core + two factor
    matrices) of one conv layer (any number of spatial dims; see
    Conv_Tucker_NonUnified)."""
    def __init__(self, in_spatial_size, out_spatial_size, in_channels, out_channels, kernel_size,
                 training_rank, stride=1, padding=0, bias=True):
        super(Conv_Tucker_per_rank, self).__init__()

        self.register_buffer("training_rank", torch.tensor(training_rank, dtype=torch.int))
        self.register_buffer("in_spatial_size", torch.tensor(_as_spatial_tuple(in_spatial_size), dtype=torch.int))
        self.register_buffer("out_spatial_size", torch.tensor(_as_spatial_tuple(out_spatial_size), dtype=torch.int))
        self.register_buffer("in_channels", torch.tensor(in_channels, dtype=torch.int))
        self.register_buffer("out_channels", torch.tensor(out_channels, dtype=torch.int))
        ndims = _spatial_ndims(in_spatial_size)
        kernel_dims = _resolve_kernel_dims(kernel_size, ndims)
        self.register_buffer("kernel_size", torch.tensor(kernel_dims, dtype=torch.int))
        self._conv = _conv_nd(ndims)
        self._kernel_ones = (1,) * ndims

        self.stride = stride
        self.padding = padding
        # Decomposition params
        self.first_conv_weights = nn.Parameter(torch.zeros(training_rank, in_channels, *self._kernel_ones))     # (I, rI)
        self.core_conv_weights = nn.Parameter(torch.zeros(training_rank, training_rank, *kernel_dims))     # (rO,rI,*kernel_dims)
        self.third_conv_weights = nn.Parameter(torch.zeros(out_channels, training_rank, *self._kernel_ones))   # (O, rO)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)


    def set_weights(self, core, factors_0, factors_1, bias=None):
        """
        core: (rO, rI, *kernel_dims)
        factors: with shapes (O,rO), (I,rI)
        """
        self.core_conv_weights.data.copy_(core)
        self.third_conv_weights.data.copy_(factors_0.view(factors_0.shape[0], factors_0.shape[1], *self._kernel_ones)) #factors[0]
        self.first_conv_weights.data.copy_(factors_1.t().view(factors_1.shape[1], factors_1.shape[0], *self._kernel_ones))
        if bias is not None:
            self.bias.data.copy_(bias)

    def forward(self, x):
        # 1) First convolution
        x_proj = self._conv(x, self.first_conv_weights)

        # 2) Core convolution
        x_core = self._conv(x_proj, self.core_conv_weights, stride=self.stride, padding=self.padding)

        # 3) Output projection
        out = self._conv(x_core, self.third_conv_weights, bias=self.bias)

        return out
class Conv_Tucker_OriginalWeight(nn.Module):
    """Wraps the uncompressed original conv weight as one of the gate's
    choices (any number of spatial dims; `kernel_size` is an int or an
    N-tuple, see Conv_Tucker_NonUnified)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None):
        super(Conv_Tucker_OriginalWeight, self).__init__()
        self.register_buffer("in_channels", torch.tensor(in_channels, dtype=torch.int))
        self.register_buffer("out_channels", torch.tensor(out_channels, dtype=torch.int))
        kernel_dims = _as_spatial_tuple(kernel_size)
        self.register_buffer("kernel_size", torch.tensor(kernel_dims, dtype=torch.int))
        self._conv = _conv_nd(len(kernel_dims))
        self.Original_weight = nn.Parameter(torch.zeros(out_channels, in_channels, *kernel_dims), requires_grad=True)       # [out_channels, in_channels, *kernel_dims]
        self.stride = stride
        self.padding = padding
        if bias is not None:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)    # [out_channels]
        else:
            self.bias = None
    def set_weights(self, W, bias=None):
        if W.shape != self.Original_weight.shape:
            raise ValueError(f"Shape mismatch: W has shape {W.shape}, "
                            f"but self.Original_weight has shape {self.Original_weight.shape}")
        self.Original_weight.data.copy_(W)
        if self.bias is not None:
            self.bias.data.copy_(bias)
    def forward(self, x):
        out = self._conv(x, self.Original_weight, bias=self.bias, padding=self.padding, stride=self.stride)
        return out


class Conv_Tucker_Unified(nn.Module):
    """General conv layer (any number of spatial dims) with a single
    shared Tucker basis at max rank; compression choices slice it
    (variant used by the "_unified" architectures). See
    Conv_Tucker_NonUnified for in_spatial_size/out_spatial_size/kernel_size."""
    def __init__(self, in_spatial_size, out_spatial_size, in_channels, out_channels, kernel_size,
                 compression_ratio_list, min_rank=[], stride=1, padding=0, bias=None):
        super(Conv_Tucker_Unified, self).__init__()

        self.register_buffer("in_channels", torch.tensor(in_channels, dtype=torch.int))
        self.register_buffer("out_channels", torch.tensor(out_channels, dtype=torch.int))
        self.register_buffer("in_spatial_size", torch.tensor(_as_spatial_tuple(in_spatial_size), dtype=torch.int))
        self.register_buffer("out_spatial_size", torch.tensor(_as_spatial_tuple(out_spatial_size), dtype=torch.int))
        ndims = _spatial_ndims(in_spatial_size)
        kernel_dims = _resolve_kernel_dims(kernel_size, ndims)
        self.register_buffer("kernel_size", torch.tensor(kernel_dims, dtype=torch.int))
        self._conv = _conv_nd(ndims)
        self._kernel_ones = (1,) * ndims

        if min_rank != []:
            self.register_buffer("min_rank", torch.tensor(min_rank, dtype=torch.int))
        else:
            self.min_rank = None

        min_compression_ratio = min(k for k in compression_ratio_list)
        max_rank = calculate_conv2D_tucker_rank(min_compression_ratio, in_spatial_size, out_spatial_size, in_channels, out_channels, kernel_dims)
        self.register_buffer("max_rank", torch.tensor(max_rank, dtype=torch.int))


        self.stride = stride
        self.padding = padding
        # Decomposition params
        self.first_conv_weights = nn.Parameter(torch.zeros(max_rank, in_channels, *self._kernel_ones))     # (I, rI)
        self.core_conv_weights = nn.Parameter(torch.zeros(max_rank, max_rank, *kernel_dims))     # (rO,rI,*kernel_dims)
        self.third_conv_weights = nn.Parameter(torch.zeros(out_channels, max_rank, *self._kernel_ones))   # (O, rO)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def set_weights(self, W, core, factors, bias=None):
        self.core_conv_weights.data.copy_(core[:self.max_rank, :self.max_rank, ...])
        self.third_conv_weights.data.copy_(factors[0][:, :self.max_rank].view(-1, self.max_rank, *self._kernel_ones))
        self.first_conv_weights.data.copy_(factors[1][:, :self.max_rank].t().reshape(self.max_rank, -1, *self._kernel_ones))
        if bias is not None:
            self.bias.data.copy_(bias)

    def forward(self, x, cr = None):
        # cr is the reduction ratio. cr 90 means we keep 10 percent of computations
        if cr is None or cr == 0:
            rank = self.max_rank
        elif cr == 99: # Minimum rank
            rank = self.min_rank
        else: # Other ranks
            compression_ratio = float(cr)
            rank = calculate_conv2D_tucker_rank(compression_ratio, self.in_spatial_size.tolist(), self.out_spatial_size.tolist(),
                                                self.in_channels.item(), self.out_channels.item(), self.kernel_size.tolist())
            if rank < 1:
                rank = 1
        core_conv_weights_tmp = self.core_conv_weights[:rank, :rank, ...]
        third_conv_weights_tmp = self.third_conv_weights[:, :rank, ...]
        first_conv_weights_tmp = self.first_conv_weights[:rank, ...]

        x_proj = self._conv(x, first_conv_weights_tmp)
        x_core = self._conv(x_proj, core_conv_weights_tmp, stride=self.stride, padding=self.padding)
        out = self._conv(x_core, third_conv_weights_tmp, bias=self.bias)

        return out
