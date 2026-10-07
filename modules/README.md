# modules/

The low-rank layers the adaptive networks are built from, and the rank/FLOPs formulas they use.

## Files

- `lrf.py` - the low-rank layer classes. Each one holds a set of compression "choices" that the
  gate selects between at every forward pass:
  - `Linear_SVD_*` - fully connected layers (SVD).
  - `Conv1x1_SVD_*` - pointwise convs (SVD), 1D or 2D spatial.
  - `Conv_Tucker_*` - general convs (Tucker), 1D or 2D spatial.

  Each family has:
  - `*_NonUnified` - one independent decomposition per choice, plus an optional original-weight choice.
  - `*_Unified` - one shared basis at max rank, with every choice a slice of it.
  - `*_per_rank` and `*_OriginalWeight` - the single-choice pieces a NonUnified layer is built from.

- `flops_utils.py` - rank-from-compression-ratio and FLOPs formulas for all three families
  (Linear helpers included). Conv helpers support 1 or 2 spatial dims; 3D raises `NotImplementedError`.

## Forward-pass choice argument

Every layer's `forward(x, cr)` takes the gate's selected choice as `cr`:

- `99` - the min-rank choice (only if the layer was built with `min_rank`).
- `0` - the original weight (only if `include_original_weight`).
- Any other float - a compression ratio from the layer's `compression_ratio_list`.
- `None` - no choice given. On a Unified layer, uses the shared basis truncated to `max_rank`.
  On a NonUnified layer, uses choice `0` if `include_original_weight`, otherwise the
  least-compressed ratio in the list.

## Spatial dimensions

`Conv1x1_SVD_*` takes `spatial_size`, and `Conv_Tucker_*` takes `in_spatial_size`,
`out_spatial_size`, and `kernel_size`. Each is an `int` for 1D or an N-tuple for 2D. A bare `int`
for `kernel_size` means that value repeated across all dims (3 -> 3x3), not a 1D size.
