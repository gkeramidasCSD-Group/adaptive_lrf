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

## Gate classes

`ordered_gate_choices(compression_ratios, has_min_rank, include_original_weight)` is the
one definition of the gate's class order: `[MIN_RANK]` (if set) + the compression ratios +
`[ORIGINAL_WEIGHT]` (if set). Gate class `k` means `layer(x, choices[k])`. Every model's
`forward()` gets this list from `gate_choices_for(self, compression_ratios)`, and every
layer reports its own list with `layer.gate_choices(compression_ratios)`. The gate class
count and labels (`configs.config_utils`) and the FLOPs table (`adaptive_training.py`) use
the same order, and `adaptive_training.py` checks that they agree before training or testing.

## Forward-pass choice argument

Every layer's `forward(x, cr)` takes the gate's selected choice as `cr`:

- `MIN_RANK` (the string `'min_rank'`) - the min-rank choice (only if the layer was
  built with `min_rank`).
- `ORIGINAL_WEIGHT` (the string `'original'`) - the original weight (NonUnified only,
  and only if `include_original_weight`).
- Any float - a compression ratio from the layer's `compression_ratio_list`. This
  includes `0`, which is an ordinary SVD/Tucker choice, not the original weight, and
  `0.99`, which is not the min-rank choice.
- `None` - no choice given. On a Unified layer, uses the shared basis truncated to `max_rank`.
  On a NonUnified layer, uses the original weight if `include_original_weight`, otherwise
  the least-compressed ratio in the list.

Adaptive checkpoints saved before these names existed stored the original weight under
key `0` and the min-rank choice under key `99`. They still load: the NonUnified layers
rename those keys on load.

## Spatial dimensions

`Conv1x1_SVD_*` takes `spatial_size`, and `Conv_Tucker_*` takes `in_spatial_size`,
`out_spatial_size`, and `kernel_size`. Each is an `int` for 1D or an N-tuple for 2D. A bare `int`
for `kernel_size` means that value repeated across all dims (3 -> 3x3), not a 1D size.
