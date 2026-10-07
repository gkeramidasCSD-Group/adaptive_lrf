# aux/

Standalone analysis tools. Nothing here is imported by the training scripts.

## Files

- `flops_parameters_counter_fvcore_v3.py` - reports total and trainable parameters, plus MACs
  (via fvcore, with custom handlers for ops fvcore doesn't count, including LSTMs). It also breaks
  out 1x1-conv, Linear, and gate-related MACs as shares of the total.

## Usage

The script has no command-line arguments. Edit these lines near the top, then run it directly:

- `checkpoint_path` - the checkpoint to load (its `state_dict` key is read).
- `compression_ratios` and `min_rank` - must match the run the checkpoint came from.
- The `model = ...` line - the factory that builds that architecture, with the same
  `include_original_weight` setting the run used.

```bash
python aux/flops_parameters_counter_fvcore_v3.py
```

The gate-related MAC total counts modules whose names end with any entry in `name_keywords`
(default `gate` and `control`). Per-sample figures are printed alongside batch totals.
