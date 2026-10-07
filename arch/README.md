# arch/

Network definitions, gating modules, and the factory functions the training scripts select architectures by name.

## Files

- `gates.py` - gate modules that pick one compression choice per sample:
  `RNNGate` (LSTM, hard one-hot in forward), `SoftRNNGate` (continuous per-sample weight),
  `SEQ_Gate_V1`..`V6` (non-recurrent MLP gates of increasing depth), and `broadcast_mask`
  (reshapes a `(batch, classes)` mask so it multiplies any layer output).
- `common.py` - building blocks shared across network families (currently `DropPath`).
- `lenet5.py`, `alexnet.py`, `convnext.py`, `efficientformer.py`, `ghostnet.py` - one file per
  network, holding its baseline architecture, its adaptive architecture(s), and the factory
  functions for both.
- `models_baseline.py` - re-exports the selectable baseline factories (used by `train_base.py`).
- `models_adaptive.py` - re-exports the selectable adaptive factories (used by `adaptive_training.py`).

## Conventions

- **Factories** are the names that go into a config's `arch` field. Each is a plain module-level
  function, and the aggregators must re-export it explicitly (no `import *`), so that
  `models.__dict__[arch]` finds only the real architectures.
- **Variants:** a plain adaptive architecture keeps a separate decomposition per compression choice
  (optionally plus an "original weight" choice). A `_unified` architecture shares one low-rank
  basis across all choices. Each adaptive architecture class declares `is_unified = True/False`,
  and its factory mirrors that value. `configs/config_utils.py::is_plain_arch` reads it from there.
- **Gate pairing:** a low-rank layer named `<name>_svd_layer` or `<name>_tucker_layer` must have
  its gate registered as `<name>_gate` on the same module. `adaptive_training.py` finds the pairs
  this way.

## Adding a network

1. Create `arch/<network>.py` with the baseline architecture class and its factory, plus the
   adaptive architecture class(es), each with `is_unified` and a factory that mirrors it.
2. Add one import line per factory to `models_baseline.py` and/or `models_adaptive.py`.

No other file needs to change.
