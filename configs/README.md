# configs/

Run configuration: the defaults both training scripts merge with each run's JSON file, and the
helpers that build the training criterion, metric, and optimizer from those settings.

## Files

- `config_utils.py`
  - `BASE_DEFAULTS` - every field `train_base.py` reads, with its default.
  - `SCENARIO_DEFAULTS` - every field `adaptive_training.py` reads, with its default.
  - `load_config(path, defaults)` - merges a JSON file onto a defaults dict and returns an
    `argparse.Namespace`. Unknown keys raise `ValueError`.
  - `build_criterion(name, **kwargs)`, `build_metric(name)`, `build_optimizer(...)` - turn config
    names into the loss, metric, and optimizer objects.
  - `is_plain_arch`, `gate_num_classes_for`, `gate_class_labels` - derive the gate's output width
    and labels from the architecture and config.
  - `set_seed(seed)` - seeds python, numpy, and torch, and forces deterministic cuDNN.

- `baseline/` - JSON configs for `train_base.py`.
- `adaptive/` - JSON configs for `adaptive_training.py`.

## Usage

```bash
python train_base.py --config configs/baseline/<run>.json
python adaptive_training.py --config configs/adaptive/<run>.json
```

Each JSON file only needs the keys that differ from the defaults. The defaults dicts in
`config_utils.py` are the full field reference; the root README only covers the main ones.
