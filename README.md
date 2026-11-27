This is the official implementation of the paper "Kokhazad, Zahra, Milad Kokhazadeh, Georgios Keramidas, Vasilios Kelefouras, and Panagiotis Mousouliotis. Adaptive Low-Rank Factorization for Dynamic DNN Inference on Edge Devices. In 2026 29th Euromicro Conference on Digital System Design (DSD), Euromicro, 2026".

# Adaptive LRF Framework

Adaptive low-rank compression for CNN backbones. Each Conv/Linear
layer is replaced by a low-rank (SVD or Tucker) factorization exposing a
handful of compression-ratio "choices," and a small RNN gate learns, per
layer and per input, which choice to route to - trading accuracy for
computation under a FLOPs regularizer.

## How it fits together

Training happens in two stages, each with its own script and config schema
(`configs/config_utils.py`):

1. **`train_base.py`** - trains a plain backbone from `arch/models_baseline.py`
   with no compression at all. Its checkpoint becomes the reference
   accuracy and the required warm-start source for stage 2.
2. **`adaptive_training.py`** - builds one of the final adaptive
   architectures (`arch/models_adaptive.py`) and trains it under the
   computation-cost regularizer. `resume` always points at a plain-backbone
   checkpoint (stage 1's output): the script warm-starts the low-rank
   layers from it via SVD/Tucker decomposition
   (`apply_lrf_on_target_layers`), fine-tunes them with the gate frozen
   (`fine_tune_weights`), and warms up the gate with the weights frozen
   (`warmup_gate`) - all directly on the final architecture, in one run -
   before falling into the joint-training loop. 

   Which layers get warm-started is discovered automatically
   (`discover_adaptive_layers`): it finds every low-rank layer in the
   adaptive model by type.    A low-rank layer whose plain counterpart 
   can't be found is left at its random initialization and logged as a 
   warning, rather than aborting the whole warm-start. 

Both scripts are driven entirely by a `--config path/to/config.json` file.

## Repo layout

```
adaptive_lrf_framework/
├── train_base.py              # stage 1: plain backbone training
├── adaptive_training.py       # stage 2: backbone warm-start + gated training
├── data.py                    # dataset/dataloader factories
├── arch/
│   ├── gates.py                # RNNGate, SoftRNNGate, SEQ_Gate_* controllers
│   ├── common.py                # building blocks shared by more than one network (e.g. DropPath)
│   ├── lenet5.py                 # LeNet5: baseline + adaptive classes
│   ├── alexnet.py                # AlexNet-v2: baseline + adaptive classes
│   ├── convnext.py               # ConvNeXt: baseline + adaptive classes
│   ├── efficientformer.py        # EfficientFormer-L3: baseline + adaptive classes
│   ├── ghostnet.py                # GhostNet: baseline + adaptive classes
│   ├── models_baseline.py      # thin re-export aggregator: plain backbones (stage 1)
│   └── models_adaptive.py      # thin re-export aggregator: adaptive architectures (stage 2)
├── modules/
│   └── lrf.py                  # the low-rank Linear/Conv1x1/Conv2d (SVD + Tucker) layers
├── configs/
│   ├── config_utils.py         # config loading + the 2 default schemas
│   ├── baseline/                    # train_base.py configs go here
│   └── adaptive/                    # adaptive_training.py configs go here
└── aux/
    └── flops_parameters_counter_fvcore_v3.py  # params/MACs report for a checkpoint
```

## Installation

```bash
pip install -r requirements.txt
```

## Configs

Each script loads a JSON file via `--config`.Missing keys fall back to the defaults in `configs/config_utils.py`
(`BASE_DEFAULTS`, `SCENARIO_DEFAULTS`). A few fields worth calling out:

- `arch` - the factory function name to use, e.g. `"cifar10_ghostnet_adaptive_unified"`.
  Run `python -c "from arch import models_adaptive as m; print(sorted(n for n in m.__dict__ if n.islower() and not n.startswith('_')))"`
  to list what's available.
- `save_folder` / `save_name` - checkpoints and logs go to
  `<save_folder>/<save_name>`. `save_name` defaults to `arch` when unset, so
  two configs with the same `arch` write to the same directory unless you
  set `save_name` to tell them apart.
- `seed` - seeds python/numpy/torch (CPU+CUDA) and forces deterministic
  cuDNN algorithms (`configs.config_utils.set_seed`), so re-running the
  same config reproduces the same run.
- `epochs` / `start_epoch` - training length and resume point, in epochs
  (both scripts loop over epochs, not raw steps).
- `compression_ratios` - the list of compression-ratio choices the gate
  picks between (`adaptive_training.py` only).
- `include_original_weight` - whether to add an extra "keep the original
  weight" gate choice. Only affects the **non-unified** variant of each
  adaptive architecture - ignored for `_unified`.
- `min_rank` - `[]` (default) for none, or a rank (`[r]`, or one per stage
  for ConvNeXt) that adds an extra "min rank" gate choice to every adaptive
  layer, unified or not.

  The gate's classes are, in order: the min-rank choice (if `min_rank` is
  set), each of `compression_ratios` in the order given, then the
  original-weight choice (if present). For example `compression_ratios:
  [0.75, 0.5, 0]`, `min_rank: [2]`, `include_original_weight: true` gives
  5 classes: min rank, 75%, 50%, 0%, keep original weight. A ratio of `0`
  is an ordinary (factorized) choice, separate from the original weight.
- `optimizer` - `"sgd"` / `"adam"` / `"adamw"`.
- `target_computation_ratio` - if set, the computation-cost regularizer's
  sign flips once the model's live computation ratio drops below this
  target, so it stops over-compressing; `null` disables the flip
  (regularizer always pushes toward more compression). Not used during
  gate warm-up, which always rewards higher computation.
- `criterion` / `criterion_kwargs` - the loss trained against:
  `"cross_entropy"` (default; `criterion_kwargs` passes e.g.
  `{"label_smoothing": 0.1}`), or `"l1"`, `"mse"`, `"bce"`,
  `"bce_with_logits"`.
- `metric` - the score tracked for checkpointing and reporting:
  `"accuracy"` (default, top-1 %), or `"neg_l1"` / `"neg_mse"` for
  regression tasks. Always higher-is-better.
- `metric_base_value` / `metric_drop_thr` - the metric-drop checkpoint
  assesing. An epoch only qualifies if its metric is at least
  `metric_base_value - metric_drop_thr` (absolute units). Qualifying 
  epochs are saved only if they also use less computation than the best
  checkpoint saved so far.
- `fine_tune_epochs` / `gate_warmup_epochs` - length of the two warm-start
  sub-phases, run once at the start of every `adaptive_training.py` training run.

Example config for `adaptive_training.py` (`resume` is the stage-1
backbone checkpoint, not an adaptive one):

```json
{
  "cmd": "train",
  "arch": "cifar10_ghostnet_adaptive_unified",
  "resume": "save_checkpoints/cifar10_ghostnet/model_best.pth.tar",
  "fine_tune_epochs": 10,
  "gate_warmup_epochs": 5,
  "compression_ratios": [0.75, 0.5, 0.25],
  "epochs": 300,
  "lr": 0.001,
  "beta": 1e-3
}
```

## Run with:

```bash
python train_base.py --config configs/baseline/mnist_lenet_5.json
python adaptive_training.py --config configs/adaptive/lenet5_mnist_rnn_gate_two_fc_layers.json
```

## Built-in architectures

The framework isn't limited to these - see `arch/README.md`'s "Adding a
network" section for how to add another one.

**Backbones** (`arch/models_baseline.py`, used by `train_base.py`):
`mnist_lenet_5`, `cifar10_alexnet_v2`, `cifar10_convnext`,
`cifar10_efficienformer_l3`, `cifar10_ghostnet`.

**Adaptive families** (`arch/models_adaptive.py`, used by
`adaptive_training.py` - each with `_NonUnified` / `_unified` variants unless
noted):

| Family | `arch` values |
|---|---|
| AlexNet-v2 | `cifar10_alexnet_v2_adaptive[_unified]` |
| GhostNet | `cifar10_ghostnet_adaptive[_unified]` |
| ConvNeXt | `cifar10_convnext_adaptive[_unified]` |
| EfficientFormer-L3 | `cifar10_efficientformer_l3_adaptive[_unified]` |
| LeNet5 (1st FC layer gated) | `lenet5_mnist_rnn_gate_first_fc_layer[_unified]` |
| LeNet5 (2 FC layers gated) | `lenet5_mnist_rnn_gate_two_fc_layers` (plain only) |

## Acknowledgments

[Claude](https://claude.com/claude-code) (Anthropic) was used in developing this repository.
