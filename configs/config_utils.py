"""Shared JSON config loading for the training scripts."""

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    """Seed python/numpy/torch (CPU+CUDA) RNGs and force deterministic
    cuDNN algorithms, so re-running the same config reproduces the same
    weight init, data shuffling and training trajectory."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(path):
    """Load the checkpoint at `path` (the config's `resume`) and return it.

    Raises instead of silently continuing without it: FileNotFoundError if
    `path` is not an existing file, RuntimeError if torch.load fails or the
    result has no 'state_dict' entry. Callers only call this when `resume`
    is set; an empty `resume` still means "no checkpoint"."""
    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"resume checkpoint not found: {path!r} (set 'resume' in the "
            f"config to an existing checkpoint file, or to \"\" for none)")
    try:
        checkpoint = torch.load(path)
    except Exception as e:
        raise RuntimeError(f"could not load resume checkpoint {path!r}: {e}") from e
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise RuntimeError(
            f"resume checkpoint {path!r} has no 'state_dict' entry; "
            f"it was not saved by this framework")
    return checkpoint


def load_config(config_path, defaults):
    """Load a JSON config, overlay it onto `defaults`, and return an
    argparse.Namespace (so downstream code can keep using `args.foo`)."""
    with open(config_path, 'r') as f:
        overrides = json.load(f)

    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown config key(s) in {config_path}: {sorted(unknown)}")

    merged = dict(defaults)
    merged.update(overrides)
    return argparse.Namespace(**merged)


def build_optimizer(name, params, lr, momentum=0.9, weight_decay=0.0):
    """Builds an optimizer by name ('sgd'/'adam'/'adamw') over `params`.

    Output: a torch.optim.Optimizer instance. Raises ValueError for an
    unknown name."""
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {name!r}. Choices: sgd, adam, adamw")


def build_criterion(name, **kwargs):
    """Builds the loss trained against, by name.

    Input: name - 'cross_entropy' (classification, the default) or
    'l1'/'mse'/'bce'/'bce_with_logits' (regression/reconstruction tasks,
    e.g. a speech-enhancement mask predictor). `kwargs` are passed
    straight through to the underlying nn.*Loss (e.g. `label_smoothing`
    for cross_entropy).

    Output: an nn.Module loss instance. Raises ValueError for an unknown
    name."""
    name = name.lower()
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif name == 'l1':
        return nn.L1Loss(**kwargs)
    elif name == 'mse':
        return nn.MSELoss(**kwargs)
    elif name == 'bce':
        return nn.BCELoss(**kwargs)
    elif name == 'bce_with_logits':
        return nn.BCEWithLogitsLoss(**kwargs)
    else:
        raise ValueError(f"Unknown criterion {name!r}. Choices: cross_entropy, l1, mse, bce, bce_with_logits")


def _topk_accuracy(output, target, k=1):
    """Computes top-k accuracy as a percentage (0-100); the 'accuracy'
    metric's implementation. Input: model output and integer-label
    target. Output: a scalar tensor."""
    batch_size = target.size(0)
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return correct[:k].reshape(-1).float().sum(0) * (100.0 / batch_size)


def build_metric(name):
    """Builds the scalar "how good is this result" function tracked for
    best-checkpoint selection and the metric-drop checkpointing gate.
    Always higher-is-better, by convention: a loss-like metric is exposed
    here as its negative (e.g. 'neg_mse' is -MSE) so every comparison
    elsewhere in the training scripts stays a plain "bigger is better".

    Input: name - 'accuracy' (top-1 classification accuracy, the default)
    or 'neg_l1'/'neg_mse' (regression/reconstruction tasks).

    Output: a callable `metric_fn(output, target) -> scalar tensor`.
    Raises ValueError for an unknown name."""
    name = name.lower()
    if name == 'accuracy':
        return lambda output, target: _topk_accuracy(output, target, k=1)
    elif name == 'neg_l1':
        return lambda output, target: -F.l1_loss(output, target)
    elif name == 'neg_mse':
        return lambda output, target: -F.mse_loss(output, target)
    else:
        raise ValueError(f"Unknown metric {name!r}. Choices: accuracy, neg_l1, neg_mse")


def is_plain_arch(arch_name):
    """Checks whether an architecture is the plain (non-Unified) variant,
    whose low-rank layers have a separate per-choice decomposition (so an
    extra 'keep original weight' choice can be added to them) - as
    opposed to Unified, which shares one low-rank basis and bias across
    every choice (just sliced to a different rank per choice) and so
    never gets that extra choice.

    Input: arch_name - a factory function name in arch.models_adaptive.

    Output: True if plain (not Unified). Reads the `is_unified` class
    attribute every adaptive model declares, rather than guessing from
    the name. Raises ValueError if `arch_name` isn't a known factory, or
    AttributeError if it doesn't declare `is_unified`."""
    from arch import models_adaptive

    factory = getattr(models_adaptive, arch_name, None)
    if factory is None:
        raise ValueError(f"Unknown arch {arch_name!r}: no such factory function in arch.models_adaptive.")
    if not hasattr(factory, 'is_unified'):
        raise AttributeError(
            f"arch.models_adaptive.{arch_name} doesn't declare `is_unified` "
            f"(directly, or mirrored from its model class). Every adaptive "
            f"architecture factory must set this - see the `is_unified = "
            f"True/False` class attribute plus the one-line "
            f"`<factory>.is_unified = <ModelClass>.is_unified` pattern "
            f"already used throughout arch/models_adaptive.py - so "
            f"is_plain_arch never has to guess the variant from the name."
        )
    return not factory.is_unified


def gate_num_classes_for(arch_name, compression_rates, include_original_weight):
    """Computes how many choices the gate must output for: one per
    compression ratio, plus one more if this arch/config combination adds
    a "keep original weight" choice. Output: an int."""
    extra = 1 if (is_plain_arch(arch_name) and include_original_weight) else 0
    return len(compression_rates) + extra


def gate_class_labels(arch_name, compression_rates, include_original_weight):
    """Builds a human-readable label for each of the gate's output
    classes, in the same order the model's forward() builds them in, so a
    printed per-class distribution can name its class instead of using a
    bare index. Output: a list of strings, length gate_num_classes_for(...)."""
    labels = [f"{int(round(r * 100))}% compression" for r in compression_rates]
    if is_plain_arch(arch_name) and include_original_weight:
        labels.append("keep original weight")
    return labels


BASE_DEFAULTS = {
    'cmd': 'train',
    'seed': 42,
    'arch': 'cifar10_alexnet_v2',
    'dataset': 'cifar10',
    'workers': 8,
    'epochs': 100,
    'start_epoch': 0,
    'batch_size': 128,
    'optimizer': 'sgd',
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'print_freq': 50,
    'resume': '',
    'pretrained': False,
    'step_ratio': 0.1,
    'warm_up': False,
    'save_folder': 'save_checkpoints/',
    'save_name': None,  # the checkpoint subdirectory name; defaults to `arch` when unset
    'eval_every': 500,
    'precision': 0,

    # task: what's being trained against (see build_criterion/build_metric
    # in this file). Defaults match this script's original hardcoded
    # behavior - a classification backbone trained with label-smoothed
    # cross-entropy, evaluated by top-1 accuracy.
    'criterion': 'cross_entropy',
    'criterion_kwargs': {'label_smoothing': 0.1},
    'metric': 'accuracy',
}

SCENARIO_DEFAULTS = {
    'cmd': 'train',
    'seed': 42,
    'arch': 'cifar10_ghostnet_adaptive',
    'gate_type': 'rnn',
    'dataset': 'cifar10',
    'workers': 1,
    'epochs': 300,
    'start_epoch': 0,
    'batch_size': 128,
    'optimizer': 'sgd',
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 1e-3,
    'print_freq': 50,
    'resume': '',
    'pretrained': False,
    'step_ratio': 0.1,
    'warm_up': False,
    'save_folder': 'save_checkpoints',
    'save_name': None,  # the checkpoint subdirectory name; defaults to `arch` when unset
    'eval_every': 390,
    'verbose': False,
    'target_computation_ratio': None,
    'computation_cost': True,
    'beta': 1e-3,
    'min_rank': [],
    'metric_base_value': 0,
    'metric_drop_thr': None,
    'compression_ratios': None,  # required: set in every adaptive config
    'include_original_weight': True,

    # task: what's being trained against (see build_criterion/build_metric
    # in this file). Defaults match this script's original hardcoded
    # behavior - a classification backbone trained with plain (unsmoothed)
    # cross-entropy, evaluated by top-1 accuracy. `metric_base_value`/
    # `metric_drop_thr` (above) are the per-epoch checkpointing gate: an
    # epoch is only checkpointed once its validation metric is at or above
    # `metric_base_value - metric_drop_thr` (e.g. the plain backbone's
    # accuracy/metric_base_value, with up to metric_drop_thr tolerated
    # drop from it) - in absolute metric units, so it works the same way
    # regardless of which `metric` is chosen. Leave `metric_drop_thr` at
    # its default (None) to disable the gate and checkpoint every epoch.
    'criterion': 'cross_entropy',
    'criterion_kwargs': {},
    'metric': 'accuracy',

    # backbone warm-start (formerly the separate adaptive_initial.py stage):
    # `resume` is always a plain-backbone checkpoint. Its low-rank layers
    # are warm-started from it via SVD/Tucker decomposition, fine-tuned
    # with the gate frozen, and the gate is then warmed up with the
    # weights frozen - all directly on this (final) architecture - before
    # falling into the main joint-training loop below. (To evaluate an
    # already-adaptive checkpoint instead of retraining it, use
    # `cmd: "test"` instead, which loads it directly.)
    'fine_tune_epochs': 10,
    'gate_warmup_epochs': 5,
}
