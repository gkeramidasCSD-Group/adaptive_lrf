"""Adaptive training: builds one of the final adaptive architectures
(arch/models_adaptive.py) and trains it under a computation-cost
regularizer, so the RNN gate learns to route each layer to a compression
ratio that keeps accuracy near the backbone's while minimizing FLOPs.

`resume` (required for training) always points at a plain-backbone
checkpoint (arch/models_baseline.py, produced by train_base.py). Its
low-rank layers are warm-started from that checkpoint via SVD/Tucker
decomposition (apply_lrf_on_target_layers), fine-tuned with the gate
frozen (fine_tune_weights), and the gate is warmed up with the weights
frozen (warmup_gate), before falling into the main joint-training loop.

To evaluate an already-adaptive checkpoint (rather than retrain it), use
`cmd: "test"`, which loads it directly (see test_model) instead of running
the warm-start sequence.

Run with: python adaptive_training.py --config path/to/config.json
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import tensorly as tl
from tensorly.decomposition import tucker, partial_tucker

import os
import math
import copy
import shutil
import argparse
import inspect
import time
import logging

from arch import models_adaptive as models
from modules.lrf import (
    Linear_SVD_NonUnified, Linear_SVD_Unified,
    Conv1x1_SVD_NonUnified, Conv1x1_SVD_Unified,
    Conv_Tucker_NonUnified, Conv_Tucker_Unified,
)
from configs.config_utils import (
    load_config, load_checkpoint, SCENARIO_DEFAULTS, build_optimizer, build_criterion, build_metric,
    gate_num_classes_for, gate_class_labels, is_plain_arch, set_seed,
)
from data import *
from modules.flops_utils import (
    calculate_conv2D_tucker_ranks, calculate_tucker_flops, calculate_Conv2D_flops,
    calculate_svd_conv1x1_flops, calculate_conv1x1_flops,
    calculate_fc_ranks, calculate_fc_flops, calculate_svd_fc_flops,
)
from torchsummary import summary
from functools import reduce

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )

tl.set_backend('pytorch')
def min_rank_argument(args):
    """Converts the config's min_rank list into what the factories expect: a
    scalar for a single value, otherwise the list (per stage for ConvNeXt)."""
    return args.min_rank[0] if len(args.min_rank) == 1 else args.min_rank


def parse_args():
    """Reads --config, merges it onto SCENARIO_DEFAULTS, and validates
    `arch`/`cmd`. Returns the resulting argparse.Namespace."""
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10 training with gating')
    parser.add_argument('--config', type=str, required=True,
                        help='path to a JSON config file with all run parameters')
    cli_args = parser.parse_args()

    args = load_config(cli_args.config, SCENARIO_DEFAULTS)
    if args.arch not in model_names:
        raise ValueError(f"Unknown arch {args.arch!r} in config. Choices: {', '.join(model_names)}")
    if args.cmd not in ('train', 'test'):
        raise ValueError(f"config 'cmd' must be 'train' or 'test', got {args.cmd!r}")
    if args.compression_ratios is None:
        raise ValueError("config must set 'compression_ratios' (no default is used)")
    return args


def main():
    """Script entry point: parses the config, sets up logging, and
    dispatches to run_training or test_model based on `args.cmd`."""
    args = parse_args()
    set_seed(args.seed)

    save_path = args.save_path = os.path.join(args.save_folder, args.save_name or args.arch)
    os.makedirs(save_path, exist_ok=True)


    print("======================================== Input Parameters: =============================================")
    for arg, value in vars(args).items():
        print(f"{arg:>15}: {value}")
    print("========================================================================================================\n\n")

    # config logger file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)
_LINEAR_SVD_TYPES = (Linear_SVD_NonUnified, Linear_SVD_Unified)
_CONV1X1_SVD_TYPES = (Conv1x1_SVD_NonUnified, Conv1x1_SVD_Unified)
_TUCKER_TYPES = (Conv_Tucker_NonUnified, Conv_Tucker_Unified)
# Baseline-checkpoint weight ndim/shape "kind" each adaptive layer kind warm-starts from.
_KIND_TO_BASELINE_KIND = {'linear_svd': 'linear', 'conv1x1_svd': 'conv1x1', 'tucker': 'conv'}


_KIND_TO_GATE_SUFFIX = {'tucker': '_tucker_layer', 'conv1x1_svd': '_svd_layer', 'linear_svd': '_svd_layer'}


def discover_adaptive_layers(model):
    """Finds every low-rank (SVD/Tucker) layer in `model` and pairs each one
    with its RNN gate. Which modules are low-rank layers is found by
    isinstance checks (see _all_target_slots), architecture-agnostic; but
    pairing a layer with its gate is name-based, since a gate is just a
    plain nn.Sequential with no distinguishing type - a layer registered as
    `<name>_svd_layer`/`<name>_tucker_layer` must have its gate registered
    as `<name>_gate` on the same module.

    Input: an adaptive model instance.

    Output: a list of dicts (one per layer, in construction order) with
    keys name, module, kind ('linear_svd'/'conv1x1_svd'/'tucker'),
    adaptive, gate. Raises RuntimeError if a layer's expected gate is
    missing."""
    named_modules = dict(model.named_modules())
    low_rank, _ = _all_target_slots(model)
    layers = []
    for slot in low_rank:
        name, kind = slot['name'], slot['kind']
        suffix = _KIND_TO_GATE_SUFFIX[kind]
        if not name.endswith(suffix):
            raise RuntimeError(
                f"Low-rank layer '{name}' doesn't end in '{suffix}'; can't "
                f"derive its gate's name from it."
            )
        gate_name = name[:-len(suffix)] + '_gate'
        if gate_name not in named_modules:
            raise RuntimeError(f"Low-rank layer '{name}' has no matching gate at '{gate_name}'.")
        layers.append({**slot, 'gate': named_modules[gate_name]})
    return layers


def _baseline_layer_candidates(state_dict):
    """Lists every Linear/Conv2d weight in a plain-backbone checkpoint, as
    a candidate a low-rank layer could warm-start from.

    Input: a plain-backbone state_dict.

    Output: a list of dicts (one per weight, in the order the backbone
    constructed them) with keys prefix, weight, bias, kind
    ('linear'/'conv1x1'/'conv')."""
    candidates = []
    for key, tensor in state_dict.items():
        if not key.endswith('.weight') or tensor.dim() not in (2, 4):
            continue
        if tensor.dim() == 2:
            kind = 'linear'
        else:
            kind = 'conv1x1' if tensor.shape[2] == 1 and tensor.shape[3] == 1 else 'conv'
        prefix = key[:-len('.weight')]
        candidates.append({
            'prefix': prefix,
            'weight': tensor,
            'bias': state_dict.get(f"{prefix}.bias", None),
            'kind': kind,
        })
    return candidates


def _all_target_slots(model):
    """Splits every layer in `model` that has (or should have) a baseline
    counterpart into low-rank layers and untouched plain nn.Conv2d/
    nn.Linear layers. Excludes gate internals (the shared `control`
    controller and each layer's own `<name>_gate` feature extractor) since
    those are new to the adaptive architecture and have no baseline
    counterpart at all.

    Input: an adaptive model instance.

    Output: (low_rank, plain) - two lists of dicts, each with keys name,
    module, kind, adaptive. `plain` is used by
    match_adaptive_layers_to_baseline to keep a non-adaptive baseline
    layer from being mistaken for a low-rank layer's counterpart just
    because they share a shape."""
    named_modules = dict(model.named_modules())
    exclude_roots = {'control'}
    low_rank = []
    for name, module in named_modules.items():
        if isinstance(module, _TUCKER_TYPES):
            kind, suffix = 'tucker', '_tucker_layer'
        elif isinstance(module, _CONV1X1_SVD_TYPES):
            kind, suffix = 'conv1x1_svd', '_svd_layer'
        elif isinstance(module, _LINEAR_SVD_TYPES):
            kind, suffix = 'linear_svd', '_svd_layer'
        else:
            continue
        if name.endswith(suffix):
            exclude_roots.add(name[:-len(suffix)] + '_gate')
        low_rank.append({'name': name, 'module': module, 'kind': kind, 'adaptive': True})

    def is_excluded(name):
        return any(name == root or name.startswith(root + '.') for root in exclude_roots)

    plain = []
    for name, module in named_modules.items():
        if is_excluded(name):
            continue
        if isinstance(module, nn.Conv2d):
            kind = 'conv1x1' if tuple(module.kernel_size) == (1, 1) else 'conv'
            plain.append({'name': name, 'module': module, 'kind': kind, 'adaptive': False})
        elif isinstance(module, nn.Linear):
            plain.append({'name': name, 'module': module, 'kind': 'linear', 'adaptive': False})
    return low_rank, plain


def _expected_baseline_shape(slot):
    """Computes the baseline nn.Conv2d/nn.Linear weight shape a low-rank
    slot expects to warm-start from.

    Input: a low-rank slot dict (from _all_target_slots).

    Output: a shape tuple, read off the buffers the layer recorded at
    construction (in_features/out_features for Linear,
    in_channels/out_channels/spatial_size for Conv1x1,
    in_channels/out_channels/kernel_size for Tucker)."""
    module, kind = slot['module'], slot['kind']
    if kind == 'linear_svd':
        return (int(module.out_features.item()), int(module.in_features.item()))
    elif kind == 'conv1x1_svd':
        ndims = module.spatial_size.numel()
        return (int(module.out_channels.item()), int(module.in_channels.item())) + (1,) * ndims
    else:  # tucker
        kernel_dims = tuple(int(k) for k in module.kernel_size.tolist())
        return (int(module.out_channels.item()), int(module.in_channels.item())) + kernel_dims


def _compute_layer_ranks_and_flops(adaptive_layers, compression_rates):
    """Computes the per-choice rank and FLOPs cost of every adaptive
    layer, for the computation-cost regularizer and the FLOPs report.

    Input: the model's adaptive layers (from discover_adaptive_layers) and
    the configured compression ratios.

    Output: (ranks_per_layer, flops_per_rank, layers_full_flops) - three
    lists, one entry per layer; ranks_per_layer/flops_per_rank each list
    one value per gate choice (including the min-rank and original-weight
    choices where present), layers_full_flops is the layer's single
    uncompressed FLOPs count."""
    ranks_per_layer = []
    flops_per_rank = []
    layers_full_flops = []
    for layer in adaptive_layers:
        module = layer['module']
        layer_ranks = []
        if module.min_rank is not None:
            layer_ranks.append(int(module.min_rank.item())) # for the min-rank choice that is corresponding to the compression_ratio of 99

        if layer['kind'] == 'tucker':
            in_spatial_size = module.in_spatial_size.tolist()
            out_spatial_size = module.out_spatial_size.tolist()
            kernel_size = module.kernel_size.tolist()
            layer_ranks.extend(calculate_conv2D_tucker_ranks(compression_rates, in_spatial_size, out_spatial_size,
                                                    module.in_channels.item(), module.out_channels.item(), kernel_size))
            layer_ranks.append(math.inf) # for the original weight choice
            ranks_per_layer.append(layer_ranks)
            flops_per_rank.append(calculate_tucker_flops(layer_ranks, in_spatial_size, out_spatial_size,
                                                    module.in_channels.item(), module.out_channels.item(), kernel_size))
            layers_full_flops.append(calculate_Conv2D_flops(out_spatial_size, module.in_channels.item(),
                                                    module.out_channels.item(), kernel_size))
        elif layer['kind'] == 'conv1x1_svd':
            spatial_size = module.spatial_size.tolist()
            layer_ranks.extend(calculate_fc_ranks(compression_rates, module.in_channels.item(), module.out_channels.item()))
            layer_ranks.append(math.inf) # for the original weight choice
            ranks_per_layer.append(layer_ranks)
            flops_per_rank.append(calculate_svd_conv1x1_flops(layer_ranks, spatial_size,
                                                    module.in_channels.item(), module.out_channels.item()))
            layers_full_flops.append(calculate_conv1x1_flops(spatial_size,
                                                    module.in_channels.item(), module.out_channels.item()))
        else:  # linear_svd
            layer_ranks.extend(calculate_fc_ranks(compression_rates, module.in_features.item(), module.out_features.item()))
            layer_ranks.append(math.inf) # for the original weight choice
            ranks_per_layer.append(layer_ranks)
            flops_per_rank.append(calculate_svd_fc_flops(layer_ranks, module.in_features.item(), module.out_features.item()))
            layers_full_flops.append(calculate_fc_flops(module.in_features.item(), module.out_features.item()))
    return ranks_per_layer, flops_per_rank, layers_full_flops


def match_adaptive_layers_to_baseline(model, baseline_candidates):
    """Pairs each of `model`'s low-rank layers with the baseline-checkpoint
    weight it should warm-start from. Plain layers are matched first, by
    exact dotted name, and removed from the candidate pool; each low-rank
    layer then takes the first remaining candidate with the right kind and
    shape, walking forward in construction order (a forward-only cursor,
    not a fresh search each time, so it can't match the same baseline
    weight twice).

    Input: the model and the baseline candidates (from
    _baseline_layer_candidates).

    Output: a list of (low_rank_slot, baseline_candidate) pairs, one per
    low-rank layer. Raises RuntimeError naming the layer if no matching
    candidate is left for it."""
    low_rank, plain = _all_target_slots(model)
    baseline_by_name = {c['prefix']: c for c in baseline_candidates}
    claimed = {slot['name'] for slot in plain if slot['name'] in baseline_by_name}
    remaining = [c for c in baseline_candidates if c['prefix'] not in claimed]

    pairs = []
    cursor = 0
    for slot in low_rank:
        want_kind = _KIND_TO_BASELINE_KIND[slot['kind']]
        expected_shape = _expected_baseline_shape(slot)
        match = None
        for idx in range(cursor, len(remaining)):
            candidate = remaining[idx]
            if candidate['kind'] == want_kind and tuple(candidate['weight'].shape) == expected_shape:
                match = idx
                break
        if match is None:
            raise RuntimeError(
                f"apply_lrf_on_target_layers: no baseline-checkpoint layer found for "
                f"adaptive layer '{slot['name']}' (expected a {want_kind} weight of "
                f"shape {expected_shape}, searching forward from position {cursor} among "
                f"the {len(remaining)} baseline layer(s) not claimed by a plain/non-adaptive "
                f"layer). The baseline checkpoint and adaptive architecture appear to be "
                f"out of sync."
            )
        pairs.append((slot, remaining[match]))
        cursor = match + 1
    return pairs


def apply_lrf_on_target_layers(adaptive_model, checkpoint):
    """Warm-starts every low-rank layer in `adaptive_model` by decomposing
    (SVD for Linear/Conv1x1, Tucker for general conv) its matched weight
    in `checkpoint` (see match_adaptive_layers_to_baseline).

    Input: an adaptive model instance and a plain-backbone checkpoint
    (dict with a 'state_dict' key, or the state_dict itself).

    Output: none - mutates `adaptive_model`'s low-rank layers in place."""
    sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    baseline_candidates = _baseline_layer_candidates(sd)
    pairs = match_adaptive_layers_to_baseline(adaptive_model, baseline_candidates)

    logging.info(f"=> warm-starting {len(pairs)} low-rank layer(s) from the baseline checkpoint:")
    for layer, baseline in pairs:
        logging.info(f"     {layer['name']} ({layer['kind']}) <- {baseline['prefix']}")
        W, b, module = baseline['weight'], baseline['bias'], layer['module']

        if layer['kind'] == 'tucker':
            core, factors = partial_tucker(W, rank=None, modes=[0, 1], init="svd")
            core_tensor = core[0].float()
            third_tensor = core[1][0].float()
            first_tensor = core[1][1].float()
            module.set_weights(W, core_tensor, [third_tensor, first_tensor], b)
        else:
            if layer['kind'] == 'conv1x1_svd':
                ndims = module.spatial_size.numel()  # 1 for Conv1d, 2 for Conv2d, ...
                W_2d = W.view(W.shape[0], W.shape[1])  # drop the (all size-1) trailing spatial dims
            else:
                W_2d = W
            U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)  # U: [out, rank], S: [rank], Vh: [rank, in]
            if layer['kind'] == 'conv1x1_svd':
                kernel_ones = (1,) * ndims
                U = U.view(*U.shape, *kernel_ones)       # [out_c, rank, *spatial]
                Vh = Vh.view(*Vh.shape, *kernel_ones)    # [rank, in_c, *spatial]
                S = S.view(1, -1, *kernel_ones)          # [1, rank, *spatial]
            if len(inspect.signature(module.set_svd_weights).parameters) == 5:
                module.set_svd_weights(W, U, S, Vh, b)
            else:
                module.set_svd_weights(U, S, Vh, b)


def _freeze_all_but_biggest_choice(adaptive_layers):
    """Freezes every per-choice sub-module of each low-rank layer except
    its biggest (least-compressed) choice - the '0' (original-weight)
    choice if present, else the smallest compression ratio. Unified-variant
    layers have no per-choice sub-modules (every choice is a slice of one
    shared basis) and are left untouched.

    Input: the model's adaptive layers (from discover_adaptive_layers).

    Output: none - sets `requires_grad` on each choice's parameters in
    place."""
    for layer in adaptive_layers:
        choices = getattr(layer['module'], 'options_object_dictionary', None)
        if choices is None:
            continue
        keys = list(choices.keys())
        if '0' in keys:
            biggest_key = '0'
        else:
            biggest_key = min((k for k in keys if k != '99'), key=lambda k: float(k))
        for key, submodule in choices.items():
            for param in submodule.parameters():
                param.requires_grad = (key == biggest_key)


def fine_tune_weights(model, args, adaptive_layers, train_loader, test_loader, layers_full_flops, flops_per_rank,
                       criterion, metric_fn):
    """Backbone warm-start step 1: freezes the gate and every low-rank
    layer's smaller-rank choices, then fine-tunes only each layer's
    biggest choice (the one warm-started closest to the original baseline
    weight) for `args.fine_tune_epochs`.

    Input: the model, its adaptive layers, the data loaders, the
    precomputed FLOPs bookkeeping, and the criterion/metric to train and
    validate against.

    Output: none - loads the best-validation-metric weights back into
    `model` in place."""
    for param in model.parameters():
        param.requires_grad = True
    for param in model.control.parameters():
        param.requires_grad = False
    for layer in adaptive_layers:
        for param in layer['gate'].parameters():
            param.requires_grad = False
    _freeze_all_but_biggest_choice(adaptive_layers)

    metric_label = args.metric
    optimizer = build_optimizer(args.optimizer, filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    best_prec1 = 0
    best_state = None

    print("======================   Validation before fine-tune ========================")
    validate(args, test_loader, model, criterion, metric_fn, layers_full_flops, flops_per_rank)
    print("=============================================================================\n\n")

    for i in range(args.fine_tune_epochs):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        for batch_idx, (input, target) in enumerate(train_loader):
            # measuring data loading time
            data_time.update(time.time() - end)

            target = target.squeeze().long().cuda()
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            # compute output
            output, masks = model(input_var, args.compression_ratios)
            loss = criterion(output, target_var)

            # measure metric and record loss
            prec1 = metric_fn(output.data, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print log
            if batch_idx % args.print_freq == 0:
                logging.info("Fine-tune epoch: [{0}/{1}]\t"
                            "Batch: {2}\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                            "{metric_label} {top1.val:.3f} ({top1.avg:.3f})\t".format(
                                i,
                                args.fine_tune_epochs,
                                batch_idx,
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=losses,
                                metric_label=metric_label,
                                top1=top1)
                )

        with torch.no_grad():
            prec1, _ = validate(args, test_loader, model, criterion, metric_fn, layers_full_flops, flops_per_rank)
        if prec1 > best_prec1:
            best_prec1 = prec1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)


def warmup_gate(model, args, adaptive_layers, train_loader, test_loader, compression_rates, network_depth,
                 layers_full_flops, flops_per_rank, criterion, metric_fn):
    """Backbone warm-start step 2: freezes every SVD/Tucker weight and
    trains only the gate for `args.gate_warmup_epochs`, with the
    computation-cost regularizer forced to always reward higher
    computation - so every sample routes toward each layer's biggest
    choice, the one fine_tune_weights actually trained - instead of the
    normal minimize-FLOPs pressure used later in run_epoch.

    Input: the model, its adaptive layers, the data loaders, the
    compression ratios/FLOPs bookkeeping, and the criterion/metric to
    train and validate against.

    Output: none - mutates `model` in place and leaves it as the last
    warm-up epoch produced (no best-epoch tracking/reload, unlike
    fine_tune_weights)."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.control.parameters():
        param.requires_grad = True
    for layer in adaptive_layers:
        for param in layer['gate'].parameters():
            param.requires_grad = True

    metric_label = args.metric
    optimizer = build_optimizer(args.optimizer, filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # unreachable target (cp_ratio is always <= 100%) so reg below is
    # always -1, i.e. always rewards higher computation - see docstring
    warmup_target_computation_ratio = 100

    reg = 1
    for i in range(args.gate_warmup_epochs):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        cp_record = AverageMeter()
        end = time.time()
        for batch_idx, (input, target) in enumerate(train_loader):
            # measuring data loading time
            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            # compute output
            output, masks = model(input_var, compression_rates)

            computation_cost = 0
            computation_all = 0
            for layer in range(network_depth):
                sample_number = reduce((lambda x, y: x * y), masks[layer][0].shape)
                full_layer = sample_number * layers_full_flops[layer]
                computation_all += full_layer
                for k in range(gate_num_classes_for(args.arch, compression_rates, args.include_original_weight)):
                    computation_cost += masks[layer][k].sum() * flops_per_rank[layer][k]

            cp_ratio = (float(computation_cost.item()) / float(computation_all)) * 100
            normalized_computation_cost = computation_cost / computation_all * 100 * args.beta
            if cp_ratio <= warmup_target_computation_ratio:
                reg = -1
            else:
                reg = 1

            if args.computation_cost:
                loss = criterion(output, target_var) + normalized_computation_cost * reg
            else:
                loss = criterion(output, target_var)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure metric and record loss
            prec1 = metric_fn(output.data, target)
            losses.update(loss.detach().item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            cp_record.update(cp_ratio, input.size(0))

            # repackage hidden units for RNN Gate
            if args.gate_type == 'rnn':
                model.control.repackage_hidden()

            batch_time.update(time.time() - end)
            end = time.time()

            # print log
            if batch_idx % args.print_freq == 0:
                logging.info("Gate warmup epoch: [{0}/{1}]\t"
                            "Batch:{2}\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                            "{metric_label} {top1.val:.3f} ({top1.avg:.3f})\t"
                            "Computation_Percentage: {cp_record.val:.3f}({cp_record.avg:.3f})\t".format(
                                i,
                                args.gate_warmup_epochs,
                                batch_idx,
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=losses,
                                metric_label=metric_label,
                                top1=top1,
                                cp_record=cp_record)
                )

        with torch.no_grad():
            validate(args, test_loader, model, criterion, metric_fn, layers_full_flops, flops_per_rank)
    # no best-epoch tracking here - unlike fine_tune_weights, `model` is kept
    # exactly as the last warm-up epoch left it, not reloaded to any earlier state


def run_epoch(model, args, epoch, train_loader, test_loader, compression_rates, network_depth,
               layers_full_flops, flops_per_rank, optimizer, criterion, metric_fn,
               best_prec1, threshold_metric, best_computation):
    """One epoch of the main joint-training loop: trains every parameter
    (weights and gate) under the computation-cost regularizer, then
    validates and checkpoints to disk - but only if this epoch both
    clears `threshold_metric` and costs less (in FLOPs) than
    `best_computation`, so saved checkpoints only ever get cheaper while
    staying at or above the metric bar.

    Input: the model, its training state (optimizer, criterion, metric),
    the data loaders, the compression ratios/FLOPs bookkeeping, and the
    current best metric/computation seen so far.

    Output: the updated (best_prec1, best_computation)."""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()
    cp_record = AverageMeter()
    metric_label = args.metric
    end = time.time()
    reg = 1
    # adjust_learning_rate(args, optimizer, epoch)
    for batch_idx, (input, target) in enumerate(train_loader):

        # measuring data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = Variable(input).cuda()
        target_var = Variable(target).cuda()

        # compute output
        output, masks = model(input_var, compression_rates)

        computation_cost = 0
        computation_all = 0
        normalized_computation_cost = 0
        for layer in range(network_depth):
            sample_number = reduce((lambda x, y: x * y), masks[layer][0].shape)
            full_layer = sample_number * layers_full_flops[layer]
            computation_all  += full_layer
            for k in range(gate_num_classes_for(args.arch, compression_rates, args.include_original_weight)):
                computation_cost += masks[layer][k].sum() * flops_per_rank[layer][k]

        cp_ratio = (float(computation_cost.item()) / float(computation_all)) * 100
        normalized_computation_cost = computation_cost / computation_all * 100 * args.beta
        if args.target_computation_ratio is not None and cp_ratio <= args.target_computation_ratio:
            reg = -1
        else:
            reg = 1

        loss_part_1 = criterion(output, target_var)
        if args.computation_cost:
            loss_part_2 = normalized_computation_cost * reg
            loss = loss_part_1 + loss_part_2
        else:
            loss = loss_part_1
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure metric and record loss
        prec1 = metric_fn(output.data, target)
        losses.update(loss.detach().item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        cp_record.update(cp_ratio, input.size(0))

        # repackage hidden units for RNN Gate
        if args.gate_type == 'rnn':
            model.control.repackage_hidden()

        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if batch_idx % args.print_freq == 0:
            logging.info("Epoch: [{0}/{1}]\t"
                         "Batch:{2}\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                        "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                        "{metric_label} {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Computation_Percentage: {cp_record.val:.3f}({cp_record.avg:.3f})\t".format(
                            epoch,
                            args.epochs,
                            batch_idx,
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            metric_label=metric_label,
                            top1=top1,
                            cp_record=cp_record)
            )

    # evaluate; only checkpoint if this epoch clears the metric bar AND
    # costs less (in FLOPs) than the cheapest checkpoint saved so far - the
    # eval-mode computation ratio from validate() (hard per-sample gate
    # choices), not the training loop's own cp_record above (soft/weighted
    # choices, a proxy used only for the regularizer, not real deployed cost)
    with torch.no_grad():
        prec1, cp_ratio_avg = validate(args, test_loader, model, criterion, metric_fn, layers_full_flops, flops_per_rank)

    if prec1 >= threshold_metric and cp_ratio_avg < best_computation:
        best_prec1 = prec1
        best_computation = cp_ratio_avg
        checkpoint_path = os.path.join(args.save_path,
                                        'checkpoint_{:05d}.pth.tar'.format(
                                            epoch))
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_computation': best_computation,
        },
            is_best = True, filename=checkpoint_path)
        shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                                                        'checkpoint_latest'
                                                        '.pth.tar'))

    return best_prec1, best_computation


def run_training(args):
    """Entry point for `cmd: "train"`: builds the adaptive model, warm-starts
    it from `args.resume` if given (apply_lrf_on_target_layers ->
    fine_tune_weights -> warmup_gate), then runs the main joint-training
    loop (run_epoch) for `args.epochs`.

    Input: the parsed config (see configs/config_utils.py's
    SCENARIO_DEFAULTS for every field used).

    Output: none - trains in place and writes checkpoints to
    `args.save_path`."""
    compression_rates = args.compression_ratios
    costs = compression_rates
    # create model
    min_compression = 99 if not compression_rates else compression_rates[-1]
    min_rank_arg = min_rank_argument(args)
    gate_num_classes = gate_num_classes_for(args.arch, compression_rates, args.include_original_weight)
    model_kwargs = {'gate_num_classes': gate_num_classes}
    if is_plain_arch(args.arch):
        # Unified layers share one basis across all choices, so there's no
        # separate "keep original weight" choice to add or omit.
        model_kwargs['include_original_weight'] = args.include_original_weight
    model = models.__dict__[args.arch](compression_rates, args.pretrained, min_rank_arg, **model_kwargs)
    model = model.cuda()

    adaptive_layers = discover_adaptive_layers(model)
    network_depth = len(adaptive_layers)
    assert network_depth == model.num_adaptive_layers, (
        f"discover_adaptive_layers found {network_depth} low-rank layer(s) but "
        f"model.num_adaptive_layers is {model.num_adaptive_layers} - the model's "
        f"forward() and the layer-discovery walk disagree on the layer count."
    )
    ranks_per_layer, flops_per_rank, layers_full_flops = _compute_layer_ranks_and_flops(adaptive_layers, compression_rates)

    print("======================================== Compression/Rank/FLOPs info ===================================")
    print(f"{'compression_rates':>20}: {str(compression_rates):<15}")
    print(f"{'Ranks':>20}: {str(ranks_per_layer):<15}")
    print(f"{'Ranks_FLOPs':>20}: {str(flops_per_rank):<15}")
    values = [x.item() if isinstance(x, torch.Tensor) else x for x in layers_full_flops]
    print(f"{'Full_FLOPs':>20}: {str(values):<15}")
    print("========================================================================================================\n\n")

    train_loader = prepare_train_data(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # built once and threaded through every phase below (warm-start and
    # joint training alike), so they stay consistent across the whole run
    criterion = build_criterion(args.criterion, **args.criterion_kwargs).cuda()
    metric_fn = build_metric(args.metric)

    best_prec1 = 0
    # `resume`, when given, is always a plain-backbone checkpoint (from
    # train_base.py): warm-start this final architecture's low-rank layers
    # from it, fine-tune the SVD/Tucker weights with the gate frozen, then
    # warm up the gate with the weights frozen, before falling into the
    # joint training loop below. (To evaluate an already-adaptive
    # checkpoint instead of retraining it, use `cmd: "test"` / test_model.)
    if args.resume:
        logging.info('=> loading checkpoint `{}`'.format(args.resume))
        checkpoint = load_checkpoint(args.resume)

        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info('=> loaded backbone checkpoint `{}`'.format(args.resume))

        apply_lrf_on_target_layers(model, checkpoint)
        logging.info('=> LRF applied to the model')
        fine_tune_weights(model, args, adaptive_layers, train_loader, test_loader, layers_full_flops, flops_per_rank,
                           criterion, metric_fn)
        logging.info('=> low-rank weights fine-tuned')

        warmup_gate(model, args, adaptive_layers, train_loader, test_loader, compression_rates, network_depth,
                    layers_full_flops, flops_per_rank, criterion, metric_fn)
        logging.info('=> gate warmed up')

    cudnn.benchmark = False

    # joint training: every parameter (weights and gate alike) is trainable,
    # so unfreeze everything *before* building the optimizer - building it
    # first would filter model.parameters() down to whatever was left
    # trainable by the warm-start phase above (just the gate), and
    # unfreezing afterward wouldn't add the newly-unfrozen weight params to
    # an optimizer that's already been constructed.
    for param in model.parameters():
        param.requires_grad = True

    optimizer = build_optimizer(args.optimizer, filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # an epoch is only checkpointed once its validation metric is at or above
    # this - in absolute metric units (not a percentage of the baseline), so
    # it works the same way whether `metric` is accuracy or something else -
    # see `metric_base_value`/`metric_drop_thr` in configs/config_utils.py.
    # -math.inf (rather than 0) when disabled, so the gate is a true no-op
    # for a metric that can go negative (e.g. neg_mse), not just for accuracy.
    if args.metric_drop_thr is not None:
        threshold_metric = args.metric_base_value - args.metric_drop_thr
    else:
        threshold_metric = -math.inf

    # a disk checkpoint only ever gets written when it both clears
    # threshold_metric and is cheaper (in FLOPs) than the best one written
    # so far - math.inf means "nothing saved yet", so the first qualifying
    # epoch always wins regardless of its own computation
    best_computation = math.inf

    for i in range(args.start_epoch, args.epochs):
        best_prec1, best_computation = run_epoch(model, args, i, train_loader, test_loader, compression_rates,
                                network_depth, layers_full_flops, flops_per_rank,
                                optimizer, criterion, metric_fn, best_prec1, threshold_metric, best_computation)

def validate(args, test_loader, model, criterion, metric_fn, layers_full_flops_arr=None, flops_per_rank_arr=None):
    """Runs `model` in eval mode over `test_loader`, logging loss/metric
    and each layer's gate-choice distribution.

    Input: the model and test loader, the criterion/metric to evaluate
    with, and optionally precomputed FLOPs bookkeeping (recomputed from
    `model` if omitted).

    Output: (metric_avg, computation_ratio_avg) - the average metric and
    the average % of full-network FLOPs actually used under eval-mode
    (hard, per-sample) gate choices."""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()
    cp_record = AverageMeter()
    metric_label = args.metric

    compression_rates = args.compression_ratios
    costs = compression_rates
    adaptive_layers = discover_adaptive_layers(model)
    network_depth = len(adaptive_layers)
    assert network_depth == model.num_adaptive_layers, (
        f"discover_adaptive_layers found {network_depth} low-rank layer(s) but "
        f"model.num_adaptive_layers is {model.num_adaptive_layers} - the model's "
        f"forward() and the layer-discovery walk disagree on the layer count."
    )
    layerwise_decision_statistics = []
    for k in range(network_depth):
        layerwise_decision_statistics.append([])
        for j in range(gate_num_classes_for(args.arch, compression_rates, args.include_original_weight)):
            ratio = AverageMeter()
            layerwise_decision_statistics[k].append(ratio)

    if layers_full_flops_arr is None:
        ranks_per_layer, flops_per_rank, layers_full_flops = _compute_layer_ranks_and_flops(adaptive_layers, compression_rates)
    else:
        flops_per_rank = flops_per_rank_arr
        layers_full_flops = layers_full_flops_arr

    # switch to evaluation mode
    model.eval()
    end = time.time()
    count_c = [ [0] * gate_num_classes_for(args.arch, compression_rates, args.include_original_weight) for _ in range(network_depth)]
    final_computations_all = 0
    final_computations_dynamic = 0
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda(non_blocking=True)
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        # compute output
        output, masks = model(input_var,compression_rates)

        computation_cost = 0
        computation_all = 0

        for layer in range(network_depth):
            sample_number = reduce((lambda x, y: x * y), masks[layer][0].shape)
            full_layer = sample_number * layers_full_flops[layer]
            computation_all += full_layer

            # per-layer histogram of which compression choice the gate picked
            gate_tensor = torch.stack(masks[layer])  # shape: (num_gates, batch_size)
            gate_tensor = gate_tensor.T              # shape: (batch_size, num_gates)
            selected_layer_per_sample = gate_tensor.argmax(dim=1)
            counts = torch.bincount(selected_layer_per_sample, minlength=(len(compression_rates)+1))
            for c in range(gate_num_classes_for(args.arch, compression_rates, args.include_original_weight)):
                count_c[layer][c] += counts[c].item()

            for k in range(gate_num_classes_for(args.arch, compression_rates, args.include_original_weight)):
                dynamic_choice = masks[layer][k].sum()
                ratio = dynamic_choice / sample_number
                layerwise_decision_statistics[layer][k].update(ratio, 1)
                computation_cost += int(masks[layer][k].sum()) * flops_per_rank[layer][k]
        final_computations_dynamic += computation_cost
        final_computations_all += computation_all
        cp_ratio = (float(computation_cost) / float(computation_all)) * 100

        loss = criterion(output, target_var)

        # measure metric and record loss
        prec1 = metric_fn(output.data, target)
        top1.update(prec1.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        cp_record.update(cp_ratio, input.size(0))


        final_computations_dynamic_2 = 0.0
        for layer in  range(network_depth):
            for k in range(gate_num_classes_for(args.arch, compression_rates, args.include_original_weight)):
                final_computations_dynamic_2 += count_c[layer][k] * flops_per_rank[layer][k]
        if i % args.print_freq == 0 or (i == (len(test_loader) - 1)):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                '{metric_label}: {top1.val:.3f}({top1.avg:.3f})\t'
                'Computation_Percentage:{cp_record.val:.3f}({cp_record.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses,
                    metric_label=metric_label,
                    top1=top1,
                    cp_record=cp_record
                )
            )

    logging.info(' * {metric_label} {top1.avg:.3f}, Loss {loss.avg:.3f}'.format(
        metric_label=metric_label, top1=top1, loss=losses))
    print("======================================== Computations info =============================================")
    print(f"{'Compression rates':>30}: {str(compression_rates):<30}")
    print(f"{'final_computations_dynamic':>30}: {str(final_computations_dynamic):<20}")
    print(f"{'final_computations_all':>30}: {final_computations_all:<20}")
    print("========================================================================================================\n\n")

    class_labels = gate_class_labels(args.arch, compression_rates, args.include_original_weight)
    for layer in range(network_depth):
        print(f"{adaptive_layers[layer]['name']} decision distribution:")
        for g, label in enumerate(class_labels):
            print(f"  {label}: {layerwise_decision_statistics[layer][g].avg}")

    # per-layer count of how many samples picked each compression choice
    for i in range(network_depth):
        print(f"{adaptive_layers[i]['name']} decision counts:")
        for j, label in enumerate(class_labels):
            print(f"  {label}: {count_c[i][j]}")

    # cp_record.avg: the average % of full-network FLOPs actually used under
    # eval-mode (hard, per-sample) gate choices - the real deployed
    # computation cost, used by run_epoch to pick which checkpoint to keep
    return top1.avg, cp_record.avg


def test_model(args):
    """Entry point for `cmd: "test"`: loads an already-adaptive checkpoint
    from `args.resume` directly (no warm-start) and evaluates it.

    Input: the parsed config.

    Output: none - prints the evaluation results."""
    compression_rates = args.compression_ratios
    costs = compression_rates
    min_compression = 99 if not compression_rates else compression_rates[-1]
    gate_num_classes = gate_num_classes_for(args.arch, compression_rates, args.include_original_weight)
    model_kwargs = {'gate_num_classes': gate_num_classes}
    if is_plain_arch(args.arch):
        # Unified layers share one basis across all choices, so there's no
        # separate "keep original weight" choice to add or omit.
        model_kwargs['include_original_weight'] = args.include_original_weight
    # create model
    model = models.__dict__[args.arch](compression_rates, min_rank=min_rank_argument(args), pretrained=args.pretrained,
                                        **model_kwargs)
    model = model.cuda()

    if args.resume:
        logging.info('=> loading checkpoint `{}`'.format(args.resume))
        checkpoint = load_checkpoint(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
            args.resume, checkpoint['epoch']
        ))
    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = build_criterion(args.criterion, **args.criterion_kwargs).cuda()
    metric_fn = build_metric(args.metric)

    with torch.no_grad():
        validate(args, test_loader, model, criterion, metric_fn)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves `state` to `filename`, and also to `model_best.pth.tar` in
    the same directory if `is_best`. No return value."""
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count


def adjust_learning_rate(args, optimizer, _epoch):
    """Step-decays `optimizer`'s LR by `args.step_ratio` at 50% of
    `args.epochs`, and again by `args.step_ratio` at 75% (the original
    32k/48k thresholds were sized for counting iterations, not epochs, and
    never triggered at this codebase's epoch scale - a few hundred at
    most). Not currently called anywhere - run_epoch's call to it is
    commented out."""
    decay_epoch_1 = int(args.epochs * 0.5)
    decay_epoch_2 = int(args.epochs * 0.75)
    if args.warm_up and (_epoch < 400):
        lr = 0.01
    elif decay_epoch_1 <= _epoch < decay_epoch_2:
        lr = args.lr * (args.step_ratio ** 1)
    elif _epoch >= decay_epoch_2:
        lr = args.lr * (args.step_ratio ** 2)
    else:
        lr = args.lr

    if _epoch % args.eval_every == 0:
        logging.info('Epoch [{}] learning rate = {}'.format(_epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
