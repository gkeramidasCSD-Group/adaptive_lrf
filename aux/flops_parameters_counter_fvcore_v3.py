"""Reports parameter count and MACs (via fvcore) for a trained checkpoint,
broken down by conv1x1 / FC / gate-related modules.

Run with --help for the full option list. Example:

    python aux/flops_parameters_counter_fvcore_v3.py \\
        --arch cifar10_efficientformer_l3_adaptive \\
        --checkpoint save_checkpoints/cifar10_efficientformer_l3_adaptive/model_best.pth.tar \\
        --compression-ratios 0.88,0.75,0.5,0.2 \\
        --no-include-original-weight
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape

import arch.models_baseline as models_baseline
import arch.models_adaptive as models_adaptive
from configs.config_utils import is_plain_arch, gate_num_classes_for


def parse_args():
    """Parses the command-line options this script runs with. Output: an
    argparse.Namespace."""
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", required=True,
                        help="factory name in arch.models_adaptive or arch.models_baseline")
    parser.add_argument("--checkpoint", required=True, help="path to the checkpoint's .pth.tar file")
    parser.add_argument("--input-size", default="3,32,32",
                        help="comma-separated (channels,height,width) of one sample, e.g. 1,28,28 for MNIST")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--compression-ratios", default="",
                        help="comma-separated compression ratios, e.g. 0.88,0.75,0.5,0.2 "
                             "(required for an adaptive --arch; ignored for a baseline one)")
    parser.add_argument("--min-rank", default="",
                        help="comma-separated min-rank value(s); a single value, or one per "
                             "ConvNeXt stage. Empty (default) means no min-rank choice.")
    parser.add_argument("--gate-type", default="rnn")
    parser.add_argument("--include-original-weight", dest="include_original_weight",
                        action="store_true", default=True)
    parser.add_argument("--no-include-original-weight", dest="include_original_weight",
                        action="store_false",
                        help="drop the \"keep original weight\" gate choice - see README.md's "
                             "\"Weight-sharing variants\" section")
    parser.add_argument("--name-keywords", default="gate,control",
                        help="comma-separated module-name suffixes whose MACs are reported "
                             "separately as the \"target\" subtotal")
    parser.add_argument("--device", default=None, help="defaults to cuda:0 if available, else cpu")
    args = parser.parse_args()

    args.input_size = tuple(int(v) for v in args.input_size.split(","))
    args.compression_ratios = [float(v) for v in args.compression_ratios.split(",") if v != ""]
    args.min_rank = [int(v) for v in args.min_rank.split(",") if v != ""]
    if len(args.min_rank) == 1:
        args.min_rank = args.min_rank[0]
    args.name_keywords = [v for v in args.name_keywords.split(",") if v != ""]
    args.device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args


def build_model(args):
    """Builds the model named by `args.arch`, from either models_adaptive
    (using args.compression_ratios/min_rank/gate_type/include_original_weight)
    or models_baseline (ignoring those). Output: a model instance. Raises
    ValueError if `args.arch` is in neither module."""
    if hasattr(models_adaptive, args.arch):
        if not args.compression_ratios:
            raise ValueError(f"--compression-ratios is required for adaptive arch {args.arch!r}")
        factory = getattr(models_adaptive, args.arch)
        include_original_weight = is_plain_arch(args.arch) and args.include_original_weight
        kwargs = {
            "gate_type": args.gate_type,
            "gate_num_classes": gate_num_classes_for(args.arch, args.compression_ratios, include_original_weight, args.min_rank),
        }
        if is_plain_arch(args.arch):
            kwargs["include_original_weight"] = args.include_original_weight
        return factory(args.compression_ratios, min_rank=args.min_rank, **kwargs)
    if hasattr(models_baseline, args.arch):
        return getattr(models_baseline, args.arch)()
    raise ValueError(f"Unknown --arch {args.arch!r}: not in arch.models_adaptive or arch.models_baseline")

# -------------------------------
# Custom JIT MAC HANDLERS
# -------------------------------
def elementwise_mul_mac_jit(inputs, outputs):
    """
    2 FLOPs = 1 MAC convention
    1 multiply = 0.5 MAC
    """
    shape = get_shape(outputs[0])
    if shape is None:
        return 0

    numel = int(torch.prod(torch.tensor(shape)))
    return numel // 2
def elementwise_mac_jit(inputs, outputs):
    # 1 MAC per element for add/mul
    shape = get_shape(outputs[0])
    return int(torch.prod(torch.tensor(shape)))
def add_inplace_mac_jit(inputs, outputs):
    """
    MACs for in-place addition: 1 MAC per element.
    """
    shape = get_shape(outputs[0])
    macs = int(torch.prod(torch.tensor(shape)))
    return macs
def relu6_mac_jit(inputs, outputs):
    """
    MACs for ReLU6: 1 MAC per element (elementwise operation).
    """
    shape = get_shape(outputs[0])
    macs = int(torch.prod(torch.tensor(shape)))
    return macs

def pow_mac_jit(inputs, outputs):
    # use fvcore helper
    shape = get_shape(outputs[0])
    macs = int(torch.prod(torch.tensor(shape)))
    return macs
def mean_mac_jit(inputs, outputs):
    out_shape = get_shape(outputs[0])
    in_shape = get_shape(inputs[0])
    reduction_size = int(torch.prod(torch.tensor(in_shape))) // int(torch.prod(torch.tensor(out_shape)))
    macs = int(torch.prod(torch.tensor(out_shape))) * reduction_size
    return macs

def sqrt_mac_jit(inputs, outputs):
    shape = get_shape(outputs[0])
    macs = int(torch.prod(torch.tensor(shape)))
    return macs

def div_mac_jit(inputs, outputs):
    shape = get_shape(outputs[0])
    macs = int(torch.prod(torch.tensor(shape)))
    return macs

def gelu_mac_jit(inputs, outputs):
    shape = get_shape(outputs[0])
    num_elements = int(torch.prod(torch.tensor(shape)))
    macs = num_elements * 5
    return macs





def max_pool2d_mac_jit(inputs, outputs):
    x_shape = get_shape(inputs[0])
    out_shape = get_shape(outputs[0])

    # kernel size
    # inputs[1] = weight? No, MaxPool doesn't have weight in aten::max_pool2d
    # So we get kernel size from input vs output shape
    B, C, H_out, W_out = out_shape
    H_in, W_in = x_shape[2], x_shape[3]

    # approximate kernel: how many input pixels contributed to each output pixel
    # assume stride divides exactly
    kH = H_in // H_out
    kW = W_in // W_out

    macs = B * C * H_out * W_out * kH * kW  # 1 comparison per element in kernel
    return macs

def conv2d_mac_jit(inputs, outputs):
    x_shape = get_shape(inputs[0])
    w_shape = get_shape(inputs[1])
    out_shape = get_shape(outputs[0])

    B, Cout, H, W = out_shape
    Cin = w_shape[1]
    kH, kW = w_shape[2], w_shape[3]

    macs = B * H * W * Cout * Cin * kH * kW  # standard conv MACs

    # Bias counts as 1 MAC per output element
    if len(inputs) > 2 and inputs[2] is not None:
        macs += B * H * W * Cout

    return macs

def linear_mac_jit(inputs, outputs):
    x_shape = get_shape(inputs[0])
    w_shape = get_shape(inputs[1])
    B = x_shape[0]
    in_features = w_shape[1]
    out_features = w_shape[0]
    macs = B * in_features * out_features
    # Add bias as 1 MAC per output element
    if len(inputs) > 2 and inputs[2] is not None:
        macs += B * out_features
    return macs

def sub_mac_jit(inputs, outputs):
    shape = get_shape(outputs[0])
    return int(torch.prod(torch.tensor(shape)))

def lstm_mac_jit(inputs, outputs):
    """
    Safe JIT handler for aten::lstm

    Counts:
        - Matrix multiplies
        - Bias adds as MACs
        - 2 FLOPs = 1 MAC
    """

    x_shape = get_shape(inputs[0])       # (seq_len, batch, input_size)
    out_shape = get_shape(outputs[0])    # (seq_len, batch, num_directions * hidden_size)

    if x_shape is None or out_shape is None:
        return 0

    seq_len, batch_size, input_size = x_shape
    _, _, hidden_total = out_shape

    # bidirectionality isn't visible in this op, so this always assumes 1
    # direction (works for every LSTM gate in this codebase, which are not
    # bidirectional)
    num_directions = 1
    hidden_size = hidden_total

    # number of layers also isn't visible here; approximated as 1 layer
    # per aten::lstm call (true for every gate in this codebase)
    num_layers = 1

    # Matrix MACs per timestep per layer per direction
    mac_per_timestep = 4 * (
        input_size * hidden_size +
        hidden_size * hidden_size
    )

    # Bias: 8H per timestep
    mac_per_timestep += 4 * hidden_size

    total_macs = (
        mac_per_timestep *
        seq_len *
        batch_size *
        num_layers *
        num_directions
    )

    return total_macs

def svdlinear_module_macs(module, input, output):
    """MACs for an SVD-decomposed Linear at its currently selected rank.

    Input: the module, its input/output tensors. Expects
    `module.current_rank` to have been set externally before the forward
    call. Output: an int MAC count."""

    x = input[0]
    B = x.shape[0]

    in_features = module.in_features.item()
    out_features = module.out_features.item()

    # assumes forward was called with module.current_rank set externally
    rank = module.current_rank

    macs = B * (
        in_features * rank +
        rank * out_features
    )

    selected_obj = module.options_object_dictionary[module.current_key]

    return macs

# -------------------------------
# LSTM MAC COUNT
# -------------------------------

def count_lstm_macs(model, input_tensor):
    """Counts LSTM MACs by running one forward pass with hooks on every
    nn.LSTM. Input: model and a sample input. Output: total LSTM MACs (int)."""
    lstm_macs = 0
    hooks = []

    def hook_fn(module, inp, out):
        nonlocal lstm_macs
        x = inp[0]
        if module.batch_first:
            B, T, _ = x.shape
        else:
            T, B, _ = x.shape

        input_size = module.input_size
        hidden_size = module.hidden_size
        num_layers = module.num_layers
        bidirectional = 2 if module.bidirectional else 1

        flops_per_timestep = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
        lstm_macs += B * T * num_layers * bidirectional * flops_per_timestep  # already 1 MAC per multiply-add

    for m in model.modules():
        if isinstance(m, nn.LSTM):
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    return lstm_macs

# -------------------------------
# PARAMETER COUNT
# -------------------------------

def count_parameters(model):
    """Input: a model. Output: (total_params, trainable_params) as ints."""
    total_params = 0
    trainable_params = 0
    for name, p in model.named_parameters():
        num = p.numel()
        total_params += num
        if p.requires_grad:
            trainable_params += num
    return total_params, trainable_params

# -------------------------------
# MAC COMPUTATION
# -------------------------------

class _FixedRatios(nn.Module):
    """Wraps an adaptive model so FLOPs tracing only sees the input tensor -
    a no-op pass-through for a baseline model (empty `compression_ratios`),
    since those take no second forward argument."""
    def __init__(self, model, compression_ratios):
        super().__init__()
        self.model = model
        self.compression_ratios = compression_ratios

    def forward(self, x):
        if self.compression_ratios:
            return self.model(x, self.compression_ratios)
        return self.model(x)


def compute_macs(model, input_tensor, compression_ratios, name_keywords):
    """Counts total MACs with fvcore plus the custom handlers above, and
    breaks out the 1x1-conv, Linear, and name_keywords-matched (gate)
    subtotals.

    Input: model, a sample input, the compression ratios to trace with
    (empty for a baseline model), and the module-name suffixes to total
    separately. Output: a 7-tuple of (total MACs, 1x1-conv MACs, its % of
    total, Linear MACs, its %, name-keyword MACs, its %). Percentages are
    0 if total is 0."""
    flops = FlopCountAnalysis(_FixedRatios(model, compression_ratios), input_tensor)

    # register handlers
    flops.set_op_handle("aten::add", elementwise_mac_jit)
    flops.set_op_handle("aten::sub", sub_mac_jit)
    flops.set_op_handle("aten::mul", elementwise_mac_jit)
    flops.set_op_handle("aten::conv2d", conv2d_mac_jit)
    flops.set_op_handle("aten::linear", linear_mac_jit)
    flops.set_op_handle("aten::lstm", lstm_mac_jit)
    flops.set_op_handle("aten::max_pool2d", max_pool2d_mac_jit)
    flops.set_op_handle("aten::pow", pow_mac_jit)
    flops.set_op_handle("aten::sqrt", sqrt_mac_jit)
    flops.set_op_handle("aten::div", div_mac_jit)
    flops.set_op_handle("aten::gelu", gelu_mac_jit)
    flops.set_op_handle("aten::add_", add_inplace_mac_jit)
    flops.set_op_handle("aten::relu6", relu6_mac_jit)

    total_macs = flops.total()

    by_module = flops.by_module()

    conv1x1_macs = 0
    fc_macs = 0
    target_macs = 0

    for name, module in model.named_modules():
        module_macs = by_module.get(f"model.{name}" if name else "model", by_module.get(name, 0))
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            conv1x1_macs += module_macs
        if isinstance(module, nn.Linear):
            fc_macs += module_macs
        if any(name.endswith(k) for k in name_keywords):
            print(f"name:{name}, macs:{module_macs}")
            target_macs += module_macs

    return (
        total_macs,
        conv1x1_macs,
        100 * conv1x1_macs / total_macs if total_macs > 0 else 0,
        fc_macs,
        100 * fc_macs / total_macs if total_macs > 0 else 0,
        target_macs,
        100 * target_macs / total_macs if total_macs > 0 else 0,
    )

# -------------------------------
# LOAD MODEL
# -------------------------------

def load_model(model, checkpoint_path, device):
    """Loads `checkpoint['state_dict']` into `model`, moves it to `device`,
    and puts it in eval mode. Output: the same model instance."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":
    args = parse_args()
    model = build_model(args)
    model = load_model(model, args.checkpoint, args.device)
    total_params, trainable_params = count_parameters(model)

    dummy_input = torch.randn(args.batch_size, *args.input_size).to(args.device)
    (
        total_macs,
        conv1x1_macs, conv1x1_percentage,
        fc_macs, fc_percentage,
        target_macs, target_percentage,
    ) = compute_macs(model, dummy_input, args.compression_ratios, args.name_keywords)
    print(args.checkpoint)
    print("========== PARAMETERS ==========")
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    print("\n========== MACs ==========")
    print(f"Total MACs: {total_macs}")
    print(f"Conv1x1 MACs: {conv1x1_macs}")
    print(f"FC MACs: {fc_macs}")
    print(f"Target ({args.name_keywords}) MACs (Per SAMPLE): {target_macs/args.batch_size}")

    print("\n========== PERCENTAGES ==========")
    print(f"Conv1x1 MACs %: {conv1x1_percentage:.2f}%")
    print(f"FC MACs %: {fc_percentage:.2f}%")
    print(f"Target MACs %: {target_percentage:.2f}%")
