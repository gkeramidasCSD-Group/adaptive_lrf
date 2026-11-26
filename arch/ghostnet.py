import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.lrf import *
from .gates import (
    broadcast_mask,
    RNNGate,
    SoftRNNGate,
    SEQ_Gate_V1,
    SEQ_Gate_V2,
    SEQ_Gate_V3,
    SEQ_Gate_V4,
    SEQ_Gate_V5,
    SEQ_Gate_V6,
)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    """Piecewise-linear sigmoid approximation, relu6(x+3)/6, used as the squeeze-excite gate. Input/output: same-shape tensor."""
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class ConvBnAct(nn.Module):
    """Conv, BatchNorm, then activation. Input: (B, in_chs, H, W). Output: (B, out_chs, H', W')."""
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class SqueezeExcite(nn.Module):
    """Squeeze-and-excite channel attention block. Input/output: (B, C, H, W)."""
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()

        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)

        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):

        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)

        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)

        x = x * self.gate_fn(x_se)
        return x


class GhostModule(nn.Module):
    """Ghost module: a primary conv produces half the channels, a cheap depthwise conv produces the rest, and the two are concatenated. Input: (B, inp, H, W). Output: (B, oup, H, W)."""
    def __init__(self, inp, oup,  ghost_id = 0, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        self.ghost1 = GhostModule(in_chs, mid_chs, ghost_id = 0, relu=True)

        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModule(mid_chs, out_chs,  ghost_id = 1, relu=False)

        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        x = self.ghost1(x)

        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        if self.se is not None:
            x = self.se(x)

        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    """Baseline GhostNet for CIFAR-10. Input: (B, 3, H, W). Output: (B, num_classes) logits."""
    def __init__(self, num_classes=10, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()

        self.cfgs = [

        [[3,  16,  16, 0, 1]],

        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],

        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],

        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],

        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
        self.dropout = dropout

        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def cifar10_ghostnet(pretrained=False, **kwargs):
    """Factory for the baseline GhostNet. Output: a model instance."""
    model = GhostNet(num_classes=10)
    return model


# ---------------------------------------------------------------------------
# Adaptive (non-unified)
# ---------------------------------------------------------------------------

class ConvBnAct_adaptive(nn.Module):
    """Adaptive ConvBnAct: its 1x1 conv is a gated low-rank layer (plain variant). Input: (B, in_chs, H, W). Output: (features, masks)."""
    def __init__(self, in_chs, out_chs, kernel_size, compression_ratio_list, min_rank = 1,
                 embed_dim = 10, stride=1, act_layer=nn.ReLU, include_original_weight=True):
        super(ConvBnAct_adaptive, self).__init__()

        self.include_original_weight = include_original_weight
        prev_H_W = 1
        pool_size = prev_H_W
        self.conv = self.make_svd_block(pool_size, in_chs, out_chs, compression_ratio_list, min_rank, embed_dim, "conv")
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def make_svd_block(self, pool_size, in_channels, out_channels, compression_ratio_list, min_rank, embed_dim, prefix, bias = None):
        W = H = pool_size
        layer = Conv1x1_SVD_NonUnified((W, H), in_channels, out_channels, compression_ratio_list, min_rank, bias = bias,
                                        include_original_weight=self.include_original_weight)
        gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=1,
                      stride=1)
            )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios, control_module = None):
        # gate class k <-> choices[k] (see modules.lrf.ordered_gate_choices)
        choices = gate_choices_for(self, compression_ratios)
        masks = []

        prefix = "conv"
        batch_size = x.size(0)
        control_module.hidden_one = control_module.init_hidden(batch_size)
        gate_feature = getattr(self, f"{prefix}_gate")(x)
        mask_one = control_module(gate_feature)

        output_candidates = []
        if self.training:
            for k in range(len(choices)):
                out = getattr(self, f"{prefix}_svd_layer")(x, choices[k])
                output_candidates.append(out)
        mask_list = []
        for j in range(len(choices)):
            mask_list.append(mask_one[:,j])
        if self.training:
            x = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(choices))])
        else:
            index = mask_one.argmax(dim=1)
            x_out = []
            for i in range(x.size(0)):
                idx = index[i].item()
                ratio = choices[idx]
                out_i = getattr(self, f"{prefix}_svd_layer")(x[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x = torch.cat(x_out, dim=0)
        mask_list = [mask.squeeze() for mask in mask_list]
        masks.append(mask_list)

        x = self.bn1(x)
        x = self.act1(x)
        return x, masks


class SqueezeExcite_adaptive(nn.Module):
    """Adaptive squeeze-excite block: both 1x1 convs are gated low-rank layers (plain variant). Input: (B, C, H, W). Output: (features, masks)."""
    def __init__(self, in_chs, compression_ratio_list, min_rank = 1, embed_dim = 10, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, stage_index = 1, block_index = 1, include_original_weight=True, **_):
        super(SqueezeExcite_adaptive, self).__init__()

        self.include_original_weight = include_original_weight
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        prefix = "conv_reduce"
        if stage_index == 1 or (stage_index != 1 and block_index != 1):
            prev_H_W = int(32 / (2 ** stage_index))

        else:
            prev_H_W = int(32 / (2 ** (stage_index-1)))
        pool_size = prev_H_W
        self.conv_reduce = self.make_svd_block(pool_size, in_chs, reduced_chs, compression_ratio_list, min_rank, embed_dim, prefix, bias = True)
        self.act1 = act_layer(inplace=True)

        prefix = "conv_expand"
        self.conv_expand = self.make_svd_block(pool_size, reduced_chs, in_chs, compression_ratio_list, min_rank, embed_dim, prefix, bias = True)

    def make_svd_block(self, pool_size, in_channels, out_channels, compression_ratio_list, min_rank, embed_dim, prefix, bias = None):
        W = H = pool_size
        layer = Conv1x1_SVD_NonUnified((W, H), in_channels, out_channels, compression_ratio_list, min_rank, bias = bias,
                                        include_original_weight=self.include_original_weight)
        gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=1,
                      stride=1)
            )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios, control_module = None):

        # gate class k <-> choices[k] (see modules.lrf.ordered_gate_choices)
        choices = gate_choices_for(self, compression_ratios)
        masks = []
        x_se = self.avg_pool(x)

        prefix = "conv_reduce"
        batch_size = x_se.size(0)
        control_module.hidden_one = control_module.init_hidden(batch_size)
        gate_feature = getattr(self, f"{prefix}_gate")(x_se)
        mask_one = control_module(gate_feature)

        output_candidates = []
        if self.training:
            for k in range(len(choices)):
                out = getattr(self, f"{prefix}_svd_layer")(x_se, choices[k])
                output_candidates.append(out)
        mask_list = []
        for j in range(len(choices)):
            mask_list.append(mask_one[:,j])
        if self.training:
            x_se = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(choices))])
        else:
            index = mask_one.argmax(dim=1)
            x_out = []
            for i in range(x_se.size(0)):
                idx = index[i].item()
                ratio = choices[idx]
                out_i = getattr(self, f"{prefix}_svd_layer")(x_se[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x_se = torch.cat(x_out, dim=0)
        mask_list = [mask.squeeze() for mask in mask_list]
        masks.append(mask_list)

        x_se = self.act1(x_se)

        prefix = "conv_expand"
        batch_size = x_se.size(0)
        control_module.hidden_one = control_module.init_hidden(batch_size)
        gate_feature = getattr(self, f"{prefix}_gate")(x_se)
        mask_one = control_module(gate_feature)

        output_candidates_2 = []
        if self.training:
            for k in range(len(choices)):
                out = getattr(self, f"{prefix}_svd_layer")(x_se, choices[k])
                output_candidates_2.append(out)
        mask_list_2 = []
        for j in range(len(choices)):
            mask_list_2.append(mask_one[:,j])
        if self.training:
            x_se = sum([broadcast_mask(mask_list_2[k], output_candidates_2[0]) * output_candidates_2[k] for k in range(len(choices))])
        else:
            index = mask_one.argmax(dim=1)
            x_out = []
            for i in range(x_se.size(0)):
                idx = index[i].item()
                ratio = choices[idx]
                out_i = getattr(self, f"{prefix}_svd_layer")(x_se[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x_se = torch.cat(x_out, dim=0)
        mask_list_2 = [mask.squeeze() for mask in mask_list_2]
        masks.append(mask_list_2)

        x = x * self.gate_fn(x_se)
        return x, masks


class GhostModule_adaptive(nn.Module):
    """Adaptive ghost module: its primary 1x1 conv is a gated low-rank layer (plain variant). Input: (B, inp, H, W). Output: (features, masks)."""
    def __init__(self, inp, oup, compression_ratio_list, ghost_id = 0, min_rank = 1, embed_dim = 10,
    kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, stage_index = 1, block_index = 1, include_original_weight=True):
        super(GhostModule_adaptive, self).__init__()
        self.include_original_weight = include_original_weight
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)
        aux_conv = None
        self.adaptive = False
        if kernel_size == 1:
            prefix = "primary_conv_0"
            if stage_index == 1:
                prev_H_W = int(32 / (2 ** stage_index))
            elif block_index == 1 and ghost_id == 0:
                prev_H_W = int(32 / (2 ** stage_index-1))
            else:
                prev_H_W = int(32 / (2 ** (stage_index)))
            pool_size = prev_H_W

            self.make_svd_block(pool_size, inp, init_channels, compression_ratio_list, min_rank, embed_dim, prefix)
            self.adaptive = True
            aux_conv = nn.Identity()
        else:
            aux_conv = nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False)

        self.primary_conv = nn.Sequential(
            aux_conv,
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def make_svd_block(self, pool_size, in_channels, out_channels, compression_ratio_list, min_rank, embed_dim, prefix, bias = None):
        W = H = pool_size
        layer = Conv1x1_SVD_NonUnified((W, H), in_channels, out_channels, compression_ratio_list, min_rank, bias = bias,
                                        include_original_weight=self.include_original_weight)
        gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=1,
                      stride=1)
            )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios, control_module = None):
        # gate class k <-> choices[k] (see modules.lrf.ordered_gate_choices)
        choices = gate_choices_for(self, compression_ratios)
        masks = []
        if self.adaptive is True:
            prefix = "primary_conv_0"
            batch_size = x.size(0)
            control_module.hidden_one = control_module.init_hidden(batch_size)
            gate_feature = getattr(self, f"{prefix}_gate")(x)
            mask_one = control_module(gate_feature)

            output_candidates = []
            if self.training:
                for k in range(len(choices)):
                    out = getattr(self, f"{prefix}_svd_layer")(x, choices[k])
                    output_candidates.append(out)
            mask_list = []
            for j in range(len(choices)):
                mask_list.append(mask_one[:,j])
            if self.training:
                x = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(choices))])
            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x.size(0)):
                    idx = index[i].item()
                    ratio = choices[idx]
                    out_i = getattr(self, f"{prefix}_svd_layer")(x[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x = torch.cat(x_out, dim=0)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)
        else:
            prefix = "primary_conv.0"
            x = getattr(self, prefix)(x)

        x1 = self.primary_conv[1](x)
        x1 = self.primary_conv[2](x1)

        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)

        if self.adaptive is True:
            return out[:,:self.oup,:,:], masks
        else:
            return out[:,:self.oup,:,:]


class GhostBottleneck_adaptive(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, compression_ratio_list, dw_kernel_size=3,
                 stride=1, min_rank = 1, embed_dim = 10, act_layer=nn.ReLU,
                 se_ratio=0., stage_index = 1, block_index=1, include_original_weight=True):
        super(GhostBottleneck_adaptive, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        self.adaptive_shortcut = False
        self.include_original_weight = include_original_weight

        self.ghost1 = GhostModule_adaptive(in_chs, mid_chs, ghost_id = 0, compression_ratio_list = compression_ratio_list,
        min_rank = min_rank, embed_dim = embed_dim, relu=True, stage_index = stage_index, block_index = block_index,
        include_original_weight = include_original_weight)

        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        if has_se:
            self.se = SqueezeExcite_adaptive(mid_chs, compression_ratio_list = compression_ratio_list, min_rank = min_rank,
            embed_dim = embed_dim, se_ratio=se_ratio, stage_index = stage_index, block_index = block_index,
            include_original_weight = include_original_weight)
        else:
            self.se = None

        self.ghost2 = GhostModule_adaptive(mid_chs, out_chs,  ghost_id = 1, compression_ratio_list = compression_ratio_list,
        min_rank = min_rank, embed_dim = embed_dim, relu=False, stage_index = stage_index, block_index = block_index,
        include_original_weight = include_original_weight)

        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.adaptive_shortcut = True

            prev_H_W = int(32 / (2 ** stage_index))

            pool_size = prev_H_W
            self.make_svd_block(pool_size, in_chs, out_chs, compression_ratio_list, min_rank, embed_dim, "shortcut_2")
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Identity(),

                nn.BatchNorm2d(out_chs),
            )

    def make_svd_block(self, pool_size, in_channels, out_channels, compression_ratio_list, min_rank, embed_dim, prefix, bias = None):
        W = H = pool_size
        layer = Conv1x1_SVD_NonUnified((W, H), in_channels, out_channels, compression_ratio_list, min_rank, bias = bias,
                                        include_original_weight=self.include_original_weight)
        gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=1,
                      stride=1)
            )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios, control=None):
        # gate class k <-> choices[k] (see modules.lrf.ordered_gate_choices)
        choices = gate_choices_for(self, compression_ratios)
        masks = []
        residual = x

        x, gmasks1 = self.ghost1(x, compression_ratios, control)
        masks.extend(gmasks1)

        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        if self.se is not None:
            x, se_masks = self.se(x, compression_ratios, control)
            masks.extend(se_masks)

        x, gmasks2 = self.ghost2(x, compression_ratios, control)
        masks.extend(gmasks2)

        if self.adaptive_shortcut is True:
            x_res = self.shortcut[0](residual)
            x_res = self.shortcut[1](x_res)

            prefix = "shortcut_2"
            batch_size = x_res.size(0)
            control.hidden_one = control.init_hidden(batch_size)
            gate_feature = getattr(self, f"{prefix}_gate")(x_res)
            mask_one = control(gate_feature)

            output_candidates = []
            if self.training:
                for k in range(len(choices)):
                    out = getattr(self, f"{prefix}_svd_layer")(x_res, choices[k])
                    output_candidates.append(out)
            mask_list = []
            for j in range(len(choices)):
                mask_list.append(mask_one[:,j])
            if self.training:
                x_res = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(choices))])
            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x_res.size(0)):
                    idx = index[i].item()
                    ratio = choices[idx]
                    out_i = getattr(self, f"{prefix}_svd_layer")(x_res[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x_res = torch.cat(x_out, dim=0)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)

            x_res = self.shortcut[3](x_res)
            x += x_res
        else:
            x += self.shortcut(residual)
        return x, masks


class GhostNet_adaptive(nn.Module):
    """Adaptive GhostNet with gated low-rank layers (plain variant). Constructor: compression_ratio_list, min_rank, gate_type ('rnn'/'soft'/'seq_v1'..'seq_v6'), include_original_weight. Input: (B, 3, H, W). Output: (logits, masks)."""
    is_unified = False  # see configs.config_utils.is_plain_arch

    def __init__(self, compression_ratio_list, min_rank = 1, num_classes=10, embed_dim=10, hidden_dim=10, gate_type='rnn', gate_num_classes=2, width=1.0, dropout=0.2, include_original_weight=True):
        super(GhostNet_adaptive, self).__init__()

        self.gate_type = gate_type
        self.include_original_weight = include_original_weight
        self.num_adaptive_layers = 53
        self.cfgs = [

        [[3,  16,  16, 0, 1, 1, 1]],

        [[3,  48,  24, 0, 2, 2, 1]],
        [[3,  72,  24, 0, 1, 2, 2]],

        [[5,  72,  40, 0.25, 2, 3, 1]],
        [[5, 120,  40, 0.25, 1, 3, 2]],

        [[3, 240,  80, 0, 2, 4, 1]],
        [[3, 200,  80, 0, 1, 4, 2],
         [3, 184,  80, 0, 1, 4, 3],
         [3, 184,  80, 0, 1, 4, 4],
         [3, 480, 112, 0.25, 1, 4, 5],
         [3, 672, 112, 0.25, 1, 4, 6]
        ],

        [[5, 672, 160, 0.25, 2, 5, 1]],
        [[5, 960, 160, 0, 1, 5, 2],
         [5, 960, 160, 0.25, 1, 5, 3],
         [5, 960, 160, 0, 1, 5, 4],
         [5, 960, 160, 0.25, 1, 5, 5]
        ]
        ]

        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm', gate_num_classes=gate_num_classes)
        elif gate_type == 'soft':
            self.control = SoftRNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        elif gate_type == 'seq_v1':
            self.control = SEQ_Gate_V1(embed_dim, gate_num_classes)
        elif gate_type == 'seq_v2':
            self.control = SEQ_Gate_V2(embed_dim, gate_num_classes)
        elif gate_type == 'seq_v3':
            self.control = SEQ_Gate_V3(embed_dim, gate_num_classes)
        elif gate_type == 'seq_v4':
            self.control = SEQ_Gate_V4(embed_dim, gate_num_classes)
        elif gate_type == 'seq_v5':
            self.control = SEQ_Gate_V5(embed_dim, gate_num_classes)
        elif gate_type == 'seq_v6':
            self.control = SEQ_Gate_V6(embed_dim, gate_num_classes)
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None

        self.dropout = dropout

        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        stages = []
        block = GhostBottleneck_adaptive
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s, stage_index, block_index in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, compression_ratio_list, dw_kernel_size=k, stride=s,
                              min_rank=min_rank, embed_dim = embed_dim,
                              se_ratio=se_ratio, stage_index = stage_index, block_index=block_index,
                              include_original_weight = include_original_weight))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct_adaptive(input_channel, output_channel, 1, embed_dim = embed_dim, compression_ratio_list = compression_ratio_list, min_rank = min_rank,
                                                         include_original_weight = include_original_weight)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.make_svd_block(1, input_channel, output_channel, compression_ratio_list, min_rank, embed_dim, "conv_head", bias = True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def make_svd_block(self, pool_size, in_channels, out_channels, compression_ratio_list, min_rank, embed_dim, prefix, bias = None):
        W = H = pool_size
        layer = Conv1x1_SVD_NonUnified((W, H), in_channels, out_channels, compression_ratio_list, min_rank, bias = bias,
                                        include_original_weight=self.include_original_weight)
        gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=1,
                      stride=1)
            )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios):
        # gate class k <-> choices[k] (see modules.lrf.ordered_gate_choices)
        choices = gate_choices_for(self, compression_ratios)
        masks = []
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        for block in self.blocks:
            for l in block:
                x, layer_masks = l(x, compression_ratios, self.control)
                masks.extend(layer_masks)

        x = self.global_pool(x)

        prefix = "conv_head"
        batch_size = x.size(0)
        if self.gate_type == 'rnn':
            self.control.hidden_one = self.control.init_hidden(batch_size)
        gate_feature = getattr(self, f"{prefix}_gate")(x)
        mask_one = self.control(gate_feature)

        output_candidates = []
        if self.training:
            for k in range(len(choices)):
                out = getattr(self, f"{prefix}_svd_layer")(x, choices[k])
                output_candidates.append(out)
        mask_list = []
        for j in range(len(choices)):
            mask_list.append(mask_one[:,j])
        if self.training:
            x = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(choices))])
        else:
            index = mask_one.argmax(dim=1)
            x_out = []
            for i in range(x.size(0)):
                idx = index[i].item()
                ratio = choices[idx]
                out_i = getattr(self, f"{prefix}_svd_layer")(x[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x = torch.cat(x_out, dim=0)
        mask_list = [mask.squeeze() for mask in mask_list]
        masks.append(mask_list)

        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x, masks


def cifar10_ghostnet_adaptive(compression_ratios, pretrained=False, min_rank=1, **kwargs):
    """Factory for GhostNet_adaptive, CIFAR-10 settings. Inputs: compression_ratios (list), min_rank, kwargs. Output: a model instance."""
    model = GhostNet_adaptive(compression_ratios, min_rank, num_classes=10, **kwargs)
    return model
cifar10_ghostnet_adaptive.is_unified = GhostNet_adaptive.is_unified


# ---------------------------------------------------------------------------
# Adaptive (unified)
# ---------------------------------------------------------------------------

class ConvBnAct_adaptive_unified(nn.Module):
    """Unified-variant adaptive ConvBnAct (shared basis across choices). Input: (B, in_chs, H, W). Output: (features, masks)."""
    def __init__(self, in_chs, out_chs, kernel_size, compression_ratio_list, min_rank = 1,
                 embed_dim = 10, stride=1, act_layer=nn.ReLU):
        super(ConvBnAct_adaptive_unified, self).__init__()

        prev_H_W = 1
        pool_size = prev_H_W
        self.conv = self.make_svd_block(pool_size, in_chs, out_chs, compression_ratio_list, min_rank, embed_dim, "conv")
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def make_svd_block(self, pool_size, in_channels, out_channels, compression_ratio_list, min_rank, embed_dim, prefix, bias = None):
        W = H = pool_size
        layer = Conv1x1_SVD_Unified((W, H), in_channels, out_channels, compression_ratio_list, min_rank, bias = bias)
        gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=1,
                      stride=1)
            )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios, control_module = None):
        # gate class k <-> choices[k] (see modules.lrf.ordered_gate_choices)
        choices = gate_choices_for(self, compression_ratios)
        masks = []

        prefix = "conv"
        batch_size = x.size(0)
        control_module.hidden_one = control_module.init_hidden(batch_size)
        gate_feature = getattr(self, f"{prefix}_gate")(x)
        mask_one = control_module(gate_feature)

        output_candidates = []
        if self.training:
            for k in range(len(choices)):
                out = getattr(self, f"{prefix}_svd_layer")(x, choices[k])
                output_candidates.append(out)

        mask_list = []
        for j in range(len(choices)):
            mask_list.append(mask_one[:,j])
        if self.training:
            x = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(choices))])
        else:
            index = mask_one.argmax(dim=1)
            x_out = []
            for i in range(x.size(0)):
                idx = index[i].item()
                ratio = choices[idx]
                out_i = getattr(self, f"{prefix}_svd_layer")(x[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x = torch.cat(x_out, dim=0)
        mask_list = [mask.squeeze() for mask in mask_list]
        masks.append(mask_list)

        x = self.bn1(x)
        x = self.act1(x)
        return x, masks


class SqueezeExcite_adaptive_unified(nn.Module):
    """Unified-variant adaptive squeeze-excite block (shared basis across choices). Input: (B, C, H, W). Output: (features, masks)."""
    def __init__(self, in_chs, compression_ratio_list, min_rank = 1, embed_dim = 10, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, stage_index = 1, block_index = 1,  **_):
        super(SqueezeExcite_adaptive_unified, self).__init__()

        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        prefix = "conv_reduce"
        if stage_index == 1 or (stage_index != 1 and block_index != 1):
            prev_H_W = int(32 / (2 ** stage_index))

        else:
            prev_H_W = int(32 / (2 ** (stage_index-1)))
        pool_size = prev_H_W
        self.conv_reduce = self.make_svd_block(pool_size, in_chs, reduced_chs, compression_ratio_list, min_rank, embed_dim, prefix, bias = True)
        self.act1 = act_layer(inplace=True)

        prefix = "conv_expand"
        self.conv_expand = self.make_svd_block(pool_size, reduced_chs, in_chs, compression_ratio_list, min_rank, embed_dim, prefix, bias = True)

    def make_svd_block(self, pool_size, in_channels, out_channels, compression_ratio_list, min_rank, embed_dim, prefix, bias = None):
        W = H = pool_size
        layer = Conv1x1_SVD_Unified((W, H), in_channels, out_channels, compression_ratio_list, min_rank, bias = bias)
        gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=1,
                      stride=1)
            )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios, control_module = None):

        # gate class k <-> choices[k] (see modules.lrf.ordered_gate_choices)
        choices = gate_choices_for(self, compression_ratios)
        masks = []
        x_se = self.avg_pool(x)

        prefix = "conv_reduce"
        batch_size = x_se.size(0)
        control_module.hidden_one = control_module.init_hidden(batch_size)
        gate_feature = getattr(self, f"{prefix}_gate")(x_se)
        mask_one = control_module(gate_feature)

        output_candidates = []
        if self.training:
            for k in range(len(choices)):
                out = getattr(self, f"{prefix}_svd_layer")(x_se, choices[k])
                output_candidates.append(out)

        mask_list = []
        for j in range(len(choices)):
            mask_list.append(mask_one[:,j])
        if self.training:
            x_se = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(choices))])
        else:
            index = mask_one.argmax(dim=1)
            x_out = []
            for i in range(x_se.size(0)):
                idx = index[i].item()

                ratio = choices[idx]
                out_i = getattr(self, f"{prefix}_svd_layer")(x_se[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x_se = torch.cat(x_out, dim=0)
        mask_list = [mask.squeeze() for mask in mask_list]
        masks.append(mask_list)

        x_se = self.act1(x_se)

        prefix = "conv_expand"
        batch_size = x_se.size(0)
        control_module.hidden_one = control_module.init_hidden(batch_size)
        gate_feature = getattr(self, f"{prefix}_gate")(x_se)
        mask_one = control_module(gate_feature)

        output_candidates_2 = []
        if self.training:
            for k in range(len(choices)):
                out = getattr(self, f"{prefix}_svd_layer")(x_se, choices[k])
                output_candidates_2.append(out)

        mask_list_2 = []
        for j in range(len(choices)):
            mask_list_2.append(mask_one[:,j])
        if self.training:
            x_se = sum([broadcast_mask(mask_list_2[k], output_candidates_2[0]) * output_candidates_2[k] for k in range(len(choices))])
        else:
            index = mask_one.argmax(dim=1)
            x_out = []
            for i in range(x_se.size(0)):
                idx = index[i].item()

                ratio = choices[idx]
                out_i = getattr(self, f"{prefix}_svd_layer")(x_se[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x_se = torch.cat(x_out, dim=0)
        mask_list_2 = [mask.squeeze() for mask in mask_list_2]
        masks.append(mask_list_2)

        x = x * self.gate_fn(x_se)
        return x, masks


class GhostModule_adaptive_unified(nn.Module):
    """Unified-variant adaptive ghost module (shared basis across choices). Input: (B, inp, H, W). Output: (features, masks)."""
    def __init__(self, inp, oup, compression_ratio_list, ghost_id = 0, min_rank = 1, embed_dim = 10,
    kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, stage_index = 1, block_index = 1):
        super(GhostModule_adaptive_unified, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)
        aux_conv = None
        self.adaptive = False
        if kernel_size == 1:
            prefix = "primary_conv_0"
            if stage_index == 1:
                prev_H_W = int(32 / (2 ** stage_index))
            elif block_index == 1 and ghost_id == 0:
                prev_H_W = int(32 / (2 ** stage_index-1))
            else:
                prev_H_W = int(32 / (2 ** (stage_index)))
            pool_size = prev_H_W

            self.make_svd_block(pool_size, inp, init_channels, compression_ratio_list, min_rank, embed_dim, prefix)
            self.adaptive = True
            aux_conv = nn.Identity()
        else:
            aux_conv = nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False)

        self.primary_conv = nn.Sequential(
            aux_conv,
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def make_svd_block(self, pool_size, in_channels, out_channels, compression_ratio_list, min_rank, embed_dim, prefix, bias = None):
        W = H = pool_size
        layer = Conv1x1_SVD_Unified((W, H), in_channels, out_channels, compression_ratio_list, min_rank, bias = bias)
        gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=1,
                      stride=1)
            )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios, control_module = None):
        # gate class k <-> choices[k] (see modules.lrf.ordered_gate_choices)
        choices = gate_choices_for(self, compression_ratios)
        masks = []
        if self.adaptive is True:
            prefix = "primary_conv_0"
            batch_size = x.size(0)
            control_module.hidden_one = control_module.init_hidden(batch_size)
            gate_feature = getattr(self, f"{prefix}_gate")(x)
            mask_one = control_module(gate_feature)

            output_candidates = []
            if self.training:
                for k in range(len(choices)):
                    out = getattr(self, f"{prefix}_svd_layer")(x, choices[k])
                    output_candidates.append(out)

            mask_list = []
            for j in range(len(choices)):
                mask_list.append(mask_one[:,j])
            if self.training:
                x = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(choices))])
            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x.size(0)):
                    idx = index[i].item()

                    ratio = choices[idx]
                    out_i = getattr(self, f"{prefix}_svd_layer")(x[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x = torch.cat(x_out, dim=0)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)
        else:
            prefix = "primary_conv.0"
            x = getattr(self, prefix)(x)

        x1 = self.primary_conv[1](x)
        x1 = self.primary_conv[2](x1)

        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)

        if self.adaptive is True:
            return out[:,:self.oup,:,:], masks
        else:
            return out[:,:self.oup,:,:]


class GhostBottleneck_adaptive_unified(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, compression_ratio_list, dw_kernel_size=3,
                 stride=1, min_rank = 1, embed_dim = 10, act_layer=nn.ReLU,
                 se_ratio=0., stage_index = 1, block_index=1):
        super(GhostBottleneck_adaptive_unified, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        self.adaptive_shortcut = False

        self.ghost1 = GhostModule_adaptive_unified(in_chs, mid_chs, ghost_id = 0, compression_ratio_list = compression_ratio_list,
        min_rank = min_rank, embed_dim = embed_dim, relu=True, stage_index = stage_index, block_index = block_index)

        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        if has_se:
            self.se = SqueezeExcite_adaptive_unified(mid_chs, compression_ratio_list = compression_ratio_list, min_rank = min_rank,
            embed_dim = embed_dim, se_ratio=se_ratio, stage_index = stage_index, block_index = block_index)
        else:
            self.se = None

        self.ghost2 = GhostModule_adaptive_unified(mid_chs, out_chs,  ghost_id = 1, compression_ratio_list = compression_ratio_list,
        min_rank = min_rank, relu=False, stage_index = stage_index, block_index = block_index)

        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.adaptive_shortcut = True

            prev_H_W = int(32 / (2 ** stage_index))

            pool_size = prev_H_W
            self.make_svd_block(pool_size, in_chs, out_chs, compression_ratio_list, min_rank, embed_dim, "shortcut_2")
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Identity(),

                nn.BatchNorm2d(out_chs),
            )

    def make_svd_block(self, pool_size, in_channels, out_channels, compression_ratio_list, min_rank, embed_dim, prefix, bias = None):
        W = H = pool_size
        layer = Conv1x1_SVD_Unified((W, H), in_channels, out_channels, compression_ratio_list, min_rank, bias = bias)
        gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=1,
                      stride=1)
            )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios, control=None):
        # gate class k <-> choices[k] (see modules.lrf.ordered_gate_choices)
        choices = gate_choices_for(self, compression_ratios)
        masks = []
        residual = x

        x, gmasks1 = self.ghost1(x, compression_ratios, control)
        masks.extend(gmasks1)

        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        if self.se is not None:
            x, se_masks = self.se(x, compression_ratios, control)
            masks.extend(se_masks)

        x, gmasks2 = self.ghost2(x, compression_ratios, control)
        masks.extend(gmasks2)

        if self.adaptive_shortcut is True:
            x_res = self.shortcut[0](residual)
            x_res = self.shortcut[1](x_res)

            prefix = "shortcut_2"
            batch_size = x_res.size(0)
            control.hidden_one = control.init_hidden(batch_size)
            gate_feature = getattr(self, f"{prefix}_gate")(x_res)
            mask_one = control(gate_feature)

            output_candidates = []
            if self.training:
                for k in range(len(choices)):
                    out = getattr(self, f"{prefix}_svd_layer")(x_res, choices[k])
                    output_candidates.append(out)

            mask_list = []
            for j in range(len(choices)):
                mask_list.append(mask_one[:,j])
            if self.training:
                x_res = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(choices))])
            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x_res.size(0)):
                    idx = index[i].item()

                    ratio = choices[idx]
                    out_i = getattr(self, f"{prefix}_svd_layer")(x_res[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x_res = torch.cat(x_out, dim=0)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)

            x_res = self.shortcut[3](x_res)
            x += x_res
        else:
            x += self.shortcut(residual)
        return x, masks


class GhostNet_adaptive_unified(nn.Module):
    """Unified-variant adaptive GhostNet: every low-rank layer shares one basis across choices. Constructor: compression_ratio_list, min_rank, gate_type ('rnn' or 'soft'). Input: (B, 3, H, W). Output: (logits, masks)."""
    is_unified = True  # see configs.config_utils.is_plain_arch

    def __init__(self, compression_ratio_list, min_rank = 1, num_classes=10, embed_dim=10, hidden_dim=10, gate_type='rnn', gate_num_classes=2, width=1.0, dropout=0.2):
        super(GhostNet_adaptive_unified, self).__init__()

        self.num_adaptive_layers = 53
        self.cfgs = [

        [[3,  16,  16, 0, 1, 1, 1]],

        [[3,  48,  24, 0, 2, 2, 1]],
        [[3,  72,  24, 0, 1, 2, 2]],

        [[5,  72,  40, 0.25, 2, 3, 1]],
        [[5, 120,  40, 0.25, 1, 3, 2]],

        [[3, 240,  80, 0, 2, 4, 1]],
        [[3, 200,  80, 0, 1, 4, 2],
         [3, 184,  80, 0, 1, 4, 3],
         [3, 184,  80, 0, 1, 4, 4],
         [3, 480, 112, 0.25, 1, 4, 5],
         [3, 672, 112, 0.25, 1, 4, 6]
        ],

        [[5, 672, 160, 0.25, 2, 5, 1]],
        [[5, 960, 160, 0, 1, 5, 2],
         [5, 960, 160, 0.25, 1, 5, 3],
         [5, 960, 160, 0, 1, 5, 4],
         [5, 960, 160, 0.25, 1, 5, 5]
        ]
        ]

        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm', gate_num_classes=gate_num_classes)
        elif gate_type == 'soft':
            self.control = SoftRNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None

        self.dropout = dropout

        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        stages = []
        block = GhostBottleneck_adaptive_unified
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s, stage_index, block_index in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, compression_ratio_list, dw_kernel_size=k, stride=s,
                              min_rank=min_rank, embed_dim = embed_dim,
                              se_ratio=se_ratio, stage_index = stage_index, block_index=block_index))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct_adaptive_unified(input_channel, output_channel, 1, embed_dim = embed_dim, compression_ratio_list = compression_ratio_list, min_rank = min_rank)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.make_svd_block(1, input_channel, output_channel, compression_ratio_list, min_rank, embed_dim, "conv_head", bias = True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def make_svd_block(self, pool_size, in_channels, out_channels, compression_ratio_list, min_rank, embed_dim, prefix, bias = None):
        W = H = pool_size
        layer = Conv1x1_SVD_Unified((W, H), in_channels, out_channels, compression_ratio_list, min_rank, bias = bias)
        gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=1,
                      stride=1)
            )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios):
        # gate class k <-> choices[k] (see modules.lrf.ordered_gate_choices)
        choices = gate_choices_for(self, compression_ratios)
        masks = []
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        for block in self.blocks:
            for l in block:
                x, layer_masks = l(x, compression_ratios, self.control)
                masks.extend(layer_masks)

        x = self.global_pool(x)

        prefix = "conv_head"
        batch_size = x.size(0)
        self.control.hidden_one = self.control.init_hidden(batch_size)
        gate_feature = getattr(self, f"{prefix}_gate")(x)
        mask_one = self.control(gate_feature)

        output_candidates = []
        if self.training:
            for k in range(len(choices)):
                out = getattr(self, f"{prefix}_svd_layer")(x, choices[k])
                output_candidates.append(out)

        mask_list = []
        for j in range(len(choices)):
            mask_list.append(mask_one[:,j])
        if self.training:
            x = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(choices))])
        else:
            index = mask_one.argmax(dim=1)
            x_out = []
            for i in range(x.size(0)):
                idx = index[i].item()

                ratio = choices[idx]
                out_i = getattr(self, f"{prefix}_svd_layer")(x[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x = torch.cat(x_out, dim=0)
        mask_list = [mask.squeeze() for mask in mask_list]
        masks.append(mask_list)

        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x, masks


def cifar10_ghostnet_adaptive_unified(compression_ratios, pretrained=False, min_rank=1, **kwargs):
    """Factory for GhostNet_adaptive_unified, CIFAR-10 settings. Inputs: compression_ratios (list), min_rank, kwargs. Output: a model instance."""
    model = GhostNet_adaptive_unified(compression_ratios, min_rank, num_classes=10, **kwargs)
    return model
cifar10_ghostnet_adaptive_unified.is_unified = GhostNet_adaptive_unified.is_unified
