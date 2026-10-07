"""ConvNeXt (CIFAR-10): the plain backbone (cifar10_convnext, trained by
train_base.py) and its adaptive variants
(cifar10_convnext_adaptive*, trained by
adaptive_training.py, which warm-starts from the backbone's
checkpoint)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.lrf import *
from .common import DropPath
from .gates import broadcast_mask, RNNGate, SoftRNNGate, SEQ_Gate_V1, SEQ_Gate_V2


class LayerNorm_V1(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = normalized_shape

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    """Baseline ConvNeXt residual block: depthwise conv, LayerNorm, two pointwise (1x1) linears with GELU, layer scale, DropPath. Input/output: (B, C, H, W)."""
    def __init__(self, dim, stage_id, block_id, drop_path_rate=0.0, layer_scale_init_value=1e-6):
        super(ConvNeXtBlock, self).__init__()

        self.stage_id =stage_id
        self.block_id = block_id

        prefix = f"stage{stage_id}_block{block_id}"


        setattr(self, f"{prefix}_depthwise", nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim))


        self.norm = LayerNorm_V1(dim, eps=1e-6, data_format="channels_first")


        setattr(self, f"{prefix}_pw1", nn.Conv2d(dim, 4 * dim, kernel_size=1))
        setattr(self, f"{prefix}_pw2", nn.Conv2d(4 * dim, dim, kernel_size=1))


        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
        else:
            self.layer_scale = None
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        residual = x
        prefix = f"stage{self.stage_id}_block{self.block_id}"

        x = getattr(self, f"{prefix}_depthwise")(x)

        x = self.norm(x)
        x = getattr(self, f"{prefix}_pw1")(x)
        x = F.gelu(x)
        x = getattr(self, f"{prefix}_pw2")(x)

        if self.layer_scale is not None:
            x = x * self.layer_scale.view(1, -1, 1, 1)


        x = self.drop_path(x)
        return residual + x


class ConvNeXt(nn.Module):
    """Baseline ConvNeXt for CIFAR-10. Input: (B, 3, H, W). Output: (B, num_classes) logits."""
    def __init__(self, num_classes=10, input_channels=3):
        super(ConvNeXt, self).__init__()

        self.dims = [96, 192, 384, 768]
        self.depths = [3, 3, 9, 3]
        self.drop_path_rate = 0.0
        self.layer_scale_init_value = 1e-6


        self.stem_conv = nn.Conv2d(input_channels, self.dims[0], kernel_size=4, stride=4)
        self.stem_norm = LayerNorm_V1([96, 8, 8], eps=1e-6)


        self.stages = nn.ModuleList()
        for stage_idx, (dim, depth) in enumerate(zip(self.dims, self.depths)):
            blocks = []
            if stage_idx > 0:
                prev_dim = self.dims[stage_idx - 1]
                prev_H = 32 // (2 ** (stage_idx+1))
                prev_W = 32 // (2 ** (stage_idx+1))
                downsample_norm = LayerNorm_V1([prev_dim, prev_H, prev_W])
                downsample_conv = nn.Conv2d(self.dims[stage_idx-1], self.dims[stage_idx], kernel_size=2, stride=2)

                setattr(self, f"stage{stage_idx}_downsample_norm", downsample_norm)
                setattr(self, f"stage{stage_idx}_downsample_conv", downsample_conv)

                blocks.append(nn.Sequential(downsample_norm, downsample_conv))

            for block_idx in range(depth):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dim,
                        stage_id=stage_idx + 1,
                        block_id=block_idx + 1,
                        drop_path_rate=self.drop_path_rate,
                        layer_scale_init_value=self.layer_scale_init_value
                    )
                )


            self.stages.append(nn.Sequential(*blocks))


        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head_norm =nn.LayerNorm(self.dims[-1], eps=1e-6)
        self.classifier = nn.Linear(self.dims[-1], num_classes)
    def promote_attrs(self, block_module):
        for name, module in block_module.named_children():
            setattr(self, name, module)
    def forward(self, x):

        x = self.stem_conv(x)
        x = self.stem_norm(x)


        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)






        x = self.global_avg_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.head_norm(x)
        x = self.classifier(x)
        return x


def cifar10_convnext(pretrained=False, **kwargs):
    """Factory for the baseline ConvNeXt. Output: a model instance."""
    model = ConvNeXt(num_classes=10)
    return model


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock_Adaptive(nn.Module):
    """ConvNeXt block whose two pointwise (1x1) convs are low-rank and gated per sample (plain variant). Input/output: (B, C, H, W) plus gate masks."""
    def __init__(self, dim, stage_id, block_id, compression_ratio_list, adaptive=False, min_rank = [], embed_dim = 10, drop_path_rate=0.0, layer_scale_init_value=1e-6, gate_type = 'rnn', include_original_weight=True):
        super(ConvNeXtBlock_Adaptive, self).__init__()
        self.embed_dim = embed_dim
        self.stage_id =stage_id
        self.block_id = block_id
        self.adaptive = adaptive
        self.gate_type = gate_type
        self.include_original_weight = include_original_weight

        prefix = f"stage{stage_id}_block{block_id}"


        setattr(self, f"{prefix}_depthwise", nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim))


        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")


        if self.adaptive is True:
            prev_H_W = 32 // (2 ** (stage_id+1))
            pool_size = prev_H_W
            self.make_svd_block(pool_size, dim, 4 * dim, compression_ratio_list, min_rank, f"{prefix}_pw1", bias = True)
            self.make_svd_block(pool_size, 4 * dim, dim, compression_ratio_list, min_rank, f"{prefix}_pw2", bias = True)
        else:
            setattr(self, f"{prefix}_pw1", nn.Conv2d(dim, 4 * dim, kernel_size=1))
            setattr(self, f"{prefix}_pw2", nn.Conv2d(4 * dim, dim, kernel_size=1))


        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
        else:
            self.layer_scale = None

        self.drop_path = DropPath(drop_path_rate)
    def make_svd_block(self, pool_size, in_channels, out_channels, compression_ratio_list, min_rank, prefix, bias = None):
        W = H = pool_size
        layer = Conv1x1_SVD_NonUnified((W, H), in_channels, out_channels, compression_ratio_list, min_rank, bias = bias,
                                        include_original_weight=self.include_original_weight)
        gate_layer = nn.Sequential(
            nn.AvgPool2d(pool_size),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.embed_dim,
                      kernel_size=1,
                      stride=1)
            )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios, control_module = None):
        masks = []
        residual = x
        prefix = f"stage{self.stage_id}_block{self.block_id}"
        num_classes = len(compression_ratios) + (1 if self.include_original_weight else 0)


        x = getattr(self, f"{prefix}_depthwise")(x)
        x = self.norm(x)


        if self.adaptive is True:
            batch_size = x.size(0)
            if self.gate_type == 'rnn':
                control_module.hidden_one = control_module.init_hidden(batch_size)
            gate_feature = getattr(self, f"{prefix}_pw1_gate")(x)
            mask_one = control_module(gate_feature)
            output_candidates = []
            if self.training:

                for k in range(len(compression_ratios)):
                    out = getattr(self, f"{prefix}_pw1_svd_layer")(x, compression_ratios[k])
                    output_candidates.append(out)
                if self.include_original_weight:
                    output_candidates.append(getattr(self, f"{prefix}_pw1_svd_layer")(x, 0))
            mask_list = []
            for j in range(num_classes):

                mask_list.append(mask_one[:,j])
            if self.training:


                x = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(num_classes)])
            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x.size(0)):
                    idx = index[i].item()


                    if self.include_original_weight and idx == len(compression_ratios):
                        ratio = 0
                    else:
                        ratio = compression_ratios[idx]
                    out_i = getattr(self, f"{prefix}_pw1_svd_layer")(x[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x = torch.cat(x_out, dim=0)
            x = F.gelu(x)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)

            gate_feature = getattr(self, f"{prefix}_pw2_gate")(x)
            mask_one = control_module(gate_feature)
            output_candidates = []
            if self.training:

                for k in range(len(compression_ratios)):
                    out = getattr(self, f"{prefix}_pw2_svd_layer")(x, compression_ratios[k])
                    output_candidates.append(out)
                if self.include_original_weight:
                    output_candidates.append(getattr(self, f"{prefix}_pw2_svd_layer")(x, 0))
            mask_list = []
            for j in range(num_classes):

                mask_list.append(mask_one[:,j])
            if self.training:


                x = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(num_classes)])
            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x.size(0)):
                    idx = index[i].item()


                    if self.include_original_weight and idx == len(compression_ratios):
                        ratio = 0
                    else:
                        ratio = compression_ratios[idx]
                    out_i = getattr(self, f"{prefix}_pw2_svd_layer")(x[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x = torch.cat(x_out, dim=0)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)
        else:
            x = getattr(self, f"{prefix}_pw1")(x)
            x = F.gelu(x)
            x = getattr(self, f"{prefix}_pw2")(x)

        if self.layer_scale is not None:
            x = x * self.layer_scale.view(1, -1, 1, 1)

        x = self.drop_path(x)
        if self.adaptive is True:
            return residual + x, masks
        else:
            return residual + x


class ConvNeXt_Adaptive(nn.Module):
    """Adaptive ConvNeXt where every pointwise layer is low-rank and gated (plain variant). Input: (B, 3, H, W). Output: (logits, masks)."""
    # See configs.config_utils.is_plain_arch: every top-level adaptive
    # model class must declare this, so the "which variant" check never
    # relies on the arch/factory-function name.
    is_unified = False

    def __init__(self, compression_ratio_list, min_ranks=[], num_classes=10, embed_dim=10, hidden_dim=10, gate_type='rnn', gate_num_classes=2, input_channels=3, include_original_weight=True):
        super(ConvNeXt_Adaptive, self).__init__()

        self.num_adaptive_layers = 36
        self.include_original_weight = include_original_weight

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dims = [96, 192, 384, 768]
        self.depths = [3, 3, 9, 3]
        if isinstance(min_ranks, int):
            min_ranks = [min_ranks] * len(self.depths)
        self.drop_path_rate = 0.0
        self.layer_scale_init_value = 1e-6

        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm', gate_num_classes=gate_num_classes)
        elif gate_type == 'soft':
            self.control = SoftRNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        elif gate_type == 'seq_v1':
            self.control = SEQ_Gate_V1(embed_dim, gate_num_classes)
        elif gate_type == 'seq_v2':
            self.control = SEQ_Gate_V2(embed_dim, gate_num_classes)
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None


        self.stem_conv = nn.Conv2d(input_channels, self.dims[0], kernel_size=4, stride=4)
        self.stem_norm = nn.LayerNorm([self.dims[0], 8, 8], eps=1e-6)


        self.stages = nn.ModuleList()
        min_rank_index = 0
        minr = []
        for stage_idx, (dim, depth) in enumerate(zip(self.dims, self.depths)):
            blocks = []
            if stage_idx > 0:
                prev_dim = self.dims[stage_idx - 1]
                prev_H = 32 // (2 ** (stage_idx+1))
                prev_W = 32 // (2 ** (stage_idx+1))
                downsample_norm = nn.LayerNorm([prev_dim, prev_H, prev_W])
                downsample_conv = nn.Conv2d(self.dims[stage_idx-1], self.dims[stage_idx], kernel_size=2, stride=2)

                setattr(self, f"stage{stage_idx}_downsample_norm", downsample_norm)
                setattr(self, f"stage{stage_idx}_downsample_conv", downsample_conv)

                blocks.append(nn.Sequential(downsample_norm, downsample_conv))
            if min_ranks != []:
                minr = min_ranks[min_rank_index]
            for block_idx in range(depth):
                blocks.append(
                    ConvNeXtBlock_Adaptive(
                        dim=dim,
                        stage_id=stage_idx + 1,
                        block_id=block_idx + 1,
                        adaptive = True,
                        compression_ratio_list = compression_ratio_list,
                        min_rank = minr,
                        embed_dim = embed_dim,
                        drop_path_rate=self.drop_path_rate,
                        layer_scale_init_value=self.layer_scale_init_value,
                        gate_type = gate_type,
                        include_original_weight = include_original_weight
                    )
                )
            min_rank_index = min_rank_index + 1
            self.stages.append(nn.Sequential(*blocks))


        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head_norm = nn.LayerNorm(self.dims[-1], eps=1e-6)
        self.classifier = nn.Linear(self.dims[-1], num_classes)

    def forward(self, x, compression_ratios):
        masks = []

        x = self.stem_conv(x)
        x = self.stem_norm(x)

        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block in enumerate(stage):
                if hasattr(block, "adaptive") and block.adaptive is True:
                    x, mask_list = block(x, compression_ratios = compression_ratios, control_module = self.control)
                    masks.extend(mask_list)
                else:
                    x = block(x)


        x = self.global_avg_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.head_norm(x)
        x = self.classifier(x)
        return x, masks


def cifar10_convnext_adaptive(compression_ratios, pretrained=False, min_rank=1, **kwargs):
    """Factory for ConvNeXt_Adaptive, CIFAR-10 settings. Inputs: compression_ratios (list), min_rank (per-stage list or int), kwargs. Output: a model instance."""
    model = ConvNeXt_Adaptive(compression_ratios, min_rank, num_classes=10, **kwargs)
    return model
cifar10_convnext_adaptive.is_unified = ConvNeXt_Adaptive.is_unified


class ConvNeXtBlock_Adaptive_unified(nn.Module):
    """Unified-variant ConvNeXt block: both pointwise layers share one low-rank basis across choices. Input/output: (B, C, H, W) plus gate masks."""
    def __init__(self, dim, stage_id, block_id, compression_ratio_list, adaptive=False, min_rank = [], embed_dim = 10, drop_path_rate=0.0, layer_scale_init_value=1e-6):
        super(ConvNeXtBlock_Adaptive_unified, self).__init__()
        self.embed_dim = embed_dim
        self.stage_id =stage_id
        self.block_id = block_id
        self.adaptive = adaptive

        prefix = f"stage{stage_id}_block{block_id}"


        setattr(self, f"{prefix}_depthwise", nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim))


        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")


        if self.adaptive is True:
            prev_H_W = 32 // (2 ** (stage_id+1))
            pool_size = prev_H_W
            self.make_svd_block(pool_size, dim, 4 * dim, compression_ratio_list, min_rank, f"{prefix}_pw1", bias = True)
            self.make_svd_block(pool_size, 4 * dim, dim, compression_ratio_list, min_rank, f"{prefix}_pw2", bias = True)
        else:
            setattr(self, f"{prefix}_pw1", nn.Conv2d(dim, 4 * dim, kernel_size=1))
            setattr(self, f"{prefix}_pw2", nn.Conv2d(4 * dim, dim, kernel_size=1))


        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
        else:
            self.layer_scale = None

        self.drop_path = DropPath(drop_path_rate)
    def make_svd_block(self, pool_size, in_channels, out_channels, compression_ratio_list, min_rank, prefix, bias = None):
        W = H = pool_size
        layer = Conv1x1_SVD_Unified((W, H), in_channels, out_channels, compression_ratio_list, min_rank, bias = bias)
        gate_layer = nn.Sequential(
            nn.AvgPool2d(pool_size),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.embed_dim,
                      kernel_size=1,
                      stride=1)
            )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios, control_module = None):
        masks = []
        residual = x
        prefix = f"stage{self.stage_id}_block{self.block_id}"


        x = getattr(self, f"{prefix}_depthwise")(x)
        x = self.norm(x)


        if self.adaptive is True:
            batch_size = x.size(0)
            control_module.hidden_one = control_module.init_hidden(batch_size)
            gate_feature = getattr(self, f"{prefix}_pw1_gate")(x)
            mask_one = control_module(gate_feature)
            output_candidates = []
            if self.training:

                for k in range(len(compression_ratios)):
                    out = getattr(self, f"{prefix}_pw1_svd_layer")(x, compression_ratios[k])
                    output_candidates.append(out)

            mask_list = []
            for j in range(len(compression_ratios)):

                mask_list.append(mask_one[:,j])
            if self.training:


                x = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(compression_ratios))])
            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x.size(0)):
                    idx = index[i].item()


                    ratio = compression_ratios[idx]
                    out_i = getattr(self, f"{prefix}_pw1_svd_layer")(x[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x = torch.cat(x_out, dim=0)
            x = F.gelu(x)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)

            gate_feature = getattr(self, f"{prefix}_pw2_gate")(x)
            mask_one = control_module(gate_feature)
            output_candidates = []
            if self.training:

                for k in range(len(compression_ratios)):
                    out = getattr(self, f"{prefix}_pw2_svd_layer")(x, compression_ratios[k])
                    output_candidates.append(out)

            mask_list = []
            for j in range(len(compression_ratios)):

                mask_list.append(mask_one[:,j])
            if self.training:


                x = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(compression_ratios))])
            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x.size(0)):
                    idx = index[i].item()


                    ratio = compression_ratios[idx]
                    out_i = getattr(self, f"{prefix}_pw2_svd_layer")(x[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x = torch.cat(x_out, dim=0)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)
        else:
            x = getattr(self, f"{prefix}_pw1")(x)
            x = F.gelu(x)
            x = getattr(self, f"{prefix}_pw2")(x)

        if self.layer_scale is not None:
            x = x * self.layer_scale.view(1, -1, 1, 1)

        x = self.drop_path(x)
        if self.adaptive is True:
            return residual + x, masks
        else:
            return residual + x


class ConvNeXt_Adaptive_Unified(nn.Module):
    """Unified-variant adaptive ConvNeXt; all pointwise layers share one basis across choices. Input: (B, 3, H, W). Output: (logits, masks)."""
    is_unified = True  # see configs.config_utils.is_plain_arch

    def __init__(self, compression_ratio_list, min_ranks=[], num_classes=10, embed_dim=10, hidden_dim=10, gate_type='rnn', gate_num_classes=2, input_channels=3):
        super(ConvNeXt_Adaptive_Unified, self).__init__()

        self.num_adaptive_layers = 36

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dims = [96, 192, 384, 768]
        self.depths = [3, 3, 9, 3]
        if isinstance(min_ranks, int):
            min_ranks = [min_ranks] * len(self.depths)
        self.drop_path_rate = 0.0
        self.layer_scale_init_value = 1e-6

        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm', gate_num_classes=gate_num_classes)
        elif gate_type == 'soft':
            self.control = SoftRNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None


        self.stem_conv = nn.Conv2d(input_channels, self.dims[0], kernel_size=4, stride=4)
        self.stem_norm = nn.LayerNorm([self.dims[0], 8, 8], eps=1e-6)


        self.stages = nn.ModuleList()
        min_rank_index = 0
        minr = []
        for stage_idx, (dim, depth) in enumerate(zip(self.dims, self.depths)):
            blocks = []
            if stage_idx > 0:
                prev_dim = self.dims[stage_idx - 1]
                prev_H = 32 // (2 ** (stage_idx+1))
                prev_W = 32 // (2 ** (stage_idx+1))
                downsample_norm = nn.LayerNorm([prev_dim, prev_H, prev_W])
                downsample_conv = nn.Conv2d(self.dims[stage_idx-1], self.dims[stage_idx], kernel_size=2, stride=2)

                setattr(self, f"stage{stage_idx}_downsample_norm", downsample_norm)
                setattr(self, f"stage{stage_idx}_downsample_conv", downsample_conv)

                blocks.append(nn.Sequential(downsample_norm, downsample_conv))
            if min_ranks != []:
                minr = min_ranks[min_rank_index]
            for block_idx in range(depth):
                blocks.append(
                    ConvNeXtBlock_Adaptive_unified(
                        dim=dim,
                        stage_id=stage_idx + 1,
                        block_id=block_idx + 1,
                        adaptive = True,
                        compression_ratio_list = compression_ratio_list,
                        min_rank = minr,
                        embed_dim = embed_dim,
                        drop_path_rate=self.drop_path_rate,
                        layer_scale_init_value=self.layer_scale_init_value
                    )
                )
            min_rank_index = min_rank_index + 1
            self.stages.append(nn.Sequential(*blocks))


        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head_norm = nn.LayerNorm(self.dims[-1], eps=1e-6)
        self.classifier = nn.Linear(self.dims[-1], num_classes)

    def forward(self, x, compression_ratios):
        masks = []

        x = self.stem_conv(x)
        x = self.stem_norm(x)

        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block in enumerate(stage):
                if hasattr(block, "adaptive") and block.adaptive is True:
                    x, mask_list = block(x, compression_ratios = compression_ratios, control_module = self.control)
                    masks.extend(mask_list)
                else:
                    x = block(x)


        x = self.global_avg_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.head_norm(x)
        x = self.classifier(x)
        return x, masks


def cifar10_convnext_adaptive_unified(compression_ratios, pretrained=False, min_rank=1, **kwargs):
    """Factory for ConvNeXt_Adaptive_Unified, CIFAR-10 settings. Inputs: compression_ratios (list), min_rank, kwargs. Output: a model instance."""
    model = ConvNeXt_Adaptive_Unified(compression_ratios, min_rank, num_classes=10, **kwargs)
    return model
cifar10_convnext_adaptive_unified.is_unified = ConvNeXt_Adaptive_Unified.is_unified
