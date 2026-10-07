"""EfficientFormer-L3 (CIFAR-10): the plain backbone
(cifar10_efficienformer_l3, trained by train_base.py) and its adaptive
variants (cifar10_efficientformer_l3_adaptive*, trained by
adaptive_training.py, which warm-starts from the backbone's
checkpoint)."""

import torch
import torch.nn as nn

from modules.lrf import *
from .common import DropPath
from .gates import broadcast_mask, RNNGate, SoftRNNGate, SEQ_Gate_V1, SEQ_Gate_V2

EfficientFormer_width = {
    'l1': [48, 96, 224, 448],
    'l3': [64, 128, 320, 512],
    'l7': [96, 192, 384, 768],
}


EfficientFormer_depth = {
    'l1': [3, 2, 6, 4],
    'l3': [4, 4, 12, 6],
    'l7': [6, 6, 18, 8],
}


class Pooling(nn.Module):
    """Pooling token mixer (average pool minus identity) used in the early stages. Input/output: (B, C, H, W)."""
    def __init__(self, size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(size, stride=1, padding=size // 2)

    def forward(self, x):
        return self.pool(x) - x


class ConvMLP(nn.Module):
    """Pointwise-conv MLP block (1x1 conv, GELU, 1x1 conv). Input/output: (B, C, H, W)."""
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.gelu = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, 1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.pw2(x)
        x = self.bn2(x)
        return x


class Meta4D(nn.Module):
    """Meta block on 4D feature maps: a token mixer followed by a ConvMLP, with residual connections. Input/output: (B, C, H, W)."""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.pool = Pooling()
        self.mlp = ConvMLP(dim)
        self.drop = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop(self.pool(x))
        x = x + self.drop(self.mlp(x))
        return x


class PatchEmbed(nn.Module):
    """Strided conv patch embedding that downsamples between stages. Input: (B, C_in, H, W). Output: (B, C_out, H/s, W/s)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class EfficientFormer(nn.Module):
    """Baseline EfficientFormer classifier. Input: (B, 3, H, W). Output: (B, num_classes) logits."""
    def __init__(self, dims, depths, num_classes=10):
        super().__init__()

        self.stem = PatchEmbed(3, dims[0])
        self.stages = nn.ModuleList()

        for i in range(len(dims)):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(Meta4D(dims[i]))
            self.stages.append(nn.Sequential(*blocks))

            if i < len(dims) - 1:
                self.stages.append(
                    PatchEmbed(dims[i], dims[i + 1])
                )

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)

        for block in self.stages:
            x = block(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.mean(dim=1)

        return self.head(x)


def efficientformer_l3(num_classes = 10):
    """Builds the baseline EfficientFormer with the L3 width/depth config. Input: num_classes. Output: a model instance."""
    return EfficientFormer(dims=EfficientFormer_width['l3'], depths=EfficientFormer_depth['l3'], num_classes=num_classes)


def cifar10_efficienformer_l3(pretrained=False, **kwargs):
    """Factory for the baseline L3 model (name keeps its original spelling). Output: a model instance."""
    model = efficientformer_l3(num_classes=10)
    return model


class ConvMLP_adaptive(nn.Module):
    """Adaptive ConvMLP: both pointwise layers are low-rank and gated per sample (plain variant). Input/output: (B, C, H, W) plus gate masks."""
    def __init__(self, dim, compression_ratios, mlp_ratio=4, stage_id=0, min_rank = [], embed_dim = 10, gate_type = 'rnn', include_original_weight=True):
        super().__init__()
        self.gate_type = gate_type
        self.include_original_weight = include_original_weight
        hidden = int(dim * mlp_ratio)

        pool_size = int(32 / (2**stage_id))
        self.pw1 = self.make_svd_block(pool_size, dim, hidden, compression_ratios, min_rank, embed_dim, "pw1", bias = True)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.gelu = nn.GELU()

        pool_size = int(32 / (2**stage_id))
        self.pw2 = self.make_svd_block(pool_size, hidden, dim, compression_ratios, min_rank, embed_dim, "pw2", bias = True)
        self.bn2 = nn.BatchNorm2d(dim)
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

    def forward(self, x, compression_ratios = None, control_module=None):
        masks = []
        num_classes = len(compression_ratios) + (1 if self.include_original_weight else 0)

        prefix = "pw1"
        batch_size = x.size(0)
        if self.gate_type == 'rnn':
            control_module.hidden_one = control_module.init_hidden(batch_size)
        gate_feature = getattr(self, f"{prefix}_gate")(x)
        mask_one = control_module(gate_feature)
        output_candidates = []
        if self.training:
            for k in range(len(compression_ratios)):
                out = getattr(self, f"{prefix}_svd_layer")(x, compression_ratios[k])
                output_candidates.append(out)
            if self.include_original_weight:
                output_candidates.append(getattr(self, f"{prefix}_svd_layer")(x, 0))
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
                out_i = getattr(self, f"{prefix}_svd_layer")(x[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x = torch.cat(x_out, dim=0)
        mask_list = [mask.squeeze() for mask in mask_list]
        masks.append(mask_list)

        x = self.bn1(x)
        x = self.gelu(x)


        prefix = "pw2"


        gate_feature = getattr(self, f"{prefix}_gate")(x)
        mask_one = control_module(gate_feature)
        output_candidates_2 = []
        if self.training:
            for k in range(len(compression_ratios)):
                out = getattr(self, f"{prefix}_svd_layer")(x, compression_ratios[k])
                output_candidates_2.append(out)
            if self.include_original_weight:
                output_candidates_2.append(getattr(self, f"{prefix}_svd_layer")(x, 0))
        mask_list = []
        for j in range(num_classes):
            mask_list.append(mask_one[:,j])
        if self.training:
            x = sum([broadcast_mask(mask_list[k], output_candidates_2[0]) * output_candidates_2[k] for k in range(num_classes)])
        else:
            index = mask_one.argmax(dim=1)
            x_out = []
            for i in range(x.size(0)):
                idx = index[i].item()
                if self.include_original_weight and idx == len(compression_ratios):
                    ratio = 0
                else:
                    ratio = compression_ratios[idx]
                out_i = getattr(self, f"{prefix}_svd_layer")(x[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x = torch.cat(x_out, dim=0)
        mask_list = [mask.squeeze() for mask in mask_list]
        masks.append(mask_list)

        x = self.bn2(x)
        return x, masks


class Meta4D_adaptive(nn.Module):
    """Adaptive Meta4D block built from adaptive ConvMLP (plain variant). Input/output: (B, C, H, W) plus gate masks."""
    def __init__(self, dim, compression_ratios, stage_id=0, drop_path=0., min_rank=[], gate_type = 'rnn', include_original_weight=True):
        super().__init__()
        self.adaptive = True
        self.pool = Pooling()
        self.mlp = ConvMLP_adaptive(dim, stage_id=stage_id, compression_ratios=compression_ratios, min_rank=min_rank, gate_type = gate_type,
                                     include_original_weight=include_original_weight)
        self.drop = DropPath(drop_path)

    def forward(self, x, compression_ratios, control_module=None):
        x = x + self.drop(self.pool(x))
        x_aux, masks = self.mlp(x, compression_ratios, control_module)
        x = x + self.drop(x_aux)
        return x, masks


class EfficientFormer_adaptive(nn.Module):
    """Adaptive EfficientFormer-L3 with gated low-rank layers (plain variant). Input: (B, 3, H, W). Output: (logits, masks)."""
    is_unified = False  # see configs.config_utils.is_plain_arch

    def __init__(self, dims, depths, compression_ratios, min_rank=[],
    embed_dim=10, num_classes=10, hidden_dim=10, gate_type='rnn', gate_num_classes=2, include_original_weight=True):
        super().__init__()
        self.num_adaptive_layers = int(sum(depths) * 2)
        self.embed_dim = embed_dim


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

        self.stem = PatchEmbed(3, dims[0])
        self.stages = nn.ModuleList()

        for i in range(len(dims)):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(Meta4D_adaptive(dims[i], stage_id=i, compression_ratios=compression_ratios, min_rank=min_rank, gate_type = gate_type,
                                               include_original_weight=include_original_weight))
            self.stages.append(nn.Sequential(*blocks))

            if i < len(dims) - 1:
                self.stages.append(
                    PatchEmbed(dims[i], dims[i + 1])
                )

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x, compression_ratios):
        masks = []
        x = self.stem(x)

        for stage in self.stages:
            if isinstance(stage, nn.Sequential):
                for block in stage:
                    if hasattr(block, "adaptive") and block.adaptive:
                        x, mask_list = block(x, compression_ratios, self.control)
                        masks.extend(mask_list)
                    else:
                        x = block(x)
            else:

                x = stage(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)

        return x, masks


def efficientformer_l3_adaptive(compression_ratios, pretrained=False, min_rank=[], num_classes = 10, gate_type = 'rnn', **kwargs):
    """Builds EfficientFormer_adaptive with the L3 config. Inputs: compression_ratios, min_rank, num_classes, gate_type, kwargs. Output: a model instance."""
    return EfficientFormer_adaptive(dims=EfficientFormer_width['l3'], depths=EfficientFormer_depth['l3'],
    compression_ratios=compression_ratios, min_rank=min_rank, num_classes=num_classes, gate_type = gate_type, **kwargs)
efficientformer_l3_adaptive.is_unified = EfficientFormer_adaptive.is_unified


def cifar10_efficientformer_l3_adaptive(compression_ratios, pretrained=False, min_rank=[], **kwargs):
    """Factory for EfficientFormer_adaptive, CIFAR-10 settings. Inputs: compression_ratios (list), min_rank, kwargs. Output: a model instance."""
    model = efficientformer_l3_adaptive(compression_ratios, min_rank, num_classes=10, **kwargs)
    return model
cifar10_efficientformer_l3_adaptive.is_unified = EfficientFormer_adaptive.is_unified


class ConvMLP_adaptive_unified(nn.Module):
    """Unified-variant adaptive ConvMLP: pointwise layers share one basis across choices. Input/output: (B, C, H, W) plus gate masks."""
    def __init__(self, dim, compression_ratios, mlp_ratio=4, stage_id=0, min_rank = [], embed_dim = 10):
        super().__init__()
        hidden = int(dim * mlp_ratio)

        pool_size = int(32 / (2**stage_id))
        self.pw1 = self.make_svd_block(pool_size, dim, hidden, compression_ratios, min_rank, embed_dim, "pw1", bias = True)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.gelu = nn.GELU()

        pool_size = int(32 / (2**stage_id))
        self.pw2 = self.make_svd_block(pool_size, hidden, dim, compression_ratios, min_rank, embed_dim, "pw2", bias = True)
        self.bn2 = nn.BatchNorm2d(dim)
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

    def forward(self, x, compression_ratios = None, control_module=None):
        masks = []

        prefix = "pw1"
        batch_size = x.size(0)
        control_module.hidden_one = control_module.init_hidden(batch_size)
        gate_feature = getattr(self, f"{prefix}_gate")(x)
        mask_one = control_module(gate_feature)
        output_candidates = []
        if self.training:
            for k in range(len(compression_ratios)):
                out = getattr(self, f"{prefix}_svd_layer")(x, compression_ratios[k])
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
                out_i = getattr(self, f"{prefix}_svd_layer")(x[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x = torch.cat(x_out, dim=0)
        mask_list = [mask.squeeze() for mask in mask_list]
        masks.append(mask_list)

        x = self.bn1(x)
        x = self.gelu(x)


        prefix = "pw2"


        gate_feature = getattr(self, f"{prefix}_gate")(x)
        mask_one = control_module(gate_feature)
        output_candidates_2 = []
        if self.training:
            for k in range(len(compression_ratios)):
                out = getattr(self, f"{prefix}_svd_layer")(x, compression_ratios[k])
                output_candidates_2.append(out)

        mask_list = []
        for j in range(len(compression_ratios)):
            mask_list.append(mask_one[:,j])
        if self.training:
            x = sum([broadcast_mask(mask_list[k], output_candidates_2[0]) * output_candidates_2[k] for k in range(len(compression_ratios))])
        else:
            index = mask_one.argmax(dim=1)
            x_out = []
            for i in range(x.size(0)):
                idx = index[i].item()
                if idx == len(compression_ratios):
                    ratio = 0
                else:
                    ratio = compression_ratios[idx]
                out_i = getattr(self, f"{prefix}_svd_layer")(x[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x = torch.cat(x_out, dim=0)
        mask_list = [mask.squeeze() for mask in mask_list]
        masks.append(mask_list)

        x = self.bn2(x)
        return x, masks


class Meta4D_adaptive_unified(nn.Module):
    """Unified-variant adaptive Meta4D block. Input/output: (B, C, H, W) plus gate masks."""
    def __init__(self, dim, compression_ratios, stage_id=0, drop_path=0., min_rank=[]):
        super().__init__()
        self.adaptive = True
        self.pool = Pooling()
        self.mlp = ConvMLP_adaptive_unified(dim, stage_id=stage_id, compression_ratios=compression_ratios, min_rank=min_rank)
        self.drop = DropPath(drop_path)

    def forward(self, x, compression_ratios, control_module=None):
        x = x + self.drop(self.pool(x))
        x_aux, masks = self.mlp(x, compression_ratios, control_module)
        x = x + self.drop(x_aux)
        return x, masks


class EfficientFormer_adaptive_unified(nn.Module):
    """Unified-variant adaptive EfficientFormer-L3 (shared basis per low-rank layer). Input: (B, 3, H, W). Output: (logits, masks)."""
    is_unified = True  # see configs.config_utils.is_plain_arch

    def __init__(self, dims, depths, compression_ratios, min_rank=[],
    embed_dim=10, num_classes=10, hidden_dim=10, gate_type='rnn', gate_num_classes=2):
        super().__init__()
        self.num_adaptive_layers = int(sum(depths) * 2)
        self.embed_dim = embed_dim


        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm', gate_num_classes=gate_num_classes)
        elif gate_type == 'soft':
            self.control = SoftRNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None

        self.stem = PatchEmbed(3, dims[0])
        self.stages = nn.ModuleList()

        for i in range(len(dims)):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(Meta4D_adaptive_unified(dims[i], stage_id=i, compression_ratios=compression_ratios, min_rank=min_rank))
            self.stages.append(nn.Sequential(*blocks))

            if i < len(dims) - 1:
                self.stages.append(
                    PatchEmbed(dims[i], dims[i + 1])
                )

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x, compression_ratios):
        masks = []
        x = self.stem(x)

        for stage in self.stages:
            if isinstance(stage, nn.Sequential):
                for block in stage:
                    if hasattr(block, "adaptive") and block.adaptive:
                        x, mask_list = block(x, compression_ratios, self.control)
                        masks.extend(mask_list)
                    else:
                        x = block(x)
            else:

                x = stage(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)

        return x, masks


def efficientformer_l3_adaptive_unified(compression_ratios, pretrained=False, min_rank=[], num_classes = 10, **kwargs):
    """Builds EfficientFormer_adaptive_unified with the L3 config. Inputs: compression_ratios, min_rank, num_classes, kwargs. Output: a model instance."""
    return EfficientFormer_adaptive_unified(dims=EfficientFormer_width['l3'], depths=EfficientFormer_depth['l3'],
    compression_ratios=compression_ratios, min_rank=min_rank, num_classes=num_classes, **kwargs)
efficientformer_l3_adaptive_unified.is_unified = EfficientFormer_adaptive_unified.is_unified


def cifar10_efficientformer_l3_adaptive_unified(compression_ratios, pretrained=False, min_rank=[], **kwargs):
    """Factory for EfficientFormer_adaptive_unified, CIFAR-10 settings. Inputs: compression_ratios (list), min_rank, kwargs. Output: a model instance."""
    model = efficientformer_l3_adaptive_unified(compression_ratios, min_rank, num_classes=10, **kwargs)
    return model
cifar10_efficientformer_l3_adaptive_unified.is_unified = EfficientFormer_adaptive_unified.is_unified
