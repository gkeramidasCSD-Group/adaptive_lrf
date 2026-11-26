"""AlexNet-v2 (CIFAR-10): the plain backbone (cifar10_alexnet_v2, trained
by train_base.py) and its adaptive variants
(cifar10_alexnet_v2_adaptive*, trained by adaptive_training.py,
which warm-starts from the backbone's checkpoint)."""

import math

import torch
import torch.nn as nn

from modules.lrf import *
from .gates import broadcast_mask, RNNGate, SoftRNNGate, SEQ_Gate_V1, SEQ_Gate_V2


class AlexNet_v2(nn.Module):
    """Baseline AlexNet-v2 for CIFAR-10 (5 conv + 3 FC layers). Input: (B, 3, 32, 32). Output: (B, num_classes) logits."""
    def __init__(self, num_classes=10):
        super(AlexNet_v2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(256 * 2 * 2, 4096)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.act2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)

        x = self.drop(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


def cifar10_alexnet_v2(pretrained=False, **kwargs):
    """Factory for the baseline AlexNet_v2. Output: a model instance."""
    model = AlexNet_v2(num_classes=10)
    return model


class AlexNetV2_Adaptive(nn.Module):
    """Adaptive AlexNet-v2 with every conv (Tucker) and FC (SVD) layer low-rank and gated per sample. Constructor takes compression_ratio_list, min_rank, gate settings, include_original_weight. Input: (B, 3, 32, 32). Output: (logits, masks)."""
    is_unified = False  # see configs.config_utils.is_plain_arch

    def __init__(self, compression_ratio_list, min_rank = [], num_classes=10, embed_dim=10, hidden_dim=10, gate_type='rnn', gate_num_classes=2, include_original_weight=True):
        super(AlexNetV2_Adaptive, self).__init__()
        self.num_adaptive_layers = 8
        self.embed_dim = embed_dim
        self.gate_type = gate_type
        self.include_original_weight = include_original_weight

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


        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = self.make_tucker_block(32, 32, 16, 16, in_channels=3, out_channels=64, kernel_size=3, padding=1,
                                            compression_ratio_list=compression_ratio_list, min_ranks=min_rank, stride = 2, bias=True, conv=1)
        self.conv2 = self.make_tucker_block(8, 8, 8, 8, in_channels=64, out_channels=192, kernel_size=3, padding=1,
                                            compression_ratio_list=compression_ratio_list, min_ranks=min_rank, bias=True, conv=2)
        self.conv3 = self.make_tucker_block(4, 4, 4, 4, in_channels=192, out_channels=384, kernel_size=3, padding=1,
                                            compression_ratio_list=compression_ratio_list, min_ranks=min_rank, bias=True, conv=3)
        self.conv4 = self.make_tucker_block(4, 4, 4, 4, in_channels=384, out_channels=256, kernel_size=3, padding=1,
                                            compression_ratio_list=compression_ratio_list, min_ranks=min_rank, bias=True, conv=4)
        self.conv5 = self.make_tucker_block(4, 4, 4, 4, in_channels=256, out_channels=256, kernel_size=3, padding=1,
                                            compression_ratio_list=compression_ratio_list, min_ranks=min_rank, bias=True, conv=5)

        self.drop = nn.Dropout()

        self.make_svd_block(256 * 2 * 2, 4096, compression_ratio_list, min_rank, embed_dim, prefix = "fc1", inchannels = 256, bias = True)
        self.act1 = nn.ReLU(inplace=True)

        self.make_svd_block(4096, 4096, compression_ratio_list, min_rank, embed_dim, prefix = "fc2", bias = True)
        self.act2 = nn.ReLU(inplace=True)

        self.make_svd_block(4096, num_classes, compression_ratio_list, min_rank, embed_dim, prefix = "fc3", bias = True)

    def make_svd_block(self, insize, outsize, compression_ratio_list, min_rank, embed_dim, prefix, bias = None, inchannels = None):
        layer = Linear_SVD_NonUnified(insize, outsize, compression_ratio_list, min_rank = min_rank, bias = bias,
                                       include_original_weight=self.include_original_weight)
        if inchannels is not None:
            spatial = insize / inchannels
            pool_size = int(math.sqrt(spatial))
            gate_layer = nn.Sequential(
                nn.MaxPool2d(kernel_size=pool_size),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features=inchannels, out_features=self.embed_dim)
                )
        else:
            gate_layer = nn.Sequential(
                nn.Linear(in_features=insize, out_features=self.embed_dim)
                )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def make_tucker_block(self, w_in, h_in, w_out, h_out, in_channels, out_channels, kernel_size, compression_ratio_list, min_ranks=[],
                       stride = 1, padding = 0, bias = None, conv=1):
        layer = Conv_Tucker_NonUnified((w_in, h_in), (w_out, h_out), in_channels, out_channels, kernel_size,
                                        compression_ratio_list, min_rank=min_ranks, stride = stride, padding = padding, bias=bias,
                                        include_original_weight=self.include_original_weight)
        gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=in_channels, out_features=self.embed_dim)
            )

        setattr(self, 'conv{}_tucker_layer'.format(conv), layer)
        setattr(self, 'conv{}_gate'.format(conv), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios):
        # gate class k <-> choices[k] (see modules.lrf.ordered_gate_choices)
        choices = gate_choices_for(self, compression_ratios)
        masks = []

        batch_size = x.size(0)
        self.control.hidden_one = self.control.init_hidden(batch_size)

        for l in range (5):
            gate_feature = getattr(self, 'conv{}_gate'.format(l+1))(x)
            mask_one = self.control(gate_feature)
            mask_list = []
            for j in range(len(choices)):
                mask_list.append(mask_one[:,j])
            output_candidates = []
            if self.training:
                for k in range(len(choices)):
                    out = getattr(self, 'conv{}_tucker_layer'.format(l+1))(x, choices[k])
                    output_candidates.append(out)

            if self.training:
                x = sum([
                    broadcast_mask(mask_list[k], output_candidates[k]) * output_candidates[k]
                    for k in range(len(choices))
                ])

            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x.size(0)):
                    idx = index[i].item()
                    ratio = choices[idx]
                    out_i = getattr(self, 'conv{}_tucker_layer'.format(l+1))(x[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x = torch.cat(x_out, dim=0)
            x = self.relu(x)
            if l in [0, 1, 4]:
                x = self.maxpool(x)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)


        batch_size = x.size(0)
        if self.gate_type == 'rnn':
            self.control.hidden_one = self.control.init_hidden(batch_size)
        for l in range (3):
            gate_feature = getattr(self, 'fc{}_gate'.format(l+1))(x)
            mask_one = self.control(gate_feature)
            mask_list = []
            for j in range(len(choices)):
                mask_list.append(mask_one[:,j])


            if l == 0:
                x = x.view(x.size(0), 256 * 2 * 2)
                x = self.drop(x)
            output_candidates = []
            if self.training:
                for k in range(len(choices)):
                    out = getattr(self, 'fc{}_svd_layer'.format(l+1))(x, choices[k])
                    output_candidates.append(out)


            if self.training:


                x = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(choices))])
            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x.size(0)):
                    idx = index[i].item()
                    ratio = choices[idx]
                    out_i = getattr(self, 'fc{}_svd_layer'.format(l+1))(x[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x = torch.cat(x_out, dim=0)
            if l == 0:
                x = self.act1(x)
                x = self.drop(x)
            elif l == 1:
                x = self.act2(x)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)

        return x, masks


def cifar10_alexnet_v2_adaptive(compression_ratios, pretrained=False, min_rank=[], **kwargs):
    """Factory for AlexNetV2_Adaptive, CIFAR-10 settings. Inputs: compression_ratios (list), min_rank, kwargs passed to the model. Output: a model instance."""
    model = AlexNetV2_Adaptive(compression_ratios, min_rank, num_classes=10, **kwargs)
    return model
cifar10_alexnet_v2_adaptive.is_unified = AlexNetV2_Adaptive.is_unified


class AlexNetV2_Adaptive_Unified(nn.Module):
    """Unified-variant adaptive AlexNet-v2: every low-rank layer shares one basis (and bias) across all compression choices. Input: (B, 3, 32, 32). Output: (logits, masks)."""
    is_unified = True  # see configs.config_utils.is_plain_arch

    def __init__(self, compression_ratio_list, min_rank = [], num_classes=10, embed_dim=10, hidden_dim=10, gate_type='rnn', gate_num_classes=2):
        super(AlexNetV2_Adaptive_Unified, self).__init__()
        self.num_adaptive_layers = 8
        self.embed_dim = embed_dim
        self.gate_type = gate_type

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


        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = self.make_tucker_block(32, 32, 16, 16, in_channels=3, out_channels=64, kernel_size=3, padding=1,
                                            compression_ratio_list=compression_ratio_list, min_ranks=min_rank, stride = 2, bias=True, conv=1)
        self.conv2 = self.make_tucker_block(8, 8, 8, 8, in_channels=64, out_channels=192, kernel_size=3, padding=1,
                                            compression_ratio_list=compression_ratio_list, min_ranks=min_rank, bias=True, conv=2)
        self.conv3 = self.make_tucker_block(4, 4, 4, 4, in_channels=192, out_channels=384, kernel_size=3, padding=1,
                                            compression_ratio_list=compression_ratio_list, min_ranks=min_rank, bias=True, conv=3)
        self.conv4 = self.make_tucker_block(4, 4, 4, 4, in_channels=384, out_channels=256, kernel_size=3, padding=1,
                                            compression_ratio_list=compression_ratio_list, min_ranks=min_rank, bias=True, conv=4)
        self.conv5 = self.make_tucker_block(4, 4, 4, 4, in_channels=256, out_channels=256, kernel_size=3, padding=1,
                                            compression_ratio_list=compression_ratio_list, min_ranks=min_rank, bias=True, conv=5)

        self.drop = nn.Dropout()

        self.make_svd_block(256 * 2 * 2, 4096, compression_ratio_list, min_rank, embed_dim, prefix = "fc1", inchannels = 256, bias = True)
        self.act1 = nn.ReLU(inplace=True)

        self.make_svd_block(4096, 4096, compression_ratio_list, min_rank, embed_dim, prefix = "fc2", bias = True)
        self.act2 = nn.ReLU(inplace=True)

        self.make_svd_block(4096, num_classes, compression_ratio_list, min_rank, embed_dim, prefix = "fc3", bias = True)

    def make_svd_block(self, insize, outsize, compression_ratio_list, min_rank, embed_dim, prefix, bias = None, inchannels = None):
        layer = Linear_SVD_Unified(insize, outsize, compression_ratio_list, min_rank = min_rank, bias = bias)
        if inchannels is not None:
            spatial = insize / inchannels
            pool_size = int(math.sqrt(spatial))
            gate_layer = nn.Sequential(
                nn.MaxPool2d(kernel_size=pool_size),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features=inchannels, out_features=self.embed_dim)
                )
        else:
            gate_layer = nn.Sequential(
                nn.Linear(in_features=insize, out_features=self.embed_dim)
                )
        setattr(self, '{}_svd_layer'.format(prefix), layer)
        setattr(self, '{}_gate'.format(prefix), gate_layer)
        return layer, gate_layer

    def make_tucker_block(self, w_in, h_in, w_out, h_out, in_channels, out_channels, kernel_size, compression_ratio_list, min_ranks=[],
                       stride = 1, padding = 0, bias = None, conv=1):
        layer = Conv_Tucker_Unified((w_in, h_in), (w_out, h_out), in_channels, out_channels, kernel_size,
                                        compression_ratio_list, min_rank=min_ranks, stride = stride, padding = padding, bias=bias)
        gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=in_channels, out_features=self.embed_dim)
            )

        setattr(self, 'conv{}_tucker_layer'.format(conv), layer)
        setattr(self, 'conv{}_gate'.format(conv), gate_layer)
        return layer, gate_layer

    def forward(self, x, compression_ratios):
        # gate class k <-> choices[k] (see modules.lrf.ordered_gate_choices)
        choices = gate_choices_for(self, compression_ratios)
        masks = []

        batch_size = x.size(0)
        self.control.hidden_one = self.control.init_hidden(batch_size)

        for l in range (5):
            gate_feature = getattr(self, 'conv{}_gate'.format(l+1))(x)
            mask_one = self.control(gate_feature)
            mask_list = []
            for j in range(len(choices)):
                mask_list.append(mask_one[:,j])
            output_candidates = []
            if self.training:
                for k in range(len(choices)):
                    out = getattr(self, 'conv{}_tucker_layer'.format(l+1))(x, choices[k])
                    output_candidates.append(out)


            if self.training:
                x = sum([
                    broadcast_mask(mask_list[k], output_candidates[k]) * output_candidates[k]
                    for k in range(len(choices))
                ])

            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x.size(0)):
                    idx = index[i].item()


                    ratio = choices[idx]
                    out_i = getattr(self, 'conv{}_tucker_layer'.format(l+1))(x[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x = torch.cat(x_out, dim=0)
            x = self.relu(x)
            if l in [0, 1, 4]:
                x = self.maxpool(x)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)


        batch_size = x.size(0)
        if self.gate_type == 'rnn':
            self.control.hidden_one = self.control.init_hidden(batch_size)
        for l in range (3):
            gate_feature = getattr(self, 'fc{}_gate'.format(l+1))(x)
            mask_one = self.control(gate_feature)
            mask_list = []
            for j in range(len(choices)):
                mask_list.append(mask_one[:,j])


            if l == 0:
                x = x.view(x.size(0), 256 * 2 * 2)
                x = self.drop(x)
            output_candidates = []
            if self.training:
                for k in range(len(choices)):
                    out = getattr(self, 'fc{}_svd_layer'.format(l+1))(x, choices[k])
                    output_candidates.append(out)


            if self.training:


                x = sum([broadcast_mask(mask_list[k], output_candidates[0]) * output_candidates[k] for k in range(len(choices))])
            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x.size(0)):
                    idx = index[i].item()


                    ratio = choices[idx]
                    out_i = getattr(self, 'fc{}_svd_layer'.format(l+1))(x[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x = torch.cat(x_out, dim=0)
            if l == 0:
                x = self.act1(x)
                x = self.drop(x)
            elif l == 1:
                x = self.act2(x)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)

        return x, masks


def cifar10_alexnet_v2_adaptive_unified(compression_ratios, pretrained=False, min_rank=[], **kwargs):
    """Factory for AlexNetV2_Adaptive_Unified, CIFAR-10 settings. Inputs: compression_ratios (list), min_rank, kwargs. Output: a model instance."""
    model = AlexNetV2_Adaptive_Unified(compression_ratios, min_rank, num_classes=10, **kwargs)
    return model
cifar10_alexnet_v2_adaptive_unified.is_unified = AlexNetV2_Adaptive_Unified.is_unified
