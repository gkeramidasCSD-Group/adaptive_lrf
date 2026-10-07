"""LeNet5 (MNIST): the plain backbone (mnist_lenet_5, trained by
train_base.py) and its adaptive variants (lenet5_mnist_rnn_gate_*, trained
by adaptive_training.py, which warm-starts from the backbone's
checkpoint)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.lrf import *
from .gates import broadcast_mask, RNNGate, SoftRNNGate


class LeNet5_mnist(nn.Module):
    """Baseline LeNet-5 for MNIST (2 conv + 3 FC layers). Input: (B, 1, 28, 28). Output: (B, 10) logits."""
    def __init__(self, num_classes=10):
        super(LeNet5_mnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def mnist_lenet_5(pretrained=False, **kwargs):
    """Factory for the baseline LeNet5_mnist. Output: a model instance."""
    model = LeNet5_mnist(num_classes=10)
    return model


class LeNet5_mnist_RecurrentGateSP_fisrtFCLayer(nn.Module):
    """Adaptive LeNet-5 whose first FC layer is low-rank (SVD) with an RNN gate choosing a compression ratio per sample. Input: (B, 1, 28, 28). Output: (logits, masks)."""
    is_unified = False  # see configs.config_utils.is_plain_arch

    def __init__(self, compression_ratio_list, min_ranks, num_classes=10, embed_dim=10, hidden_dim=10, gate_type='rnn', gate_num_classes=2, include_original_weight=True):
        super(LeNet5_mnist_RecurrentGateSP_fisrtFCLayer, self).__init__()

        self.num_adaptive_layers = 1
        self.include_original_weight = include_original_weight


        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2)


        self.make_svd_block(16*5*5, 120, compression_ratio_list=compression_ratio_list, min_ranks=min_ranks, bias=True, fc=1)

        self.fc2 = nn.Linear(in_features=120, out_features=84)


        self.fc3 = nn.Linear(in_features=84, out_features=10)


        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm', gate_num_classes=gate_num_classes)
        elif gate_type == 'soft':
            self.control = SoftRNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None

    def make_svd_block(self, insize, outsize, compression_ratio_list, min_ranks=[], bias = None, fc=1):
        pooling_output_size = 8
        layer = Linear_SVD_NonUnified(insize, outsize, compression_ratio_list=compression_ratio_list, min_rank=min_ranks, bias=bias,
                                       include_original_weight=self.include_original_weight)
        gate_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(output_size=pooling_output_size),
            nn.Linear(in_features=pooling_output_size, out_features=self.embed_dim)

            )

        setattr(self, 'fc{}_svd_layer'.format(fc), layer)
        setattr(self, 'fc{}_gate'.format(fc), gate_layer)


        return layer, gate_layer


    def forward(self, x, compression_ratios):
        batch_size = x.size(0)

        self.control.hidden_one = self.control.init_hidden(batch_size)
        self.control = self.control.to(x.device)
        masks = []
        num_classes = len(compression_ratios) + (1 if self.include_original_weight else 0)


        x = F.tanh(self.conv1(x))
        x = self.pool1(x)

        x = F.tanh(self.conv2(x))
        x = self.pool1(x)

        x = x.view(x.size(0), -1)

        gate_feature = getattr(self, 'fc1_gate')(x)
        mask_one = self.control(gate_feature)
        output_candidates = []
        if self.training:
            for k in range(len(compression_ratios)):
                out = getattr(self, 'fc1_svd_layer')(x, compression_ratios[k])
                output_candidates.append(out)
            if self.include_original_weight:
                output_candidates.append(getattr(self, 'fc1_svd_layer')(x, 0))
        mask_list = []
        for j in range(num_classes):
            mask_list.append(mask_one[:,j])

        if self.training:
            x = sum([broadcast_mask(mask_list[k], out) * output_candidates[k] for k in range(num_classes)])
        else:
            index = mask_one.argmax(dim=1)
            x_out = []
            for i in range(x.size(0)):
                idx = index[i].item()
                if self.include_original_weight and idx == len(compression_ratios):
                    ratio = 0
                else:
                    ratio = compression_ratios[idx]
                out_i = getattr(self, 'fc1_svd_layer')(x[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x = torch.cat(x_out, dim=0)
        x = F.tanh(x)
        mask_list = [mask.squeeze() for mask in mask_list]
        masks.append(mask_list)

        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)

        return x, masks


def lenet5_mnist_rnn_gate_first_fc_layer(compression_ratios, pretrained=False, min_rank=[], **kwargs):
    """LeNet5 trained on mnist with Recurrent Gate"""
    model = LeNet5_mnist_RecurrentGateSP_fisrtFCLayer(compression_ratio_list=compression_ratios, min_ranks=min_rank,
                                                      num_classes=10, embed_dim=10, hidden_dim=10, **kwargs)
    return model
lenet5_mnist_rnn_gate_first_fc_layer.is_unified = LeNet5_mnist_RecurrentGateSP_fisrtFCLayer.is_unified


class LeNet5_mnist_RecurrentGateSP_fisrtFCLayer_Unified(nn.Module):
    """Unified-variant adaptive LeNet-5: the first FC layer shares one low-rank basis across all compression choices, gated per sample. Input: (B, 1, 28, 28). Output: (logits, masks)."""
    is_unified = True  # see configs.config_utils.is_plain_arch

    def __init__(self, compression_ratio_list, min_ranks, num_classes=10, embed_dim=10, hidden_dim=10, gate_type='rnn', gate_num_classes=2):
        super(LeNet5_mnist_RecurrentGateSP_fisrtFCLayer_Unified, self).__init__()

        self.num_adaptive_layers = 1


        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2)


        self.make_svd_block(16*5*5, 120, compression_ratio_list=compression_ratio_list, min_ranks=min_ranks, bias=True, fc=1)

        self.fc2 = nn.Linear(in_features=120, out_features=84)


        self.fc3 = nn.Linear(in_features=84, out_features=10)


        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm', gate_num_classes=gate_num_classes)
        elif gate_type == 'soft':
            self.control = SoftRNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None

    def make_svd_block(self, insize, outsize, compression_ratio_list, min_ranks=[], bias = None, fc=1):
        pooling_output_size = 8
        layer = Linear_SVD_Unified(insize, outsize, compression_ratio_list=compression_ratio_list, min_rank=min_ranks, bias=bias)
        gate_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(output_size=pooling_output_size),
            nn.Linear(in_features=pooling_output_size, out_features=self.embed_dim)

            )

        setattr(self, 'fc{}_svd_layer'.format(fc), layer)
        setattr(self, 'fc{}_gate'.format(fc), gate_layer)


        return layer, gate_layer


    def forward(self, x, compression_ratios):
        batch_size = x.size(0)

        self.control.hidden_one = self.control.init_hidden(batch_size)
        masks = []


        x = F.tanh(self.conv1(x))
        x = self.pool1(x)

        x = F.tanh(self.conv2(x))
        x = self.pool1(x)

        x = x.view(x.size(0), -1)

        gate_feature = getattr(self, 'fc1_gate')(x)
        mask_one = self.control(gate_feature)
        output_candidates = []
        if self.training:
            for k in range(len(compression_ratios)):
                out = getattr(self, 'fc1_svd_layer')(x, compression_ratios[k])
                output_candidates.append(out)

        mask_list = []
        for j in range(len(compression_ratios)):
            mask_list.append(mask_one[:,j])

        if self.training:
            x = sum([broadcast_mask(mask_list[k], out) * output_candidates[k] for k in range(len(compression_ratios))])
        else:
            index = mask_one.argmax(dim=1)
            x_out = []
            for i in range(x.size(0)):
                idx = index[i].item()


                ratio = compression_ratios[idx]
                out_i = getattr(self, 'fc1_svd_layer')(x[i].unsqueeze(0), ratio)
                x_out.append(out_i)
            x = torch.cat(x_out, dim=0)
        x = F.tanh(x)
        mask_list = [mask.squeeze() for mask in mask_list]
        masks.append(mask_list)

        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)

        return x, masks


def lenet5_mnist_rnn_gate_first_fc_layer_unified(compression_ratios, pretrained=False, min_rank=[], **kwargs):
    """LeNet5 trained on mnist with Recurrent Gate"""
    model = LeNet5_mnist_RecurrentGateSP_fisrtFCLayer_Unified(compression_ratio_list=compression_ratios, min_ranks=min_rank,
                                                      num_classes=10, embed_dim=10, hidden_dim=10, **kwargs)
    return model
lenet5_mnist_rnn_gate_first_fc_layer_unified.is_unified = LeNet5_mnist_RecurrentGateSP_fisrtFCLayer_Unified.is_unified


class LeNet5_mnist_RecurrentGateSP_2FCLayers(nn.Module):
    """Adaptive LeNet-5 with two low-rank FC layers, each with its own RNN gate. Input: (B, 1, 28, 28). Output: (logits, masks)."""
    is_unified = False  # see configs.config_utils.is_plain_arch

    def __init__(self, compression_ratio_list, min_ranks, num_classes=10, embed_dim=10, hidden_dim=10, gate_type='rnn', gate_num_classes=2, include_original_weight=True):
        super(LeNet5_mnist_RecurrentGateSP_2FCLayers, self).__init__()

        self.num_adaptive_layers = 2
        self.include_original_weight = include_original_weight


        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2)


        self.make_svd_block(16*5*5, 120, compression_ratio_list=compression_ratio_list, min_ranks=min_ranks, bias=True, fc=1)


        self.make_svd_block(120, 84, compression_ratio_list=compression_ratio_list, min_ranks=min_ranks, bias=True, fc=2)

        self.fc3 = nn.Linear(in_features=84, out_features=10)


        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm', gate_num_classes=gate_num_classes)
        elif gate_type == 'soft':
            self.control = SoftRNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None

    def make_svd_block(self, insize, outsize, compression_ratio_list, min_ranks=[], bias = None, fc=1):
        pooling_output_size = 8
        layer = Linear_SVD_NonUnified(insize, outsize, compression_ratio_list=compression_ratio_list, min_rank=min_ranks, bias=bias,
                                       include_original_weight=self.include_original_weight)
        gate_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(output_size=pooling_output_size),
            nn.Linear(in_features=pooling_output_size, out_features=self.embed_dim)

            )

        setattr(self, 'fc{}_svd_layer'.format(fc), layer)
        setattr(self, 'fc{}_gate'.format(fc), gate_layer)


        return layer, gate_layer


    def forward(self, x, compression_ratios):
        batch_size = x.size(0)

        self.control.hidden_one = self.control.init_hidden(batch_size)
        masks = []
        num_classes = len(compression_ratios) + (1 if self.include_original_weight else 0)


        x = F.tanh(self.conv1(x))
        x = self.pool1(x)

        x = F.tanh(self.conv2(x))
        x = self.pool1(x)

        x = x.view(x.size(0), -1)

        for l in range (self.num_adaptive_layers):
            gate_feature = getattr(self, 'fc{}_gate'.format(l+1))(x)
            mask_one = self.control(gate_feature)
            output_candidates = []
            if self.training:
                for k in range(len(compression_ratios)):
                    out = getattr(self, 'fc{}_svd_layer'.format(l+1))(x, compression_ratios[k])
                    output_candidates.append(out)
                if self.include_original_weight:
                    output_candidates.append(getattr(self, 'fc{}_svd_layer'.format(l+1))(x, 0))
            mask_list = []
            for j in range(num_classes):
                mask_list.append(mask_one[:,j])

            if self.training:
                x = sum([broadcast_mask(mask_list[k], out) * output_candidates[k] for k in range(num_classes)])
            else:
                index = mask_one.argmax(dim=1)
                x_out = []
                for i in range(x.size(0)):
                    idx = index[i].item()
                    if self.include_original_weight and idx == len(compression_ratios):
                        ratio = 0
                    else:
                        ratio = compression_ratios[idx]
                    out_i = getattr(self, 'fc{}_svd_layer'.format(l+1))(x[i].unsqueeze(0), ratio)
                    x_out.append(out_i)
                x = torch.cat(x_out, dim=0)
            x = F.tanh(x)
            mask_list = [mask.squeeze() for mask in mask_list]
            masks.append(mask_list)


        x = self.fc3(x)

        return x, masks


def lenet5_mnist_rnn_gate_two_fc_layers(compression_ratios, pretrained=False, min_rank=[], **kwargs):
    """LeNet5 trained on mnist with Recurrent Gate"""
    model = LeNet5_mnist_RecurrentGateSP_2FCLayers(compression_ratio_list=compression_ratios, min_ranks=min_rank,
                                                      num_classes=10, embed_dim=10, hidden_dim=10, **kwargs)
    return model
lenet5_mnist_rnn_gate_two_fc_layers.is_unified = LeNet5_mnist_RecurrentGateSP_2FCLayers.is_unified
