"""Thin re-export aggregator for all baseline (non-adaptive) architectures.

Each network's baseline classes live in their own module under arch/ (see
arch/<network>.py); this module only re-exports the selectable factory
functions so that `from arch import models_baseline as models` and
`models.__dict__[args.arch]` keep working unchanged.
"""

from .lenet5 import mnist_lenet_5
from .alexnet import cifar10_alexnet_v2
from .convnext import cifar10_convnext
from .efficientformer import cifar10_efficienformer_l3
from .ghostnet import cifar10_ghostnet
