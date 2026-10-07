"""Thin re-export aggregator for all adaptive (low-rank + gating) architectures.

Each network's adaptive classes live in their own module under arch/ (see
arch/<network>.py); this module only re-exports the selectable factory
functions so that `from arch import models_adaptive as models` and
`models.__dict__[args.arch]` keep working unchanged.
"""

from .lenet5 import (
    lenet5_mnist_rnn_gate_first_fc_layer,
    lenet5_mnist_rnn_gate_first_fc_layer_unified,
    lenet5_mnist_rnn_gate_two_fc_layers,
)
from .alexnet import (
    cifar10_alexnet_v2_adaptive,
    cifar10_alexnet_v2_adaptive_unified,
)
from .convnext import (
    cifar10_convnext_adaptive,
    cifar10_convnext_adaptive_unified,
)
from .efficientformer import (
    cifar10_efficientformer_l3_adaptive,
    cifar10_efficientformer_l3_adaptive_unified,
)
from .ghostnet import (
    cifar10_ghostnet_adaptive,
    cifar10_ghostnet_adaptive_unified,
)
