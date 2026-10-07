"""Gate modules shared by the adaptive network architectures.

Every gate below returns a plain `(batch_size, num_classes)` mask - it has
no idea whether the layer it's about to gate is a Linear (2D output) or a
Conv2d (4D output), and it shouldn't need to: that's a fact about the
specific layer, not about the gate. Reshaping the mask to broadcast
correctly against a specific layer's output is the caller's job - see
`broadcast_mask` below - which is also why there's only one of each gate
class instead of separate "plain" and "_conv" versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable


def repackage_hidden(h):
    """Detaches an RNN hidden state (or nested tuple/list of them) from
    its computation history, so it can be reused as the next forward
    call's input without backpropagating through previous batches."""
    if type(h) == Variable:
        return Variable(h.data)
    elif isinstance(h, (tuple, list)):
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h


def broadcast_mask(mask, target):
    """Reshape a per-sample gate mask, e.g. `mask_one[:, j, :]`, so it
    broadcasts correctly against `target` (a specific layer's output)
    regardless of whether that output is 2D (Linear) or 4D (Conv2d) - or
    any other rank.

    Broadcasting aligns shapes from the right, by position, not by
    meaning: a `(B, 1)` mask multiplied directly against a `(B, C, H, W)`
    output would align its `B` against `W`, not against the output's own
    batch dimension - silently wrong whenever a batch size happens to
    equal one of the target's other dimensions, and a shape-mismatch
    error otherwise. Reshaping the mask to `target`'s exact number of
    dimensions first (keeping its own batch dimension in the first slot
    and padding the rest with size-1 dims) makes broadcasting stretch
    those into every remaining axis correctly, no matter what kind of
    layer produced `target`.
    """
    return mask.reshape(mask.size(0), *([1] * (target.dim() - 1)))


class SEQ_Gate_V1(nn.Module):
    """Non-recurrent gate: a single linear layer maps the pooled input to
    `num_classes` logits.

    Input: (batch, input_size) features. Output: (batch, num_classes)
    one-hot mask (straight-through Gumbel-softmax)."""
    def __init__(self, input_size, num_classes):
        super(SEQ_Gate_V1, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.input_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        out_one = self.dropout(x.view(batch_size, -1))

        out_one = self.fc1(out_one)
        out_one = F.gumbel_softmax(out_one, tau=1, hard=True)

        return out_one


class SEQ_Gate_V2(nn.Module):
    """Non-recurrent gate: two linear layers (hidden width 10) map the
    pooled input to `num_classes` logits.

    Input: (batch, input_size) features. Output: (batch, num_classes)
    one-hot mask (straight-through Gumbel-softmax)."""
    def __init__(self, input_size, num_classes):
        super(SEQ_Gate_V2, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.input_size, 10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        out_one = self.dropout(x.view(batch_size, -1))

        out_one = self.fc1(out_one)
        out_one = self.fc2(out_one)
        out_one = F.gumbel_softmax(out_one, tau=1, hard=True)

        return out_one


class SEQ_Gate_V3(nn.Module):
    """Non-recurrent gate: two linear layers (hidden width 20) map the
    pooled input to `num_classes` logits.

    Input: (batch, input_size) features. Output: (batch, num_classes)
    one-hot mask (straight-through Gumbel-softmax)."""
    def __init__(self, input_size, num_classes):
        super(SEQ_Gate_V3, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.input_size, 20)
        self.fc2 = nn.Linear(20, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        out_one = self.dropout(x.view(batch_size, -1))

        out_one = self.fc1(out_one)
        out_one = self.fc2(out_one)
        out_one = F.gumbel_softmax(out_one, tau=1, hard=True)

        return out_one


class SEQ_Gate_V4(nn.Module):
    """Non-recurrent gate: three linear layers (hidden widths 5 and 10)
    map the pooled input to `num_classes` logits.

    Input: (batch, input_size) features. Output: (batch, num_classes)
    one-hot mask (straight-through Gumbel-softmax)."""
    def __init__(self, input_size, num_classes):
        super(SEQ_Gate_V4, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.input_size, 5)
        self.fc2 = nn.Linear(5, 10)
        self.fc3 = nn.Linear(10, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        out_one = self.dropout(x.view(batch_size, -1))

        out_one = self.fc1(out_one)
        out_one = self.fc2(out_one)
        out_one = self.fc3(out_one)
        out_one = F.gumbel_softmax(out_one, tau=1, hard=True)

        return out_one


class SEQ_Gate_V5(nn.Module):
    """Non-recurrent gate: three linear layers (hidden widths 10 and 20)
    map the pooled input to `num_classes` logits.

    Input: (batch, input_size) features. Output: (batch, num_classes)
    one-hot mask (straight-through Gumbel-softmax)."""
    def __init__(self, input_size, num_classes):
        super(SEQ_Gate_V5, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.input_size, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        out_one = self.dropout(x.view(batch_size, -1))

        out_one = self.fc1(out_one)
        out_one = self.fc2(out_one)
        out_one = self.fc3(out_one)
        out_one = F.gumbel_softmax(out_one, tau=1, hard=True)

        return out_one


class SEQ_Gate_V6(nn.Module):
    """Non-recurrent gate: three linear layers (hidden widths 15 and 10)
    map the pooled input to `num_classes` logits.

    Input: (batch, input_size) features. Output: (batch, num_classes)
    one-hot mask (straight-through Gumbel-softmax)."""
    def __init__(self, input_size, num_classes):
        super(SEQ_Gate_V6, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.input_size, 15)
        self.fc2 = nn.Linear(15, 10)
        self.fc3 = nn.Linear(10, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        out_one = self.dropout(x.view(batch_size, -1))

        out_one = self.fc1(out_one)
        out_one = self.fc2(out_one)
        out_one = self.fc3(out_one)
        out_one = F.gumbel_softmax(out_one, tau=1, hard=True)

        return out_one


class RNNGate(nn.Module):
    """Recurrent (LSTM) gate that picks one discrete choice per sample.

    Inputs to the constructor: input_dim (embedding size), hidden_dim
    (LSTM width), rnn_type ('lstm'), gate_num_classes (number of choices).

    forward input: (batch, input_dim) pooled features; output: a
    (batch, gate_num_classes) one-hot mask in the forward pass, with a
    straight-through gradient to the softmax probabilities. The LSTM's
    hidden state persists across calls until `repackage_hidden()`; it must
    be reset via `init_hidden(batch_size)` at the start of each batch.
    """
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm', gate_num_classes=2):
        super(RNNGate, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'lstm':
            self.rnn_one = nn.LSTM(input_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden_one = None

        self.proj = nn.Linear(hidden_dim, gate_num_classes)

        self.prob = nn.Sigmoid()
        self.prob_layer = nn.Softmax()

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()))

    def repackage_hidden(self):
        self.hidden_one = repackage_hidden(self.hidden_one)

    def forward(self, x):
        batch_size = x.size(0)
        self.rnn_one.flatten_parameters()

        out_one, self.hidden_one = self.rnn_one(x.view(1, batch_size, -1), self.hidden_one)
        x_one = self.proj(out_one.squeeze())

        prob = self.prob_layer(x_one)
        x_one = prob.detach().cpu().numpy()

        hard = (x_one == x_one.max(axis=1)[:,None]).astype(int)
        hard = torch.from_numpy(hard)
        hard = hard.cuda()
        x_one = hard.float().detach() -\
                prob.detach() + prob

        return x_one


class SoftRNNGate(nn.Module):
    """Recurrent (LSTM) gate that outputs a continuous per-sample weight in
    [0, 1] for a single choice. Output is thresholded at 0.5 in eval mode.

    Inputs to the constructor: input_dim, hidden_dim, rnn_type ('lstm').
    forward input: (batch, input_dim); output: a tuple (gated value, raw
    sigmoid probability), both (batch, 1)."""
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm'):
        super(SoftRNNGate, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden = None

        self.proj = nn.Linear(hidden_dim, 1)
        self.prob = nn.Sigmoid()

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    def forward(self, x):
        batch_size = x.size(0)
        self.rnn.flatten_parameters()
        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)

        proj = self.proj(out.squeeze())
        prob = self.prob(proj)

        x = prob
        if not self.training:
            x = (x > 0.5).float()
        return x, prob
