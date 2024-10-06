from collections import OrderedDict
from itertools import tee

import math
import torch.nn.functional as F

from utils import *


def name2nn(name):
    if name is None:
        return None
    elif isinstance(name, nn.Module):
        return name
    else:
        return name


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class SequentialNet(nn.Module):
    def __init__(
            self, hiddens,
            layer_fn=nn.Linear, bias=True,
            norm_fn=None,
            activation_fn=nn.ReLU,
            dropout=None):
        super().__init__()
        # hack to prevent cycle imports

        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)

        net = []

        for i, (f_in, f_out) in enumerate(pairwise(hiddens)):
            net.append((f"layer_{i}", layer_fn(f_in, f_out, bias=bias)))
            if norm_fn is not None:
                net.append((f"norm_{i}", norm_fn(f_out)))
            if dropout is not None:
                net.append((f"drop_{i}", nn.Dropout(dropout)))
            if activation_fn is not None:
                net.append((f"activation_{i}", activation_fn()))

        self.net = torch.nn.Sequential(OrderedDict(net))

    def forward(self, x):
        x = self.net.forward(x)
        return x




class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1) + self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        out = []
        if len(input.size()) == 3:
            support = input@self.weight
            out = adj @ support
        else:
            support = torch.mm(input, self.weight)
            out= torch.spmm(adj, support)

        if self.bias is not None:
            return out + self.bias
        else:
            return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.001, n_layers=3):
        super(GCN, self).__init__()

        self.n_layers = n_layers
        self.gc_start = GraphConvolution(nfeat, nhid, bias=False)
        hiddens_dims = [ nhid if i == 0 or i == 2*n_layers-1 else 2*nhid  for i in range(2*n_layers)]
        self.layers = torch.nn.ModuleList([GraphConvolution(hiddens_dims[i], hiddens_dims[i+1], bias=False) for i in range(0, 2*n_layers, 2)])
        self.gc_final = GraphConvolution(nhid, nclass, bias=True)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc_start(x, adj))
        for i in range(self.n_layers):
            x = F.dropout(x, p=self.dropout)
            x = F.relu(self.layers[i](x, adj))
        x = F.dropout(x, p=self.dropout)
        x = self.gc_final(x, adj)
        return x



class ConvNet(nn.Module):
    def __init__(
            self, output_size,
            conv_hiddens, linear_hiddens,
            conv_kwargs=None,
            state_shape=[15, 15], num_channels=1, out_embd=8, dropout_rate=0.05,
            activation_fn=nn.ReLU, norm_fn=None, bias=True, gcn=False, gcn_conv=False):
        super().__init__()
        self.gcn = gcn
        conv_hiddens = [32, 16]
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)
        self.input_shape = [num_channels] + state_shape
        self.gcn_conv = gcn_conv
        default_conv_kwargs = {
            "kernel_size": 3,
            "stride": 2,
            "padding": 0
        }

        if gcn:
            self.out_shape_embd = out_embd
            self.conv_net = GCN(nfeat=state_shape[0], nhid=28, nclass=self.out_shape_embd, dropout=dropout_rate,
                                n_layers=5)
            self.linear_input = self.out_shape_embd * state_shape[0]
            linear_input = [self.linear_input, linear_hiddens[-2], linear_hiddens[-1]]
        else:
            conv_kwargs = conv_kwargs or default_conv_kwargs

            def conv_fn(f_in, f_out, bias=True):
                return nn.Conv2d(f_in, f_out, bias=bias, **conv_kwargs)

            self.conv_net = SequentialNet(
                hiddens=[num_channels] + conv_hiddens,
                layer_fn=conv_fn,
                activation_fn=activation_fn,
                norm_fn=None,
                bias=bias)
            hw = state_shape[0]
            padding = conv_kwargs.get("padding", 0)
            ksize = conv_kwargs["kernel_size"]
            stride = conv_kwargs.get("stride", 1)
            for i in range(len(conv_hiddens)):
                hw = (hw + 2 * padding - (ksize - 1) - 1) // stride + 1
            linear_input_size = conv_hiddens[-1] * hw ** 2
            self.linear_input = linear_input_size
            linear_input = [linear_input_size] + linear_hiddens
        if self.gcn_conv:
            conv_kwargs = conv_kwargs or default_conv_kwargs

            def conv_fn(f_in, f_out, bias=True):
                return nn.Conv2d(f_in, f_out, bias=bias, **conv_kwargs)

            self.conv_net2 = nn.Sequential(
                nn.Conv1d(state_shape[0], 32, 3, stride=2),
                nn.ReLU(),
                nn.Conv1d(32, 32, 2, stride=1),
                nn.Tanh(),
            )
            linear_input_size = 32 * 3
            self.linear_input = linear_input_size
            linear_input = [linear_input_size] + linear_hiddens

        self.linear_net = SequentialNet(
            hiddens=linear_input,
            layer_fn=nn.Linear,
            activation_fn=activation_fn,
            norm_fn=None,
            bias=bias)
        self.policy_net = SequentialNet(
            hiddens=[linear_hiddens[-1], output_size],
            layer_fn=nn.Linear,
            activation_fn=None,
            norm_fn=None,
            bias=True)
        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        self.conv_net.apply(inner_init)
        self.linear_net.apply(inner_init)
        self.policy_net.apply(out_init)

    def forward(self, states, x=None):
        if x is None and not self.gcn:
            # use simple convolution if X is None
            if len(states.shape) == 3:
                states = states[..., None]
            x = states.permute(0, 3, 1, 2)
            x = self.conv_net(x)

        else:
            x = self.conv_net(x, states)  # .unsqueeze(0)#.unsqueeze(1).permute(0,1,3,2)
            if self.gcn_conv:
                if len(states.shape) == 2:
                    x = x.unsqueeze(0)
                    print(x.size())
                x = self.conv_net2(x)

        if len(states.shape) == 2 and (self.gcn):
            x = x.view(-1, self.linear_input)
        else:
            x = x.view(states.shape[0], -1)
        x = self.linear_net(x)
        logits = self.policy_net(x)
        return logits


class Actor(ConvNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, states, mask, X=None, with_dist=False, greedy=False):
        logits = super().forward(x=X, states=states)
        probs = F.softmax(logits, dim=-1)
        probs = (probs + 1e-6) * mask  # fix hint
        # probs = probs * mask
        probs = probs / probs.sum(dim=-1)[..., None]
        dist = torch.distributions.Categorical(probs=probs)
        if greedy:
            _, action = torch.max(probs, -1)
        else:
            action = dist.sample()
        if with_dist:
            return action, dist
        return action


class Critic(ConvNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, states, X=None, with_log_pi=False):
        values = super().forward(x=X, states=states)
        return values
