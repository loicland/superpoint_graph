"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable
from learning import ecc

HAS_PYG = False
try:
    from torch_geometric.nn.conv import MessagePassing
    from torch_geometric.nn.inits import uniform
    HAS_PYG = True
except:
    pass

if HAS_PYG:
    class NNConv(MessagePassing):
        r"""The continuous kernel-based convolutional operator from the
        `"Neural Message Passing for Quantum Chemistry"
        <https://arxiv.org/abs/1704.01212>`_ paper.
        This convolution is also known as the edge-conditioned convolution from the
        `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
        Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
        :class:`torch_geometric.nn.conv.ECConv` for an alias):

        .. math::
            \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
            \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
            h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

        where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
        a MLP.

        Args:
            in_channels (int): Size of each input sample.
            out_channels (int): Size of each output sample.
            nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
                maps edge features :obj:`edge_attr` of shape :obj:`[-1,
                num_edge_features]` to shape
                    :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
                    :class:`torch.nn.Sequential`.
                aggr (string, optional): The aggregation scheme to use
                    (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
                    (default: :obj:`"add"`)
                root_weight (bool, optional): If set to :obj:`False`, the layer will
                    not add the transformed root node features to the output.
                    (default: :obj:`True`)
                bias (bool, optional): If set to :obj:`False`, the layer will not learn
                    an additive bias. (default: :obj:`True`)
                **kwargs (optional): Additional arguments of
                    :class:`torch_geometric.nn.conv.MessagePassing`.
            """
        def __init__(self,
                    in_channels,
                    out_channels,
                    aggr='mean',
                    root_weight=False,
                    bias=False,
                    vv=True,
                    flow="target_to_source",
                    negative_slope=0.2,
                    softmax=False,
                    **kwargs):
            super(NNConv, self).__init__(aggr=aggr, **kwargs)

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.aggr = aggr
            self.vv = vv
            self.negative_slope = negative_slope
            self.softmax = softmax

            if root_weight:
                self.root = Parameter(torch.Tensor(in_channels, out_channels))
            else:
                self.register_parameter('root', None)

            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)

            self.reset_parameters()

        def reset_parameters(self):
            uniform(self.in_channels, self.root)
            uniform(self.in_channels, self.bias)

        def forward(self, x, edge_index, weights):
            """"""
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            return self.propagate(edge_index, x=x, weights=weights)

        def message(self, edge_index_i, x_j, size_i, weights):
            if not self.vv:
                weight = weights.view(-1, self.in_channels, self.out_channels)
                if self.softmax: # APPLY A TWO DIMENSIONAL NON-DEPENDENT SPARSE SOFTMAX
                    weight = F.leaky_relu(weight, self.negative_slope)
                    weight = torch.cat([softmax(weight[:, k, :], edge_index_i, size_i).unsqueeze(1) for k in range(self.out_channels)], dim=1)
                return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)
            else:
                weight = weights.view(-1, self.in_channels)
                if self.softmax:
                    weight = F.leaky_relu(weight, self.negative_slope)
                    weight = torch.cat([softmax(w.unsqueeze(-1), edge_index_i, size_i).t() for w in weight.t()], dim=0).t()
                return x_j *  weight

        def update(self, aggr_out, x):
            if self.root is not None:
                aggr_out = aggr_out + torch.mm(x, self.root)
            if self.bias is not None:
                aggr_out = aggr_out + self.bias
            return aggr_out

        def __repr__(self):
            return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                        self.out_channels)


class RNNGraphConvModule(nn.Module):
    """
    Computes recurrent graph convolution using filter weights obtained from a Filter generating network (`filter_net`).
    Its result is passed to RNN `cell` and the process is repeated over `nrepeats` iterations.
    Weight sharing over iterations is done both in RNN cell and in Filter generating network.
    """
    def __init__(self, cell, filter_net, nfeat, vv = True, gc_info=None, nrepeats=1, cat_all=False, edge_mem_limit=1e20, use_pyg = True, cuda = True):
        super(RNNGraphConvModule, self).__init__()
        self._cell = cell
        self._isLSTM = 'LSTM' in type(cell).__name__
        self._fnet = filter_net
        self._nrepeats = nrepeats
        self._cat_all = cat_all
        self._edge_mem_limit = edge_mem_limit
        self.set_info(gc_info)
        self.use_pyg = use_pyg
        if use_pyg:
            self.nn = NNConv(nfeat, nfeat, vv = vv)
            if cuda:
                self.nn = self.nn.cuda()

    def set_info(self, gc_info):
        self._gci = gc_info

    def forward(self, hx):
        # get graph structure information tensors
        idxn, idxe, degs, degs_gpu, edgefeats = self._gci.get_buffers()

        edge_indexes = self._gci.get_pyg_buffers()
        ###edgefeats = Variable(edgefeats, requires_grad=False)

        # evalute and reshape filter weights (shared among RNN iterations)
        weights = self._fnet(edgefeats)
        nc = hx.size(1)
        assert hx.dim()==2 and weights.dim()==2 and weights.size(1) in [nc, nc*nc]
        if weights.size(1) != nc:
            weights = weights.view(-1, nc, nc)

        # repeatedly evaluate RNN cell
        hxs = [hx]
        if self._isLSTM:
            cx = Variable(hx.data.new(hx.size()).fill_(0))

        for r in range(self._nrepeats):
            if self.use_pyg:
                input = self.nn(hx, edge_indexes, weights)
            else:
                input = ecc.GraphConvFunction.apply(hx, weights, nc, nc, idxn, idxe, degs, degs_gpu,
                                                    self._edge_mem_limit)
            if self._isLSTM:
                hx, cx = self._cell(input, (hx, cx))
            else:
                hx = self._cell(input, hx)
            hxs.append(hx)

        return torch.cat(hxs,1) if self._cat_all else hx

class ECC_CRFModule(nn.Module):
    """
    Adapted "Conditional Random Fields as Recurrent Neural Networks" (https://arxiv.org/abs/1502.03240)
    `propagation` should be ECC with Filter generating network producing 2D matrix.
    """
    def __init__(self, propagation, nrepeats=1):
        super(ECC_CRFModule, self).__init__()
        self._propagation = propagation
        self._nrepeats = nrepeats

    def forward(self, input):
        Q = nnf.softmax(input)
        for i in range(self._nrepeats):
            Q = self._propagation(Q) # todo: speedup possible by sharing computation of fnet
            Q = input - Q
            if i < self._nrepeats-1:
                Q = nnf.softmax(Q) # last softmax will be part of cross-entropy loss
        return Q


class GRUCellEx(nn.GRUCell):
    """ Usual GRU cell extended with layer normalization and input gate.
    """
    def __init__(self, input_size, hidden_size, bias=True, layernorm=True, ingate=True):
        super(GRUCellEx, self).__init__(input_size, hidden_size, bias)
        self._layernorm = layernorm
        self._ingate = ingate
        if layernorm:
            self.add_module('ini', nn.InstanceNorm1d(1, eps=1e-5, affine=False, track_running_stats=False))
            self.add_module('inh', nn.InstanceNorm1d(1, eps=1e-5, affine=False, track_running_stats=False))
        if ingate:
            self.add_module('ig', nn.Linear(hidden_size, input_size, bias=True))

    def _normalize(self, gi, gh):
        if self._layernorm: # layernorm on input&hidden, as in https://arxiv.org/abs/1607.06450 (Layer Normalization)
            gi = self._modules['ini'](gi.unsqueeze(1)).squeeze(1)
            gh = self._modules['inh'](gh.unsqueeze(1)).squeeze(1)
        return gi, gh

    def forward(self, input, hidden):
        if self._ingate:
            input = torch.sigmoid(self._modules['ig'](hidden)) * input

        # GRUCell in https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py extended with layer normalization
        if input.is_cuda and torch.__version__.split('.')[0]=='0':
            gi = nnf.linear(input, self.weight_ih)
            gh = nnf.linear(hidden, self.weight_hh)
            gi, gh = self._normalize(gi, gh)
            state = torch.nn._functions.thnn.rnnFusedPointwise.GRUFused
            try: #pytorch >=0.3
                return state.apply(gi, gh, hidden) if self.bias_ih is None else state.apply(gi, gh, hidden, self.bias_ih, self.bias_hh)
            except: #pytorch <=0.2
                return state()(gi, gh, hidden) if self.bias_ih is None else state()(gi, gh, hidden, self.bias_ih, self.bias_hh)

        gi = nnf.linear(input, self.weight_ih)
        gh = nnf.linear(hidden, self.weight_hh)
        gi, gh = self._normalize(gi, gh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        bih_r, bih_i, bih_n = self.bias_ih.chunk(3)
        bhh_r, bhh_i, bhh_n = self.bias_hh.chunk(3)

        resetgate = torch.sigmoid(i_r + bih_r + h_r + bhh_r)
        inputgate = torch.sigmoid(i_i + bih_i + h_i + bhh_i)
        newgate = torch.tanh(i_n + bih_n + resetgate * (h_n + bhh_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def __repr__(self):
        s = super(GRUCellEx, self).__repr__() + '('
        if self._ingate:
            s += 'ingate'
        if self._layernorm:
            s += ' layernorm'
        return s + ')'


class LSTMCellEx(nn.LSTMCell):
    """ Usual LSTM cell extended with layer normalization and input gate.
    """
    def __init__(self, input_size, hidden_size, bias=True, layernorm=True, ingate=True):
        super(LSTMCellEx, self).__init__(input_size, hidden_size, bias)
        self._layernorm = layernorm
        self._ingate = ingate
        if layernorm:
            self.add_module('ini', nn.InstanceNorm1d(1, eps=1e-5, affine=False, track_running_stats=False))
            self.add_module('inh', nn.InstanceNorm1d(1, eps=1e-5, affine=False, track_running_stats=False))
        if ingate:
            self.add_module('ig', nn.Linear(hidden_size, input_size, bias=True))

    def _normalize(self, gi, gh):
        if self._layernorm: # layernorm on input&hidden, as in https://arxiv.org/abs/1607.06450 (Layer Normalization)
            gi = self._modules['ini'](gi.unsqueeze(1)).squeeze(1)
            gh = self._modules['inh'](gh.unsqueeze(1)).squeeze(1)
        return gi, gh

    def forward(self, input, hidden):
        if self._ingate:
            input = torch.sigmoid(self._modules['ig'](hidden[0])) * input

        # GRUCell in https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py extended with layer normalization
        if input.is_cuda and torch.__version__.split('.')[0]=='0':
            gi = nnf.linear(input, self.weight_ih)
            gh = nnf.linear(hidden[0], self.weight_hh)
            gi, gh = self._normalize(gi, gh)
            state = torch.nn._functions.thnn.rnnFusedPointwise.LSTMFused
            try: #pytorch >=0.3
                return state.apply(gi, gh, hidden[1]) if self.bias_ih is None else state.apply(gi, gh, hidden[1], self.bias_ih, self.bias_hh)
            except: #pytorch <=0.2
                return state()(gi, gh, hidden[1]) if self.bias_ih is None else state()(gi, gh, hidden[1], self.bias_ih, self.bias_hh)

        gi = nnf.linear(input, self.weight_ih, self.bias_ih)
        gh = nnf.linear(hidden[0], self.weight_hh, self.bias_hh)
        gi, gh = self._normalize(gi, gh)

        ingate, forgetgate, cellgate, outgate = (gi+gh).chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * hidden[1]) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, cy

    def __repr__(self):
        s = super(LSTMCellEx, self).__repr__() + '('
        if self._ingate:
            s += 'ingate'
        if self._layernorm:
            s += ' layernorm'
        return s + ')'
