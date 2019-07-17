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
import ecc


class RNNGraphConvModule(nn.Module):
    """
    Computes recurrent graph convolution using filter weights obtained from a Filter generating network (`filter_net`).
    Its result is passed to RNN `cell` and the process is repeated over `nrepeats` iterations.
    Weight sharing over iterations is done both in RNN cell and in Filter generating network.
    """
    def __init__(self, cell, filter_net, gc_info=None, nrepeats=1, cat_all=False, edge_mem_limit=1e20):
        super(RNNGraphConvModule, self).__init__()
        self._cell = cell
        self._isLSTM = 'LSTM' in type(cell).__name__
        self._fnet = filter_net
        self._nrepeats = nrepeats
        self._cat_all = cat_all
        self._edge_mem_limit = edge_mem_limit
        self.set_info(gc_info)

    def set_info(self, gc_info):
        self._gci = gc_info

    def forward(self, hx):
        # get graph structure information tensors
        idxn, idxe, degs, degs_gpu, edgefeats = self._gci.get_buffers()
        edgefeats = Variable(edgefeats, requires_grad=False)

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
            input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx, weights)
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
            input = nnf.sigmoid(self._modules['ig'](hidden)) * input

        # GRUCell in https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py extended with layer normalization
        if input.is_cuda:
            gi = nnf.linear(input, self.weight_ih)
            gh = nnf.linear(hidden, self.weight_hh)
            gi, gh = self._normalize(gi, gh)
            state = torch.nn._functions.thnn.rnnFusedPointwise.GRUFused
            try: #pytorch >=0.3
                return state.apply(gi, gh, hidden) if self.bias_ih is None else state.apply(gi, gh, hidden, self.bias_ih, self.bias_hh)
            except: #pytorch <=0.2
                return state()(gi, gh, hidden) if self.bias_ih is None else state()(gi, gh, hidden, self.bias_ih, self.bias_hh)

        gi = nnf.linear(input, self.weight_ih, self.bias_ih)
        gh = nnf.linear(hidden, self.weight_hh, self.bias_hh)
        gi, gh = self._normalize(gi, gh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = nnf.sigmoid(i_r + h_r)
        inputgate = nnf.sigmoid(i_i + h_i)
        newgate = nnf.tanh(i_n + resetgate * h_n)
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
            input = nnf.sigmoid(self._modules['ig'](hidden[0])) * input

        # GRUCell in https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py extended with layer normalization
        if input.is_cuda:
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
        ingate = nnf.sigmoid(ingate)
        forgetgate = nnf.sigmoid(forgetgate)
        cellgate = nnf.tanh(cellgate)
        outgate = nnf.sigmoid(outgate)

        cy = (forgetgate * hidden[1]) + (ingate * cellgate)
        hy = outgate * nnf.tanh(cy)
        return hy, cy

    def __repr__(self):
        s = super(LSTMCellEx, self).__repr__() + '('
        if self._ingate:
            s += 'ingate'
        if self._layernorm:
            s += ' layernorm'
        return s + ')'
