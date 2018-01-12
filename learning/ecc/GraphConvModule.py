"""
    Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs
    https://github.com/mys007/ecc
    https://arxiv.org/abs/1704.02901
    2017 Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from .GraphConvInfo import GraphConvInfo
from . import cuda_kernels
from . import utils

class GraphConvFunction(Function):
    """ Computes operations for each edge and averages the results over respective nodes. The operation is either matrix-vector multiplication (for 3D weight tensors) or element-wise vector-vector multiplication (for 2D weight tensors). The evaluation is computed in blocks of size `edge_mem_limit` to reduce peak memory load. See `GraphConvInfo` for info on `idxn, idxe, degs`.
    """

    def __init__(self, in_channels, out_channels, idxn, idxe, degs, degs_gpu, edge_mem_limit=1e20):
        super(Function, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._idxn = idxn
        self._idxe = idxe
        self._degs = degs
        self._degs_gpu = degs_gpu
        self._shards = utils.get_edge_shards(degs, edge_mem_limit)

    def _multiply(self, a, b, out, f_a=None, f_b=None):
        """ Performs operation on edge weights and node signal """
        if self._full_weight_mat:
            # weights are full in_channels x out_channels matrices -> mm
            torch.bmm(f_a(a) if f_a else a, f_b(b) if f_b else b, out=out)
        else:
            # weights represent diagonal matrices -> mul
            torch.mul(a, b.expand_as(a), out=out)
       
    def forward(self, input, weights):
        self.save_for_backward(input, weights)

        self._full_weight_mat = weights.dim()==3
        assert self._full_weight_mat or (self._in_channels==self._out_channels and weights.size(1) == self._in_channels)

        output = input.new(self._degs.numel(), self._out_channels)       
        
        # loop over blocks of output nodes
        startd, starte = 0, 0
        for numd, nume in self._shards:            

            # select sequence of matching pairs of node and edge weights            
            sel_input = torch.index_select(input, 0, self._idxn.narrow(0,starte,nume))
            
            if self._idxe is not None:
                sel_weights = torch.index_select(weights, 0, self._idxe.narrow(0,starte,nume))
            else:
                sel_weights = weights.narrow(0,starte,nume)
                
            # compute matrix-vector products
            products = input.new()
            self._multiply(sel_input, sel_weights, products, lambda a: a.unsqueeze(1))

            # average over nodes
            if self._idxn.is_cuda:
                cuda_kernels.conv_aggregate_fw(output.narrow(0,startd,numd), products.view(-1,self._out_channels), self._degs_gpu.narrow(0,startd,numd))
            else:
                k = 0
                for i in range(startd, startd+numd):
                    if self._degs[i]>0:
                        torch.mean(products.narrow(0,k,self._degs[i]), 0, out=output[i])
                    else:
                        output[i].fill_(0)
                    k = k + self._degs[i]
 
            startd += numd
            starte += nume  
            del sel_input, sel_weights, products
        
        return output

    def backward(self, grad_output):
        input, weights = self.saved_tensors

        grad_input = input.new(input.size()).fill_(0)
        grad_weights = weights.new(weights.size())
        if self._idxe is not None: grad_weights.fill_(0)

        # loop over blocks of output nodes
        startd, starte = 0, 0
        for numd, nume in self._shards:         
            
            grad_products, tmp = input.new(nume, self._out_channels), input.new()

            if self._idxn.is_cuda:
                cuda_kernels.conv_aggregate_bw(grad_products, grad_output.narrow(0,startd,numd), self._degs_gpu.narrow(0,startd,numd))
            else:           
                k = 0
                for i in range(startd, startd+numd):
                    if self._degs[i]>0:
                        torch.div(grad_output[i], self._degs[i], out=grad_products[k])
                        if self._degs[i]>1:
                            grad_products.narrow(0, k+1, self._degs[i]-1).copy_( grad_products[k].expand(self._degs[i]-1,1,self._out_channels).squeeze(1) )
                        k = k + self._degs[i]    

            # grad wrt weights
            sel_input = torch.index_select(input, 0, self._idxn.narrow(0,starte,nume))
            
            if self._idxe is not None:
                self._multiply(sel_input, grad_products, tmp, lambda a: a.unsqueeze(1).transpose_(2,1), lambda b: b.unsqueeze(1))
                grad_weights.index_add_(0, self._idxe.narrow(0,starte,nume), tmp)
            else:
                self._multiply(sel_input, grad_products, grad_weights.narrow(0,starte,nume), lambda a: a.unsqueeze(1).transpose_(2,1), lambda b: b.unsqueeze(1))

            # grad wrt input
            if self._idxe is not None:
                torch.index_select(weights, 0, self._idxe.narrow(0,starte,nume), out=tmp)
                self._multiply(grad_products, tmp, sel_input, lambda a: a.unsqueeze(1), lambda b: b.transpose_(2,1))
                del tmp
            else:
                self._multiply(grad_products, weights.narrow(0,starte,nume), sel_input, lambda a: a.unsqueeze(1), lambda b: b.transpose_(2,1))

            grad_input.index_add_(0, self._idxn.narrow(0,starte,nume), sel_input)
                    
            startd += numd
            starte += nume  
            del grad_products, sel_input
       
        return grad_input, grad_weights



class GraphConvModule(nn.Module):
    """ Computes graph convolution using filter weights obtained from a filter generating network (`filter_net`).
        The input should be a 2D tensor of size (# nodes, `in_channels`). Multiple graphs can be concatenated in the same tensor (minibatch).
    
    Parameters:
    in_channels: number of input channels
    out_channels: number of output channels
    filter_net: filter-generating network transforming a 2D tensor (# edges, # edge features) to (# edges, in_channels*out_channels) or (# edges, in_channels)
    gc_info: GraphConvInfo object containing graph(s) structure information, can be also set with `set_info()` method.
    edge_mem_limit: block size (number of evaluated edges in parallel) for convolution evaluation, a low value reduces peak memory. 
    """

    def __init__(self, in_channels, out_channels, filter_net, gc_info=None, edge_mem_limit=1e20):
        super(GraphConvModule, self).__init__()
        
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._fnet = filter_net
        self._edge_mem_limit = edge_mem_limit
        
        self.set_info(gc_info)
        
    def set_info(self, gc_info):
        self._gci = gc_info
    
    def forward(self, input):       
        # get graph structure information tensors
        idxn, idxe, degs, degs_gpu, edgefeats = self._gci.get_buffers()
        edgefeats = Variable(edgefeats, requires_grad=False)
        
        # evalute and reshape filter weights
        weights = self._fnet(edgefeats)
        assert input.dim()==2 and weights.dim()==2 and (weights.size(1) == self._in_channels*self._out_channels or
               (self._in_channels == self._out_channels and weights.size(1) == self._in_channels))
        if weights.size(1) == self._in_channels*self._out_channels:
            weights = weights.view(-1, self._in_channels, self._out_channels)

        return GraphConvFunction(self._in_channels, self._out_channels, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(input, weights)
        





class GraphConvModulePureAutograd(nn.Module):
    """
    Autograd-only equivalent of `GraphConvModule` + `GraphConvFunction`. Unfortunately, autograd needs to store intermediate products, which makes the module work only for very small graphs. The module is kept for didactic purposes only.
    """

    def __init__(self, in_channels, out_channels, filter_net, gc_info=None):
        super(GraphConvModulePureAutograd, self).__init__()
        
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._fnet = filter_net
        
        self.set_info(gc_info)
        
    def set_info(self, gc_info):
        self._gci = gc_info

    def forward(self, input):
        # get graph structure information tensors
        idxn, idxe, degs, edgefeats = self._gci.get_buffers()
        idxn = Variable(idxn, requires_grad=False)
        edgefeats = Variable(edgefeats, requires_grad=False)
        
        # evalute and reshape filter weights
        weights = self._fnet(edgefeats)
        assert input.dim()==2 and weights.dim()==2 and weights.size(1) == self._in_channels*self._out_channels
        weights = weights.view(-1, self._in_channels, self._out_channels)
            
        # select sequence of matching pairs of node and edge weights            
        if idxe is not None:
            idxe = Variable(idxe, requires_grad=False)
            weights = torch.index_select(weights, 0, idxe)        
        
        sel_input = torch.index_select(input, 0, idxn)

        # compute matrix-vector products
        products = torch.bmm(sel_input.view(-1,1,self._in_channels), weights)
        
        output = Variable(input.data.new(len(degs), self._out_channels))
        
        # average over nodes
        k = 0
        for i in range(len(degs)):
            if degs[i]>0:
                output.index_copy_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), torch.mean(products.narrow(0,k,degs[i]), 0).view(1,-1))
            else:
                output.index_fill_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), 0)
            k = k + degs[i]

        return output
    
