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
from .GraphPoolInfo import GraphPoolInfo
from . import cuda_kernels
from . import utils

class GraphPoolFunction(Function):
    """ Computes node feature aggregation for each node of the coarsened graph. The evaluation is computed in blocks of size `edge_mem_limit` to reduce peak memory load. See `GraphPoolInfo` for info on `idxn, degs`.
    """

    AGGR_MEAN = 0
    AGGR_MAX = 1

    def __init__(self, idxn, degs, degs_gpu, aggr, edge_mem_limit=1e20):
        super(Function, self).__init__()
        self._idxn = idxn
        self._degs = degs
        self._degs_gpu = degs_gpu
        self._aggr = aggr
        self._shards = utils.get_edge_shards(degs, edge_mem_limit)
                
    def forward(self, input):
        output = input.new(self._degs.numel(), input.size(1))
        if self._aggr==GraphPoolFunction.AGGR_MAX:
            self._max_indices = self._idxn.new(self._degs.numel(), input.size(1)).fill_(-1)
        
        self._input_size = input.size()
        
        # loop over blocks of output nodes
        startd, starte = 0, 0
        for numd, nume in self._shards: 
            
            sel_input = torch.index_select(input, 0, self._idxn.narrow(0,starte,nume))
            
            # aggregate over nodes
            if self._idxn.is_cuda:
                if self._aggr==GraphPoolFunction.AGGR_MEAN:
                    cuda_kernels.avgpool_fw(output.narrow(0,startd,numd), sel_input, self._degs_gpu.narrow(0,startd,numd))            
                elif self._aggr==GraphPoolFunction.AGGR_MAX:
                    cuda_kernels.maxpool_fw(output.narrow(0,startd,numd), self._max_indices.narrow(0,startd,numd), sel_input, self._degs_gpu.narrow(0,startd,numd))        
            else:
                k = 0
                for i in range(startd, startd+numd):
                    if self._degs[i]>0:
                        if self._aggr==GraphPoolFunction.AGGR_MEAN:
                            torch.mean(sel_input.narrow(0,k,self._degs[i]), 0, out=output[i])
                        elif self._aggr==GraphPoolFunction.AGGR_MAX:
                            torch.max(sel_input.narrow(0,k,self._degs[i]), 0, out=(output[i], self._max_indices[i]))
                    else:
                        output[i].fill_(0)
                    k = k + self._degs[i]
                    
            startd += numd
            starte += nume 
            del sel_input
    
        return output

        
    def backward(self, grad_output):
        grad_input = grad_output.new(self._input_size).fill_(0)

        # loop over blocks of output nodes
        startd, starte = 0, 0
        for numd, nume in self._shards:                  
            
            grad_sel_input = grad_output.new(nume, grad_output.size(1))

            # grad wrt input
            if self._idxn.is_cuda:
                if self._aggr==GraphPoolFunction.AGGR_MEAN:
                    cuda_kernels.avgpool_bw(grad_input, self._idxn.narrow(0,starte,nume), grad_output.narrow(0,startd,numd), self._degs_gpu.narrow(0,startd,numd))            
                elif self._aggr==GraphPoolFunction.AGGR_MAX:
                    cuda_kernels.maxpool_bw(grad_input, self._idxn.narrow(0,starte,nume), self._max_indices.narrow(0,startd,numd), grad_output.narrow(0,startd,numd), self._degs_gpu.narrow(0,startd,numd))  
            else:
                k = 0
                for i in range(startd, startd+numd):
                    if self._degs[i]>0:
                        if self._aggr==GraphPoolFunction.AGGR_MEAN:
                            torch.div(grad_output[i], self._degs[i], out=grad_sel_input[k])
                            if self._degs[i]>1:
                                grad_sel_input.narrow(0, k+1, self._degs[i]-1).copy_( grad_sel_input[k].expand(self._degs[i]-1,1,grad_output.size(1)) )
                        elif self._aggr==GraphPoolFunction.AGGR_MAX:
                            grad_sel_input.narrow(0, k, self._degs[i]).fill_(0).scatter_(0, self._max_indices[i].view(1,-1), grad_output[i].view(1,-1))
                        k = k + self._degs[i]             

                grad_input.index_add_(0, self._idxn.narrow(0,starte,nume), grad_sel_input)
                    
            startd += numd
            starte += nume   
            del grad_sel_input
       
        return grad_input
        
        
        
class GraphPoolModule(nn.Module):
    """ Performs graph pooling.
        The input should be a 2D tensor of size (# nodes, `in_channels`). Multiple graphs can be concatenated in the same tensor (minibatch).    
    
    Parameters:
    aggr: aggregation type (GraphPoolFunction.AGGR_MEAN, GraphPoolFunction.AGGR_MAX)
    gp_info: GraphPoolInfo object containing node mapping information, can be also set with `set_info()` method.
    edge_mem_limit: block size (number of evaluated edges in parallel), a low value reduces peak memory.
    """
    
    def __init__(self, aggr, gp_info=None, edge_mem_limit=1e20):
        super(GraphPoolModule, self).__init__()
        
        self._aggr = aggr
        self._edge_mem_limit = edge_mem_limit       
        self.set_info(gp_info)
        
    def set_info(self, gp_info):
        self._gpi = gp_info
        
    def forward(self, input):       
        idxn, degs, degs_gpu = self._gpi.get_buffers()
        return GraphPoolFunction(idxn, degs, degs_gpu, self._aggr, self._edge_mem_limit)(input)
        
        
class GraphAvgPoolModule(GraphPoolModule):
    def __init__(self, gp_info=None, edge_mem_limit=1e20):
        super(GraphAvgPoolModule, self).__init__(GraphPoolFunction.AGGR_MEAN, gp_info, edge_mem_limit)        
        
class GraphMaxPoolModule(GraphPoolModule):
    def __init__(self, gp_info=None, edge_mem_limit=1e20):
        super(GraphMaxPoolModule, self).__init__(GraphPoolFunction.AGGR_MAX, gp_info, edge_mem_limit)                