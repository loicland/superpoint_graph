"""
    Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs
    https://github.com/mys007/ecc
    https://arxiv.org/abs/1704.02901
    2017 Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, gradcheck

from .GraphConvModule import *
from .GraphConvInfo import GraphConvInfo


class TestGraphConvModule(unittest.TestCase):

    def test_gradcheck(self):
    
        torch.set_default_tensor_type('torch.DoubleTensor') #necessary for proper numerical gradient    

        for cuda in range(0,2):
            # without idxe
            n,e,in_channels, out_channels = 20,50,10, 15
            input = torch.randn(n,in_channels)
            weights = torch.randn(e,in_channels,out_channels)
            idxn = torch.from_numpy(np.random.randint(n,size=e))
            idxe = None
            degs = torch.LongTensor([5, 0, 15, 20, 10])  #strided conv
            degs_gpu = degs
            edge_mem_limit = 30 # some nodes will be combined, some not
            if cuda:
                input = input.cuda(); weights = weights.cuda(); idxn = idxn.cuda(); degs_gpu = degs_gpu.cuda()
            
            func = GraphConvFunction(in_channels, out_channels, idxn, idxe, degs, degs_gpu, edge_mem_limit=edge_mem_limit)
            data = (Variable(input, requires_grad=True), Variable(weights, requires_grad=True))

            ok = gradcheck(func, data)
            self.assertTrue(ok)
            
            # with idxe
            weights = torch.randn(30,in_channels,out_channels)
            idxe = torch.from_numpy(np.random.randint(30,size=e))
            if cuda:
                weights = weights.cuda(); idxe = idxe.cuda()

            func = GraphConvFunction(in_channels, out_channels, idxn, idxe, degs, degs_gpu, edge_mem_limit=edge_mem_limit)

            ok = gradcheck(func, data)
            self.assertTrue(ok)

        torch.set_default_tensor_type('torch.FloatTensor')
        
    def test_batch_splitting(self):
    
        n,e,in_channels, out_channels = 20,50,10, 15
        input = torch.randn(n,in_channels)
        weights = torch.randn(e,in_channels,out_channels)
        idxn = torch.from_numpy(np.random.randint(n,size=e))
        idxe = None
        degs = torch.LongTensor([5, 0, 15, 20, 10])  #strided conv
        
        func = GraphConvFunction(in_channels, out_channels, idxn, idxe, degs, degs, edge_mem_limit=1e10)
        data = (Variable(input, requires_grad=True), Variable(weights, requires_grad=True))
        output1 = func(*data)

        func = GraphConvFunction(in_channels, out_channels, idxn, idxe, degs, degs, edge_mem_limit=1)
        output2 = func(*data)

        self.assertLess((output1-output2).norm().data[0], 1e-6)

    
        
if __name__ == '__main__':
    unittest.main()        