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
from torch.autograd import Variable, gradcheck

from .GraphPoolModule import *
from .GraphPoolInfo import GraphPoolInfo


class TestGraphConvModule(unittest.TestCase):

    def test_gradcheck(self):

        torch.set_default_tensor_type('torch.DoubleTensor') #necessary for proper numerical gradient
        
        for cuda in range(0,2):
            for aggr in range(0,2):
                n,in_channels = 20,10
                input = torch.randn(n,in_channels)
                idxn = torch.from_numpy(np.random.permutation(n))
                degs = torch.LongTensor([2, 0, 3, 10, 5])
                degs_gpu = degs
                edge_mem_limit = 30 # some nodes will be combined, some not
                if cuda:
                    input = input.cuda(); idxn = idxn.cuda(); degs_gpu = degs_gpu.cuda()
                
                func = GraphPoolFunction(idxn, degs, degs_gpu, aggr=aggr, edge_mem_limit=edge_mem_limit)
                data = (Variable(input, requires_grad=True),)

                ok = gradcheck(func, data)
                self.assertTrue(ok)
        
        torch.set_default_tensor_type('torch.FloatTensor')
        
    def test_batch_splitting(self):
        n,in_channels = 20,10
        input = torch.randn(n,in_channels)
        idxn = torch.from_numpy(np.random.permutation(n))
        degs = torch.LongTensor([2, 0, 3, 10, 5])
        
        func = GraphPoolFunction(idxn, degs, degs, aggr=GraphPoolFunction.AGGR_MAX, edge_mem_limit=1e10)
        data = (Variable(input, requires_grad=True),)
        output1 = func(*data)

        func = GraphPoolFunction(idxn, degs, degs, aggr=GraphPoolFunction.AGGR_MAX, edge_mem_limit=1)
        output2 = func(*data)

        self.assertLess((output1-output2).norm(), 1e-6)        
          
if __name__ == '__main__':
    unittest.main()        