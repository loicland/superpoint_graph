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


class GraphPoolInfo(object):          
    """ Holds information about pooling in a vectorized form useful to `GraphPoolModule`. 
    
    We assume that the node feature tensor (given to `GraphPoolModule` as input) is ordered by igraph vertex id, e.g. the fifth row corresponds to vertex with id=4. Batch processing is realized by concatenating all graphs into a large graph of disconnected components (and all node feature tensors into a large tensor).
    """
    
    def __init__(self, *args, **kwargs):
        self._idxn = None           #indices into input tensor of convolution (node features)
        self._degrees = None        #in-degrees of output nodes (slices _idxn)
        self._degrees_gpu = None
        if len(args)>0 or len(kwargs)>0:
            self.set_batch(*args, **kwargs)
            
    def set_batch(self, poolmaps, graphs_from, graphs_to):
        """ Creates a representation of a given batch of graph poolings.
        
        Parameters:
        poolmaps: dict(s) mapping vertex id in coarsened graph to a list of vertex ids in input graph (defines pooling)
        graphs_from: input graph(s)
        graphs_to: coarsened graph(s)
        """
    
        poolmaps = poolmaps if isinstance(poolmaps,(list,tuple)) else [poolmaps]
        graphs_from = graphs_from if isinstance(graphs_from,(list,tuple)) else [graphs_from]
        graphs_to = graphs_to if isinstance(graphs_to,(list,tuple)) else [graphs_to]
        
        idxn = []
        degrees = []   
        p = 0        
              
        for map, G_from, G_to in zip(poolmaps, graphs_from, graphs_to):
            for v in range(G_to.vcount()):
                nlist = map.get(v, [])
                idxn.extend([n+p for n in nlist])
                degrees.append(len(nlist))
            p += G_from.vcount()
         
        self._idxn = torch.LongTensor(idxn)
        self._degrees = torch.LongTensor(degrees)
        self._degrees_gpu = None  
        
    def cuda(self):
        self._idxn = self._idxn.cuda()
        self._degrees_gpu = self._degrees.cuda()
        
    def get_buffers(self):
        """ Provides data to `GraphPoolModule`.
        """    
        return self._idxn, self._degrees, self._degrees_gpu
 