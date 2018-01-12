"""
    Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs
    https://github.com/mys007/ecc
    https://arxiv.org/abs/1704.02901
    2017 Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import torch

import ecc

def graph_info_collate_classification(batch, edge_func):
    """ Collates a list of dataset samples into a single batch. We assume that all samples have the same number of resolutions.
    
    Each sample is a tuple of following elements:
        features: 2D Tensor of node features
        classes: LongTensor of class ids
        graphs: list of graphs, each for one resolution
        pooldata: list of triplets, each for one resolution: (pooling map, finer graph, coarser graph)   
    """
    features, classes, graphs, pooldata = list(zip(*batch))
    graphs_by_layer = list(zip(*graphs))
    pooldata_by_layer = list(zip(*pooldata))
    
    features = torch.cat([torch.from_numpy(f) for f in features])
    if features.dim()==1: features = features.view(-1,1)
    
    classes = torch.LongTensor(classes)
    
    GIs, PIs = [], []    
    for graphs in graphs_by_layer:
        GIs.append( ecc.GraphConvInfo(graphs, edge_func) )
    for pooldata in pooldata_by_layer:
        PIs.append( ecc.GraphPoolInfo(*zip(*pooldata)) )  
       
    return features, classes, GIs, PIs
    
    
def unique_rows(data):
    """ Filters unique rows from a 2D np array and also returns inverse indices. Used for edge feature compaction. """
    # https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    uniq, indices = np.unique(data.view(data.dtype.descr * data.shape[1]), return_inverse=True)
    return uniq.view(data.dtype).reshape(-1, data.shape[1]), indices
    
def one_hot_discretization(feat, clip_min, clip_max, upweight):
    indices = np.clip(np.round(feat), clip_min, clip_max).astype(int).reshape((-1,))
    onehot = np.zeros((feat.shape[0], clip_max - clip_min + 1))
    onehot[np.arange(onehot.shape[0]), indices] = onehot.shape[1] if upweight else 1
    return onehot    
    
def get_edge_shards(degs, edge_mem_limit):
    """ Splits iteration over nodes into shards, approximately limited by `edge_mem_limit` edges per shard. 
    Returns a list of pairs indicating how many output nodes and edges to process in each shard."""
    d = degs if isinstance(degs, np.ndarray) else degs.numpy()
    cs = np.cumsum(d)
    cse = cs // edge_mem_limit
    _, cse_i, cse_c = np.unique(cse, return_index=True, return_counts=True)
    
    shards = []
    for b in range(len(cse_i)):
        numd = cse_c[b]
        nume = (cs[-1] if b==len(cse_i)-1 else cs[cse_i[b+1]-1]) - cs[cse_i[b]] + d[cse_i[b]]   
        shards.append( (int(numd), int(nume)) )
    return shards
