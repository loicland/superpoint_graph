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
import torch.nn.init as init
from learning import ecc
from learning.modules import RNNGraphConvModule, ECC_CRFModule, GRUCellEx, LSTMCellEx


def create_fnet(widths, orthoinit, llbias, bnidx=-1):
    """ Creates feature-generating network, a multi-layer perceptron.
    Parameters:
    widths: list of widths of layers (including input and output widths)
    orthoinit: whether to use orthogonal weight initialization
    llbias: whether to use bias in the last layer
    bnidx: index of batch normalization (-1 if not used)
    """
    fnet_modules = []
    for k in range(len(widths)-2):
        fnet_modules.append(nn.Linear(widths[k], widths[k+1]))
        if orthoinit: init.orthogonal_(fnet_modules[-1].weight, gain=init.calculate_gain('relu'))
        if bnidx==k: fnet_modules.append(nn.BatchNorm1d(widths[k+1]))
        fnet_modules.append(nn.ReLU(True))
    fnet_modules.append(nn.Linear(widths[-2], widths[-1], bias=llbias))
    if orthoinit: init.orthogonal_(fnet_modules[-1].weight)
    if bnidx==len(widths)-1: fnet_modules.append(nn.BatchNorm1d(fnet_modules[-1].weight.size(0)))
    return nn.Sequential(*fnet_modules)


class GraphNetwork(nn.Module):
    """ It is constructed in a flexible way based on `config` string, which contains sequence of comma-delimited layer definiton tokens layer_arg1_arg2_... See README.md for examples.
    """
    def __init__(self, config, nfeat, fnet_widths, fnet_orthoinit=True, fnet_llbias=True, fnet_bnidx=-1, edge_mem_limit=1e20, use_pyg = True, cuda = True):
        super(GraphNetwork, self).__init__()
        self.gconvs = []

        for d, conf in enumerate(config.split(',')):
            conf = conf.strip().split('_')

            if conf[0]=='f':    #Fully connected layer;  args: output_feats
                self.add_module(str(d), nn.Linear(nfeat, int(conf[1])))
                nfeat = int(conf[1])
            elif conf[0]=='b':  #Batch norm;             args: not_affine
                self.add_module(str(d), nn.BatchNorm1d(nfeat, eps=1e-5, affine=len(conf)==1))
            elif conf[0]=='r':  #ReLU;
                self.add_module(str(d), nn.ReLU(True))
            elif conf[0]=='d':  #Dropout;                args: dropout_prob
                self.add_module(str(d), nn.Dropout(p=float(conf[1]), inplace=False))

            elif conf[0]=='crf': #ECC-CRF;               args: repeats
                nrepeats = int(conf[1])

                fnet = create_fnet(fnet_widths + [nfeat*nfeat], fnet_orthoinit, fnet_llbias, fnet_bnidx)
                gconv = ecc.GraphConvModule(nfeat, nfeat, fnet, edge_mem_limit=edge_mem_limit)
                crf = ECC_CRFModule(gconv, nrepeats)
                self.add_module(str(d), crf)
                self.gconvs.append(gconv)

            elif conf[0]=='gru' or conf[0]=='lstm': #RNN-ECC     args: repeats, mv=False, layernorm=True, ingate=True, cat_all=True
                nrepeats = int(conf[1])
                vv = bool(int(conf[2])) if len(conf)>2 else True # whether ECC does matrix-value mult or element-wise mult
                layernorm = bool(int(conf[3])) if len(conf)>3 else True
                ingate = bool(int(conf[4])) if len(conf)>4 else True
                cat_all = bool(int(conf[5])) if len(conf)>5 else True

                fnet = create_fnet(fnet_widths + [nfeat**2 if not vv else nfeat], fnet_orthoinit, fnet_llbias, fnet_bnidx)
                if conf[0]=='gru':
                    cell = GRUCellEx(nfeat, nfeat, bias=True, layernorm=layernorm, ingate=ingate)
                else:
                    cell = LSTMCellEx(nfeat, nfeat, bias=True, layernorm=layernorm, ingate=ingate)
                gconv = RNNGraphConvModule(cell, fnet, nfeat, vv = vv, nrepeats=nrepeats, cat_all=cat_all, edge_mem_limit=edge_mem_limit, use_pyg = use_pyg, cuda = cuda)
                self.add_module(str(d), gconv)
                self.gconvs.append(gconv)
                if cat_all: nfeat *= nrepeats + 1

            elif len(conf[0])>0:
                raise NotImplementedError('Unknown module: ' + conf[0])


    def set_info(self, gc_infos, cuda):
        """ Provides convolution modules with graph structure information for the current batch.
        """
        gc_infos = gc_infos if isinstance(gc_infos,(list,tuple)) else [gc_infos]
        for i,gc in enumerate(self.gconvs):
            if cuda: gc_infos[i].cuda()
            gc.set_info(gc_infos[i])

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

