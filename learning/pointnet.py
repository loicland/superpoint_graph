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


class STNkD(nn.Module):
    """
    Spatial Transformer Net for PointNet, producing a KxK transformation matrix.
    Parameters:
      nfeat: number of input features
      nf_conv: list of layer widths of point embeddings (before maxpool)
      nf_fc: list of layer widths of joint embeddings (after maxpool)
    """
    def __init__(self, nfeat, nf_conv, nf_fc, K=2):
        super(STNkD, self).__init__()

        modules = []
        for i in range(len(nf_conv)):
            modules.append(nn.Conv1d(nf_conv[i-1] if i>0 else nfeat, nf_conv[i], 1))
            modules.append(nn.BatchNorm1d(nf_conv[i]))
            modules.append(nn.ReLU(True))
        self.convs = nn.Sequential(*modules)

        modules = []
        for i in range(len(nf_fc)):
            modules.append(nn.Linear(nf_fc[i-1] if i>0 else nf_conv[-1], nf_fc[i]))
            modules.append(nn.BatchNorm1d(nf_fc[i]))
            modules.append(nn.ReLU(True))
        self.fcs = nn.Sequential(*modules)

        self.proj = nn.Linear(nf_fc[-1], K*K)
        nn.init.constant(self.proj.weight, 0); nn.init.constant(self.proj.bias, 0)
        self.eye = torch.eye(K).unsqueeze(0)

    def forward(self, input):
        self.eye = self.eye.cuda() if input.is_cuda else self.eye
        input = self.convs(input)
        input = nnf.max_pool1d(input, input.size(2)).squeeze(2)
        input = self.fcs(input)
        input = self.proj(input)
        return input.view(-1,self.eye.size(1),self.eye.size(2)) + Variable(self.eye)

class PointNet(nn.Module):
    """
    PointNet with only one spatial transformer and additional "global" input concatenated after maxpool.
    Parameters:
      nf_conv: list of layer widths of point embeddings (before maxpool)
      nf_fc: list of layer widths of joint embeddings (after maxpool)
      nfeat: number of input features
      nf_conv_stn, nf_fc_stn, nfeat_stn: as above but for Spatial transformer
      nfeat_global: number of features concatenated after maxpooling
      prelast_do: dropout after the pre-last parameteric layer
      last_ac: whether to use batch norm and relu after the last parameteric layer
    """
    def __init__(self, nf_conv, nf_fc, nf_conv_stn, nf_fc_stn, nfeat, nfeat_stn=2, nfeat_global=1, prelast_do=0.5, last_ac=False):
        super(PointNet, self).__init__()
        self.stn = STNkD(nfeat_stn, nf_conv_stn, nf_fc_stn)
        self.nfeat_stn = nfeat_stn

        modules = []
        for i in range(len(nf_conv)):
            modules.append(nn.Conv1d(nf_conv[i-1] if i>0 else nfeat, nf_conv[i], 1))
            modules.append(nn.BatchNorm1d(nf_conv[i]))
            modules.append(nn.ReLU(True))
        self.convs = nn.Sequential(*modules)

        modules = []
        for i in range(len(nf_fc)):
            modules.append(nn.Linear(nf_fc[i-1] if i>0 else nf_conv[-1]+nfeat_global, nf_fc[i]))
            if i<len(nf_fc)-1 or last_ac:
                modules.append(nn.BatchNorm1d(nf_fc[i]))
                modules.append(nn.ReLU(True))
            if i==len(nf_fc)-2 and prelast_do>0:
                modules.append(nn.Dropout(prelast_do))
        self.fcs = nn.Sequential(*modules)

    def forward(self, input, input_global):
        T = self.stn(input[:,:self.nfeat_stn,:])
        xy_transf = torch.bmm(input[:,:2,:].transpose(1,2), T).transpose(1,2)
        input = torch.cat([xy_transf, input[:,2:,:]], 1)

        input = self.convs(input)
        input = nnf.max_pool1d(input, input.size(2)).squeeze(2)
        if input_global is not None:
            input = torch.cat([input, input_global.view(-1,1)], 1)
        return self.fcs(input)



class CloudEmbedder():
    """ Evaluates PointNet on superpoints. Too small superpoints are assigned zero embeddings. Can optionally apply memory mongering
        (https://arxiv.org/pdf/1604.06174.pdf) to decrease memory usage.
    """
    def __init__(self, args):
        self.args = args
        self.bw_hook = lambda: None  # could be more elegant in the upcoming pytorch release: http://bit.ly/2A8PI7p
        self.run = self.run_full_monger if args.ptn_mem_monger else self.run_full

    def run_full(self, model, clouds_meta, clouds_flag, clouds, clouds_global):
        """ Simply evaluates all clouds in a differentiable way, assumes that all pointnet's feature maps fit into mem."""
        idx_valid = torch.nonzero(clouds_flag.eq(0)).squeeze()
        if self.args.cuda:
            clouds, clouds_global, idx_valid = clouds.cuda(), clouds_global.cuda(), idx_valid.cuda()
        clouds, clouds_global = Variable(clouds, volatile=not model.training), Variable(clouds_global, volatile=not model.training)
        #print('Ptn with', clouds.size(0), 'clouds')

        out = model.ptn(clouds, clouds_global)
        descriptors = Variable(out.data.new(clouds_flag.size(0), out.size(1)).fill_(0))
        descriptors.index_copy_(0, Variable(idx_valid), out)
        return descriptors

    def run_full_monger(self, model, clouds_meta, clouds_flag, clouds, clouds_global):
        """ Evaluates all clouds in forward pass, but uses memory mongering to compute backward pass."""
        idx_valid = torch.nonzero(clouds_flag.eq(0)).squeeze()
        if self.args.cuda:
            clouds, clouds_global, idx_valid = clouds.cuda(), clouds_global.cuda(), idx_valid.cuda()
        #print('Ptn with', clouds.size(0), 'clouds')

        out = model.ptn(Variable(clouds, volatile=True), Variable(clouds_global, volatile=True))
        out = Variable(out.data, requires_grad=model.training, volatile=not model.training) # cut autograd

        def bw_hook():
            out_v2 = model.ptn(Variable(clouds), Variable(clouds_global)) # re-run fw pass
            out_v2.backward(out.grad)

        self.bw_hook = bw_hook

        descriptors = Variable(out.data.new(clouds_flag.size(0), out.size(1)).fill_(0))
        descriptors.index_copy_(0, Variable(idx_valid), out)
        return descriptors

