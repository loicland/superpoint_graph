"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import numpy as np
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
    def __init__(self, nfeat, nf_conv, nf_fc, K=2, norm = 'batch', affine = True, n_group = 1):
        super(STNkD, self).__init__()

        modules = []
        for i in range(len(nf_conv)):
            modules.append(nn.Conv1d(nf_conv[i-1] if i>0 else nfeat, nf_conv[i], 1))
            if norm == 'batch':
                modules.append(nn.BatchNorm1d(nf_conv[i]))
            elif norm == 'layer':
                modules.append(nn.GroupNorm(1,nf_conv[i]))
            elif norm == 'group':
                 modules.append(nn.GroupNorm(n_group,nf_conv[i]))
            modules.append(nn.ReLU(True))
        self.convs = nn.Sequential(*modules)

        modules = []
        for i in range(len(nf_fc)):
            modules.append(nn.Linear(nf_fc[i-1] if i>0 else nf_conv[-1], nf_fc[i]))
            if norm == 'batch':
                modules.append(nn.BatchNorm1d(nf_fc[i]))
            elif norm == 'layer':
                modules.append(nn.GroupNorm(1,nf_fc[i]))
            elif norm == 'group':
                 modules.append(nn.GroupNorm(n_group,nf_fc[i]))
            modules.append(nn.ReLU(True))
        self.fcs = nn.Sequential(*modules)

        self.proj = nn.Linear(nf_fc[-1], K*K)
        nn.init.constant_(self.proj.weight, 0); nn.init.constant_(self.proj.bias, 0)
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
    def __init__(self, nf_conv, nf_fc, nf_conv_stn, nf_fc_stn, nfeat, nfeat_stn=2, nfeat_global=1, prelast_do=0.5, last_ac=False, is_res=False, norm = 'batch', affine = True, n_group = 1, last_bn = False):

        super(PointNet, self).__init__()
        torch.manual_seed(0)
        if nfeat_stn > 0:
            self.stn = STNkD(nfeat_stn, nf_conv_stn, nf_fc_stn, norm=norm, n_group = n_group)
        self.nfeat_stn = nfeat_stn
        
        modules = []
        for i in range(len(nf_conv)):
            modules.append(nn.Conv1d(nf_conv[i-1] if i>0 else nfeat, nf_conv[i], 1))
            if norm == 'batch':
                modules.append(nn.BatchNorm1d(nf_conv[i]))
            elif norm == 'layer':
                modules.append(nn.GroupNorm(1, nf_conv[i]))
            elif norm == 'group':
                 modules.append(nn.GroupNorm(n_group, nf_conv[i]))
            modules.append(nn.ReLU(True))
        
        # Initialization of BN parameters.
        
        self.convs = nn.Sequential(*modules)

        modules = []
        for i in range(len(nf_fc)):
            modules.append(nn.Linear(nf_fc[i-1] if i>0 else nf_conv[-1]+nfeat_global, nf_fc[i]))
            if i<len(nf_fc)-1 or last_ac:
                if norm == 'batch':
                    modules.append(nn.BatchNorm1d(nf_fc[i]))
                elif norm == 'layer':
                    modules.append(nn.GroupNorm(1,nf_fc[i]))
                elif norm == 'group':
                 modules.append(nn.GroupNorm(n_group,nf_fc[i]))
                modules.append(nn.ReLU(True))
            if i==len(nf_fc)-2 and prelast_do>0:
                modules.append(nn.Dropout(prelast_do))
        if is_res: #init with small number so that at first the residual pointnet is close to zero
            nn.init.normal_(modules[-1].weight, mean=0, std = 1e-2)
            nn.init.normal_(modules[-1].bias, mean=0, std = 1e-2)
        
        #if last_bn:
            #modules.append(nn.BatchNorm1d(nf_fc[-1]))
        
        self.fcs = nn.Sequential(*modules)

    def forward(self, input, input_global):
        if self.nfeat_stn > 0:
            T = self.stn(input[:,:self.nfeat_stn,:])
            xy_transf = torch.bmm(input[:,:2,:].transpose(1,2), T).transpose(1,2)
            input = torch.cat([xy_transf, input[:,2:,:]], 1)

        input = self.convs(input)
        input = nnf.max_pool1d(input, input.size(2)).squeeze(2)
        if input_global is not None:
            if len(input_global.shape)== 1 or input_global.shape[1]==1:
                input = torch.cat([input, input_global.view(-1,1)], 1)
            else:
                input = torch.cat([input, input_global], 1)
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
        with torch.no_grad():
            out = model.ptn(Variable(clouds), (clouds_global))
            if not model.training:
                out = Variable(out.data, requires_grad=model.training) # cut autograd
        if model.training:
            out = Variable(out.data, requires_grad=model.training)
        def bw_hook():
            out_v2 = model.ptn(Variable(clouds), Variable(clouds_global)) # re-run fw pass
            out_v2.backward(out.grad)

        self.bw_hook = bw_hook

        descriptors = Variable(out.data.new(clouds_flag.size(0), out.size(1)).fill_(0))
        descriptors.index_copy_(0, Variable(idx_valid), out)
        return descriptors
    
class LocalCloudEmbedder():
    """ Local PointNet
    """
    def __init__(self, args):
        self.nfeat_stn = args.ptn_nfeat_stn
        self.stn_as_global = args.stn_as_global
        
    def run_batch(self, model, clouds, clouds_global, *excess):
        """ Evaluates all clouds in a differentiable way, use a batch approach.
        Use when embedding many small point clouds with small PointNets at once"""
        #cudnn cannot handle arrays larger than 2**16 in one go, uses batch
        batch_size = 2**16-1
        n_batches = int((clouds.shape[0]-1)/batch_size)
        if self.nfeat_stn > 0:
            T = model.stn(clouds[:batch_size,:self.nfeat_stn,:])
            for i in range(1,n_batches+1):
                T = torch.cat((T,model.stn(clouds[i * batch_size:(i+1) * batch_size,:self.nfeat_stn,:])))
            xy_transf = torch.bmm(clouds[:,:2,:].transpose(1,2), T).transpose(1,2)
            clouds = torch.cat([xy_transf, clouds[:,2:,:]], 1)
            if self.stn_as_global:
                clouds_global = torch.cat([clouds_global, T.view(-1,4)], 1)
        
        out = model.ptn(clouds[:batch_size,:,:], clouds_global[:batch_size,:])
        for i in range(1,n_batches+1):
            out = torch.cat((out,model.ptn(clouds[i * batch_size:(i+1) * batch_size,:,:], clouds_global[i * batch_size:(i+1) * batch_size,:])))
        return nnf.normalize(out)

    def run_batch_cpu(self, model, clouds, clouds_global, *excess):
        """ Evaluates the cloud on CPU, but put the values in the CPU as soon as they are computed"""
        #cudnn cannot handle arrays larger than 2**16 in one go, uses batch
        batch_size = 2**16-1
        n_batches = int(clouds.shape[0]/batch_size)
        emb_total = self.run_batch(model, clouds[:batch_size,:,:], clouds_global[:batch_size,:]).cpu()
        for i in range(1,n_batches+1):
            emb = self.run_batch(model, clouds[i * batch_size:(i+1) * batch_size,:,:], clouds_global[i * batch_size:(i+1) * batch_size,:])
            emb_total = torch.cat((emb_total,emb.cpu()))
            print("%d / %d %d / %d " % (i, n_batches, emb_total.shape[0], clouds.shape[0]))
        return emb_total

