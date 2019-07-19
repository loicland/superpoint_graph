#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:56:33 2018

@author: landrieuloic

"""
import os
import sys
import math
import numpy as np
import torch

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, '..'))
sys.path.append(os.path.join(DIR_PATH,"../partition/cut-pursuit/src"))

from partition.provider import *
from partition.ply_c import libply_c 

import libcp

def zhang(x, lam, dist_type):
    if dist_type == 'euclidian' or dist_type == 'scalar':
       beta = 1
    elif dist_type == 'intrinsic':
        beta = 1.0471975512
    return torch.clamp(-lam * x + lam * beta, min = 0)

def compute_dist(embeddings, edg_source, edg_target, dist_type):
    if dist_type == 'euclidian':
        dist = ((embeddings[edg_source,:] - embeddings[edg_target,:])**2).sum(1)
    elif dist_type == 'intrinsic':
        smoothness = 0.999
        dist = (torch.acos((embeddings[edg_source,:] * embeddings[edg_target,:]).sum(1) * smoothness)-np.arccos(smoothness)) \
           / (np.arccos(-smoothness)-np.arccos(smoothness)) * 3.141592
    elif dist_type == 'scalar':
        dist = (embeddings[edg_source,:] * embeddings[edg_target,:]).sum(1)-1
    else:
        raise ValueError(" %s is an unknown argument of parameter --dist_type" % (dist_type))
    return dist

def compute_loss(args, diff, is_transition, weights_loss):
    intra_edg = is_transition==0
    if 'tv' in args.loss:
        loss1 =  (weights_loss[intra_edg] * (torch.sqrt(diff[intra_edg]+1e-10))).sum()
    elif 'laplacian' in args.loss:
        loss1 =  (weights_loss[intra_edg] * (diff[intra_edg])).sum()
    elif 'TVH' in args.loss:
        delta = 0.2
        loss1 =  delta * (weights_loss[intra_edg] * (torch.sqrt(1+diff[intra_edg]/delta**2)-1)).sum()
    else:
        raise ValueError(" %s is an unknown argument of parameter --loss" % (args.loss))
        
    inter_edg = is_transition==1
    
    if 'zhang' in args.loss:
        loss2 = (zhang(torch.sqrt(diff[inter_edg]+1e-10), weights_loss[inter_edg], args.dist_type)).sum()
    elif 'TVminus' in args.loss:
        loss2 = (torch.sqrt(diff[inter_edg]+1e-10) * weights_loss[inter_edg]).sum()
 
    #return loss1/ weights_loss.sum(), loss2/ weights_loss.sum()
    return loss1, loss2


def compute_partition(args, embeddings, edg_source, edg_target, diff, xyz=0):
    edge_weight = np.ones_like(edg_source).astype('f4')
    if args.edge_weight_threshold>0:
        edge_weight[diff>1]=args.edge_weight_threshold
    if args.edge_weight_threshold<0:
        edge_weight = torch.exp(diff * args.edge_weight_threshold).detach().cpu().numpy()/np.exp(args.edge_weight_threshold)

    ver_value = np.zeros((embeddings.shape[0],0), dtype='f4')
    use_spatial = 0
    ver_value = np.hstack((ver_value,embeddings.detach().cpu().numpy()))
    if args.spatial_emb>0:
        ver_value = np.hstack((ver_value, args.spatial_emb * xyz))# * math.sqrt(args.reg_strength)))
        #ver_value = xyz * args.spatial_emb
        use_spatial = 1#!!!
        
    pred_components, pred_in_component = libcp.cutpursuit(ver_value, \
        edg_source.astype('uint32'), edg_target.astype('uint32'), edge_weight, \
        args.reg_strength / (4 * args.k_nn_adj), cutoff=args.CP_cutoff, spatial = use_spatial, weight_decay = 0.7)
    #emb2 = libcp.cutpursuit2(ver_value, edg_source.astype('uint32'), edg_target.astype('uint32'), edge_weight, args.reg_strength, cutoff=0, spatial =0)
    #emb2 = emb2.reshape(ver_value.shape)
    #((ver_value-emb2)**2).sum(0)
    #cut = pred_in_component[edg_source]!=pred_in_component[edg_target]
    return pred_components, pred_in_component

def compute_weight_loss(args, embeddings, objects, edg_source, edg_target, is_transition, diff, return_partition, xyz=0):
    
    if args.loss_weight == 'seal' or args.loss_weight == 'crosspartition' or return_partition:
        pred_components, pred_in_component = compute_partition(args, embeddings, edg_source, edg_target, diff, xyz)

    if args.loss_weight=='none':
        weights_loss = np.ones_like(edg_target).astype('f4')
    elif args.loss_weight=='proportional':
        weights_loss = np.ones_like(edg_target).astype('f4') * float(len(is_transition)) / (1-is_transition).sum().float()
        weights_loss[is_transition.nonzero()] = float(len(is_transition)) / float(is_transition.sum()) * args.transition_factor
        weights_loss = weights_loss.cpu().numpy()
    elif args.loss_weight=='seal':
        weights_loss = compute_weights_SEAL(pred_components, pred_in_component, objects, edg_source, edg_target, is_transition, args.transition_factor)
    elif args.loss_weight=='crosspartition':
        weights_loss = compute_weights_XPART(pred_components, pred_in_component, objects.cpu().numpy(), edg_source, edg_target, is_transition.cpu().numpy(), args.transition_factor * 2 * args.k_nn_adj, xyz)
    else:
        raise ValueError(" %s is an unknown argument of parameter --loss" % (args.loss_weight))       

    if args.cuda:
        weights_loss = torch.from_numpy(weights_loss).cuda()
    else:
        weights_loss = torch.from_numpy(weights_loss)
    
    if return_partition:
        return weights_loss, pred_components, pred_in_component
    else:
        return weights_loss

def compute_weights_SEAL(pred_components, pred_in_component, objects, edg_source, edg_target, is_transition, transition_factor):
    
    SEAL_weights = np.ones((len(edg_source),), dtype='float32')
    w_per_component = np.empty((len(pred_components),), dtype='uint32')
    for i_com in range(len(pred_components)):
        w_per_component[i_com] = len(pred_components[i_com]) - mode(objects[pred_components[i_com]], only_frequency=True)
    SEAL_weights[is_transition.nonzero()] += np.stack(\
                (w_per_component[pred_in_component[edg_source[is_transition.nonzero()]]]
            ,   w_per_component[pred_in_component[edg_target[is_transition.nonzero()]]])).max(0)  * transition_factor# 1 if not transition 1+w otherwise
    return SEAL_weights

def compute_weights_XPART(pred_components, pred_in_component, objects, edg_source, edg_target, is_transition, transition_factor, xyz):

    SEAGL_weights = np.ones((len(edg_source),), dtype='float32')
    pred_transition = pred_in_component[edg_source]!=pred_in_component[edg_target]
    components_x, in_component_x = libply_c.connected_comp(pred_in_component.shape[0] \
           , edg_source.astype('uint32'), edg_target.astype('uint32') \
           , (is_transition+pred_transition==0).astype('uint8'), 0)
    
    edg_transition = is_transition.nonzero()[0]
    edg_source_trans = edg_source[edg_transition]
    edg_target_trans = edg_target[edg_transition]
    
    comp_x_weight = [len(c) for c in components_x]
    n_compx = len(components_x)
    
    edg_id = np.min((in_component_x[edg_source_trans],in_component_x[edg_target_trans]),0) * n_compx \
           + np.max((in_component_x[edg_source_trans],in_component_x[edg_target_trans]),0)
    
    edg_id_unique , in_edge_id, sedg_weight = np.unique(edg_id, return_index=True, return_counts=True)
    
    for i_edg in range(len(in_edge_id)):
            i_com_1 = in_component_x[edg_source_trans[in_edge_id[i_edg]]]
            i_com_2 = in_component_x[edg_target_trans[in_edge_id[i_edg]]]
            weight = min(comp_x_weight[i_com_1], comp_x_weight[i_com_2]) \
                   / sedg_weight[i_edg] * transition_factor
            corresponding_trans_edg = edg_transition[\
                ((in_component_x[edg_source_trans]==i_com_1) * (in_component_x[edg_target_trans]==i_com_2) \
              + (in_component_x[edg_target_trans]==i_com_1) * (in_component_x[edg_source_trans]==i_com_2))]
            SEAGL_weights[corresponding_trans_edg] = SEAGL_weights[corresponding_trans_edg] + weight
        
    #missed_transition = ((is_transition==1)*(pred_transition==False)+(is_transition==0)*(pred_transition==True)).nonzero()[0]
    #missed_transition = ((is_transition==1)*(pred_transition==False)).nonzero()[0]
    #SEAGL_weights[missed_transition] = SEAGL_weights[missed_transition] * boosting_factor
    #scalar2ply('full_par.ply', xyz,pred_in_component)
    #scalar2ply('full_parX.ply', xyz,in_component_x)
    #edge_weight2ply2('w.ply', SEAGL_weights, xyz, edg_source, edg_target)
    return SEAGL_weights

def mode(array, only_frequency=False):
    """compute the mode and the corresponding frequency of a given distribution"""
    u, counts = np.unique(array, return_counts=True)
    if only_frequency: return np.amax(counts)
    else:
        return u[np.argmax(counts)], np.amax(counts)
    
def relax_edge_binary(edg_binary, edg_source, edg_target, n_ver, tolerance):
    if torch.is_tensor(edg_binary):
        relaxed_binary = edg_binary.cpu().numpy().copy()
    else:
        relaxed_binary = edg_binary.copy()
    transition_vertex = np.full((n_ver,), 0, dtype = 'uint8')
    for i_tolerance in range(tolerance):
        transition_vertex[edg_source[relaxed_binary.nonzero()]] = True
        transition_vertex[edg_target[relaxed_binary.nonzero()]] = True
        relaxed_binary[transition_vertex[edg_source]] = True
        relaxed_binary[transition_vertex[edg_target]>0] = True
    return relaxed_binary

