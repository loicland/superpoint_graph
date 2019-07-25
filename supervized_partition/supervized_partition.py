# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:26:49 2018
@author: landrieuloic
"""
import os
import sys
import ast
import h5py
import numpy as np
import random

import time
import logging
import argparse   
import json
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchnet as tnt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, '..'))

from partition.ply_c import libply_c

from learning.pointnet import STNkD
from learning.pointnet import LocalCloudEmbedder
from learning.pointnet import PointNet

from supervized_partition.graph_processing import *
from partition.provider import embedding2ply
from partition.provider import scalar2ply
from partition.provider import edge_class2ply2
from partition.provider import perfect_prediction
from partition.provider import write_spg

from learning.metrics import compute_OOA
from learning.metrics import compute_boundary_precision
from learning.metrics import compute_boundary_recall

from supervized_partition.losses import compute_weight_loss
from supervized_partition.losses import compute_partition
from supervized_partition.losses import relax_edge_binary
from supervized_partition.losses import compute_loss
from supervized_partition.losses import compute_dist

from learning import metrics
from partition.graphs import compute_sp_graph
from learning.main import create_optimizer
from folderhierarchy import FolderHierachy

def parse_args():
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    # Dataset
    parser.add_argument('--dataset', default='s3dis', help='Dataset name: sema3d|s3dis|vkitti')
    parser.add_argument('--cvfold', default=1, type=int, help='Fold left-out for testing in leave-one-out setting (S3DIS)')
    parser.add_argument('--resume', default='', help='Loads a previously saved model.')
    parser.add_argument('--db_train_name', default='trainval', help='Training set (Sema3D)')
    parser.add_argument('--db_test_name', default='testred', help='Test set (Sema3D)')
    parser.add_argument('--ROOT_PATH', default='datasets/s3dis')
    parser.add_argument('--odir', default='results_emb/s3dis', help='folder for saving the trained model')
    parser.add_argument('--spg_out', default=1, type=int, help='wether to compute the SPG for linking with the SPG semantic segmentation method')
    
    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--nworkers', default=0, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--test_nth_epoch', default=10, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')
    parser.add_argument('--test_multisamp_n', default=10, type=int, help='Average logits obtained over runs with different seeds')
    # Optimization arguments
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.7, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default='[20,35,45]', help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=5, type=int, help='Batch size')
    parser.add_argument('--optim', default='adam', help='Optimizer: sgd|adam')
    parser.add_argument('--grad_clip', default=1, type=float, help='Element-wise clipping of gradient. If 0, does not clip')
    # Point cloud processing
    parser.add_argument('--pc_attribs', default='', help='Point attributes fed to PointNets, if empty then all possible.')
    parser.add_argument('--pc_augm_scale', default=2, type=float, help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=1, type=int, help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0, type=float, help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', default=1, type=int, help='Training augmentation: Bool, Gaussian jittering of all attributes')
    # Point net
    parser.add_argument('--ptn_embedding', default='ptn', help='configuration of the learned cloud emebdder (ptn): uses PointNets for vertices embeddings. no other options so far :)')
    parser.add_argument('--ptn_widths', default='[[32,128], [34,32,32,4]]', help='PointNet widths')
    parser.add_argument('--ptn_widths_stn', default='[[16,64],[32,16]]', help='PointNet\'s Transformer widths')
    parser.add_argument('--use_color', default='rgb', help='How to use color in the local cloud embedding : rgb, lab or no')
    parser.add_argument('--ptn_nfeat_stn', default=2, type=int, help='PointNet\'s Transformer number of input features')
    parser.add_argument('--ptn_prelast_do', default=0, type=float)
    parser.add_argument('--ptn_norm', default='batch', help='Type of norm layers in PointNets, "batch or "layer" or "group"')
    parser.add_argument('--ptn_n_group', default=2, type=int, help='Number of groups in groupnorm. Only compatible with ptn_norm=group')
    parser.add_argument('--stn_as_global', default=1, type=int, help='Wether to use the STN output as a global variable')
    parser.add_argument('--global_feat', default='eXYrgb', help='Use rgb to embed points')
    parser.add_argument('--use_rgb', default=1, type=int , help='Wether to use radiometry value to use for cloud embeding')
    parser.add_argument('--ptn_mem_monger', default=0, type=int, help='Bool, save GPU memory by recomputing PointNets in back propagation.')
    #parser.add_argument('--cascade_color_neigh', default=0, type=int, help='Wether to feed the color to the residual embeddings')

    #Loss
    parser.add_argument('--loss_weight', default='crosspartition', help='[none, proportional, sqrt, seal, crosspartition] which loss weighting scheme to choose to train the model. unweighted: use classic cross_entropy loss, proportional: weight inversely by transition count,  SEAL: use SEAL loss as proposed in http://jankautz.com/publications/LearningSuperpixels_CVPR2018.pdf, crosspartition : our crosspartition weighting scheme')
    parser.add_argument('--loss', default='TVH_zhang', help='Structure of the loss : first term for intra edge (chose from : tv, laplacian, TVH (pseudo-huber)), second one for interedge (chose from: zhang, scad, tv)')
    parser.add_argument('--transition_factor', default=5, type=float, help='Weight for transition edges in the graph structured contrastive loss')
    parser.add_argument('--dist_type', default='euclidian', help='[euclidian, intrisic, scalar] How to measure the distance between embeddings')

    #Graph-Clustering
    parser.add_argument('--ver_value', default='ptn', help='what value to use for vertices (ptn): uses PointNets, (geof) : uses geometric features, (xyz) uses position, (rgb) uses color')
    parser.add_argument('--max_ver_train', default=1e4, type=int, help='Size of the subgraph taken in each point cloud for the training')
    parser.add_argument('--k_nn_adj', default=5, type=int, help='number of neighbors for the adjacency graph')
    parser.add_argument('--k_nn_local', default=20, type=int, help='number of neighbors to describe the local geometry')
    parser.add_argument('--reg_strength', default=1, type = float, help='Regularization strength or the generalized minimum partition problem.')
    parser.add_argument('--CP_cutoff', default=10, type=int, help='Minimum accepted component size in cut pursuit. if negative, chose with respect tot his number and the reg_strength as explained in the paper')
    parser.add_argument('--spatial_emb', default=0.2, type=float, help='Weight of xyz in the spatial embedding. When 0 : no xyz')
    parser.add_argument('--edge_weight_threshold', default=-0.5, type=float, help='Edge weight value when diff>1. if negative, then switch to weight = exp(-diff * edge_weight_threshold)')

    #Metrics
    parser.add_argument('--BR_tolerance', default=1, type=int, help='How far an edge must be from an actual transition to be considered a true positive')

    args = parser.parse_args()

    args.start_epoch = 0
    args.lr_steps = ast.literal_eval(args.lr_steps)
    args.ptn_widths = ast.literal_eval(args.ptn_widths)
    args.ptn_widths_stn = ast.literal_eval(args.ptn_widths_stn)
    args.learned_embeddings = ('ptn' in args.ver_value) or args.ver_value == 'xyz' #wether we actually do some learning
    if args.CP_cutoff<0: #adaptive cutoff: strong regularization will set a larger cutoff
        args.CP_cutoff = int(max(-args.CP_cutoff/2, -args.CP_cutoff/2 * np.log(args.reg_strength) -args.CP_cutoff))

    return args

def dataset(args):
        # Decide on the dataset
    if args.dataset=='s3dis':
        dbinfo = get_s3dis_info(args)
        create_dataset = create_s3dis_datasets
    elif args.dataset=='sema3d':
        dbinfo = get_sema3d_info(args)
        create_dataset = create_sema3d_datasets
    elif args.dataset=='vkitti':
        dbinfo = get_vkitti_info(args)
        create_dataset = create_vkitti_datasets
    else:
        raise NotImplementedError('Unknown dataset ' + args.dataset)
    return dbinfo, create_dataset

def embed(args):
    random.seed(0)  
    root = args.ROOT_PATH+'/'
    folder_hierarchy = FolderHierachy(args.odir, args.dataset, root, args.cvfold)

    # Save command line arguments
    with open(os.path.join(folder_hierarchy.outputdir, 'cmdline.txt'), 'w') as f:
        f.write(" ".join(["'"+a+"'" if (len(a)==0 or a[0]!='-') else a for a in sys.argv]))
    
    if (args.dataset=='sema3d' and args.db_test_name.startswith('test')) or (args.dataset.startswith('s3dis_02') and args.cvfold==2):
        # very large graphs
        torch.backends.cudnn.enabled = False
    
    dbinfo, create_dataset = dataset(args) 

    # Create model and optimizer
    if args.resume != '':
        if args.resume=='RESUME': 
            if os.path.isfile(folder_hierarchy.model_path):
                args.resume = folder_hierarchy.model_path
            else:
                raise NameError('Cant find pretrained model')
        model, optimizer, stats = resume(args)
    else:
        model = create_model(args)
        optimizer = create_optimizer(args, model)
            
        stats = []
                
    train_dataset, test_dataset = create_dataset(args)
    
    if args.learned_embeddings and args.ptn_embedding == 'ptn':
        ptnCloudEmbedder = LocalCloudEmbedder(args)
    elif 'geof' in args.ver_value:
        ptnCloudEmbedder = spatialEmbedder(args)
    else:
        raise NameError('Do not know model ' + args.learned_embeddings)
        
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_decay, last_epoch=args.start_epoch-1)

    def train(i_epoch):
        """ Trains for one epoch """
        #return 0
        model.train()
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=graph_collate, num_workers=args.nworkers, shuffle=True, drop_last=True)
        
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=100)
    
        loss_meter = tnt.meter.AverageValueMeter()
        n_clusters_meter = tnt.meter.AverageValueMeter()

        t0 = time.time()
    
        for bidx, (fname, edg_source, edg_target, is_transition, labels, objects, clouds_data, xyz) in enumerate(loader):
            
            if args.cuda:
                is_transition = is_transition.to('cuda',non_blocking=True)
                #labels = torch.from_numpy(labels).cuda()
                objects = objects.to('cuda',non_blocking=True)
                clouds, clouds_global, nei = clouds_data
                clouds_data = (clouds.to('cuda',non_blocking=True),clouds_global.to('cuda',non_blocking=True),nei) 

            t_loader = 1000*(time.time()-t0)
            optimizer.zero_grad()
            t0 = time.time()

            embeddings = ptnCloudEmbedder.run_batch(model, *clouds_data, xyz)
            
            diff = compute_dist(embeddings, edg_source, edg_target, args.dist_type)
            
            weights_loss, pred_comp, in_comp = compute_weight_loss(args, embeddings, objects, edg_source, edg_target, is_transition, diff, True, xyz)
            
            loss1, loss2 = compute_loss(args, diff, is_transition, weights_loss)
            
            factor = 1000 #scaling for better usage of float precision
            
            loss = (loss1 + loss2) / weights_loss.shape[0]*factor
            
            loss.backward()
            
            if args.grad_clip>0:
                for p in model.parameters():
                    p.grad.data.clamp_(-args.grad_clip*factor, args.grad_clip*factor)
                    
            optimizer.step()

            t_trainer = 1000*(time.time()-t0)
            loss_meter.add(loss.item()/factor)#/weights_loss.mean().item())
            n_clusters_meter.add(embeddings.shape[0] / len(pred_comp))
            
            logging.debug('Batch loss %f, Loader time %f ms, Trainer time %f ms.', loss.item() / factor, t_loader, t_trainer)
            t0 = time.time()
            
        #return 0,0,0
        return loss_meter.value()[0], n_clusters_meter.value()[0]
    
    def evaluate(i_epoch):
        """ Evaluated model on test set """
        model.eval()
        
        with torch.no_grad():
            
            loader = torch.utils.data.DataLoader(test_dataset , batch_size=1, collate_fn=graph_collate, num_workers=args.nworkers)
            
            if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=100)
    
            loss_meter = tnt.meter.AverageValueMeter()
            n_clusters_meter = tnt.meter.AverageValueMeter()
            BR_meter = tnt.meter.AverageValueMeter()
            BP_meter = tnt.meter.AverageValueMeter()
            CM_classes = metrics.ConfusionMatrix(dbinfo['classes'])
    
            # iterate over dataset in batches
            for bidx, (fname, edg_source, edg_target, is_transition, labels, objects, clouds_data, xyz) in enumerate(loader):
                
                if args.cuda:
                    is_transition = is_transition.to('cuda',non_blocking=True)
                    #labels = torch.from_numpy(labels).cuda()
                    objects = objects.to('cuda',non_blocking=True)
                    clouds, clouds_global, nei = clouds_data
                    clouds_data = (clouds.to('cuda',non_blocking=True),clouds_global.to('cuda',non_blocking=True),nei) 

                embeddings = ptnCloudEmbedder.run_batch(model, *clouds_data, xyz)
            
                diff = compute_dist(embeddings, edg_source, edg_target, args.dist_type)
                
                if len(is_transition)>1:
                    weights_loss, pred_components, pred_in_component = compute_weight_loss(args, embeddings, objects, edg_source, edg_target, is_transition, diff, True, xyz)
                    loss1, loss2 = compute_loss(args, diff, is_transition, weights_loss)
                    loss = (loss1 + loss2) / weights_loss.shape[0]
                    pred_transition = pred_in_component[edg_source]!=pred_in_component[edg_target]
                    per_pred = perfect_prediction(pred_components, labels)
                    CM_classes.count_predicted_batch(labels[:,1:], per_pred)
                else:
                    loss = 0
                    
                if len(is_transition)>1:
                    loss_meter.add(loss.item())#/weights_loss.sum().item())
                    is_transition = is_transition.cpu().numpy()
                    n_clusters_meter.add(len(pred_components))
                    BR_meter.add((is_transition.sum())*compute_boundary_recall(is_transition, relax_edge_binary(pred_transition, edg_source, edg_target, xyz.shape[0], args.BR_tolerance)),n=is_transition.sum())
                    BP_meter.add((pred_transition.sum())*compute_boundary_precision(relax_edge_binary(is_transition, edg_source, edg_target, xyz.shape[0], args.BR_tolerance), pred_transition),n=pred_transition.sum())
        CM = CM_classes.confusion_matrix
        return loss_meter.value()[0], n_clusters_meter.value()[0], 100*CM.trace() / CM.sum(), BR_meter.value()[0], BP_meter.value()[0]
    
    def evaluate_final():
        """ Evaluated model on test set """
        
        print("Final evaluation")
        model.eval()
        
        loss_meter = tnt.meter.AverageValueMeter()
        n_clusters_meter = tnt.meter.AverageValueMeter()
        confusion_matrix_classes = metrics.ConfusionMatrix(dbinfo['classes'])
        confusion_matrix_BR = metrics.ConfusionMatrix(2)
        confusion_matrix_BP = metrics.ConfusionMatrix(2)
            
        with torch.no_grad():
            
            loader = torch.utils.data.DataLoader(test_dataset , batch_size=1, collate_fn=graph_collate, num_workers=args.nworkers)
                
            if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=100)
    
    # iterate over dataset in batches
            for bidx, (fname, edg_source, edg_target, is_transition, labels, objects, clouds_data, xyz) in enumerate(loader):

                if args.cuda:
                    is_transition = is_transition.to('cuda',non_blocking=True)
                    #labels = torch.from_numpy(labels).cuda()
                    objects = objects.to('cuda',non_blocking=True)
                    clouds, clouds_global, nei = clouds_data
                    clouds_data = (clouds.to('cuda',non_blocking=True),clouds_global.to('cuda',non_blocking=True),nei) 
                
                if args.dataset=='sema3d':
                    embeddings = ptnCloudEmbedder.run_batch_cpu(model, *clouds_data, xyz)
                else:
                    embeddings = ptnCloudEmbedder.run_batch(model, *clouds_data, xyz)
                
                diff = compute_dist(embeddings, edg_source, edg_target, args.dist_type)
                    
                pred_components, pred_in_component = compute_partition(args, embeddings, edg_source, edg_target, diff, xyz)
                    
                if len(is_transition)>1:
                    pred_transition = pred_in_component[edg_source]!=pred_in_component[edg_target]
                    is_transition = is_transition.cpu().numpy()
                        
                    n_clusters_meter.add(len(pred_components))
    
                    per_pred = perfect_prediction(pred_components, labels)                    
                    confusion_matrix_classes.count_predicted_batch(labels[:,1:], per_pred)
                    confusion_matrix_BR.count_predicted_batch_hard(is_transition, relax_edge_binary(pred_transition, edg_source, edg_target, xyz.shape[0], args.BR_tolerance).astype('uint8'))
                    confusion_matrix_BP.count_predicted_batch_hard(relax_edge_binary(is_transition, edg_source, edg_target, xyz.shape[0], args.BR_tolerance),pred_transition.astype('uint8'))
              
                if args.spg_out:
                    graph_sp = compute_sp_graph(xyz, 100, pred_in_component, pred_components, labels, dbinfo["classes"])
                    spg_file = os.path.join(folder_hierarchy.spg_folder, fname[0])
                    if not os.path.exists(os.path.dirname(spg_file)):
                        os.makedirs(os.path.dirname(spg_file))
                    try:
                        os.remove(spg_file)
                    except OSError:
                        pass
                    write_spg(spg_file, graph_sp, pred_components, pred_in_component)

                    # Debugging purpose - write the embedding file and an exemple of scalar files
                    # if bidx % 0 == 0:
                    #     embedding2ply(os.path.join(folder_hierarchy.emb_folder , fname[0][:-3] + '_emb.ply'), xyz, embeddings.detach().cpu().numpy())
                    #     scalar2ply(os.path.join(folder_hierarchy.scalars , fname[0][:-3] + '_elevation.ply') , xyz, clouds_data[1][:,1].cpu())
                    #     edg_class = is_transition + 2*pred_transition
                    #     edge_class2ply2(os.path.join(folder_hierarchy.emb_folder , fname[0][:-3] + '_transition.ply'), edg_class, xyz, edg_source, edg_target)
            if len(is_transition)>1:
                res_name = folder_hierarchy.outputdir+'/res.h5'
                res_file = h5py.File(res_name, 'w')
                res_file.create_dataset('confusion_matrix_classes'
                                 , data=confusion_matrix_classes.confusion_matrix, dtype='uint64')
                res_file.create_dataset('confusion_matrix_BR'
                                 , data=confusion_matrix_BR.confusion_matrix, dtype='uint64')
                res_file.create_dataset('confusion_matrix_BP'
                                 , data=confusion_matrix_BP.confusion_matrix, dtype='uint64')
                res_file.create_dataset('n_clusters'
                                 , data=n_clusters_meter.value()[0], dtype='uint64')
                res_file.close()
                
        return
    
    # Training loop
    #
    for epoch in range(args.start_epoch, args.epochs):
        if not args.learned_embeddings:
            break
        print('Epoch {}/{} ({}):'.format(epoch, args.epochs, folder_hierarchy.outputdir))
        scheduler.step()

        loss, n_sp = train(epoch)

        if (epoch+1) % args.test_nth_epoch == 0: #or epoch+1==args.epochs:
            loss_test, n_clusters_test, ASA_test, BR_test, BP_test = evaluate(epoch)
            print('-> Train loss: %1.5f - Test Loss: %1.5f  |  n_clusters:  %5.1f  |  ASA: %3.2f %%  |  Test BR: %3.2f %%  |  BP : %3.2f%%' % (loss, loss_test, n_clusters_test, ASA_test, BR_test, BP_test))
        else:
            loss_test, n_clusters_test, ASA_test, BR_test, BP_test = 0,0,0,0,0
            print('-> Train loss: %1.5f  superpoints size : %5.0f' % (loss, n_sp))

        stats.append({'epoch': epoch, 'loss': loss, 'loss_test': loss_test, 'n_clusters_test': n_clusters_test, 'ASA_test': ASA_test, 'BR_test': BR_test, 'BP_test': BP_test})

        with open(os.path.join(folder_hierarchy.outputdir, 'trainlog.json'), 'w') as outfile:
            json.dump(stats, outfile, indent=4)

        if epoch % args.save_nth_epoch == 0 or epoch==args.epochs-1:            
            model_name = 'model.pth.tar'
            print("Saving model to " + model_name)
            model_name = 'model.pth.tar'
            torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()},
                        os.path.join(folder_hierarchy.outputdir, model_name))

        if math.isnan(loss): break

    evaluate_final()


def create_model(args):
    """ Creates model """
    model = nn.Module()
    if args.learned_embeddings and 'ptn' in args.ptn_embedding and args.ptn_nfeat_stn > 0:
        model.stn = STNkD(args.ptn_nfeat_stn, args.ptn_widths_stn[0], args.ptn_widths_stn[1], norm=args.ptn_norm, n_group = args.ptn_n_group)
    
    if args.learned_embeddings and 'ptn' in args.ptn_embedding:
        n_embed = args.ptn_widths[1][-1]
        n_feat = 3 + 3 * args.use_rgb
        nfeats_global = len(args.global_feat) + 4 * args.stn_as_global + 1 #we always add the diameter
        model.ptn = PointNet(args.ptn_widths[0], args.ptn_widths[1], [], [], n_feat, 0, prelast_do=args.ptn_prelast_do, nfeat_global=nfeats_global, norm=args.ptn_norm, is_res = False, last_bn = True)# = args.normalize_intermediary==0)

    if args.ver_value  == 'geofrgb':
        n_embed = 7
        model.placeholder = nn.Parameter(torch.tensor(0.0))
    if args.ver_value  == 'geof':
        n_embed = 4
        model.placeholder = nn.Parameter(torch.tensor(0.0))
        
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print(model)    
    if args.cuda: 
        model.cuda()
    return model

def resume(args):
    """ Loads model and optimizer state from a previous checkpoint. """
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model = create_model(checkpoint['args']) #use original arguments, architecture can't change
    if not args.cuda:
        model = model.cpu()
    optimizer = create_optimizer(args, model)
    
    model.load_state_dict(checkpoint['state_dict'])
    if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
    for group in optimizer.param_groups: group['initial_lr'] = args.lr
    args.start_epoch = checkpoint['epoch']
    try:
        stats = json.loads(open(os.path.join(os.path.dirname(args.resume), 'trainlog.json')).read())
    except:
        stats = []
    return model, optimizer, stats
    
    
if __name__ == "__main__": 
    logging.getLogger().setLevel(logging.INFO)  #set to logging.DEBUG to allow for more prints
    args = parse_args()
    embed(args)
