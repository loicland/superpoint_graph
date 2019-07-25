#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:45:16 2018
@author: landrieuloic
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import spg

def get_datasets(args, test_seed_offset=0):
    """ Gets training and test datasets. """

    # Load superpoints graphs
    testlist, trainlist, validlist = [], [], []
    valid_names = ['0001_00000.h5','0001_00085.h5', '0001_00170.h5','0001_00230.h5','0001_00325.h5','0001_00420.h5', \
                   '0002_00000.h5','0002_00111.h5','0002_00223.h5','0018_00030.h5','0018_00184.h5','0018_00338.h5',\
                   '0020_00080.h5','0020_00262.h5','0020_00444.h5','0020_00542.h5','0020_00692.h5', '0020_00800.h5']
    
    for n in range(1,7):
        if n != args.cvfold:
            path = '{}/superpoint_graphs/0{:d}/'.format(args.VKITTI_PATH, n)
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".h5") and not (args.use_val_set and fname in valid_names):
                    #training set
                    trainlist.append(spg.spg_reader(args, path + fname, True))
                if fname.endswith(".h5") and (args.use_val_set  and fname in valid_names):
                    #validation set
                    validlist.append(spg.spg_reader(args, path + fname, True))
    path = '{}/superpoint_graphs/0{:d}/'.format(args.VKITTI_PATH, args.cvfold)
    #evaluation set
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".h5"):
            testlist.append(spg.spg_reader(args, path + fname, True))

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist, validlist, scaler = spg.scaler01(trainlist, testlist, validlist=validlist)
        
    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.VKITTI_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.VKITTI_PATH, test_seed_offset=test_seed_offset)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in validlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.VKITTI_PATH, test_seed_offset=test_seed_offset)), \
            scaler


def get_info(args):
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1
    if args.loss_weights == 'none':
        weights = np.ones((13,),dtype='f4')
    else:
        weights = h5py.File(args.VKITTI_PATH + "/parsed/class_count.h5")["class_count"][:].astype('f4')
        weights = weights[:,[i for i in range(6) if i != args.cvfold-1]].sum(1)
        weights = (weights+1).mean()/(weights+1)
    if args.loss_weights == 'sqrt':
        weights = np.sqrt(weights)
    weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)
    return {
        'node_feats': 9 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'classes': 13,
        'class_weights': weights,
        'inv_class_map': {0:'Terrain', 1:'Tree', 2:'Vegetation', 3:'Building', 4:'Road', 5:'GuardRail', 6:'TrafficSign', 7:'TrafficLight', 8:'Pole', 9:'Misc', 10:'Truck', 11:'Car', 12:'Van'},
    }

def preprocess_pointclouds(VKITTI_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""
    class_count = np.zeros((13,6),dtype='int')
    for n in range(1,7):
        pathP = '{}/parsed/0{:d}/'.format(VKITTI_PATH, n)
        pathD = '{}/features_supervision/0{:d}/'.format(VKITTI_PATH, n)
        pathC = '{}/superpoint_graphs/0{:d}/'.format(VKITTI_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(n)

        for file in os.listdir(pathC):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')
                xyz = f['xyz'][:]
                rgb = f['rgb'][:].astype(np.float)
                
                labels = f['labels'][:]
                hard_labels = np.argmax(labels[:,1:],1)
                label_count = np.bincount(hard_labels, minlength=13)
                class_count[:,n-1] = class_count[:,n-1] + label_count
                
                e = (f['xyz'][:,2][:] -  np.min(f['xyz'][:,2]))/ (np.max(f['xyz'][:,2]) -  np.min(f['xyz'][:,2]))-0.5

                rgb = rgb/255.0 - 0.5
                
                xyzn = (xyz - np.array([30,0,0])) / np.array([30,5,3])
                
                lpsv = np.zeros((e.shape[0],4))

                P = np.concatenate([xyz, rgb, e[:,np.newaxis], lpsv, xyzn], axis=1)

                f = h5py.File(pathC + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(pathP + file, 'w') as hf:
                    hf.create_dataset(name='centroid',data=xyz.mean(0))
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])
    path = '{}/parsed/'.format(VKITTI_PATH)
    data_file = h5py.File(path+'class_count.h5', 'w')
    data_file.create_dataset('class_count', data=class_count, dtype='int')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--VKITTI_PATH', default='datasets/s3dis')
    args = parser.parse_args()
    preprocess_pointclouds(args.VKITTI_PATH)