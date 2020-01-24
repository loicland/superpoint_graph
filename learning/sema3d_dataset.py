"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
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

    train_names = ['bildstein_station1', 'bildstein_station5', 'domfountain_station1', 'domfountain_station3', 'neugasse_station1', 'sg27_station1', 'sg27_station2', 'sg27_station5', 'sg27_station9', 'sg28_station4', 'untermaederbrunnen_station1']
    valid_names = ['bildstein_station3', 'domfountain_station2', 'sg27_station4', 'untermaederbrunnen_station3']

    if args.db_train_name == 'train':
        trainset = ['train/' + f for f in train_names]
    elif args.db_train_name == 'trainval':
        trainset = ['train/' + f for f in train_names + valid_names]

    validset = []
    testset = []
    if args.use_val_set:
        validset = ['train/' + f for f in valid_names]
    if args.db_test_name == 'testred':
        testset = ['test_reduced/' + os.path.splitext(f)[0] for f in os.listdir(args.SEMA3D_PATH + '/superpoint_graphs/test_reduced')]
    elif args.db_test_name == 'testfull':
        testset = ['test_full/' + os.path.splitext(f)[0] for f in os.listdir(args.SEMA3D_PATH + '/superpoint_graphs/test_full')]
        
    # Load superpoints graphs
    testlist, trainlist, validlist = [], [],  []
    for n in trainset:
        trainlist.append(spg.spg_reader(args, args.SEMA3D_PATH + '/superpoint_graphs/' + n + '.h5', True))
    for n in validset:
        validlist.append(spg.spg_reader(args, args.SEMA3D_PATH + '/superpoint_graphs/' + n + '.h5', True))
    for n in testset:
        testlist.append(spg.spg_reader(args, args.SEMA3D_PATH + '/superpoint_graphs/' + n + '.h5', True))

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist, validlist, scaler = spg.scaler01(trainlist, testlist, validlist=validlist)

    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.SEMA3D_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.SEMA3D_PATH, test_seed_offset=test_seed_offset)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in validlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.SEMA3D_PATH, test_seed_offset=test_seed_offset)),\
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
        weights = np.ones((8,),dtype='f4')
    else:
        weights = h5py.File(args.SEMA3D_PATH + "/parsed/class_count.h5")["class_count"][:].astype('f4')
        weights = weights.mean()/weights
    if args.loss_weights == 'sqrt':
        weights = np.sqrt(weights)
    weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)
    return {
        'node_feats': 14 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'class_weights': weights,
        'classes': 8,
        'inv_class_map': {0:'terrain_man', 1:'terrain_nature', 2:'veget_hi', 3:'veget_low', 4:'building', 5:'scape', 6:'artefact', 7:'cars'},
    }

def preprocess_pointclouds(SEMA3D_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""
    class_count = np.zeros((8,),dtype='int')
    for n in ['train', 'test_reduced', 'test_full']:
        pathP = '{}/parsed/{}/'.format(SEMA3D_PATH, n)
        if args.supervised_partition :
            pathD = '{}/features_supervision/{}/'.format(SEMA3D_PATH, n)
        else:
            pathD = '{}/features/{}/'.format(SEMA3D_PATH, n)
        pathC = '{}/superpoint_graphs/{}/'.format(SEMA3D_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(0)

        for file in os.listdir(pathC):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')

                if n == 'train':
                    labels = f['labels'][:]
                    hard_labels = np.argmax(labels[:,1:],1)
                    label_count = np.bincount(hard_labels, minlength=8)
                    class_count = class_count + label_count
                
                xyz = f['xyz'][:]
                rgb = f['rgb'][:].astype(np.float)
                elpsv = np.concatenate((f['xyz'][:,2][:,None], f['geof'][:]), axis=1)

                # rescale to [-0.5,0.5]; keep xyz
                elpsv[:,0] /= 100 # (rough guess)
                elpsv[:,1:] -= 0.5
                rgb = rgb/255.0 - 0.5
                
                P = np.concatenate([xyz, rgb, elpsv], axis=1)

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
    path = '{}/parsed/'.format(SEMA3D_PATH)
    data_file = h5py.File(path+'class_count.h5', 'w')
    data_file.create_dataset('class_count', data=class_count, dtype='int')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--SEMA3D_PATH', default='datasets/semantic3d')
    parser.add_argument('--supervised_partition', default=0, type=int, help = 'wether to use supervized partition features')
    args = parser.parse_args()
    preprocess_pointclouds(args.SEMA3D_PATH)
