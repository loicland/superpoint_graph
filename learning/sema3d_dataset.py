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
        trainset = train_names
    elif args.db_train_name == 'trainval':
        trainset = train_names + valid_names

    if args.db_test_name == 'val':
        testset = valid_names
    elif args.db_test_name == 'testred':
        testset = ['testred/' + os.path.splitext(f)[0] for f in os.listdir(args.SEMA3D_PATH + '/descriptors/testred')]
    elif args.db_test_name == 'testfull':
        testset = ['testfull/' + os.path.splitext(f)[0] for f in os.listdir(args.SEMA3D_PATH + '/descriptors/testfull')]

    # Load superpoints graphs
    testlist, trainlist = [], []
    for n in trainset:
        trainlist.append(spg.spg_reader(args, args.SEMA3D_PATH + '/reduced_graphs/' + n + '_graph.h5'))
    for n in testset:
        testlist.append(spg.spg_reader(args, args.SEMA3D_PATH + '/reduced_graphs/' + n + '_graph.h5', args.db_test_name.startswith('test')))

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist = spg.scaler01(trainlist, testlist)

    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.SEMA3D_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.SEMA3D_PATH, test_seed_offset=test_seed_offset))

def get_info(args):
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1

    return {
        'node_feats': 11 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'classes': 8,
        'inv_class_map': {0:'terrain_man', 1:'terrain_nature', 2:'veget_hi', 3:'veget_low', 4:'building', 5:'scape', 6:'artefact', 7:'cars'},
    }


###### TODO ADAPT:

from plyfile import PlyData, PlyElement

def preprocess_pointclouds():
    """ Converts .ply clouds into .h5, splitting them by components and normalizing."""
    import scipy.io
    random.seed(0)    # note: I preprocessed train+val and both tests independently!
    SEMA3D_PATH = 'datasets/semantic3d/'

    prefix = 'full/'
    for file in os.listdir(SEMA3D_PATH+ 'descriptors/' + prefix):
        print(file)
        if file.endswith(".ply"):
            plydata = PlyData.read(SEMA3D_PATH + 'descriptors/' + prefix + file)
            xyz = np.stack([ plydata['vertex'][n] for n in ['x','y','z'] ], axis=1)
            try:
                rgb = np.stack([ plydata['vertex'][n] for n in ['red', 'green', 'blue'] ], axis=1).astype(np.float)
            except ValueError:
                rgb = np.stack([ plydata['vertex'][n] for n in ['r', 'g', 'b'] ], axis=1).astype(np.float)
            elpsv = np.stack([ plydata['vertex'][n] for n in ['z', 'linearity', 'planarity', 'scattering', 'verticality'] ], axis=1) # todo: unsure about z=elevation !
            #elpsv = np.concatenate([np.zeros((lpsv.shape[0],1)), lpsv], axis=1)   # TODO: missing

            print(np.amin(xyz[:,2]),np.amax(xyz[:,2]))

            # rescale to [-0.5,0.5]; keep xyz
            elpsv[:,0] /= 100 # (rough guess)
            elpsv[:,1:] -= 0.5
            rgb = rgb/255.0 - 0.5

            P = np.concatenate([xyz, rgb, elpsv], axis=1)

            mat = scipy.io.loadmat(SEMA3D_PATH + 'components/' + prefix + os.path.splitext(file)[0] + '_components.mat')
            components=mat['components']

            with h5py.File(SEMA3D_PATH + 'parsed/' + prefix + os.path.splitext(file)[0]+ '.h5', 'w') as hf:
                for c in range(len(components)):
                    idx = components[c][0].flatten()

                    if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                        ii = random.sample(range(idx.size), k=10000)
                        idx = idx[ii]

                    hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])

            # todo: do https://flothesof.github.io/farthest-neighbors.html right here? Also PointNet recommends farthest point subsampling...




if __name__ == "__main__":
    preprocess_pointclouds()
