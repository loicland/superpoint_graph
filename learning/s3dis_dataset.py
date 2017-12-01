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


def reader_postprocess(entry):
    """ Remove class "stairs" by handling it as unlabeled (as in previous work) """
    node_gt, node_gt_size, edges, edge_feats, fname = entry

    node_gt_size[:,0] += node_gt_size[:,-1]
    node_gt_size = node_gt_size[:,:-1]
    node_gt = np.argmax(node_gt_size[:,1:], 1)[:,None]
    node_gt[node_gt_size[:,1:].sum(1)==0,:] = -100

    return node_gt, node_gt_size, edges, edge_feats, fname


def get_datasets(args, test_seed_offset=0):
    """ Gets training and test datasets. """

    # Load superpoints graphs
    testlist, trainlist = [], []
    for n in range(1,7):
        if n != args.cvfold:
            path = '{}/reduced_graphs/Area_{:d}/'.format(args.S3DIS_PATH, n)
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".h5"):
                    trainlist.append(reader_postprocess(spg.spg_reader(args, path + fname, True)))
    path = '{}/reduced_graphs/Area_{:d}/'.format(args.S3DIS_PATH, args.cvfold)
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".h5"):
            testlist.append(reader_postprocess(spg.spg_reader(args, path + fname, True)))

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist = spg.scaler01(trainlist, testlist)

    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.S3DIS_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.S3DIS_PATH, test_seed_offset=test_seed_offset))


def get_info(args):
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1

    return {
        'node_feats': 14 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'classes': 13,
        'inv_class_map': {0:'ceiling', 1:'floor', 2:'wall', 3:'column', 4:'beam', 5:'window', 6:'door', 7:'table', 8:'chair', 9:'bookcase', 10:'sofa', 11:'board', 12:'clutter'},
    }

###### TODO ADAPT:

from plyfile import PlyData, PlyElement

def preprocess_pointclouds():
    """ Converts .ply clouds into .h5, splitting them by components and normalizing."""
    import scipy.io
    S3DIS_PATH = 'datasets/s3dis_02/'

    for n in range(1,7):
    #for n in [6]:
        pathP = '{}/parsed/Area_{:d}/'.format(S3DIS_PATH, n)
        pathD = '{}/descriptors/Area_{:d}/'.format(S3DIS_PATH, n)
        pathC = '{}/components/Area_{:d}/'.format(S3DIS_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(n)

        for file in os.listdir(pathD):
            print(file)
            if file.endswith(".ply"):
                plydata = PlyData.read(pathD + file)
                xyz = np.stack([ plydata['vertex'][n] for n in ['x','y','z'] ], axis=1)
                try:
                    rgb = np.stack([ plydata['vertex'][n] for n in ['red', 'green', 'blue'] ], axis=1).astype(np.float)
                except ValueError:
                    rgb = np.stack([ plydata['vertex'][n] for n in ['r', 'g', 'b'] ], axis=1).astype(np.float)
                elpsv = np.stack([ plydata['vertex'][n] for n in ['z', 'linearity', 'planarity', 'scattering', 'verticality'] ], axis=1) # todo: unsure about z=elevation

                print(np.amin(xyz[:,2]),np.amax(xyz[:,2]))

                # rescale to [-0.5,0.5]; keep xyz
                elpsv[:,0] = elpsv[:,0] / 4 - 0.5 # (4m rough guess)    -- different from sema3d
                elpsv[:,1:] -= 0.5
                rgb = rgb/255.0 - 0.5

                ma, mi = np.max(xyz,axis=0,keepdims=True), np.min(xyz,axis=0,keepdims=True)  #-- different from sema3d
                xyzn = (xyz - mi) / (ma - mi + 1e-8)   # as in PointNet ("normalized location as to the room (from 0 to 1)")

                P = np.concatenate([xyz, rgb, elpsv, xyzn], axis=1)

                mat = scipy.io.loadmat(pathC + os.path.splitext(file)[0] + '_components.mat')
                components=mat['components']

                with h5py.File(pathP + os.path.splitext(file)[0] + '.h5', 'w') as hf:
                    for c in range(len(components)):
                        idx = components[c][0].flatten()

                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])

                # todo: do https://flothesof.github.io/farthest-neighbors.html right here? Also PointNet recommends farthest point subsampling...




if __name__ == "__main__":
    preprocess_pointclouds()
