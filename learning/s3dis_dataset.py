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
    """ Gets training and test datasets. """

    # Load superpoints graphs
    testlist, trainlist = [], []
    for n in range(1,7):
        if n != args.cvfold:
            path = '{}/superpoint_graphs/Area_{:d}/'.format(args.S3DIS_PATH, n)
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".h5"):
                    trainlist.append(spg.spg_reader(args, path + fname, True))
    path = '{}/superpoint_graphs/Area_{:d}/'.format(args.S3DIS_PATH, args.cvfold)
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".h5"):
            testlist.append(spg.spg_reader(args, path + fname, True))

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist = spg.scaler01(trainlist, testlist)

    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.S3DIS_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.S3DIS_PATH, test_seed_offset=test_seed_offset))


def get_info(edge_attribs,pc_attribs):
    edge_feats = 0
    for attrib in edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1

    return {
        'node_feats': 14 if pc_attribs=='' else len(pc_attribs),
        'edge_feats': edge_feats,
        'classes': 13,
        'inv_class_map': {0:'ceiling', 1:'floor', 2:'wall', 3:'column', 4:'beam', 5:'window', 6:'door', 7:'table', 8:'chair', 9:'bookcase', 10:'sofa', 11:'board', 12:'clutter'},
    }



def preprocess_pointclouds(S3DIS_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""

    for n in range(1,7):
        pathP = '{}/parsed/Area_{:d}/'.format(S3DIS_PATH, n)
        pathD = '{}/features/Area_{:d}/'.format(S3DIS_PATH, n)
        pathC = '{}/superpoint_graphs/Area_{:d}/'.format(S3DIS_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(n)

        for file in os.listdir(pathC):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')
                xyz = f['xyz'][:]
                rgb = f['rgb'][:].astype(np.float)
                elpsv = np.stack([ f['xyz'][:,2][:], f['linearity'][:], f['planarity'][:], f['scattering'][:], f['verticality'][:] ], axis=1)

                # rescale to [-0.5,0.5]; keep xyz
                elpsv[:,0] = elpsv[:,0] / 4 - 0.5 # (4m rough guess)
                elpsv[:,1:] -= 0.5
                rgb = rgb/255.0 - 0.5

                ma, mi = np.max(xyz,axis=0,keepdims=True), np.min(xyz,axis=0,keepdims=True)
                xyzn = (xyz - mi) / (ma - mi + 1e-8)   # as in PointNet ("normalized location as to the room (from 0 to 1)")

                P = np.concatenate([xyz, rgb, elpsv, xyzn], axis=1)

                f = h5py.File(pathC + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(pathP + file, 'w') as hf:
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--S3DIS_PATH', default='datasets/s3dis')
    args = parser.parse_args()
    preprocess_pointclouds(args.S3DIS_PATH)
