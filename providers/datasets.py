import open3d as o3d
import numpy as np
import functools
import os
import functools
import sys
import torchnet as tnt
import h5py
import random

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path,'../learning'))

import spg

class HelixDataset:

    def __init__(self):
        self.name = "helix_v1"
        self.folders = ["test/"]
        self.extension = ".ply"
        self.labels = {
            'ceiling': 0,
            'floor': 1,
            'wall': 2,
            'column': 3,
            'beam': 4,
            'window': 5,
            'door': 6,
            'table': 7,
            'chair': 8,
            'bookcase': 9,
            'sofa': 10,
            'board': 11,
            'clutter': 12
        }
    
    def get_info(self,edge_attribs,pc_attribs):
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
            'inv_class_map': {value:key for (key,value) in self.labels.items()},
        }
    
    def get_datasets(self,args, test_seed_offset=0):
        """ Gets training and test datasets. """
        # Load superpoints graphs
        testlist, trainlist = [], []
        for folder in self.folders:
            path = os.path.join(args.ROOT_PATH,'superpoint_graphs',folder)
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".h5"):
                    testlist.append(spg.spg_reader(args, path + fname, True))
           
        # Load training data for normalisation purposes mainly
        for n in range(2,7):
            path = '{}/superpoint_graphs/Area_{:d}/'.format(args.S3DIS_PATH, n)
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".h5"):
                    trainlist.append(spg.spg_reader(args, path + fname, True))

        # Normalize edge features
        if args.spg_attribs01:
            trainlist, testlist = spg.scaler01(trainlist, testlist)

        return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                        functools.partial(spg.loader, train=True, args=args, db_path=args.ROOT_PATH)), \
               tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                        functools.partial(spg.loader, train=False, args=args, db_path=args.ROOT_PATH, test_seed_offset=test_seed_offset))

    def read_pointcloud(self,filename):
        cloud = o3d.read_point_cloud(filename)
        # Align x,y,z with  origin
        pts = np.asarray(cloud.points)
        pts = pts - np.min(pts,axis=0,keepdims=True)
        return pts
    
    def preprocess_pointclouds(self,ROOT_PATH):
        """ Preprocesses data by splitting them by components and normalizing."""

        for n,folder in enumerate(self.folders):
            pathP = os.path.join(ROOT_PATH,'parsed',folder)
            pathD = os.path.join(ROOT_PATH,'features',folder)
            pathC = os.path.join(ROOT_PATH,'superpoint_graphs',folder)
            if not os.path.exists(pathP):
                os.makedirs(pathP)
            random.seed(n)

            for file in os.listdir(pathC):
                print(file)
                if file.endswith(".h5"):
                    f = h5py.File(pathD + file, 'r')
                    xyz = f['xyz'][:]
                    elpsv = np.stack([ f['xyz'][:,2][:], f['linearity'][:], f['planarity'][:], f['scattering'][:], f['verticality'][:] ], axis=1)

                    # rescale to [-0.5,0.5]; keep xyz
                    elpsv[:,0] = elpsv[:,0] / np.max(elpsv[:,0]) - 0.5 
                    elpsv[:,1:] -= 0.5

                    ma, mi = np.max(xyz,axis=0,keepdims=True), np.min(xyz,axis=0,keepdims=True)
                    xyzn = (xyz - mi) / (ma - mi + 1e-8)   # as in PointNet ("normalized location as to the room (from 0 to 1)")

                    rgb = np.zeros((xyz.shape[0],3))
                    P = np.concatenate([xyz, rgb,elpsv, xyzn], axis=1)

                    f = h5py.File(os.path.join(pathC, file), 'r')
                    numc = len(f['components'].keys())

                    with h5py.File(os.path.join(pathP, file), 'w') as hf:
                        for c in range(numc):
                            idx = f['components/{:d}'.format(c)][:].flatten()
                            if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                                ii = random.sample(range(idx.size), k=10000)
                                idx = idx[ii]

                            hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])