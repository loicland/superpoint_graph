"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import sys
sys.path.append("./learning")

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import spg
from sklearn.linear_model import RANSACRegressor

def get_datasets(args, test_seed_offset=0):
    """ Gets training and test datasets. """

    # Load superpoints graphs
    testlist, trainlist, validlist = [], [], []
    valid_names = ['hallway_1.h5', 'hallway_6.h5', 'hallway_11.h5', 'office_1.h5' \
                 , 'office_6.h5', 'office_11.h5', 'office_16.h5', 'office_21.h5', 'office_26.h5' \
                 , 'office_31.h5', 'office_36.h5'\
                 ,'WC_2.h5', 'storage_1.h5', 'storage_5.h5', 'conferenceRoom_2.h5', 'auditorium_1.h5']
    
     #if args.db_test_name == 'test' then the test set is the evaluation set
     #otherwise it serves as valdiation set to select the best epoch
    
    for n in range(1,7):
        if n != args.cvfold:
            path = '{}/superpoint_graphs/Area_{:d}/'.format(args.S3DIS_PATH, n)
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".h5") and not (args.use_val_set and fname in valid_names):
                    #training set
                    trainlist.append(spg.spg_reader(args, path + fname, True))
                if fname.endswith(".h5") and (args.use_val_set  and fname in valid_names):
                    #validation set
                    validlist.append(spg.spg_reader(args, path + fname, True))
    path = '{}/superpoint_graphs/Area_{:d}/'.format(args.S3DIS_PATH, args.cvfold)
    
    #evaluation set
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".h5"):
            testlist.append(spg.spg_reader(args, path + fname, True))

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist, validlist, scaler = spg.scaler01(trainlist, testlist, validlist=validlist)

    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.S3DIS_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.S3DIS_PATH, test_seed_offset=test_seed_offset)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in validlist], 
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.S3DIS_PATH, test_seed_offset=test_seed_offset)), \
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
        weights = h5py.File(args.S3DIS_PATH + "/parsed/class_count.h5")["class_count"][:].astype('f4')
        weights = weights[:,[i for i in range(6) if i != args.cvfold-1]].sum(1)
        weights = weights.mean()/weights
    if args.loss_weights == 'sqrt':
        weights = np.sqrt(weights)
    weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)
    return {
        'node_feats': 14 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'class_weights': weights,
        'classes': 13,
        'inv_class_map': {0:'ceiling', 1:'floor', 2:'wall', 3:'column', 4:'beam', 5:'window', 6:'door', 7:'table', 8:'chair', 9:'bookcase', 10:'sofa', 11:'board', 12:'clutter'},
    }



def preprocess_pointclouds(args):
    """ Preprocesses data by splitting them by components and normalizing."""
    S3DIS_PATH = args.S3DIS_PATH
    class_count = np.zeros((13,6),dtype='int')
    for n in range(1,7):
        pathP = '{}/parsed/Area_{:d}/'.format(S3DIS_PATH, n)
        if args.supervized_partition:
            pathD = '{}/features_supervision/Area_{:d}/'.format(S3DIS_PATH, n)
        else:
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

                labels = f['labels'][:]
                hard_labels = np.argmax(labels[:,1:],1)
                label_count = np.bincount(hard_labels, minlength=13)
                class_count[:,n-1] = class_count[:,n-1] + label_count
                
                if not args.supervized_partition:
                    lpsv = f['geof'][:]
                    lpsv -= 0.5 #normalize
                else:
                    lpsv = np.stack([f["geof"][:] ]).squeeze()
                # rescale to [-0.5,0.5]; keep xyz
                
                if args.plane_model_elevation:
                    if args.supervized_partition: #already computed
                        e = f['elevation'][:]
                    else:   #simple plane model
                        low_points = ((xyz[:,2]-xyz[:,2].min() < 0.5)).nonzero()[0]
                        reg = RANSACRegressor(random_state=0).fit(xyz[low_points,:2], xyz[low_points,2])
                        e = xyz[:,2]-reg.predict(xyz[:,:2])
                else:   #compute elevation from zmin
                    e = xyz[:,2] / 4 - 0.5 # (4m rough guess)
                    
                rgb = rgb/255.0 - 0.5
                
                room_center = xyz[:,[0,1]].mean(0) #compute distance to room center, useful to detect walls and doors
                distance_to_center = np.sqrt(((xyz[:,[0,1]]-room_center)**2).sum(1))
                distance_to_center = (distance_to_center - distance_to_center.mean())/distance_to_center.std()

                ma, mi = np.max(xyz,axis=0,keepdims=True), np.min(xyz,axis=0,keepdims=True)
                xyzn = (xyz - mi) / (ma - mi + 1e-8)   # as in PointNet ("normalized location as to the room (from 0 to 1)")

                P = np.concatenate([xyz, rgb, e[:,np.newaxis], lpsv, xyzn, distance_to_center[:,None]], axis=1)

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

    path = '{}/parsed/'.format(S3DIS_PATH)
    data_file = h5py.File(path+'class_count.h5', 'w')
    data_file.create_dataset('class_count', data=class_count, dtype='int')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--S3DIS_PATH', default='datasets/s3dis')
    parser.add_argument('--supervized_partition', type=int, default=0)
    parser.add_argument('--plane_model_elevation', type=int, default=0, help='compute elevation with a simple RANSAC based plane model')
    args = parser.parse_args()
    preprocess_pointclouds(args)
