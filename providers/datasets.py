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
            'clutter': 12,
            'stairs': 13
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
            'classes': 14,
            'inv_class_map': {value:key for (key,value) in self.labels.items()},
        }
    
    def get_data_for_inference_only(self, args, filename, folder_s, test_seed_offset=0):
        """ get data for inference """
        # Load superpoints graphs
        testlist, trainlist = [], []
        path = os.path.join(args.ROOT_PATH,'superpoint_graphs',folder_s)
        if filename.endswith(".h5"):
            testlist.append(spg.spg_reader(args, path + filename, True))
            
        # need to load the whole training set if we want to normalize the edge feature wrt to the stats of the training set.
        for n in range(1,7):
            if n != args.cvfold:
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
        pts = pts  - np.min(pts,axis=0,keepdims=True) 
        return pts
    
    def preprocess_pointclouds(self,ROOT_PATH, pc_attribs, single_file = False, filename = '', folder= ''):
        """ Preprocesses data by splitting them by components and normalizing."""
        if not single_file :
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

                        if pc_attribs == 'xyzelpsvXYZ':
                            ma, mi = np.max(xyz,axis=0,keepdims=True), np.min(xyz,axis=0,keepdims=True)
                            xyzn = (xyz - mi) / (ma - mi + 1e-8)   # as in PointNet ("normalized location as to the room (from 0 to 1)")
                            rgb = np.zeros((xyz.shape[0],3))
                            P = np.concatenate([xyz, rgb,elpsv, xyzn], axis=1)
                        elif pc_attribs == 'xyzelpsv':
                            rgb = np.zeros((xyz.shape[0],3))
                            P = np.concatenate([xyz, rgb,elpsv], axis=1)

                        f = h5py.File(os.path.join(pathC, file), 'r')
                        numc = len(f['components'].keys())

                        with h5py.File(os.path.join(pathP, file), 'w') as hf:
                            for c in range(numc):
                                idx = f['components/{:d}'.format(c)][:].flatten()
                                if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                                    ii = random.sample(range(idx.size), k=10000)
                                    idx = idx[ii]

                                hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])
        else :
            pathP = os.path.join(ROOT_PATH,'parsed',folder)
            pathD = os.path.join(ROOT_PATH,'features',folder)
            pathC = os.path.join(ROOT_PATH,'superpoint_graphs',folder)
            if not os.path.exists(pathP):
                os.makedirs(pathP)
            file = filename
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')
                xyz = f['xyz'][:]
                elpsv = np.stack([ f['xyz'][:,2][:], f['linearity'][:], f['planarity'][:], f['scattering'][:], f['verticality'][:] ], axis=1)
                # rescale to [-0.5,0.5]; keep xyz
                elpsv[:,0] = elpsv[:,0] / np.max(elpsv[:,0]) - 0.5 
                elpsv[:,1:] -= 0.5
                
                if pc_attribs == 'xyzelspvXYZ':
                    ma, mi = np.max(xyz,axis=0,keepdims=True), np.min(xyz,axis=0,keepdims=True)
                    xyzn = (xyz - mi) / (ma - mi + 1e-8)   # as in PointNet ("normalized location as to the room (from 0 to 1)")
                    rgb = np.zeros((xyz.shape[0],3))
                    P = np.concatenate([xyz, rgb,elpsv, xyzn], axis=1)
                elif pc_attribs == 'xyzelpsv':
                    rgb = np.zeros((xyz.shape[0],3))
                    P = np.concatenate([xyz, rgb,elpsv], axis=1)
                
                f = h5py.File(os.path.join(pathC, file), 'r')
                numc = len(f['components'].keys())

                with h5py.File(os.path.join(pathP, file), 'w') as hf:
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])               
    

class CustomS3DISDataset:

    def __init__(self):
        self.name = "custom_s3dis"
        self.folders = ["Area_1/", "Area_2/", "Area_3/", "Area_4/", "Area_5/", "Area_6/"]
        self.extension = ".txt"
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
            'clutter': 12,
            'stairs': 13
        }
    
    def get_info(self,args):
        edge_attribs = args.edge_attribs
        pc_attribs = args.pc_attribs
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
            'classes': 14,
            'inv_class_map': {value:key for (key,value) in self.labels.items()},
        }
    
    def get_datasets(self, args, test_seed_offset=0):
        """ Gets training and test datasets. """
        # Load superpoints graphs
        testlist, trainlist = [], []
        for n in range(1,7):
            if n != args.cvfold:
                path = '{}/superpoint_graphs/Area_{:d}/'.format(args.S3DIS_PATH, n)
                for fname in sorted(os.listdir(path)):
                    if fname.endswith(".h5"):
                        trainlist.append(spg.spg_reader(args, path + fname, True))
        
        """path = '{}/superpoint_graphs/Area_{:d}/'.format(args.S3DIS_PATH, 5)
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".h5"):
                trainlist.append(spg.spg_reader(args, path + fname, True))"""
        
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
    
    def read_custom_s3dis_format(self, raw_path, label_out=True):
    #S3DIS specific
        """extract data from a room folder"""
        room_ver = np.genfromtxt(raw_path, delimiter=' ')
        xyz = np.array(room_ver[:, 0:3], dtype='float32')
        rgb = np.array(room_ver[:, 3:6], dtype='uint8')
        if not label_out:
            return xyz, rgb
        # label has to start with 1 and not 0, so adding 1 since ceiling : 0
        room_labels = np.array(room_ver[:, 6], dtype='uint8') + 1
        # keeping only a few labels
        #room_labels = np.array(room_ver[:, 6], dtype='uint8')
        #room_labels[room_labels == 3] = 2 # considering columns as walls
        #room_labels[room_labels == 4] = 12 # considering beam as clutter
        #room_labels[room_labels == 7] = 12 # and so on
        #room_labels[room_labels == 8] = 12
        #room_labels[room_labels == 9] = 12
        #room_labels[room_labels == 10] = 12
        #room_labels[room_labels == 11] = 2 # considering board as walls
        #room_labels[room_labels == 13] = 12
        #room_labels += 1
        # Align x,y,z with origin
        xyz = xyz  - np.min(xyz,axis=0,keepdims=True)
        return xyz, rgb, room_labels
    
    def preprocess_pointclouds(self, ROOT_PATH, pc_attribs):
        """ Preprocesses data by splitting them by components and normalizing."""
        for n,folder in enumerate(self.folders):
            pathP = os.path.join(ROOT_PATH,'parsed',folder)
            pathD = os.path.join(ROOT_PATH,'features',folder)
            pathC = os.path.join(ROOT_PATH,'superpoint_graphs',folder)
            if not os.path.exists(pathP):
                os.makedirs(pathP)
            random.seed(n)
            for file in os.listdir(pathC):
                print(folder+file)
                if file.endswith(".h5"):
                    f = h5py.File(pathD + file, 'r')
                    xyz = f['xyz'][:]
                    elpsv = np.stack([ f['xyz'][:,2][:], f['linearity'][:], f['planarity'][:], f['scattering'][:], f['verticality'][:] ], axis=1)

                    # rescale to [-0.5,0.5]; keep xyz # need to be aligned with the origin beforehand
                    elpsv[:,0] = elpsv[:,0] / np.max(elpsv[:,0]) - 0.5 
                    elpsv[:,1:] -= 0.5

                    if pc_attribs == 'xyzelpsvXYZ':
                        ma, mi = np.max(xyz,axis=0,keepdims=True), np.min(xyz,axis=0,keepdims=True)
                        xyzn = (xyz - mi) / (ma - mi + 1e-8)   # as in PointNet ("normalized location as to the room (from 0 to 1)")
                        rgb = np.zeros((xyz.shape[0],3))
                        P = np.concatenate([xyz, rgb,elpsv, xyzn], axis=1)
                    elif pc_attribs == 'xyzelpsv':
                        rgb = np.zeros((xyz.shape[0],3))
                        P = np.concatenate([xyz, rgb,elpsv], axis=1)

                    f = h5py.File(os.path.join(pathC, file), 'r')
                    numc = len(f['components'].keys())

                    with h5py.File(os.path.join(pathP, file), 'w') as hf:
                        for c in range(numc):
                            idx = f['components/{:d}'.format(c)][:].flatten()
                            if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                                ii = random.sample(range(idx.size), k=10000)
                                idx = idx[ii]

                            hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])