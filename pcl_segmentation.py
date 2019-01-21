
# coding: utf-8

# # Point Cloud Segmentation

# This notebook aims at coding a single step to process point cloud : It details each processing step from a raw point cloud to a semantic segmented point cloud. The method will take the point cloud in input and will output the corresponding segmented point cloud

# In[1]:


import os.path
import sys
import numpy as np
from timeit import default_timer as timer
sys.path.append('partition/cut-pursuit/src')
sys.path.append('partition/ply_c')
sys.path.append('partition')
import libcp
import libply_c
from graphs import *
from provider import *
sys.path.append('./providers')
from datasets import *

import torch
import torch.nn as nn
from providers.datasets import HelixDataset
from collections import defaultdict
import h5py
import os
from plyfile import PlyData, PlyElement
import open3d as o3d

sys.path.append('learning')
sys.path.append('partition')
import spg
import graphnet
import pointnet
import metrics
import provider
import s3dis_dataset
import custom_dataset

from visualisation import display_cloud

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# # **Point Cloud Segmentation Steps**

# ## ***Partitioning Point Cloud***

# This subsubsection adapt the code used in partition/parition.py.

# ### Partitionning 

# In[2]:


def _partition(path_to_pcl, k_nn_geof = 45, k_nn_adj = 10, lambda_edge_weight = 1., reg_strength = 0.03, d_se_max = 0, voxel_width = 0.03, ver_batch = 0, overwrite = 0 ):
        """ Large-scale Point Cloud Segmentation with Superpoint Graphs
        Input : - path_to_pcl : path to the raw point cloud .ply file.
                - k_nn_geof : number of neighbors for the geometric features, type=int
                - k_nn_adj : adjacency structure for the minimal partition, type=int
                - lambda_edge_weight : parameter determine the edge weight for minimal part, type=float
                - reg_strength : regularization strength for the minimal partition, type=float
                - d_se_max : max length of super edges, type=float
                - voxel_width : voxel size when subsampling (in m), type=float
                - ver_batch : Batch size for reading large files, 0 do disable batch loading, type=int
                - overwrite : Wether to read existing files or overwrite them, type=int
        """
        
        root, file =  os.path.dirname(os.path.dirname(os.path.split(os.path.abspath(path_to_pcl))[0])), os.path.split(os.path.abspath(path_to_pcl))[1]
        root = root + '/'
        
        helix_data = HelixDataset()
        folder = helix_data.folders[0]
        n_labels = len(helix_data.labels.keys())
        
        times = [0,0,0] #time for computing: features / partition / spg

        if not os.path.isdir(root + "clouds"):
            os.mkdir(root + "clouds")
        if not os.path.isdir(root + "features"):
            os.mkdir(root + "features")
        if not os.path.isdir(root + "superpoint_graphs"):
            os.mkdir(root + "superpoint_graphs")
            
        print("=================\n   "+ 'Start Partitioning {}'.format(file)+"\n=================")

        data_folder = root   + "data/"              + folder
        cloud_folder  = root + "clouds/"            + folder
        fea_folder  = root   + "features/"          + folder
        spg_folder  = root   + "superpoint_graphs/" + folder
        if not os.path.isdir(data_folder):
            raise ValueError("%s does not exist" % data_folder)

        if not os.path.isdir(cloud_folder):
            os.mkdir(cloud_folder)
        if not os.path.isdir(fea_folder):
            os.mkdir(fea_folder)
        if not os.path.isdir(spg_folder):
            os.mkdir(spg_folder)

        if not os.path.isfile(data_folder + file):
            raise ValueError('{} does not exist in {}'.format(file, data_folder))

        file_name   = os.path.splitext(os.path.basename(file))[0]

        data_file   = data_folder      + file_name + helix_data.extension
        cloud_file  = cloud_folder     + file_name
        fea_file    = fea_folder       + file_name + '.h5'
        spg_file    = spg_folder       + file_name + '.h5'
       
        #--- build the geometric feature file h5 file ---
        if os.path.isfile(fea_file) and not overwrite:
            print("    reading the existing feature file...")
            geof, xyz, rgb, graph_nn, labels = read_features(fea_file)
        else :
            print("    creating the feature file...")
            #--- read the data files and compute the labels---
            
            xyz = helix_data.read_pointcloud(data_file).astype(dtype='float32')
            if voxel_width > 0:
                xyz = libply_c.prune(xyz, voxel_width, np.zeros(xyz.shape,dtype='u1'), np.array(1,dtype='u1'), 0)[0]
            labels = []
            rgb = []
            
            start = timer()
            #---compute 10 nn graph-------
            graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
            #---compute geometric features-------
            geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
            end = timer()
            times[0] = times[0] + end - start
            del target_fea
            write_features(fea_file, geof, xyz, rgb, graph_nn, labels)
        #--compute the partition------
        sys.stdout.flush()
        if os.path.isfile(spg_file) and not overwrite:
            print("    reading the existing superpoint graph file...")
            graph_sp, components, in_component = read_spg(spg_file)
        else:
            print("    computing the superpoint graph...")
            #--- build the spg h5 file --
            features = geof
            geof[:,3] = 2. * geof[:, 3]
            
            graph_nn["edge_weight"] = np.array(1. / (lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
            print("        minimal partition...")
            components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                         , graph_nn["edge_weight"], reg_strength)
            components = np.array(components, dtype = 'object')
            end = timer()
            times[1] = times[1] + end - start
            print("        computation of the SPG...")
            start = timer()
            graph_sp = compute_sp_graph(xyz, d_se_max, in_component, components, labels, n_labels)
            end = timer()
            times[2] = times[2] + end - start
            write_spg(spg_file, graph_sp, components, in_component)

        print("Timer : %5.1f / %5.1f / %5.1f " % (times[0], times[1], times[2]))
        print("=================\n   "+ 'Ended Partitioning {}'.format(file)+"\n=================")
        return root[:-1], folder + file_name


# In[3]:


#root, file = _partition('data/TEST/data/test/room_1900.ply')


# In[4]:


# normal if it fails
#_partition('data/TEST/data/test/room_19065.ply')


# ## ***Embedding Semantic Informations***

# ### Loading model and Weights

# In[7]:


def load_weights(model_path,model_config,db_info):
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    
    checkpoint['args'].model_config = model_config 
    cloud_embedder = pointnet.CloudEmbedder(checkpoint['args'])
    model = create_model(checkpoint['args'], db_info) #use original arguments, architecture can't change    
    model.load_state_dict(checkpoint['state_dict'])
    return model, cloud_embedder, checkpoint['args']


# In[8]:


def create_model(args, dbinfo):
    """ Creates model """
    model = nn.Module()

    nfeat = args.ptn_widths[1][-1]
    model.ecc = graphnet.GraphNetwork(args.model_config, nfeat, [dbinfo['edge_feats']] + args.fnet_widths, args.fnet_orthoinit, args.fnet_llbias,args.fnet_bnidx, args.edge_mem_limit)

    model.ptn = pointnet.PointNet(args.ptn_widths[0], args.ptn_widths[1], args.ptn_widths_stn[0], args.ptn_widths_stn[1], dbinfo['node_feats'], args.ptn_nfeat_stn, prelast_do=args.ptn_prelast_do)

    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()]))) 
    if args.cuda: 
        model.cuda()
    return model 


# In[9]:


"""
MODEL_PATH = 'results/s3dis/bw/cv1/model.pth.tar'
model_config = 'gru_10_0,f_13'
edge_attribs = 'delta_avg,delta_std,nlength/ld,surface/ld,volume/ld,size/ld,xyz/d'
pc_attribs = 'xyzelspvXYZ'
dbinfo = HelixDataset().get_info(edge_attribs,pc_attribs)
"""
#model,cloud_embedder, args = load_weights(MODEL_PATH,model_config,dbinfo)


# ### Run Inferences

# In[12]:


def predict(args, root):
    args.ROOT_PATH = root
    HelixDataset().preprocess_pointclouds(args.ROOT_PATH)
    create_dataset = HelixDataset().get_datasets

    collected = defaultdict(list)
    for ss in range(args.test_multisamp_n):
        eval_data = create_dataset(args,ss)[1]
        loader = torch.utils.data.DataLoader(eval_data, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=8)
        for bidx, (targets, GIs, clouds_data) in enumerate(loader):
            model.ecc.set_info(GIs, args.cuda)
            label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1).float()
            data = clouds_data
            embeddings = cloud_embedder.run(model, *clouds_data)
            outputs = model.ecc(embeddings)
            fname = clouds_data[0][0][:clouds_data[0][0].rfind('.')]
            collected[fname].append((outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy()))
    predictions = {}
    with h5py.File(os.path.join(args.ROOT_PATH, 'predictions.h5'), 'w') as hf:
        for fname,output in collected.items():
            o_cpu, t_cpu, tvec_cpu = list(zip(*output))
            o_cpu = np.mean(np.stack(o_cpu,0),0)
            prediction = np.argmax(o_cpu,axis=-1)
            predictions[fname] = prediction
            hf.create_dataset(name=fname, data=prediction) #(0-based classes)
    return predictions


# In[13]:


#predictions = predict(args, root)


# ### Outputs Segmented Point Cloud

# In[2]:


def visualise(root_path, filename, predictions):
    n_labels = 13
    
    folder = os.path.split(filename)[0] + '/'
    file_name = os.path.split(filename)[1]
    
    #---load the values------------------------------------------------------------
    fea_file   = os.path.join(root_path,'features',folder,file_name + '.h5')
    spg_file   = os.path.join(root_path,'superpoint_graphs',folder,file_name + '.h5')
    ply_folder = os.path.join(root_path,'clouds',folder)
    ply_file   = os.path.join(ply_folder,file_name)

    if not os.path.isdir(ply_folder ):
        os.mkdir(ply_folder)
    if (not os.path.isfile(fea_file)) :
        raise ValueError("%s does not exist and is needed" % fea_file)
    
    geof, xyz, rgb, graph_nn, labels = provider.read_features(fea_file)

    if not os.path.isfile(spg_file):    
        raise ValueError("%s does not exist and is needed to output the partition  or result ply" % spg_file) 
    else:
        graph_spg, components, in_component = provider.read_spg(spg_file)
        
    pred_red  = predictions[filename]        
    if (len(pred_red) != len(components)):
        raise ValueError("It looks like the spg is not adapted to the result file") 
    pred_full = provider.reduced_labels2full(pred_red, components, len(xyz))
    
    print("writing the prediction file...")
    provider.prediction2ply(ply_file + "_pred.ply", xyz, pred_full+1, n_labels,  args.dataset)


# In[16]:


#visualise(root, file, predictions)


# # **Regrouping in a Class**

# In[8]:


class PointCloudSegmentation(object):
    """
    Collection of functions used to segment a point cloud
    """
    
    def __init__(self, MODEL_PATH, model_config, edge_attribs, pc_attribs, dataset):
        self._model_path = MODEL_PATH
        self._model_config = model_config
        self._edge_attribs = edge_attribs
        self._pc_attribs = pc_attribs
        self._dataset = dataset
        self._model = None
        self._cloud_embedder = None
        self._args = None
        
    
    def process(self, input_pcl, save_model = False):
        """ take a raw point cloud as input and output a segmented point cloud in a .ply file"""
        root, folder, file = self._partition(input_pcl)
        print("=================\n   "+ 'Running Inferences' +"\n=================")
        predictions = self._predict(root, folder, file)
        xyz, xyz_labels = self._save(root, folder+file , predictions, save_model)
        return xyz, xyz_labels
    
    
    def display(self, xyz, xyz_labels):
        n_labels = 14
        clouds = []
        labels = []
        colors = []
        
        if self._dataset == 's3dis':
            dataset_info = s3dis_dataset.get_info(self._edge_attribs,self._pc_attribs)
        elif self._dataset == 'helix':
            dataset_info = HelixDataset().get_info(self._edge_attribs,self._pc_attribs)
        
        for i_label in range(n_labels):
            cloud = xyz[np.where(xyz_labels == i_label)]
            # converting simple array to open3d.PointCloud object
            pcd = o3d.PointCloud()
            pcd.points = o3d.Vector3dVector(cloud)
            if len(pcd.points) != 0 :
                clouds.append(pcd)
                labels.append(dataset_info['inv_class_map'][i_label])
                colors.append(provider.get_color_from_label(i_label, self._dataset))
        
        colors = np.asarray(colors)/255
        display_cloud(clouds = clouds, labels = labels, colors = colors)
       
    
    def load_model(self):
        """ load the weiths of the model """
        print("=================\n   "+ 'Preparing Model' +"\n=================")
        print("=> loading checkpoint '{}'".format(self._model_path))
        checkpoint = torch.load(self._model_path)
        print(checkpoint['args'])
        checkpoint['args'].model_config = self._model_config 
        cloud_embedder = pointnet.CloudEmbedder(checkpoint['args'])
        dbinfo = HelixDataset().get_info(self._edge_attribs,self._pc_attribs)
        model = self._create_model(checkpoint['args'], dbinfo) #use original arguments, architecture can't change    
        model.load_state_dict(checkpoint['state_dict'])
        self._model = model
        self._cloud_embedder = cloud_embedder
        self._args = checkpoint['args']
        return 
    
    
    def load_prediction(self, root_path, filename, prediction_file):
        """ load the predictions from a file
        root_path : relative path to the data folder (containing features, superpoint graph... folders)
        filename : filename without the extension and its parent directory. ex : if test_01.ply in in directory test : test\test_01
        prediction_file : name of the prediction file : ex test_01_predictions.h5
        """
        n_labels = 14

        folder = os.path.split(filename)[0] + '/'
        file_name = os.path.split(filename)[1]

        #---load the values------------------------------------------------------------
        fea_file   = os.path.join(root_path,'features',folder,file_name + '.h5')
        spg_file   = os.path.join(root_path,'superpoint_graphs',folder,file_name + '.h5')
        ply_folder = os.path.join(root_path,'clouds',folder)
        #pred_folder = os.path.join(root_path,'predictions',folder)
        ply_file   = os.path.join(ply_folder,file_name)
        res_file   = os.path.join(root_path,'predictions', folder, prediction_file)

        if not os.path.isdir(ply_folder ):
            os.mkdir(ply_folder)
        if (not os.path.isfile(fea_file)) :
            raise ValueError("%s does not exist and is needed" % fea_file)

        geof, xyz, rgb, graph_nn, labels = provider.read_features(fea_file)

        if not os.path.isfile(spg_file):    
            raise ValueError("%s does not exist and is needed to output the partition  or result ply" % spg_file) 
        else:
            graph_spg, components, in_component = provider.read_spg(spg_file)

        if not os.path.isfile(res_file):
            raise ValueError("%s does not exist and is needed." % res_file) 
        try:
            pred_red  = np.array(h5py.File(res_file, 'r').get(folder + file_name))        
            if (len(pred_red) != len(components)):
                raise ValueError("It looks like the spg is not adapted to the result file") 
            pred_full = provider.reduced_labels2full(pred_red, components, len(xyz))
        except OSError:
            raise ValueError("%s does not exist in %s" % (folder + file_name, res_file))
        
        return xyz, pred_full
        
        
    def _create_model(self, args, dbinfo):
        """ Creates the model """
        model = nn.Module()
        nfeat = args.ptn_widths[1][-1]
        model.ecc = graphnet.GraphNetwork(args.model_config, nfeat, [dbinfo['edge_feats']] + args.fnet_widths, args.fnet_orthoinit, args.fnet_llbias,args.fnet_bnidx, args.edge_mem_limit)
        model.ptn = pointnet.PointNet(args.ptn_widths[0], args.ptn_widths[1], args.ptn_widths_stn[0], args.ptn_widths_stn[1], dbinfo['node_feats'], args.ptn_nfeat_stn, prelast_do=args.ptn_prelast_do)
        print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()]))) 
        if args.cuda: 
            model.cuda()
        return model 
    
    
    def _partition(self, path_to_pcl, k_nn_geof = 45, k_nn_adj = 10, lambda_edge_weight = 1., reg_strength = 0.03, d_se_max = 0, voxel_width = 0.03, ver_batch = 0, overwrite = 0 ):
            """ Large-scale Point Cloud Segmentation with Superpoint Graphs
            Input : - path_to_pcl : path to the raw point cloud .ply file.
                    - k_nn_geof : number of neighbors for the geometric features, type=int
                    - k_nn_adj : adjacency structure for the minimal partition, type=int
                    - lambda_edge_weight : parameter determine the edge weight for minimal part, type=float
                    - reg_strength : regularization strength for the minimal partition, type=float
                    - d_se_max : max length of super edges, type=float
                    - voxel_width : voxel size when subsampling (in m), type=float
                    - ver_batch : Batch size for reading large files, 0 do disable batch loading, type=int
                    - overwrite : Wether to read existing files or overwrite them, type=int
            """
            
            if self._dataset == 's3dis':
                root, file =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.split(os.path.abspath(path_to_pcl))[0]))), os.path.split(os.path.abspath(path_to_pcl))[1]
                root = root + '/'
            elif self._dataset == 'helix':
                root, file =  os.path.dirname(os.path.dirname(os.path.split(os.path.abspath(path_to_pcl))[0])), os.path.split(os.path.abspath(path_to_pcl))[1]
                root = root + '/'
            
            #list of subfolders to be processed
            if self._dataset == 's3dis':
                folder = os.path.split(os.path.dirname(os.path.dirname(path_to_pcl)))[1] + '/'
                n_labels = 13
            elif self._dataset == 'helix':
                helix_data = HelixDataset()
                folder = os.path.split(os.path.split(os.path.abspath(path_to_pcl))[0])[1] + '/'
                n_labels = len(helix_data.labels.keys())
            else:
                raise ValueError('%s is an unknown data set' % dataset)

            times = [0,0,0] #time for computing: features / partition / spg

            if not os.path.isdir(root + "clouds"):
                os.mkdir(root + "clouds")
            if not os.path.isdir(root + "features"):
                os.mkdir(root + "features")
            if not os.path.isdir(root + "superpoint_graphs"):
                os.mkdir(root + "superpoint_graphs")

            print("=================\n   "+ 'Start Partitioning {}'.format(file)+"\n=================")
            
            data_folder = root   + "data/"              + folder
            cloud_folder  = root + "clouds/"            + folder
            fea_folder  = root   + "features/"          + folder
            spg_folder  = root   + "superpoint_graphs/" + folder
            if not os.path.isdir(data_folder):
                raise ValueError("%s does not exist" % data_folder)

            if not os.path.isdir(cloud_folder):
                os.mkdir(cloud_folder)
            if not os.path.isdir(fea_folder):
                os.mkdir(fea_folder)
            if not os.path.isdir(spg_folder):
                os.mkdir(spg_folder)
            
            file_name = os.path.splitext(os.path.basename(file))[0]
           
            if self._dataset == 's3dis':
                if not os.path.isfile(data_folder +  file_name + '/' + file):
                    raise ValueError('{} does not exist in {}'.format(file, data_folder +  file_name + '/'))
            elif self._dataset == 'helix':
                if not os.path.isfile(data_folder +  file):
                    raise ValueError('{} does not exist in {}'.format(file, data_folder))
            
            if self._dataset =='s3dis':
                data_file   = data_folder      + file_name + '/' + file_name + ".txt"
                cloud_file  = cloud_folder     + file_name
                fea_file    = fea_folder       + file_name + '.h5'
                spg_file    = spg_folder       + file_name + '.h5'
            elif self._dataset =='helix':
                data_file   = data_folder      + file_name + helix_data.extension
                cloud_file  = cloud_folder     + file_name
                fea_file    = fea_folder       + file_name + '.h5'
                spg_file    = spg_folder       + file_name + '.h5'

            #--- build the geometric feature file h5 file ---
            if os.path.isfile(fea_file) and not overwrite:
                print("    reading the existing feature file...")
                geof, xyz, rgb, graph_nn, labels = read_features(fea_file)
            else :
                print("    creating the feature file...")
                #--- read the data files and compute the labels---
                
                if self._dataset ==' s3dis':
                    xyz, rgb, labels = read_s3dis_format(data_file)
                    if voxel_width > 0:
                        xyz, rgb, labels = libply_c.prune(xyz, voxel_width, rgb, labels, n_labels)
                elif self._dataset == 'helix' :
                    xyz = helix_data.read_pointcloud(data_file).astype(dtype='float32')
                    if voxel_width > 0:
                        xyz = libply_c.prune(xyz, voxel_width, np.zeros(xyz.shape,dtype='u1'), np.array(1,dtype='u1'), 0)[0]
                    labels = []
                    rgb = []

                start = timer()
                #---compute 10 nn graph-------
                graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
                #---compute geometric features-------
                geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
                end = timer()
                times[0] = times[0] + end - start
                del target_fea
                write_features(fea_file, geof, xyz, rgb, graph_nn, labels)
            #--compute the partition------
            sys.stdout.flush()
            if os.path.isfile(spg_file) and not overwrite:
                print("    reading the existing superpoint graph file...")
                graph_sp, components, in_component = read_spg(spg_file)
            else:
                print("    computing the superpoint graph...")
                #--- build the spg h5 file --
                features = geof
                geof[:,3] = 2. * geof[:, 3]
                if self._dataset =='s3dis':
                    features = np.hstack((geof, rgb/255.)).astype('float32')#add rgb as a feature for partitioning
                    features[:,3] = 2. * features[:,3] #increase importance of verticality (heuristic)
                elif self._dataset =='helix':
                    features = geof
                    geof[:,3] = 2. * geof[:, 3]

                graph_nn["edge_weight"] = np.array(1. / (lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
                print("        minimal partition...")
                components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                             , graph_nn["edge_weight"], reg_strength)
                components = np.array(components, dtype = 'object')
                end = timer()
                times[1] = times[1] + end - start
                print("        computation of the SPG...")
                start = timer()
                graph_sp = compute_sp_graph(xyz, d_se_max, in_component, components, labels, n_labels)
                end = timer()
                times[2] = times[2] + end - start
                write_spg(spg_file, graph_sp, components, in_component)
            print("Timer : %5.1f / %5.1f / %5.1f " % (times[0], times[1], times[2]))
            return root[:-1], folder, file_name
    
    def _predict(self, root, folder, file):
        self._args.ROOT_PATH = root
        self._args.S3DIS_PATH = 'data/custom_S3DIS'
        file_name = file+'.h5'
        if self._dataset == 's3dis':
            create_dataset = s3dis_dataset.get_datasets
        elif self._dataset == 'helix':
            HelixDataset().preprocess_pointclouds(self._args.ROOT_PATH, single_file = True, filename = file_name, folder = folder)
            create_data = HelixDataset().get_data
        
        if not os.path.isdir(root + "/predictions"):
                os.mkdir(root + "/predictions")
        pred_folder = root   + "/predictions/" + folder
        if not os.path.isdir(pred_folder):
                os.mkdir(pred_folder)
            
        collected = defaultdict(list)
        eval_data = create_data(self._args, filename = file_name, folder_s = folder)[1]
        loader = torch.utils.data.DataLoader(eval_data, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=8)
        for bidx, (targets, GIs, clouds_data) in enumerate(loader):
            self._model.ecc.set_info(GIs, self._args.cuda)
            label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1).float()
            data = clouds_data
            embeddings = self._cloud_embedder.run(self._model, *clouds_data)
            outputs = self._model.ecc(embeddings)
            fname = clouds_data[0][0][:clouds_data[0][0].rfind('.')]
            collected[fname].append((outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy()))
        predictions = {}
        with h5py.File(os.path.join(pred_folder, os.path.splitext(os.path.basename(file))[0] +'_predictions.h5'), 'w') as hf:
            for fname,output in collected.items():
                o_cpu, t_cpu, tvec_cpu = list(zip(*output))
                o_cpu = np.mean(np.stack(o_cpu,0),0)
                prediction = np.argmax(o_cpu,axis=-1)
                predictions[fname] = prediction
                hf.create_dataset(name=fname, data=prediction) #(0-based classes)
        return predictions  
    
    
    def _save(self, root_path, filename,predictions, save_model):
        n_labels = 14

        folder = os.path.split(filename)[0] + '/'
        file_name = os.path.split(filename)[1]

        #---load the values------------------------------------------------------------
        fea_file   = os.path.join(root_path,'features',folder,file_name + '.h5')
        spg_file   = os.path.join(root_path,'superpoint_graphs',folder,file_name + '.h5')
        ply_folder = os.path.join(root_path,'clouds',folder)
        ply_file   = os.path.join(ply_folder,file_name)

        if not os.path.isdir(ply_folder ):
            os.mkdir(ply_folder)
        if (not os.path.isfile(fea_file)) :
            raise ValueError("%s does not exist and is needed" % fea_file)

        geof, xyz, rgb, graph_nn, labels = provider.read_features(fea_file)

        if not os.path.isfile(spg_file):    
            raise ValueError("%s does not exist and is needed to output the partition  or result ply" % spg_file) 
        else:
            graph_spg, components, in_component = provider.read_spg(spg_file)
        
        pred_red  = predictions[filename]        
        if (len(pred_red) != len(components)):
            raise ValueError("It looks like the spg is not adapted to the result file") 
        pred_full = provider.reduced_labels2full(pred_red, components, len(xyz))
        
        if save_model:
            print("=================\n   "+ 'Saving Segmented Point Cloud' +"\n=================")
            print("writing the prediction file (i.e Semantic Segmented Point Cloud) in {}...".format(ply_folder))
            provider.prediction2ply(ply_file + "_pred.ply", xyz, pred_full+1, n_labels,  self._dataset)
        
        return xyz, pred_full


# # **How to use the Class**

# ## Initialize the model

# In[9]:


MODEL_PATH = 'results/s3dis/bw/cv1_3/model.pth.tar'
model_config = 'gru_10_0,f_14'
edge_attribs = 'delta_avg,delta_std,nlength/ld,surface/ld,volume/ld,size/ld,xyz/d'
#pc_attribs = 'xyzelspvXYZ'
pc_attribs = 'xyzelspv'
dataset = 'helix'


# In[10]:


model = PointCloudSegmentation(MODEL_PATH, model_config, edge_attribs, pc_attribs, dataset)


# 
# ## Load the Weights

# In[11]:


model.load_model()


# ## Segment the Point Cloud

# In[12]:


xyz, xyz_labels = model.process('data/TEST/data/test/test_02.ply') #set save_model to True if you want to write out the segmented point cloud. 


# ## Or reading an existing file

# In[6]:


xyz, xyz_labels = model.load_prediction('data/TEST', 'test/test_02', 'test_02_predictions.h5')


# ## Visualisation

# In[12]:


model.display(xyz, xyz_labels)

